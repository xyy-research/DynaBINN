"""MAG, POGA, SCA and multi-level classifier (unchanged computations)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustOmicsFusion(nn.Module):
    def __init__(self, n_genes, n_omics):
        super(RobustOmicsFusion, self).__init__()
        self.ln = nn.LayerNorm(n_genes)
        self.global_omics_logits = nn.Parameter(torch.ones(n_omics))
        self.gene_scale = nn.Parameter(torch.ones(n_genes))
        self.bias = nn.Parameter(torch.zeros(n_genes))
        # 固化全激活掩码
        self.register_buffer('omics_mask', torch.ones(n_omics))

    def forward(self, x):
        # 显式掩码乘法（代码一的关键稳定化操作）
        x = x * self.omics_mask.unsqueeze(0).unsqueeze(0)
        
        # attention融合 + 强制归一化
        attn_weights = F.softmax(self.global_omics_logits, dim=0)
        attn_weights = attn_weights * self.omics_mask
        attn_weights = attn_weights / attn_weights.sum()
        
        fused_x = torch.sum(x * attn_weights.unsqueeze(0).unsqueeze(0), dim=2)
        
        out = fused_x * self.gene_scale + self.bias
        out = self.ln(out)
        return torch.tanh(out)

class DualStreamOrthogonalGATLayer(nn.Module):
    def __init__(self, n_src_nodes, n_dst_nodes, prior_mask, augmented_mask, is_discovery_phase=False, alpha=0.2, dropout=0.2):
        super(DualStreamOrthogonalGATLayer, self).__init__()
        self.is_discovery_phase = is_discovery_phase
        self.register_buffer('prior_mask', torch.tensor(prior_mask, dtype=torch.float32).t())
        self.register_buffer('augmented_mask', torch.tensor(augmented_mask, dtype=torch.float32).t())
        
        self.W_base = nn.Parameter(torch.empty(n_dst_nodes, n_src_nodes))
        self.W_gate = nn.Parameter(torch.empty(n_dst_nodes, n_src_nodes))
        nn.init.xavier_uniform_(self.W_base)
        nn.init.xavier_uniform_(self.W_gate)
        self.bias = nn.Parameter(torch.zeros(n_dst_nodes))

        hidden_dim = 8
        self.W_src = nn.Linear(1, hidden_dim, bias=False) 
        self.H_dst = nn.Parameter(torch.empty(1, n_dst_nodes, hidden_dim))
        nn.init.xavier_uniform_(self.H_dst)
        self.a_src = nn.Parameter(torch.empty(hidden_dim, 1))
        self.a_dst = nn.Parameter(torch.empty(hidden_dim, 1))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.W_out = nn.Linear(hidden_dim, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(dropout)
        self.gamma_gat = nn.Parameter(torch.ones(1) * 0.5)

        if self.is_discovery_phase:
            self.W_adj = nn.Parameter(torch.empty_like(self.prior_mask))
            nn.init.normal_(self.W_adj, mean=-1.0, std=0.1) 

    def forward(self, x_src):
        batch_size = x_src.size(0)
        mask_to_use = self.augmented_mask if not self.is_discovery_phase else torch.clamp(self.prior_mask + torch.sigmoid(self.W_adj), min=0.0, max=1.0)
        masked_w_base = self.W_base * mask_to_use
        masked_w_gate = self.W_gate * mask_to_use
        val_base = F.linear(x_src, masked_w_base)
        gate_base = torch.sigmoid(F.linear(x_src, masked_w_gate))
        h_base = val_base * gate_base + self.bias
        
        x_src_ext = x_src.unsqueeze(-1) 
        h_src_gat = self.W_src(x_src_ext) 
        h_dst_gat = self.H_dst.expand(batch_size, -1, -1) 
        
        score_src = torch.matmul(h_src_gat, self.a_src) 
        score_dst = torch.matmul(h_dst_gat, self.a_dst) 
        e = self.leakyrelu(score_src + score_dst.transpose(1, 2)) 
        attention_dropped = self.dropout_layer(torch.sigmoid(e)) 
        
        if self.is_discovery_phase:
            learned_edges = torch.sigmoid(self.W_adj)
            adj_mask_t = torch.clamp(self.prior_mask + learned_edges, min=0.0, max=1.0).t().unsqueeze(0).expand(batch_size, -1, -1)
            deg_mask = adj_mask_t.sum(dim=1, keepdim=True).clamp(min=1.0)
            attn_norm = (attention_dropped * adj_mask_t) / deg_mask 
            h_prime = torch.matmul(attn_norm.transpose(1, 2), h_src_gat) + h_dst_gat 
            h_gat_out = self.W_out(F.elu(h_prime)).squeeze(-1) 
            out = torch.tanh(h_base + self.gamma_gat * h_gat_out)
            struct_loss = torch.mean(learned_edges)
            return out, struct_loss
        else:
            adj_prior_t = self.prior_mask.t().unsqueeze(0).expand(batch_size, -1, -1) 
            deg_prior = adj_prior_t.sum(dim=1, keepdim=True).clamp(min=1.0) 
            attn_prior = (attention_dropped * adj_prior_t) / deg_prior 
            h_prior = torch.matmul(attn_prior.transpose(1, 2), h_src_gat) 
            
            adj_new_t = (self.augmented_mask - self.prior_mask).clamp(min=0.0).t().unsqueeze(0).expand(batch_size, -1, -1) 
            if adj_new_t.sum() > 0:
                deg_new = adj_new_t.sum(dim=1, keepdim=True).clamp(min=1.0)
                attn_new = (attention_dropped * adj_new_t) / deg_new
                h_new = torch.matmul(attn_new.transpose(1, 2), h_src_gat)
            else:
                h_new = 0.0 
                
            h_prime = h_prior + h_new + h_dst_gat
            h_gat_out = self.W_out(F.elu(h_prime)).squeeze(-1)
            out = torch.tanh(h_base + self.gamma_gat * h_gat_out)
            return out, torch.tensor(0.0, device=x_src.device)

class GatedLateralCrosstalkLayer(nn.Module):
    def __init__(self, node_dim=1, hidden_dim=16, dropout=0.2, top_k=20): 
        super(GatedLateralCrosstalkLayer, self).__init__()
        self.query = nn.Linear(node_dim, hidden_dim)
        self.key = nn.Linear(node_dim, hidden_dim)
        self.value = nn.Linear(node_dim, node_dim)
        self.dropout = nn.Dropout(dropout)
        self.top_k = top_k 
        self.gamma_cross = nn.Parameter(torch.ones(1) * 0.2) 

    def forward(self, x):
        Q = self.query(x) 
        K = self.key(x)    
        V = self.value(x)  
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)

        if self.top_k is not None and self.top_k < attention_scores.size(-1):
            topk_values, topk_indices = torch.topk(attention_scores, self.top_k, dim=-1)
            mask = torch.full_like(attention_scores, -1e9)
            attention_scores = mask.scatter_(-1, topk_indices, topk_values)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights_dropped = self.dropout(attention_weights)
        crosstalk_msg = torch.matmul(attention_weights_dropped, V) 
        out = x + self.gamma_cross * F.elu(crosstalk_msg)
        return out

class UltimatePNet(nn.Module):
    def __init__(self, n_genes, n_omics, prior_masks, augmented_masks, num_classes, is_discovery_phase=False, use_crosstalk=True, use_base_bypass=True):
        super(UltimatePNet, self).__init__()
        self.is_discovery_phase = is_discovery_phase
        self.use_crosstalk = use_crosstalk
        self.use_base_bypass = use_base_bypass
        self.node_dim = 1
        self.omics_attention = RobustOmicsFusion(n_genes, n_omics)
        
        self.drop0 = nn.Dropout(0.2) 
        self.out0 = nn.Linear(n_genes, num_classes)
        
        self.pathway_layers = nn.ModuleList()
        self.out_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for prior_m, aug_m in zip(prior_masks, augmented_masks):
            n_src_nodes, n_dst_nodes = prior_m.shape
            self.pathway_layers.append(DualStreamOrthogonalGATLayer(n_src_nodes, n_dst_nodes, prior_m, aug_m, is_discovery_phase))
            self.dropouts.append(nn.Dropout(0.3))
            self.out_layers.append(nn.Linear(n_dst_nodes, num_classes))
            
        if self.use_crosstalk:
            self.crosstalk_layers = nn.ModuleList([
                GatedLateralCrosstalkLayer(node_dim=self.node_dim, hidden_dim=16, top_k=20)
                for _ in range(len(augmented_masks))
            ])

    def get_orthogonal_loss(self):
        if not self.is_discovery_phase:
            return torch.tensor(0.0, device=self.pathway_layers[0].W_base.device)
        ortho_loss = 0.0
        for layer in self.pathway_layers:
            W = layer.W_base * layer.augmented_mask 
            W_norm = F.normalize(W, p=2, dim=1)
            cosine_sim = torch.mm(W_norm, W_norm.t())
            eye = torch.eye(cosine_sim.size(0), device=cosine_sim.device)
            ortho_loss += torch.norm(cosine_sim - eye, p='fro')
        return ortho_loss

    def forward(self, x):
        out_preds = []
        total_struct_loss = 0.0
        h = self.omics_attention(x)
        current_h = self.drop0(h)
        
        if self.use_base_bypass:
            out_preds.append(self.out0(current_h))
        
        for i in range(len(self.pathway_layers)):
            current_h, struct_loss = self.pathway_layers[i](current_h)
            total_struct_loss += struct_loss
            
            if self.use_crosstalk:
                current_h_ext = current_h.unsqueeze(-1)
                current_h_ext = self.crosstalk_layers[i](current_h_ext)
                current_h = current_h_ext.squeeze(-1)
                
            current_h_drop = self.dropouts[i](current_h)
            out_preds.append(self.out_layers[i](current_h_drop))
            
        final_logits = torch.mean(torch.stack(out_preds, dim=1), dim=1)
        return final_logits, total_struct_loss
