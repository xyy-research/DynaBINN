"""Reactome hierarchy and multi-omics construction. Input matrices are supplied by the user."""
import os
import copy
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_dataset(input_root, dataset_name, cohort_files):
    INPUT_ROOT = Path(input_root).expanduser().resolve()
    REACTOME_DIR = INPUT_ROOT / "data"
    COHORT_FILES = cohort_files
    def input_path(relative):
        path = INPUT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing input file: {path}")
        return path

    print("Loading Reactome pathway hierarchy...")
    data = pd.read_csv(input_path('data/ReactomePathwaysRelation_new_download.txt'), sep='\t', header=None)
    data.columns = ['child', 'parent']
    human_hierarchy = data[data['child'].str.contains('HSA')] 
    net = nx.from_pandas_edgelist(human_hierarchy, 'child', 'parent', create_using=nx.DiGraph()) 
    net.name = 'reactome'
    roots = [n for n, d in net.in_degree() if d == 0] 
    root_node = 'root'
    net.add_edges_from([(root_node, n) for n in roots]) 

    def get_nodes_at_level(net, distance):
        nodes = set(nx.ego_graph(net, 'root', radius=distance))
        if distance >= 1.:
            nodes -= set(nx.ego_graph(net, 'root', radius=distance - 1))
        return sorted(list(nodes))

    net_nodes = [get_nodes_at_level(net, i) for i in range(1, 6)]

    def add_node(net, net_nodes):
        for i in range(len(net_nodes) - 2, -1, -1):
            data_temp = copy.deepcopy(net_nodes[i])
            for n in net_nodes[i]:
                nexts = net.successors(n)        
                temp = [nex for nex in nexts] 
                if len(temp) == 0 or len(set(temp).intersection(set(net_nodes[i + 1]))) == 0:
                    data_temp.remove(n)
            net_nodes[i] = data_temp
        return net_nodes
    net_nodes = add_node(net, net_nodes)

    def get_note_relation(net_nodes):
        node_mat = []
        for i in range(len(net_nodes) - 1):
            dicts = {n: [nex for nex in net.successors(n) if nex in net_nodes[i + 1]] for n in net_nodes[i]}
            mat = np.zeros((len(net_nodes[i]), len(net_nodes[i + 1]))) 
            for p, gs in dicts.items():    
                mat[net_nodes[i].index(p), [net_nodes[i + 1].index(g) for g in gs]] = 1
            node_mat.append(pd.DataFrame(mat, index=net_nodes[i], columns=net_nodes[i + 1]).T)
        return node_mat
    Get_Node_relation = get_note_relation(net_nodes)

    def load_data_dict(filename):
        dict_map = {}
        with open(filename) as gmt:
            for row in gmt:
                genes = [i.replace('\n', '') for i in row.split('\t')]
                dict_map[genes[1]] = genes[3:]
        return dict_map
    gene_data = load_data_dict(input_path('data/ReactomePathways.gmt'))

    # ==========================================
    # 2. Multi-omics loading and sample alignment
    # ==========================================
    print("\nLoading multi-omics data and labels...")
    for key in ('snv', 'cnv_amp', 'cnv_del', 'mrna', 'labels'):
        if key not in COHORT_FILES:
            raise KeyError(f"Missing {key!r} for {args.dataset} in {DATASET_CONFIG}")

    snv_data = pd.read_csv(input_path(COHORT_FILES['snv']), index_col=0)
    amp_data = pd.read_csv(input_path(COHORT_FILES['cnv_amp']), index_col=0)
    del_data = pd.read_csv(input_path(COHORT_FILES['cnv_del']), index_col=0)
    mrna_data = pd.read_csv(input_path(COHORT_FILES['mrna']), index_col=0)
    response = pd.read_csv(input_path(COHORT_FILES['labels']), index_col=0)

    snv_data = snv_data.loc[~snv_data.index.duplicated(keep='first'), ~snv_data.columns.duplicated(keep='first')]
    amp_data = amp_data.loc[~amp_data.index.duplicated(keep='first'), ~amp_data.columns.duplicated(keep='first')]
    del_data = del_data.loc[~del_data.index.duplicated(keep='first'), ~del_data.columns.duplicated(keep='first')]
    mrna_data = mrna_data.loc[~mrna_data.index.duplicated(keep='first'), ~mrna_data.columns.duplicated(keep='first')]
    response = response.loc[~response.index.duplicated(keep='first')]

    common_samples = response.index.intersection(snv_data.index).intersection(amp_data.index)\
                     .intersection(del_data.index).intersection(mrna_data.index)
    print(f"Aligned {len(common_samples)} samples across 4 modalities.")
    response = response.loc[common_samples]
    snv_data = snv_data.loc[common_samples]
    amp_data = amp_data.loc[common_samples]
    del_data = del_data.loc[common_samples]
    mrna_data = mrna_data.loc[common_samples]

    pathways = list(gene_data.keys())  
    pathway_union = sorted(list(set(Get_Node_relation[3].index).intersection(set(pathways))))
    Get_Node_relation[3] = Get_Node_relation[3].loc[pathway_union]

    all_pathway_genes = set(g for p in pathway_union for g in gene_data.get(p, []))
    common_omics_genes = set(snv_data.columns) & set(amp_data.columns) & set(del_data.columns) & set(mrna_data.columns)
    final_genes = common_omics_genes.intersection(all_pathway_genes)

    gene_file_path = REACTOME_DIR / 'protein-coding_gene_with_coordinate_minimal.txt'
    if os.path.exists(gene_file_path):
        coding_genes_df = pd.read_csv(gene_file_path, sep='\t', header=None)
        final_genes = final_genes.intersection(set(coding_genes_df.iloc[:, 3].unique()))
    final_genes = sorted(list(final_genes))

    # ==========================================
    # 3. 构造先验矩阵与 3D 张量，自动判断任务类型
    # ==========================================
    pathways_gene = np.zeros((len(pathway_union), len(final_genes))) 
    for p_idx, p_name in enumerate(pathway_union):
        gs = gene_data.get(p_name, [])      
        g_inds = [final_genes.index(g) for g in gs if g in final_genes]  
        pathways_gene[p_idx, g_inds] = 1

    prior_h0_matrix = pd.DataFrame(pathways_gene, index=pathway_union, columns=final_genes).values.T 

    snv_np = np.nan_to_num(snv_data.loc[common_samples, final_genes].values, nan=0.0)
    amp_np = np.nan_to_num(amp_data.loc[common_samples, final_genes].values, nan=0.0)
    del_np = np.nan_to_num(del_data.loc[common_samples, final_genes].values, nan=0.0)
    mrna_raw = np.nan_to_num(mrna_data.loc[common_samples, final_genes].values, nan=0.0)

    # Apply the original per-sample log1p transform; fit mRNA MinMaxScaler inside each fold.
    mrna_np = np.log1p(mrna_raw)

    X_all = np.stack([snv_np, amp_np, del_np, mrna_np], axis=-1).astype(np.float32)

    # 自动处理标签连续化，防止跨数据集标签格式跳跃
    label_encoder = LabelEncoder()
    Y_all = label_encoder.fit_transform(response['label'].values).astype(np.int64)
    NUM_CLASSES = len(np.unique(Y_all))
    IS_BINARY = (NUM_CLASSES == 2)
    print(f"Task Detected: {'Binary Classification' if IS_BINARY else 'Multi-class Classification'} ({NUM_CLASSES} Classes)")

    original_pathway_masks = [
        prior_h0_matrix,
        Get_Node_relation[3].values,
        Get_Node_relation[2].values,
        Get_Node_relation[1].values,
        Get_Node_relation[0].values
    ]

    return {
        "X_all": X_all, "Y_all": Y_all,
        "NUM_CLASSES": NUM_CLASSES, "IS_BINARY": IS_BINARY,
        "original_pathway_masks": original_pathway_masks,
        "prior_h0_matrix": prior_h0_matrix,
        "Get_Node_relation": Get_Node_relation,
        "final_genes": final_genes, "pathway_union": pathway_union,
        "label_encoder": label_encoder
    }
