"""DynaBINN two-stage, five-fold main experiment."""
import os
import gc
import copy
from pathlib import Path
import numpy as np
import joblib
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from .model import UltimatePNet
from .metrics import evaluate_task
from .losses import DynamicFocalLoss
from .utils import SEED, seed_everything

def run_experiment(bundle, save_dir, device):
    SAVE_DIR = str(Path(save_dir))
    os.makedirs(SAVE_DIR, exist_ok=True)
    X_all = bundle["X_all"]
    Y_all = bundle["Y_all"]
    NUM_CLASSES = bundle["NUM_CLASSES"]
    IS_BINARY = bundle["IS_BINARY"]
    original_pathway_masks = bundle["original_pathway_masks"]
    prior_h0_matrix = bundle["prior_h0_matrix"]
    Get_Node_relation = bundle["Get_Node_relation"]
    final_genes = bundle["final_genes"]
    pathway_union = bundle["pathway_union"]
    label_encoder = bundle["label_encoder"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10590)
    EPOCHS_DISCOVERY = 50 
    EPOCHS_FULL = 1500
    BATCH_SIZE = 64

    fold_results = []

    for p, (train_index, test_index) in enumerate(skf.split(X_all, Y_all)):
        print(f"\n\n{'='*30} FOLD {p+1}/5 {'='*30}")
        
        # ----------------------------------------------------------------
        # Fit mRNA scaler and class weights on the optimization split only
        # ----------------------------------------------------------------
        X_train_fold_raw, Y_train_fold = X_all[train_index], Y_all[train_index]
        X_test_raw, Y_test = X_all[test_index], Y_all[test_index]
        X_train_raw, X_val_raw, Y_train, Y_val = train_test_split(
            X_train_fold_raw, Y_train_fold, test_size=0.15, stratify=Y_train_fold, random_state=SEED + p
        )

        # Fit the mRNA scaler on the optimization split (channel 3).
        scaler_mrna = MinMaxScaler(feature_range=(0, 1))
        X_train = X_train_raw.copy().astype(np.float32)
        X_val   = X_val_raw.copy().astype(np.float32)
        X_test  = X_test_raw.copy().astype(np.float32)
        X_train[:, :, 3] = scaler_mrna.fit_transform(X_train_raw[:, :, 3]).astype(np.float32)
        X_val[:, :, 3]   = scaler_mrna.transform(X_val_raw[:, :, 3]).astype(np.float32)
        X_test[:, :, 3]  = scaler_mrna.transform(X_test_raw[:, :, 3]).astype(np.float32)

        # Compute class weights from the optimization labels.
        fold_class_counts = np.bincount(Y_train, minlength=NUM_CLASSES).astype(np.float64)
        fold_class_counts = np.maximum(fold_class_counts, 1.0)  # Avoid division by zero for absent classes.
        fold_weights = np.sqrt(len(Y_train) / (NUM_CLASSES * fold_class_counts))
        fold_alpha = torch.tensor(fold_weights, dtype=torch.float32).to(device)
        criterion = DynamicFocalLoss(alpha=fold_alpha, gamma=1.5)

        # ----------------------------------------------------------------
        # Step A: Topology discovery
        # ----------------------------------------------------------------
        print(f"--> [Step A] Topology Discovery (Epochs: {EPOCHS_DISCOVERY})...")
        seed_everything(SEED + p)
        
        discovery_model = UltimatePNet(
            n_genes=len(final_genes), n_omics=4, prior_masks=original_pathway_masks, augmented_masks=original_pathway_masks, 
            num_classes=NUM_CLASSES, is_discovery_phase=True, use_crosstalk=True, use_base_bypass=True
        ).to(device)

        base_params = [param for name, param in discovery_model.named_parameters() if 'W_adj' not in name]
        adj_params = [param for name, param in discovery_model.named_parameters() if 'W_adj' in name]
        discovery_optimizer = optim.Adam([
            {'params': base_params, 'weight_decay': 1e-3, 'lr': 1e-3},
            {'params': adj_params, 'weight_decay': 0.0, 'lr': 5e-3} 
        ])

        discovery_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train)) 
        discovery_loader = DataLoader(discovery_dataset, batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(EPOCHS_DISCOVERY):
            discovery_model.train()
            for batch_x, batch_y in discovery_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                discovery_optimizer.zero_grad()
                logits, struct_loss = discovery_model(batch_x)
                pred_loss = criterion(logits, batch_y)
                ortho_loss = discovery_model.get_orthogonal_loss()
                loss = pred_loss + 1e-4 * ortho_loss + 5e-3 * struct_loss
                loss.backward()
                discovery_optimizer.step()

        # Select fold-specific added edges and construct augmented masks.
        discovery_model.eval()
        learned_weights = torch.sigmoid(discovery_model.pathway_layers[0].W_adj).detach().cpu().numpy()
        prior_mask = discovery_model.pathway_layers[0].prior_mask.detach().cpu().numpy()
        dst_indices, src_indices = np.where(prior_mask == 0)
        weights = learned_weights[dst_indices, src_indices]
        
        top_indices = np.argsort(weights)[::-1][:60] 
        fold_augmented_matrix = copy.deepcopy(prior_h0_matrix) 
        for idx in top_indices:
            fold_augmented_matrix[src_indices[idx], dst_indices[idx]] = weights[idx]

        fold_augmented_masks = [
            fold_augmented_matrix, Get_Node_relation[3].values,
            Get_Node_relation[2].values, Get_Node_relation[1].values, Get_Node_relation[0].values
        ]

        del discovery_model, discovery_optimizer, discovery_loader, discovery_dataset
        gc.collect()
        torch.cuda.empty_cache()
        
        # ----------------------------------------------------------------
        # Step B: Predictive training
        # ----------------------------------------------------------------
        print(f"--> [Step B] Training Full Model (Max Epochs: {EPOCHS_FULL})...")
        seed_everything(SEED + p)
        
        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        model = UltimatePNet(
            n_genes=len(final_genes), n_omics=4, prior_masks=original_pathway_masks, augmented_masks=fold_augmented_masks,
            num_classes=NUM_CLASSES, is_discovery_phase=False, use_crosstalk=True, use_base_bypass=True
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
        
        best_val_loss = float('inf')
        best_model_state = None
        patience = 40 
        patience_counter = 0
        min_delta = 1e-4

        for epoch in range(EPOCHS_FULL):
            model.train()
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits, _ = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            model.eval()
            with torch.no_grad():
                val_x_tensor = torch.tensor(X_val).to(device)
                val_y_tensor = torch.tensor(Y_val, dtype=torch.int64).to(device)
                val_logits, _ = model(val_x_tensor)
                val_loss = criterion(val_logits, val_y_tensor).item()
                
                val_preds_prob = F.softmax(val_logits, dim=1).cpu().numpy()
                val_metrics = evaluate_task(Y_val, val_preds_prob, NUM_CLASSES)
                display_metric_name = "F1" if IS_BINARY else "Macro-F1"
                display_metric_val = val_metrics[display_metric_name]

                if val_loss < (best_val_loss - min_delta):
                    best_val_loss = val_loss
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

            if (epoch + 1) % 50 == 0 or epoch == EPOCHS_FULL - 1:
                print(f"Epoch {epoch+1:04d}/{EPOCHS_FULL} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Best Val Loss: {best_val_loss:.4f} | (Val {display_metric_name}: {display_metric_val:.4f})")

            if patience_counter >= patience:
                print(f" Early stopping triggered at epoch {epoch+1}. Best Val Loss: {best_val_loss:.4f}")
                break

        # Evaluate the held-out test partition.
        model.load_state_dict(best_model_state)
        model.to(device)
        model.eval()
        with torch.no_grad():
            test_logits, _ = model(torch.tensor(X_test).to(device))
            test_preds_prob = F.softmax(test_logits, dim=1).cpu().numpy()

        fold_res_dict = evaluate_task(Y_test, test_preds_prob, NUM_CLASSES)
        fold_results.append(fold_res_dict)

        # Report fold metrics.
        print("Fold Test Results:", ", ".join([f"{k} = {v:.3f}" for k, v in fold_res_dict.items()]))

        # Save model and fold results.
        fold_save_dir = os.path.join(SAVE_DIR, f"fold_{p+1}")
        os.makedirs(fold_save_dir, exist_ok=True)
        
        # Save model weights.
        torch.save({
            'model_state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
            'fold': p + 1,
            'best_val_loss': best_val_loss,
        }, os.path.join(fold_save_dir, 'best_model.pt'))
        
        # Save prior and augmented masks.
        np.save(os.path.join(fold_save_dir, 'prior_mask_layer0.npy'), original_pathway_masks[0])
        np.save(os.path.join(fold_save_dir, 'augmented_mask_layer0.npy'), fold_augmented_masks[0])
        
        # Save held-out sample indices, labels, and predictions.
        np.save(os.path.join(fold_save_dir, 'test_indices.npy'), test_index)
        np.save(os.path.join(fold_save_dir, 'test_labels.npy'), Y_test)
        np.save(os.path.join(fold_save_dir, 'test_predictions.npy'), test_preds_prob)

        # Save the fold-specific scaler for evaluation/inference reproducibility
        joblib.dump(scaler_mrna, os.path.join(fold_save_dir, 'scaler_mrna.pkl'))
        
        print(f"   Saved to {fold_save_dir}")
        # End fold outputs.

        del model, optimizer, train_loader, train_dataset
        gc.collect() 
        torch.cuda.empty_cache()

    # ====================================================================
    # Summarize five-fold metrics.
    # ====================================================================
    print("\n" + "="*80)
    print(f"FINAL 5-FOLD CV RESULTS: DynaBINN ({'Binary' if IS_BINARY else 'Multi-Class'})")
    print("="*80)

    keys = fold_results[0].keys()
    for key in keys:
        metric_values = [fold[key] for fold in fold_results]
        mean_val = np.mean(metric_values)
        std_val = np.std(metric_values)
        print(f"{key:<14}: {mean_val:.3f} ± {std_val:.3f}")
    print("="*80)

    # Save cohort metadata.
    print(f"\nSaving global metadata to {SAVE_DIR}")

    # X_all includes log1p-transformed mRNA before fold-specific MinMax scaling.
    # Apply the saved fold-specific scaler for downstream inference.
    np.savez(os.path.join(SAVE_DIR, 'global_info.npz'), 
             final_genes=np.array(final_genes),
             pathway_union=np.array(pathway_union),
             num_classes=NUM_CLASSES,
             is_binary=IS_BINARY,
             label_classes=label_encoder.classes_)

    np.save(os.path.join(SAVE_DIR, 'X_all.npy'), X_all)
    np.save(os.path.join(SAVE_DIR, 'Y_all.npy'), Y_all)

    print(f"All data saved to {SAVE_DIR}")
