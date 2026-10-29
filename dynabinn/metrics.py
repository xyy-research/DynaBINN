"""Metrics for binary and multiclass tasks."""
import numpy as np
from sklearn import metrics

def evaluate_task(y_true, y_pred_prob, num_classes):
    y_pred_class = np.argmax(y_pred_prob, axis=1)
    acc = metrics.accuracy_score(y_true, y_pred_class)

    if num_classes == 2:
        prob_pos = y_pred_prob[:, 1]
        aupr = metrics.average_precision_score(y_true, prob_pos)
        auc = metrics.roc_auc_score(y_true, prob_pos)
        f1 = metrics.f1_score(y_true, y_pred_class, average='binary', zero_division=0)
        return {"AUPR": aupr, "AUC": auc, "Accuracy": acc, "F1": f1}
    else:
        macro_f1 = metrics.f1_score(y_true, y_pred_class, average='macro', zero_division=0)
        weighted_f1 = metrics.f1_score(y_true, y_pred_class, average='weighted', zero_division=0)
        mcc = metrics.matthews_corrcoef(y_true, y_pred_class)
        return {"Macro-F1": macro_f1, "Weighted-F1": weighted_f1, "MCC": mcc, "Accuracy": acc}
