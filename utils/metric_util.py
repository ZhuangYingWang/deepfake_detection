import numpy as np
from sklearn.metrics import roc_auc_score


def calculate_auc(labels: np.ndarray, probs: np.ndarray, true_label: int = 1) -> float:
    if true_label == 0:
        labels = 1 - labels
    auc_score = roc_auc_score(labels, probs)
    return auc_score
