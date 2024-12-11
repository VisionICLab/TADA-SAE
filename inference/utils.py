import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve


def concat_scores(norm_scores, ano_scores):
    """
    Concatenates the normal and anomaly scores into a single array prior to evaluation

    Args:
        norm_scores (list): list of normal scores
        ano_scores (list): list of anomaly scores

    Returns:
        y_true (np.array): array of true labels
        y_pred (np.array): array of predicted scores

    """
    y_true = np.concatenate(
        (np.zeros_like(norm_scores), np.ones_like(ano_scores))
    )
    y_pred = np.concatenate(
        (np.array(norm_scores), np.array(ano_scores))
    )

    return y_true, y_pred


def gmean_from_scores(normal_scores, anomalous_scores):
    y_true, y_pred = concat_scores(normal_scores, anomalous_scores)
    fpr, tpr, ths = roc_curve(y_true, y_pred)
    gms, ix = gmean(fpr, tpr)
    return gms[ix], ths[ix]


def gmean(fpr, tpr):
    gms = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gms)
    return gms, ix


def binarize_scores_opt(y_true, y_scores, metric="gmean"):
    """
    Compute the optimal threshold for a given set of true labels and predicted scores
    optimizing the F1 score and returns the binarized scores
    https://stats.stackexchange.com/questions/518616/how-to-find-the-optimal-threshold-for-the-weighted-f1-score-in-a-binary-classifi

    Args:
        y_true (np.array): array of true labels (0,1)
        y_scores (np.array): array of predicted scores
    Returns:
        opt_th (float): optimal threshold
        y_pred (np.array): array of binarized scores
    """
    if metric not in ["gmean", "f1"]:
        raise ValueError(f'Invalid metric. Supported metrics are: {["gmean", "f1"]}')
    if metric == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * recall * precision / (recall + precision + 1e-6)
        opt_th = thresholds[np.argmax(f1_scores)]
        y_pred = np.where(np.array(y_scores) >= opt_th, 1, 0)
    elif metric == "gmean":
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        _, ix = gmean(fpr, tpr)
        opt_th = thresholds[ix]
        y_pred = np.where(np.array(y_scores) >= opt_th, 1, 0)
    return opt_th, y_pred
