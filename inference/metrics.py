import numpy as np
from inference.utils import concat_scores, binarize_scores_opt
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
)


def binarize_scores(norm_anomaly_scores, an_anomaly_scores):
    """
    Binaries the scores and concatenates them into a single array
    
    Args:
        norm_anomaly_scores (list): list of normal scores
        an_anomaly_scores (list): list of anomaly scores
        
    Returns:
        y_true (np.array): array of true labels
        y_pred (np.array): array of predicted scores
    """
    y_true, y_pred = concat_scores(norm_anomaly_scores, an_anomaly_scores)
    _, y_pred = binarize_scores_opt(y_true, y_pred)
    return y_true, y_pred


def roc_auc_from_scores(norm_anomaly_scores, an_anomaly_scores):
    """
    Compute the ROC AUC score from the normal and anomaly scores
    
    Args:
        norm_anomaly_scores (list): list of normal scores
        an_anomaly_scores (list): list of anomaly scores
        
    Returns:
        float: ROC AUC score
    """
    y_true, y_pred = concat_scores(norm_anomaly_scores, an_anomaly_scores)
    return roc_auc_score(y_true, y_pred)


def pr_auc_from_scores(norm_anomaly_scores, an_anomaly_scores):
    """
    Compute the PR AUC score from the normal and anomaly scores
    
    Args:
        norm_anomaly_scores (list): list of normal scores
        an_anomaly_scores (list): list of anomaly scores
        
    Returns:
        float: PR AUC score
    """
    y_true, y_pred = concat_scores(norm_anomaly_scores, an_anomaly_scores)
    prec, rec, _ = precision_recall_curve(y_true, y_pred)
    return auc(rec, prec)


def precision_from_scores(norm_anomaly_scores, an_anomaly_scores):
    """
    Compute the precision score from the normal and anomaly scores
    
    Args:
        norm_anomaly_scores (list): list of normal scores
        an_anomaly_scores (list): list of anomaly scores
        
    Returns:
        float: Precision score
    """
    y_true, y_pred = binarize_scores(norm_anomaly_scores, an_anomaly_scores)
    return precision_score(y_true, y_pred)


def recall_from_scores(norm_anomaly_scores, an_anomaly_scores):
    """
    Compute the recall score from the normal and anomaly scores
    
    Args:
        norm_anomaly_scores (list): list of normal scores
        an_anomaly_scores (list): list of anomaly scores
        
    Returns:
        float: Recall score
    """
    y_true, y_pred = binarize_scores(norm_anomaly_scores, an_anomaly_scores)
    return recall_score(y_true, y_pred)

def f1_from_scores(norm_anomaly_scores, an_anomaly_scores):
    """
    Compute the F1 score from the normal and anomaly scores
    
    Args:
        norm_anomaly_scores (list): list of normal scores
        an_anomaly_scores (list): list of anomaly scores
        
    Returns:
        float: F1 score
    """
    y_true, y_pred = binarize_scores(norm_anomaly_scores, an_anomaly_scores)
    return f1_score(y_true, y_pred)


def specificity_from_scores(norm_anomaly_scores, an_anomaly_scores):
    """
    Compute the specificity score from the normal and anomaly scores    
    
    Args:
        norm_anomaly_scores (list): list of normal scores
        an_anomaly_scores (list): list of anomaly scores
        
    Returns:
        float: Specificity score
    """
    y_true, y_pred = concat_scores(norm_anomaly_scores, an_anomaly_scores)
    _, y_pred = binarize_scores_opt(y_true, y_pred)
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def accuracy_from_scores(norm_anomaly_scores, an_anomaly_scores):
    """
    Compute the accuracy score from the normal and anomaly scores
    
    Args:
        norm_anomaly_scores (list): list of normal scores
        an_anomaly_scores (list): list of anomaly scores
        
    Returns:
        float: Accuracy score
    """
    y_true, y_pred = binarize_scores(norm_anomaly_scores, an_anomaly_scores)
    return accuracy_score(y_true, y_pred)


def classification_report(scores_normal, scores_outliers):
    """
    Print the classification report from the normal and anomaly scores with std 
    of all defined metrics above
    
    Args:
        scores_normal (list): list of normal scores
        scores_outliers (list): list of anomaly scores
        
    Returns:
        None
    """
    if isinstance(scores_normal[0], (int, float)):
        scores_normal = [scores_normal]
        scores_outliers = [scores_outliers]

    acc, spec, prec, rec, f1, auroc, aupr = [], [], [], [], [], [], []
    for norm_arr, ano_arr in zip(scores_normal, scores_outliers):
        acc.append(accuracy_from_scores(norm_arr, ano_arr))
        spec.append(specificity_from_scores(norm_arr, ano_arr))
        prec.append(precision_from_scores(norm_arr, ano_arr))
        rec.append(recall_from_scores(norm_arr, ano_arr))
        f1.append(f1_from_scores(norm_arr, ano_arr))
        auroc.append(roc_auc_from_scores(norm_arr, ano_arr))
        aupr.append(pr_auc_from_scores(norm_arr, ano_arr))

    print(f"Accuracy: {np.mean(acc):.4f}; std {np.std(acc):.4f}")
    print(f"Specificity: {np.mean(spec):.4f}; std {np.std(spec):.4f}")
    print(f"Precision: {np.mean(prec):.4f}; std {np.std(prec):.4f}")
    print(f"Recall: {np.mean(rec):.4f}; std {np.std(rec):.4f}")
    print(f"F1: {np.mean(f1):.4f}; std {np.std(f1):.4f}")
    print(f"AUROC: {np.mean(auroc):.4f}; std {np.std(auroc):.4f}")
    print(f"AUPR: {np.mean(aupr):.4f}; std {np.std(aupr):.4f}")
