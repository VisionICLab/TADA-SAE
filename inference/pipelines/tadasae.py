import torch
import numpy as np
from sklearn.utils import shuffle
from inference.utils import binarize_scores_opt
from sklearn.base import clone
from torch.nn.functional import cosine_similarity
from enum import Enum
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

class VecDist(Enum):
    EUCLIDIAN = 'euclidian'
    COSINE = 'cosine'
    DOT = 'dot'
    ABS = 'abs'


@torch.no_grad()
def generate_interpolation(sae_encoder, sae_generator, struct_im, text_im, alpha=1):
    structure1, texture1 = sae_encoder(struct_im, multi_tex=False)
    structure2, texture2 = sae_encoder(text_im, multi_tex=False)
    recons_img1 = sae_generator(structure1, texture1)
    recons_img2 = sae_generator(structure2, texture2)

    text_interp_im1 = alpha * texture2 + (1 - alpha) * texture1
    interp_img1 = sae_generator(structure1, text_interp_im1)

    text_interp_im2 = alpha * texture1 + (1 - alpha) * texture2
    interp_img2 = sae_generator(structure2, text_interp_im2)

    return (recons_img1, recons_img2), (interp_img1, interp_img2)


class AnomalyDetectionPipeline:
    def __init__(self, encoder, scaler, classifier):
        self.device = encoder.device
        self.encoder = encoder.eval()
        self.scaler = scaler
        self.classifier=classifier

    def reset(self):
        """
        Resets the classifier and the scaler.
        
        Args:
            None
        Returns:
            None
        """
        self.scaler = clone(self.scaler)
        self.classifier = clone(self.classifier)
    
    def build_features(self, dataset):
        feats = []
        for i in range(len(dataset)):
            im = dataset[i]
            out = self.encoder(im.unsqueeze(0).to(self.device))
            out = out.cpu().detach().numpy()
            feats.append(out)
        
        return np.array(feats).squeeze()
    
    def score_features(self, features):
        """
        Computes the decision function of the classifier on the given features.
        
        Args:
            features (np.ndarray): The features to score.
        Returns:
            np.ndarray: The decision function of the classifier.
        """
        X = self.scaler.transform(features)
        return self.classifier.decision_function(X)
    
    def score_dataset(self, dataset):
        """
        Computes the decision function of the classifier on the given dataset.
        
        Args:
            dataset (torch.utils.data.Dataset): The dataset to score.
        Returns:
            np.ndarray: The decision function of the classifier.
        """
        features = self.build_features(dataset)
        return self.score_features(features)


class SymmetryClassifierPipeline(AnomalyDetectionPipeline):
    """
    A pipeline for training and evaluating a symmetry detection model using a supervised approach.
    """
    def __init__(self, sae_texture_encoder, scaler, classifier, device="cpu"):
        super().__init__(sae_texture_encoder, scaler, classifier, device)
    
    def build_features(self, dataset):
        """
        Builds the left-right image features for the given dataset.
        
        Args:
            dataset (torch.utils.data.Dataset): The dataset to build the features from.
        Returns:
            (np.ndarray, np.ndarray): The left and right image features.
        """
        l_lats, r_lats = [], []
        for i in range(len(dataset)):
            l, r = dataset[i]
            _, l_tex = self.sae_texture_encoder(l.unsqueeze(0).to(self.device), run_str=False, multi_tex=False)
            _, r_tex = self.sae_texture_encoder(r.unsqueeze(0).to(self.device), run_str=False, multi_tex=False)
            l_tex = l_tex.cpu().detach().numpy()
            r_tex = r_tex.cpu().detach().numpy()
            l_lats.append(l_tex)
            r_lats.append(r_tex)
        return np.array(l_lats).squeeze(), np.array(r_lats).squeeze()
    
    def build_symmetry_features(self, dataset):
        """
        Builds the symmetry features for the given dataset by taking 
        the absolute difference between the left and right image features.
        
        Args:
            dataset (torch.utils.data.Dataset): The dataset to build the features from.
        Returns:
            np.ndarray: The symmetry features.
        """
        l_lats, r_lats = self.build_features(dataset)
        return np.abs(l_lats - r_lats)
    
    def score_features(self, features):
        """
        Computes the class probability function of the classifier on the given features.
        
        Args:
            features (np.ndarray): The features to score.
        Returns:
            np.ndarray: The decision function of the classifier.
        """
        X = self.scaler.transform(features)
        return self.classifier.predict_proba(X)
    
    def score_dataset(self, dataset):
        """
        Computes the decision function of the classifier on the given dataset.
        
        Args:
            dataset (torch.utils.data.Dataset): The dataset to score.
        Returns:
            np.ndarray: The decision function of the classifier.
        """
        features = self.build_symmetry_features(dataset)
        return self.score_features(features)

    def fit_from_dataset(self, normal_dataset, anomalous_dataset=None):
        """
        Fits the supervised anomaly detection classifier on the given normal and anomalous datasets.
        
        Args:
            normal_dataset (torch.utils.data.Dataset): The normal dataset.
            anomalous_dataset (torch.utils.data.Dataset): The anomalous dataset.
        Returns:
            sklearn.base.BaseEstimator: The fitted classifier
        """
        X_normal = self.build_symmetry_features(normal_dataset)
        X_anomalous = self.build_symmetry_features(anomalous_dataset)
        self.fit_from_features(X_normal, X_anomalous)

    def fit_from_features(self, normal_features, anomalous_features):
        """
        Fits the supervised anomaly detection classifier on the given normal and anomalous features.
        
        Args:
            normal_features (np.ndarray): The normal features.
            anomalous_features (np.ndarray): The anomalous features.
        Returns:
            sklearn.base.BaseEstimator: The fitted classifier
        """
        X = np.concatenate([normal_features, anomalous_features])
        y_norm = np.zeros(normal_features.shape[0])
        y_anom = np.ones(anomalous_features.shape[0])
        y = np.concatenate([y_norm, y_anom])
        X, y = shuffle(X, y)
        X = self.scaler.fit_transform(X)
        self.classifier.fit(X, y)

    def evaluate_features(self, normal_features, anomalous_features):
        """
        Evaluates the supervised anomaly detection classifier on the given normal and anomalous features.
        
        Args:
            normal_features (np.ndarray): The normal features.
            anomalous_features (np.ndarray): The anomalous features.
        Returns:
            ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)): The normal and anomalous predictions and ground truths.
        """
        y_normal_gt = np.zeros(normal_features.shape[0])
        y_anomalous_gt = np.ones(anomalous_features.shape[0])
        y_normal_pred = self.score_features(normal_features)
        y_anomalous_pred = self.score_features(anomalous_features)
        return (y_normal_pred, y_normal_gt), (y_anomalous_pred, y_anomalous_gt)

    def evaluate_dataset(self, normal_dataset, anomalous_dataset):
        """
        Evaluates the supervised anomaly detection classifier on the given normal and anomalous datasets.
        
        Args:
            normal_dataset (torch.utils.data.Dataset): The normal dataset.
            anomalous_dataset (torch.utils.data.Dataset): The anomalous dataset.
        """
        X_normal = self.build_symmetry_features(normal_dataset)
        X_anomalous = self.build_symmetry_features(anomalous_dataset)
        return self.evaluate_features(X_normal, X_anomalous)
