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


class SymmetryHybridPipeline:
    """
    A pipeline for training and evaluating a symmetry detection model using a hybrid approach.
    Gets the latent space of a given model and trains a classifier on top of it.
    """
    def __init__(self, sae_texture_encoder, scaler, classifier, device="cpu"):
        self.sae_texture_encoder = sae_texture_encoder.to(device).eval()
        self.scaler = scaler
        self.classifier = classifier
        self.device = device

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
        features = self.build_symmetry_features(dataset)
        return self.score_features(features)


class SymmetryUnsupervisedPipeline(SymmetryHybridPipeline):
    """
    A pipeline for training and evaluating a symmetry detection model using an unsupervised approach.
    Inherits from SymmetryHybridPipeline.
    """
    def __init__(self, sae_texture_encoder, scaler, classifier, device="cpu"):
        super().__init__(sae_texture_encoder, scaler, classifier, device)

    def fit_from_dataset(self, normal_dataset, anomalous_dataset):
        """
        Fits the unsupervised anomaly detection classifier on the given normal and anomalous datasets.
        
        Args:
            normal_dataset (torch.utils.data.Dataset): The normal dataset.
            anomalous_dataset (torch.utils.data.Dataset): The anomalous dataset.
        Returns:
            sklearn.base.BaseEstimator: The fitted classifier.
        """
        X_normal = self.build_symmetry_features(normal_dataset)
        X_anomalous = self.build_symmetry_features(anomalous_dataset)
        return self.fit_from_features(X_normal, X_anomalous)

    def evaluate_dataset(self, normal_dataset, anomalous_dataset, to_binary=True):
        """
        Evaluates the unsupervised anomaly detection classifier on the given normal and anomalous datasets.
        
        Args:
            normal_dataset (torch.utils.data.Dataset): The normal dataset.
            anomalous_dataset (torch.utils.data.Dataset): The anomalous dataset.
            to_binary (bool): Whether to binarize the scores or not.
        Returns:
            ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)): The normal and anomalous scores and ground truths.
        """
        y_normal = self.score_dataset(normal_dataset)
        y_normal_gt = np.zeros_like(y_normal)
        y_anomalous = self.score_dataset(anomalous_dataset)
        y_anomalous_gt = np.ones_like(y_anomalous)
        if to_binary:
            y_normal, _ = binarize_scores_opt(y_normal_gt, y_normal)
            y_anomalous, _ = binarize_scores_opt(y_anomalous_gt, y_anomalous)
        return (y_normal, y_normal_gt), (y_anomalous, y_anomalous_gt)
    
    def evaluate_features(self, normal_features, anomalous_features):
        y_normal_gt = np.zeros(normal_features.shape[0])
        y_anomalous_gt = np.ones(anomalous_features.shape[0])
        y_normal_pred = self.score_features(normal_features)
        y_anomalous_pred = self.score_features(anomalous_features)
        return (y_normal_pred, y_normal_gt), (y_anomalous_pred, y_anomalous_gt)

    def fit_from_features(self, normal_features, anomalous_features=None):
        """
        Fits the unsupervised anomaly detection classifier on the given normal and anomalous features.
        
        Args:
            normal_features (np.ndarray): The normal features.
            anomalous_features (np.ndarray): The anomalous features.
        Returns:
            sklearn.base.BaseEstimator: The fitted classifier.
        """
        X = normal_features
        if anomalous_features is not None:
            X = np.concatenate([X, anomalous_features])
        X = self.scaler.fit_transform(X)
        self.classifier.fit(X)
        return self.classifier


class SymmetryClassifierPipeline(SymmetryHybridPipeline):
    """
    A pipeline for training and evaluating a symmetry detection model using a supervised approach.
    """
    def __init__(self, sae_texture_encoder, scaler, classifier, device="cpu"):
        super().__init__(sae_texture_encoder, scaler, classifier, device)

    def predict_features(self, features):
        """
        Predicts the class of the given features.
        
        Args:
            features (np.ndarray): The features to predict.
        Returns:
            np.ndarray: The class ids
        """
        X = self.scaler.transform(features)
        return self.classifier.predict_proba(X)
    
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
        y_normal_pred = self.predict_features(normal_features)
        y_anomalous_pred = self.predict_features(anomalous_features)
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


class SymmetryMultimodalUnsupervisedPipeline(SymmetryUnsupervisedPipeline):
    """
    A pipeline for training and evaluating a multimodal symmetry detection model using an unsupervised approach.
    Modalities are tabular data and image features.
    """
    def __init__(self, sae_texture_encoder, scaler, classifier, vec_dist: VecDist, device="cpu"):
        super().__init__(sae_texture_encoder, scaler, classifier, device)
        try: 
            VecDist(vec_dist) 
        except: 
            ValueError()
        self.vec_dist = vec_dist
        
    
    def build_features(self, dataset, proj_scaler=None, projector=None):
        """
        Builds the tabular and image features for the given dataset.
        The image features are a distance metric between the left and right image textures.
        
        Args:
            dataset (torch.utils.data.Dataset): The dataset to build the features from.
        Returns:
            np.ndarray: The tabular and image features.
        """
        
        pat_data_feats = []
        tex_feats = []
        for i in range(len(dataset)):
            l, r, pat_data = dataset[i]
            _, l_tex = self.sae_texture_encoder(l.unsqueeze(0).cuda(), run_str=False, multi_tex=False)
            _, r_tex = self.sae_texture_encoder(r.unsqueeze(0).cuda(), run_str=False, multi_tex=False)
            if self.vec_dist == VecDist.EUCLIDIAN:
                tex_dist = torch.norm((l_tex - r_tex), dim=1)
            elif self.vec_dist == VecDist.COSINE:
                tex_dist = cosine_similarity(l_tex, r_tex)
            elif self.vec_dist == VecDist.ABS:
                tex_dist = torch.abs(l_tex - r_tex).squeeze()
            else:
                tex_dist = torch.einsum('ij,ij->i', l_tex, r_tex)
            
            tex_dist = np.expand_dims(tex_dist.cpu().numpy(), 0)
            pat_data_feats.append(pat_data)
            tex_feats.append(tex_dist)
            
        pat_data_feats = np.array(pat_data_feats).squeeze(1)
        tex_feats = np.array(tex_feats).squeeze(1)
        if self.vec_dist == VecDist.ABS and proj_scaler is not None and projector is not None:
            try:
                check_is_fitted(proj_scaler)
            except:
                proj_scaler.fit(tex_feats)
                
            try:
                check_is_fitted(projector)
            except:
                projector.fit(tex_feats)

            tex_feats = proj_scaler.transform(tex_feats)
            tex_feats = projector.transform(tex_feats)

        return np.concatenate([pat_data_feats, tex_feats], axis=1)
            
    def build_symmetry_features(self, dataset, proj_scaler=None, projector=None):
        """
        Builds the symmetry features, placeholder for compatibility.
        
        Args:
            dataset (torch.utils.data.Dataset): The dataset to build the features from.
        Returns:
            np.ndarray: The symmetry features.
        """
        return self.build_features(dataset, proj_scaler, projector)
