"""
utf-8
knn_recommender.py

-------------------------------------------------------------------------------------------------- xx
Author: Pedro Ruiz
Creation : Pedro Ruiz 10-01-2024
Version: 2.1
-------------------------------------------------------------------------------------------------- xx

This module contains the implementation of the K-Nearest Neighbors (KNN) model.

The model uses numerical and textual features processed with TF-IDF to recommend similar videos.

Classes:
- KNNRecommenderModel: encapsulates the logic of the KNN recommendation model.

Main methods:
- __init__: initializes the model with the configuration.
- fit: Fits the model.
- predict: Makes recommendations using the trained model.
- save: Saves the trained model.
- load: Loads the trained model.
- load_video_ids: Loads the video identifiers used for the recommendations.

The model uses a configuration file (config.conf) to specify hyperparameters, data paths, and other options.
"""

import os
import pickle
import configparser
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack, csr_matrix
import logging


class KNNRecommenderModel:
    def __init__(self, config_path, logger=None):
        self.logger = logger or logging.getLogger(__name__)

        config = configparser.ConfigParser()
        config.read(config_path)

        # Validate that sections exist
        hyperparams_section = 'knn_recommendation'
        paths_section = 'knn_rec'

        if hyperparams_section not in config:
            raise ValueError(f"Section '{hyperparams_section}' not found.") 
        if paths_section not in config:
            raise ValueError(f"Section '{paths_section}' not found.")

        # Load hyperparameters
        self.k = config.getint(hyperparams_section, 'k')
        self.metric = config.get(hyperparams_section, 'metric')
        self.algorithm = config.get(hyperparams_section, 'algorithm')
        self.num_features = [feature.strip() for feature in config.get(hyperparams_section, 'num_features').split(',')]
        self.target_feature = config.get(hyperparams_section, 'target_feature')
        self.text_features = [feature.strip() for feature in config.get(hyperparams_section, 'text_features').split(',')]

        # Load paths
        self.video_ids_path = config[paths_section]['video_ids_path']

        # Paths of TF-IDF matrices
        self.tfidf_paths = {}
        for text_feature in self.text_features:
            if text_feature in config['tfidf_paths']:
                tfidf_path = config['tfidf_paths'][text_feature]
                if os.path.exists(tfidf_path):
                    self.tfidf_paths[text_feature] = tfidf_path
                    self.logger.info(f"TF-IDF path for '{text_feature}' loaded: {tfidf_path}.")
                else:
                    self.logger.warning(f"TF-IDF path for '{text_feature}' does not exist: {tfidf_path}. The feature will be skipped.")
            else:
                self.logger.warning(f"TF-IDF path '{text_feature}' not found in [tfidf_paths]. The feature will be skipped.")

        # Initialize KNN
        self.model = NearestNeighbors(n_neighbors=self.k, metric=self.metric, algorithm=self.algorithm)
        self.video_ids = None

        self.logger.info("KNNRecommenderModel initialized successfully.")


    def fit(self, df):
        """
        Fits the KNN model.

        Parameters:
            df (pd.DataFrame)
        """
        self.logger.info("Training KNN Recommender.")

        # Numerical features
        X_num = df[self.num_features]
        X_num_sparse = csr_matrix(X_num.values)

        # TF-IDF matrices
        X_text = []
        for text_feature, tfidf_path in self.tfidf_paths.items():
            with open(tfidf_path, 'rb') as f:
                tfidf_matrix = pickle.load(f)
                X_text.append(tfidf_matrix)
                self.logger.info(f"TF-IDF matrix for '{text_feature}' loaded successfully.")

        if X_text:
            X_combined = hstack([X_num_sparse] + X_text)
            self.logger.info("Combined features.")
        else:
            X_combined = X_num_sparse
            self.logger.warning("No TF-IDF matrices found. Only numerical features will be used.")

        # Train
        self.model.fit(X_combined)
        self.logger.info("KNN model trained successfully.")

        # Save the list of video_ids
        self.video_ids = df[self.target_feature].tolist()
        os.makedirs(os.path.dirname(self.video_ids_path), exist_ok=True)
        with open(self.video_ids_path, 'wb') as f:
            pickle.dump(self.video_ids, f)
        self.logger.info("List of video_ids saved.")


    def save(self, model_path):
        self.logger.info(f"Saving KNN to {model_path}.")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        self.logger.debug("KNN model saved successfully.")


    def load(self, model_path):
        self.logger.info(f"Loading KNN from {model_path}.")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.logger.debug("KNN model loaded successfully.")


    def load_video_ids(self, video_ids_path):
        self.logger.info(f"Loading video_ids from {video_ids_path}.")
        with open(video_ids_path, 'rb') as f:
            self.video_ids = pickle.load(f)
        self.logger.debug("video_ids loaded successfully.")


    def predict(self, video_id, top_k=5):
        """
        Makes recommendations.

        Parameters:
            video_id (str): ID of the requested video.
            top_k (int): Number of recommendations.

        Returns:
            recommended_ids (list): List of recommended IDs.
            distances (list): Distances to the recommended videos.
        """
        if self.video_ids is None:
            raise ValueError("The list of video_ids has not been loaded.")

        if video_id not in self.video_ids:
            raise ValueError(f"Video ID '{video_id}' not found in the data.")

        index = self.video_ids.index(video_id)
        video_features = self.model._fit_X[index]

        # Correct feature vector form
        if isinstance(video_features, csr_matrix):
            video_features = video_features.toarray()
        video_features = video_features.reshape(1, -1)

        # Get more neighbors to cover possible duplicates
        distances, indices = self.model.kneighbors(video_features, n_neighbors=top_k + 1) 

        neighbor_indices = indices.flatten()
        neighbor_distances = distances.flatten()

        recommended_indices = []
        recommended_distances = []
        recommended_video_ids = set()
        recommended_video_ids.add(video_id)

        for idx, dist in zip(neighbor_indices, neighbor_distances):
            vid_id = self.video_ids[idx]
            if vid_id not in recommended_video_ids:
                recommended_indices.append(idx)
                recommended_distances.append(dist)
                recommended_video_ids.add(vid_id)
            if len(recommended_indices) == top_k:
                break

        recommended_ids = [self.video_ids[i] for i in recommended_indices]
        distances = recommended_distances

        return recommended_ids, distances
