"""
'utf-8'
dbscan_clusterization.py

-------------------------------------------------------------------------------------------------- xx
Author: Pedro Ruiz
Creation: 10-01-2024
Version: 2.1
-------------------------------------------------------------------------------------------------- xx

This module contains the implementation of the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm for clustering tasks.

The model uses specified features from the dataset to perform DBSCAN clustering and save the results and the trained model.

Classes:
- DBSCANClusterizationModel: encapsulates the logic of the DBSCAN model.

Main methods:
- __init__: Initializes the model with the specified configuration.
- fit: Trains the DBSCAN model with the data provided and calculates clustering metrics.
- save: Saves the trained DBSCAN model.
- load: Loads a previously trained DBSCAN model.
- run_training: Executes the training process, saves the model, and the clustering results.

The model uses a configuration file (train.conf) to specify hyperparameters, data paths, and other relevant options.
"""

import os
import configparser
import argparse
import pandas as pd
import json
import logging
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pickle


def ensure_dir_exists(path):
    os.makedirs(path, exist_ok=True)


class DBSCANClusterizationModel:
    def __init__(self, config_path, logger=None):
        """
        Initializes DBSCAN with the specific configuration.

        Parameters:
            config_path (str): Path to the configuration file.
            logger (logging.Logger, optional): Logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)
        
        config = configparser.ConfigParser()
        config.read(config_path)

        # section of the configuration file
        self.config_section = 'dbscan_clusterization'

        # Configurations
        self.eps = config.getfloat(self.config_section, 'eps')
        self.min_samples = config.getint(self.config_section, 'min_samples')
        self.metric = config.get(self.config_section, 'metric')
        self.algorithm = config.get(self.config_section, 'algorithm')

        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric, algorithm=self.algorithm)

        # Paths
        self.data_path = config['pca_clus']['transformed_data_output']
        self.model_path = config['dbscan_clus']['output_model']
        self.metrics_output = config['dbscan_clus']['metrics_output']
        self.results_output = 'models/clusterization/results/dbscan_results.csv'

        self.logger.info(f"DBSCAN initialized with: '{self.config_section}'.")

    def fit(self, X):
        """
        Trains the DBSCAN model.

        Parameters:
            X (array-like): Data for clustering.

        Returns:
            tuple: (labels, model metrics)
        """
        self.logger.info("Training DBSCAN.")
        labels = self.dbscan.fit_predict(X)
        self.logger.info("DBSCAN training completed.")

        # Metrics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        metrics = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'eps': self.eps,
            'min_samples': self.min_samples,
            'metric': self.metric,
            'algorithm': self.algorithm
        }

        # Calculate quality metrics if there are clusters
        if n_clusters > 1:
            X_core = X[labels != -1]
            labels_core = labels[labels != -1]
            if len(X_core) > 0:
                silhouette_avg = silhouette_score(X_core, labels_core)
                calinski_harabasz = calinski_harabasz_score(X_core, labels_core)
                davies_bouldin = davies_bouldin_score(X_core, labels_core)
                metrics.update({
                    'silhouette_score': silhouette_avg,
                    'calinski_harabasz_score': calinski_harabasz,
                    'davies_bouldin_score': davies_bouldin
                })
                self.logger.info(f"Calculated quality metrics: {metrics}")
            else:
                self.logger.warning("There are not enough non-noisy points to calculate metrics.")
        else:
            self.logger.warning("No. insufficient number of clusters.")

        return labels, metrics

    def save(self, model_path):
        """
        Saves the trained model.

        Parameters:
            model_path (str): Path to save.
        """
        self.logger.info(f"Saving DBSCAN to {model_path}.")
        ensure_dir_exists(os.path.dirname(model_path))
        with open(model_path, 'wb') as f:
            pickle.dump(self.dbscan, f)
        self.logger.debug("DBSCAN saved successfully.")

    def load(self, model_path):
        """
        Loads a previously trained model.

        Parameters:
            model_path (str): Path to the file
        """
        self.logger.info(f"Loading DBSCAN from {model_path}.")
        with open(model_path, 'rb') as f:
            self.dbscan = pickle.load(f)
        self.logger.debug("DBSCAN loaded successfully.")

    def run_training(self):
        """
        Executes the complete training process, saving the model and the results.
        """
        # charge data from PCA
        self.logger.info(f"Loading PCA data from {self.data_path}.")
        with open(self.data_path, 'rb') as f:
            X = pickle.load(f)

        # Entrenar el modelo
        labels, metrics = self.fit(X)

        # Guardar el modelo
        self.save(self.model_path)

        # Guardar las etiquetas de cluster
        ensure_dir_exists(os.path.dirname(self.results_output))
        results_df = pd.DataFrame({'cluster_label': labels})
        results_df.to_csv(self.results_output, index=False)
        self.logger.info(f"Clustering results saved to: {self.results_output}.")

        # Guardar métricas
        ensure_dir_exists(os.path.dirname(self.metrics_output))
        with open(self.metrics_output, 'w') as f:
            json.dump(metrics, f, indent=4)
        self.logger.info(f"Metrics saved to {self.metrics_output}.")

