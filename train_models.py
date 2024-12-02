<<<<<<< HEAD
"""
utf-8
train_models.py
-------------------------------------------------------------------------------------------------- xx
Author: Pedro Ruiz
Creation: 10-01-2024
Version: 2.1
-------------------------------------------------------------------------------------------------- xx

This module trains the selected machine learning models for the following objectives:
- Logistic Regression
- MLP
- LightGBM
- KNN 
- PCA
- DBSCAN

The preprocessed data is loaded from a pickle file specified in the configuration file.
Each model is trained with the data and saved to the path specified in the configuration file.
Test data is also saved for future inference.

Functions:
- setup_logger
- ensure_dir_exists
- train_models
- main

Usage:
This script can be run from the command line as follows:
-->      python train_models.py train.conf 
"""

import os
import configparser
import argparse
import pandas as pd
import pickle
import gc
import logging
from models.classification_categories.logistic_regression import LogisticRegressionModel
from models.classification_video_disabled.mlp import MLPClassifierModel
from models.regression_like_ratio.lightgbm_regressor import LightGBMRegressorModel as LightGBMRegressorLikeDislikeRatioModel
from models.regression_number_of_likes.lightgbm_regressor import LightGBMRegressorModel as LightGBMRegressorNumberOfLikesModel
from models.clusterization.pca_clusterization import PCAClusterizationModel
from models.recommendation.knn_recommender import KNNRecommenderModel
from models.clusterization.dbscan_clusterization import DBSCANClusterizationModel


def ensure_dir_exists(path):
    """Creates directories if they do not exist."""
    os.makedirs(path, exist_ok=True)


def setup_logger(name, level=logging.INFO):
    """
    Configures and returns a logger with the specified name and logging level.
    The logger will output messages to both console and file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Check if handlers are already added to avoid duplicate logs
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = 'logs/train_models.log'
        ensure_dir_exists(os.path.dirname(log_file))
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = False
    return logger


def train_models(config_path):
    """
    Trains all models.

    Parameters:
        config_path (str): Path to the configuration file (train.conf).
        logger (logging.Logger): Logger instance to record events.
    """
    logger = logging.getLogger('train_models_logger')
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Load preprocessed data
    data_path = config['data']['processed_data']
    if not os.path.exists(data_path):
        logger.error(f"The data file does not exist at {data_path}.")
        raise FileNotFoundError(f"The data file does not exist at {data_path}.")
    
    try:
        data = pd.read_pickle(data_path)
        logger.info(f"Data loaded from {data_path}.")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    
    # List of models to train
    models_to_train = [
        {
            'model_type': 'logistic_regression',
            'class': LogisticRegressionModel,
            'config_section': 'logistic_regression',
            'model_save_path': config['paths']['logistic_regression_model'],
            'test_data_paths': {
                'X_test': config['log_reg_clf']['X_test'],
                'y_test': config['log_reg_clf']['y_test']
            }
        },
        {
            'model_type': 'mlp',
            'class': MLPClassifierModel,
            'config_section': 'mlp',
            'model_save_path': config['paths']['mlp_model'],
            'test_data_paths': {
                'X_test': config['mlp_clf']['X_test'],
                'y_test': config['mlp_clf']['y_test']
            }
        },
        {
            'model_type': 'lightgbm_regressor_like_dislike_ratio',
            'class': LightGBMRegressorLikeDislikeRatioModel,
            'config_section': 'lightgbm_reg_two',
            'model_save_path': config['paths']['lightgbm_regression_like_dislike_ratio_model'],
            'test_data_paths': {
                'X_test': config['lightgbm_reg_two_eval']['X_test'],
                'y_test': config['lightgbm_reg_two_eval']['y_test']
            }
        },
        {
            'model_type': 'lightgbm_regressor_number_of_likes',
            'class': LightGBMRegressorNumberOfLikesModel,
            'config_section': 'lightgbm_reg_one',
            'model_save_path': config['paths']['lightgbm_regression_model'],
            'test_data_paths': {
                'X_test': config['lightgbm_reg_one_eval']['X_test'],
                'y_test': config['lightgbm_reg_one_eval']['y_test']
            }
        },
    ]

    # Train models
    for model_info in models_to_train:
        model_type = model_info['model_type']
        model_class = model_info['class']
        config_section = model_info['config_section']
        model_save_path = model_info['model_save_path']
        test_data_paths = model_info['test_data_paths']
        
        try:
            logger.info(f"Starting training for the model '{model_type}'.")
            
            # Instantiate the model 
            model = model_class(config_path=config_path, logger=logger)
            logger.info(f"Model '{model_type}' instantiated successfully.")
            
            # Train the model
            X_test, y_test, report = model.fit(data)
            logger.info(f"Model '{model_type}' trained successfully.")
            
            # Save the trained model
            ensure_dir_exists(os.path.dirname(model_save_path))
            model.save(model_save_path)
            logger.info(f"Model '{model_type}' saved at {model_save_path}.")
            
            # Save data for inference
            X_test_path = test_data_paths['X_test']
            y_test_path = test_data_paths['y_test']
            ensure_dir_exists(os.path.dirname(X_test_path))
            ensure_dir_exists(os.path.dirname(y_test_path))
            
            with open(X_test_path, 'wb') as f:
                pickle.dump(X_test, f)
            y_test.to_pickle(y_test_path)
            logger.info(f"Inference dataset for model '{model_type}' saved successfully.")
            
        except Exception as e:
            logger.error(f"Error training the model '{model_type}': {e}")
            raise
        
        finally:
            # Free memory if the model was created
            if 'model' in locals():
                del model
            if 'X_test' in locals():
                del X_test
            if 'y_test' in locals():
                del y_test
            if 'report' in locals():
                del report
            gc.collect()
            
            logger.debug(f"Memory freed after training the model '{model_type}'.")

    # KNNRecommenderModel
    try:
        logger.info("Starting KNN training.")
        knn_model = KNNRecommenderModel(config_path, logger=logger)
        knn_model.fit(data)
        knn_model_save_path = config['paths']['knn_model']
        ensure_dir_exists(os.path.dirname(knn_model_save_path))
        knn_model.save(knn_model_save_path)
        logger.info(f"KNN model saved at {knn_model_save_path}.")
    except Exception as e:
        logger.error(f"Error during KNN training: {e}")
        raise

    # PCAClusterizationModel
    try:
        logger.info("Starting PCA training.")
        pca_model = PCAClusterizationModel(config_path, logger=logger)
        pca_model.fit(data)
        pca_model_save_path = config['pca_clus']['output_pca_model']
        ensure_dir_exists(os.path.dirname(pca_model_save_path))
        pca_model.save(pca_model_save_path)
        logger.info(f"PCA model saved at {pca_model_save_path}.")
    except Exception as e:
        logger.error(f"Error during PCA training: {e}")
        raise

    # Load PCA data and train DBSCAN
    pca_transformed_data_path = config['pca_clus']['transformed_data_output']
    try:
        logger.info("Loading PCA data for DBSCAN.")
        with open(pca_transformed_data_path, 'rb') as f:
            pca_transformed_data = pickle.load(f)
        
        logger.info("Starting DBSCAN training.")
        dbscan_model = DBSCANClusterizationModel(config_path, logger=logger)
        dbscan_model.run_training()
        dbscan_model_save_path = config['dbscan_clus']['output_model']
        ensure_dir_exists(os.path.dirname(dbscan_model_save_path))
        logger.info(f"DBSCAN model saved at {dbscan_model_save_path}.")
    except Exception as e:
        logger.error(f"Error during DBSCAN training: {e}")
        raise

    logger.info("All models trained and saved successfully.")


def main():
    logger = setup_logger('train_models_logger')
    logger.info("Training started.")
    
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('config_path', type=str, help='Path to the configuration file (train.conf)')
    args = parser.parse_args()
    
    try:
        train_models(args.config_path)
        logger.info("Model training completed.")
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise

if __name__ == "__main__":
=======
"""
utf-8
train_models.py
-------------------------------------------------------------------------------------------------- xx
Author: Pedro Ruiz
Creation: 10-01-2024
Version: 2.1
-------------------------------------------------------------------------------------------------- xx

This module trains the selected machine learning models for the following objectives:
- Logistic Regression
- MLP
- LightGBM
- KNN 
- PCA
- DBSCAN

The preprocessed data is loaded from a pickle file specified in the configuration file.
Each model is trained with the data and saved to the path specified in the configuration file.
Test data is also saved for future inference.

Functions:
- setup_logger
- ensure_dir_exists
- train_models
- main

Usage:
This script can be run from the command line as follows:
-->      python train_models.py train.conf 
"""

import os
import configparser
import argparse
import pandas as pd
import pickle
import gc
import logging
from models.classification_categories.logistic_regression import LogisticRegressionModel
from models.classification_video_disabled.mlp import MLPClassifierModel
from models.regression_like_ratio.lightgbm_regressor import LightGBMRegressorModel as LightGBMRegressorLikeDislikeRatioModel
from models.regression_number_of_likes.lightgbm_regressor import LightGBMRegressorModel as LightGBMRegressorNumberOfLikesModel
from models.clusterization.pca_clusterization import PCAClusterizationModel
from models.recommendation.knn_recommender import KNNRecommenderModel
from models.clusterization.dbscan_clusterization import DBSCANClusterizationModel


def ensure_dir_exists(path):
    """Creates directories if they do not exist."""
    os.makedirs(path, exist_ok=True)


def setup_logger(name, level=logging.INFO):
    """
    Configures and returns a logger with the specified name and logging level.
    The logger will output messages to both console and file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Check if handlers are already added to avoid duplicate logs
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = 'logs/train_models.log'
        ensure_dir_exists(os.path.dirname(log_file))
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = False
    return logger


def train_models(config_path):
    """
    Trains all models.

    Parameters:
        config_path (str): Path to the configuration file (train.conf).
        logger (logging.Logger): Logger instance to record events.
    """
    logger = logging.getLogger('train_models_logger')
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Load preprocessed data
    data_path = config['data']['processed_data']
    if not os.path.exists(data_path):
        logger.error(f"The data file does not exist at {data_path}.")
        raise FileNotFoundError(f"The data file does not exist at {data_path}.")
    
    try:
        data = pd.read_pickle(data_path)
        logger.info(f"Data loaded from {data_path}.")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    
    # List of models to train
    models_to_train = [
        {
            'model_type': 'logistic_regression',
            'class': LogisticRegressionModel,
            'config_section': 'logistic_regression',
            'model_save_path': config['paths']['logistic_regression_model'],
            'test_data_paths': {
                'X_test': config['log_reg_clf']['X_test'],
                'y_test': config['log_reg_clf']['y_test']
            }
        },
        {
            'model_type': 'mlp',
            'class': MLPClassifierModel,
            'config_section': 'mlp',
            'model_save_path': config['paths']['mlp_model'],
            'test_data_paths': {
                'X_test': config['mlp_clf']['X_test'],
                'y_test': config['mlp_clf']['y_test']
            }
        },
        {
            'model_type': 'lightgbm_regressor_like_dislike_ratio',
            'class': LightGBMRegressorLikeDislikeRatioModel,
            'config_section': 'lightgbm_reg_two',
            'model_save_path': config['paths']['lightgbm_regression_like_dislike_ratio_model'],
            'test_data_paths': {
                'X_test': config['lightgbm_reg_two_eval']['X_test'],
                'y_test': config['lightgbm_reg_two_eval']['y_test']
            }
        },
        {
            'model_type': 'lightgbm_regressor_number_of_likes',
            'class': LightGBMRegressorNumberOfLikesModel,
            'config_section': 'lightgbm_reg_one',
            'model_save_path': config['paths']['lightgbm_regression_model'],
            'test_data_paths': {
                'X_test': config['lightgbm_reg_one_eval']['X_test'],
                'y_test': config['lightgbm_reg_one_eval']['y_test']
            }
        },
    ]

    # Train models
    for model_info in models_to_train:
        model_type = model_info['model_type']
        model_class = model_info['class']
        config_section = model_info['config_section']
        model_save_path = model_info['model_save_path']
        test_data_paths = model_info['test_data_paths']
        
        try:
            logger.info(f"Starting training for the model '{model_type}'.")
            
            # Instantiate the model 
            model = model_class(config_path=config_path, logger=logger)
            logger.info(f"Model '{model_type}' instantiated successfully.")
            
            # Train the model
            X_test, y_test, report = model.fit(data)
            logger.info(f"Model '{model_type}' trained successfully.")
            
            # Save the trained model
            ensure_dir_exists(os.path.dirname(model_save_path))
            model.save(model_save_path)
            logger.info(f"Model '{model_type}' saved at {model_save_path}.")
            
            # Save data for inference
            X_test_path = test_data_paths['X_test']
            y_test_path = test_data_paths['y_test']
            ensure_dir_exists(os.path.dirname(X_test_path))
            ensure_dir_exists(os.path.dirname(y_test_path))
            
            with open(X_test_path, 'wb') as f:
                pickle.dump(X_test, f)
            y_test.to_pickle(y_test_path)
            logger.info(f"Inference dataset for model '{model_type}' saved successfully.")
            
        except Exception as e:
            logger.error(f"Error training the model '{model_type}': {e}")
            raise
        
        finally:
            # Free memory if the model was created
            if 'model' in locals():
                del model
            if 'X_test' in locals():
                del X_test
            if 'y_test' in locals():
                del y_test
            if 'report' in locals():
                del report
            gc.collect()
            
            logger.debug(f"Memory freed after training the model '{model_type}'.")

    # KNNRecommenderModel
    try:
        logger.info("Starting KNN training.")
        knn_model = KNNRecommenderModel(config_path, logger=logger)
        knn_model.fit(data)
        knn_model_save_path = config['paths']['knn_model']
        ensure_dir_exists(os.path.dirname(knn_model_save_path))
        knn_model.save(knn_model_save_path)
        logger.info(f"KNN model saved at {knn_model_save_path}.")
    except Exception as e:
        logger.error(f"Error during KNN training: {e}")
        raise

    # PCAClusterizationModel
    try:
        logger.info("Starting PCA training.")
        pca_model = PCAClusterizationModel(config_path, logger=logger)
        pca_model.fit(data)
        pca_model_save_path = config['pca_clus']['output_pca_model']
        ensure_dir_exists(os.path.dirname(pca_model_save_path))
        pca_model.save(pca_model_save_path)
        logger.info(f"PCA model saved at {pca_model_save_path}.")
    except Exception as e:
        logger.error(f"Error during PCA training: {e}")
        raise

    # Load PCA data and train DBSCAN
    pca_transformed_data_path = config['pca_clus']['transformed_data_output']
    try:
        logger.info("Loading PCA data for DBSCAN.")
        with open(pca_transformed_data_path, 'rb') as f:
            pca_transformed_data = pickle.load(f)
        
        logger.info("Starting DBSCAN training.")
        dbscan_model = DBSCANClusterizationModel(config_path, logger=logger)
        dbscan_model.run_training()
        dbscan_model_save_path = config['dbscan_clus']['output_model']
        ensure_dir_exists(os.path.dirname(dbscan_model_save_path))
        logger.info(f"DBSCAN model saved at {dbscan_model_save_path}.")
    except Exception as e:
        logger.error(f"Error during DBSCAN training: {e}")
        raise

    logger.info("All models trained and saved successfully.")


def main():
    logger = setup_logger('train_models_logger')
    logger.info("Training started.")
    
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('config_path', type=str, help='Path to the configuration file (train.conf)')
    args = parser.parse_args()
    
    try:
        train_models(args.config_path)
        logger.info("Model training completed.")
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise

if __name__ == "__main__":
>>>>>>> 4c9b743 (Proyecto Git LFS)
    main()