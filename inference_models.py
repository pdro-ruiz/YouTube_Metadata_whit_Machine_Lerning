"""
utf-8
inference_models.py

-------------------------------------------------------------------------------------------------- xx
Author: Pedro Ruiz
Creation: 10-01-2024
Version: 2.1
-------------------------------------------------------------------------------------------------- xx

This module performs inference using the trained machine learning models specified in the configuration file.

It loads the trained models and test datasets, performs predictions, calculates metrics, and saves the results.

Functions:
- setup_logger
- ensure_dir_exists
- inference_models
- main

Usage:
This script can be run from the command line as follows:
-->    python inference_models.py train.conf
"""

import os
import configparser
import argparse
import pandas as pd
import pickle
import json
import logging
import numpy as np
from models.classification_categories.logistic_regression import LogisticRegressionModel
from models.classification_video_disabled.mlp import MLPClassifierModel
from models.regression_like_ratio.lightgbm_regressor import LightGBMRegressorModel as LightGBMRegressorLikeDislikeRatioModel
from models.regression_number_of_likes.lightgbm_regressor import LightGBMRegressorModel as LightGBMRegressorNumberOfLikesModel
from models.recommendation.knn_recommender import KNNRecommenderModel


def ensure_dir_exists(path):
    """Creates the directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def setup_logger(name, level=logging.INFO):
    """
    Configures and returns a logger with the specified name and logging level.
    The logger will send messages to both the console and a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Checks if handlers are added and avoids duplicate records
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = 'logs/inference_models.log'
        ensure_dir_exists(os.path.dirname(log_file))
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = False
    return logger


def inference_models(config_path):
    """
    Performs inference for the models.

    Parameters:
        config_path (str): Path to the configuration file (train.conf).
    """
    logger = logging.getLogger('inference_models_logger')
    config = configparser.ConfigParser()
    config.read(config_path)

    # List of models
    models_to_infer = [
        {
            'model_type': 'logistic_regression',
            'class': LogisticRegressionModel,
            'model_save_path': config['paths']['logistic_regression_model'],
            'test_data_paths': {
                'X_test': config['log_reg_clf']['X_test'],
                'y_test': config['log_reg_clf']['y_test']
            },
            'metrics_output': config['log_reg_clf']['metrics_output'],
            'results_output': config['log_reg_clf']['results_output']
        },
        {
            'model_type': 'mlp',
            'class': MLPClassifierModel,
            'model_save_path': config['paths']['mlp_model'],
            'test_data_paths': {
                'X_test': config['mlp_clf']['X_test'],
                'y_test': config['mlp_clf']['y_test']
            },
            'metrics_output': config['mlp_clf']['metrics_output'],
            'results_output': config['mlp_clf']['results_output']
        },
        {
            'model_type': 'lightgbm_regressor_like_dislike_ratio',
            'class': LightGBMRegressorLikeDislikeRatioModel,
            'model_save_path': config['paths']['lightgbm_regression_like_dislike_ratio_model'],
            'test_data_paths': {
                'X_test': config['lightgbm_reg_two_eval']['X_test'],
                'y_test': config['lightgbm_reg_two_eval']['y_test']
            },
            'metrics_output': config['lightgbm_reg_two_eval']['metrics_output'],
            'results_output': config['lightgbm_reg_two_eval']['results_output']
        },
        {
            'model_type': 'lightgbm_regressor_number_of_likes',
            'class': LightGBMRegressorNumberOfLikesModel,
            'model_save_path': config['paths']['lightgbm_regression_model'],
            'test_data_paths': {
                'X_test': config['lightgbm_reg_one_eval']['X_test'],
                'y_test': config['lightgbm_reg_one_eval']['y_test']
            },
            'metrics_output': config['lightgbm_reg_one_eval']['metrics_output'],
            'results_output': config['lightgbm_reg_one_eval']['results_output']
        },
        {
            'model_type': 'knn_recommender',
            'class': KNNRecommenderModel,
            'model_save_path': config['paths']['knn_model'],
            'video_ids_path': config['knn_rec']['video_ids_path'],
            'metrics_output': config['knn_rec']['metrics_output'],
            'results_output': config['knn_rec']['results_output'],
            'processed_data_path': config['data']['processed_data']
        },
    ]

    for model_info in models_to_infer:
        model_type = model_info['model_type']
        model_class = model_info['class']
        model_save_path = model_info['model_save_path']
        metrics_output = model_info['metrics_output']
        results_output = model_info['results_output']
        
        try:
            logger.info(f"Starting inference for model '{model_type}'.")

            # Instantiate model
            model = model_class(config_path)

            # Load the model
            model.load(model_save_path)
            logger.info(f"Model '{model_type}' loaded from {model_save_path}.")

            # If it is KNN
            if model_type == 'knn_recommender':
                # Load video_ids and data
                model.load_video_ids(model_info['video_ids_path'])

                # Load the data
                processed_data_path = model_info['processed_data_path']
                logger.info(f"Loading processed data from {processed_data_path}.")
                df = pd.read_pickle(processed_data_path)

                # Load features and TF-IDF
                logger.info("Loading features.")
                title_tfidf = pickle.load(open(config['tfidf_paths']['title_tfidf'], 'rb'))
                tags_tfidf = pickle.load(open(config['tfidf_paths']['tags_tfidf'], 'rb'))
                description_tfidf = pickle.load(open(config['tfidf_paths']['description_tfidf'], 'rb'))

                # Concatenate the features
                features = df[config['knn_recommendation']['num_features'].split(",")].values
                X_combined = np.hstack([title_tfidf.toarray(), tags_tfidf.toarray(), description_tfidf.toarray(), features])

                # Prompt user for video index
                video_index = int(input("Enter the video index number for which you want to get recommendations: "))

                # Validate the index
                if video_index < 0 or video_index >= len(df):
                    logger.error(f"Provided index ({video_index}) is out of range.")
                    raise ValueError(f"Incorrect index. It must be between 0 and {len(df)-1}")

                # Features of the input video
                video_features = X_combined[video_index].reshape(1, -1)

                # Get recommendations
                logger.info(f"Getting recommendations for video with index {video_index}.")
                distances, indices_knn = model.model.kneighbors(video_features, n_neighbors=model.k + 1)

                # Filter to exclude input video from results
                mask = indices_knn[0] != video_index
                neighbor_indices = indices_knn[0][mask][:model.k]
                neighbor_distances = distances[0][mask][:model.k]

                # Recommended video details
                recommended_videos = df.iloc[neighbor_indices].copy()
                recommended_videos['similarity'] = 1 - neighbor_distances

                # Save results
                ensure_dir_exists(os.path.dirname(results_output))
                recommended_videos[['video_id', 'channel_title', 'title_words', 'category_name', 'similarity']].to_csv(results_output, index=False)
                logger.info(f"Recommendations saved to {results_output}.")

                # Display recommendations
                print(f"\nRecommended videos for video with index '{video_index}':")
                print(recommended_videos[['video_id', 'channel_title', 'title_words', 'category_name', 'similarity']])

            # If it is not KNN
            else:
                # Load data
                X_test_path = model_info['test_data_paths']['X_test']
                y_test_path = model_info['test_data_paths']['y_test']

                with open(X_test_path, 'rb') as f:
                    X_test = pickle.load(f)
                y_test = pd.read_pickle(y_test_path)

                # Predictions
                logger.info(f"Making predictions for '{model_type}'.")
                predictions = model.predict(X_test)

                # Calculate metrics and save results
                logger.info(f"Calculating metrics for '{model_type}'.")
                if 'regressor' in model_type:  # If it is a regression model
                    from sklearn.metrics import mean_squared_error, r2_score
                    mse = mean_squared_error(y_test, predictions)
                    r2 = r2_score(y_test, predictions)
                    metrics = {'mean_squared_error': mse, 'r2_score': r2}                       
                else:
                    from sklearn.metrics import accuracy_score, classification_report
                    accuracy = accuracy_score(y_test, predictions)
                    report = classification_report(y_test, predictions, output_dict=True)
                    metrics = {'accuracy': accuracy, 'classification_report': report}

                # Save metrics and results
                ensure_dir_exists(os.path.dirname(metrics_output))
                with open(metrics_output, 'w') as f:
                    json.dump(metrics, f, indent=4)
                logger.info(f"Metrics saved to {metrics_output}.")

                ensure_dir_exists(os.path.dirname(results_output))
                results_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
                results_df.to_csv(results_output, index=False)
                logger.info(f"Results saved to {results_output}.")

        except Exception as e:
            logger.error(f"Error during inference for model '{model_type}': {e}")
            raise


def main():
    # Set up the logger
    logger = setup_logger('inference_models_logger')
    logger.info("Model inference started.")

    parser = argparse.ArgumentParser(description='Model Inference')
    parser.add_argument('config_path', type=str, help='Path to the configuration file (train.conf)')
    args = parser.parse_args()

    try:
        inference_models(args.config_path)
        logger.info("Model inference completed successfully.")
    except Exception as e:
        logger.error(f"Model inference failed: {e}")
        raise

if __name__ == "__main__":
    main()