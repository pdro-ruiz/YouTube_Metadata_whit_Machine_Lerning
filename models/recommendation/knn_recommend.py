"""
utf-8
knn_recommend.py

-------------------------------------------------------------------------------------------------- xx
Author: Pedro Ruiz
Creation : Pedro Ruiz 10-01-2024
Version: 2.1
-------------------------------------------------------------------------------------------------- xx

This script allows generating K-Nearest Neighbors (KNN) recommendations for a specific video.

Unlike knn_recommender.py, which encapsulates the recommendation logic in a class, this script (knn_recommend.py) 
provides an interface to interactively request recommendations from an existing KNN model. It allows users to input 
a video ID and receive recommendations based on the pre-trained model.

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
import argparse
import pandas as pd
import configparser
import json
from logger.logger import setup_logger
from models.recommendation.knn_recommender import KNNRecommenderModel


def ensure_dir_exists(path):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='KNN Recommendations for a specific video.')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    log_file = 'logs/knn_recommend.log'
    logger = setup_logger('knn_recommend_logger', log_file)
    logger.info("Starting KNN recommendations.")

    config = configparser.ConfigParser()
    config.read(args.config_path)

    try:
        # Instantiate KNN
        knn_model = KNNRecommenderModel(args.config_path, logger=logger)

        # Load pre-trained KNN
        knn_model.load(config['paths']['knn_model'])

        # Load video_ids
        knn_model.load_video_ids(config['knn_rec']['video_ids_path'])

        # Request video_id from the user
        video_id = input("Enter the video ID for which you want to get recommendations: ")

        # Check if the provided video_id exists within the data
        if video_id not in knn_model.video_ids:
            logger.error(f"Video ID '{video_id}' was not found in the database.")
            raise ValueError(f"Video ID '{video_id}' not found.")

        # Recommendations
        recommended_ids, distances = knn_model.predict(video_id, top_k=knn_model.k)

        # Load processed data
        processed_data_path = config['data']['processed_data']
        df = pd.read_pickle(processed_data_path)
        logger.info(f"Processed data loaded from {processed_data_path}.")

        # Recommended videos
        recommended_videos = df[df['video_id'].isin(recommended_ids)].copy()
        recommended_videos['similarity'] = 1 - distances

        # Display recommendations
        print(f"\nRecommended videos for ID '{video_id}':")
        print(recommended_videos[['video_id', 'channel_title', 'title_words', 'category_name', 'similarity']])

        # Save results
        results_output = config['knn_rec']['results_output']
        ensure_dir_exists(os.path.dirname(results_output))
        recommended_videos[['video_id', 'channel_title', 'title_words', 'category_name', 'similarity']].to_csv(results_output, index=False)
        logger.info(f"Recommendations saved to {results_output}.")

        # Save metrics
        metrics_output = config['knn_rec']['metrics_output']
        ensure_dir_exists(os.path.dirname(metrics_output))
        with open(metrics_output, 'w') as f:
            json.dump(knn_model.metrics, f, indent=4)
        logger.info(f"Metrics saved to {metrics_output}.")

    except Exception as e:
        logger.error(f"Error during KNN recommendations: {e}")
        raise

    logger.info("KNN recommendation process completed successfully.")


if __name__ == "__main__":
    main()
