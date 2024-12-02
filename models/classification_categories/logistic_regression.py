"""
utf-8
logistic_regression.py

-------------------------------------------------------------------------------------------------- xx
Author: Pedro Ruiz
Creation: 10-01-2024
Version: 2.1
-------------------------------------------------------------------------------------------------- xx

This module contains the implementation of the Logistic Regression classification model for video categories. Classification challenge 1.1

The model uses textual features (title, tags, description) processed with TF-IDF,
along with other features to predict the category of a video.

Classes:
- LogisticRegressionModel: encapsulates the logic of the model.

Main methods:
- __init__: Initializes the model with the specified configuration.
- fit: Trains the model with the data provided.
- predict: Makes predictions using the trained model.
- save: Saves the trained model and associated metrics.
- load: Loads a previously trained model.

The model uses a configuration file (train.conf) to specify hyperparameters, data paths, and other relevant options.
"""

import os
import json
import pickle
import configparser
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging


class LogisticRegressionModel:
    def __init__(self, config_path, logger=None):
        self.config_path = config_path
        # Initialize logger
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Avoid adding multiple handlers if they already exist
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        self.logger.propagate = False  # Avoid propagation to parent loggers

        self.logger.info("LogisticRegressionModel initialized.")

        config = configparser.ConfigParser()
        config.read(config_path)

        # Model configuration
        C = config.getfloat('logistic_regression', 'C')
        penalty = config.get('logistic_regression', 'penalty')
        multi_class = config.get('logistic_regression', 'multi_class')
        solver = config.get('logistic_regression', 'solver')
        max_iter = config.getint('logistic_regression', 'max_iter')
        random_state = config.getint('logistic_regression', 'random_state')

        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            multi_class=multi_class,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state
        )

        self.features = [feature.strip() for feature in config.get('logistic_regression', 'features').split(',')]

        # TF-IDF paths
        try:
            self.tfidf_paths = {
                'title_tfidf': config.get('tfidf_paths', 'title_tfidf'),
                'tags_tfidf': config.get('tfidf_paths', 'tags_tfidf'),
                'description_tfidf': config.get('tfidf_paths', 'description_tfidf')
            }
        except configparser.NoOptionError as e:
            self.logger.error(f"Missing TF-IDF path: {e}")
            raise

        self.target = config.get('logistic_regression', 'target')

        # Test data paths
        try:
            self.X_test_path = config['log_reg_clf']['X_test']
            self.y_test_path = config['log_reg_clf']['y_test']
        except KeyError as e:
            self.logger.error(f"Missing test data path in config: {e}")
            raise

    def __preprocess_data(self, df):
        """
        Preprocesses the data by loading the TF-IDF matrices and the other features from the DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame.

        Returns:
            tuple: The preprocessed data.
        """
        self.logger.debug("Preparing the data.")
        tfidf_matrices = [pickle.load(open(self.tfidf_paths[feature], 'rb')) for feature in self.tfidf_paths]
        features = df[self.features]
        X = hstack(tfidf_matrices + [features])
        y = df[self.target]
        return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    def predict(self, X):
        self.logger.debug("Making predictions.")
        return self.model.predict(X)

    def fit(self, df):
        """
        Fits the model.

        Parameters:
            df (pd.DataFrame): The DataFrame.

        Returns:
            tuple: Test data and the classification report.
        """

        self.logger.info("Starting logistic regression training process.")

        X_train, X_test, y_train, y_test = self.__preprocess_data(df)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Create the test data directories
        os.makedirs(os.path.dirname(self.X_test_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.y_test_path), exist_ok=True)

        # Save the test data
        with open(self.X_test_path, 'wb') as f:
            pickle.dump(X_test, f)
        y_test.to_pickle(self.y_test_path)

        self.logger.info("Finalizing the Logistic Regression training process.")

        return X_test, y_test, report

    def save(self, model_path):
        """
        Saves the trained model.

        Parameters:
            model_path (str): Save path.
        """
        self.logger.info(f"Saving logistic regression model to {model_path}.")

        # Create the directory
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save the model
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file)

        self.logger.debug("Logistic regression model saved successfully.")

    def load(self, model_name):
        """
        Loads a previously trained model.

        Parameters:
            model_name (str): The name of the model.
        """
        self.logger.info(f"Starting to load {model_name}.")
        base_name = os.path.splitext(model_name)[0]

        if not base_name.endswith('_weights'):
            base_name = f"{base_name}"

        with open(f'{base_name}.pkl', 'rb') as f:
            self.model = pickle.load(f)

        self.logger.debug("Logistic regression model loaded successfully.")
