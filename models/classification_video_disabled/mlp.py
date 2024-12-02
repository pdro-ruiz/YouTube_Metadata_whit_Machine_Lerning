"""
'utf-8'
mlp.py

-------------------------------------------------------------------------------------------------- xx
Author: Pedro Ruiz
Creation: 10-01-2024
Version: 2.1
-------------------------------------------------------------------------------------------------- xx

This module contains the implementation of the MLP (Multi-Layer Perceptron) classification model for video categories. Classification challenge 1.2

The model uses textual features (title, tags, description) processed with TF-IDF,
along with other features to predict the category of a video.

Classes:
- MLPClassifierModel: encapsulates the logic of the model.

Main methods:
- __init__: Initializes the model with the specified configuration.
- fit: Trains the model with the data provided.
- predict: Makes predictions using the trained model.
- save: Saves the trained model and associated metrics.
- load: Loads a previously trained model.

The model uses a configuration file (train.conf) to specify hyperparameters, data paths, and other relevant options.
"""

import os
import pickle
import configparser
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import logging

class MLPClassifierModel:
    def __init__(self, config_path, config_section='mlp', logger=None):
        """
        Initializes the MLP
        
        Parameters:
            config_path (str): Path to the configuration file.
            config_section (str): Section in the configuration file for the MLP model.
            logger (logging.Logger): Logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)
        config = configparser.ConfigParser()
        config.read(config_path)
        
        if self.logger:
            self.logger.info("MLPClassifierModel initialized.")

        # Hiperparámetros de la configuración
        self.hidden_layers = [int(x.strip()) for x in config.get(config_section, 'hidden_layers').split(',')]
        self.dropout_rate = config.getfloat(config_section, 'dropout_rate')
        self.activation = config.get(config_section, 'activation')
        self.optimizer = config.get(config_section, 'optimizer')
        self.loss = config.get(config_section, 'loss')
        self.metrics = [metric.strip() for metric in config.get(config_section, 'metrics').split(',')]
        self.early_stopping_monitor = config.get(config_section, 'early_stopping_monitor')
        self.early_stopping_patience = config.getint(config_section, 'early_stopping_patience')
        self.epochs = config.getint(config_section, 'epochs')
        self.batch_size = config.getint(config_section, 'batch_size')
        self.random_state = config.getint(config_section, 'random_state')
        self.features = [feature.strip() for feature in config.get(config_section, 'features').split(',')]
        self.target = config.get(config_section, 'target')

        # Rutas TF-IDF
        self.tfidf_paths = {
            'title_tfidf': config.get('tfidf_paths', 'title_tfidf'),
            'tags_tfidf': config.get('tfidf_paths', 'tags_tfidf'),
            'description_tfidf': config.get('tfidf_paths', 'description_tfidf')
        }

        # Rutas X_test y y_test
        try:
            self.X_test_path = config['mlp_clf']['X_test']
            self.y_test_path = config['mlp_clf']['y_test']
        except KeyError as e:
            self.logger.error(f"Missing test data path in config: {e}")
            raise

        # Inicializar MLP
        self.model = Sequential()

    def __preprocess_data(self, df):
        """
        Preprocesses the data loaded from the DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the data.

        Returns:
            tuple: (X_train, X_test, y_train, y_test).
        """
        self.logger.debug("Preprocessing data for MLP")
        # Cargar matrices TF-IDF
        tfidf_matrices = [pickle.load(open(self.tfidf_paths[feature], 'rb')) for feature in self.tfidf_paths]
        features = df[self.features]
        X = hstack(tfidf_matrices + [features])
        y = df[self.target]
        return train_test_split(X, y, test_size=0.2, stratify=y, random_state=self.random_state)

    def __build_model(self, input_dim):
        """
        Builds the MLP model based on the hyperparameters.

        Parameters:
            input_dim (int): no. of input features.
        """
        self.model.add(Dense(self.hidden_layers[0], input_dim=input_dim, activation=self.activation))
        self.model.add(Dropout(self.dropout_rate))
        
        for units in self.hidden_layers[1:]:
            self.model.add(Dense(units, activation=self.activation))
            self.model.add(Dropout(self.dropout_rate))
        
        # Clasificación binaria
        self.model.add(Dense(1, activation='sigmoid'))
        
        # Compilar el modelo
        optimizer_instance = Adam() if self.optimizer == 'adam' else None
        self.model.compile(optimizer=optimizer_instance, loss=self.loss, metrics=self.metrics)

    def fit(self, df):
        """
        Trains the model.

        Parameters:
            df (pd.DataFrame)

        Returns:
            tuple: test data and classification report.
        """
        self.logger.info("Training MLP.")
        
        X_train, X_test, y_train, y_test = self.__preprocess_data(df)

        # Convertir matrices dispersas a densas
        X_train_dense = X_train.toarray()
        X_test_dense = X_test.toarray()

        # Construir el modelo
        input_dim = X_train_dense.shape[1]
        self.__build_model(input_dim)

        # Definir early stopping
        early_stopping = EarlyStopping(monitor=self.early_stopping_monitor, patience=self.early_stopping_patience, restore_best_weights=True)

        # Entrenar el modelo
        self.model.fit(
            X_train_dense, 
            y_train, 
            epochs=self.epochs, 
            batch_size=self.batch_size, 
            validation_data=(X_test_dense, y_test),
            callbacks=[early_stopping],
            verbose=1
        )

        # Hacer predicciones
        y_pred_prob = self.model.predict(X_test_dense)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        # Generar reporte de clasificación
        report = classification_report(y_test, y_pred, output_dict=True)

        # Guardar los datos de prueba
        os.makedirs(os.path.dirname(self.X_test_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.y_test_path), exist_ok=True)

        with open(self.X_test_path, 'wb') as f:
            pickle.dump(X_test, f)
        y_test.to_pickle(self.y_test_path)

        self.logger.info("MLP training completed.")
        
        return X_test, y_test, report

    def predict(self, X):
        """
        Makes predictions using the trained MLP model.

        Parameters:
            X (array-like): Input data.

        Returns:
            array: Predictions.
        """
        self.logger.debug("Predicting with MLP")
        # Convertir matrices dispersas a densas si es necesario
        if not isinstance(X, pd.DataFrame):
            X = X.toarray()
        
        y_pred_prob = self.model.predict(X)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        return y_pred

    def save(self, model_name):
        """
        Saves the model.
        
        Parameters:
            model_name (str): Path to save.
        """
        self.logger.info(f"Saving MLP model in {model_name}.")
        
        model_dir = os.path.dirname(model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.model.save(model_name)
        self.logger.debug("MLP model saved successfully.")

    def load(self, model_name):
        """
        Loads a trained model.

        Parameters:
            model_name (str): Path to the model to load.
        """
        self.logger.info(f"Loading MLP model from {model_name}.")
        self.model = load_model(model_name)
        self.logger.debug("MLP model loaded successfully.")
