"""
'utf-8'
pca_clusterization.py

-------------------------------------------------------------------------------------------------- xx
Author: Pedro Ruiz
Creation: 10-01-2024
Version: 2.1
-------------------------------------------------------------------------------------------------- xx

This module contains the implementation of the PCA (Principal Component Analysis) for dimensionality reduction in clustering tasks.

The model uses specified features from the dataset to perform PCA and save the transformed data and the trained model.

Classes:
- PCAClusterizationModel: encapsulates the logic of the PCA model.

Main methods:
- __init__: Initializes the model with the specified configuration.
- fit: Trains the PCA model with the data provided and saves the transformed data.
- save: Saves the trained PCA model.
- run: Executes the training process, saves the model, and the transformed data.

The model uses a configuration file (train.conf) to specify hyperparameters, data paths, and other relevant options.
"""


import pickle
import os
import configparser
from sklearn.decomposition import PCA
import pandas as pd
import logging

class PCAClusterizationModel:
    def __init__(self, config_path, logger=None):
        """
        Initializes PCA for clustering.

        Parameters:
            config_path (str): Path to the configuration file.
            logger (logging.Logger, optional): Logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Load configuration
        config = configparser.ConfigParser()
        config.read(config_path)

        # Configuration
        self.n_components = config.getint('pca_clusterization', 'n_components')
        self.random_state = config.getint('pca_clusterization', 'random_state')

        self.model = PCA(n_components=self.n_components, random_state=self.random_state)

        # Features
        self.features = [feature.strip() for feature in config.get('pca_clusterization', 'features').split(',')]

        # Paths
        self.data_path = config['data']['processed_data']
        self.transformed_data_output = config['pca_clus']['transformed_data_output']
        self.output_pca_model = config['pca_clus']['output_pca_model']

        self.logger.info("PCAClusterizationModel inicializado con éxito.")

    def fit(self, df):
        """
        Fits PCA with the provided data and saves the transformed data.

        Parameters:
            df (pd.DataFrame): DataFrame
        """
        self.logger.info("Training PCA model for dimensionality reduction.")
        X = df[self.features]
        X_pca = self.model.fit_transform(X)
        self.logger.info("PCA trained successfully.")

        # Guardar los datos transformados
        os.makedirs(os.path.dirname(self.transformed_data_output), exist_ok=True)
        with open(self.transformed_data_output, 'wb') as f:
            pickle.dump(X_pca, f)
        self.logger.info(f"Transformed data saved in {self.transformed_data_output}.")

    def save(self, model_path):
        """
        Saves trained PCA.

        Parameters:
            model_path (str): Save path.
        """
        self.logger.info(f"Saving PCA model to {model_path}.")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        self.logger.debug("PCA saved successfully.")

    def run(self):
        """
        Executes the PCA process, saves the model and the transformed data.
        """
        # Cargar los datos
        self.logger.info(f"Loading data from {self.data_path}.")
        data = pd.read_pickle(self.data_path)

        # Entrenar el modelo y transformar los datos
        self.fit(data)

        # Guardar el modelo entrenado
        self.save()


def setup_logger(name, log_file, level=logging.INFO):
    """
    Configures a logger with the given name and log file.

    Parameters:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        level (int): Logging level.

    Returns:
        logging.Logger: Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Crear directorio para logs si no existe
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def main():
    parser = argparse.ArgumentParser(description='PCA Clusterization Model')
    parser.add_argument('config_path', type=str, help='Path to the configuration file (train.conf)')
    args = parser.parse_args()

    # Configurar el logger
    log_file = 'logs/pca_clusterization.log'
    logger = setup_logger('pca_clusterization_logger', log_file)
    logger.info("Starting PCA.")

    try:
        # Instantiate the model and execute the training
        pca_model = PCAClusterizationModel(config_path=args.config_path, logger=logger)
        pca_model.run()

    except Exception as e:
        logger.error(f"Error: {e}")
        raise

    logger.info("PCA process completed.")

if __name__ == "__main__":
    main()