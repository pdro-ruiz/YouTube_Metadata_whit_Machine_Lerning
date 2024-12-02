<<<<<<< HEAD
# main.py
"""
utf-8
main.py

-------------------------------------------------------------------------------------------------- xx
Author: Pedro Ruiz
Creation: 10-01-2024
Version: 2.1
-------------------------------------------------------------------------------------------------- xx

This module orchestrates the machine learning pipeline, including data preprocessing, model training, and inference.

It parses command-line arguments, sets up logging, and calls the main functions from the respective scripts.

Functions:
- setup_main_logger
- main

Usage:
This script can be run from the command line as follows:
-->    python main.py --config_path train.conf --input_dir data/raw_data --output_dir data/industrialized
"""

import logging
import sys
import argparse
from preprocess_data import main as preprocess_main
from train_models import main as train_main
from inference_models import main as inference_main


def setup_main_logger():
    logger = logging.getLogger('main_logger')
    logger.setLevel(logging.INFO)
    
    # Handle duplicate handlers
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.propagate = False        
    return logger


def main():
    logger = setup_main_logger()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Pipeline Orchestrator')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to raw data')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to processed data')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the train.conf file')
    args = parser.parse_args()

    logger.info("Task 1. Preprocessing...")
    # Call function for preprocess_main with its arguments
    sys.argv = ['preprocess_data.py', '--input_dir', args.input_dir, '--output_dir', args.output_dir]
    preprocess_main()

    logger.info("Task 2. Training...")
    # Call function for train_main with its arguments
    sys.argv = ['train_models.py', args.config_path]
    train_main()

    logger.info("Task 3. Inference...")
    # Call function for inference_main with its arguments
    sys.argv = ['inference_models.py', args.config_path]
    inference_main()

    logger.info("Pipeline completed.")

if __name__ == '__main__':
    main()

=======
# main.py
"""
utf-8
main.py

-------------------------------------------------------------------------------------------------- xx
Author: Pedro Ruiz
Creation: 10-01-2024
Version: 2.1
-------------------------------------------------------------------------------------------------- xx

This module orchestrates the machine learning pipeline, including data preprocessing, model training, and inference.

It parses command-line arguments, sets up logging, and calls the main functions from the respective scripts.

Functions:
- setup_main_logger
- main

Usage:
This script can be run from the command line as follows:
-->    python main.py --config_path train.conf --input_dir data/raw_data --output_dir data/industrialized
"""

import logging
import sys
import argparse
from preprocess_data import main as preprocess_main
from train_models import main as train_main
from inference_models import main as inference_main


def setup_main_logger():
    logger = logging.getLogger('main_logger')
    logger.setLevel(logging.INFO)
    
    # Handle duplicate handlers
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.propagate = False        
    return logger


def main():
    logger = setup_main_logger()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Pipeline Orchestrator')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to raw data')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to processed data')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the train.conf file')
    args = parser.parse_args()

    logger.info("Task 1. Preprocessing...")
    # Call function for preprocess_main with its arguments
    sys.argv = ['preprocess_data.py', '--input_dir', args.input_dir, '--output_dir', args.output_dir]
    preprocess_main()

    logger.info("Task 2. Training...")
    # Call function for train_main with its arguments
    sys.argv = ['train_models.py', args.config_path]
    train_main()

    logger.info("Task 3. Inference...")
    # Call function for inference_main with its arguments
    sys.argv = ['inference_models.py', args.config_path]
    inference_main()

    logger.info("Pipeline completed.")

if __name__ == '__main__':
    main()

>>>>>>> 4c9b743 (Proyecto Git LFS)
