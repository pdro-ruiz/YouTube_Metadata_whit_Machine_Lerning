<<<<<<< HEAD
import logging
import os

def setup_logger(name, log_file, level=logging.INFO):
    """
    Configures a logger with the specified name and log file, along with a console handler.
    
    Parameters:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        level (int): Logging level.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent multiple handlers from being added to the same logger
    if not logger.handlers:
        # Handler for the log file
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)
        
        # Handler for the console
        ch = logging.StreamHandler()
        ch.setLevel(level)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(console_formatter)
        logger.addHandler(ch)
    
    return logger
=======
import logging
import os

def setup_logger(name, log_file, level=logging.INFO):
    """
    Configures a logger with the specified name and log file, along with a console handler.
    
    Parameters:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        level (int): Logging level.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent multiple handlers from being added to the same logger
    if not logger.handlers:
        # Handler for the log file
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)
        
        # Handler for the console
        ch = logging.StreamHandler()
        ch.setLevel(level)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(console_formatter)
        logger.addHandler(ch)
    
    return logger
>>>>>>> 4c9b743 (Proyecto Git LFS)
