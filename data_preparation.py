import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

logger = logging.getLogger(__name__)

def get_data():
    logger.info("Reading dataset")

    X, y = load_breast_cancer(return_X_y=True)
    logger.info(f"Shape: {X.shape}")
    
    return X, y

def split(X, y, test_size):

    logger.info("Splitting data")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    logger.info(f"Training dataset shape: {X_train.shape}")
    logger.info(f"Test dataset shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test

