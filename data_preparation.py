import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

def get_data(log):
    log.info("Reading dataset")

    dataset = load_breast_cancer()

    X = dataset.data
    y = dataset.target

    log.info(f"Shape: {X.shape}")
    
    return X, y

def split(X, y, test_size,log):

    log.info("Splitting data")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    log.info(f"Traning dataset shape: {X_train.shape}")
    log.info(f"Test dataset shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test

