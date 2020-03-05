import hydra
import logging
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from omegaconf import DictConfig, OmegaConf
from pickle import dump

logger = logging.getLogger(__name__)

def build_pipeline(cfg):

    logger.info("Building pipeline:")
    pipe = []
    
    if cfg.scaling.type == 'std':

        logger.info("Adding StandardScaler to Pipeline")

        pipe.append(
            ('sc',StandardScaler())
            ) 
    elif cfg.scaling.type == 'minmax':   
        logger.info("Adding MinMaxScaler to Pipeline")   
        pipe.append(
            ('sc',MinMaxScaler())
            ) 

    if cfg.decomp.type == 'pca':
        logger.info("Adding PCA to Pipeline")
        pipe.append(
            ('pca',PCA())
            ) 

    if cfg.model.type =='lr':
        logger.info("Adding LogisticRegression to Pipeline")
        pipe.append(
            ('lr',LogisticRegression())
        )

    return Pipeline(pipe)

def hpo(pipe,X_train, y_train,cfg):

    logger.info("Hyperparameters optimization:")
    
    model_name=cfg.model.pop('type', None)

    _=cfg.decomp.pop('type', None)

    model = OmegaConf.to_container(cfg.model)
    decomp = OmegaConf.to_container(cfg.decomp)
    args = {**model, **decomp}

    logger.info(f"Search space for HPO: {args}")

    gs = GridSearchCV(pipe, args, cv=5, scoring='f1')
    gs.fit(X_train, y_train)

    model = gs.best_estimator_
    
    logger.info(f"Best estimator: \n {model}")

    dump(model,open(f"{os.getcwd()}/{model_name}.pkl",'wb'))

    return model
