import hydra
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from omegaconf import DictConfig
from pickle import dump


def build_pipeline(cfg,log):

    log.info("Building pipeline:")
    pipe = []
    
    if cfg['scaling']['type'] == 'std':

        log.info("Adding StandardScaler to Pipeline")

        pipe.append(
            ('sc',StandardScaler())
            ) 
    elif cfg['scaling']['type'] == 'minmax':   
        log.info("Adding MinMaxScaler to Pipeline")   
        pipe.append(
            ('sc',MinMaxScaler())
            ) 

    if cfg['decomp']['type'] == 'pca':
        log.info("Adding PCA to Pipeline")
        pipe.append(
            ('pca',PCA())
            ) 

    if cfg['model']['type'] =='lr':
        log.info("Adding LogisticRegression to Pipeline")
        pipe.append(
            ('lr',LogisticRegression())
        )

    return Pipeline(pipe)

def hpo(pipe,X_train, y_train,cfg,log):

    log.info("Hyperparameters optimization:")
    
    model_name=cfg['model'].pop('type', None)

    _=cfg['decomp'].pop('type', None)

    args = {}
    args.update(cfg['model'])
    args.update(cfg['decomp'])

    log.info(f"Search space for HPO: {args}")

    gs = GridSearchCV(pipe, args, cv=5, scoring='f1')
    gs.fit(X_train, y_train)

    model = gs.best_estimator_
    
    log.info(f"Best estimator: \n {model}")

    dump(model,open(f"{os.getcwd()}/{model_name}.pkl",'wb'))

    return model
