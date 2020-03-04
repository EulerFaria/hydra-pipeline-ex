import hydra
from omegaconf import DictConfig
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler


def build_pipeline(cfg):

    pipe = []
    
    if cfg['scaling']['type'] == 'std':
        pipe.append(
            ('sc',StandardScaler())
            ) 
    elif cfg['scaling']['type'] == 'minmax':      
        pipe.append(
            ('sc',MinMaxScaler())
            ) 

    if cfg['decomp']['type'] == 'pca':
        pipe.append(
            ('pca',PCA())
            ) 

    if cfg['model']['type'] =='lr':
        pipe.append(
            ('lr',LogisticRegression())
        )

    return Pipeline(pipe)

def hpo(pipe,X_train, y_train, cfg):
    
    cfg['model'].pop('type', None)

    cfg['decomp'].pop('type', None)
    print(cfg)

    args = {}
    args.update(cfg['model'])
    args.update(cfg['decomp'])


    gs = GridSearchCV(pipe, args, cv=5, scoring='f1')
    gs.fit(X_train, y_train)

    return gs.best_estimator_
