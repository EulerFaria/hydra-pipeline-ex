import hydra
from omegaconf import DictConfig
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def hpo(X_train, y_train, cfg):
    pl = Pipeline([
        ('scalar', StandardScaler()),
        ('pca', PCA()),
        ('lr', LogisticRegression())
    ])

    
    print(cfg['gs_params'])
    gs = GridSearchCV(pl, cfg['gs_params'], cv=5, scoring='f1')
    gs.fit(X_train, y_train)

    return gs.best_estimator_
