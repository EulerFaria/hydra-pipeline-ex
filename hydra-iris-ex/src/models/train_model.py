"""
Use this space to describe this module
"""
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os
import logging

from .. hydra_pipeline import add_step, optimize_hyperparams

from sklearn.decomposition import PCA
from sklearn.svm import SVC


logger = logging.getLogger(__name__)

def run(data, cfg=None):
    """
    Runs all methods to train the model
  
    Parameters
    ----------
    data: pandas DataFrame or dict
        Input data

    cfg: dict or omegaconf.dictconfig.DictConfig, default = None
        Configuration values
    """
    X = data.drop('Target', axis=1)
    y = data.Target

    X_train, X_test, y_train, y_test = train_test_split( X, y , test_size =  0.3, random_state = 42 )

    pipeline = Pipeline(
        steps=[
            ('PCA', PCA()),
            add_step(cfg, "model", logger)
        ]
    )
    

    opt = optimize_hyperparams(cfg, pipeline, logger)

    opt.fit(X_train, y_train)
    
    logger.info(f'Best score-> Mean:{opt.best_score_[0]}, Stddev:{opt.best_score_[1]}')

    joblib.dump(opt, filename=f"{os.getcwd()}/pipeline.joblib")
    