"""
Use this space to describe this module
"""
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os
import logging

from .. hydra_pipeline import add_step, optimize_hyperparams



logger = logging.getLogger(__name__)


def split_data(data,test_size=0.2):
    """Split data in test and train

    Parameters
    ----------
    data: pandas DataFrame or dict
        Input data
    test_size : float, optional
        Test size, by default 0.2
    """

    X = data.drop('Target', axis=1)
    y = data.Target

    logger.info("Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =  0.3, random_state = 42 )

    return X_train, X_test, y_train, y_test

def train(X,y, cfg=None):
    """
    Runs all methods to train the model
  
    Parameters
    ----------
    X: numpy array or pandas Dataframe 
        Input data
    y: numpy array or pandas Dataframe 
        Target data

    cfg: dict or omegaconf.dictconfig.DictConfig, default = None
        Configuration values
    """
 

    logger.info("Building pipeline")

    pipeline = Pipeline(
        steps=[
            add_step(cfg, "scaling"),
            add_step(cfg, "model"),
        ]
    )
    
    opt = optimize_hyperparams(cfg, pipeline)

    opt.fit(X, y)
    
    try:
        logger.info(f'Best score-> Mean:{opt.best_score_[0]}, Stddev:{opt.best_score_[1]}')
    except:
        logger.info(f"Best score-> Mean:{opt.best_score_}, Std: {opt.cv_results_['std_test_score'][opt.best_index_]}")
       
    joblib.dump(opt.best_estimator_, filename=f"{os.getcwd()}/pipeline.joblib")
    
    return opt 