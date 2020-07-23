import pandas as pd
import os
from sklearn.datasets import load_iris

import logging

logger = logging.getLogger(__name__)

def data_acq(cfg=None):
    """
    Runs the data acquision methods

    Parameters
    ----------
    cfg: dict or omegaconf.dictconfig.DictConfig, default = None
        Configuration values

    Returns
    -------
    data: pandas DataFrame or dict
    """
    logger.info("Getting dataset")
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['Target'] = iris.target

    return data
