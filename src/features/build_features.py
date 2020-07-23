import logging

logger = logging.getLogger(__name__)

def data_prep(data, cfg=None):
    """
    Runs the data preparation methods

    Parameters
    ----------
    data: pandas DataFrame or dict
        Input data

    cfg: dict or omegaconf.dictconfig.DictConfig, default = None
        Configuration values

    Returns
    -------
    processed_data: pandas DataFrame or dict
    """
    processed_data = data
    return processed_data
    
