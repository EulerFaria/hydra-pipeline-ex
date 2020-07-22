"""
Tools for creating sklearn Pipelines with Hydra
"""

import importlib
from omegaconf import DictConfig, OmegaConf
import logging

logger = logging.getLogger(__name__)


def _get_class(step_class):
    """
    Returns an instatiated object given a string that indicates the class

    Parameters
    ----------
    step_class: str
        Class to be instantiated

    Returns
    -------
    class_: object
        Instantiated object

    class_name: str
        Name of the class
    """
    class_module, class_name = '.'.join(step_class.split('.')[:-1]), step_class.split('.')[-1]

    module = importlib.import_module(class_module)
    class_ = getattr(module, class_name)

    return class_, class_name

def add_step(cfg, step):    
    """
    Returns an object defined by hydra configuration (``cfg``).

    Parameters
    ----------
    cfg: omegaconf.dictconfig.DictConfig
        Hydra configuration

    step: str
        Type of pipeline step to be added. The options are ``feature_selection``, ``scaling``, ``model``

    Returns
    -------
    class_name: str
        Name of the step instantiated class

    class_: instance of ``class_name``
        Instantiated object of ``class_name``
    """
    pipeline_step = getattr(cfg, step)
    step_class = str(pipeline_step['class'])
    
    params = pipeline_step['params']

    class_, class_name = _get_class(step_class)

    logger.info(f"Adding {class_name} to Pipeline")

    if "hyperparam_opt" not in cfg:
        return (class_name, class_(**params))

    else:
        setattr(pipeline_step, "params", {f'{class_name}__{k}': v for k, v in params.items()})
        return (class_name, class_())

def optimize_hyperparams(cfg, pipeline):
    """
    Runs hyperparameter optmization based on hydra's definitions

    Parameters
    ----------
    cfg: omegaconf.dictconfig.DictConfig
        Hydra configuration

    pipeline: sklearn.pipeline.Pipeline
        Model Pipeline

    Returns
    -------
    class_: instance of optmizer
        Instantiated optmizer object
    """
    pipeline_step = getattr(cfg, 'hyperparam_opt')

    class_, _ = _get_class(str(pipeline_step['class']))

    param_space = dict()
    for key in cfg:
        if key != 'hyperparam_opt':
            param_space = dict(param_space, **OmegaConf.to_container(cfg[key].params, resolve=True))


    logger.info(f"Performing Hyperparameter Optimization for the following parameter space {param_space}")
    return class_(pipeline, param_space, **cfg.hyperparam_opt.params)
