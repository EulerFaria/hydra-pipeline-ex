import hydra
from omegaconf import DictConfig, OmegaConf
from data_preparation import get_data, split
from train import hpo
import sklearn

@hydra.main(config_path="conf/config.yaml")
def my_app(cfg: DictConfig):
    primitive = OmegaConf.to_container(cfg, resolve=True)
    print(primitive['dataset'])
    return primitive


if __name__ == "__main__":
    cfg = my_app()
    print(cfg)
    base_conf = OmegaConf.load('conf/config.yaml')
    print(base_conf.pretty())

    X, y = get_data()
    X_train, X_test, y_train, y_test = split(X, y, 0.2)
    print(X_train.shape)
    print(X_test.shape)

    
    #print(cfg['gs_params'])

    model = hpo(X_train, y_train, cfg)
    
    print(model)
    
