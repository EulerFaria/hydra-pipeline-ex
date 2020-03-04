import hydra
from omegaconf import DictConfig, OmegaConf
from data_preparation import get_data, split
from train import hpo, build_pipeline
import sklearn


@hydra.main(config_path="conf/config.yaml")
def my_app(cfg: DictConfig)-> None:
    cfg = OmegaConf.to_container(cfg, resolve=True)
    

    X, y = get_data()
    X_train, X_test, y_train, y_test = split(X, y, 0.2)
    print(X_train.shape)
    print(X_test.shape)

    
    pipe = build_pipeline(cfg)
    model = hpo(pipe,X_train, y_train,cfg)
    
    print(model)


if __name__ == "__main__":
    my_app()
    

    
