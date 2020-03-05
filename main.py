import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from data_preparation import get_data, split
from train import hpo, build_pipeline
from evaluate import generate_report


@hydra.main(config_path="conf/config.yaml")
def experiment(cfg: DictConfig)-> None:
    # Reading dataset
    X, y = get_data()

    # Splitting data   
    X_train, X_test, y_train, y_test = split(X, y, 0.2)
    
    #Building pipeline
    pipe = build_pipeline(cfg)

    #Hyperparameter optimization & Dump best estimator
    model = hpo(pipe,X_train, y_train,cfg)

    #Evaluation report
    generate_report(y_train,model.predict(X_train),
                    y_test,model.predict(X_test))

    

if __name__ == "__main__":
    experiment()
    

    
