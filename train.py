from src.data.make_dataset import data_acq
from src.features.build_features import  data_prep
from src.models.train_model import train, split_data
from src.report import generate_report
from pathlib import Path
import hydra

@hydra.main(config_path="config/config.yaml", strict=False)
def run(cfg=None):
    #Getting dataset 
    raw_data = data_acq(cfg)
    #Building features
    data = data_prep(raw_data, cfg)
    #Split data 
    X_train, X_test, y_train, y_test= split_data(data)
    #Hyperparameter optimization
    opt = train(X_train,y_train, cfg)
    #Evaluation report
    generate_report(opt,X_test,y_test)

if __name__ == "__main__":    
    run()