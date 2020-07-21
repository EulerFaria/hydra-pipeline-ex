from src.data.make_dataset import run as data_acq
from src.features.build_features import run as data_prep
from src.models.train_model import run as train
from pathlib import Path
import hydra
import os

@hydra.main(config_path="config/config.yaml", strict=False)
def run(cfg=None):
    os.environ['path'] =  str(Path(__file__).resolve().parent)
    data = data_acq(cfg)
    merged_data = data_prep(data, cfg)
    train(merged_data, cfg)

if __name__ == "__main__":    
    run()