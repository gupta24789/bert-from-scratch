import torch
import yaml
import json
import pandas as pd
from pathlib import Path

def load_config(config_path):
    config = yaml.safe_load(open(config_path))
    return config


def load_data():
    df = pd.read_pickle("data/pairs.pkl")
    data_pairs = df.values.tolist()
    return data_pairs[:1000]

def get_model_path(config, epoch):
    model_folder = Path(config['model_folder'], config['checkpoints_folder'])
    model_folder.mkdir(parents=True, exist_ok=True)

    model_filename = f"{config['model_name']}{str(epoch).zfill(2)}.pt"
    model_filepath = model_folder/model_filename
    return str(model_filepath)


def save_model(config, model, optimizer, epoch):

    ## Get model path
    model_path = get_model_path(config, epoch)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
    }

    torch.save(state, model_path)



