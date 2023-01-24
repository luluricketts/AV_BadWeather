import json

import numpy as np
from yaml import safe_load as yload
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

def class_sampler(cfg_file, json_file):

    with open(cfg_file, "r") as file:
        cfg = yload(file)
        counts = np.array(list(cfg["class_counts"].values()))
    with open(json_file, "r") as file:
        train_data = json.load(file)

    weights = 1. / counts
    class_weights = np.array([weights[v["label"]] for k,v in train_data.items()])
    samples_weight = torch.from_numpy(class_weights).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    return sampler


class WeatherDataset(Dataset):

    def __init__(self, metadata_json, transform=None):
        with open(metadata_json) as file:
            self.metadata = json.load(file)
        
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        idx = str(idx)
        img_path = self.metadata[idx]["img_path"]
        img = Image.open(img_path)
        img = img.resize((224, 224), Image.ANTIALIAS)

        label = torch.tensor(self.metadata[idx]["label"])
        
        if self.transform:
            img = self.transform(img)

        return img, label

