import json

import numpy as np
from yaml import safe_load as yload
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, default_collate
from torchvision import transforms


transform = transforms.Compose([
    # transforms.RandomAffine(90),
    # transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def class_sampler(class_counts, json_file):

    counts = np.array(list(class_counts.values()))
    with open(json_file, "r") as file:
        train_data = json.load(file)

    weights = 1. / counts
    class_weights = np.array([weights[v["label"]] for k,v in train_data.items()])
    samples_weight = torch.from_numpy(class_weights).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    return sampler


def collate_fn(batch):
    batch = list(filter(lambda x: x[0].shape[0] == 3, batch))
    return default_collate(batch)


def collate_noMWI(batch):
    batch = list(filter(lambda x:x[0].shape[0] == 3 and x[3] != 'MWI', batch))
    return default_collate(batch)


class WeatherDataset(Dataset):

    def __init__(self, metadata_json, transform=None, img_size=(224, 224)):
        with open(metadata_json) as file:
            self.metadata = json.load(file)
        
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        idx = str(idx)
        img_path = self.metadata[idx]["img_path"]
        img = Image.open(img_path)
        img = img.resize(self.img_size, Image.ANTIALIAS)

        label = torch.tensor(self.metadata[idx]["label"])
        
        if self.transform:
            try:
                img = self.transform(img)
            except:
                img = transforms.ToTensor()(img) # will get collated out

        return img, label, self.metadata[idx]["source"]


class WeatherDatasetToResults(Dataset):

    def __init__(self, metadata_json, transform=None, img_size=(224, 224)):
        with open(metadata_json) as file:
            self.metadata = json.load(file)
        
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        idx = str(idx)
        img_path = self.metadata[idx]["img_path"]
        img = Image.open(img_path)
        img = img.resize(self.img_size, Image.ANTIALIAS)

        label = torch.tensor(self.metadata[idx]["label"])
        
        if self.transform:
            try:
                img = self.transform(img)
            except:
                img = transforms.ToTensor()(img) # will get collated out

        return img, label, self.metadata[idx]['img_path'], self.metadata[idx]['source']
