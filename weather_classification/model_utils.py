import json

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

def class_sampler(json_file, n_classes=5):

    with open(json_file) as file:
        metadata = json.load(file)

    classes = np.zeros(n_classes)
    labels = []
    for k,v in metadata.items():
        classes[int(v['label'])] += 1
        labels.append(int(v['label']))

    weights = 1. / classes
    class_weights = np.array([weights[i] for i in labels])
    samples_weight = torch.from_numpy(class_weights).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    return sampler


