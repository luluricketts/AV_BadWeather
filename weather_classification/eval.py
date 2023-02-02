import os
import argparse

import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from yaml import safe_load as yload

import models
from data.data import WeatherDataset, eval_transform, collate_fn

def configure_model(model_type, params):

    model = getattr(models, model_type)()
    model = model.to('cuda')
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(params))
    print('matched parameters!')
    return model


def generate_results_report(model, dataloader):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        logging.warning('Could not establish connection to cuda')
    model.eval()

    correct = 0
    all_preds = None
    all_y = None
    srcs = None
    for i,(X,y,src) in enumerate(dataloader): 
        src = np.array(src)

        if i % 10 == 0:
            print(f'Processing batch {i}/{len(dataloader)}')

        X = X.to(device).float()
        pred = model(X)
        pred_labels = torch.argmax(pred, axis=1).detach().cpu()
        correct += torch.sum(pred_labels == y).item()

        if all_preds is None:
            all_preds = pred_labels
            srcs = src
            all_y = y
        else:
            all_preds = torch.cat((all_preds, pred_labels))
            srcs = np.concatenate((srcs, src))
            all_y = torch.cat((all_y, y))

        del X,y,pred
        torch.cuda.empty_cache()

    print('------------------------------------------------------')
    print('Reports analysis')
    print('------------------------------------------------------')
    print(f'Overall accuracy: {correct / len(dataloader.dataset)}')
    print('------------------------------------------------------')
    print('Per Class Accuracies:')
    for i in range(4):
        idxs = torch.where(all_y == i, all_y, -1)
        n = torch.sum(idxs != -1)
        print(f'Class {i}: \t {torch.sum(idxs == all_preds) / n}')
    print('Confusion Matrix')
    confmat = ConfusionMatrix(task="multiclass", num_classes=4)
    print(confmat(all_preds, all_y))
    print('------------------------------------------------------')
    print('Per Source Accuracies:')
    for i in ['MWI', 'Dawn', 'BDD100k']:
        idxs = np.where(srcs == i)
        print(f'Source {i}: \t {torch.sum(all_y[idxs] == all_preds[idxs]) / len(idxs)}')



parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='data/config.yaml')
parser.add_argument('--test', action='store_true')
parser.add_argument('--train', action='store_true')
if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, "r") as file:
        cfg = yload(file)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["device_IDs"]  # specify which GPU(s) to be used

    if not cfg["params"]:
        raise ValueError("Must specify parameters to load in config")

    model = configure_model(cfg["model"], cfg["params"])

    if args.train:
        train_dataset = WeatherDataset(cfg["metadata_train"], transform=eval_transform)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=len(train_dataset), 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=8
        )
        generate_results_report(model, train_dataloader)
    
    if args.test:
        test_dataset = WeatherDataset(cfg["metadata_test"], transform=eval_transform)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=128,
            collate_fn=collate_fn,
            shuffle=False
        )    
        generate_results_report(model, test_dataloader)