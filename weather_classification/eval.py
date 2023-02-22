import os
import argparse

import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from yaml import safe_load as yload
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import models
from data.data import WeatherDataset, eval_transform, collate_fn

def configure_model(model_type, params):

    model = getattr(models, model_type)(False, False)
    model = model.to('cuda')
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(params))
    print('matched parameters!')
    return model


def generate_results_report(model, dataloader, dataset_name, directory, file):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        logging.warning('Could not establish connection to cuda')
    model.eval()

    correct = 0
    all_preds = None
    all_y = None
    all_confs = None
    srcs = None
    for i,(X,y,src) in enumerate(tqdm(dataloader)): 
        src = np.array(src)

        X = X.to(device).float()
        pred = model(X)
        preds = torch.max(pred, axis=1)
        pred_labels = preds.indices.detach().cpu()
        conf = preds.values.detach().cpu()
        correct += torch.sum(pred_labels == y).item()

        if all_preds is None:
            all_preds = pred_labels
            srcs = src
            all_y = y
            all_confs = conf
        else:
            all_preds = torch.cat((all_preds, pred_labels))
            srcs = np.concatenate((srcs, src))
            all_y = torch.cat((all_y, y))
            all_confs = torch.cat((all_confs, conf))

        del X,y,pred,preds
        torch.cuda.empty_cache()

    file.write('------------------------------------------------------\n')
    file.write(f'Reports analysis: {dataset_name}\n')
    file.write('------------------------------------------------------\n')
    file.write(f'Overall accuracy: {correct / len(dataloader.dataset)}\n')
    file.write('------------------------------------------------------\n')
    file.write('Per Class Accuracies:\n')
    for i in range(4):
        idxs = torch.where(all_y == i, all_y, -1)
        n = torch.sum(idxs != -1)
        file.write(f'Class {i}: \t {torch.sum(idxs == all_preds) / n}\n')
    file.write('Confusion Matrix\n')
    confmat = ConfusionMatrix(task="multiclass", num_classes=4)
    cmat = confmat(all_preds, all_y)
    for r in cmat:
        for c in r:
            file.write(f'{c}\t')
        file.write('\n')
    
    file.write('------------------------------------------------------\n')

    # confidence/degree analysis
    conf_th = np.linspace(0.001, 0.999, 200)
    n_gt = []
    acc = []
    for c in conf_th:
        idxs = torch.where(all_confs >= c, all_confs, 0)
        n = torch.count_nonzero(idxs).item()
        n_gt.append(n)
        idxs = torch.where(all_confs >= c, all_preds, -1)
        acc.append(torch.sum(idxs == all_y) / n)

    plt.plot(conf_th, n_gt)
    plt.xlabel('Confidence Threshold')
    plt.ylabel('n >= confidence')
    plt.savefig(os.path.join(directory, f'n_per_conf{dataset_name}.jpg'))
    plt.clf()

    plt.plot(conf_th, acc)
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(directory, f'confidence_acc{dataset_name}.jpg'))
    plt.clf()



    # TODO fix  this
    # file.write('Per Source Accuracies:\n')
    # for i in ['MWI', 'Dawn', 'BDD100k']:
    #     idxs = np.where(srcs == i)
    #     file.write(f'Source {i}: \t {torch.sum(all_y[idxs] == all_preds[idxs]) / len(idxs)}\n')



parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True, help='directory UNDER RESULTS/ containing config and params')
if __name__ == "__main__":
    args = parser.parse_args()
    directory = os.path.join('results', args.dir)
    with open(os.path.join(directory, 'config.yaml'), "r") as file:
        cfg = yload(file)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["device_IDs"]  # specify which GPU(s) to be used

    params = os.path.join(directory, f'{args.dir}.pth')
    model = configure_model(cfg["model"], params)

    file = open(os.path.join(directory, 'eval.txt'), 'w+')

    # try:
    train_dataset = WeatherDataset(cfg["metadata_train"], transform=eval_transform)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg["batch_size"], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=8
    )
    generate_results_report(model, train_dataloader, 'TRAIN', directory, file)
    # except:
    #     pass

    # try:
    val_dataset = WeatherDataset(cfg["metadata_val"], transform=eval_transform)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8
    )
    generate_results_report(model, val_dataloader, 'VAL', directory, file)
    # except:
    #     pass
    
    # try:
    test_dataset = WeatherDataset(cfg["metadata_test"], transform=eval_transform)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        collate_fn=collate_fn,
        shuffle=False
    )    
    generate_results_report(model, test_dataloader, 'TEST', directory, file)
    # except:
    #     pass


    file.close()