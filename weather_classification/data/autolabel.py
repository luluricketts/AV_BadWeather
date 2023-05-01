import json 

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision import transforms, models
from tqdm.auto import tqdm
from torch.optim import Adam
import matplotlib.pyplot as plt

from data import collate_fn

import sys
sys.path.insert(0, '../')
from feature import get_all_features, saturation


class RelabelDataset(Dataset):

    def __init__(self, data, label='tod', ret_idx=False, n=None):
        
        self.data = data

        self.img_size = (224, 224)
        self.label = label
        self.ret_idx = ret_idx
        self.n = n

    def __len__(self):
        if self.n is None:
            return len(self.data)
        return self.n

    def __getitem__(self, idx):

        idx = str(idx)
        img_path = self.data[idx]["img_path"]
        img = Image.open(img_path)
        img = img.resize(self.img_size, Image.LANCZOS)

        if f'{self.label}_label' in self.data[idx]:
            label = torch.tensor(self.data[idx][f"{self.label}_label"])
        else:
            label = -1
        img = transforms.ToTensor()(img)

        # print(img, label)
        if self.ret_idx:
            return img, label, idx
        return img, label


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.head = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Softmax()
        )

    def forward(self, x):
        x, hx = get_all_features(x, hist=False)
        # x = saturation(x.cpu(), hist=False).cuda()
        x = self.model(x)
        x = self.head(x)
        return x


# load jsons
with open('train.json', 'r') as file:
    train_json_tmp = json.load(file)
with open('val.json', 'r') as file:
    val_json = json.load(file)
with open('test.json', 'r') as file:
    test_json_tmp = json.load(file)

lname = 'rc'

# merge train / val
i = 0
train_json = {}
for k,v in train_json_tmp.items():
    if f'{lname}_label' in v.keys() and v[f'{lname}_label'] is not None and v[f'{lname}_label'] != 3:
        train_json[str(i)] = v
        i += 1

print('i should be 0:', i)

for _,v in val_json.items():
    if v[f'{lname}_label'] is not None and v[f'{lname}_label'] != 3:
        train_json[str(i)] = v
        i += 1

i_test = 0
test_json = {}
for k,v in test_json_tmp.items():
    if v[f'{lname}_label'] is not None and v[f'{lname}_label'] != 3:
        if i < 9173:
            train_json[str(i)] = v
            i += 1
        else:
            test_json[str(i_test)] = v
            i_test += 1


print(len(train_json), len(test_json))


# sampler
counts = np.array([0, 0, 0])
for k,v in train_json.items():
    counts[int(v[f'{lname}_label'])] += 1

weights = 1. / counts
class_weights = np.array([weights[v[f"{lname}_label"]] for k,v in train_json.items()])
samples_weight = torch.from_numpy(class_weights).double()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))


# create dataloaders
train_dataset = RelabelDataset(train_json, label=lname)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=128, 
    sampler=sampler,
    collate_fn=collate_fn,
    num_workers=8)
test_dataset = RelabelDataset(test_json)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=128,
    collate_fn=collate_fn,
    num_workers=8
)

def train():

    # define model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model = Model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)
    epochs = 10

    model.train()
    for ep in range(epochs):
        print(f'epoch {ep}')

        train_loss = 0
        train_acc = 0
        for X,y in tqdm(train_dataloader):

            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            train_loss += loss.detach().item()
            train_acc += torch.sum(torch.argmax(pred, axis=1) == y).detach().item()
            
            loss.backward()
            optimizer.step()

            del X,y,pred,loss
            torch.cuda.empty_cache()

        print(f'Train acc {train_acc / len(train_dataloader.dataset)}')
        print(f'Train loss {train_loss}')


        model.eval()
        test_loss = 0
        test_acc = 0
        all_confs = None
        all_preds = None
        all_y = None
        for X,y in tqdm(test_dataloader): 
            X = X.to(device).float()
            y = y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.detach().item()
            test_acc += torch.sum(torch.argmax(pred, axis=1) == y).detach().item()
            
            y = y.detach().cpu()
            preds = torch.max(pred, axis=1)
            pred_labels = preds.indices.detach().cpu()
            conf = preds.values.detach().cpu()

            if all_confs is None:
                all_preds = pred_labels
                all_y = y
                all_confs = conf
            else:
                all_preds = torch.cat((all_preds, pred_labels))
                all_y = torch.cat((all_y, y))
                all_confs = torch.cat((all_confs, conf))


            del X,y,loss,preds,pred
            torch.cuda.empty_cache()

        print(f'Test acc {test_acc / len(test_dataloader.dataset)}')
        print(f'Test loss {test_loss}')

    # confidence/degree analysis
    conf_th = np.linspace(0.001, 0.999, 200)
    acc = []
    for c in conf_th:
        idxs = torch.where(all_confs >= c, all_confs, 0)
        n = torch.count_nonzero(idxs).item()
        idxs = torch.where(all_confs >= c, all_preds, -1)
        acc.append(torch.sum(idxs == all_y) / n)

    plt.plot(conf_th, acc)
    plt.savefig('tod_conf_th.jpg')

    path = 'autolabel_tod.pth'
    torch.save(model.state_dict(), path)


def relabel():

    # get data
    with open('train.json', 'r') as file:
        train_json_tmp = json.load(file)

    json_file = {}
    label_map = {}
    i = 0
    for k,v in train_json_tmp.items():
        if f'{lname}_label' in v.keys():
            continue
        json_file[str(i)] = v
        label_map[i] = k
        i += 1

    dataset = RelabelDataset(json_file, ret_idx=True, n=i)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=128,
        collate_fn=collate_fn,
        num_workers=8
    )

    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model = Model().to(device)
    model.load_state_dict(torch.load('autolabel_tod.pth'))
    thr = np.linspace(0.001, 0.999, 200)[-2]
    # print(train_json_tmp)

    model.eval()
    n_labeled = 0
    for X,_,ret_idx in tqdm(dataloader):
        X = X.to(device)

        pred = model(X)
        preds = torch.max(pred, axis=1)
        pred_labels = preds.indices.detach().cpu()
        conf = preds.values.detach().cpu()
        
        for idx in range(conf.shape[0]):
            if conf[idx] >= thr:
                train_json_tmp[label_map[int(ret_idx[idx])]][f'{lname}_label'] = pred_labels[idx].item()
                n_labeled += 1

        del X,ret_idx,pred
        torch.cuda.empty_cache()

    print(f'Relabeled: {n_labeled}')

    with open('train_relabeled.json', 'w') as file:
        json.dump(train_json_tmp, file)






if __name__ == "__main__":

    train()

    # relabel()