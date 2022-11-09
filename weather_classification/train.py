import argparse
import logging

import torch
print(torch.__version__)
import torch.nn as nn
from torch.utils.data import DataLoader, default_collate
from torchvision.models import resnet50
from torchvision import transforms

from data import WeatherDataset


logging_levels = {
    1: logging.DEBUG,
    2: logging.INFO,
    3: logging.WARNING,
    4: logging.ERROR,
    5: logging.CRITICAL
}

transform = transforms.Compose([
    transforms.RandomCrop(224,224),
    transforms.RandomAffine(90),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
])


def collate_fn(batch):
    batch = list(filter(lambda x: x[0].shape[0] == 3, batch))
    return default_collate(batch)


class WeatherClass:

    def __init__(self, model, train_dataloader, test_dataloader):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device != 'cuda':
            logging.warning('Could not establish connection to cuda')

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.model = model.to(self.device)
        



    def train(self, eps, lr):
        # TODO pass these in as params
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        optimizer.zero_grad()
        
        self.model.train()
        for ep in range(eps):
            for i,(X,y) in enumerate(self.train_dataloader):

                X = X.to(self.device)
                y = y.to(self.device)

                pred = self.model(X)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()

                logging.info(f'EPOCH {ep} BATCH {i}/{len(self.train_dataloader)}: Loss {loss}')
        

        
parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, default='../../data/MWI-reformat/train_data')
parser.add_argument('--train_labels', type=str, default='../../data/MWI-reformat/train_labels')
parser.add_argument('--test_data', type=str, default='../../data/MWI-reformat/test_data')
parser.add_argument('--test_labels', type=str, default='../../data/MWI-reformat/test_labels')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--logging', type=int, default=2)
if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(level=logging_levels[args.logging])

    # define base model
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 4)

    train_data = WeatherDataset(args.train_data, args.train_labels, transform=transform)
    test_data = WeatherDataset(args.test_data, args.test_labels, transform=transform)
    train_dataloader = DataLoader(
        train_data, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=8
    )
    test_dataloader = DataLoader(
        test_data, 
        batch_size=len(test_data), 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=8)


    Weather = WeatherClass(model, train_dataloader, test_dataloader)

    # train
    epochs = 1
    learning_rate = 0.001
    Weather.train(epochs, learning_rate)
