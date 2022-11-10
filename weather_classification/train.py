import argparse
import logging
import os 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, default_collate
from torch.optim import lr_scheduler, Adam
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from data import WeatherDataset


logging_levels = {
    1: logging.DEBUG,
    2: logging.INFO,
    3: logging.WARNING,
    4: logging.ERROR,
    5: logging.CRITICAL
}

transform = transforms.Compose([
    # transforms.RandomCrop(224,224),
    transforms.Resize((224,224)),
    # transforms.RandomAffine(90),
    # transforms.RandomRotation(90),
    transforms.ToTensor(),
])


def collate_fn(batch):
    batch = list(filter(lambda x: x[0].shape[0] == 3, batch))
    return default_collate(batch)


class WeatherClass:

    def __init__(self, model, train_dataloader, test_dataloader, comment):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device != 'cuda':
            logging.warning('Could not establish connection to cuda')

        self.writer = SummaryWriter(log_dir=f'results/{comment}')

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.model = model.to(self.device)
        

    def train(self, eps, lr):
        # TODO pass these in as params
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1**(epoch+1))

        for ep in range(eps):
            logging.info(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
            self.model.train()

            train_loss = 0
            train_acc = 0
            for i,(X,y) in enumerate(self.train_dataloader):
                optimizer.zero_grad()
                X = X.to(self.device)
                y = y.to(self.device)

                pred = self.model(X)
                loss = loss_fn(pred, y)
                train_loss += loss.detach().item()
                train_acc += torch.sum(torch.argmax(pred, axis=1) == y).detach().item() / X.shape[0]
                loss.backward()
                optimizer.step()

                del X,y,loss,pred
                torch.cuda.empty_cache()

            train_acc /= len(self.train_dataloader)
            logging.info(f'EPOCH {ep}: Loss {round(train_loss, 4)}\t Accuracy {train_acc}')
            self.writer.add_scalar('Loss/train', train_loss, ep)
            self.writer.add_scalar('Accuracy/train', train_acc, ep)
            self.evaluate(loss_fn, ep)
            scheduler.step()

    def evaluate(self, loss_fn, ep):
        """
        Computes train/test accuracy and test loss
        """
        self.model.eval()

        test_loss = 0
        test_acc = 0
        for X,y in self.test_dataloader: 
            X = X.to(self.device)
            y = y.to(self.device)

            pred = self.model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.detach().item()
            test_acc += torch.sum(torch.argmax(pred, axis=1) == y).detach().item() / X.shape[0]
    
            del X,y,loss,pred
            torch.cuda.empty_cache()
        
        test_acc /= len(self.test_dataloader)
        self.writer.add_scalar('Loss/test', test_loss, ep)
        self.writer.add_scalar('Accuracy/test', test_acc, ep)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


        
parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, default='../../data/MWI/train_data')
parser.add_argument('--train_labels', type=str, default='../../data/MWI/train_labels')
parser.add_argument('--test_data', type=str, default='../../data/MWI/test_data')
parser.add_argument('--test_labels', type=str, default='../../data/MWI/test_labels')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--logging', type=int, default=2)
parser.add_argument('--comment', type=str, default='')
if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(level=logging_levels[args.logging])

    # define base model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
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
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=8)


    Weather = WeatherClass(model, train_dataloader, test_dataloader, args.comment)

    # train
    epochs = 20
    learning_rate = 0.001
    Weather.train(epochs, learning_rate)

    path = os.path.join('results', 'model_nofeatures_ep20lr001.pth')
    Weather.save_model(path)