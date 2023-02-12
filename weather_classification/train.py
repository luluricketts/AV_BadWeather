import argparse
import logging
import os 

from tqdm.auto import tqdm
import yaml
from yaml import safe_load as yload
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, Adam
from torch.utils.tensorboard import SummaryWriter

from data.data import WeatherDataset, class_sampler, collate_fn, transform
import models


logging_levels = {
    1: logging.DEBUG,
    2: logging.INFO,
    3: logging.WARNING,
    4: logging.ERROR,
    5: logging.CRITICAL
}


class WeatherClass:

    def __init__(self, model, train_dataloader, test_dataloader, exp):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device != 'cuda':
            logging.warning('Could not establish connection to cuda')

        self.writer = SummaryWriter(log_dir=f'results/{exp}')

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        model = nn.DataParallel(model)
        self.model = model.to(self.device)
        

    def train(self, eps, lr):
        # TODO pass these in as params
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        
        losses = []
        for ep in range(eps):
            logging.info(f'-----------EPOCH {ep}-----------')
            logging.info(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
            self.model.train()

            train_loss = 0
            train_acc = 0
            for i,(X,y,_) in enumerate(tqdm(self.train_dataloader)):
                
                optimizer.zero_grad()
                X = X.to(self.device).float()
                y = y.to(self.device)

                pred = self.model(X)
                loss = loss_fn(pred, y)
                train_loss += loss.detach().item()
                train_acc += torch.sum(torch.argmax(pred, axis=1) == y).detach().item()
                
                loss.backward()
                optimizer.step()

                del X,y,loss,pred
                torch.cuda.empty_cache()


            train_acc /= len(self.train_dataloader.dataset)
            losses.append(train_loss)
            logging.info(f'EPOCH {ep}: Loss {round(train_loss, 4)}\t Accuracy {train_acc}')
            self.writer.add_scalar('Loss/train', train_loss, ep)
            self.writer.add_scalar('Accuracy/train', train_acc, ep)
            self.evaluate(loss_fn, ep)
            scheduler.step()

        if len(losses) > 1 and losses[-1] - losses[-2] <= 0.001:
            return


    def evaluate(self, loss_fn, ep):
        """
        Computes train/test accuracy and test loss
        """
        self.model.eval()

        test_loss = 0
        test_acc = 0
        for X,y,_ in self.test_dataloader: 
            X = X.to(self.device).float()
            y = y.to(self.device)

            pred = self.model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.detach().item()
            test_acc += torch.sum(torch.argmax(pred, axis=1) == y).detach().item()
    
            del X,y,loss,pred
            torch.cuda.empty_cache()
        
        test_acc /= len(self.test_dataloader.dataset)
        self.writer.add_scalar('Loss/test', test_loss, ep)
        self.writer.add_scalar('Accuracy/test', test_acc, ep)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


        
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--config', type=str, default='data/config.yaml')
if __name__ == "__main__":

    args = parser.parse_args()
    with open(args.config, "r") as file:
        cfg = yload(file)
    logging.basicConfig(level=logging_levels[cfg["logging"]])
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["device_IDs"]  # specify which GPU(s) to be used
    
    # define base model
    model = getattr(models, cfg["model"])(cfg["params"])

    train_data = WeatherDataset(cfg["metadata_train"], transform=transform)
    test_data = WeatherDataset(cfg["metadata_test"], transform=transform)
    train_dataloader = DataLoader(
        train_data, 
        batch_size=cfg["batch_size"], 
        sampler=class_sampler(cfg["class_counts"], cfg["metadata_train"]),
        collate_fn=collate_fn,
        num_workers=8
    )
    test_dataloader = DataLoader(
        test_data, 
        batch_size=cfg["batch_size"], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=8)

    Weather = WeatherClass(model, train_dataloader, test_dataloader, args.exp_name)
    with open(os.path.join("results", args.exp_name, "config.yaml"), "w") as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)

    # train
    Weather.train(cfg["epochs"], cfg["lr"])

    path = os.path.join('results', args.exp_name, f'{args.exp_name}.pth')
    Weather.save_model(path)