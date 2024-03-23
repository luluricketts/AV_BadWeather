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

from data.data import WeatherDataset, class_sampler, collate_fn, transform, eval_transform
import models


logging_levels = {
    1: logging.DEBUG,
    2: logging.INFO,
    3: logging.WARNING,
    4: logging.ERROR,
    5: logging.CRITICAL
}


class WeatherClass:

    def __init__(self, model, train_dataloader, val_dataloader, exp):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device != 'cuda':
            logging.warning('Could not establish connection to cuda')

        self.writer = SummaryWriter(log_dir=f'results/{exp}')

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        model = nn.DataParallel(model)
        self.model = model.to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_2 = nn.MSELoss()

    def train(self, eps, lr, early_stop_epoch):

        optimizer = Adam(self.model.parameters(), lr=lr)#, weight_decay=0.01)
        # scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.008, total_steps=eps * len(self.train_dataloader))
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=0.0000001)
        # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        
        patience = 0
        best_val_loss = 1000000
        for ep in range(eps):
            logging.info(f'-----------EPOCH {ep}-----------')
            logging.info(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
            self.model.train()

            train_loss = 0
            train_acc1 = 0
            train_acc2 = 0
            for i,(X,y,z,_) in enumerate(tqdm(self.train_dataloader)):
                
                optimizer.zero_grad()
                X = X.to(self.device).float()
                y = y.to(self.device)
                z = z.to(self.device)
                pred1, pred2 = self.model(X)
                loss = self.loss_fn(pred1, y)+ self.loss_fn(pred2, z)
                train_loss += loss.detach().item()
                train_acc1 += torch.sum(torch.argmax(pred1, axis=1) == y).detach().item()
                train_acc2 += torch.sum(torch.argmax(pred2, axis=1) == z).detach().item()
                loss.backward()
                optimizer.step()
                # scheduler.step()

                del X,y,z,loss,pred1,pred2
                torch.cuda.empty_cache()

            
            scheduler.step()

            val_loss = 0
            val_acc1 = 0
            val_acc2 = 0
            self.model.eval()
            for X,y,z,_ in self.val_dataloader: 
                X = X.to(self.device).float()
                y = y.to(self.device)
                z = z.to(self.device)

                pred1, pred2 = self.model(X)
                loss = self.loss_fn(pred1, y) 
                loss+= self.loss_fn(pred2, z)
                val_loss += loss.detach().item()
                val_acc1 += torch.sum(torch.argmax(pred1, axis=1) == y).detach().item()
                val_acc2 += torch.sum(torch.argmax(pred2, axis=1) == z).detach().item()
                del X,y,z,loss,pred1, pred2
                torch.cuda.empty_cache()
            
            val_acc1 /= len(self.val_dataloader.dataset)
            val_acc2 /= len(self.val_dataloader.dataset)
            train_acc2 /= len(self.train_dataloader.dataset)
            logging.info(f'EPOCH {ep}: \nTrain Loss {round(train_loss, 4)}\t Train Accuracy {train_acc2}')
            logging.info(f'\tVal Loss {round(val_loss, 4)}\t Val Accuracy {val_acc2}')
            self.writer.add_scalar('Loss/train', train_loss, ep)
            self.writer.add_scalar('Accuracy/train', train_acc2, ep)
            self.writer.add_scalar('Loss/val', val_loss, ep)
            self.writer.add_scalar('Accuracy/val', val_acc2, ep)

            # early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
            else:
                patience += 1
            
            if patience == early_stop_epoch:
                print(f"Early Stopped at Epoch: {ep}")
                return train_acc2, val_acc2
        return train_acc2, val_acc2
            
    def evaluate(self, dataloader):
        """
        Computes train/test accuracy and test loss
        """
        self.model.eval()

        test_loss = 0
        test_acc = 0
        for X,y,z,_ in dataloader: 
            X = X.to(self.device).float()
            y = y.to(self.device)
            z = z.to(self.device)
            pred1, pred2 = self.model(X)
            loss = self.loss_fn(pred1, y) 
            loss+= self.loss_fn(pred2, z)
            test_loss += loss.detach().item()
            test_acc += torch.sum(torch.argmax(pred2, axis=1) == z).detach().item()
    
            del X,y,loss,pred2
            torch.cuda.empty_cache()
        
        test_acc /= len(dataloader.dataset)
        return test_acc

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
    model = getattr(models, cfg["model"])(cfg["eng_feats"], cfg["freeze_backbone"], cfg["params"])

    train_data = WeatherDataset(cfg["metadata_train"], transform=transform if cfg["data_aug"] else eval_transform)
    val_data = WeatherDataset(cfg["metadata_val"], transform=eval_transform)
    test_data = WeatherDataset(cfg["metadata_test"], transform=eval_transform)
    train_dataloader = DataLoader(
        train_data, 
        batch_size=cfg["batch_size"], 
        sampler=class_sampler(cfg["class_counts"], cfg["metadata_train"]),
        collate_fn=collate_fn,
        num_workers=8
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8
    )
    test_dataloader = DataLoader(
        test_data, 
        batch_size=cfg["batch_size"], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=8)

    Weather = WeatherClass(model, train_dataloader, val_dataloader, args.exp_name)

    # train
    train_acc, val_acc = Weather.train(cfg["epochs"], cfg["lr"], cfg["early_stop_epoch"])
    test_accuracy = Weather.evaluate(test_dataloader)
    cfg["train_accuracy"] = train_acc
    cfg["val_accuracy"] = val_acc
    cfg["test_accuracy"] = test_accuracy
    print(train_acc, val_acc, test_accuracy)

    path = os.path.join('results', args.exp_name, f'{args.exp_name}.pth')
    Weather.save_model(path)

    with open(os.path.join("results", args.exp_name, "fog_config.yaml"), "w") as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)
