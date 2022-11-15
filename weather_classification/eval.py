import argparse
import torch 
from torch.utils.data import DataLoader

from model import *
from data import WeatherDataset2
from train import transform, collate_dawn_only

def evaluate(model, dataloader):
        """
        Computes accuracy 
        """
        y_map = {0:0, 1:2, 2:1, 4:3}

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()

        acc = 0
        for X,y,_ in dataloader: 
            newy = torch.zeros(y.size())
            for i in range(y.size()[0]):
                newy[i] = torch.tensor(y_map[int(y[i])])

            X = X.to(device)
            y = newy.to(device)
            pred = model(X)
            acc += torch.sum(torch.argmax(pred, axis=1) == y).detach().item() / X.shape[0]
    
            del X,y,pred
            torch.cuda.empty_cache()
        
        test_acc /= len(dataloader)
        print(test_acc)


parser = argparse.ArgumentParser()
parser.add_argument('--params', type=str, default='results/model_nofeatures_ep40lr001gp8.pth')
parser.add_argument('--data_dir', type=str, default='../../data/weather_classification/data')
parser.add_argument('--train', type=str, default='../../data/weather_classification/train.json')
if __name__ == "__main__":
    args = parser.parse_args()
    
    model = get_resnet50(args.params, 4)

    dataset = WeatherDataset2(args.data_dir, args.train, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=len(dataset) // 10, 
        shuffle=True, 
        collate_fn=collate_dawn_only,
        num_workers=8
    )

    evaluate(model, dataloader)