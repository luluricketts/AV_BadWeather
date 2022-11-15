
# from torch.utils.data import DataLoader
# import cv2
# import matplotlib.pyplot as plt

# from data import WeatherDataset
# from train import collate_fn, transform
# from feature import *

# data = '../../data/MWI/train_data'
# labels = '../../data/MWI/train_labels'
# train_data = WeatherDataset(data, labels, transform=transform)
# train_dataloader = DataLoader(
#         train_data, 
#         batch_size=1, 
#         shuffle=True, 
#         collate_fn=collate_fn,
#         num_workers=1
#     )


# for X,y in train_dataloader:
#     print(X.size(), X.dtype)

#     dark = dark_channel(X[0]).numpy()
#     print(dark.shape)
#     plt.imshow(dark, cmap='gray')
#     plt.savefig('dark.jpg')
#     plt.imshow(torch.movedim(X[0], 0, 2).numpy())
#     plt.savefig('img.jpg')
#     break





# combining all these datasets
import os
import shutil

home_dir = '../'
dirs = ['Dawn/20221010/Fog',
        'Dawn/20221010/Rain',
        'Dawn/20221010/Sand',
        'Dawn/20221010/Snow',
        'MWI/MWI-HAZE/',
        'MWI/MWI-SNOWY',
        'MWI/MWI-RAINY',
        'MWI/MWI-SUNNY'
        ]

labels = [2, 1, 3, 4, 2, 4, 1, 0]
srcs = ['D', 'D', 'D', 'D', 'MWI', 'MWI', 'MWI', 'MWI']
extensions = ['.jpg', '.JPG', '.png']



global_n = 0

def recurse(d):
    for file in os.listdir(d):

        if any(e in file for e in extensions):
            # shutil.copy(os.path.join(d, file), f'{global_n}.jpg')
            print(file)
        elif os.path.isdir(os.path.join(d, file)):
            recurse(os.path.join(d, file))
                                        
for d in dirs:
    recurse(os.path.join(home_dir, d))