from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer 
from detectron2.config import get_cfg
from detectron2 import model_zoo

import os
import cv2
import random
import matplotlib.pyplot as plt

def onImage(dataset_name, output_dir, predictor=None, n=1):

    dataset_custom = DatasetCatalog.get(dataset_name)
    metadata_custom = MetadataCatalog.get(dataset_name)

    for i,s in enumerate(random.sample(dataset_custom, n)):
        img = cv2.imread(s['file_name'])
        
        if predictor:
            fig = plt.figure(figsize=(20, 15))
            v = Visualizer(img[:,:,::-1], metadata=metadata_custom, scale=0.5)
            output = predictor(img)
            v = v.draw_instance_predictions(output['instances'].to('cpu'))
            fig.add_subplot(2, 1, 1)
            plt.imshow(v.get_image())
            plt.axis('off')
            plt.title('Predictions')
            
            v2 = Visualizer(img[:,:,::-1], metadata=metadata_custom, scale=0.5)
            v2 = v2.draw_dataset_dict(s)
            fig.add_subplot(2, 1, 2)
            plt.imshow(v2.get_image())
            plt.axis('off')
            plt.title('Ground Truth')

            plt.savefig(os.path.join(output_dir, f'img{i}.jpg'))
            

        else:
            v = v.draw_dataset_dict(s)
            plt.figure(figsize=(20, 15))    
            plt.imshow(v.get_image())
            plt.savefig(os.path.join(output_dir, f'img{i}.jpg'))


def get_train_cfg(cfg_file_path, checkpoint_url, train_name, test_name, output_dir):
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (test_name,)
    cfg.TEST.EVAL_PERIOD = 100

    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 0.00005
    cfg.SOLVER.MAX_ITER = 5000

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
    cfg.MODEL.RETINANET.NUM_CLASSES = 7
    cfg.MODEL.DEVICE = 'cuda'
    cfg.OUTPUT_DIR = output_dir

    return cfg

