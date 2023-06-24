import os
import argparse
import pickle

from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from train_net import Trainer
from utils import *

def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, val_dataset_name, num_classes, device, output_dir):
    cfg = get_cfg()
    
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)
    
    cfg.DATALOADER.NUM_WORKERS = 2
    
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = []
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir
    
    return cfg

class Detector:

    def __init__(self, exp_name):

        self.output_dir = os.path.join('results', exp_name)
        self.device = 'cuda'
        os.makedirs(self.output_dir, exist_ok=True)
        
        cfg_path = 'COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml'
        checkpoint_url = 'COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml'
        print("training started")
        # cfg_path = 'COCO-Detection/retinanet_R_50_FPN_3x.yaml'
        # checkpoint_url = 'COCO-Detection/retinanet_R_50_FPN_3x.yaml'
        print("checkpoints loaded")
        # root_dir = '/home/lulu/datasets/KITTI-coco'
        root_dir = '/home/roshini/AV_BadWeather/TransWeather/data/bdd/'
        root_dir_test = '/home/roshini/AV_BadWeather/TransWeather/data/dawn_new/Fog_coco'
        self.train_name = 'kitti-train'
        self.test_name = 'kitti-val'
        register_coco_instances(
            self.train_name, 
            metadata={}, 
            json_file=os.path.join(root_dir, 'labels_coco/bdd100k_labels_new_train.json'),
            image_root=os.path.join(root_dir, 'train/input')
        )
        register_coco_instances(
            self.test_name,
            metadata={},
            json_file=os.path.join(root_dir_test, 'labels.json'),
            image_root=os.path.join(root_dir_test, 'data')
        )


        cfg = get_cfg()
        # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.merge_from_file(model_zoo.get_config_file(cfg_path))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
        cfg.DATASETS.TRAIN = (self.train_name,)
        cfg.DATASETS.TEST = (self.test_name,)
        
        cfg.DATALOADER.NUM_WORKERS = 2
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.MAX_ITER = 3000
        # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
        self.cfg = cfg

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        self.trainer = DefaultTrainer(cfg)
        self.cfg_path = os.path.join(self.output_dir, 'cfg.yaml')
        with open(self.cfg_path, 'wb') as f:
            pickle.dump(self.cfg, f, protocol=pickle.HIGHEST_PROTOCOL)



    def train(self):
        # self.trainer = DefaultTrainer(self.cfg)
        self.trainer.resume_or_load(resume=False)
        self.trainer.train()


    def evaluate(self, test_cfg_path=None):
        
        if not test_cfg_path:
            test_cfg_path = self.cfg_path
        with open(test_cfg_path, 'rb') as f:
            test_cfg = pickle.load(f)
            
        test_cfg.MODEL.WEIGHTS = os.path.join(test_cfg.OUTPUT_DIR, 'model_final.pth')
        test_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        test_cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
        predictor = DefaultPredictor(test_cfg)

        evaluator = COCOEvaluator(self.test_name, ('bbox',), False, output_dir=self.output_dir)
        self.trainer.test(test_cfg, self.trainer.model, evaluators=evaluator)
        # test_loader = build_detection_test_loader(test_cfg, self.test_name)
        # inference_on_dataset(predictor.model, test_loader, evaluator)

        onImage(self.test_name, self.output_dir, predictor=predictor, n=50)




parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--eval_only', action='store_true')
if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    kitti_detector = Detector(args.exp_name)

    if not args.eval_only:
        kitti_detector.train()
    
    kitti_detector.evaluate()
