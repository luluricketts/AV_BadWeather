import os
import argparse
import pickle

from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from train_net import Trainer
from utils import *


class Detector:

    def __init__(self, exp_name):

        self.output_dir = os.path.join('results', exp_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # cfg_path = 'COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml'
        # checkpoint_url = 'COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml'
        cfg_path = 'COCO-Detection/retinanet_R_50_FPN_3x.yaml'
        checkpoint_url = 'COCO-Detection/retinanet_R_50_FPN_3x.yaml'

        # root_dir = '/home/lulu/datasets/KITTI-coco'
        root_dir = '/home/lulu/datasets/Dawn/Fog_coco_restore'
        self.train_name = 'kitti-train'
        self.test_name = 'kitti-val'
        register_coco_instances(
            self.train_name, 
            metadata={}, 
            json_file=os.path.join(root_dir, 'labels.json'),
            image_root=os.path.join(root_dir, 'data')
        )
        register_coco_instances(
            self.test_name,
            metadata={},
            json_file=os.path.join(root_dir, 'labels.json'),
            image_root=os.path.join(root_dir, 'data')
        )


        self.cfg = get_train_cfg(cfg_path, checkpoint_url, self.train_name, self.test_name, self.output_dir)
        self.cfg_path = os.path.join(self.output_dir, 'cfg.yaml')
        with open(self.cfg_path, 'wb') as f:
            pickle.dump(self.cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.trainer = Trainer(self.cfg)


    def train(self):
        
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
