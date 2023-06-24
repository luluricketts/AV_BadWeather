import random

import cv2
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
import os

from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer
root_dir = '/home/roshini/AV_BadWeather/TransWeather/data/bdd/'
root_dir_test = '/home/roshini/AV_BadWeather/TransWeather/data/dawn_new/Fog_coco_restore'
        
register_coco_instances("kitti-train",
                        {},
                        os.path.join(root_dir, 'labels_coco/bdd100k_labels_new_train.json'),
                        image_root=os.path.join(root_dir, 'train/input'))
register_coco_instances("val",
                        {},
                        json_file=os.path.join(root_dir_test, 'labels.json'),
                        image_root=os.path.join(root_dir_test, 'data'))
metadata = MetadataCatalog.get("train")
# dataset_dicts = DatasetCatalog.get("train")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.DATASETS.TEST = ("val", )
predictor = DefaultPredictor(cfg)
# evaluator = COCOEvaluator(self.test_name, ('bbox',), False, output_dir=self.output_dir)
# self.trainer.test(test_cfg, self.trainer.model, evaluators=evaluator)
# # test_loader = build_detection_test_loader(test_cfg, self.test_name)
# # inference_on_dataset(predictor.model, test_loader, evaluator)

# onImage(self.test_name, self.output_dir, predictor=predictor, n=50)
img = cv2.imread("/home/roshini/AV_BadWeather/TransWeather/data/dawn_new/Fog_coco/data/foggy-001.jpg")
outputs = predictor(img)

# for d in random.sample(dataset_dicts, 1):
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=metadata,
#                    scale=0.8)
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2.imwrite('output_retrained.jpg', out.get_image()[:, :, ::-1])