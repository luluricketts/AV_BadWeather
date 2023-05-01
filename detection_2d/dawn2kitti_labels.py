
import os
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='dawn2coco')
parser.add_argument('--dawn', type=str, default='../../datasets/Dawn/Fog_coco')
cfg = parser.parse_args()


if __name__ == "__main__":

    with open(os.path.join(cfg.dawn, 'labels.json'), 'r') as f:
        labels = json.load(f)


    attr_dict = {"categories":
        [
        {"supercategory": "none", "id": 1, "name": "Car"},
        {"supercategory": "none", "id": 2, "name": "Cyclist"},
        {"supercategory": "none", "id": 3, "name": "Pedestrian"},
        {"supercategory": "none", "id": 4, "name": "Person_sitting"},
        {"supercategory": "none", "id": 5, "name": "Tram"},
        {"supercategory": "none", "id": 6, "name": "Truck"},
        {"supercategory": "none", "id": 7, "name": "Van"},
        ]}

    old_dict = {"categories": 
        [
            {"id": 0,"name": "bicycle","supercategory": None},
            {"id": 1,"name": "bus","supercategory": None},
            {"id": 2,"name": "car","supercategory": None},
            {"id": 3,"name": "motorcycle","supercategory": None},
            {"id": 4,"name": "person","supercategory": None},
            {"id": 5,"name": "truck","supercategory": None}
        ]
    }
    dawn_dict = {i['id']: i['name'] for i in old_dict['categories']}

    category_map = {
        "person": "Pedestrian",
        "car": "Car",
        "bicycle": "Cyclist",
        "truck": "Truck",
        "bus": "Van",
        "motorcycle": "Cyclist",
    }
    id_dict = {i['name']: i['id'] for i in attr_dict['categories']}

    for i in labels['annotations']:
        i['category_id'] = id_dict[category_map[dawn_dict[i['category_id']]]]

    labels['categories'] = attr_dict['categories']


    with open(os.path.join(cfg.dawn, 'labels.json'), 'w') as f:
        json.dump(labels, f)