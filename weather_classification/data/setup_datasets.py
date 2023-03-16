import os
import json
import yaml
import numpy as np

def reset_cfg(cfg):
    cfg["UID_train"] = 0
    cfg["UID_val"] = 0
    cfg["UID_test"] = 0
    for k in cfg["class_counts"]:
        cfg["class_counts"][k] = 0
    for k in cfg["val_class_counts"]:
        cfg["val_class_counts"][k] = 0
    for k in cfg["test_class_counts"]:
        cfg["test_class_counts"][k] = 0
    return cfg


def load_json(cfg):
    train_path = cfg["metadata_train"]
    val_path = cfg["metadata_val"]
    test_path = cfg["metadata_test"]
    if not os.path.isfile(train_path) or not os.path.isfile(test_path) or not os.path.isfile(val_path):
        open(train_path, "w+")
        open(val_path, "w+")
        open(test_path, "w+")
        cfg = reset_cfg(cfg)
    with open(train_path) as file: 
        try:
            train_json = json.load(file)
        except json.JSONDecodeError:
            train_json = {}
    with open(val_path) as file:
        try:
            val_json = json.load(file)
        except json.JSONDecodeError:
            val_json = {}
    with open(test_path) as file:
        try:
            test_json = json.load(file)
        except json.JSONDecodeError:
            test_json = {}

    return train_json, val_json, test_json, cfg


def add_MWI(cfg_file):
    train_json, val_json, test_json, cfg_file = load_json(cfg_file)

    mwi_base = os.path.join(cfg_file["dataset_path"], "MWI-Dataset")
    mwi_fog = os.path.join(mwi_base, "MWI-HAZE")
    mwi_rain = os.path.join(mwi_base, "MWI-RAINY")
    mwi_snow = os.path.join(mwi_base, "MWI-SNOWY")
    mwi_clear = os.path.join(mwi_base, "MWI-SUNNY")

    for i,cls_dir in enumerate([mwi_clear, mwi_rain, mwi_snow, mwi_fog]):
        
        shuffled = np.random.permutation(os.listdir(cls_dir))
        train_idx = int(len(shuffled) * cfg_file["train_ratio"])
        val_idx = int((len(shuffled) - train_idx) // 2) + train_idx
        for file in shuffled[:train_idx]:
            train_json[cfg_file["UID_train"]] = {
                "img_path" : os.path.join(cls_dir, file),
                "source" : "MWI",
                "label" : i,
            }
            cfg_file["UID_train"] += 1
            cfg_file["class_counts"][i] += 1
        for file in shuffled[train_idx:val_idx]:
            val_json[cfg_file["UID_val"]] = {
                "img_path" : os.path.join(cls_dir, file),
                "source" : "MWI",
                "label" : i,
            }
            cfg_file["UID_val"] += 1
            cfg_file["val_class_counts"][i] += 1
        for file in shuffled[val_idx:]:    
            test_json[cfg_file["UID_test"]] = {
                "img_path" : os.path.join(cls_dir, file),
                "source" : "MWI",
                "label" : i,
            }
            cfg_file["UID_test"] += 1
            cfg_file["test_class_counts"][i] += 1

    with open(cfg_file["metadata_train"], "w") as file:
        json.dump(train_json, file)
    with open(cfg_file["metadata_val"], "w") as file:
        json.dump(val_json, file)
    with open(cfg_file["metadata_test"], "w") as file:
        json.dump(test_json, file)

    return cfg_file


def add_Dawn(cfg_file):
    train_json, val_json, test_json, cfg_file = load_json(cfg_file)

    dawn_base = os.path.join(cfg_file["dataset_path"], "Dawn")
    dawn_fog = os.path.join(dawn_base, "Fog")
    dawn_rain = os.path.join(dawn_base, "Rain")
    dawn_snow = os.path.join(dawn_base, "Snow")

    for i,cls_dir in enumerate([dawn_rain, dawn_snow, dawn_fog], start=1):

        shuffled = np.random.permutation(os.listdir(cls_dir))
        train_idx = int(len(shuffled) * cfg_file["train_ratio"])
        val_idx = int((len(shuffled) - train_idx) // 2) + train_idx
        for file in shuffled[:train_idx]:
            if '.jpg' not in file:
                continue
            train_json[cfg_file["UID_train"]] = {
                "img_path" : os.path.join(cls_dir, file),
                "source" : "Dawn",
                "label" : i,
            }
            cfg_file["UID_train"] += 1
            cfg_file["class_counts"][i] += 1
        for file in shuffled[train_idx:val_idx]:
            if '.jpg' not in file:
                continue
            val_json[cfg_file["UID_val"]] = {
                "img_path" : os.path.join(cls_dir, file),
                "source" : "Dawn",
                "label" : i,
            }
            cfg_file["UID_val"] += 1
            cfg_file["val_class_counts"][i] += 1
        for file in shuffled[train_idx:]:
            if '.jpg' not in file:
                continue
            test_json[cfg_file["UID_test"]] = {
                "img_path" : os.path.join(cls_dir, file),
                "source" : "Dawn",
                "label" : i,
            }
            cfg_file["UID_test"] += 1
            cfg_file["test_class_counts"][i] += 1

    with open(cfg_file["metadata_train"], "w") as file:
        json.dump(train_json, file)
    with open(cfg_file["metadata_val"], "w") as file:
        json.dump(val_json, file)
    with open(cfg_file["metadata_test"], "w") as file:
        json.dump(test_json, file)

    return cfg_file


def add_bdd100k(cfg_file):
    train_json, val_json, test_json, cfg_file = load_json(cfg_file)

    bdd_base = os.path.join(cfg_file["dataset_path"], "bdd100k")
    bdd_train_img = os.path.join(bdd_base, "images", "100k", "train")
    bdd_val_img = os.path.join(bdd_base, "images", "100k", "val")
    with open(os.path.join(bdd_base, "labels", "bdd100k_labels_images_train.json"), "r") as file:
        bdd_train = json.load(file)
    with open(os.path.join(bdd_base, "labels", "bdd100k_labels_images_val.json"), "r") as file:
        bdd_val = json.load(file)

    for item in bdd_train:
        if item["attributes"]["weather"] not in cfg_file["labels"].values():
            continue
        label = list(cfg_file["labels"].values()).index(item["attributes"]["weather"])
        train_json[cfg_file["UID_train"]] = {
            "img_path" : os.path.join(bdd_train_img, item["name"]),
            "source" : "BDD100k",
            "label" : label
        }
        cfg_file["UID_train"] += 1
        cfg_file["class_counts"][label] += 1

    val_idx = int(len(bdd_val) // 2)
    for i,item in enumerate(bdd_val):
        if item["attributes"]["weather"] not in cfg_file["labels"].values():
            continue
        label = list(cfg_file["labels"].values()).index(item["attributes"]["weather"])
        if i < val_idx:
            val_json[cfg_file["UID_val"]] = {
                "img_path" : os.path.join(bdd_val_img, item["name"]),
                "source" : "BDD100k",
                "label" : label
            }
            cfg_file["UID_val"] += 1
            cfg_file["val_class_counts"][label] += 1
        else:
            test_json[cfg_file["UID_test"]] = {
                "img_path" : os.path.join(bdd_val_img, item["name"]),
                "source" : "BDD100k",
                "label" : label
            }
            cfg_file["UID_test"] += 1
            cfg_file["test_class_counts"][label] += 1

    with open(cfg_file["metadata_train"], "w") as file:
        json.dump(train_json, file)
    with open(cfg_file["metadata_val"], "w") as file:
        json.dump(val_json, file)
    with open(cfg_file["metadata_test"], "w") as file:
        json.dump(test_json, file)

    return cfg_file
    


def add_DENSE(cfg_file):

    train_json, val_json, test_json, cfg_file = load_json(cfg_file)

    dense_base = '/media/kolossus_data3/bad_weather_data/DENSE'
    dense_imgs = os.path.join(dense_base, 'cam_stereo_left_lut')
    dense_labels = os.path.join(dense_base, 'labeltool_labels')

    def read_label(file):
        
        with open(os.path.join(dense_labels, file.split('.')[0] + '.json'), 'r') as file:
            data = json.load(file)
        if data['weather']['clear']:
            return 0
        if data['weather']['rain']:
            return 1
        if data['weather']['snow']:
            return 2
        if data['weather']['dense_fog'] or data['weather']['light_fog']:
            return 3
  
        return -1


    shuffled = np.random.permutation(os.listdir(dense_imgs))
    train_idx = int(len(shuffled) * cfg_file["train_ratio"])
    val_idx = int((len(shuffled) - train_idx) // 2) + train_idx

    for file in shuffled[:train_idx]:
        if '.png' not in file:
            continue
        try:
            label = read_label(file)
            if label == -1: continue
        except:
            print('FAILED TO READ JSON')
            continue
        train_json[cfg_file["UID_train"]] = {
            "img_path" : os.path.join(dense_imgs, file),
            "source" : "DENSE",
            "label" : label,
        }
        cfg_file["UID_train"] += 1
        cfg_file["class_counts"][label] += 1
    
    for file in shuffled[train_idx:val_idx]:
        if '.png' not in file:
            continue
        try:
            label = read_label(file)
            if label == -1: continue
        except:
            print(f'Failed to read json for {file}')
            continue
        val_json[cfg_file["UID_val"]] = {
            "img_path" : os.path.join(dense_imgs, file),
            "source" : "DENSE",
            "label" : label,
        }
        cfg_file["UID_val"] += 1
        cfg_file["val_class_counts"][label] += 1
    
    for file in shuffled[val_idx:]:
        if '.png' not in file:
            continue
        try:
            label = read_label(file)
            if label == -1: continue
        except:
            print('FAILED TO READ JSON')
            continue
        test_json[cfg_file["UID_test"]] = {
            "img_path" : os.path.join(dense_imgs, file),
            "source" : "DENSE",
            "label" : label,
        }
        cfg_file["UID_test"] += 1
        cfg_file["test_class_counts"][label] += 1

    with open(cfg_file["metadata_train"], "w") as file:
        json.dump(train_json, file)
    with open(cfg_file["metadata_val"], "w") as file:
        json.dump(val_json, file)
    with open(cfg_file["metadata_test"], "w") as file:
        json.dump(test_json, file)

    return cfg_file