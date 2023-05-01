import os
import json
import logging
import contextlib
import io

import numpy as np

from detectron2.utils.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode

logger = logging.getLogger(__name__)

def register_bdd100k(datasets_base="./datasets"):
    # cfg.INPUT.MASK_FORMAT = "bitmask" # necessary for the whole thing to work
    bdd100k_base = os.path.join(datasets_base, "bdd100k/")
    register_coco_instances(
        "bdd100k_ins_seg_train", {}, 
        os.path.join(bdd100k_base, "labels/ins_seg/ins_seg_cocofied_train.json"),
        os.path.join(bdd100k_base, "images/10k/train"))
    register_coco_instances(
        "bdd100k_ins_seg_val", {}, 
        os.path.join(bdd100k_base, "labels/ins_seg/ins_seg_cocofied_val.json"), 
        os.path.join(bdd100k_base, "images/10k/val"))
    return ("bdd100k_ins_seg_train", "bdd100k_ins_seg_val")

## Modified from detectron2/data/datasets/coco.py
def load_argoverse_hd_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    from pycocotools.coco import COCO

    json_file = PathManager.get_local_path(json_file)
    with open(json_file) as f:
        coco_json = json.load(f)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        MetadataCatalog.get(dataset_name).set(
            json_file=json_file, 
            image_root=image_root, 
            evaluator_type="coco",
            thing_classes=thing_classes,
            thing_dataset_id_to_contiguous_id=id_map,
            )

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        # the argoverse way of life
        path = os.path.join(coco_json["seq_dirs"][img_dict["sid"]], img_dict["name"])
        record["file_name"] = os.path.join(image_root, path)
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    return dataset_dicts

def register_argoverse_hd(datasets_base="./datasets"):
    argoverse_base = os.path.join(datasets_base, "Argoverse/")
    image_root = os.path.join(argoverse_base, "Argoverse-1.1/tracking/")
    json_file_train = os.path.join(argoverse_base, "Argoverse-HD/annotations/train.json")
    DatasetCatalog.register("argoverse_hd_train", 
        lambda : load_argoverse_hd_json(json_file_train, image_root, "argoverse_hd_train")
    )
    json_file_val = os.path.join(argoverse_base, "Argoverse-HD/annotations/val.json")
    DatasetCatalog.register("argoverse_hd_val", 
        lambda : load_argoverse_hd_json(json_file_val, image_root, "argoverse_hd_val")
    )
    return ("argoverse_hd_train", "argoverse_hd_val")


def load_nuimages_dicts(nuimages_base, split):

    from nuimages.nuimages import NuImages
    from pycocotools import mask
    import pycocotools.mask as mask_util

    version = "v1.0-{}".format(split)
    with contextlib.redirect_stdout(io.StringIO()):
        nuim = NuImages(dataroot=nuimages_base, version=version, verbose=True, lazy=True)

    categories = [data["name"] for data in nuim.category]
    categories_token = [data['token'] for data in nuim.category]
    id_map = {
        i : i for i in range(len(categories))
    }
    dataset_name = "nuimages_v1.0_{}".format(split)
    MetadataCatalog.get(dataset_name).set(
            image_root=nuimages_base, 
            evaluator_type="coco",
            thing_classes=categories,
            thing_dataset_id_to_contiguous_id=id_map,
    )

    dataset_dicts = []
    nuim.load_tables(['object_ann', 'sample_data', 'category', 'attribute'])

    for idx in range(0, len(nuim.sample)):
        data = nuim.sample_data[idx]
        # if only want CAM_FRONT, uncomment this 2 line
        if not (data['filename'][:18] =="samples/CAM_FRONT/"):
             continue
        record = {}
        record["file_name"] = os.path.join(nuimages_base, data["filename"])
        record["image_id"] = idx
        record["height"] = data["height"]
        record["width"] = data["width"]
        objs = []
        if data['is_key_frame']:

            objects = []
            for i in nuim.object_ann:
                if i['sample_data_token']==nuim.sample_data[idx]['token']:
                    objects.append(i)
            _, segs = nuim.get_segmentation(data['token'])
            objnum=1
            for object in objects:
                seg = (segs == objnum)
                objnum += 1
                seg = seg.astype('uint8')
                catid = categories_token.index(object['category_token'])
                obj = {
                    "bbox": object['bbox'],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": catid,
                    "iscrowd": 0,
                    "segmentation": mask.encode(np.asarray(seg, order="F"))
                }
                objs.append(obj)
        record["annotations"] = objs
        if len(objs) > 0:
            dataset_dicts.append(record)
    return dataset_dicts

def register_nuimages(datasets_base="./datasets"):
    nuimages_base = os.path.join(datasets_base, "nuimages/")

    DatasetCatalog.register("nuimages_v1.0_train", 
        lambda : load_nuimages_dicts(nuimages_base, "train")
    )
    DatasetCatalog.register("nuimages_v1.0_val", 
        lambda : load_nuimages_dicts(nuimages_base, "val")
    )
    return ("nuimages_v1.0_train", "nuimages_v1.0_val")

# datasets_base="./datasets"
# argoverse_base = os.path.join(datasets_base, "Argoverse/")
# image_root = os.path.join(argoverse_base, "Argoverse-1.1/tracking/")
# json_file_train = os.path.join(argoverse_base, "Argoverse-HD/annotations/train.json")
# load_argoverse_hd_json(json_file_train, image_root, "argoverse_hd_train")
# register_nuimages()