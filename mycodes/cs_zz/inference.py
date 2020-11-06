import os
import glob
import sys
import xml.etree.ElementTree as ElementTree
sys.path.extend(['..', '../..'])

from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

label_map = {
    'smfpw': 0
}

data_dir = '/data/csy/cs_zz/mislead_v1/'


def get_rdd_dicts(mode='train'):
    with open(data_dir + 'annotations/' + mode + '.txt') as f:
        xml_files = f.readlines()

    dataset_dicts = []
    for idx, xml_file in enumerate(xml_files):
        xml_file = data_dir + 'annotations/xmls/' + xml_file.strip() + '.xml'
        if not os.path.isfile(xml_file):
            print(xml_file)
            continue
        tree = ElementTree.parse(xml_file)
        root = tree.getroot()

        filename = os.path.join(data_dir, 'images', root.find('filename').text)
        if not os.path.isfile(filename):
            print(filename)
            continue

        record = {"file_name": filename,
                  "image_id": idx,
                  "height": int(root.find('size').find('height').text),
                  "width": int(root.find('size').find('width').text)}

        objs = []
        for member in root.findall('object'):
            if len(member) < 5:
                continue
            if label_map.get(member.find('name').text) is None:
                continue
            obj = {
                "bbox": [int(member.find('bndbox')[0].text),
                         int(member.find('bndbox')[1].text),
                         int(member.find('bndbox')[2].text),
                         int(member.find('bndbox')[3].text)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [],
                "category_id": label_map[member.find('name').text],
                "iscrowd": 0
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


if __name__ == "__main__":
    from detectron2.data import DatasetCatalog, MetadataCatalog

    img_path = "/data/csy/cs_zz/mislead_v1/images/20201105"
    output_path = "/data/csy/cs_zz/mislead_v1/test/"
    for mode in ['train', 'val']:
        DatasetCatalog.register('flotage_'+mode, lambda mode=mode:get_flotage_dicts(args.data_dir, mode))
        MetadataCatalog.get('flotage_'+mode).set(thing_classes=list(label_map.keys()))

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ("flotage_train",)
    cfg.DATASETS.TEST = ("flotage_val",)
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 3000  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(label_map)

    cfg.MODEL.WEIGHTS = os.path.join('/data/csy/cs_zz/output/mislead_v1', "model_final.pth")
    predictor = DefaultPredictor(cfg)
    for img_name in os.listdir(img_path):
        if img_name.startswith("."):
            continue
        im = cv2.imread(os.path.join(img_path, img_name))

        outputs = predictor(im)
        print(outputs)
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        cv2.imwrite(os.path.join(output_path, img_name.split(".")[0] + ".jpg"), v.get_image()[:, :, ::-1])
