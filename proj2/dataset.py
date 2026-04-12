import os

import cv2
import torch
from albumentations import BboxParams
from matplotlib.transforms import Bbox
from torch.utils.data import Dataset
from torch import nn
import albumentations as A

from proj2.utils import yolo2torch
from utils import yolo2yolo, yolo2matplotlib, yolo2torch, bbox_area
from extract_data import get_mappings

from config import DATA_DIR


def _read_data_from_file(path):
    labels = []
    bboxes = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip().split()

            if len(line) != 5:
                raise ValueError("Expected 5 values for a label - class and bounding box coordinates")
            labels.append(int(line[0]))
            bboxes.append(tuple(map(float, line[1:])))
    return labels, bboxes


def _gather_labels(is_train):
    labels = {}

    if is_train:
        labels_path = DATA_DIR / "train" / "labels"
    else:
        labels_path = DATA_DIR / "test" / "labels"

    for idx, file in enumerate(labels_path.iterdir()):
        if not file.is_file():
            continue
        try:
            file_labels, file_bboxes = _read_data_from_file(file)
            labels[idx] = {
                "filename" : file.stem,
                "labels" : file_labels,
                "bboxes" : file_bboxes,
            }
        except ValueError as e:
            print(e)

    return labels


class FireDatasetForDetection(Dataset):
    def __init__(self, train=True, transforms: list | None = None, bbox_params: A.BboxParams | None = None, target_resolution = (512, 512)):

        if bbox_params is None:
            bbox_params = A.BboxParams(format = 'yolo', label_fields=['labels'], min_visibility=0.2, clip=True, filter_invalid_bboxes=True)

        if transforms is None or any([not isinstance(transforms, A.BasicTransform) for transforms in transforms]):
            print("Transforms can be only from albumentations library")
            self.transforms = [A.NoOp()]
        else:
            self.transforms = transforms

        self.target_res = target_resolution
        self.transforms = A.Compose([A.Resize(height=self.target_res[0], width=self.target_res[1])] + transforms, bbox_params=bbox_params, seed=42)

        self.data = _gather_labels(train)
        self.imgs_path = DATA_DIR / "train" / "images" if train else DATA_DIR / "test"
        self.file_mappings = get_mappings(train)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        datapoint_dict = self.data[idx]
        filename, labels, bboxes = datapoint_dict.values()
        image = cv2.imread(self.imgs_path / self.file_mappings[filename])

        transform_dict = self.transforms(image=image, labels=labels, bboxes=bboxes)

        transformed_img = transform_dict["image"]
        transformed_labels = transform_dict["labels"]
        transformed_bboxes = transform_dict["bboxes"]

        transformed_bboxes = yolo2torch(transformed_bboxes, self.target_res)
        bboxes_areas = bbox_area(transformed_bboxes)
        transformed_labels = torch.tensor([label + 1 for label in transformed_labels], dtype=torch.int8)

        return {
            "image" : transformed_img,
            "boxes" : transformed_bboxes,
            "labels" : transformed_labels,
            "image_id" : filename,
            "area" : bboxes_areas,
            "iscrowd" : torch.zeros(len(transformed_labels), dtype=torch.uint8)
        }

