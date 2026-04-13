from math import floor

import numpy as np
import torch
from torchmetrics import Metric

def yolo2matplotlib(bboxes: list[float], img_shape: tuple[int, int]) -> list[float]:
    """
    Converts the yolo bbox format to x, y, width and height for matplotlib plotting
    """
    img_x, img_y = img_shape

    width = floor(bboxes[2] * img_x)
    height = floor(bboxes[3] * img_y)

    left_x = floor(bboxes[0] * img_x - width/2)
    left_y = floor(bboxes[1] * img_y - height/2)

    return [left_x, left_y, width, height]

def yolo2torch(bboxes, img_shape: tuple[int, int]) -> torch.Tensor:
    if bboxes is None or len(bboxes) == 0:
        return torch.zeros((0, 4), dtype=torch.float32)

    bboxes = torch.tensor(bboxes, dtype=torch.float32)

    img_x, img_y = img_shape

    width = bboxes[:, 2] * img_x
    height = bboxes[:, 3] * img_y

    left_x = bboxes[:, 0] * img_x - width / 2
    left_y = bboxes[:, 1] * img_y - height / 2
    right_x = bboxes[:, 0] * img_x + width / 2
    right_y = bboxes[:, 1] * img_y + height / 2

    converted_bboxes = torch.stack((left_x, left_y, right_x, right_y), dim=1)

    return converted_bboxes


def yolo2yolo(bboxes: list[float], img_shape: tuple[int, int]) -> list[float]:
    """
    Dummy function for training YOLO model on dataset
    """
    return bboxes

def bbox_area(bboxes: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if bboxes is None or len(bboxes) == 0:
        return torch.zeros((0,), dtype=torch.float32)

    if isinstance(bboxes, np.ndarray):
        bboxes = torch.from_numpy(bboxes)

    area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    return area

def collate_fn(batch):
    return tuple(zip(*batch))

def is_better_value(val1, val2, metric_used: Metric):
    """Returns whether val2 is better than val1 based on metric_used"""
    if metric_used.higher_is_better:
        return val1 < val2
    else:
        return val1 > val2