from math import floor

import numpy as np
import torch


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

def yolo2torch(bboxes: list[float], img_shape: tuple[int, int]) -> torch.Tensor:
    img_x, img_y = img_shape

    width = floor(bboxes[2] * img_x)
    height = floor(bboxes[3] * img_y)

    left_x = floor(bboxes[0] * img_x - width / 2)
    left_y = floor(bboxes[1] * img_y - height / 2)
    right_x = floor(bboxes[0] * img_x + width / 2)
    right_y = floor(bboxes[1] * img_y + height / 2)

    return torch.tensor([left_x, left_y, right_x, right_y], dtype=torch.float32)


def yolo2yolo(bboxes: list[float], img_shape: tuple[int, int]) -> list[float]:
    """
    Dummy function for training YOLO model on dataset
    """
    return bboxes

def bbox_area(bboxes: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    return area