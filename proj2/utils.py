from math import floor

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

def yolo2yolo(bboxes: list[float], img_shape: tuple[int, int]) -> list[float]:
    """
    Dummy function for training YOLO model on dataset
    """
    return bboxes