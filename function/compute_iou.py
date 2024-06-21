import torch
from torchvision.ops import box_iou


def compute_iou(boxes1, boxes2):
    """
    Compute Intersection over Union (IoU) between two sets of boxes.
    Each box is [x1, y1, x2, y2].
    """
    iou_matrix = box_iou(boxes1, boxes2)
    return iou_matrix