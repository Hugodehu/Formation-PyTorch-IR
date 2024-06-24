import numpy as np
import torch
from torchvision.ops import box_iou


def compute_iou(boxes1, boxes2):
    """
    Compute Intersection over Union (IoU) between two lists of boxes.
    
    Args:
        boxes1 (list): List of boxes in the format [x1, y1, x2, y2].
        boxes2 (list): List of boxes in the format [x1, y1, x2, y2].
        
    Returns:
        list: IoU matrix representing the intersection over union between each pair of boxes.
    """
    iou_matrix = box_iou(boxes1, boxes2)
    return iou_matrix


import numpy as np

def bb_iou_array(boxes, new_box):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    - boxes (numpy.ndarray): An array representing the coordinates of the first bounding box [x1, y1, x2, y2].
    - new_box (tuple): A tuple representing the coordinates of the second bounding box (x1, y1, x2, y2).

    Returns:
    - iou (float): The IoU value of the two bounding boxes.

    The IoU is calculated as the ratio of the intersection area to the union area of the two bounding boxes.
    """
    # bb intersection over union
    boxX1, boxY1, boxX2, boxY2 = boxes.cpu().numpy()
    new_boxX1, new_boxY1, new_boxX2, new_boxY2 = new_box
    xA = np.maximum(boxX1, new_boxX1)
    yA = np.maximum(boxY1, new_boxY1)
    xB = np.minimum(boxX2, new_boxX2)
    yB = np.minimum(boxY2, new_boxY2)

    # compute the area of intersection rectangle
    interArea = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxX2 - boxX1) * (boxY2 - boxY1)
    boxBArea = (new_boxX2 - new_boxX1) * (new_boxY2 - new_boxY1)

    # compute the intersection over union by taking the intersection
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

