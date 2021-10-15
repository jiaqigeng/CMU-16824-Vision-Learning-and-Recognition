import os
import random
import time
import copy

import torch

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np


# TODO: given bounding boxes and corresponding scores, perform non max suppression
def nms(bounding_boxes, confidence_score, threshold=0.05):
    """
    bounding boxes of shape     Nx4
    confidence scores of shape  N
    threshold: confidence threshold for boxes to be considered

    return: list of bounding boxes and scores
    """
    filtered_bboxes = bounding_boxes[confidence_score > threshold, :].cpu().detach().numpy()
    filtered_scores = confidence_score[confidence_score > threshold].cpu().detach().numpy()

    boxes, scores = [], []
    while filtered_bboxes.shape[0] != 0:
        best_bbox = filtered_bboxes[np.argmax(filtered_scores), :]
        best_score = np.max(filtered_scores)
        boxes.append(best_bbox)
        scores.append(best_score)

        to_maintain = []
        for idx, proposed_box in enumerate(filtered_bboxes):
            if iou(best_bbox, proposed_box) < 0.3:
                to_maintain.append(idx)

        filtered_bboxes = np.take(filtered_bboxes, to_maintain, axis=0)
        filtered_scores = np.take(filtered_scores, to_maintain, axis=0)

    return boxes, scores


# TODO: calculate the intersection over union of two boxes
def iou(box1, box2):
    """
    Calculates Intersection over Union for two bounding boxes (xmin, ymin, xmax, ymax)
    returns IoU vallue
    """

    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)

    intersect_xmin = max(xmin1, xmin2)
    intersect_ymin = max(ymin1, ymin2)
    intersect_xmax = min(xmax1, xmax2)
    intersect_ymax = min(ymax1, ymax2)

    if intersect_xmax - intersect_xmin < 0 or intersect_ymax - intersect_ymin < 0:
        return 0

    intersect_area = (intersect_xmax - intersect_xmin) * (intersect_ymax - intersect_ymin)
    union_area = box1_area + box2_area - intersect_area

    iou = intersect_area / union_area
    return iou


def tensor_to_PIL(image):
    """
    converts a tensor normalized image (imagenet mean & std) into a PIL RGB image
    will not work with batches (if batch size is 1, squeeze before using this)
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255],
    )

    inv_tensor = inv_normalize(image)
    inv_tensor = torch.clamp(inv_tensor, 0, 1)
    original_image = transforms.ToPILImage()(inv_tensor).convert("RGB")
    return original_image


def get_box_data(classes, bbox_coordinates):
    """
    classes : tensor containing class predictions/gt
    bbox_coordinates: tensor containing [[xmin0, ymin0, xmax0, ymax0], [xmin1, ymin1, ...]] (Nx4)

    return list of boxes as expected by the wandb bbox plotter
    """
    box_list = [{
            "position": {
                "minX": bbox_coordinates[i][0],
                "minY": bbox_coordinates[i][1],
                "maxX": bbox_coordinates[i][2],
                "maxY": bbox_coordinates[i][3],
            },
            "class_id" : classes[i],
        } for i in range(len(classes))
        ]

    return box_list
