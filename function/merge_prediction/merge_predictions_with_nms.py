import torch
from torchvision.ops import nms

from function.merge_prediction.regroup_predictions_filter_low_confidence_box import RegroupPredictionsFilterLowConfidenceBox


def merge_predictions_with_nms(predictionsModel1, predictionsModel2, iou_threshold=0.5, threshold=0.5):
    output = []
    for predictions, predictions2 in zip(predictionsModel1, predictionsModel2):
        merged_predictions = []
        for pred1, pred2 in zip(predictions, predictions2):
            combined_boxes, combined_scores, combined_labels = RegroupPredictionsFilterLowConfidenceBox(pred1, pred2, threshold)
            
            # Apply NMS to the combined boxes
            keep = nms(combined_boxes, combined_scores, iou_threshold)
            
            merged_boxes = combined_boxes[keep]
            merged_scores = combined_scores[keep]
            merged_labels = combined_labels[keep]
            
            merged_predictions.append({'boxes': merged_boxes, 'labels': merged_labels, 'scores': merged_scores})
        output.append(merged_predictions)
    return output