import torch
from torchvision.ops import nms


def merge_predictions_with_nms(predictionsModel1, predictionsModel2, iou_threshold=0.5, threshold=0.5):
    output = []
    for predictions, predictions2 in zip(predictionsModel1, predictionsModel2):
        merged_predictions = []
        for pred1, pred2 in zip(predictions, predictions2):
            boxes1, scores1 = pred1['boxes'], pred1['scores']
            boxes2, scores2 = pred2['boxes'], pred2['scores']
            
            # Combine boxes and scores from both models
            combined_boxes = torch.cat((boxes1, boxes2), dim=0)
            combined_scores = torch.cat((scores1, scores2), dim=0)

            # Filter out low-confidence boxes
            combined_boxes = combined_boxes[combined_scores >= threshold]
            combined_scores = combined_scores[combined_scores >= threshold]
            
            # Apply NMS to the combined boxes
            keep = nms(combined_boxes, combined_scores, iou_threshold)
            
            merged_boxes = combined_boxes[keep]
            merged_scores = combined_scores[keep]
            
            merged_predictions.append({'boxes': merged_boxes, 'scores': merged_scores})
        output.append(merged_predictions)
    return output

def merge_predictions_without_nms(predictionsModel1, predictionsModel2, threshold=0.5):
    output = []
    for predictions1, predictions2 in zip(predictionsModel1, predictionsModel2):
        merged_predictions = []
        for pred1, pred2 in zip(predictions1, predictions2):
            boxes1, scores1 = pred1['boxes'], pred1['scores']
            boxes2, scores2 = pred2['boxes'], pred2['scores']
            
            # Combine boxes and scores
            combined_boxes = torch.cat((boxes1, boxes2), dim=0)
            combined_scores = torch.cat((scores1, scores2), dim=0)

            # Filter out low-confidence boxes
            combined_boxes = combined_boxes[combined_scores >= threshold]
            combined_scores = combined_scores[combined_scores >= threshold]
            
            merged_predictions.append({'boxes': combined_boxes, 'scores': combined_scores})
        output.append(merged_predictions)
    return output