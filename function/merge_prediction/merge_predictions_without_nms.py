import torch

def merge_predictions_without_nms(predictionsModel1, predictionsModel2, threshold=0.5):
    output = []
    for predictions1, predictions2 in zip(predictionsModel1, predictionsModel2):
        merged_predictions = []
        for pred1, pred2 in zip(predictions1, predictions2):
            boxes1, scores1, labels1 = pred1['boxes'], pred1['scores'], pred1['labels']
            boxes2, scores2, labels2 = pred2['boxes'], pred2['scores'], pred2['labels']
            
            # Combine boxes and scores
            combined_boxes = torch.cat((boxes1, boxes2), dim=0)
            combined_scores = torch.cat((scores1, scores2), dim=0)
            combined_labels = torch.cat((labels1, labels2), dim=0)

            # Filter out low-confidence boxes
            high_conf_idx = combined_scores >= threshold
            combined_boxes = combined_boxes[high_conf_idx]
            combined_scores = combined_scores[high_conf_idx]
            combined_labels = combined_labels[high_conf_idx]
            
            merged_predictions.append({'boxes': combined_boxes, 'labels': combined_labels, 'scores': combined_scores})
        output.append(merged_predictions)
    return output
