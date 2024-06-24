import torch


def RegroupPredictionsFilterLowConfidenceBox(pred1, pred2, threshold=0.5):
    boxes1, scores1, labels1 = pred1['boxes'], pred1['scores'], pred1['labels']
    boxes2, scores2, labels2 = pred2['boxes'], pred2['scores'], pred2['labels']
            
     # Combine boxes and scores from both models
    combined_boxes = torch.cat((boxes1, boxes2), dim=0)
    combined_scores = torch.cat((scores1, scores2), dim=0)
    combined_labels = torch.cat((labels1, labels2), dim=0)

    # Filter out low-confidence boxes
    high_conf_idx = combined_scores >= threshold
    combined_boxes = combined_boxes[high_conf_idx]
    combined_scores = combined_scores[high_conf_idx]
    combined_labels = combined_labels[high_conf_idx] 
    
    return combined_boxes, combined_scores, combined_labels