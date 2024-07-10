import torch
from ensemble_boxes import weighted_boxes_fusion

def merge_predictions_with_stats_filter_wbf(predictionsList, IoU_threshold=0.5, method='mean_std', factor=1.0, reduction_factor=0.5, percentileFactor=0.75):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ListListBoxes, ListListScores, ListListLables = regroup_predictions(predictionsList, method=method, factor=factor, reduction_factor=reduction_factor, percentileFactor= percentileFactor)
    fusionListList = []

    for boxes, scores, labels in zip(ListListBoxes, ListListScores, ListListLables):
        fusionList = []

        for box, score, label in zip(boxes, scores, labels):
            for idx, b in enumerate(box):
                box[idx] = b.cpu().numpy()
            for idx, s in enumerate(score):
                score[idx] = s.cpu().numpy()
            for idx, l in enumerate(label):
                label[idx] = l.cpu().numpy()

            fusion = {'boxes': [], 'scores': [], 'labels': []}
            resboxs, resscores, reslables = weighted_boxes_fusion(
                box, score, label, weights=None, iou_thr=IoU_threshold, skip_box_thr=0.0, conf_type='avg', allows_overflow=False
            )
            fusion['boxes'] = torch.tensor(resboxs, device=device)
            fusion['scores'] = torch.tensor(resscores, device=device)
            fusion['labels'] = torch.tensor(reslables, device=device).to(torch.int)

            fusionList.append(fusion)

        fusionListList.append(fusionList)

    return fusionListList

def regroup_predictions(predictionsList, method='mean_std', factor=1.0, reduction_factor=0.5, percentileFactor=0.75):
    boxes = []
    scores = []
    labels = []

    num_images = len(predictionsList[0])
    for i in range(num_images):
        boxe = []
        score = []
        label = []
        for model_predictions in predictionsList:
            pred = model_predictions[i]
            threshold = calculate_adaptive_threshold(pred[0]['scores'], method=method, factor=factor, percentileFactor=percentileFactor)

            pred_boxes, pred_scores, pred_labels = adjust_scores_below_threshold(pred[0]['boxes'], pred[0]['scores'], pred[0]['labels'], threshold, reduction_factor=reduction_factor)
            
            boxe.append(pred_boxes)
            score.append(pred_scores)
            label.append(pred_labels)
            
        boxes.append([boxe])
        scores.append([score])
        labels.append([label])
    

    return boxes, scores, labels

def adjust_scores_below_threshold(boxes, scores, labels, threshold, reduction_factor=0.5):
    adjusted_scores = scores.clone()
    adjusted_scores[scores < threshold] *= reduction_factor
    return boxes, adjusted_scores, labels

def calculate_adaptive_threshold(scores, method='mean_std', factor=1.0, percentileFactor=0.5):
    if method == 'mean_std':
        mean_score = scores.mean().item()
        std_score = scores.std().item()
        threshold = mean_score - factor * std_score
        if threshold < 0:
            threshold = -threshold
    elif method == 'median':
        median_score = scores.median().item()
        threshold = median_score
    elif method == 'percentile':
        threshold = torch.quantile(scores, percentileFactor).item()
    else:
        raise ValueError(f"Unknown method {method}. Choose from 'mean_std', 'median', or 'percentile'.")
    
    return threshold
