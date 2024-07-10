import torch
from ensemble_boxes import weighted_boxes_fusion

def merge_predictions_with_extension_wbf(predictionsList, IoU_threshold=0.5, threshold=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ListListBoxes, ListListScores, ListListLables = regroup_predictions(predictionsList, threshold)
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
            resboxs, resscores, reslables = weighted_boxes_fusion(box, score, label, weights=None, iou_thr=IoU_threshold, skip_box_thr=0.0, conf_type='avg', allows_overflow=False)            
            fusion['boxes'] = torch.tensor(resboxs, device=device)
            fusion['scores'] = torch.tensor(resscores, device=device)
            fusion['labels'] = torch.tensor(reslables, device=device).to(torch.int)

            fusionList.append(fusion)

        fusionListList.append(fusionList)

    return fusionListList


def regroup_predictions(predictionsList, threshold=0.5):
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

            predscore = pred[0]['scores']
            idxscore = predscore > threshold
            pred_boxes = pred[0]['boxes'][idxscore]
            pred_labels = pred[0]['labels'][idxscore]
            pred_scores = predscore[idxscore]

            boxe.append(pred_boxes)
            score.append(pred_scores)
            label.append(pred_labels)
            
        boxes.append([boxe])
        scores.append([score])
        labels.append([label])
    

    return boxes, scores, labels