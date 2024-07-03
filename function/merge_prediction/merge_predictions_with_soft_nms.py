import torch
from ensemble_boxes import soft_nms

def merge_predictions_with_soft_nms(predictionsModel1, predictionsModel2, IoU_threshold=0.5, threshold=0.5, sigma=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ListListBoxes, ListListScores, ListListLables = regroup_predictions(predictionsModel1, predictionsModel2, threshold)
    fusionListList = []
    clusterListList = []
    
    for boxes, scores, labels in zip(ListListBoxes, ListListScores, ListListLables):
        fusionList = []
        clusterList = []

        for box, score, label in zip(boxes, scores, labels):
            for idx, b in enumerate(box):
                box[idx] = b.cpu().numpy()
            for idx, s in enumerate(score):
                score[idx] = s.cpu().numpy()
            for idx, l in enumerate(label):
                label[idx] = l.cpu().numpy()
            fusion = {'boxes': [], 'scores': [], 'labels': []}
            cluster = {'boxes': [], 'scores': [], 'labels': []}
            fusionboxes = fusion['boxes']
            resboxs, resscores, reslables = soft_nms(box, score, label, iou_thr=IoU_threshold, sigma=sigma)            
            fusion['boxes'] = torch.tensor(resboxs, device=device)
            fusion['scores'] = torch.tensor(resscores, device=device)
            fusion['labels'] = torch.tensor(reslables, device=device).to(torch.int)

            fusionList.append(fusion)
            clusterList.append(cluster)
        fusionListList.append(fusionList)
        clusterListList.append(clusterList)

    return fusionListList


def regroup_predictions(predictionsModel1, predictionsModel2, threshold=0.5):
    boxes = []
    scores = []
    labels = []

    for predictions1, predictions2 in zip(predictionsModel1, predictionsModel2):
        boxesImages = []
        scoresImages = []
        labelsImages = []

        for pred1, pred2 in zip(predictions1, predictions2):

            scores1 = pred1['scores']
            scores2 = pred2['scores']
            scores1 = scores1 > threshold
            scores2 = scores2 > threshold
            pred1boxes = pred1['boxes']
            pred2boxes = pred2['boxes']
            pred1scores = pred1['scores']
            pred2scores = pred2['scores']
            pred1labels = pred1['labels']
            pred2labels = pred2['labels']
            if len(scores1) == 0:
                pred1boxes = torch.tensor([[0, 0, 0, 0]], dtype=torch.float32)
                pred1scores = torch.tensor([0], dtype=torch.float32)
                pred1labels = torch.tensor([0])
            else:
                pred1boxes = pred1['boxes'][scores1]
                pred1scores = pred1['scores'][scores1]
                pred1labels = pred1['labels'][scores1]
            if len(scores2) == 0:
                pred2boxes = torch.tensor([[0, 0, 0, 0]], dtype=torch.float32)
                pred2scores = torch.tensor([0], dtype=torch.float32)
                pred2labels = torch.tensor([0])
            else:
                pred2boxes = pred2['boxes'][scores2]
                pred2scores = pred2['scores'][scores2]
                pred2labels = pred2['labels'][scores2]

            boxe = [pred1boxes, pred2boxes]
            score = [pred1scores, pred2scores]
            label = [pred1labels, pred2labels]
            
            boxesImages.append(boxe)
            scoresImages.append(score)
            labelsImages.append(label)

        boxes.append(boxesImages)
        scores.append(scoresImages)
        labels.append(labelsImages)

    return boxes, scores, labels