import torch
from function.evaluate_performance_model import evaluate_performance_model
from function.merge_prediction.merge_predictions_with_wbf import merge_predictions_with_wbf
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import pytest

# # Test case 1: Empty prediction and target lists
# prediction = []
# targetsOut = []
# precision, recall, target_boxes, pred_boxes, mAP = evaluate_performance_model(prediction, targetsOut)
# assert precision == 0.0
# assert recall == 0.0
# assert target_boxes == 0
# assert pred_boxes == 0
# assert mAP == 0.0

# # Test case 2: Non-empty prediction and target lists
# prediction = [
#     [
#         {
#             'boxes': torch.tensor([[0, 0, 10, 10]]),
#             'scores': torch.tensor([0.9]),
#             'labels': torch.tensor([1])
#         }
#     ]
# ]
# targetsOut = [
#     [
#         {
#             'boxes': torch.tensor([[0, 0, 10, 10]]),
#             'labels': torch.tensor([1])
#         }
#     ]
# ]
# precision, recall, target_boxes, pred_boxes, mAP = evaluate_performance_model(prediction, targetsOut)
# assert precision == 1.0
# assert recall == 1.0
# assert target_boxes == 1
# assert pred_boxes == 1
# assert mAP == 100.0

# # Test case 3: Multiple predictions and targets
# prediction = [
#     [
#         {
#             'boxes': torch.tensor([[0, 0, 10, 10]]),
#             'scores': torch.tensor([0.9]),
#             'labels': torch.tensor([1])
#         },
#         {
#             'boxes': torch.tensor([[20, 20, 30, 30]]),
#             'scores': torch.tensor([0.8]),
#             'labels': torch.tensor([2])
#         }
#     ],
#     [
#         {
#             'boxes': torch.tensor([[0, 0, 10, 10]]),
#             'scores': torch.tensor([0.7]),
#             'labels': torch.tensor([1])
#         }
#     ]
# ]
# targetsOut = [
#     [
#         {
#             'boxes': torch.tensor([[0, 0, 10, 10]]),
#             'labels': torch.tensor([1])
#         },
#         {
#             'boxes': torch.tensor([[20, 20, 30, 30]]),
#             'labels': torch.tensor([2])
#         }
#     ],
#     [
#         {
#             'boxes': torch.tensor([[0, 0, 10, 10]]),
#             'labels': torch.tensor([1])
#         }
#     ]
# ]
# precision, recall, target_boxes, pred_boxes, mAP = evaluate_performance_model(prediction, targetsOut)
# assert precision == 1
# assert recall == 1
# assert target_boxes == 3
# assert pred_boxes == 3
# assert mAP == 100

#test 4 Multiprediction but missing a detection +1FN

# prediction = [
#     [
#         {
#             'boxes': torch.tensor([[0, 0, 10, 10]]),
#             'scores': torch.tensor([0.9]),
#             'labels': torch.tensor([1])
#         },
#         {
#             'boxes': torch.tensor([[20, 20, 30, 30]]),
#             'scores': torch.tensor([0.8]),
#             'labels': torch.tensor([2])
#         }
#     ],
#     [
#         {
#             'boxes': torch.tensor([[0, 0, 10, 10]]),
#             'scores': torch.tensor([0.7]),
#             'labels': torch.tensor([1])
#         }
#     ]
# ]

# targetsOut = [
#     [
#         {
#             'boxes': torch.tensor([[0, 0, 10, 10]]),
#             'labels': torch.tensor([1])
#         },
#         {
#             'boxes': torch.tensor([[20, 20, 30, 30]]),
#             'labels': torch.tensor([2])
#         }
#     ],
#     [
#         {
#             'boxes': torch.tensor([[0, 0, 10, 10], [40, 40, 50, 50]]),
#             'labels': torch.tensor([1, 1 ])
#         }
#     ]
# ]
 
# precision, recall, target_boxes, pred_boxes, mAP = evaluate_performance_model(prediction, targetsOut)
# assert precision == 1
# assert recall == 0.75
# assert target_boxes == 4
# assert pred_boxes == 3
# assert mAP == 100


#test 5 Multiprediction 4 boxes but 1FP so MAP = 75

prediction = [
    [
        {
            'boxes': torch.tensor([[0, 0, 10, 10]]),
            'scores': torch.tensor([0.7]),
            'labels': torch.tensor([1])
        }
        
    ],
    [
        {
            'boxes': torch.tensor([[20, 20, 30, 30]]),
            'scores': torch.tensor([0.6]),
            'labels': torch.tensor([2])
        }
    ],
    [
        {
            'boxes': torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]]),
            'scores': torch.tensor([0.7, 0.8]),
            'labels': torch.tensor([1, 2])
        }
    ]
]
targetsOut = [
    [
        {
            'boxes': torch.tensor([[0, 0, 10, 10]]),
            'labels': torch.tensor([1])
        }
        
    ],
    [
        {
            'boxes': torch.tensor([[20, 20, 30, 30]]),
            'labels': torch.tensor([2])
        }
    ],
    [
        {
            'boxes': torch.tensor([[0, 0, 10, 10]]),
            'labels': torch.tensor([1])
        }
    ]
]
metric = MeanAveragePrecision(class_metrics=True, extended_summary=True)
for listoutputs, listtargets in zip(prediction, targetsOut):
    print(listoutputs)
    print(listtargets)
    metric.update(preds=listoutputs, target=listtargets)
mAPDirect = metric.compute()
precision, recall, target_boxes, pred_boxes, mAP = evaluate_performance_model(prediction, targetsOut)
assert precision == 0.75
assert recall == 1
assert target_boxes == 3
assert pred_boxes == 4
assert mAP == 75
assert mAP == mAPDirect['map'].item()*100

# #test 6 Multiprediction 5 boxes but 1FP and a FN so MAP = 66.67

prediction = [
    [
        {
            'boxes': torch.tensor([[0, 0, 10, 10]]),
            'scores': torch.tensor([0.7]),
            'labels': torch.tensor([1])
        }
        
    ],
    [
        {
            'boxes': torch.tensor([[20, 20, 30, 30]]),
            'scores': torch.tensor([0.6]),
            'labels': torch.tensor([2])
        }
    ],
    [
        {
            'boxes': torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]]),
            'scores': torch.tensor([0.7, 0.8]),
            'labels': torch.tensor([1, 2])
        }
    ]
]
targetsOut = [
    [
        {
            'boxes': torch.tensor([[0, 0, 10, 10]]),
            'labels': torch.tensor([1])
        }
        
    ],
    [
        {
            'boxes': torch.tensor([[20, 20, 30, 30]]),
            'labels': torch.tensor([2])
        }
    ],
    [
        {
            'boxes': torch.tensor([[0, 0, 10, 10], [40, 40, 50, 50]]),
            'labels': torch.tensor([1, 1])
        }
    ]
]

metric = MeanAveragePrecision(class_metrics=True, extended_summary=True)
for listoutputs, listtargets in zip(prediction, targetsOut):
    print(listoutputs)
    print(listtargets)
    metric.update(preds=listoutputs, target=listtargets)
mAPDirect = metric.compute()
precision, recall, target_boxes, pred_boxes, mAP = evaluate_performance_model(prediction, targetsOut)
assert precision == 0.75
assert recall == 0.75
assert target_boxes == 4
assert pred_boxes == 4
assert mAP == mAPDirect['map'].item()*100


# #test 7 Multiprediction 4 boxes but IoU < 0.5 

prediction = [
    [
        {
            'boxes': torch.tensor([[0, 0, 8, 7]]),
            'scores': torch.tensor([0.7]),
            'labels': torch.tensor([1])
        }
        
    ],
    [
        {
            'boxes': torch.tensor([[12, 12, 22, 22]]),
            'scores': torch.tensor([0.6]),
            'labels': torch.tensor([2])
        }
    ],
    [
        {
            'boxes': torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]]),
            'scores': torch.tensor([0.7, 0.8]),
            'labels': torch.tensor([1, 2])
        }
    ]
]

targetsOut = [
    [
        {
            'boxes': torch.tensor([[0, 0, 10, 10]]),
            'labels': torch.tensor([1])
        }
        
    ],
    [
        {
            'boxes': torch.tensor([[20, 20, 30, 30]]),
            'labels': torch.tensor([2])
        }
    ],
    [
        {
            'boxes': torch.tensor([[0, 0, 10, 10], [40, 40, 50, 50]]),
            'labels': torch.tensor([1, 1])
        }
    ]
]

metric = MeanAveragePrecision(class_metrics=True, extended_summary=True)
for listoutputs, listtargets in zip(prediction, targetsOut):
    print(listoutputs)
    print(listtargets)
    metric.update(preds=listoutputs, target=listtargets)
mAPDirect = metric.compute()
precision, recall, target_boxes, pred_boxes, mAP = evaluate_performance_model(prediction, targetsOut)
assert precision == 0.5
assert recall == 0.5
assert target_boxes == 4
assert pred_boxes == 4
assert mAP == mAPDirect['map'].item()*100

#test 2 boites dont une fausses 

# targetsOut = [
#         {
#             'boxes': torch.tensor([[0, 0, 10, 10]]),
#             'labels': torch.tensor([1])
#         }
#     ]

# prediction = [
#     {
#             'boxes': torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]]),
#             'scores': torch.tensor([0.6, 0.7]),
#             'labels': torch.tensor([1, 1])
#         }
#     ]

# metric = MeanAveragePrecision(class_metrics=True, extended_summary=True)
# metric.update(preds=prediction, target=targetsOut)
# mAP = metric.compute()
# print(mAP)
print("All test cases passed!")