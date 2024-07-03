import torch
from torchvision import models
from function.evaluate_performance_model import evaluate_performance_model
from function.get_prediction_model import getPredictionModel
from function.merge_prediction.merge_predictions_with_nms import merge_predictions_with_nms 
from function.merge_prediction.merge_predictions_without_nms import merge_predictions_without_nms
from function.merge_prediction.merge_predictions_with_wbf import merge_predictions_with_wbf
from function.merge_prediction.merge_predictions_with_extension_wbf import merge_predictions_with_extension_wbf
from function.merge_prediction.merge_predictions_with_soft_nms import merge_predictions_with_soft_nms
from function.visualize_prediction_image import plot_precision_recall_curve, show_comparison_image_models, visualize_prediction
from function.init_dataloader import initialiseDataloader

BDD100K_dataloader, Coco_dataloader = initialiseDataloader(isSubset=False)

# récupération du cpu ou gpu pour l'évaluation.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else"cpu"
print("Using {} device".format(device))

try:
    FasterRCNNModel = torch.load("models/FasterRCNNModel.pt")
except FileNotFoundError as e:
    FasterRCNNModel = models.detection.fasterrcnn_resnet50_fpn_v2(pretrained= True)
    FasterRCNNModel.to(device)
    torch.save(FasterRCNNModel, "models/FasterRCNNModel.pt")
    pass


try:
    RetinaNetModel = torch.load("models/RetinaNetModel.pt")
except FileNotFoundError as e:
    RetinaNetModel = models.detection.retinanet_resnet50_fpn_v2(pretrained=True)
    RetinaNetModel.to(device)
    torch.save(RetinaNetModel, "models/RetinaNetModel.pt")
    pass

try:
    FcosModel = torch.load("models/FcosModel.pt")
except FileNotFoundError as e:
    FcosModel = models.detection.fcos_resnet50_fpn(pretrained=True)
    FcosModel.to(device)
    torch.save(FcosModel, "models/FcosModel.pt")
    pass


# print("Evaluation du modèle FasterRCNNModel sans modification")
# predictionFasterRCNN, targetsOutFasterRCNN = getPredictionModel(FasterRCNNModel, BDD100K_dataloader, device)
# FasterRCNNModelBdd100KPrecision, FasterRCNNModelBdd100KRecall, FasterRCNNModelBdd100KTargetBoxes, FasterRCNNModelBdd100KPredBoxs, FasterRCNNModelBdd100KMap  = evaluate_performance_model(predictionFasterRCNN, targetsOutFasterRCNN)

# print("Evaluation du modèle RetinaNetModel sans modification")
# predictionRetinaNet, targetsOutRetinaNet = getPredictionModel(RetinaNetModel, BDD100K_dataloader, device)
# RetinaNetModelBdd100KPrecision, RetinaNetModelBdd100KRecall, RetinaNetModelBdd100KTargetBoxes, RetinaNetModelBdd100KPredBoxs, RetinaNetModelBdd100KMap = evaluate_performance_model(predictionRetinaNet, targetsOutRetinaNet)
           

# print("Evaluation du modèle FcosModel sans modification")
# predictionFcos, targetsOutFcos = getPredictionModel(FcosModel, BDD100K_dataloader, device)
# FcosModelBdd100KPrecision, FcosModelBdd100KRecall, FcosModelBdd100KTargetBoxes, FcosModelBdd100KPredBoxs, FcosModelBdd100KMap = evaluate_performance_model(predictionFcos, targetsOutFcos)


# print("Evaluation de la combinaisons des modèles FasrerRCNNModel et RetinaNet avec filtre de confiance à 0.5 sans NMS")
# merge_predictionsFasterRCNNRetinaNetWithoutNMS = merge_predictions_without_nms(predictionFasterRCNN, predictionRetinaNet)
# FasterRCNNRetinaNetModelBdd100KPrecision, FasterRCNNRetinaNetModelBdd100KRecall, FasterRCNNRetinaNetModelBdd100KTargetBoxes, FasterRCNNRetinaNetModelBdd100KPredBoxs, FasterRCNNRetinaNetModelBdd100KMap = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetWithoutNMS, targetsOutFasterRCNN)


# print("Evaluation de la combinaisons des modèles FasrerRCNNModel et RetinaNet avec filtre de confiance à 0.5 avec NMS")
# merge_predictionsFasterRCNNRetinaNetWithNMS = merge_predictions_with_nms(predictionFasterRCNN, predictionRetinaNet)
# FasterRCNNRetinaNetModelBdd100KPrecisionNMS, FasterRCNNRetinaNetModelBdd100KRecallNMS, FasterRCNNRetinaNetModelBdd100KTargetBoxesNMS, FasterRCNNRetinaNetModelBdd100KPredBoxsNMS, FasterRCNNRetinaNetModelBdd100KMapNMS = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetWithNMS, targetsOutFasterRCNN)

# print("Evaluation WBF avec filtre avant la fusion")
# merge_predictionsFasterRCNNRetinaNetWithWBF = merge_predictions_with_wbf(predictionFasterRCNN, predictionRetinaNet)
# FasterRCNNRetinaNetModelBdd100KPrecisionWBF, FasterRCNNRetinaNetModelBdd100KRecallWBF, FasterRCNNRetinaNetModelBdd100KTargetBoxesWBF, FasterRCNNRetinaNetModelBdd100KPredBoxsWBF, FasterRCNNRetinaNetModelBdd100KMapWBF = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetWithWBF, targetsOutFasterRCNN)

# # show_comparison_image_models([merge_predictionsFasterRCNNRetinaNetWithoutNMS, merge_predictionsFasterRCNNRetinaNetWithNMS, merge_predictionsFasterRCNNRetinaNetWithWBF], BDD100K_dataloader, targetsOutFasterRCNN, ["F-RCNN RNet Without NMS", "F-RCNN RNet NMS", "F-RCNN RNet WBF"], device)

# print("Evaluation de la combinaisons des modèles FasrerRCNNModel et Fcos avec filtre de confiance à 0.5 sans NMS")
# merge_predictionsFasterRCNNFcosWithoutNMS = merge_predictions_without_nms(predictionFasterRCNN, predictionFcos)
# FasterRCNNFcosModelBdd100KPrecision, FasterRCNNFcosModelBdd100KRecall, FasterRCNNFcosModelBdd100KTargetBoxes, FasterRCNNFcosModelBdd100KPredBoxs, FasterRCNNFcosModelBdd100KMap = evaluate_performance_model(merge_predictionsFasterRCNNFcosWithoutNMS, targetsOutFasterRCNN)

# print("Evaluation de la combinaisons des modèles FasrerRCNNModel et Fcos avec filtre de confiance à 0.5 avec NMS")
# merge_predictionsFasterRCNNFcosWithNMS = merge_predictions_with_nms(predictionFasterRCNN, predictionFcos)
# FasterRCNNFcosModelBdd100KPrecisionNMS, FasterRCNNFcosModelBdd100KRecallNMS, FasterRCNNFcosModelBdd100KTargetBoxesNMS, FasterRCNNFcosModelBdd100KPredBoxsNMS, FasterRCNNFcosModelBdd100KMapNMS = evaluate_performance_model(merge_predictionsFasterRCNNFcosWithNMS, targetsOutFasterRCNN)

# print("Evaluation de la combinaisons des modèles RetinaNetModel et Fcos avec IoU threshold à 0.5 et threshold évaluation à 0 avec wbf")
# merge_predictionsFasterRCNNFcosWithWBF = merge_predictions_with_wbf(predictionFasterRCNN, predictionFcos)
# FasterRCNNFcosModelBdd100KPrecisionWBFThreshold0, FasterRCNNFcosModelBdd100KRecallWBFThreshold0, FasterRCNNFcosModelBdd100KTargetBoxesWBFThreshold0, FasterRCNNFcosModelBdd100KPredBoxsWBFThreshold0, FasterRCNNFcosModelBdd100KMapWBFThreshold0 = evaluate_performance_model(merge_predictionsFasterRCNNFcosWithWBF, targetsOutFasterRCNN)

# # show_comparison_image_models([merge_predictionsFasterRCNNFcosWithoutNMS, merge_predictionsFasterRCNNFcosWithNMS, merge_predictionsFasterRCNNFcosWithWBF], BDD100K_dataloader, targetsOutFasterRCNN, ["F-RCNN Fcos Without NMS", "F-RCNN Fcos NMS", "F-RCNN Fcos WBF"], device)

# print("Evaluation de la combinaisons des modèles RetinaNetModel et Fcos avec filtre de confiance à 0.5 sans NMS")
# merge_predictionsRetinaNetFcosWithoutNMS = merge_predictions_without_nms(predictionRetinaNet, predictionFcos)
# RetinaNetFcosModelBdd100KPrecision, RetinaNetFcosModelBdd100KRecall, RetinaNetFcosModelBdd100KTargetBoxes, RetinaNetFcosModelBdd100KPredBoxs, RetinaNetFcosModelBdd100KMap = evaluate_performance_model(merge_predictionsRetinaNetFcosWithoutNMS, targetsOutFasterRCNN)

# print("Evaluation de la combinaisons des modèles RetinaNetModel et Fcos avec filtre de confiance à 0.5 avec NMS")
# merge_predictionsRetinaNetFcosWithNMS = merge_predictions_with_nms(predictionRetinaNet, predictionFcos)
# RetinaNetFcosModelBdd100KPrecisionNMS, RetinaNetFcosModelBdd100KRecallNMS, RetinaNetFcosModelBdd100KTargetBoxesNMS, RetinaNetFcosModelBdd100KPredBoxsNMS, RetinaNetFcosModelBdd100KMapNMS = evaluate_performance_model(merge_predictionsRetinaNetFcosWithNMS, targetsOutFasterRCNN)

# print("Evaluation de la combinaisons des modèles RetinaNetModel et Fcos avec IoU threshold à 0.5 et threshold évaluation à 0 avec wbf")
# merge_predictionsRetinaNetFcosWithWBF = merge_predictions_with_wbf(predictionRetinaNet, predictionFcos)
# RetinaNetFcosModelBdd100KPrecisionWBFThreshold0, RetinaNetFcosModelBdd100KRecallWBFThreshold0, RetinaNetFcosModelBdd100KTargetBoxesWBFThreshold0, RetinaNetFcosModelBdd100KPredBoxsWBFThreshold0, RetinaNetFcosModelBdd100KMapWBFThreshold0 = evaluate_performance_model(merge_predictionsRetinaNetFcosWithWBF, targetsOutFasterRCNN)

# print("Evaluation de la combinaisons des modèles FasrerRCNNModel, RetinaNet et Fcos avec filtre de confiance à 0.5 sans NMS")
# merge_predictionsFasterRCNNRetinaNetFcosWithoutNMS = merge_predictions_without_nms(merge_predictionsFasterRCNNRetinaNetWithWBF, predictionFcos)
# FasterRCNNRetinaNetFcosModelBdd100KPrecision, FasterRCNNRetinaNetFcosModelBdd100KRecall, FasterRCNNRetinaNetFcosModelBdd100KTargetBoxes, FasterRCNNRetinaNetFcosModelBdd100KPredBoxs, FasterRCNNRetinaNetFcosModelBdd100KMap = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetFcosWithoutNMS, targetsOutFasterRCNN)

# print("Evaluation de la combinaisons des modèles FasrerRCNNModel, RetinaNet et Fcos avec filtre de confiance à 0.5 avec NMS")
# merge_predictionsFasterRCNNRetinaNetFcosWithNMS = merge_predictions_with_nms(merge_predictionsFasterRCNNRetinaNetWithNMS, predictionFcos)
# FasterRCNNRetinaNetFcosModelBdd100KPrecisionNMS, FasterRCNNRetinaNetFcosModelBdd100KRecallNMS, FasterRCNNRetinaNetFcosModelBdd100KTargetBoxesNMS, FasterRCNNRetinaNetFcosModelBdd100KPredBoxsNMS, FasterRCNNRetinaNetFcosModelBdd100KMapNMS = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetFcosWithNMS, targetsOutFasterRCNN)

# print("Evaluation de la combinaisons des modèles FasrerRCNNModel, RetinaNet et Fcos avec IoU threshold à 0.5 et threshold évaluation à 0 avec wbf")
# merge_predictionsFasterRCNNRetinaNetFcosWithWBF = merge_predictions_with_wbf(merge_predictionsFasterRCNNRetinaNetWithWBF, predictionFcos, number_of_models=3)
# FasterRCNNRetinaNetFcosModelBdd100KPrecisionWBFThreshold0, FasterRCNNRetinaNetFcosModelBdd100KRecallWBFThreshold0, FasterRCNNRetinaNetFcosModelBdd100KTargetBoxesWBFThreshold0, FasterRCNNRetinaNetFcosModelBdd100KPredBoxsWBFThreshold0, FasterRCNNRetinaNetFcosModelBdd100KMapWBFThreshold0 = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetFcosWithWBF, targetsOutFasterRCNN)

# # show_comparison_image_models([merge_predictionsFasterRCNNRetinaNetFcosWithoutNMS, merge_predictionsFasterRCNNRetinaNetFcosWithNMS, merge_predictionsFasterRCNNRetinaNetFcosWithWBF], BDD100K_dataloader, targetsOutFasterRCNN, ["F-RCNN RNet Fcos Without NMS", "F-RCNN RNet Fcos NMS", "F-RCNN RNet Fcos WBF"], device)
# print("fini")


print("-------------------Evaluation des modèles sur le dataset COCO-------------------")

print("Evaluation du modèle FasterRCNNModel sans modification")
predictionFasterRCNN, targetsOutFasterRCNN = getPredictionModel(FasterRCNNModel, Coco_dataloader, device)
FasterRCNNModelCocoPrecision, FasterRCNNModelCocoRecall, FasterRCNNModelCocoTargetBoxes, FasterRCNNModelCocoPredBoxs,  FasterRCNNModelCocoMap = evaluate_performance_model(predictionFasterRCNN, targetsOutFasterRCNN)

print("Evaluation du modèle RetinaNetModel sans modification")
predictionRetinaNet, targetsOutRetinaNet = getPredictionModel(RetinaNetModel, Coco_dataloader, device)
RetinaNetModelCocoPrecision, RetinaNetModelCocoRecall, RetinaNetModelCocoTargetBoxes, RetinaNetModelCocoPredBoxs, RetinaNetModelCocoMap = evaluate_performance_model(predictionRetinaNet, targetsOutRetinaNet)

print("Evaluation du modèle FcosModel sans modification")
predictionFcos, targetsOutFcos = getPredictionModel(FcosModel, Coco_dataloader, device)
FcosModelCocoPrecision, FcosModelCocoRecall, FcosModelCocoTargetBoxes, FcosModelCocoPredBoxs, FcosModelCocoMap = evaluate_performance_model(predictionFcos, targetsOutFcos)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel et RetinaNet avec filtre de confiance à 0.5 sans NMS")
merge_predictionsFasterRCNNRetinaNetWithoutNMS = merge_predictions_without_nms(predictionFasterRCNN, predictionRetinaNet, threshold=0.001)
FasterRCNNRetinaNetModelCocoPrecision, FasterRCNNRetinaNetModelCocoRecall, FasterRCNNRetinaNetModelCocoTargetBoxes, FasterRCNNRetinaNetModelCocoPredBoxs, FasterRCNNRetinaNetModelCocoMap = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetWithoutNMS, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel et RetinaNet avec filtre de confiance à 0.5 avec NMS")
merge_predictionsFasterRCNNRetinaNetWithNMS = merge_predictions_with_nms(predictionFasterRCNN, predictionRetinaNet, threshold=0.001)
FasterRCNNRetinaNetModelCocoPrecisionNMS, FasterRCNNRetinaNetModelCocoRecallNMS, FasterRCNNRetinaNetModelCocoTargetBoxesNMS, FasterRCNNRetinaNetModelCocoPredBoxsNMS, FasterRCNNRetinaNetModelCocoMapNMS = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetWithNMS, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel et RetinaNet avec IoU threshold à 0.5 et threshold évaluation à 0 avec wbf")
merge_predictionsFasterRCNNRetinaNetWithWBF = merge_predictions_with_wbf(predictionFasterRCNN, predictionRetinaNet, threshold=0.001)
FasterRCNNRetinaNetModelCocoPrecisionWBFThreshold0, FasterRCNNRetinaNetModelCocoRecallWBFThreshold0, FasterRCNNRetinaNetModelCocoTargetBoxesWBFThreshold0, FasterRCNNRetinaNetModelCocoPredBoxsWBFThreshold0, FasterRCNNRetinaNetModelCocoMapWBFThreshold0 = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetWithWBF, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel et RetinaNet avec l'extension wbf")
FRCNNRNetExtensionWBF = merge_predictions_with_extension_wbf(predictionFasterRCNN, predictionRetinaNet, threshold=0.001)
FRCNNRNetExtensionWBFPrecision, FRCNNRNetExtensionWBFRecall, FRCNNRNetExtensionWBFTargetBoxes, FRCNNRNetExtensionWBFPredBoxs, FRCNNRNetExtensionWBFMap = evaluate_performance_model(FRCNNRNetExtensionWBF, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel et RetinaNet avec Soft NMS")
merge_predictionsFasterRCNNRetinaNetWithSoftNMS = merge_predictions_with_soft_nms(predictionFasterRCNN, predictionRetinaNet, threshold=0.001)
FasterRCNNRetinaNetModelCocoPrecisionSoftNMS, FasterRCNNRetinaNetModelCocoRecallSoftNMS, FasterRCNNRetinaNetModelCocoTargetBoxesSoftNMS, FasterRCNNRetinaNetModelCocoPredBoxsSoftNMS, FasterRCNNRetinaNetModelCocoMapSoftNMS = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetWithSoftNMS, targetsOutFasterRCNN)


print("Evaluation de la combinaisons des modèles FasrerRCNNModel et Fcos avec filtre de confiance à 0.5 sans NMS")
merge_predictionsFasterRCNNFcosWithoutNMS = merge_predictions_without_nms(predictionFasterRCNN, predictionFcos, threshold=0.001)
FasterRCNNFcosModelCocoPrecision, FasterRCNNFcosModelCocoRecall, FasterRCNNFcosModelCocoTargetBoxes, FasterRCNNFcosModelCocoPredBoxs, FasterRCNNFcosModelCocoMap = evaluate_performance_model(merge_predictionsFasterRCNNFcosWithoutNMS, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel et Fcos avec filtre de confiance à 0.5 avec NMS")
merge_predictionsFasterRCNNFcosWithNMS = merge_predictions_with_nms(predictionFasterRCNN, predictionFcos, threshold=0.001)
FasterRCNNFcosModelCocoPrecisionNMS, FasterRCNNFcosModelCocoRecallNMS, FasterRCNNFcosModelCocoTargetBoxesNMS, FasterRCNNFcosModelCocoPredBoxsNMS, FasterRCNNFcosModelCocoMapNMS = evaluate_performance_model(merge_predictionsFasterRCNNFcosWithNMS, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel et Fcos avec IoU threshold à 0.5 et threshold évaluation à 0 avec wbf")
merge_predictionsFasterRCNNFcosWithWBF = merge_predictions_with_wbf(predictionFasterRCNN, predictionFcos, threshold=0.001)
FasterRCNNFcosModelCocoPrecisionWBFThreshold0, FasterRCNNFcosModelCocoRecallWBFThreshold0, FasterRCNNFcosModelCocoTargetBoxesWBFThreshold0, FasterRCNNFcosModelCocoPredBoxsWBFThreshold0, FasterRCNNFcosModelCocoMapWBFThreshold0 = evaluate_performance_model(merge_predictionsFasterRCNNFcosWithWBF, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel et Fcos extension avec wbf")
FFcosExtensionWBF = merge_predictions_with_extension_wbf(predictionFasterRCNN, predictionFcos, threshold=0.001)
FFcosExtensionWBFPrecision, FFcosExtensionWBFRecall, FFcosExtensionWBFTargetBoxes, FFcosExtensionWBFPredBoxs, FFcosExtensionWBFMap = evaluate_performance_model(FFcosExtensionWBF, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel et Fcos avec Soft NMS")
merge_predictionsFasterRCNNFcosWithSoftNMS = merge_predictions_with_soft_nms(predictionFasterRCNN, predictionFcos, threshold=0.001)
FasterRCNNFcosModelCocoPrecisionSoftNMS, FasterRCNNFcosModelCocoRecallSoftNMS, FasterRCNNFcosModelCocoTargetBoxesSoftNMS, FasterRCNNFcosModelCocoPredBoxsSoftNMS, FasterRCNNFcosModelCocoMapSoftNMS = evaluate_performance_model(merge_predictionsFasterRCNNFcosWithSoftNMS, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles RetinaNetModel et Fcos avec filtre de confiance à 0.5 sans NMS")
merge_predictionsRetinaNetFcosWithoutNMS = merge_predictions_without_nms(predictionRetinaNet, predictionFcos, threshold=0.001)
RetinaNetFcosModelCocoPrecision, RetinaNetFcosModelCocoRecall, RetinaNetFcosModelCocoTargetBoxes, RetinaNetFcosModelCocoPredBoxs, RetinaNetFcosModelCocoMap = evaluate_performance_model(merge_predictionsRetinaNetFcosWithoutNMS, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles RetinaNetModel et Fcos avec filtre de confiance à 0.5 avec NMS")
merge_predictionsRetinaNetFcosWithNMS = merge_predictions_with_nms(predictionRetinaNet, predictionFcos, threshold=0.001)
RetinaNetFcosModelCocoPrecisionNMS, RetinaNetFcosModelCocoRecallNMS, RetinaNetFcosModelCocoTargetBoxesNMS, RetinaNetFcosModelCocoPredBoxsNMS, RetinaNetFcosModelCocoMapNMS = evaluate_performance_model(merge_predictionsRetinaNetFcosWithNMS, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles RetinaNetModel et Fcos avec IoU threshold à 0.5 et threshold évaluation à 0 avec wbf")
merge_predictionsRetinaNetFcosWithWBF = merge_predictions_with_wbf(predictionRetinaNet, predictionFcos, threshold=0.001)
RetinaNetFcosModelCocoPrecisionWBFThreshold0, RetinaNetFcosModelCocoRecallWBFThreshold0, RetinaNetFcosModelCocoTargetBoxesWBFThreshold0, RetinaNetFcosModelCocoPredBoxsWBFThreshold0, RetinaNetFcosModelCocoMapWBFThreshold0 = evaluate_performance_model(merge_predictionsRetinaNetFcosWithWBF, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles RetinaNet et Fcos avec l'extension wbf")
RetinaNetFcosExtensionWBF = merge_predictions_with_extension_wbf(predictionRetinaNet, predictionFcos, threshold=0.001)
RetinaNetFcosExtensionWBFPrecision, RetinaNetFcosExtensionWBFRecall, RetinaNetFcosExtensionWBFTargetBoxes, RetinaNetFcosExtensionWBFPredBoxs, RetinaNetFcosExtensionWBFMap = evaluate_performance_model(RetinaNetFcosExtensionWBF, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles RetinaNet et Fcos avec Soft NMS")
merge_predictionsRetinaNetFcosWithSoftNMS = merge_predictions_with_soft_nms(predictionRetinaNet, predictionFcos, threshold=0.001)
RetinaNetFcosModelCocoPrecisionSoftNMS, RetinaNetFcosModelCocoRecallSoftNMS, RetinaNetFcosModelCocoTargetBoxesSoftNMS, RetinaNetFcosModelCocoPredBoxsSoftNMS, RetinaNetFcosModelCocoMapSoftNMS = evaluate_performance_model(merge_predictionsRetinaNetFcosWithSoftNMS, targetsOutFasterRCNN)
print("Evaluation de la combinaisons des modèles FasrerRCNNModel, RetinaNet et Fcos avec filtre de confiance à 0.5 sans NMS")
merge_predictionsFasterRCNNRetinaNetFcosWithoutNMS = merge_predictions_without_nms(merge_predictionsFasterRCNNRetinaNetWithoutNMS, predictionFcos, threshold=0.001)
FasterRCNNRetinaNetFcosModelCocoPrecision, FasterRCNNRetinaNetFcosModelCocoRecall, FasterRCNNRetinaNetFcosModelCocoTargetBoxes, FasterRCNNRetinaNetFcosModelCocoPredBoxs, FasterRCNNRetinaNetFcosModelCocoMap = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetFcosWithoutNMS, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel, RetinaNet et Fcos avec filtre de confiance à 0.5 avec NMS")
merge_predictionsFasterRCNNRetinaNetFcosWithNMS = merge_predictions_with_nms(merge_predictionsFasterRCNNRetinaNetWithNMS, predictionFcos, threshold=0.001)
FasterRCNNRetinaNetFcosModelCocoPrecisionNMS, FasterRCNNRetinaNetFcosModelCocoRecallNMS, FasterRCNNRetinaNetFcosModelCocoTargetBoxesNMS, FasterRCNNRetinaNetFcosModelCocoPredBoxsNMS, FasterRCNNRetinaNetFcosModelCocoMapNMS = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetFcosWithNMS, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel, RetinaNet et Fcos avec IoU threshold à 0.5 et threshold évaluation à 0 avec wbf")
merge_predictionsFasterRCNNRetinaNetFcosWithWBF = merge_predictions_with_wbf(merge_predictionsFasterRCNNRetinaNetWithWBF, predictionFcos, threshold=0.001)
FasterRCNNRetinaNetFcosModelCocoPrecisionWBFThreshold0, FasterRCNNRetinaNetFcosModelCocoRecallWBFThreshold0, FasterRCNNRetinaNetFcosModelCocoTargetBoxesWBFThreshold0, FasterRCNNRetinaNetFcosModelCocoPredBoxsWBFThreshold0, FasterRCNNRetinaNetFcosModelCocoMapWBFThreshold0 = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetFcosWithWBF, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel, RetinaNet et Fcos avec l'extension wbf")
FRCNNRNetFcosExtensionWBF = merge_predictions_with_extension_wbf(merge_predictionsFasterRCNNRetinaNetWithWBF, predictionFcos, threshold=0.001)
FRCNNRNetFcosExtensionWBFPrecision, FRCNNRNetFcosExtensionWBFRecall, FRCNNRNetFcosExtensionWBFTargetBoxes, FRCNNRNetFcosExtensionWBFPredBoxs, FRCNNRNetFcosExtensionWBFMap = evaluate_performance_model(FRCNNRNetFcosExtensionWBF, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel, RetinaNet et Fcos avec Soft NMS")
merge_predictionsFasterRCNNRetinaNetFcosWithSoftNMS = merge_predictions_with_soft_nms(merge_predictionsFasterRCNNRetinaNetWithSoftNMS, predictionFcos, threshold=0.001)
FasterRCNNRetinaNetFcosModelCocoPrecisionSoftNMS, FasterRCNNRetinaNetFcosModelCocoRecallSoftNMS, FasterRCNNRetinaNetFcosModelCocoTargetBoxesSoftNMS, FasterRCNNRetinaNetFcosModelCocoPredBoxsSoftNMS, FasterRCNNRetinaNetFcosModelCocoMapSoftNMS = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetFcosWithSoftNMS, targetsOutFasterRCNN)
print("fini")
# # Ecrire dans un fichier tex les informations sous forme de tableau
# with open("Resultats/performance_model_mAPTest.tex", "w") as f:
#     f.write("\\documentclass{article}\n")
#     f.write("\\usepackage{graphicx} % Required for inserting images\n")
#     f.write("\\begin{document}\n")
#     f.write("\\begin{table}[h!]\n")
#     f.write("\\centering\n")
#     f.write("\\begin{tabular}{|c||c|c|c||c|c|c|} \n")
#     f.write("\\hline\n")
#     f.write("Model & \\multicolumn{3}{|c||}{BDD100K} & \\multicolumn{3}{|c|}{COCO} \\\\ \n")
#     f.write(" & precision & recall & mAP  & precision & recall & mAP  \\\\ [0.5ex] \n")
#     f.write("\\hline\n")
#     f.write("F R-CNN & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNModelBdd100KPrecision, FasterRCNNModelBdd100KRecall, FasterRCNNModelBdd100KMap, FasterRCNNModelCocoPrecision, FasterRCNNModelCocoRecall, FasterRCNNModelCocoMap))
#     f.write("\\hline\n")
#     f.write("RetinaNet & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RetinaNetModelBdd100KPrecision, RetinaNetModelBdd100KRecall, RetinaNetModelBdd100KMap, RetinaNetModelCocoPrecision, RetinaNetModelCocoRecall, RetinaNetModelCocoMap))
#     f.write("\\hline\n")
#     f.write("FCOS & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FcosModelBdd100KPrecision, FcosModelBdd100KRecall, FcosModelBdd100KMap, FcosModelCocoPrecision, FcosModelCocoRecall, FcosModelCocoMap))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RetinaNet Cat & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetModelBdd100KPrecision, FasterRCNNRetinaNetModelBdd100KRecall,FasterRCNNRetinaNetModelBdd100KMap , FasterRCNNRetinaNetModelCocoPrecision, FasterRCNNRetinaNetModelCocoRecall, FasterRCNNRetinaNetModelCocoMap))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RetinaNet NMS & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetModelBdd100KPrecisionNMS, FasterRCNNRetinaNetModelBdd100KRecallNMS, FasterRCNNRetinaNetModelBdd100KMapNMS, FasterRCNNRetinaNetModelCocoPrecisionNMS, FasterRCNNRetinaNetModelCocoRecallNMS, FasterRCNNRetinaNetModelCocoMapNMS))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RetinaNet WBF & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetModelBdd100KPrecisionWBF, FasterRCNNRetinaNetModelBdd100KRecallWBF, FasterRCNNRetinaNetModelBdd100KMapWBF, FasterRCNNRetinaNetModelCocoPrecisionWBFThreshold0, FasterRCNNRetinaNetModelCocoRecallWBFThreshold0, FasterRCNNRetinaNetModelCocoMapWBFThreshold0))
#     f.write("\\hline\n")
#     f.write("F R-CNN, FCOS Cat & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNFcosModelBdd100KPrecision, FasterRCNNFcosModelBdd100KRecall, FasterRCNNFcosModelBdd100KMap, FasterRCNNFcosModelCocoPrecision, FasterRCNNFcosModelCocoRecall, FasterRCNNFcosModelCocoMap))
#     f.write("\\hline\n")
#     f.write("F R-CNN, FCOS NMS & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNFcosModelBdd100KPrecisionNMS, FasterRCNNFcosModelBdd100KRecallNMS, FasterRCNNFcosModelBdd100KMapNMS, FasterRCNNFcosModelCocoPrecisionNMS, FasterRCNNFcosModelCocoRecallNMS, FasterRCNNFcosModelCocoMapNMS))
#     f.write("\\hline\n")
#     f.write("F R-CNN, FCOS WBF & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNFcosModelBdd100KPrecisionWBFThreshold0, FasterRCNNFcosModelBdd100KRecallWBFThreshold0, FasterRCNNFcosModelBdd100KMapWBFThreshold0, FasterRCNNFcosModelCocoPrecisionWBFThreshold0, FasterRCNNFcosModelCocoRecallWBFThreshold0, FasterRCNNFcosModelCocoMapWBFThreshold0))
#     f.write("\\hline\n")
#     f.write("RetinaNet, FCOS Cat & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RetinaNetFcosModelBdd100KPrecision, RetinaNetFcosModelBdd100KRecall, RetinaNetFcosModelBdd100KMap, RetinaNetFcosModelCocoPrecision, RetinaNetFcosModelCocoRecall, RetinaNetFcosModelCocoMap))
#     f.write("\\hline\n")
#     f.write("RetinaNet, FCOS NMS & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RetinaNetFcosModelBdd100KPrecisionNMS, RetinaNetFcosModelBdd100KRecallNMS, RetinaNetFcosModelBdd100KMapNMS, RetinaNetFcosModelCocoPrecisionNMS, RetinaNetFcosModelCocoRecallNMS, RetinaNetFcosModelCocoMapNMS))
#     f.write("\\hline\n")
#     f.write("RetinaNet, FCOS WBF & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RetinaNetFcosModelBdd100KPrecisionWBFThreshold0, RetinaNetFcosModelBdd100KRecallWBFThreshold0, RetinaNetFcosModelBdd100KMapWBFThreshold0, RetinaNetFcosModelCocoPrecisionWBFThreshold0, RetinaNetFcosModelCocoRecallWBFThreshold0, RetinaNetFcosModelCocoMapWBFThreshold0))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RetinaNet, FCOS Cat & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetFcosModelBdd100KPrecision, FasterRCNNRetinaNetFcosModelBdd100KRecall, FasterRCNNRetinaNetFcosModelBdd100KMap, FasterRCNNRetinaNetFcosModelCocoPrecision, FasterRCNNRetinaNetFcosModelCocoRecall, FasterRCNNRetinaNetFcosModelCocoMap))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RetinaNet, FCOS NMS & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetFcosModelBdd100KPrecisionNMS, FasterRCNNRetinaNetFcosModelBdd100KRecallNMS, FasterRCNNRetinaNetFcosModelBdd100KMapNMS, FasterRCNNRetinaNetFcosModelCocoPrecisionNMS, FasterRCNNRetinaNetFcosModelCocoRecallNMS, FasterRCNNRetinaNetFcosModelCocoMapNMS))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RetinaNet, FCOS WBF & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetFcosModelBdd100KPrecisionWBFThreshold0, FasterRCNNRetinaNetFcosModelBdd100KRecallWBFThreshold0, FasterRCNNRetinaNetFcosModelBdd100KMapWBFThreshold0, FasterRCNNRetinaNetFcosModelCocoPrecisionWBFThreshold0, FasterRCNNRetinaNetFcosModelCocoRecallWBFThreshold0, FasterRCNNRetinaNetFcosModelCocoMapWBFThreshold0))
#     f.write("\\hline\n")
#     f.write("\\end{tabular}\n")
#     f.write("\\caption{Résultats des différents modèles en fonction de la base de test.}\n")
#     f.write("\\label{table:data}\n")
#     f.write("\\end{table}\n")
#     f.write("\\end{document}\n")
#     f.close()
#     pass


# Affichage score coco
with open("Resultats/performance_model_mAPCocoFullData.tex", "w") as f:
    f.write("\\documentclass{article}\n")
    f.write("\\usepackage{graphicx} % Required for inserting images\n")
    f.write("\\begin{document}\n")
    f.write("\\begin{table}[h!]\n")
    f.write("\\centering\n")
    f.write("\\begin{tabular}{|c||c|c|c|} \n")
    f.write("\\hline\n")
    f.write("Model & precision & recall & mAP \\\\ \n")
    f.write("\\hline\n")
    f.write("F R-CNN & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNModelCocoPrecision, FasterRCNNModelCocoRecall, FasterRCNNModelCocoMap))
    f.write("\\hline\n")
    f.write("RetinaNet & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RetinaNetModelCocoPrecision, RetinaNetModelCocoRecall, RetinaNetModelCocoMap))
    f.write("\\hline\n")
    f.write("FCOS & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FcosModelCocoPrecision, FcosModelCocoRecall, FcosModelCocoMap))
    f.write("\\hline\n")
    f.write("F R-CNN, RetinaNet Cat & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetModelCocoPrecision, FasterRCNNRetinaNetModelCocoRecall, FasterRCNNRetinaNetModelCocoMap))
    f.write("\\hline\n")
    f.write("F R-CNN, RetinaNet NMS & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetModelCocoPrecisionNMS, FasterRCNNRetinaNetModelCocoRecallNMS, FasterRCNNRetinaNetModelCocoMapNMS))
    f.write("\\hline\n")
    f.write("F R-CNN, RetinaNet WBF & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetModelCocoPrecisionWBFThreshold0, FasterRCNNRetinaNetModelCocoRecallWBFThreshold0, FasterRCNNRetinaNetModelCocoMapWBFThreshold0))
    f.write("\\hline\n")
    f.write("F RCNN, RetinaNet Extension WBF & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetExtensionWBFPrecision, FRCNNRNetExtensionWBFRecall, FRCNNRNetExtensionWBFMap))
    f.write("\\hline\n")
    f.write("F R-CNN, RetinaNet Soft NMS & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetModelCocoPrecisionSoftNMS, FasterRCNNRetinaNetModelCocoRecallSoftNMS, FasterRCNNRetinaNetModelCocoMapSoftNMS))
    f.write("\\hline\n")
    f.write("F R-CNN, FCOS Cat & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNFcosModelCocoPrecision, FasterRCNNFcosModelCocoRecall, FasterRCNNFcosModelCocoMap))
    f.write("\\hline\n")
    f.write("F R-CNN, FCOS NMS & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNFcosModelCocoPrecisionNMS, FasterRCNNFcosModelCocoRecallNMS, FasterRCNNFcosModelCocoMapNMS))
    f.write("\\hline\n")
    f.write("F R-CNN, FCOS WBF & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNFcosModelCocoPrecisionWBFThreshold0, FasterRCNNFcosModelCocoRecallWBFThreshold0, FasterRCNNFcosModelCocoMapWBFThreshold0))
    f.write("\\hline\n")
    f.write("F R-CNN, FCOS Extension WBF & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FFcosExtensionWBFPrecision, FFcosExtensionWBFRecall, FFcosExtensionWBFMap))
    f.write("\\hline\n")
    f.write("F R-CNN, FCOS Soft NMS & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNFcosModelCocoPrecisionSoftNMS, FasterRCNNFcosModelCocoRecallSoftNMS, FasterRCNNFcosModelCocoMapSoftNMS))
    f.write("\\hline\n")
    f.write("RetinaNet, FCOS Cat & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RetinaNetFcosModelCocoPrecision, RetinaNetFcosModelCocoRecall, RetinaNetFcosModelCocoMap))
    f.write("\\hline\n")
    f.write("RetinaNet, FCOS NMS & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RetinaNetFcosModelCocoPrecisionNMS, RetinaNetFcosModelCocoRecallNMS, RetinaNetFcosModelCocoMapNMS))
    f.write("\\hline\n")
    f.write("RetinaNet, FCOS WBF & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RetinaNetFcosModelCocoPrecisionWBFThreshold0, RetinaNetFcosModelCocoRecallWBFThreshold0, RetinaNetFcosModelCocoMapWBFThreshold0))
    f.write("\\hline\n")
    f.write("RetinaNet, FCOS Extension WBF & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RetinaNetFcosExtensionWBFPrecision, RetinaNetFcosExtensionWBFRecall, RetinaNetFcosExtensionWBFMap))
    f.write("\\hline\n")
    f.write("RetinaNet, FCOS Soft NMS & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RetinaNetFcosModelCocoPrecisionSoftNMS, RetinaNetFcosModelCocoRecallSoftNMS, RetinaNetFcosModelCocoMapSoftNMS))
    f.write("\\hline\n")
    f.write("F R-CNN, RetinaNet, FCOS Cat & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetFcosModelCocoPrecision, FasterRCNNRetinaNetFcosModelCocoRecall, FasterRCNNRetinaNetFcosModelCocoMap))
    f.write("\\hline\n")
    f.write("F R-CNN, RetinaNet, FCOS NMS & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetFcosModelCocoPrecisionNMS, FasterRCNNRetinaNetFcosModelCocoRecallNMS, FasterRCNNRetinaNetFcosModelCocoMapNMS))
    f.write("\\hline\n")
    f.write("F R-CNN, RetinaNet, FCOS WBF & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetFcosModelCocoPrecisionWBFThreshold0, FasterRCNNRetinaNetFcosModelCocoRecallWBFThreshold0, FasterRCNNRetinaNetFcosModelCocoMapWBFThreshold0))
    f.write("\\hline\n")
    f.write("F R-CNN, RetinaNet, FCOS Extension WBF & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetFcosExtensionWBFPrecision, FRCNNRNetFcosExtensionWBFRecall, FRCNNRNetFcosExtensionWBFMap))
    f.write("\\hline\n")
    f.write("F R-CNN, RetinaNet, FCOS Soft NMS & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetFcosModelCocoPrecisionSoftNMS, FasterRCNNRetinaNetFcosModelCocoRecallSoftNMS, FasterRCNNRetinaNetFcosModelCocoMapSoftNMS))
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\caption{Résultats des différents modèles en fonction de la base de test.}\n")
    f.write("\\label{table:data}\n")
    f.write("\\end{table}\n")
    f.write("\\end{document}\n")
    f.close()
    pass
