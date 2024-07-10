from matplotlib import pyplot as plt
import torch
from torchvision import models
from function.evaluate_performance_model import evaluate_performance_model
from function.get_prediction_model import getPredictionModel
from function.merge_prediction.merge_predictions_with_nms import merge_predictions_with_nms 
from function.merge_prediction.merge_predictions_without_nms import merge_predictions_without_nms
from function.merge_prediction.merge_predictions_with_wbf import merge_predictions_with_wbf
from function.merge_prediction.merge_predictions_with_extension_wbf import merge_predictions_with_extension_wbf
from function.merge_prediction.merge_predictions_with_soft_nms import merge_predictions_with_soft_nms
from function.merge_prediction.merge_predictions_with_stats_filter_wbf import merge_predictions_with_stats_filter_wbf
from function.visualize_prediction_image import plot_precision_recall_curve, show_comparison_image_models, visualize_prediction
from function.init_dataloader import initialiseDataloader

BDD100K_dataloader, Coco_dataloader = initialiseDataloader(isSubset=True, subsetSize=1)

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


print("Evaluation du modèle FasterRCNNModel sans modification")
predictionFasterRCNN, targetsOutFasterRCNN = getPredictionModel(FasterRCNNModel, BDD100K_dataloader, device)
FasterRCNNModelBdd100KPrecision, FasterRCNNModelBdd100KRecall, FasterRCNNModelBdd100KTargetBoxes, FasterRCNNModelBdd100KPredBoxs, FasterRCNNModelBdd100KMap  = evaluate_performance_model(predictionFasterRCNN, targetsOutFasterRCNN)

print("Evaluation du modèle RetinaNetModel sans modification")
predictionRetinaNet, targetsOutRetinaNet = getPredictionModel(RetinaNetModel, BDD100K_dataloader, device)
RetinaNetModelBdd100KPrecision, RetinaNetModelBdd100KRecall, RetinaNetModelBdd100KTargetBoxes, RetinaNetModelBdd100KPredBoxs, RetinaNetModelBdd100KMap = evaluate_performance_model(predictionRetinaNet, targetsOutRetinaNet)
           

print("Evaluation du modèle FcosModel sans modification")
predictionFcos, targetsOutFcos = getPredictionModel(FcosModel, BDD100K_dataloader, device)
FcosModelBdd100KPrecision, FcosModelBdd100KRecall, FcosModelBdd100KTargetBoxes, FcosModelBdd100KPredBoxs, FcosModelBdd100KMap = evaluate_performance_model(predictionFcos, targetsOutFcos)


show_comparison_image_models([predictionFasterRCNN, predictionRetinaNet, predictionFcos], BDD100K_dataloader, targetsOutFasterRCNN, ["FasterRCNNModel", "RetinaNetModel", "FcosModel"], device, threshold=0)

def evaluate_hyperparameters(predictionsList, IoU_threshold=0.5, methods=['percentile'], reduction_factors=[0.5], percentileFactors=[0.75], factors=[1.0]):
    results = {}
    for method in methods:
        results[method] = []
        if(method == 'percentile'):
            for reduction_factor in reduction_factors:
                for percentileFactor in percentileFactors:
                    fused_predictions = merge_predictions_with_stats_filter_wbf(predictionsList, IoU_threshold, method=method, reduction_factor=reduction_factor, percentileFactor=percentileFactor)                   
                    _, _, _, _, fused_predictionsMap = evaluate_performance_model(fused_predictions, targetsOutFasterRCNN)
                    results[method].append((reduction_factor, percentileFactor, fused_predictionsMap))
        elif(method == 'mean_std'):
            for factor in factors:
                for reduction_factor in reduction_factors:
                    fused_predictions = merge_predictions_with_stats_filter_wbf(predictionsList, IoU_threshold, method=method, reduction_factor=reduction_factor, factor=factor)                   
                    _, _, _, _, fused_predictionsMap = evaluate_performance_model(fused_predictions, targetsOutFasterRCNN)
                    results[method].append((factor, factor, fused_predictionsMap))
    return results

def plot_results_percentile(results, dataset = "BDD100K"):
    for method, data in results.items():
        _, percentileFactor, mAPs = zip(*data)
        plt.figure()
        plt.scatter(percentileFactor, mAPs, label=f'Percentile Factor = {percentileFactor}')
        plt.xlabel('Percentile Factor')
        plt.ylabel('mAP')
        plt.title(f'Impact of {method} on mAP on {dataset} dataset')
        plt.legend()
        plt.show()

def plot_results_reduction(results, dataset = "BDD100K"):
    for method, data in results.items():
        reduction_factors, _, mAPs = zip(*data)
        plt.figure()
        plt.scatter(reduction_factors, mAPs, label=f'Reduction Factor = {reduction_factors}')
        plt.xlabel('Reduction Factor')
        plt.ylabel('mAP')
        plt.title(f'Impact of {method} on mAP on {dataset} dataset')
        plt.legend()
        plt.show()

# factor= [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
# results = evaluate_hyperparameters([predictionFasterRCNN, predictionRetinaNet, predictionFcos], IoU_threshold=0.5, methods=['mean_std'], reduction_factors=[0.5], factors=factor)
# plot_results_percentile(results)

# reduction_factors = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
# results = evaluate_hyperparameters([predictionFasterRCNN, predictionRetinaNet, predictionFcos], IoU_threshold=0.5, methods=['mean_std'], reduction_factors=reduction_factors, factors=[1])
# plot_results_reduction(results)


# # percentileFactor = [0,0.1,0.15,0.2,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0]
# factor= [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
# reduction_factors = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
# results = evaluate_hyperparameters([predictionFasterRCNN, predictionRetinaNet, predictionFcos], IoU_threshold=0.5, methods=['mean_std'], reduction_factors=reduction_factors, factors=factor)
# plot_results_percentile(results)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel et RetinaNet avec filtre de confiance à 0.5 sans NMS")
merge_predictionsFasterRCNNRetinaNetWithoutNMS = merge_predictions_without_nms(predictionFasterRCNN, predictionRetinaNet, threshold=0.001)
FasterRCNNRetinaNetModelBdd100KPrecision, FasterRCNNRetinaNetModelBdd100KRecall, FasterRCNNRetinaNetModelBdd100KTargetBoxes, FasterRCNNRetinaNetModelBdd100KPredBoxs, FasterRCNNRetinaNetModelBdd100KMap = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetWithoutNMS, targetsOutFasterRCNN)


print("Evaluation de la combinaisons des modèles FasrerRCNNModel et RetinaNet avec filtre de confiance à 0.5 avec NMS")
merge_predictionsFasterRCNNRetinaNetWithNMS = merge_predictions_with_nms(predictionFasterRCNN, predictionRetinaNet, threshold=0.001)
FasterRCNNRetinaNetModelBdd100KPrecisionNMS, FasterRCNNRetinaNetModelBdd100KRecallNMS, FasterRCNNRetinaNetModelBdd100KTargetBoxesNMS, FasterRCNNRetinaNetModelBdd100KPredBoxsNMS, FasterRCNNRetinaNetModelBdd100KMapNMS = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetWithNMS, targetsOutFasterRCNN)

# print("Evaluation WBF avec filtre avant la fusion")
# merge_predictionsFasterRCNNRetinaNetWithWBF = merge_predictions_with_wbf(predictionFasterRCNN, predictionRetinaNet)
# FasterRCNNRetinaNetModelBdd100KPrecisionWBF, FasterRCNNRetinaNetModelBdd100KRecallWBF, FasterRCNNRetinaNetModelBdd100KTargetBoxesWBF, FasterRCNNRetinaNetModelBdd100KPredBoxsWBF, FasterRCNNRetinaNetModelBdd100KMapWBF = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetWithWBF, targetsOutFasterRCNN)

print("Evaluation FRCNN Rnet extension WBF")
FRCNNRNetExtensionWBFBDD100K = merge_predictions_with_extension_wbf([predictionFasterRCNN, predictionRetinaNet], threshold=0.001)
FRCNNRNetBDD100KExtensionWBFPrecision, FRCNNRNetBDD100KExtensionWBFRecall, FRCNNRNetBDD100KExtensionWBFTargetBoxes, FRCNNRNetBDD100KExtensionWBFPredBoxs, FRCNNRNetBDD100KExtensionWBFMap = evaluate_performance_model(FRCNNRNetExtensionWBFBDD100K, targetsOutFasterRCNN)

show_comparison_image_models([merge_predictionsFasterRCNNRetinaNetWithoutNMS, merge_predictionsFasterRCNNRetinaNetWithNMS, FRCNNRNetExtensionWBFBDD100K], BDD100K_dataloader, targetsOutFasterRCNN, ["F-RCNN RNet Without NMS", "F-RCNN RNet NMS", "F-RCNN RNet WBF"], device, threshold=0)

print("Evaluation FRCNN Rnet moyenne 1")
FRCNNRNetMoyenne1BDD100K = merge_predictions_with_stats_filter_wbf([predictionFasterRCNN, predictionRetinaNet], method='mean_std', factor=1.0)
FRCNNRNetBDD100KMoyenne1Precision, FRCNNRNetBDD100KMoyenne1Recall, FRCNNRNetBDD100KMoyenne1TargetBoxes, FRCNNRNetBDD100KMoyenne1PredBoxs, FRCNNRNetBDD100KMoyenne1Map = evaluate_performance_model(FRCNNRNetMoyenne1BDD100K, targetsOutFasterRCNN)

print("Evaluation FRCNN Rnet moyenne 2")
FRCNNRNetMoyenne2BDD100K = merge_predictions_with_stats_filter_wbf([predictionFasterRCNN, predictionRetinaNet], method='mean_std', factor=2.0)
FRCNNRNetBDD100KMoyenne2Precision, FRCNNRNetBDD100KMoyenne2Recall, FRCNNRNetBDD100KMoyenne2TargetBoxes, FRCNNRNetBDD100KMoyenne2PredBoxs, FRCNNRNetBDD100KMoyenne2Map = evaluate_performance_model(FRCNNRNetMoyenne2BDD100K, targetsOutFasterRCNN)

print("Evaluation FRCNN Rnet mediane")
FRCNNRNetMedianeBDD100K = merge_predictions_with_stats_filter_wbf([predictionFasterRCNN, predictionRetinaNet], method='median', factor=1.0)
FRCNNRNetBDD100KMedianePrecision, FRCNNRNetBDD100KMedianeRecall, FRCNNRNetBDD100KMedianeTargetBoxes, FRCNNRNetBDD100KMedianePredBoxs, FRCNNRNetBDD100KMedianeMap = evaluate_performance_model(FRCNNRNetMedianeBDD100K, targetsOutFasterRCNN)

print("Evaluation FRCNN Rnet percentile")
FRCNNRNetPercentileBDD100K = merge_predictions_with_stats_filter_wbf([predictionFasterRCNN, predictionRetinaNet], method='percentile', factor=1.0)
FRCNNRNetBDD100KPercentilePrecision, FRCNNRNetBDD100KPercentileRecall, FRCNNRNetBDD100KPercentileTargetBoxes, FRCNNRNetBDD100KPercentilePredBoxs, FRCNNRNetBDD100KPercentileMap = evaluate_performance_model(FRCNNRNetPercentileBDD100K, targetsOutFasterRCNN)

# # show_comparison_image_models([merge_predictionsFasterRCNNRetinaNetWithoutNMS, merge_predictionsFasterRCNNRetinaNetWithNMS, merge_predictionsFasterRCNNRetinaNetWithWBF], BDD100K_dataloader, targetsOutFasterRCNN, ["F-RCNN RNet Without NMS", "F-RCNN RNet NMS", "F-RCNN RNet WBF"], device)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel et Fcos avec filtre de confiance à 0.5 sans NMS")
merge_predictionsFasterRCNNFcosWithoutNMS = merge_predictions_without_nms(predictionFasterRCNN, predictionFcos)
FasterRCNNFcosModelBdd100KPrecision, FasterRCNNFcosModelBdd100KRecall, FasterRCNNFcosModelBdd100KTargetBoxes, FasterRCNNFcosModelBdd100KPredBoxs, FasterRCNNFcosModelBdd100KMap = evaluate_performance_model(merge_predictionsFasterRCNNFcosWithoutNMS, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel et Fcos avec filtre de confiance à 0.5 avec NMS")
merge_predictionsFasterRCNNFcosWithNMS = merge_predictions_with_nms(predictionFasterRCNN, predictionFcos)
FasterRCNNFcosModelBdd100KPrecisionNMS, FasterRCNNFcosModelBdd100KRecallNMS, FasterRCNNFcosModelBdd100KTargetBoxesNMS, FasterRCNNFcosModelBdd100KPredBoxsNMS, FasterRCNNFcosModelBdd100KMapNMS = evaluate_performance_model(merge_predictionsFasterRCNNFcosWithNMS, targetsOutFasterRCNN)

# print("Evaluation de la combinaisons des modèles RetinaNetModel et Fcos avec IoU threshold à 0.5 et threshold évaluation à 0 avec wbf")
# merge_predictionsFasterRCNNFcosWithWBF = merge_predictions_with_wbf(predictionFasterRCNN, predictionFcos)
# FasterRCNNFcosModelBdd100KPrecisionWBFThreshold0, FasterRCNNFcosModelBdd100KRecallWBFThreshold0, FasterRCNNFcosModelBdd100KTargetBoxesWBFThreshold0, FasterRCNNFcosModelBdd100KPredBoxsWBFThreshold0, FasterRCNNFcosModelBdd100KMapWBFThreshold0 = evaluate_performance_model(merge_predictionsFasterRCNNFcosWithWBF, targetsOutFasterRCNN)

print("Evaluation FRCNN Fcos extension WBF")
FFcosExtensionWBFBDD100K = merge_predictions_with_extension_wbf([predictionFasterRCNN, predictionFcos], threshold=0.001)
FFcosExtensionWBFBDD100KPrecision, FFcosExtensionWBFBDD100KRecall, FFcosExtensionWBFBDD100KTargetBoxes, FFcosExtensionWBFBDD100KPredBoxs, FFcosExtensionWBFBDD100KMap = evaluate_performance_model(FFcosExtensionWBFBDD100K, targetsOutFasterRCNN)

print("Evaluation FRCNN Fcos moyenne 1")
FFcosMoyenne1BDD100K = merge_predictions_with_stats_filter_wbf([predictionFasterRCNN, predictionFcos], method='mean_std', factor=1.0)
FFcosMoyenne1BDD100KPrecision, FFcosMoyenne1BDD100KRecall, FFcosMoyenne1BDD100KTargetBoxes, FFcosMoyenne1BDD100KPredBoxs, FFcosMoyenne1BDD100KMap = evaluate_performance_model(FFcosMoyenne1BDD100K, targetsOutFasterRCNN)

print("Evaluation FRCNN Fcos moyenne 2")
FFcosMoyenne2BDD100K = merge_predictions_with_stats_filter_wbf([predictionFasterRCNN, predictionFcos], method='mean_std', factor=2.0)
FFcosMoyenne2BDD100KPrecision, FFcosMoyenne2BDD100KRecall, FFcosMoyenne2BDD100KTargetBoxes, FFcosMoyenne2BDD100KPredBoxs, FFcosMoyenne2BDD100KMap = evaluate_performance_model(FFcosMoyenne2BDD100K, targetsOutFasterRCNN)

print("Evaluation FRCNN Fcos mediane")
FFcosMedianeBDD100K = merge_predictions_with_stats_filter_wbf([predictionFasterRCNN, predictionFcos], method='median', factor=1.0)
FFcosMedianeBDD100KPrecision, FFcosMedianeBDD100KRecall, FFcosMedianeBDD100KTargetBoxes, FFcosMedianeBDD100KPredBoxs, FFcosMedianeBDD100KMap = evaluate_performance_model(FFcosMedianeBDD100K, targetsOutFasterRCNN)

print("Evaluation FRCNN Fcos percentile")
FFcosPercentileBDD100K = merge_predictions_with_stats_filter_wbf([predictionFasterRCNN, predictionFcos], method='percentile', factor=1.0)
FFcosPercentileBDD100KPrecision, FFcosPercentileBDD100KRecall, FFcosPercentileBDD100KTargetBoxes, FFcosPercentileBDD100KPredBoxs, FFcosPercentileBDD100KMap = evaluate_performance_model(FFcosPercentileBDD100K, targetsOutFasterRCNN)

# # show_comparison_image_models([merge_predictionsFasterRCNNFcosWithoutNMS, merge_predictionsFasterRCNNFcosWithNMS, merge_predictionsFasterRCNNFcosWithWBF], BDD100K_dataloader, targetsOutFasterRCNN, ["F-RCNN Fcos Without NMS", "F-RCNN Fcos NMS", "F-RCNN Fcos WBF"], device)

print("Evaluation de la combinaisons des modèles RetinaNetModel et Fcos avec filtre de confiance à 0.5 sans NMS")
merge_predictionsRetinaNetFcosWithoutNMS = merge_predictions_without_nms(predictionRetinaNet, predictionFcos)
RetinaNetFcosModelBdd100KPrecision, RetinaNetFcosModelBdd100KRecall, RetinaNetFcosModelBdd100KTargetBoxes, RetinaNetFcosModelBdd100KPredBoxs, RetinaNetFcosModelBdd100KMap = evaluate_performance_model(merge_predictionsRetinaNetFcosWithoutNMS, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles RetinaNetModel et Fcos avec filtre de confiance à 0.5 avec NMS")
merge_predictionsRetinaNetFcosWithNMS = merge_predictions_with_nms(predictionRetinaNet, predictionFcos)
RetinaNetFcosModelBdd100KPrecisionNMS, RetinaNetFcosModelBdd100KRecallNMS, RetinaNetFcosModelBdd100KTargetBoxesNMS, RetinaNetFcosModelBdd100KPredBoxsNMS, RetinaNetFcosModelBdd100KMapNMS = evaluate_performance_model(merge_predictionsRetinaNetFcosWithNMS, targetsOutFasterRCNN)

# print("Evaluation de la combinaisons des modèles RetinaNetModel et Fcos avec IoU threshold à 0.5 et threshold évaluation à 0 avec wbf")
# merge_predictionsRetinaNetFcosWithWBF = merge_predictions_with_wbf(predictionRetinaNet, predictionFcos)
# RetinaNetFcosModelBdd100KPrecisionWBFThreshold0, RetinaNetFcosModelBdd100KRecallWBFThreshold0, RetinaNetFcosModelBdd100KTargetBoxesWBFThreshold0, RetinaNetFcosModelBdd100KPredBoxsWBFThreshold0, RetinaNetFcosModelBdd100KMapWBFThreshold0 = evaluate_performance_model(merge_predictionsRetinaNetFcosWithWBF, targetsOutFasterRCNN)


print("Evaluation RNet Fcos extension WBF")
RNetFcosExtensionWBFBDD100K = merge_predictions_with_extension_wbf([predictionRetinaNet, predictionFcos], threshold=0.001)
RNetFcosBDD100KExtensionWBFPrecision, RNetFcosBDD100KExtensionWBFRecall, RNetFcosBDD100KExtensionWBFTargetBoxes, RNetFcosBDD100KExtensionWBFPredBoxs, RNetFcosBDD100KExtensionWBFMap = evaluate_performance_model(RNetFcosExtensionWBFBDD100K, targetsOutFasterRCNN)

print("Evaluation RNet Fcos moyenne 1")
RNetFcosMoyenne1BDD100K = merge_predictions_with_stats_filter_wbf([predictionRetinaNet, predictionFcos], method='mean_std', factor=1.0)
RNetFcosBDD100KMoyenne1Precision, RNetFcosBDD100KMoyenne1Recall, RNetFcosBDD100KMoyenne1TargetBoxes, RNetFcosBDD100KMoyenne1PredBoxs, RNetFcosBDD100KMoyenne1Map = evaluate_performance_model(RNetFcosMoyenne1BDD100K, targetsOutFasterRCNN)

print("Evaluation RNet Fcos moyenne 2")
RNetFcosMoyenne2BDD100K = merge_predictions_with_stats_filter_wbf([predictionRetinaNet, predictionFcos], method='mean_std', factor=2.0)
RNetFcosBDD100KMoyenne2Precision, RNetFcosBDD100KMoyenne2Recall, RNetFcosBDD100KMoyenne2TargetBoxes, RNetFcosBDD100KMoyenne2PredBoxs, RNetFcosBDD100KMoyenne2Map = evaluate_performance_model(RNetFcosMoyenne2BDD100K, targetsOutFasterRCNN)

print("Evaluation RNet Fcos mediane")
RNetFcosMedianeBDD100K = merge_predictions_with_stats_filter_wbf([predictionRetinaNet, predictionFcos], method='median', factor=1.0)
RNetFcosBDD100KMedianePrecision, RNetFcosBDD100KMedianeRecall, RNetFcosBDD100KMedianeTargetBoxes, RNetFcosBDD100KMedianePredBoxs, RNetFcosBDD100KMedianeMap = evaluate_performance_model(RNetFcosMedianeBDD100K, targetsOutFasterRCNN)

print("Evaluation RNet Fcos percentile")
RNetFcosPercentileBDD100K = merge_predictions_with_stats_filter_wbf([predictionRetinaNet, predictionFcos], method='percentile', factor=1.0)
RNetFcosBDD100KPercentilePrecision, RNetFcosBDD100KPercentileRecall, RNetFcosBDD100KPercentileTargetBoxes, RNetFcosBDD100KPercentilePredBoxs, RNetFcosBDD100KPercentileMap = evaluate_performance_model(RNetFcosPercentileBDD100K, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel, RetinaNet et Fcos avec filtre de confiance à 0.5 sans NMS")
merge_predictionsFasterRCNNRetinaNetFcosWithoutNMS = merge_predictions_without_nms(merge_predictionsFasterRCNNFcosWithoutNMS, predictionFcos)
FasterRCNNRetinaNetFcosModelBdd100KPrecision, FasterRCNNRetinaNetFcosModelBdd100KRecall, FasterRCNNRetinaNetFcosModelBdd100KTargetBoxes, FasterRCNNRetinaNetFcosModelBdd100KPredBoxs, FasterRCNNRetinaNetFcosModelBdd100KMap = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetFcosWithoutNMS, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel, RetinaNet et Fcos avec filtre de confiance à 0.5 avec NMS")
merge_predictionsFasterRCNNRetinaNetFcosWithNMS = merge_predictions_with_nms(merge_predictionsFasterRCNNRetinaNetWithNMS, predictionFcos)
FasterRCNNRetinaNetFcosModelBdd100KPrecisionNMS, FasterRCNNRetinaNetFcosModelBdd100KRecallNMS, FasterRCNNRetinaNetFcosModelBdd100KTargetBoxesNMS, FasterRCNNRetinaNetFcosModelBdd100KPredBoxsNMS, FasterRCNNRetinaNetFcosModelBdd100KMapNMS = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetFcosWithNMS, targetsOutFasterRCNN)

# print("Evaluation de la combinaisons des modèles FasrerRCNNModel, RetinaNet et Fcos avec IoU threshold à 0.5 et threshold évaluation à 0 avec wbf")
# merge_predictionsFasterRCNNRetinaNetFcosWithWBF = merge_predictions_with_wbf(merge_predictionsFasterRCNNRetinaNetWithWBF, predictionFcos, number_of_models=3)
# FasterRCNNRetinaNetFcosModelBdd100KPrecisionWBFThreshold0, FasterRCNNRetinaNetFcosModelBdd100KRecallWBFThreshold0, FasterRCNNRetinaNetFcosModelBdd100KTargetBoxesWBFThreshold0, FasterRCNNRetinaNetFcosModelBdd100KPredBoxsWBFThreshold0, FasterRCNNRetinaNetFcosModelBdd100KMapWBFThreshold0 = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetFcosWithWBF, targetsOutFasterRCNN)


print("Evaluation FRCNN Rnet Fcos extension WBF")
FRCNNRNetFcosExtensionWBFBDD100K = merge_predictions_with_extension_wbf([predictionFasterRCNN, predictionRetinaNet, predictionFcos], threshold=0.001)
FRCNNRNetFcosBDD100KExtensionWBFPrecision, FRCNNRNetFcosBDD100KExtensionWBFRecall, FRCNNRNetFcosBDD100KExtensionWBFTargetBoxes, FRCNNRNetFcosBDD100KExtensionWBFPredBoxs, FRCNNRNetFcosBDD100KExtensionWBFMap = evaluate_performance_model(FRCNNRNetFcosExtensionWBFBDD100K, targetsOutFasterRCNN)

print("Evaluation FRCNN Rnet Fcos moyenne 1")
FRCNNRNetFcosMoyenne1BDD100K = merge_predictions_with_stats_filter_wbf([predictionFasterRCNN, predictionRetinaNet, predictionFcos], method='mean_std', factor=1.0)
FRCNNRNetFcosBDD100KMoyenne1Precision, FRCNNRNetFcosBDD100KMoyenne1Recall, FRCNNRNetFcosBDD100KMoyenne1TargetBoxes, FRCNNRNetFcosBDD100KMoyenne1PredBoxs, FRCNNRNetFcosBDD100KMoyenne1Map = evaluate_performance_model(FRCNNRNetFcosMoyenne1BDD100K, targetsOutFasterRCNN)

print("Evaluation FRCNN Rnet Fcos moyenne 2")
FRCNNRNetFcosMoyenne2BDD100K = merge_predictions_with_stats_filter_wbf([predictionFasterRCNN, predictionRetinaNet, predictionFcos], method='mean_std', factor=2.0)
FRCNNRNetFcosBDD100KMoyenne2Precision, FRCNNRNetFcosBDD100KMoyenne2Recall, FRCNNRNetFcosBDD100KMoyenne2TargetBoxes, FRCNNRNetFcosBDD100KMoyenne2PredBoxs, FRCNNRNetFcosBDD100KMoyenne2Map = evaluate_performance_model(FRCNNRNetFcosMoyenne2BDD100K, targetsOutFasterRCNN)

print("Evaluation FRCNN Rnet Fcos mediane")
FRCNNRNetFcosMedianeBDD100K = merge_predictions_with_stats_filter_wbf([predictionFasterRCNN, predictionRetinaNet, predictionFcos], method='median', factor=1.0)
FRCNNRNetFcosBDD100KMedianePrecision, FRCNNRNetFcosBDD100KMedianeRecall, FRCNNRNetFcosBDD100KMedianeTargetBoxes, FRCNNRNetFcosBDD100KMedianePredBoxs, FRCNNRNetFcosBDD100KMedianeMap = evaluate_performance_model(FRCNNRNetFcosMedianeBDD100K, targetsOutFasterRCNN)

print("Evaluation FRCNN Rnet Fcos percentile")
FRCNNRNetFcosPercentileBDD100K = merge_predictions_with_stats_filter_wbf([predictionFasterRCNN, predictionRetinaNet, predictionFcos], method='percentile', factor=1.0)
FRCNNRNetFcosBDD100KPercentilePrecision, FRCNNRNetFcosBDD100KPercentileRecall, FRCNNRNetFcosBDD100KPercentileTargetBoxes, FRCNNRNetFcosBDD100KPercentilePredBoxs, FRCNNRNetFcosBDD100KPercentileMap = evaluate_performance_model(FRCNNRNetFcosPercentileBDD100K, targetsOutFasterRCNN)


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

print("Evaluation de la combinaisons des modèles FasrerRCNNModel et RetinaNet avec Soft NMS")
merge_predictionsFasterRCNNRetinaNetWithSoftNMS = merge_predictions_with_soft_nms(predictionFasterRCNN, predictionRetinaNet, threshold=0.001)
FasterRCNNRetinaNetModelCocoPrecisionSoftNMS, FasterRCNNRetinaNetModelCocoRecallSoftNMS, FasterRCNNRetinaNetModelCocoTargetBoxesSoftNMS, FasterRCNNRetinaNetModelCocoPredBoxsSoftNMS, FasterRCNNRetinaNetModelCocoMapSoftNMS = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetWithSoftNMS, targetsOutFasterRCNN)

# print("Evaluation de la combinaisons des modèles FasrerRCNNModel et RetinaNet avec IoU threshold à 0.5 et threshold évaluation à 0 avec wbf")
# merge_predictionsFasterRCNNRetinaNetWithWBF = merge_predictions_with_wbf(predictionFasterRCNN, predictionRetinaNet, threshold=0.5)
# FasterRCNNRetinaNetModelCocoPrecisionWBFThreshold0, FasterRCNNRetinaNetModelCocoRecallWBFThreshold0, FasterRCNNRetinaNetModelCocoTargetBoxesWBFThreshold0, FasterRCNNRetinaNetModelCocoPredBoxsWBFThreshold0, FasterRCNNRetinaNetModelCocoMapWBFThreshold0 = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetWithWBF, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel et RetinaNet avec l'extension wbf")
FRCNNRNetExtensionWBF = merge_predictions_with_extension_wbf([predictionFasterRCNN, predictionRetinaNet], threshold=0.001)
FRCNNRNetExtensionWBFPrecision, FRCNNRNetExtensionWBFRecall, FRCNNRNetExtensionWBFTargetBoxes, FRCNNRNetExtensionWBFPredBoxs, FRCNNRNetExtensionWBFMap = evaluate_performance_model(FRCNNRNetExtensionWBF, targetsOutFasterRCNN)

print("Evaluation filtre statistique moyenne avec facteur de 1")
FRCNNRNetMoyenne1 = merge_predictions_with_stats_filter_wbf([predictionFasterRCNN, predictionRetinaNet], method='mean_std', factor=1.0)
FRCNNRNetMoyenne1Precision, FRCNNRNetMoyenne1Recall, FRCNNRNetMoyenne1TargetBoxes, FRCNNRNetMoyenne1PredBoxs, FRCNNRNetMoyenne1Map = evaluate_performance_model(FRCNNRNetMoyenne1, targetsOutFasterRCNN)

print("Evaluation filtre statistique moyenne avec facteur de 2")
FRCNNRNetMoyenne2 = merge_predictions_with_stats_filter_wbf([predictionFasterRCNN, predictionRetinaNet], method='mean_std', factor=2.0)
FRCNNRNetMoyenne2Precision, FRCNNRNetMoyenne2Recall, FRCNNRNetMoyenne2TargetBoxes, FRCNNRNetMoyenne2PredBoxs, FRCNNRNetMoyenne2Map = evaluate_performance_model(FRCNNRNetMoyenne2, targetsOutFasterRCNN)

print("Evaluation filtre statistique Mediane")
FRCNNRNetMediane = merge_predictions_with_stats_filter_wbf([predictionFasterRCNN, predictionRetinaNet], method='median', factor=1.0)
FRCNNRNetMedianePrecision, FRCNNRNetMedianeRecall, FRCNNRNetMedianeTargetBoxes, FRCNNRNetMedianePredBoxs, FRCNNRNetMedianeMap = evaluate_performance_model(FRCNNRNetMediane, targetsOutFasterRCNN)

print("Evaluation filtre statistique Percentile")
FRCNNRNetPercentile = merge_predictions_with_stats_filter_wbf([predictionFasterRCNN, predictionRetinaNet], method='percentile', factor=1.0)
FRCNNRNetPercentilePrecision, FRCNNRNetPercentileRecall, FRCNNRNetPercentileTargetBoxes, FRCNNRNetPercentilePredBoxs, FRCNNRNetPercentileMap = evaluate_performance_model(FRCNNRNetPercentile, targetsOutFasterRCNN)

# show_comparison_image_models([FRCNNRNetExtensionWBF, FRCNNRNetPercentile], Coco_dataloader, targetsOutFasterRCNN, ["F-RCNN RNet Extension WBF", "F-RCNN RNet Percentile"], device)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel et Fcos avec filtre de confiance à 0.5 sans NMS")
merge_predictionsFasterRCNNFcosWithoutNMS = merge_predictions_without_nms(predictionFasterRCNN, predictionFcos, threshold=0.001)
FasterRCNNFcosModelCocoPrecision, FasterRCNNFcosModelCocoRecall, FasterRCNNFcosModelCocoTargetBoxes, FasterRCNNFcosModelCocoPredBoxs, FasterRCNNFcosModelCocoMap = evaluate_performance_model(merge_predictionsFasterRCNNFcosWithoutNMS, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel et Fcos avec filtre de confiance à 0.5 avec NMS")
merge_predictionsFasterRCNNFcosWithNMS = merge_predictions_with_nms(predictionFasterRCNN, predictionFcos, threshold=0.001)
FasterRCNNFcosModelCocoPrecisionNMS, FasterRCNNFcosModelCocoRecallNMS, FasterRCNNFcosModelCocoTargetBoxesNMS, FasterRCNNFcosModelCocoPredBoxsNMS, FasterRCNNFcosModelCocoMapNMS = evaluate_performance_model(merge_predictionsFasterRCNNFcosWithNMS, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel et Fcos avec Soft NMS")
merge_predictionsFasterRCNNFcosWithSoftNMS = merge_predictions_with_soft_nms(predictionFasterRCNN, predictionFcos, threshold=0.001)
FasterRCNNFcosModelCocoPrecisionSoftNMS, FasterRCNNFcosModelCocoRecallSoftNMS, FasterRCNNFcosModelCocoTargetBoxesSoftNMS, FasterRCNNFcosModelCocoPredBoxsSoftNMS, FasterRCNNFcosModelCocoMapSoftNMS = evaluate_performance_model(merge_predictionsFasterRCNNFcosWithSoftNMS, targetsOutFasterRCNN)

# print("Evaluation de la combinaisons des modèles FasrerRCNNModel et Fcos avec IoU threshold à 0.5 et threshold évaluation à 0 avec wbf")
# merge_predictionsFasterRCNNFcosWithWBF = merge_predictions_with_wbf(predictionFasterRCNN, predictionFcos, threshold=0.5)
# FasterRCNNFcosModelCocoPrecisionWBFThreshold0, FasterRCNNFcosModelCocoRecallWBFThreshold0, FasterRCNNFcosModelCocoTargetBoxesWBFThreshold0, FasterRCNNFcosModelCocoPredBoxsWBFThreshold0, FasterRCNNFcosModelCocoMapWBFThreshold0 = evaluate_performance_model(merge_predictionsFasterRCNNFcosWithWBF, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel et Fcos extension avec wbf")
FFcosExtensionWBF = merge_predictions_with_extension_wbf([predictionFasterRCNN, predictionFcos], threshold=0.001)
FFcosExtensionWBFPrecision, FFcosExtensionWBFRecall, FFcosExtensionWBFTargetBoxes, FFcosExtensionWBFPredBoxs, FFcosExtensionWBFMap = evaluate_performance_model(FFcosExtensionWBF, targetsOutFasterRCNN)

print("Evaluation filtre statistique moyenne avec facteur de 1")
FFcosMoyenne1 = merge_predictions_with_stats_filter_wbf([predictionFasterRCNN, predictionFcos], method='mean_std', factor=1.0)
FFcosMoyenne1Precision, FFcosMoyenne1Recall, FFcosMoyenne1TargetBoxes, FFcosMoyenne1PredBoxs, FFcosMoyenne1Map = evaluate_performance_model(FFcosMoyenne1, targetsOutFasterRCNN)

print("Evaluation filtre statistique moyenne avec facteur de 2")
FFcosMoyenne2 = merge_predictions_with_stats_filter_wbf([predictionFasterRCNN, predictionFcos], method='mean_std', factor=2.0)
FFcosMoyenne2Precision, FFcosMoyenne2Recall, FFcosMoyenne2TargetBoxes, FFcosMoyenne2PredBoxs, FFcosMoyenne2Map = evaluate_performance_model(FFcosMoyenne2, targetsOutFasterRCNN)

print("Evaluation filtre statistique Mediane")
FFcosMediane = merge_predictions_with_stats_filter_wbf([predictionFasterRCNN, predictionFcos], method='median', factor=1.0)
FFcosMedianePrecision, FFcosMedianeRecall, FFcosMedianeTargetBoxes, FFcosMedianePredBoxs, FFcosMedianeMap = evaluate_performance_model(FFcosMediane, targetsOutFasterRCNN)

print("Evaluation filtre statistique Percentile")
FFcosPercentile = merge_predictions_with_stats_filter_wbf([predictionFasterRCNN, predictionFcos], method='percentile', factor=1.0)
FFcosPercentilePrecision, FFcosPercentileRecall, FFcosPercentileTargetBoxes, FFcosPercentilePredBoxs, FFcosPercentileMap = evaluate_performance_model(FFcosPercentile, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles RetinaNetModel et Fcos avec filtre de confiance à 0.5 sans NMS")
merge_predictionsRetinaNetFcosWithoutNMS = merge_predictions_without_nms(predictionRetinaNet, predictionFcos, threshold=0.001)
RetinaNetFcosModelCocoPrecision, RetinaNetFcosModelCocoRecall, RetinaNetFcosModelCocoTargetBoxes, RetinaNetFcosModelCocoPredBoxs, RetinaNetFcosModelCocoMap = evaluate_performance_model(merge_predictionsRetinaNetFcosWithoutNMS, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles RetinaNetModel et Fcos avec filtre de confiance à 0.5 avec NMS")
merge_predictionsRetinaNetFcosWithNMS = merge_predictions_with_nms(predictionRetinaNet, predictionFcos, threshold=0.001)
RetinaNetFcosModelCocoPrecisionNMS, RetinaNetFcosModelCocoRecallNMS, RetinaNetFcosModelCocoTargetBoxesNMS, RetinaNetFcosModelCocoPredBoxsNMS, RetinaNetFcosModelCocoMapNMS = evaluate_performance_model(merge_predictionsRetinaNetFcosWithNMS, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles RetinaNet et Fcos avec Soft NMS")
merge_predictionsRetinaNetFcosWithSoftNMS = merge_predictions_with_soft_nms(predictionRetinaNet, predictionFcos, threshold=0.001)
RetinaNetFcosModelCocoPrecisionSoftNMS, RetinaNetFcosModelCocoRecallSoftNMS, RetinaNetFcosModelCocoTargetBoxesSoftNMS, RetinaNetFcosModelCocoPredBoxsSoftNMS, RetinaNetFcosModelCocoMapSoftNMS = evaluate_performance_model(merge_predictionsRetinaNetFcosWithSoftNMS, targetsOutFasterRCNN)

# print("Evaluation de la combinaisons des modèles RetinaNetModel et Fcos avec IoU threshold à 0.5 et threshold évaluation à 0 avec wbf")
# merge_predictionsRetinaNetFcosWithWBF = merge_predictions_with_wbf(predictionRetinaNet, predictionFcos, threshold=0.5)
# RetinaNetFcosModelCocoPrecisionWBFThreshold0, RetinaNetFcosModelCocoRecallWBFThreshold0, RetinaNetFcosModelCocoTargetBoxesWBFThreshold0, RetinaNetFcosModelCocoPredBoxsWBFThreshold0, RetinaNetFcosModelCocoMapWBFThreshold0 = evaluate_performance_model(merge_predictionsRetinaNetFcosWithWBF, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles RetinaNet et Fcos avec l'extension wbf")
RetinaNetFcosExtensionWBF = merge_predictions_with_extension_wbf([predictionRetinaNet, predictionFcos], threshold=0.001)
RetinaNetFcosExtensionWBFPrecision, RetinaNetFcosExtensionWBFRecall, RetinaNetFcosExtensionWBFTargetBoxes, RetinaNetFcosExtensionWBFPredBoxs, RetinaNetFcosExtensionWBFMap = evaluate_performance_model(RetinaNetFcosExtensionWBF, targetsOutFasterRCNN)

print("Evaluation filtre statistique moyenne avec facteur de 1")
RetinaNetFcosMoyenne1 = merge_predictions_with_stats_filter_wbf([predictionRetinaNet, predictionFcos], method='mean_std', factor=1.0)
RetinaNetFcosMoyenne1Precision, RetinaNetFcosMoyenne1Recall, RetinaNetFcosMoyenne1TargetBoxes, RetinaNetFcosMoyenne1PredBoxs, RetinaNetFcosMoyenne1Map = evaluate_performance_model(RetinaNetFcosMoyenne1, targetsOutFasterRCNN)

print("Evaluation filtre statistique moyenne avec facteur de 2")
RetinaNetFcosMoyenne2 = merge_predictions_with_stats_filter_wbf([predictionRetinaNet, predictionFcos], method='mean_std', factor=2.0)
RetinaNetFcosMoyenne2Precision, RetinaNetFcosMoyenne2Recall, RetinaNetFcosMoyenne2TargetBoxes, RetinaNetFcosMoyenne2PredBoxs, RetinaNetFcosMoyenne2Map = evaluate_performance_model(RetinaNetFcosMoyenne2, targetsOutFasterRCNN)

print("Evaluation filtre statistique Mediane")
RetinaNetFcosMediane = merge_predictions_with_stats_filter_wbf([predictionRetinaNet, predictionFcos], method='median', factor=1.0)
RetinaNetFcosMedianePrecision, RetinaNetFcosMedianeRecall, RetinaNetFcosMedianeTargetBoxes, RetinaNetFcosMedianePredBoxs, RetinaNetFcosMedianeMap = evaluate_performance_model(RetinaNetFcosMediane, targetsOutFasterRCNN)

print("Evaluation filtre statistique Percentile")
RetinaNetFcosPercentile = merge_predictions_with_stats_filter_wbf([predictionRetinaNet, predictionFcos], method='percentile', factor=1.0)
RetinaNetFcosPercentilePrecision, RetinaNetFcosPercentileRecall, RetinaNetFcosPercentileTargetBoxes, RetinaNetFcosPercentilePredBoxs, RetinaNetFcosPercentileMap = evaluate_performance_model(RetinaNetFcosPercentile, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel, RetinaNet et Fcos avec filtre de confiance à 0.5 sans NMS")
merge_predictionsFasterRCNNRetinaNetFcosWithoutNMS = merge_predictions_without_nms(merge_predictionsFasterRCNNRetinaNetWithoutNMS, predictionFcos, threshold=0.001)
FasterRCNNRetinaNetFcosModelCocoPrecision, FasterRCNNRetinaNetFcosModelCocoRecall, FasterRCNNRetinaNetFcosModelCocoTargetBoxes, FasterRCNNRetinaNetFcosModelCocoPredBoxs, FasterRCNNRetinaNetFcosModelCocoMap = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetFcosWithoutNMS, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel, RetinaNet et Fcos avec filtre de confiance à 0.5 avec NMS")
merge_predictionsFasterRCNNRetinaNetFcosWithNMS = merge_predictions_with_nms(merge_predictionsFasterRCNNRetinaNetWithNMS, predictionFcos, threshold=0.001)
FasterRCNNRetinaNetFcosModelCocoPrecisionNMS, FasterRCNNRetinaNetFcosModelCocoRecallNMS, FasterRCNNRetinaNetFcosModelCocoTargetBoxesNMS, FasterRCNNRetinaNetFcosModelCocoPredBoxsNMS, FasterRCNNRetinaNetFcosModelCocoMapNMS = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetFcosWithNMS, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel, RetinaNet et Fcos avec Soft NMS")
merge_predictionsFasterRCNNRetinaNetFcosWithSoftNMS = merge_predictions_with_soft_nms(merge_predictionsFasterRCNNRetinaNetWithSoftNMS, predictionFcos, threshold=0.001)
FasterRCNNRetinaNetFcosModelCocoPrecisionSoftNMS, FasterRCNNRetinaNetFcosModelCocoRecallSoftNMS, FasterRCNNRetinaNetFcosModelCocoTargetBoxesSoftNMS, FasterRCNNRetinaNetFcosModelCocoPredBoxsSoftNMS, FasterRCNNRetinaNetFcosModelCocoMapSoftNMS = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetFcosWithSoftNMS, targetsOutFasterRCNN)

# print("Evaluation de la combinaisons des modèles FasrerRCNNModel, RetinaNet et Fcos avec IoU threshold à 0.5 et threshold évaluation à 0 avec wbf")
# merge_predictionsFasterRCNNRetinaNetFcosWithWBF = merge_predictions_with_wbf(merge_predictionsFasterRCNNRetinaNetWithWBF, predictionFcos, threshold=0.5)
# FasterRCNNRetinaNetFcosModelCocoPrecisionWBFThreshold0, FasterRCNNRetinaNetFcosModelCocoRecallWBFThreshold0, FasterRCNNRetinaNetFcosModelCocoTargetBoxesWBFThreshold0, FasterRCNNRetinaNetFcosModelCocoPredBoxsWBFThreshold0, FasterRCNNRetinaNetFcosModelCocoMapWBFThreshold0 = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetFcosWithWBF, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel, RetinaNet et Fcos avec l'extension wbf")
FRCNNRNetFcosExtensionWBF = merge_predictions_with_extension_wbf([predictionFasterRCNN, predictionRetinaNet, predictionFcos], threshold=0.001)
FRCNNRNetFcosExtensionWBFPrecision, FRCNNRNetFcosExtensionWBFRecall, FRCNNRNetFcosExtensionWBFTargetBoxes, FRCNNRNetFcosExtensionWBFPredBoxs, FRCNNRNetFcosExtensionWBFMap = evaluate_performance_model(FRCNNRNetFcosExtensionWBF, targetsOutFasterRCNN)

print("Evaluation filtre statistique moyenne avec facteur de 1")
FRCNNRNetFcosMoyenne1 = merge_predictions_with_stats_filter_wbf([predictionFasterRCNN, predictionRetinaNet, predictionFcos], method='mean_std', factor=1.0)
FRCNNRNetFcosMoyenne1Precision, FRCNNRNetFcosMoyenne1Recall, FRCNNRNetFcosMoyenne1TargetBoxes, FRCNNRNetFcosMoyenne1PredBoxs, FRCNNRNetFcosMoyenne1Map = evaluate_performance_model(FRCNNRNetFcosMoyenne1, targetsOutFasterRCNN)

print("Evaluation filtre statistique moyenne avec facteur de 2")
FRCNNRNetFcosMoyenne2 = merge_predictions_with_stats_filter_wbf([predictionFasterRCNN, predictionRetinaNet, predictionFcos], method='mean_std', factor=2.0)
FRCNNRNetFcosMoyenne2Precision, FRCNNRNetFcosMoyenne2Recall, FRCNNRNetFcosMoyenne2TargetBoxes, FRCNNRNetFcosMoyenne2PredBoxs, FRCNNRNetFcosMoyenne2Map = evaluate_performance_model(FRCNNRNetFcosMoyenne2, targetsOutFasterRCNN)

print("Evaluation filtre statistique Mediane")
FRCNNRNetFcosMediane = merge_predictions_with_stats_filter_wbf([predictionFasterRCNN, predictionRetinaNet, predictionFcos], method='median', factor=1.0)
FRCNNRNetFcosMedianePrecision, FRCNNRNetFcosMedianeRecall, FRCNNRNetFcosMedianeTargetBoxes, FRCNNRNetFcosMedianePredBoxs, FRCNNRNetFcosMedianeMap = evaluate_performance_model(FRCNNRNetFcosMediane, targetsOutFasterRCNN)

print("Evaluation filtre statistique Percentile")
FRCNNRNetFcosPercentile = merge_predictions_with_stats_filter_wbf([predictionFasterRCNN, predictionRetinaNet, predictionFcos], method='percentile', factor=1.0)
FRCNNRNetFcosPercentilePrecision, FRCNNRNetFcosPercentileRecall, FRCNNRNetFcosPercentileTargetBoxes, FRCNNRNetFcosPercentilePredBoxs, FRCNNRNetFcosPercentileMap = evaluate_performance_model(FRCNNRNetFcosPercentile, targetsOutFasterRCNN)


percentileFactor = [0,0.1,0.15,0.2,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0]
results = evaluate_hyperparameters([predictionFasterRCNN, predictionRetinaNet, predictionFcos], IoU_threshold=0.5, methods=['percentile'], reduction_factors=[0.5], percentileFactors=percentileFactor)
plot_results_percentile(results, dataset='COCO')


reduction_factors = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
results = evaluate_hyperparameters([predictionFasterRCNN, predictionRetinaNet, predictionFcos], IoU_threshold=0.5, methods=['percentile'], reduction_factors=reduction_factors, percentileFactors=[0.75])
plot_results_reduction(results, dataset='COCO')

percentileFactor = [0,0.1,0.15,0.2,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0]
reduction_factors = [0]
results = evaluate_hyperparameters([predictionFasterRCNN, predictionRetinaNet, predictionFcos], IoU_threshold=0.5, methods=['percentile'], reduction_factors=reduction_factors, percentileFactors=percentileFactor)
plot_results_percentile(results, dataset='COCO')


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
# with open("Resultats/performance_model_mAPCocoFullData.tex", "w") as f:
#     f.write("\\documentclass{article}\n")
#     f.write("\\usepackage{graphicx} % Required for inserting images\n")
#     f.write("\\begin{document}\n")
#     f.write("\\begin{table}[h!]\n")
#     f.write("\\centering\n")
#     f.write("\\begin{tabular}{|c||c|c|c|} \n")
#     f.write("\\hline\n")
#     f.write("Model & precision & recall & mAP \\\\ \n")
#     f.write("\\hline\n")
#     f.write("F R-CNN & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNModelCocoPrecision, FasterRCNNModelCocoRecall, FasterRCNNModelCocoMap))
#     f.write("\\hline\n")
#     f.write("RetinaNet & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RetinaNetModelCocoPrecision, RetinaNetModelCocoRecall, RetinaNetModelCocoMap))
#     f.write("\\hline\n")
#     f.write("FCOS & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FcosModelCocoPrecision, FcosModelCocoRecall, FcosModelCocoMap))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RetinaNet Cat & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetModelCocoPrecision, FasterRCNNRetinaNetModelCocoRecall, FasterRCNNRetinaNetModelCocoMap))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RetinaNet NMS & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetModelCocoPrecisionNMS, FasterRCNNRetinaNetModelCocoRecallNMS, FasterRCNNRetinaNetModelCocoMapNMS))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RetinaNet Soft NMS & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetModelCocoPrecisionSoftNMS, FasterRCNNRetinaNetModelCocoRecallSoftNMS, FasterRCNNRetinaNetModelCocoMapSoftNMS))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RetinaNet WBF & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetModelCocoPrecisionWBFThreshold0, FasterRCNNRetinaNetModelCocoRecallWBFThreshold0, FasterRCNNRetinaNetModelCocoMapWBFThreshold0))
#     f.write("\\hline\n")
#     f.write("F RCNN, RetinaNet Extension WBF & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetExtensionWBFPrecision, FRCNNRNetExtensionWBFRecall, FRCNNRNetExtensionWBFMap))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RNet Filtre Moyenne 1 & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetMoyenne1Precision, FRCNNRNetMoyenne1Recall, FRCNNRNetMoyenne1Map))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RNet Filtre Moyenne 2 & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetMoyenne2Precision, FRCNNRNetMoyenne2Recall, FRCNNRNetMoyenne2Map))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RNet Filtre Mediane & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetMedianePrecision, FRCNNRNetMedianeRecall, FRCNNRNetMedianeMap))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RNet Filtre Percentile & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetPercentilePrecision, FRCNNRNetPercentileRecall, FRCNNRNetPercentileMap))
#     f.write("\\hline\n")
#     f.write("F R-CNN, FCOS Cat & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNFcosModelCocoPrecision, FasterRCNNFcosModelCocoRecall, FasterRCNNFcosModelCocoMap))
#     f.write("\\hline\n")
#     f.write("F R-CNN, FCOS NMS & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNFcosModelCocoPrecisionNMS, FasterRCNNFcosModelCocoRecallNMS, FasterRCNNFcosModelCocoMapNMS))
#     f.write("\\hline\n")
#     f.write("F R-CNN, FCOS Soft NMS & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNFcosModelCocoPrecisionSoftNMS, FasterRCNNFcosModelCocoRecallSoftNMS, FasterRCNNFcosModelCocoMapSoftNMS))
#     f.write("\\hline\n")
#     f.write("F R-CNN, FCOS WBF & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNFcosModelCocoPrecisionWBFThreshold0, FasterRCNNFcosModelCocoRecallWBFThreshold0, FasterRCNNFcosModelCocoMapWBFThreshold0))
#     f.write("\\hline\n")
#     f.write("F R-CNN, FCOS Extension WBF & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FFcosExtensionWBFPrecision, FFcosExtensionWBFRecall, FFcosExtensionWBFMap))
#     f.write("\\hline\n")
#     f.write("F R-CNN, FCOS Filtre Moyenne 1 & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FFcosMoyenne1Precision, FFcosMoyenne1Recall, FFcosMoyenne1Map))
#     f.write("\\hline\n")
#     f.write("F R-CNN, FCOS Filtre Moyenne 2 & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FFcosMoyenne2Precision, FFcosMoyenne2Recall, FFcosMoyenne2Map))
#     f.write("\\hline\n")
#     f.write("F R-CNN, FCOS Filtre Mediane & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FFcosMedianePrecision, FFcosMedianeRecall, FFcosMedianeMap))
#     f.write("\\hline\n")
#     f.write("F R-CNN, FCOS Filtre Percentile & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FFcosPercentilePrecision, FFcosPercentileRecall, FFcosPercentileMap))
#     f.write("\\hline\n")
#     f.write("RetinaNet, FCOS Cat & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RetinaNetFcosModelCocoPrecision, RetinaNetFcosModelCocoRecall, RetinaNetFcosModelCocoMap))
#     f.write("\\hline\n")
#     f.write("RetinaNet, FCOS NMS & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RetinaNetFcosModelCocoPrecisionNMS, RetinaNetFcosModelCocoRecallNMS, RetinaNetFcosModelCocoMapNMS))
#     f.write("\\hline\n")
#     f.write("RetinaNet, FCOS Soft NMS & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RetinaNetFcosModelCocoPrecisionSoftNMS, RetinaNetFcosModelCocoRecallSoftNMS, RetinaNetFcosModelCocoMapSoftNMS))
#     f.write("\\hline\n")
#     f.write("RetinaNet, FCOS WBF & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RetinaNetFcosModelCocoPrecisionWBFThreshold0, RetinaNetFcosModelCocoRecallWBFThreshold0, RetinaNetFcosModelCocoMapWBFThreshold0))
#     f.write("\\hline\n")
#     f.write("RetinaNet, FCOS Extension WBF & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RetinaNetFcosExtensionWBFPrecision, RetinaNetFcosExtensionWBFRecall, RetinaNetFcosExtensionWBFMap))
#     f.write("\\hline\n")
#     f.write("RetinaNet, FCOS Filtre Moyenne 1 & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RetinaNetFcosMoyenne1Precision, RetinaNetFcosMoyenne1Recall, RetinaNetFcosMoyenne1Map))
#     f.write("\\hline\n")
#     f.write("RetinaNet, FCOS Filtre Moyenne 2 & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RetinaNetFcosMoyenne2Precision, RetinaNetFcosMoyenne2Recall, RetinaNetFcosMoyenne2Map))
#     f.write("\\hline\n")
#     f.write("RetinaNet, FCOS Filtre Mediane & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RetinaNetFcosMedianePrecision, RetinaNetFcosMedianeRecall, RetinaNetFcosMedianeMap))
#     f.write("\\hline\n")
#     f.write("RetinaNet, FCOS Filtre Percentile & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RetinaNetFcosPercentilePrecision, RetinaNetFcosPercentileRecall, RetinaNetFcosPercentileMap))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RetinaNet, FCOS Cat & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetFcosModelCocoPrecision, FasterRCNNRetinaNetFcosModelCocoRecall, FasterRCNNRetinaNetFcosModelCocoMap))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RetinaNet, FCOS NMS & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetFcosModelCocoPrecisionNMS, FasterRCNNRetinaNetFcosModelCocoRecallNMS, FasterRCNNRetinaNetFcosModelCocoMapNMS))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RetinaNet, FCOS Soft NMS & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetFcosModelCocoPrecisionSoftNMS, FasterRCNNRetinaNetFcosModelCocoRecallSoftNMS, FasterRCNNRetinaNetFcosModelCocoMapSoftNMS))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RetinaNet, FCOS WBF & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetFcosModelCocoPrecisionWBFThreshold0, FasterRCNNRetinaNetFcosModelCocoRecallWBFThreshold0, FasterRCNNRetinaNetFcosModelCocoMapWBFThreshold0))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RetinaNet, FCOS Extension WBF & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetFcosExtensionWBFPrecision, FRCNNRNetFcosExtensionWBFRecall, FRCNNRNetFcosExtensionWBFMap))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RetinaNet, FCOS Filtre Moyenne 1 & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetFcosMoyenne1Precision, FRCNNRNetFcosMoyenne1Recall, FRCNNRNetFcosMoyenne1Map))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RetinaNet, FCOS Filtre Moyenne 2 & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetFcosMoyenne2Precision, FRCNNRNetFcosMoyenne2Recall, FRCNNRNetFcosMoyenne2Map))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RetinaNet, FCOS Filtre Mediane & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetFcosMedianePrecision, FRCNNRNetFcosMedianeRecall, FRCNNRNetFcosMedianeMap))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RetinaNet, FCOS Filtre Percentile & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetFcosPercentilePrecision, FRCNNRNetFcosPercentileRecall, FRCNNRNetFcosPercentileMap))
#     f.write("\\hline\n")
#     f.write("\\end{tabular}\n")
#     f.write("\\caption{Evaluation des perfomances avec le filtre stats}\n")
#     f.write("\\label{table:data}\n")
#     f.write("\\end{table}\n")
#     f.write("\\end{document}\n")
#     f.close()
#     pass


# Affichage stats bdd100k et COCO

with open("Resultats/performance_model_mAPStatsTest.tex", "w") as f:
    f.write("\\documentclass{article}\n")
    f.write("\\usepackage{graphicx} % Required for inserting images\n")
    f.write("\\begin{document}\n")
    f.write("\\begin{table}[h!]\n")
    f.write("\\centering\n")
    f.write("\\begin{tabular}{|c||c|c|c||c|c|c|} \n")
    f.write("\\hline\n")
    f.write("Model & \\multicolumn{3}{|c||}{BDD100K} & \\multicolumn{3}{|c|}{COCO} \\\\ \n")
    f.write(" & precision & recall & mAP  & precision & recall & mAP  \\\\ [0.5ex] \n")
    f.write("\\hline\n")
    f.write("F R-CNN & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNModelBdd100KPrecision, FasterRCNNModelBdd100KRecall, FasterRCNNModelBdd100KMap, FasterRCNNModelCocoPrecision, FasterRCNNModelCocoRecall, FasterRCNNModelCocoMap))
    f.write("\\hline\n")
    f.write("RetinaNet & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RetinaNetModelBdd100KPrecision, RetinaNetModelBdd100KRecall, RetinaNetModelBdd100KMap, RetinaNetModelCocoPrecision, RetinaNetModelCocoRecall, RetinaNetModelCocoMap))
    f.write("\\hline\n")
    f.write("FCOS & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FcosModelBdd100KPrecision, FcosModelBdd100KRecall, FcosModelBdd100KMap, FcosModelCocoPrecision, FcosModelCocoRecall, FcosModelCocoMap))
    f.write("\\hline\n")
    f.write("F R-CNN, RetinaNet WBF & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetBDD100KExtensionWBFPrecision, FRCNNRNetBDD100KExtensionWBFRecall, FRCNNRNetBDD100KExtensionWBFMap, FRCNNRNetExtensionWBFPrecision, FRCNNRNetExtensionWBFRecall, FRCNNRNetExtensionWBFMap))
    f.write("\\hline\n")
    f.write("F R-CNN, RetinaNet Filtre Moyenne 1 & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetBDD100KMoyenne1Precision, FRCNNRNetBDD100KMoyenne1Recall, FRCNNRNetBDD100KMoyenne1Map, FRCNNRNetMoyenne1Precision, FRCNNRNetMoyenne1Recall, FRCNNRNetMoyenne1Map))
    f.write("\\hline\n")
    f.write("F R-CNN, RetinaNet Filtre Moyenne 2 & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetBDD100KMoyenne2Precision, FRCNNRNetBDD100KMoyenne2Recall, FRCNNRNetBDD100KMoyenne2Map, FRCNNRNetMoyenne2Precision, FRCNNRNetMoyenne2Recall, FRCNNRNetMoyenne2Map))
    f.write("\\hline\n")
    f.write("F R-CNN, RetinaNet Filtre Mediane & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetBDD100KMedianePrecision, FRCNNRNetBDD100KMedianeRecall, FRCNNRNetBDD100KMedianeMap, FRCNNRNetMedianePrecision, FRCNNRNetMedianeRecall, FRCNNRNetMedianeMap))
    f.write("\\hline\n")
    f.write("F R-CNN, RetinaNet Filtre Percentile & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetBDD100KPercentilePrecision, FRCNNRNetBDD100KPercentileRecall, FRCNNRNetBDD100KPercentileMap, FRCNNRNetPercentilePrecision, FRCNNRNetPercentileRecall, FRCNNRNetPercentileMap))
    f.write("\\hline\n")
    f.write("F R-CNN, FCOS WBF & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FFcosExtensionWBFBDD100KPrecision, FFcosExtensionWBFBDD100KRecall, FFcosExtensionWBFBDD100KMap, FFcosExtensionWBFPrecision, FFcosExtensionWBFRecall, FFcosExtensionWBFMap))
    f.write("\\hline\n")
    f.write("F R-CNN, FCOS Filtre Moyenne 1 & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FFcosMoyenne1BDD100KPrecision, FFcosMoyenne1BDD100KRecall, FFcosMoyenne1BDD100KMap, FFcosMoyenne1BDD100KPrecision, FFcosMoyenne1Recall, FFcosMoyenne1Map))
    f.write("\\hline\n")
    f.write("F R-CNN, FCOS Filtre Moyenne 2 & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FFcosMoyenne2BDD100KPrecision, FFcosMoyenne2BDD100KRecall, FFcosMoyenne2BDD100KMap, FFcosMoyenne2Precision, FFcosMoyenne2Recall, FFcosMoyenne2Map))
    f.write("\\hline\n")
    f.write("F R-CNN, FCOS Filtre Mediane & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FFcosMedianeBDD100KPrecision, FFcosMedianeBDD100KRecall, FFcosMedianeBDD100KMap, FFcosMedianePrecision, FFcosMedianeRecall, FFcosMedianeMap))
    f.write("\\hline\n")
    f.write("F R-CNN, FCOS Filtre Percentile & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FFcosPercentileBDD100KPrecision, FFcosPercentileBDD100KRecall, FFcosPercentileBDD100KMap, FFcosPercentilePrecision, FFcosPercentileRecall, FFcosPercentileMap))
    f.write("\\hline\n")
    f.write("RetinaNet, FCOS WBF & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RNetFcosBDD100KExtensionWBFPrecision, RNetFcosBDD100KExtensionWBFRecall, RNetFcosBDD100KExtensionWBFMap, RetinaNetFcosExtensionWBFPrecision, RetinaNetFcosExtensionWBFRecall, RetinaNetFcosExtensionWBFMap))
    f.write("\\hline\n")
    f.write("RetinaNet, FCOS Filtre Moyenne 1 & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RNetFcosBDD100KMoyenne1Precision, RNetFcosBDD100KMoyenne1Recall, RNetFcosBDD100KMoyenne1Map, RetinaNetFcosMoyenne1Precision, RetinaNetFcosMoyenne1Recall, RetinaNetFcosMoyenne1Map))
    f.write("\\hline\n")
    f.write("RetinaNet, FCOS Filtre Moyenne 2 & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RNetFcosBDD100KMoyenne2Precision, RNetFcosBDD100KMoyenne2Recall, RNetFcosBDD100KMoyenne2Map, RetinaNetFcosMoyenne2Precision, RetinaNetFcosMoyenne2Recall, RetinaNetFcosMoyenne2Map))
    f.write("\\hline\n")
    f.write("RetinaNet, FCOS Filtre Mediane & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RNetFcosBDD100KMedianePrecision, RNetFcosBDD100KMedianeRecall, RNetFcosBDD100KMedianeMap, RetinaNetFcosMedianePrecision, RetinaNetFcosMedianeRecall, RetinaNetFcosMedianeMap))
    f.write("\\hline\n")
    f.write("RetinaNet, FCOS Filtre Percentile & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RNetFcosBDD100KPercentilePrecision, RNetFcosBDD100KPercentileRecall, RNetFcosBDD100KPercentileMap, RetinaNetFcosPercentilePrecision, RetinaNetFcosPercentileRecall, RetinaNetFcosPercentileMap))
    f.write("\\hline\n")
    f.write("F R-CNN, RetinaNet, FCOS WBF & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetFcosBDD100KExtensionWBFPrecision, FRCNNRNetFcosBDD100KExtensionWBFRecall, FRCNNRNetFcosBDD100KExtensionWBFMap, FRCNNRNetFcosExtensionWBFPrecision, FRCNNRNetFcosExtensionWBFRecall, FRCNNRNetFcosExtensionWBFMap))
    f.write("\\hline\n")
    f.write("F R-CNN, RetinaNet, FCOS Filtre Moyenne 1 & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetFcosBDD100KMoyenne1Precision, FRCNNRNetFcosBDD100KMoyenne1Recall, FRCNNRNetFcosBDD100KMoyenne1Map, FRCNNRNetFcosMoyenne1Precision, FRCNNRNetFcosMoyenne1Recall, FRCNNRNetFcosMoyenne1Map))
    f.write("\\hline\n")
    f.write("F R-CNN, RetinaNet, FCOS Filtre Moyenne 2 & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetFcosBDD100KMoyenne2Precision, FRCNNRNetFcosBDD100KMoyenne2Recall, FRCNNRNetFcosBDD100KMoyenne2Map, FRCNNRNetFcosMoyenne2Precision, FRCNNRNetFcosMoyenne2Recall, FRCNNRNetFcosMoyenne2Map))
    f.write("\\hline\n")
    f.write("F R-CNN, RetinaNet, FCOS Filtre Mediane & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetFcosBDD100KMedianePrecision, FRCNNRNetFcosBDD100KMedianeRecall, FRCNNRNetFcosBDD100KMedianeMap, FRCNNRNetFcosMedianePrecision, FRCNNRNetFcosMedianeRecall, FRCNNRNetFcosMedianeMap))
    f.write("\\hline\n")
    f.write("F R-CNN, RetinaNet, FCOS Filtre Percentile & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetFcosBDD100KPercentilePrecision, FRCNNRNetFcosBDD100KPercentileRecall, FRCNNRNetFcosBDD100KPercentileMap, FRCNNRNetFcosPercentilePrecision, FRCNNRNetFcosPercentileRecall, FRCNNRNetFcosPercentileMap))
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\caption{Evaluation des perfomances avec le filtre stats}\n")
    f.write("\\label{table:data}\n")
    f.write("\\end{table}\n")
    f.write("\\end{document}\n")
    f.close()
    pass

# Affichage stats bdd100k et COCO

# with open("Resultats/performance_model_mAP_Percentile.tex", "w") as f:
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
#     f.write("F R-CNN, RetinaNet WBF & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetBDD100KExtensionWBFPrecision, FRCNNRNetBDD100KExtensionWBFRecall, FRCNNRNetBDD100KExtensionWBFMap, FRCNNRNetExtensionWBFPrecision, FRCNNRNetExtensionWBFRecall, FRCNNRNetExtensionWBFMap))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RetinaNet Filtre Percentile & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetBDD100KPercentilePrecision, FRCNNRNetBDD100KPercentileRecall, FRCNNRNetBDD100KPercentileMap, FRCNNRNetPercentilePrecision, FRCNNRNetPercentileRecall, FRCNNRNetPercentileMap))
#     f.write("\\hline\n")
#     f.write("F R-CNN, FCOS WBF & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FFcosExtensionWBFBDD100KPrecision, FFcosExtensionWBFBDD100KRecall, FFcosExtensionWBFBDD100KMap, FFcosExtensionWBFPrecision, FFcosExtensionWBFRecall, FFcosExtensionWBFMap))
#     f.write("\\hline\n")
#     f.write("F R-CNN, FCOS Filtre Percentile & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FFcosPercentileBDD100KPrecision, FFcosPercentileBDD100KRecall, FFcosPercentileBDD100KMap, FFcosPercentilePrecision, FFcosPercentileRecall, FFcosPercentileMap))
#     f.write("\\hline\n")
#     f.write("RetinaNet, FCOS WBF & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RNetFcosBDD100KExtensionWBFPrecision, RNetFcosBDD100KExtensionWBFRecall, RNetFcosBDD100KExtensionWBFMap, RetinaNetFcosExtensionWBFPrecision, RetinaNetFcosExtensionWBFRecall, RetinaNetFcosExtensionWBFMap))
#     f.write("\\hline\n")
#     f.write("RetinaNet, FCOS Filtre Percentile & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RNetFcosBDD100KPercentilePrecision, RNetFcosBDD100KPercentileRecall, RNetFcosBDD100KPercentileMap, RetinaNetFcosPercentilePrecision, RetinaNetFcosPercentileRecall, RetinaNetFcosPercentileMap))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RetinaNet, FCOS WBF & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetFcosBDD100KExtensionWBFPrecision, FRCNNRNetFcosBDD100KExtensionWBFRecall, FRCNNRNetFcosBDD100KExtensionWBFMap, FRCNNRNetFcosExtensionWBFPrecision, FRCNNRNetFcosExtensionWBFRecall, FRCNNRNetFcosExtensionWBFMap))
#     f.write("\\hline\n")
#     f.write("F R-CNN, RetinaNet, FCOS Filtre Percentile & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FRCNNRNetFcosBDD100KPercentilePrecision, FRCNNRNetFcosBDD100KPercentileRecall, FRCNNRNetFcosBDD100KPercentileMap, FRCNNRNetFcosPercentilePrecision, FRCNNRNetFcosPercentileRecall, FRCNNRNetFcosPercentileMap))
#     f.write("\\hline\n")
#     f.write("\\end{tabular}\n")
#     f.write("\\caption{Evaluation des perfomances avec le filtre stats}\n")
#     f.write("\\label{table:data}\n")
#     f.write("\\end{table}\n")
#     f.write("\\end{document}\n")
#     f.close()
#     pass

print("Fin de l'execution")