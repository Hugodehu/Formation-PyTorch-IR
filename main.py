import copy
from matplotlib import pyplot as plt
import torch
from torchvision import models
from function.get_list_fp_fn_map_images import getListFPFNMapImages
from function.get_prediction_model import getPredictionModel
from function.merge_prediction.merge_predictions_with_nms import merge_predictions_with_nms 
from function.merge_prediction.merge_predictions_without_nms import merge_predictions_without_nms
from function.merge_prediction.merge_predictions_with_wbf import merge_predictions_with_wbf
from function.merge_prediction.merge_predictions_with_extension_wbf import merge_predictions_with_extension_wbf
from function.merge_prediction.merge_predictions_with_soft_nms import merge_predictions_with_soft_nms
from function.merge_prediction.merge_predictions_with_stats_filter_wbf import merge_predictions_with_stats_filter_wbf
from function.visualize_prediction_image import show_comparison_image_models_for_map_inferior
from function.init_dataloader import initialiseDataloader

BDD100K_dataloader, Coco_dataloader = initialiseDataloader(isSubset=True, subsetSize=10)

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
predictionFasterRCNN, BaseOutput = getPredictionModel(FasterRCNNModel, BDD100K_dataloader, device)

print("Evaluation du modèle RetinaNetModel sans modification")
predictionRetinaNet, targetsOutRetinaNet = getPredictionModel(RetinaNetModel, BDD100K_dataloader, device)

print("Evaluation du modèle FcosModel sans modification")
predictionFcos, targetsOutFcos = getPredictionModel(FcosModel, BDD100K_dataloader, device)

predictionFasterRCNN50 = copy.deepcopy(predictionFasterRCNN)
predictionRetinaNet50 = copy.deepcopy(predictionRetinaNet)
predictionFcos50 = copy.deepcopy(predictionFcos)

for listlist in predictionFasterRCNN50:
    for prediction in listlist:
        scores =prediction['scores']
        prediction['boxes'] = prediction['boxes'][scores >= 0.5]
        prediction['labels'] = prediction['labels'][scores >= 0.5]
        prediction['scores'] = scores[scores >= 0.5]

for listlist in predictionRetinaNet50:
    for prediction in listlist:
        scores =prediction['scores']
        prediction['boxes'] = prediction['boxes'][scores >= 0.5]
        prediction['labels'] = prediction['labels'][scores >= 0.5]
        prediction['scores'] = scores[scores >= 0.5]

for listlist in predictionFcos50:
    for prediction in listlist:
        scores =prediction['scores']
        prediction['boxes'] = prediction['boxes'][scores >= 0.5]
        prediction['labels'] = prediction['labels'][scores >= 0.5]
        prediction['scores'] = scores[scores >= 0.5]

print("Evaluation FRCNN Rnet extension WBF")
FRCNNRNetExtensionWBFBDD100K = merge_predictions_with_extension_wbf([predictionFasterRCNN, predictionRetinaNet, predictionFcos], threshold=0.001)

FasterRCNNModelBdd100KMap, predictionFasterRCNNFilter, targetFasterRCNN  = getListFPFNMapImages(predictionFasterRCNN, BaseOutput)
RetinaNetModelBdd100KMap, predictionRetinaNetFilter, targetRetinaNet = getListFPFNMapImages(predictionRetinaNet, BaseOutput)
FcosModelBdd100KMap, predictionFcosFilter, targetFcos = getListFPFNMapImages(predictionFcos, BaseOutput)
FRCNNRNetBDD100KExtensionWBFMap, FRCNNRNetExtensionWBFBDD100K, targetExtensionWBFBDD100K = getListFPFNMapImages(FRCNNRNetExtensionWBFBDD100K, BaseOutput)

listIndexImagesWithMapInferiorForWBF = []

for idx, map in enumerate(FRCNNRNetBDD100KExtensionWBFMap):
    heighestMapInModelAlone = max(FasterRCNNModelBdd100KMap[idx]['map'], RetinaNetModelBdd100KMap[idx]['map'], FcosModelBdd100KMap[idx]['map'])
    if map['map'] < heighestMapInModelAlone:
        listIndexImagesWithMapInferiorForWBF.append(idx)

if(len(listIndexImagesWithMapInferiorForWBF) != 0):
    # listIndexImagesWithMapInferiorForWBF = [listIndexImagesWithMapInferiorForWBF[3]]
    predictionFasterRCNN = [predictionFasterRCNN[i] for i in listIndexImagesWithMapInferiorForWBF]
    predictionRetinaNet = [predictionRetinaNet[i] for i in listIndexImagesWithMapInferiorForWBF]
    predictionFcos = [predictionFcos[i] for i in listIndexImagesWithMapInferiorForWBF]
    predictionFasterRCNN50 = [predictionFasterRCNN50[i] for i in listIndexImagesWithMapInferiorForWBF]
    predictionRetinaNet50 = [predictionRetinaNet50[i] for i in listIndexImagesWithMapInferiorForWBF]
    predictionFcos50 = [predictionFcos50[i] for i in listIndexImagesWithMapInferiorForWBF]
    BaseOutput = [BaseOutput[i] for i in listIndexImagesWithMapInferiorForWBF]
    targetsOutRetinaNet = [targetsOutRetinaNet[i] for i in listIndexImagesWithMapInferiorForWBF]
    targetsOutFcos = [targetsOutFcos[i] for i in listIndexImagesWithMapInferiorForWBF]

    FasterRCNNModelBdd100KMap, predictionFasterRCNNFilter, targetFasterRCNN  = getListFPFNMapImages(predictionFasterRCNN, BaseOutput)
    RetinaNetModelBdd100KMap, predictionRetinaNetFilter, targetRetinaNet = getListFPFNMapImages(predictionRetinaNet, BaseOutput)
    FcosModelBdd100KMap, predictionFcosFilter, targetFcos = getListFPFNMapImages(predictionFcos, BaseOutput)

    show_comparison_image_models_for_map_inferior([predictionFasterRCNN, predictionRetinaNet, predictionFcos], BDD100K_dataloader, [BaseOutput, targetsOutRetinaNet, targetsOutFcos], ["FasterRCNNModel", "RetinaNetModel", "FcosModel"], device, threshold=0.001, listIndexImagesWithMapInferiorForWBF=listIndexImagesWithMapInferiorForWBF)
    show_comparison_image_models_for_map_inferior([predictionFasterRCNN50, predictionRetinaNet50, predictionFcos50], BDD100K_dataloader, [BaseOutput, targetsOutRetinaNet, targetsOutFcos], ["FasterRCNNModel", "RetinaNetModel", "FcosModel"], device, threshold=0.001, listIndexImagesWithMapInferiorForWBF=listIndexImagesWithMapInferiorForWBF)

    print("Evaluation de la combinaisons des modèles FasrerRCNNModel et RetinaNet avec filtre de confiance à 0.5 sans NMS")
    merge_predictionsFasterRCNNRetinaNetWithoutNMS = merge_predictions_without_nms(predictionFasterRCNN, predictionRetinaNet, threshold=0.001)
    _, merge_predictionsFasterRCNNRetinaNetWithoutNMS, targetFRCNNRetinaNetWithoutNMS = getListFPFNMapImages(merge_predictionsFasterRCNNRetinaNetWithoutNMS, BaseOutput)
    merge_predictionsFasterRCNNRetinaNetFCOSWithoutNMS = merge_predictions_without_nms(merge_predictionsFasterRCNNRetinaNetWithoutNMS, predictionFcos, threshold=0.001)
    _, merge_predictionsFasterRCNNRetinaNetFCOSWithoutNMS, targetFRCNNRetinaNetFCOSWithoutNms = getListFPFNMapImages(merge_predictionsFasterRCNNRetinaNetFCOSWithoutNMS, BaseOutput)

    print("Evaluation de la combinaisons des modèles FasrerRCNNModel et RetinaNet avec filtre de confiance à 0.5 avec NMS")
    merge_predictionsFasterRCNNRetinaNetWithNMS = merge_predictions_with_nms(predictionFasterRCNN, predictionRetinaNet, threshold=0.001)
    _, merge_predictionsFasterRCNNRetinaNetWithNMS, targetFRCNNRetinaNetWithNMS = getListFPFNMapImages(merge_predictionsFasterRCNNRetinaNetWithNMS, BaseOutput)
    merge_predictionsFasterRCNNRetinaNetFCOSWithNMS = merge_predictions_with_nms(merge_predictionsFasterRCNNRetinaNetWithNMS, predictionFcos, threshold=0.001)
    _, merge_predictionsFasterRCNNRetinaNetFCOSWithNMS, targetFRCNNRetinaNetFCOSWithNMS = getListFPFNMapImages(merge_predictionsFasterRCNNRetinaNetFCOSWithNMS, BaseOutput)

    FRCNNRNetExtensionWBFBDD100K = merge_predictions_with_extension_wbf([predictionFasterRCNN, predictionRetinaNet, predictionFcos], threshold=0.001)
    FRCNNRNetBDD100KExtensionWBFMap, FRCNNRNetExtensionWBFBDD100K, targetExtensionWBFBDD100K = getListFPFNMapImages(FRCNNRNetExtensionWBFBDD100K, BaseOutput)


    print("Evaluation FRCNN Rnet extension WBF")
    show_comparison_image_models_for_map_inferior([predictionFasterRCNNFilter, predictionRetinaNetFilter, predictionFcosFilter], BDD100K_dataloader, [targetFasterRCNN, targetRetinaNet, targetFcos], ["FasterRCNNModel", "RetinaNetModel", "FcosModel"], device, threshold=0.001, listIndexImagesWithMapInferiorForWBF=listIndexImagesWithMapInferiorForWBF)

    show_comparison_image_models_for_map_inferior([merge_predictionsFasterRCNNRetinaNetFCOSWithoutNMS, merge_predictionsFasterRCNNRetinaNetFCOSWithNMS, FRCNNRNetExtensionWBFBDD100K], BDD100K_dataloader, [targetFRCNNRetinaNetFCOSWithoutNms, targetFRCNNRetinaNetFCOSWithNMS, targetExtensionWBFBDD100K], ["FasterRCNNModel et RetinaNetModel et FcosModel sans NMS", "FasterRCNNModel et RetinaNetModel et FcosModel avec NMS", "FasterRCNNModel et RetinaNetModel et FcosModel avec WBF"], device, threshold=0.001, listIndexImagesWithMapInferiorForWBF=listIndexImagesWithMapInferiorForWBF)

    merge_predictionsFasterRCNNRetinaNetFCOSWithoutNMS = merge_predictions_without_nms(merge_predictionsFasterRCNNRetinaNetWithoutNMS, predictionFcos, threshold=0.001)
    _, merge_predictionsFasterRCNNRetinaNetFCOSWithoutNMS, targetFRCNNRetinaNetFCOSWithoutNms = getListFPFNMapImages(merge_predictionsFasterRCNNRetinaNetFCOSWithoutNMS, BaseOutput, threshold=0, goodPrediction=True)

    print("Evaluation de la combinaisons des modèles FasrerRCNNModel et RetinaNet avec filtre de confiance à 0.5 avec NMS")
    merge_predictionsFasterRCNNRetinaNetFCOSWithNMS = merge_predictions_with_nms(merge_predictionsFasterRCNNRetinaNetWithNMS, predictionFcos, threshold=0.001)
    _, merge_predictionsFasterRCNNRetinaNetFCOSWithNMS, targetFRCNNRetinaNetFCOSWithNMS = getListFPFNMapImages(merge_predictionsFasterRCNNRetinaNetFCOSWithNMS, BaseOutput, threshold=0, goodPrediction=True)
             
    FRCNNRNetExtensionWBFBDD100K = merge_predictions_with_extension_wbf([predictionFasterRCNN, predictionRetinaNet, predictionFcos], threshold=0.001)
    FRCNNRNetBDD100KExtensionWBFMap, FRCNNRNetExtensionWBFBDD100K, targetExtensionWBFBDD100K = getListFPFNMapImages(FRCNNRNetExtensionWBFBDD100K, BaseOutput, threshold=0, goodPrediction=True)

    show_comparison_image_models_for_map_inferior([merge_predictionsFasterRCNNRetinaNetFCOSWithoutNMS, merge_predictionsFasterRCNNRetinaNetFCOSWithNMS, FRCNNRNetExtensionWBFBDD100K], BDD100K_dataloader, [targetFRCNNRetinaNetFCOSWithoutNms, targetFRCNNRetinaNetFCOSWithNMS, targetExtensionWBFBDD100K], ["FasterRCNNModel et RetinaNetModel et FcosModel sans NMS", "FasterRCNNModel et RetinaNetModel et FcosModel avec NMS", "FasterRCNNModel et RetinaNetModel et FcosModel avec WBF"], device, threshold=0.001, listIndexImagesWithMapInferiorForWBF=listIndexImagesWithMapInferiorForWBF)
  


    merge_predictionsFasterRCNNRetinaNetFCOSWithoutNMS = merge_predictions_without_nms(merge_predictionsFasterRCNNRetinaNetWithoutNMS, predictionFcos, threshold=0.001)
    _, merge_predictionsFasterRCNNRetinaNetFCOSWithoutNMS, targetFRCNNRetinaNetFCOSWithoutNms = getListFPFNMapImages(merge_predictionsFasterRCNNRetinaNetFCOSWithoutNMS, BaseOutput, threshold=0.45)

    print("Evaluation de la combinaisons des modèles FasrerRCNNModel et RetinaNet avec filtre de confiance à 0.5 avec NMS")
    merge_predictionsFasterRCNNRetinaNetFCOSWithNMS = merge_predictions_with_nms(merge_predictionsFasterRCNNRetinaNetWithNMS, predictionFcos, threshold=0.001)
    _, merge_predictionsFasterRCNNRetinaNetFCOSWithNMS, targetFRCNNRetinaNetFCOSWithNMS = getListFPFNMapImages(merge_predictionsFasterRCNNRetinaNetFCOSWithNMS, BaseOutput, threshold=0.45)
             
    FRCNNRNetExtensionWBFBDD100K = merge_predictions_with_extension_wbf([predictionFasterRCNN, predictionRetinaNet, predictionFcos], threshold=0.001)
    FRCNNRNetBDD100KExtensionWBFMap, FRCNNRNetExtensionWBFBDD100K, targetExtensionWBFBDD100K = getListFPFNMapImages(FRCNNRNetExtensionWBFBDD100K, BaseOutput, threshold=0.45)

    show_comparison_image_models_for_map_inferior([merge_predictionsFasterRCNNRetinaNetFCOSWithoutNMS, merge_predictionsFasterRCNNRetinaNetFCOSWithNMS, FRCNNRNetExtensionWBFBDD100K], BDD100K_dataloader, [targetFRCNNRetinaNetFCOSWithoutNms, targetFRCNNRetinaNetFCOSWithNMS, targetExtensionWBFBDD100K], ["FasterRCNNModel et RetinaNetModel et FcosModel sans NMS", "FasterRCNNModel et RetinaNetModel et FcosModel avec NMS", "FasterRCNNModel et RetinaNetModel et FcosModel avec WBF"], device, threshold=0.001, listIndexImagesWithMapInferiorForWBF=listIndexImagesWithMapInferiorForWBF)
    
    #treashold = 0.5


print("fini")