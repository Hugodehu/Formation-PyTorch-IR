import torch
from torch.utils.data import DataLoader, Subset
from torchvision import models, datasets, transforms
from torchvision.transforms import ToTensor
from function.evaluate_performance_model import evaluate_performance_model
from function.get_prediction_model import getPredictionModel
from classes.custom_dataset_bdd100k import CustomDataset, collate_fn_for_bdd100K, collate_fn_for_coco
from function.merge_predictions import merge_predictions_with_nms, merge_predictions_without_nms
from function.visualize_prediction_image import plot_precision_recall_curve

# Usage
img_dir_bdd100K = 'data/bdd100K/bdd100K/images/100K/val'
json_file_bdd100K = 'data/bdd100k/bdd100K/labels/det_20/det_val.json'
BDD100K_dataset = CustomDataset(json_file=json_file_bdd100K, img_dir=img_dir_bdd100K, transform=ToTensor())

img_dir_coco = 'data/coco/val2017'
json_file_coco = 'data/coco/annotations/instances_val2017.json'
transform = transforms.Compose([transforms.ToTensor()])
COCO_dataset = datasets.CocoDetection(root=img_dir_coco, annFile=json_file_coco, transform=transform,)

# Créer un sous-ensemble contenant seulement les 10 premières images
subset_indices = list(range(100))
BDD100Ksubset_dataset = Subset(BDD100K_dataset, subset_indices)

batch_size = 4 # 4 images per batch

# Create data loaders.
BDD100K_dataloader = DataLoader(BDD100K_dataset, batch_size=batch_size, collate_fn=collate_fn_for_bdd100K)

COCOsubset_dataset = Subset(COCO_dataset, subset_indices)

Coco_dataloader = DataLoader(COCO_dataset, batch_size=batch_size, collate_fn=collate_fn_for_coco)

# récupération du cpu ou gpu pour l'évaluation.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else"cpu"
print("Using {} device".format(device))

# Precision: 0.2349, Recall: 0.6171, Mean IoU: 0.5180
# deuxième test en ajoutant le filtre de confiance à 0.5 : Precision: 0.6498, Recall: 0.4718, Mean IoU: 0.4100
try:
    FasterRCNNModel = torch.load("models/FasterRCNNModel.pt")
except FileNotFoundError as e:
    FasterRCNNModel = models.detection.fasterrcnn_resnet50_fpn_v2(pretrained= True)
    FasterRCNNModel.to(device)
    torch.save(FasterRCNNModel, "models/FasterRCNNModel.pt")
    pass


# Precision: 0.8695, Recall: 0.3031, Mean IoU: 0.2787
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


print("Evaluation de la combinaisons des modèles FasrerRCNNModel et RetinaNet avec filtre de confiance à 0.5 sans NMS")
merge_predictionsFasterRCNNRetinaNet = merge_predictions_without_nms(predictionFasterRCNN, predictionRetinaNet)
FasterRCNNRetinaNetModelBdd100KPrecision, FasterRCNNRetinaNetModelBdd100KRecall, FasterRCNNRetinaNetModelBdd100KTargetBoxes, FasterRCNNRetinaNetModelBdd100KPredBoxs, FasterRCNNRetinaNetModelBdd100KMap = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNet, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel et RetinaNet avec filtre de confiance à 0.5 avec NMS")
merge_predictionsFasterRCNNRetinaNet = merge_predictions_with_nms(predictionFasterRCNN, predictionRetinaNet)
FasterRCNNRetinaNetModelBdd100KPrecisionNMS, FasterRCNNRetinaNetModelBdd100KRecallNMS, FasterRCNNRetinaNetModelBdd100KTargetBoxesNMS, FasterRCNNRetinaNetModelBdd100KPredBoxsNMS, FasterRCNNRetinaNetModelBdd100KMapNMS = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNet, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel et Fcos avec filtre de confiance à 0.5 sans NMS")
merge_predictionsFasterRCNNFcos = merge_predictions_without_nms(predictionFasterRCNN, predictionFcos)
FasterRCNNFcosModelBdd100KPrecision, FasterRCNNFcosModelBdd100KRecall, FasterRCNNFcosModelBdd100KTargetBoxes, FasterRCNNFcosModelBdd100KPredBoxs, FasterRCNNFcosModelBdd100KMap = evaluate_performance_model(merge_predictionsFasterRCNNFcos, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel et Fcos avec filtre de confiance à 0.5 avec NMS")
merge_predictionsFasterRCNNFcos = merge_predictions_with_nms(predictionFasterRCNN, predictionFcos)
FasterRCNNFcosModelBdd100KPrecisionNMS, FasterRCNNFcosModelBdd100KRecallNMS, FasterRCNNFcosModelBdd100KTargetBoxesNMS, FasterRCNNFcosModelBdd100KPredBoxsNMS, FasterRCNNFcosModelBdd100KMapNMS = evaluate_performance_model(merge_predictionsFasterRCNNFcos, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles RetinaNetModel et Fcos avec filtre de confiance à 0.5 sans NMS")
merge_predictionsRetinaNetFcos = merge_predictions_without_nms(predictionRetinaNet, predictionFcos)
RetinaNetFcosModelBdd100KPrecision, RetinaNetFcosModelBdd100KRecall, RetinaNetFcosModelBdd100KTargetBoxes, RetinaNetFcosModelBdd100KPredBoxs, RetinaNetFcosModelBdd100KMap = evaluate_performance_model(merge_predictionsRetinaNetFcos, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles RetinaNetModel et Fcos avec filtre de confiance à 0.5 avec NMS")
merge_predictionsRetinaNetFcos = merge_predictions_with_nms(predictionRetinaNet, predictionFcos)
RetinaNetFcosModelBdd100KPrecisionNMS, RetinaNetFcosModelBdd100KRecallNMS, RetinaNetFcosModelBdd100KTargetBoxesNMS, RetinaNetFcosModelBdd100KPredBoxsNMS, RetinaNetFcosModelBdd100KMapNMS = evaluate_performance_model(merge_predictionsRetinaNetFcos, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel, RetinaNet et Fcos avec filtre de confiance à 0.5 sans NMS")
merge_predictionsFasterRCNNRetinaNetFcos = merge_predictions_without_nms(merge_predictionsFasterRCNNRetinaNet, predictionFcos)
FasterRCNNRetinaNetFcosModelBdd100KPrecision, FasterRCNNRetinaNetFcosModelBdd100KRecall, FasterRCNNRetinaNetFcosModelBdd100KTargetBoxes, FasterRCNNRetinaNetFcosModelBdd100KPredBoxs, FasterRCNNRetinaNetFcosModelBdd100KMap = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetFcos, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel, RetinaNet et Fcos avec filtre de confiance à 0.5 avec NMS")
merge_predictionsFasterRCNNRetinaNetFcos = merge_predictions_with_nms(merge_predictionsFasterRCNNRetinaNet, predictionFcos)
FasterRCNNRetinaNetFcosModelBdd100KPrecisionNMS, FasterRCNNRetinaNetFcosModelBdd100KRecallNMS, FasterRCNNRetinaNetFcosModelBdd100KTargetBoxesNMS, FasterRCNNRetinaNetFcosModelBdd100KPredBoxsNMS, FasterRCNNRetinaNetFcosModelBdd100KMapNMS = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetFcos, targetsOutFasterRCNN)

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
merge_predictionsFasterRCNNRetinaNet = merge_predictions_without_nms(predictionFasterRCNN, predictionRetinaNet)
FasterRCNNRetinaNetModelCocoPrecision, FasterRCNNRetinaNetModelCocoRecall, FasterRCNNRetinaNetModelCocoTargetBoxes, FasterRCNNRetinaNetModelCocoPredBoxs, FasterRCNNRetinaNetModelCocoMap = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNet, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel et RetinaNet avec filtre de confiance à 0.5 avec NMS")
merge_predictionsFasterRCNNRetinaNet = merge_predictions_with_nms(predictionFasterRCNN, predictionRetinaNet)
FasterRCNNRetinaNetModelCocoPrecisionNMS, FasterRCNNRetinaNetModelCocoRecallNMS, FasterRCNNRetinaNetModelCocoTargetBoxesNMS, FasterRCNNRetinaNetModelCocoPredBoxsNMS, FasterRCNNRetinaNetModelCocoMapNMS = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNet, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel et Fcos avec filtre de confiance à 0.5 sans NMS")
merge_predictionsFasterRCNNFcos = merge_predictions_without_nms(predictionFasterRCNN, predictionFcos)
FasterRCNNFcosModelCocoPrecision, FasterRCNNFcosModelCocoRecall, FasterRCNNFcosModelCocoTargetBoxes, FasterRCNNFcosModelCocoPredBoxs, FasterRCNNFcosModelCocoMap = evaluate_performance_model(merge_predictionsFasterRCNNFcos, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel et Fcos avec filtre de confiance à 0.5 avec NMS")
merge_predictionsFasterRCNNFcos = merge_predictions_with_nms(predictionFasterRCNN, predictionFcos)
FasterRCNNFcosModelCocoPrecisionNMS, FasterRCNNFcosModelCocoRecallNMS, FasterRCNNFcosModelCocoTargetBoxesNMS, FasterRCNNFcosModelCocoPredBoxsNMS, FasterRCNNFcosModelCocoMapNMS = evaluate_performance_model(merge_predictionsFasterRCNNFcos, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles RetinaNetModel et Fcos avec filtre de confiance à 0.5 sans NMS")
merge_predictionsRetinaNetFcos = merge_predictions_without_nms(predictionRetinaNet, predictionFcos)
RetinaNetFcosModelCocoPrecision, RetinaNetFcosModelCocoRecall, RetinaNetFcosModelCocoTargetBoxes, RetinaNetFcosModelCocoPredBoxs, RetinaNetFcosModelCocoMap = evaluate_performance_model(merge_predictionsRetinaNetFcos, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles RetinaNetModel et Fcos avec filtre de confiance à 0.5 avec NMS")
merge_predictionsRetinaNetFcos = merge_predictions_with_nms(predictionRetinaNet, predictionFcos)
RetinaNetFcosModelCocoPrecisionNMS, RetinaNetFcosModelCocoRecallNMS, RetinaNetFcosModelCocoTargetBoxesNMS, RetinaNetFcosModelCocoPredBoxsNMS, RetinaNetFcosModelCocoMapNMS = evaluate_performance_model(merge_predictionsRetinaNetFcos, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel, RetinaNet et Fcos avec filtre de confiance à 0.5 sans NMS")
merge_predictionsFasterRCNNRetinaNetFcos = merge_predictions_without_nms(merge_predictionsFasterRCNNRetinaNet, predictionFcos)
FasterRCNNRetinaNetFcosModelCocoPrecision, FasterRCNNRetinaNetFcosModelCocoRecall, FasterRCNNRetinaNetFcosModelCocoTargetBoxes, FasterRCNNRetinaNetFcosModelCocoPredBoxs, FasterRCNNRetinaNetFcosModelCocoMap = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetFcos, targetsOutFasterRCNN)

print("Evaluation de la combinaisons des modèles FasrerRCNNModel, RetinaNet et Fcos avec filtre de confiance à 0.5 avec NMS")
merge_predictionsFasterRCNNRetinaNetFcos = merge_predictions_with_nms(merge_predictionsFasterRCNNRetinaNet, predictionFcos)
FasterRCNNRetinaNetFcosModelCocoPrecisionNMS, FasterRCNNRetinaNetFcosModelCocoRecallNMS, FasterRCNNRetinaNetFcosModelCocoTargetBoxesNMS, FasterRCNNRetinaNetFcosModelCocoPredBoxsNMS, FasterRCNNRetinaNetFcosModelCocoMapNMS = evaluate_performance_model(merge_predictionsFasterRCNNRetinaNetFcos, targetsOutFasterRCNN)

# Ecrire dans un fichier tex les informations sous forme de tableau
with open("Resultats/performance_model_mAP.tex", "w") as f:
    f.write("\\documentclass{article}\n")
    f.write("\\usepackage{graphicx} % Required for inserting images\n")
    f.write("\\begin{document}\n")
    f.write("\\begin{table}[h!]\n")
    f.write("\\centering\n")
    f.write("\\begin{tabular}{|c||c|c|c|c||c|c|} \n")
    f.write("\\hline\n")
    f.write("Model & \\multicolumn{3}{|c||}{BDD100K} & \\multicolumn{3}{|c|}{COCO} \\\\ \n")
    f.write(" & precision & recall & mAP  & precision & recall & mAP  \\\\ [0.5ex] \n")
    f.write("\\hline\n")
    f.write("Faster R-CNN & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNModelBdd100KPrecision, FasterRCNNModelBdd100KRecall, FasterRCNNModelBdd100KMap, FasterRCNNModelCocoPrecision, FasterRCNNModelCocoRecall, FasterRCNNModelCocoMap))
    f.write("\\hline\n")
    f.write("RetinaNet & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RetinaNetModelBdd100KPrecision, RetinaNetModelBdd100KRecall, RetinaNetModelBdd100KMap, RetinaNetModelCocoPrecision, RetinaNetModelCocoRecall, RetinaNetModelCocoMap))
    f.write("\\hline\n")
    f.write("FCOS & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FcosModelBdd100KPrecision, FcosModelBdd100KRecall, FcosModelBdd100KMap, FcosModelCocoPrecision, FcosModelCocoRecall, FcosModelCocoMap))
    f.write("\\hline\n")
    f.write("Faster R-CNN, RetinaNet Cat & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetModelBdd100KPrecision, FasterRCNNRetinaNetModelBdd100KRecall,FasterRCNNRetinaNetModelBdd100KMap , FasterRCNNRetinaNetModelCocoPrecision, FasterRCNNRetinaNetModelCocoRecall, FasterRCNNRetinaNetModelCocoMap))
    f.write("\\hline\n")
    f.write("Faster R-CNN, RetinaNet NMS & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetModelBdd100KPrecisionNMS, FasterRCNNRetinaNetModelBdd100KRecallNMS, FasterRCNNRetinaNetModelBdd100KMapNMS, FasterRCNNRetinaNetModelCocoPrecisionNMS, FasterRCNNRetinaNetModelCocoRecallNMS, FasterRCNNRetinaNetModelCocoMapNMS))
    f.write("\\hline\n")
    f.write("Faster R-CNN, FCOS Cat & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNFcosModelBdd100KPrecision, FasterRCNNFcosModelBdd100KRecall, FasterRCNNFcosModelBdd100KMap, FasterRCNNFcosModelCocoPrecision, FasterRCNNFcosModelCocoRecall, FasterRCNNFcosModelCocoMap))
    f.write("\\hline\n")
    f.write("Faster R-CNN, FCOS NMS & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNFcosModelBdd100KPrecisionNMS, FasterRCNNFcosModelBdd100KRecallNMS, FasterRCNNFcosModelBdd100KMapNMS, FasterRCNNFcosModelCocoPrecisionNMS, FasterRCNNFcosModelCocoRecallNMS, FasterRCNNFcosModelCocoMapNMS))
    f.write("\\hline\n")
    f.write("RetinaNet, FCOS Cat & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RetinaNetFcosModelBdd100KPrecision, RetinaNetFcosModelBdd100KRecall, RetinaNetFcosModelBdd100KMap, RetinaNetFcosModelCocoPrecision, RetinaNetFcosModelCocoRecall, RetinaNetFcosModelCocoMap))
    f.write("\\hline\n")
    f.write("RetinaNet, FCOS NMS & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(RetinaNetFcosModelBdd100KPrecisionNMS, RetinaNetFcosModelBdd100KRecallNMS, RetinaNetFcosModelBdd100KMapNMS, RetinaNetFcosModelCocoPrecisionNMS, RetinaNetFcosModelCocoRecallNMS, RetinaNetFcosModelCocoMapNMS))
    f.write("\\hline\n")
    f.write("Faster R-CNN, RetinaNet, FCOS Cat & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetFcosModelBdd100KPrecision, FasterRCNNRetinaNetFcosModelBdd100KRecall, FasterRCNNRetinaNetFcosModelBdd100KMap, FasterRCNNRetinaNetFcosModelCocoPrecision, FasterRCNNRetinaNetFcosModelCocoRecall, FasterRCNNRetinaNetFcosModelCocoMap))
    f.write("\\hline\n")
    f.write("Faster R-CNN, RetinaNet, FCOS NMS & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\ \n".format(FasterRCNNRetinaNetFcosModelBdd100KPrecisionNMS, FasterRCNNRetinaNetFcosModelBdd100KRecallNMS, FasterRCNNRetinaNetFcosModelBdd100KMapNMS, FasterRCNNRetinaNetFcosModelCocoPrecisionNMS, FasterRCNNRetinaNetFcosModelCocoRecallNMS, FasterRCNNRetinaNetFcosModelCocoMapNMS))
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\caption{Résultats des différents modèles en fonction de la base de test.}\n")
    f.write("\\label{table:data}\n")
    f.write("\\end{table}\n")
    f.write("\\end{document}\n")
    f.close()
    pass

