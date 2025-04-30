# Ambiguity Constrained Nuclei Segmentation

## Instructions to Run Training and Evaluation
1. Download NuInsSeg dataset from [Kaggle](https://www.kaggle.com/datasets/ipateam/nuinsseg/data). Example raw input images, labels, and ambiguity maps are provided in the data/ folder. {[Reference Github](https://github.com/masih4/NuInsSeg?tab=readme-ov-file)}
2. Create python environment with requirements.txt
3. Run process_data.py to process raw dataset folders into a structured directory for training and evaluation. Use --help for relevant command line arguments or refer to code. 
4. Run train_test_split.py to generate train test splits for training and evaluation on processed NuInsSeg dataset. 
5. To train the model, run train_amb.py to train model on the combined semantic segmentation and ambiguity task; otherwise run train_control.py to train model solely on semantic segmentation task. Saved models with associated training and validation loss curves/Dice scores are provided in the outputs (dual task) or outputs_control (only segmentation) folders.
6. To use the model for inference, postprocessing for instance segmentation, and evaluation on test data, run eva_amb.py or eval_control.py. Outputs include instance level Dice Score, IoU metric, and three random visualizations stored in the outputs/figs or outputs_control/figs folders.

## Methodology
1. Preprocess the NuInsSeg dataset to extract raw RGB images of patches of HE stained tissues, manually annotated instance labels for distinct nuclei in the image, and manually annotated binary map indicating regions of ambiguity as identified by the pathologist. Stain normalization and denoising using CurvatureFlow are performed on raw images prior to training.
2. Model consists of a UNet pretrained on Spleen CT images for a different segmentation task, an ambiguity convolutional head, and a ambiguity conditioned semantic segmentation head. A control model without any ambiguity based inputs or outputs is also provided for comparison.
3. Model finetuning is performed with DiceLoss for segmentation and binary cross entropy loss for ambiguity prediction. An Adam optimizer and a WarmupCosine based learning rate scheduler was used. For now, the UNet was left frozen for rapid training.
4. The predicted binary maps of nuclei undergo postprocessing using SimpleITK to obtain distinct nuclear instances. This consists of morphological hole opening, ambiguity conditioned distance map generation, seed point generation using RegionalMinima, and finally a simple morphological watershed algorithm.
5. Performance on the test data is evaluated using Dice and Intersection over Union (IoU) scores.

## Results

### Evaluation on Ambiguity-Conditioned UNet
* Semantic Dice (pre‑postproc): 0.695
* Instance Dice: 0.6430
* Instance IoU: 0.4855

### Evaluation on Control UNet 
* Semantic Dice (pre‑postproc): 0.6715
* Instance Dice: 0.6496
* Instance IoU: 0.4888