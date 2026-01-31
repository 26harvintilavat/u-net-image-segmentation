# U-Net Image Segmentation (Oxford-IIIT Pet)

This project implements a **U-Net-based image segmentation pipeline** using PyTorch.  
The goal is to perform **binary segmentation** on pet images from the Oxford-IIIT Pet Dataset.

The project focuses on:
- correct data preprocessing
- stable training
- proper evaluation using Dice score
- clean project structure suitable for GitHub

## 🧠 Project Overview

- **Model**: U-Net (encoder–decoder with skip connections)
- **Task**: Binary image segmentation
- **Loss Function**: `BCEWithLogitsLoss`
- **Metrics**:
  - Dice Score
  - Pixel Accuracy
- **Training**: Mixed Precision Training (AMP)
- **Augmentation**: Albumentations

## 📊 Results (Small Subset)

> ⚠️ **Important Note**  
> To speed up experimentation and debugging, the model was trained on a **small subset** of the dataset:
>
> - **Training images**: 100  
> - **Validation images**: 20  
>
> The reported metrics reflect performance on this small subset and are **not meant to represent full-dataset performance**.

### Metrics on Validation Set:
- **Dice Score**: ~0.98  
- **Pixel Accuracy**: ~97%

These results confirm that the **pipeline, metrics, and preprocessing are correct**, not that the model is fully optimized.

## 🗂 Dataset

- **Dataset**: Oxford-IIIT Pet Dataset
- **Segmentation Type**: Binary (pet vs background)

### Mask Preprocessing
All segmentation masks are explicitly converted to **binary masks (0 or 1)** during dataset loading to ensure:
- correct Dice computation
- stable loss behavior
- consistent visualization

The dataset itself is **not included** in this repository.

## 🏗 Project Structure
U-Net-Image-Segmentation/
│
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
│
├── data/
|   ├── dataset.py
│   ├── create_small_dataset.py
│   ├── dataset_download.py
│   └── split_dataset.py
|      
├── .gitignore
├── README.md              
├── requirements.txt
