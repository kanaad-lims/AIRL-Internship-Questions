# AIRL-Internship-Coding-Assignment


## Q1. Vision Transformer (ViT) for CIFAR-10 Classification
A PyTorch implementation of Vision Transformer for image classification on the CIFAR-10 dataset.

## How to Run in Google Colab

1. **Upload the notebook** to Google Colab or select **Open in Colab** option in the uploaded notebook.
2. **Enable GPU acceleration**: Runtime -> Change runtime type -> Hardware accelerator -> GPU (T4).
3. **Install dependencies** (if needed).
4. **Run all cells** - the training will start automatically with the configured parameters.


## Best Model Configuration
The optimal configuration that achieved the reported results:

## Vision Transformer Configuration (CIFAR-10)

| Parameter       | Value  | Description                          |
|-----------------|--------|--------------------------------------|
| `img_size`      | 32     | CIFAR-10 image size                  |
| `patch_size`    | 4      | Size of image patches (4x4 -> 64 total)|
| `embed_dim`     | 384    | Embedding dimension                  |
| `depth`         | 7      | Number of transformer blocks         |
| `n_heads`       | 6      | Multi-head attention heads           |
| `mlp_ratio`     | 3      | MLP hidden dimension ratio           |
| `dropout`       | 0.1    | Dropout rate                         |
| `batch_size`    | 128    | Training batch size                  |
| `epochs`        | 200    | Training epochs                      |
| `lr`            | 3e-4   | Learning rate                        |
| `weight_decay`  | 0.05   | Weight decay for regularization      |

**Model Parameters:** 10,402,954 total parameters

## Results

| Metric | Value |
|--------|-------|
| **Overall Classification Test Accuracy** | **89.39%** |
| Train Accuracy | 89.57% |
| Train Loss | 0.7318 |
| Test Loss | 0.7823 |


## Model Architecture
- **Patch Embedding**: Splits 32×32 images into 4×4 patches (64 patches total)
- **Positional Encoding**: Learnable positional embeddings
- **Transformer Blocks**: 7 layers with multi-head self-attention and MLP
- **Classification Head**: Uses [CLS] token for final classification
- **Data Augmentation**: RandomCrop, RandomHorizontalFlip, AutoAugment, RandomErasing
- **Regularization**: Label smoothing (0.1), gradient clipping, weight decay


## Training Features
- **Optimizer**: AdamW with cosine annealing scheduler
- **Loss Function**: CrossEntropyLoss with label smoothing
- **Data Augmentation**: Comprehensive CIFAR-10 specific augmentations
- **Mixed Precision**: Compatible with automatic mixed precision training
- **Progress Tracking**: Real-time training metrics with tqdm progress bars

## Analysis

- **Patch Size Choice**: A 4×4 patch size (64 patches per image) balances fine-grained feature extraction with computational efficiency. Smaller patches improve accuracy on small objects but increase memory and computation.  
- **Depth vs. Width Trade-offs**: Using 7 transformer blocks with an embedding dimension of 384 and 6 attention heads provides sufficient representation power without overfitting. Increasing depth or width further could improve accuracy but at the cost of higher memory usage.  
- **Data Augmentation Effects**: RandomCrop, RandomHorizontalFlip, AutoAugment, and RandomErasing significantly enhance generalization, reflected in high test accuracy (89.39%).  
- **Optimizer and Scheduler**: AdamW with cosine annealing ensures stable convergence and helps avoid overfitting compared to standard SGD.  
- **Regularization and Dropout**: Dropout (0.1), label smoothing (0.1), and weight decay (0.05) effectively prevent overfitting and stabilize training.  
- **Overlapping vs. Non-Overlapping Patches**: Not explicitly used here; non-overlapping 4×4 patches provide simplicity and reasonable performance. Overlapping patches could slightly improve accuracy at additional cost.

**Summary**: The chosen patch size, moderate depth/width, comprehensive augmentations, and proper regularization yield high test accuracy while keeping training time and memory usage manageable on standard GPUs.


The implementation achieves competitive results on CIFAR-10 while maintaining a reasonable model size suitable for training on standard GPUs.

---


## Q2. Text-Driven Image Segmentation with SAM 2

## Overview

This project implements **text-prompted image segmentation** using a combination of:

1. **GLIP (Grounded Language-Image Pretraining)** – to detect object regions based on a text prompt.
2. **SAM 2 (Segment Anything Model)** – to generate precise segmentation masks from the detected regions.

The pipeline takes a single image and a user-provided text prompt describing the object to segment. It outputs the original image overlaid with segmentation masks corresponding to the requested object.

---

## Pipeline

1. **Input**: An image (URL or local file) and a text prompt describing the object.  
2. **Region Detection**: GLIP detects bounding boxes corresponding to the text prompt.  
3. **Segmentation**: SAM 2 takes the bounding boxes as seeds and predicts pixel-wise masks.  
4. **Visualization**: Masks are overlaid on the original image for easy visualization.

---

## Example

Input:  
- Image: `http://farm4.staticflickr.com/3693/9472793441_b7822c00de_z.jpg`  
- Text Prompt: `"bobble heads on top of the shelf"`

Output:  
- Image with green overlay highlighting the segmented objects.

---

## Limitations

1. **Dependencies**: The pipeline relies on GLIP and SAM 2 weights, which must be manually downloaded if links expire.  
2. **Mask Accuracy**: SAM segmentation depends on GLIP bounding boxes. If GLIP misdetects or misses regions, SAM masks will also be incorrect.  
3. **Performance**: Running SAM on large images or multiple objects can be slow on Colab without a GPU.  
4. **Model Size**: SAM vit_h and GLIP models are large, requiring significant GPU memory (~12GB).  
5. **Single Image Focus**: This notebook handles one image at a time; batch processing is not implemented.  

---

## Requirements

- Python 3.8+  
- PyTorch + CUDA  
- segment-anything  
- GLIP  
- matplotlib, OpenCV, Pillow, numpy  

---

## Usage

1. Clone the repository and open the notebook in Google Colab.  
2. Run the setup cells to install dependencies and download checkpoints.  
3. Provide an image and a text prompt to generate segmentation masks.  
4. Visualize the overlaid output.


