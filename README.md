# AIRL-Internship-Coding-Assignment


# Q1. Vision Transformer (ViT) for CIFAR-10 Classification
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

The implementation achieves competitive results on CIFAR-10 while maintaining a reasonable model size suitable for training on standard GPUs.
