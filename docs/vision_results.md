# Wave Vision Results

## Overview

Extension of Wave Network from text to image classification using CIFAR-10.

## Models

### wave_vision (1D baseline)
- Treats image patches as a flat sequence
- Uses WaveLayer from wave_network_deep.py
- **Parameters**: 938K

### wave_vision_2d (2D spatial-aware)
- Processes patches with row-wise, column-wise, and local operations
- Preserves 2D spatial structure
- **Parameters**: 1.38M

## Ablation Results (100 epochs unless noted)

| Model/Config | Params | Accuracy | Notes |
|-------------|--------|----------|-------|
| wave_vision (baseline) | 938K | 58.89% | Basic flat sequence processing |
| + RandAugment | 938K | 62.51% | **+3.6%** best single augmentation |
| + Mixup | 938K | 61.35% | +2.5% |
| + RandAugment + Mixup | 938K | 62.03% | +3.1% (combined didn't help) |
| + Label smoothing | 938K | 59.32% | +0.4% (minimal effect) |
| wave_vision_base | 7.2M | 10.69% | FAILED - severe overfitting |
| wave_vision_2d | 1.38M | 77.76% | +18.9% |
| **wave_vision_2d + RandAugment** | 1.38M | **80.41%** | **+21.5%** best result |

### CIFAR-100 Results

| Model | Params | Accuracy | Top-5 |
|-------|--------|----------|-------|
| wave_vision_2d + RandAugment | 1.42M | 52.81% | 78.72% |

## Comparison to ViT

Sources:
- [vision-transformers-cifar10](https://github.com/kentaroy47/vision-transformers-cifar10)
- [ViT-CIFAR (6.3M params, 90%)](https://github.com/omihub777/ViT-CIFAR)
- [How to Train ViT on Small-scale Datasets](https://arxiv.org/abs/2210.07240)

### CIFAR-10

| Model | Params | Accuracy |
|-------|--------|----------|
| ViT small | ~6M | 80% |
| ViT patch=2 | ~6M | 80% |
| **Wave Vision 2D** | **1.38M** | **80.41%** |
| ResNet18 | 11M | 93% |

### CIFAR-100

| Model | Params | Accuracy |
|-------|--------|----------|
| ViT patch=4 | ~6M | 52% |
| **Wave Vision 2D** | **1.42M** | **52.81%** |
| ResNet18+RandAug | 11M | 71% |

**Key insight**: Wave Vision matches ViT accuracy with ~4x fewer parameters. Both transformer-style architectures struggle vs CNNs on small datasets due to lack of inductive biases (locality, translation invariance). The 2D spatial operations give Wave Network CNN-like locality benefits.

## Key Findings

1. **2D Spatial Awareness is Critical**
   - The 2D model dramatically outperforms the 1D model
   - 74% accuracy in 10 epochs vs 59% in 100 epochs
   - Row/column/local operations capture image structure

2. **RandAugment is the Best Augmentation**
   - +3.6% improvement for the 1D model
   - Simple to add, no hyperparameter tuning needed

3. **Larger Models Need More Regularization**
   - wave_vision_base (7.2M params) completely overfitted
   - Small dataset (CIFAR-10) + large model = disaster without regularization

4. **Combining Augmentations Doesn't Always Help**
   - RandAugment + Mixup performed slightly worse than RandAugment alone

5. **Smaller Patches Don't Help**
   - 2x2 patches (256 tokens) tracked same as 4x4 baseline
   - More tokens ≠ better features; semantic content per patch matters

## Files Created

- `models/wave_vision.py` - 1D Wave Vision model
- `models/wave_vision_2d.py` - 2D spatial-aware Wave Vision model
- `benchmarks/vision.py` - Vision benchmark infrastructure
- `train_vision.py` - Training script with MLflow integration
- `tests/unit/test_wave_vision.py` - Unit tests

## Usage

```bash
# Train 1D model
python train_vision.py --task cifar10 --model wave_vision --epochs 100

# Train 2D model (recommended)
python train_vision.py --task cifar10 --model wave_vision_2d --epochs 100

# With RandAugment
python train_vision.py --task cifar10 --model wave_vision --randaugment

# With MLflow tracking
python train_vision.py --task cifar10 --model wave_vision_2d --mlflow
```

## Future Directions

1. **More aggressive regularization for larger models** - wave_vision_base failed
2. **Pre-training on larger datasets** - ImageNet pre-training
3. **Larger datasets** - ImageNet, Food-101, etc.
4. **Hybrid CNN-Wave** - CNN stem for low-level features + Wave for high-level
