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

### cnn_wave (Hybrid CNN-Wave)
- CNN stem for local feature extraction (Conv + 2 ResBlocks)
- Wave layers for global processing
- Combines CNN inductive bias with Wave efficiency
- **Parameters**: 1.63M

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
| wave_vision_2d + RandAugment | 1.38M | 80.41% | +21.5% |
| **cnn_wave + RandAugment** | **1.63M** | **92.72% ± 0.12%** | **+33.8%** (validated, 5 seeds) |

### CIFAR-100 Results

| Model | Params | Accuracy | Top-5 |
|-------|--------|----------|-------|
| **cnn_wave + RandAugment** | **1.66M** | **71.93% ± 0.17%** | **91.57%** |
| wave_vision_2d + RandAugment | 1.42M | 52.81% | 78.72% |

## Comparison to ViT

Sources:
- [vision-transformers-cifar10](https://github.com/kentaroy47/vision-transformers-cifar10)
- [ViT-CIFAR (6.3M params, 90%)](https://github.com/omihub777/ViT-CIFAR)
- [How to Train ViT on Small-scale Datasets](https://arxiv.org/abs/2210.07240)

### CIFAR-10

| Model | Params | Accuracy | Source |
|-------|--------|----------|--------|
| **CNN-Wave Hybrid** | **1.63M** | **92.72% ± 0.12%** | **this work** |
| ViT-CIFAR (omihub777) | 6.3M | 90.92% | [repo](https://github.com/omihub777/ViT-CIFAR) |
| ViT patch=4 (1000 epochs) | ~6M | 89% | [repo](https://github.com/kentaroy47/vision-transformers-cifar10) |
| Wave Vision 2D | 1.38M | 80.41% | this work |
| ViT small / patch=2 | ~6M | 80% | [repo](https://github.com/kentaroy47/vision-transformers-cifar10) |
| ResNet18 | 11M | 93% | reference |

### CIFAR-100

| Model | Params | Accuracy | Source |
|-------|--------|----------|--------|
| **CNN-Wave Hybrid** | **1.66M** | **71.93% ± 0.17%** | **this work** |
| ResNet18+RandAug | 11M | 71% | reference |
| ViT-CIFAR (omihub777) | 6.3M | 66.54% | [repo](https://github.com/omihub777/ViT-CIFAR) |
| Wave Vision 2D | 1.42M | 52.81% | this work |

**Key insight**: The CNN-Wave hybrid outperforms ViT on both datasets with ~4x fewer parameters:
- **CIFAR-10**: 92.72% ± 0.12% vs ViT's 90.92% (1.63M vs 6.3M params)
- **CIFAR-100**: 71.93% ± 0.17% vs ViT's 66.54% (1.66M vs 6.3M params)

The CNN stem provides strong local inductive bias, while Wave layers handle global relationships efficiently.

## Key Findings

1. **CNN-Wave Hybrid Achieves Strong Results** (NEW)
   - CIFAR-10: **92.72% ± 0.12%** (5 seeds) - beats ViT (90.92%)
   - CIFAR-100: **71.93% ± 0.17%** (3 seeds) - beats ViT (66.54%)
   - ~4x fewer parameters than ViT (1.6M vs 6.3M)
   - CNN stem handles local features, Wave layers handle global relationships

2. **2D Spatial Awareness is Critical**
   - The 2D model dramatically outperforms the 1D model
   - 74% accuracy in 10 epochs vs 59% in 100 epochs
   - Row/column/local operations capture image structure

3. **RandAugment is the Best Augmentation**
   - +3.6% improvement for the 1D model
   - Simple to add, no hyperparameter tuning needed

4. **Larger Models Need More Regularization**
   - wave_vision_base (7.2M params) completely overfitted
   - Small dataset (CIFAR-10) + large model = disaster without regularization

5. **Combining Augmentations Doesn't Always Help**
   - RandAugment + Mixup performed slightly worse than RandAugment alone

6. **Smaller Patches Don't Help**
   - 2x2 patches (256 tokens) tracked same as 4x4 baseline
   - More tokens ≠ better features; semantic content per patch matters

## Files Created

- `models/wave_vision.py` - 1D Wave Vision model
- `models/wave_vision_2d.py` - 2D spatial-aware Wave Vision model
- `models/wave_vision_hybrid.py` - CNN-Wave hybrid model (BEST)
- `benchmarks/vision.py` - Vision benchmark infrastructure
- `train_vision.py` - Training script with MLflow integration
- `tests/unit/test_wave_vision.py` - Unit tests

## Usage

```bash
# Train CNN-Wave hybrid (BEST - 92.36% accuracy)
python train_vision.py --task cifar10 --model cnn_wave --randaugment --mlflow

# Train 2D model
python train_vision.py --task cifar10 --model wave_vision_2d --randaugment

# Train 1D model
python train_vision.py --task cifar10 --model wave_vision --randaugment

# Custom epochs
python train_vision.py --task cifar10 --model cnn_wave --epochs 200 --randaugment
```

## Future Directions

1. **Scale to ImageNet** - Test architecture at scale (224x224 images)
2. **Larger hybrid models** - Increase capacity with proper regularization
3. **Pre-training** - Self-supervised pre-training for transfer learning
4. **Compare with other efficient architectures** - MobileNet, EfficientNet, etc.
5. **Ablate CNN stem depth** - Test 1, 2, 3+ ResBlocks in stem
