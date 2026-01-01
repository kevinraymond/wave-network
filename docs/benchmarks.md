# Benchmark Results

Detailed benchmark results for Wave Network compared to BERT and FNet baselines.

All tests run locally on a single Nvidia 4090.

---

## Classification Benchmarks

### AG News

**Latest Run:**

| Metric    | Wave Network | BERT base |
| --------- | ------------ | --------- |
| Accuracy  | 92.03%       | 94.63%    |
| Precision | 92.04%       | 94.66%    |
| Recall    | 92.03%       | 94.63%    |
| F1 Score  | 92.03%       | 94.63%    |
| Loss      | 0.2378       | 0.2022    |

```
Final Test Results:
Wave Network (batch_size=64):
Performance Metrics: {'loss': 0.23779155445449493, 'accuracy': 0.9202631578947369, 'precision': np.float64(0.9204283202989598), 'recall': np.float64(0.9202631578947369), 'f1': np.float64(0.9202632926679699)}
Resource Usage: {'parameters': 24626692, 'memory_peak': 1165.90478515625}

BERT (batch_size=32):
Performance Metrics: {'loss': 0.20216417188576163, 'accuracy': 0.9463157894736842, 'precision': np.float64(0.9466198771601457), 'recall': np.float64(0.9463157894736842), 'f1': np.float64(0.9463277574455472)}
Resource Usage: {'parameters': 109485316, 'memory_peak': 4170.14404296875}
```

### DBpedia14

**Latest Run:**

| Metric    | Wave Network | BERT base |
| --------- | ------------ | --------- |
| Accuracy  | 98.15%       | 99.26%    |
| Precision | 98.16%       | 99.26%    |
| Recall    | 98.15%       | 99.26%    |
| F1 Score  | 98.15%       | 99.26%    |
| Loss      | 0.0630       | 0.0454    |

```
Final Test Results:
Wave Network (batch_size=64):
Performance Metrics: {'loss': 0.06299331468511946, 'accuracy': 0.9815, 'precision': np.float64(0.9815559797368222), 'recall': np.float64(0.9815), 'f1': np.float64(0.9815079453378028)}
Resource Usage: {'parameters': 24634382, 'memory_peak': 1166.0244140625}

BERT (batch_size=32):
Performance Metrics: {'loss': 0.045359594953326574, 'accuracy': 0.9925714285714285, 'precision': np.float64(0.9925737777449415), 'recall': np.float64(0.9925714285714285), 'f1': np.float64(0.9925717929489069)}
Resource Usage: {'parameters': 109493006, 'memory_peak': 4170.3798828125}
```

**Best Run:**

| Metric    | Wave Network | BERT base |
| --------- | ------------ | --------- |
| Accuracy  | 98.34%       | 99.30%    |
| Precision | 98.34%       | 99.30%    |
| Recall    | 98.34%       | 99.30%    |
| F1 Score  | 98.34%       | 99.30%    |
| Loss      | 0.0596       | 0.0370    |

### IMDB

_Still requires tuning - memory usage is much higher!_

**Latest Run:**

| Metric    | Wave Network | BERT base |
| --------- | ------------ | --------- |
| Accuracy  | 87.22%       | 88.69%    |
| Precision | 87.26%       | 88.69%    |
| Recall    | 87.22%       | 88.69%    |
| F1 Score  | 87.21%       | 88.69%    |
| Loss      | 0.3014       | 0.2818    |

```
Final Test Results:
WAVE_NETWORK (batch_size=64):
Performance Metrics: {'loss': 0.3014372759653479, 'accuracy': 0.87216, 'precision': np.float64(0.8725766277036728), 'recall': np.float64(0.87216), 'f1': np.float64(0.8721242512602836)}
Resource Usage: {'parameters': 24625154, 'memory_peak': 2892.970703125}

BERT (batch_size=32):
Performance Metrics: {'loss': 0.28180385720797, 'accuracy': 0.88688, 'precision': np.float64(0.8868966495556584), 'recall': np.float64(0.88688), 'f1': np.float64(0.8868787829966989)}
Resource Usage: {'parameters': 109483778, 'memory_peak': 4163.06396484375}
```

---

## GLUE Benchmark

Wave Network vs [FNet](https://arxiv.org/abs/2105.03824) (Fourier-based model) on the full GLUE benchmark.

### Wave Network vs FNet Comparison

| Task | Metric | Wave Network | FNet | Winner | BERT |
|------|--------|-------------|------|--------|------|
| **SST-2** | Accuracy | **80.5%** | 79.5% | Wave | 93% |
| **CoLA** | Matthews | **0.129** | 0.000 | Wave | 52% |
| **MRPC** | Accuracy | **69.9%** | 68.4% | Wave | 86% |
| **QQP** | Accuracy | **79.9%** | 76.6% | Wave | 91% |
| **QNLI** | Accuracy | 59.7% | **61.9%** | FNet | 91% |
| **RTE** | Accuracy | 49.5% | **52.7%** | FNet | 66% |
| **MNLI** | Accuracy | 51.2% | **53.4%** | FNet | 84% |
| **STS-B** | Pearson | **0.126** | 0.103 | Wave | 85% |

**Summary:** Wave Network wins 5/8 tasks with **less than half the parameters** (24.6M vs 52.2M).

### Key Findings

1. **Classification strength**: Wave Network excels at sentiment analysis (SST-2) and paraphrase detection (QQP, MRPC) - tasks where semantic similarity matters most

2. **NLI weakness**: Both models struggle on natural language inference tasks (QNLI, RTE, MNLI), suggesting lightweight architectures need attention-like mechanisms for complex reasoning

3. **Parameter efficiency**: Wave Network achieves competitive results with 2x fewer parameters than FNet

4. **Stability**: Wave Network required gradient checking to prevent NaN issues during training on some tasks

### Running GLUE Benchmarks

```bash
# Install GLUE dependencies
uv pip install datasets evaluate

# Run single task
python train_glue.py --task sst2 --model wave_network --lr 0.0001

# Run all tasks
python train_glue.py --task all --model wave_network

# Compare models
python train_glue.py --task sst2 --model wave_network fnet

# List available tasks
python train_glue.py --list-tasks
```

Results are saved to `data/results/` as JSON files.

---

## Next Steps for Performance Improvement

Based on our GLUE results, here are the most promising directions for boosting Wave Network performance:

### 1. Training Improvements

- **More epochs**: Current runs use 3-5 epochs. Increasing to 10-20 epochs with early stopping could help, especially on smaller datasets (RTE, CoLA)
- **Learning rate scheduling**: Implement cosine annealing or reduce-on-plateau instead of linear warmup only
- **Larger batch sizes**: With gradient accumulation, simulate batch sizes of 128-256 for more stable gradients
- **Mixed precision training**: Enable FP16/BF16 for faster training and potential regularization benefits

### 2. Architecture Changes

- **Depth**: Current model is shallow. Adding more WaveLayers (4-6) with residual connections may improve reasoning tasks
- **Pre-LayerNorm**: Switch to pre-norm architecture for more stable deep training
- **Learned position embeddings**: Current sinusoidal positions may limit sequence understanding
- **Multi-scale wave operations**: Combine modulation and interference modes instead of selecting one

### 3. Data & Augmentation

- **Pre-training**: Pre-train on large corpus (Wikipedia, BookCorpus) before fine-tuning on GLUE
- **Data augmentation**: Back-translation, synonym replacement, or mixup for small datasets
- **Longer sequences**: Increase max_length from 128 to 256-512 for tasks with longer inputs

### 4. Task-Specific Tuning

- **NLI tasks**: Add cross-sentence attention or explicit entailment heads
- **Regression (STS-B)**: Use different loss functions (e.g., cosine embedding loss)
- **Small datasets (RTE, CoLA)**: Apply aggressive dropout (0.3-0.5) and weight decay

### 5. Hyperparameter Search

Priority hyperparameters to tune:
- Learning rate: [1e-5, 5e-5, 1e-4, 5e-4]
- Weight decay: [0.0, 0.01, 0.1]
- Dropout: [0.1, 0.2, 0.3]
- Embedding dim: [512, 768, 1024]
- Number of layers: [1, 2, 3, 4, 6]

Use W&B sweeps for systematic exploration:
```bash
wandb sweep configs/sweeps/hyperparameter_sweep.yaml
```

### Research Questions

1. Can pre-training close the gap with BERT on NLI tasks?
2. Does increasing depth help Wave Network on reasoning tasks?
3. Would hybrid attention-wave architectures get the best of both worlds?
