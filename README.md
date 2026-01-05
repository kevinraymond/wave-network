# Wave Network

[![CI](https://github.com/kevinraymond/wave-network/actions/workflows/ci.yml/badge.svg)](https://github.com/kevinraymond/wave-network/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kevinraymond/wave-network/branch/main/graph/badge.svg)](https://codecov.io/gh/kevinraymond/wave-network)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Implementation of [Wave Network: An Ultra-Small Language Model](https://arxiv.org/pdf/2411.02674) - a lightweight alternative to BERT using wave-based signal processing instead of attention.

## Results

### Text Classification

Wave Network achieves near-BERT accuracy with **4.5x fewer parameters**:

| Dataset | Wave Network | BERT | Parameters |
|---------|-------------|------|------------|
| AG News | 92.0% | 94.6% | 24.6M vs 109M |
| DBpedia | 98.2% | 99.3% | 24.6M vs 109M |
| IMDB | 87.2% | 88.7% | 24.6M vs 109M |

On GLUE benchmark, Wave Network wins 5/8 tasks vs FNet with half the parameters (24.6M vs 52.2M).

### Image Classification

CNN-Wave hybrid outperforms ViT with **4x fewer parameters**:

| Model | CIFAR-10 | CIFAR-100 | Parameters |
|-------|----------|-----------|------------|
| CNN-Wave (ours) | **92.72%** | **71.93%** | 1.6M |
| ViT-CIFAR | 90.92% | 66.54% | 6.3M |
| Wave Vision 2D | 80.41% | 52.81% | 1.4M |

The hybrid combines CNN local feature extraction with Wave's efficient global processing.

### Audio Classification

Wave Network with STFT input achieves **92.9% accuracy** on Speech Commands (35 keywords):

| Model | Val Acc | Test Acc | Params | Inference |
|-------|---------|----------|--------|-----------|
| Wave-STFT + SpecAugment | **92.9%** | **92.6%** | 9.1M | 17ms (CPU) |
| Wave-STFT Tiny | 86.9% | 85.5% | 584K | 2.4ms (CPU) |

The architecture processes magnitude and phase separately—a natural fit for wave-based operations. ONNX export included for deployment.

```bash
# Train (full model)
python train_audio.py --representation stft --specaugment --epochs 100

# Train (tiny model - 5x faster inference)
python train_audio.py --representation stft --embedding-dim 128 --num-layers 3

# Export to ONNX
python export_onnx.py --checkpoint data/checkpoints/best.pt

# Inference
python infer.py --top 3 recording.wav
```

#### Live Web Demo

**[Try it live](https://kevinraymond.github.io/wave-network/)** - runs entirely in your browser with microphone input.

Or run locally:

```bash
cd docs && python -m http.server 8000
# Open http://localhost:8000
```

The demo runs entirely client-side using ONNX Runtime Web (WASM). Features:
- Real-time keyword detection (~50ms latency with tiny model)
- Lightweight model (2.4MB)
- No server required—works offline

See [notes/benchmarks.md](notes/benchmarks.md) and [notes/vision_results.md](notes/vision_results.md) for detailed results.

## Quick Start

```bash
# Install dependencies
uv sync

# Train on AG News
uv run python train.py

# Run GLUE benchmarks
uv run python train_glue.py --task sst2 --model wave_network

# Train CNN-Wave on CIFAR-10
uv run python train_vision.py --task cifar10 --model cnn_wave --randaugment

# Train CNN-Wave on CIFAR-100
uv run python train_vision.py --task cifar100 --model cnn_wave --randaugment
```

## Documentation

- [Benchmark Results](notes/benchmarks.md) - Detailed performance data
- [Vision Results](notes/vision_results.md) - Image classification benchmarks
- [Technical Analysis](notes/ANALYSIS.md) - Implementation review
- [Improvements](notes/IMPROVEMENTS.md) - Roadmap and fixes
- [Review Summary](notes/REVIEW_SUMMARY.md) - Executive summary

## Development

```bash
# Install dev dependencies
uv sync --dev

# Set up pre-commit hooks
pre-commit install
```

## Changelog

### v0.2.0
- **Fixed**: Added positional encoding to `WaveNetwork` and `DeepWaveNetwork` text models. The initial implementation was missing this component shown in Figure 6a of the paper, which meant the models treated input as a bag-of-words (word order was ignored). This primarily affects tasks where word order matters (e.g., NLI, QA).

## License

MIT
