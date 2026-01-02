# Wave Network

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

See [docs/benchmarks.md](docs/benchmarks.md) and [docs/vision_results.md](docs/vision_results.md) for detailed results.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train on AG News
python train.py

# Run GLUE benchmarks
python train_glue.py --task sst2 --model wave_network

# Train CNN-Wave on CIFAR-10
python train_vision.py --task cifar10 --model cnn_wave --randaugment

# Train CNN-Wave on CIFAR-100
python train_vision.py --task cifar100 --model cnn_wave --randaugment
```

## Documentation

- [Benchmark Results](docs/benchmarks.md) - Detailed performance data
- [Vision Results](docs/vision_results.md) - Image classification benchmarks
- [Technical Analysis](docs/ANALYSIS.md) - Implementation review
- [Improvements](docs/IMPROVEMENTS.md) - Roadmap and fixes
- [Review Summary](docs/REVIEW_SUMMARY.md) - Executive summary

## License

MIT
