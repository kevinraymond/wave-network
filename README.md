# Wave Network

Implementation of [Wave Network: An Ultra-Small Language Model](https://arxiv.org/pdf/2411.02674) - a lightweight alternative to BERT using wave-based signal processing instead of attention.

## Results

Wave Network achieves near-BERT accuracy with **4.5x fewer parameters**:

| Dataset | Wave Network | BERT | Parameters |
|---------|-------------|------|------------|
| AG News | 92.0% | 94.6% | 24.6M vs 109M |
| DBpedia | 98.2% | 99.3% | 24.6M vs 109M |
| IMDB | 87.2% | 88.7% | 24.6M vs 109M |

On GLUE benchmark, Wave Network wins 5/8 tasks vs FNet with half the parameters (24.6M vs 52.2M).

See [docs/benchmarks.md](docs/benchmarks.md) for detailed results.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train on AG News
python train.py

# Run GLUE benchmarks
python train_glue.py --task sst2 --model wave_network
```

## Documentation

- [Benchmark Results](docs/benchmarks.md) - Detailed performance data
- [Technical Analysis](docs/ANALYSIS.md) - Implementation review
- [Improvements](docs/IMPROVEMENTS.md) - Roadmap and fixes
- [Review Summary](docs/REVIEW_SUMMARY.md) - Executive summary

## License

MIT
