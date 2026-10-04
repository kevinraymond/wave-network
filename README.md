# Wave Network

[![CI](https://github.com/kevinraymond/wave-network/actions/workflows/ci.yml/badge.svg)](https://github.com/kevinraymond/wave-network/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kevinraymond/wave-network/branch/main/graph/badge.svg)](https://codecov.io/gh/kevinraymond/wave-network)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Implementation of [Wave Network: An Ultra-Small Language Model](https://arxiv.org/pdf/2411.02674) - a lightweight alternative to BERT using wave-based signal processing instead of attention.

> **Status: archived (October 2026).** This repository is a completed replication study and is
> no longer developed. The main finding is negative: on the text tasks tested, the wave
> operation adds almost nothing over a plain bag-of-embeddings model with the same embedding
> table, and the vision wave layers reduce to energy pooling. See [Findings](#findings) below.
> The code, tests and the browser keyword-spotting demo remain available as-is.

## Findings

1. **Text: the embeddings do the work.** With the wave operation removed, accuracy drops by
   0.1-0.4 points on AG News and not at all on SST-2 (3 seeds each). The paper's AG News result
   is reproduced (91.4-91.7% here vs 91.66% reported), but a mean-pooled embedding baseline
   reaches 91.3%.
2. **`abs` after modulation is degenerate.** `|c1*c2| = G1*G2` is identical for every token, so
   the original implementation never passed per-token features to the classifier. Fixing this
   (v0.3.0, `output_mode="real_imag"`) gained about 0.3 points on AG News and nothing on SST-2.
3. **Vision: the wave layers are energy pooling.** The published CNN-Wave numbers use the same
   degenerate mode; each wave scale is a sum of squared features over a row, column or 3x3
   neighborhood. The stem-only control was implemented but never run.
4. **Audio was not ablated.** The STFT model uses a different operation and has no control run.
5. **Evaluation caveats.** The vision numbers were selected on the test set (best epoch), the
   GLUE numbers are best-epoch on the validation set, and the FNet baseline was trained from
   scratch.

## Results

### Text Classification

Wave Network gets within about 3 points of BERT with roughly 4x fewer parameters:

| Dataset | Wave Network | BERT | Parameters |
|---------|-------------|------|------------|
| AG News | 91.7% | 94.6% | 26.2M vs 109M |
| DBpedia | 98.2%* | 99.3% | 24.6M vs 109M |
| IMDB | 87.2%* | 88.7% | 24.6M vs 109M |

*Test accuracy on standard splits. AG News is the v0.3.0 model (mean of 3 seeds). Rows marked
\* are single pre-v0.3.0 runs (`output_mode="abs"`) and have not been re-run.*

**Most of that accuracy comes from the embeddings, not the wave operation.** A bag-of-embeddings
control (`WaveNetwork` with the wave operation removed: embedding, layer norm, mean pooling,
linear classifier) lands within half a point on every task re-run so far:

| Model | Params | AG News (test) | SST-2 (val) | RTE (val) |
|-------|--------|----------------|-------------|-----------|
| Bag of embeddings (control) | 23.4M | 91.29% | 80.24% | 51.38% |
| Wave Network, `abs` (pre-v0.3.0) | 25.0M | 91.37% | 80.73% | 51.99% |
| Wave Network, `real` (paper) | 25.0M | 91.39% | 80.58% | 53.91% |
| Wave Network, `real_imag` (default) | 26.2M | **91.68%** | 80.58% | 53.31% |

*Mean of 3 seeds (42, 123, 456), modulation mode. AG News: 4 epochs, test accuracy at the
best-validation epoch. SST-2 and RTE: best validation epoch, so slightly optimistic.*

- On AG News `real_imag` beats the control on all three seeds, by 0.4 points on average.
- On SST-2 the four models are indistinguishable (seed-to-seed spread is 0.6-0.9 points).
- On RTE everything is near the 52.7% majority-class baseline (277 validation examples).
- The paper's "2.4M parameters" excludes the ~23.4M token embedding table; the counts here
  include it.

On the full GLUE benchmark (pre-v0.3.0 runs), Wave Network scored higher than a from-scratch
FNet on 5/8 tasks with half the parameters (24.6M vs 52.2M). Neither model was pre-trained and
FNet was at chance on several tasks, so this is a weak comparison; see
[notes/benchmarks.md](notes/benchmarks.md).

To reproduce the comparison:

```bash
uv run python train.py --dataset ag_news --models wave_network --output-mode real_imag
uv run python train.py --dataset ag_news --models bag_of_embeddings
uv run --extra glue python train_glue.py --task sst2 --model wave_network bag_of_embeddings
```

### Image Classification

CNN-Wave hybrid outperforms ViT with **4x fewer parameters**:

| Model | CIFAR-10 | CIFAR-100 | Parameters |
|-------|----------|-----------|------------|
| CNN-Wave (ours) | **92.72%** | **71.93%** | 1.6M |
| ViT-CIFAR | 90.92% | 66.54% | 6.3M |
| Wave Vision 2D | 80.41% | 52.81% | 1.4M |

*Test accuracy on standard splits.*

The hybrid combines CNN local feature extraction with Wave's efficient global processing.

> **Note on these numbers:** they use `output_mode="abs"` with modulation, where each
> wave scale reduces to energy pooling (the sum of squared features over a row, column,
> or 3x3 neighborhood) and the phase is unused. Comparisons against the stem-only control
> (`--model cnn_stem`) and the phase-preserving modes (`--output-mode real` / `real_imag`)
> were implemented but not run before the project was archived. These numbers are also the
> best epoch as measured on the test set, and plain ResNets of similar size reach comparable
> CIFAR-10 accuracy, so the comparison with ViT overstates the result.

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
uv run python train.py --dataset ag_news

# Run GLUE benchmarks
uv run --extra glue python train_glue.py --task sst2 --model wave_network

# Train CNN-Wave on CIFAR-10
uv run python train_vision.py --task cifar10 --model cnn_wave --randaugment

# Train CNN-Wave on CIFAR-100
uv run python train_vision.py --task cifar100 --model cnn_wave --randaugment
```

## Documentation

- [Benchmark Results](notes/benchmarks.md) - Detailed performance data, including the final re-run
- [Vision Results](notes/vision_results.md) - Image classification benchmarks

Historical notes, written before the findings above and superseded by them:

- [Technical Analysis](notes/ANALYSIS.md) - Implementation review (November 2025)
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

### v0.3.0
- **Fixed**: Complex-to-real conversion in the text models (`WaveNetwork`, `DeepWaveNetwork`).
  The previous `torch.abs()` conversion is degenerate under modulation (the default mode):
  since both operands have magnitude `G_k` for *every* token, `|c1*c2| = G1*G2` is identical
  at every sequence position — per-token features never reached the classifier, only the
  norm statistics `G1*G2` (which still encode a quadratic bag of token/position energies,
  which is why the old models worked at all). The new default `output_mode="real_imag"`
  concatenates real and
  imaginary parts and projects back down (the same pattern as the audio `WaveLayerSTFT`,
  the best-performing model in this repo). `output_mode="abs"` reproduces the old behavior;
  `output_mode="real"` is the paper's local-semantics reading (but is linear under
  interference). See `models/wave_ops.py` for the math.
- **Fixed**: `sqrt(0)` infinite gradient in the phase computation when a single position
  carries a dimension's entire norm — the likely cause of the NaN issues previously noted
  in `notes/benchmarks.md`. All variants now use the eps-clamped math in `models/wave_ops.py`.
- **Fixed**: Padding positions no longer leak into the global semantics norms via the
  projection biases (masking is now applied after `linear1`/`linear2`).
- **Changed**: Shared wave math consolidated into `models/wave_ops.py` (was duplicated,
  with drift, across six files).
- **Added**: `output_mode` for the vision models (`WaveVisionNetwork`, `WaveVisionNetwork2D`,
  `CNNWaveVision`) and `--output-mode` for `train.py`, `train_glue.py` and `train_vision.py`.
  The vision default stays `"abs"` so published results and checkpoints reproduce; under
  modulation that mode has the same degeneracy as the text models had (see the note under
  Image Classification).
- **Added**: Controls for measuring what the wave operation contributes: `bag_of_embeddings`
  (`WaveNetwork` without the wave operation; `train.py --models`, `train_glue.py --model`)
  and `cnn_stem` (`CNNWaveVision` without wave layers; `train_vision.py --model`).
- **Changed**: The wave math no longer uses trigonometry (`Z = x + i*sqrt(G^2 - x^2)`), and
  the stacked layers compute it on real tensors without the complex dtype. Outputs are
  unchanged to float precision.
- **Changed**: `train_audio.py` evaluates the test set once, after training, on the
  val-selected checkpoint (was: every epoch). `train.py` and `train_glue.py` gained
  `--seed` flags.

### v0.2.0
- **Fixed**: Added positional encoding to `WaveNetwork` and `DeepWaveNetwork` text models. The initial implementation was missing this component shown in Figure 6a of the paper, which meant the models treated input as a bag-of-words (word order was ignored). This primarily affects tasks where word order matters (e.g., NLI, QA).

## License

MIT
