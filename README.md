# Wave Network

I read the [Wave Network: An Ultra-Small Language Model](https://arxiv.org/pdf/2411.02674) paper and it sounds promising - here's a quick proof of life implementation based on the paper, for Wave Network versus BERT.

Not going to give tons of step-by-step; if you're here looking, you probably have an idea how to get your system running.

Someone smarter than me can check [wave_network.py](wave_network.py) to make sure it's sound.

Making some improvements along the way. Hopefully this leads somewhere amazing - enjoy!

## WIP Updates

2024-11-08: Update Wave Network implementation with some additional focus on the signal parts. Tried to keep the comments in there clear. Last update for tonight (2 AM local) - more tomorrow.

## Results

These tests are all run locally on a single Nvidia 4090. Here are the latest/best results of everything.

### AG News

**Best Run:**

| Metric    | Wave Network | BERT base |
| --------- | ------------ | --------- |
| Accuracy  | 92.03%       | 94.63%    |
| Precision | 92.04%       | 94.66%    |
| Recall    | 92.03%       | 94.63%    |
| F1 Score  | 92.03%       | 94.63%    |
| Loss      | 0.2378       | 0.2022    |

```bash
Final Test Results:
Wave Network (batch_size=64):
Performance Metrics: {'loss': 0.23779155445449493, 'accuracy': 0.9202631578947369, 'precision': np.float64(0.9204283202989598), 'recall': np.float64(0.9202631578947369), 'f1': np.float64(0.9202632926679699)}
Resource Usage: {'parameters': 24626692, 'memory_peak': 1165.90478515625}

BERT (batch_size=32):
Performance Metrics: {'loss': 0.20216417188576163, 'accuracy': 0.9463157894736842, 'precision': np.float64(0.9466198771601457), 'recall': np.float64(0.9463157894736842), 'f1': np.float64(0.9463277574455472)}
Resource Usage: {'parameters': 109485316, 'memory_peak': 4170.14404296875}
```

I asked Claude to compare the results of the **first run** I did with AG News against the paper:

> Key observations:
> \
> The results closely match the paper's claims! The Wave Network achieved 91.64% accuracy compared to BERT's 94.53%, which is remarkably close considering it uses only 2.4M parameters versus BERT's 100M parameters.\
> \
> The consistent metrics across precision, recall, and F1 for both models suggest they're performing uniformly well across all classes (no class imbalance issues).
> BERT's lower loss (0.1778 vs 0.2495) indicates it's more confident in its correct predictions, which makes sense given its larger size and pre-training.\
> \
> The results are quite exciting as they validate the paper's claims about achieving near-BERT performance with a much smaller model!

### DBpedia14

_Update pending a new run with latest Wave Network changes._

**Best Run:**

| Metric    | Wave Network | BERT base |
| --------- | ------------ | --------- |
| Accuracy  | 98.34%       | 99.30%    |
| Precision | 98.34%       | 99.30%    |
| Recall    | 98.34%       | 99.30%    |
| F1 Score  | 98.34%       | 99.30%    |
| Loss      | 0.0596       | 0.0370    |

```bash
# PENDING
```

### IMDB

_Update pending a new run with latest Wave Network changes. Something else happening here - just breaking 81% so far._

** Best Run:**

| Metric    | Wave Network | BERT base |
| --------- | ------------ | --------- |
| Accuracy  | 00.00%       | 00.00%    |
| Precision | 00.00%       | 00.00%    |
| Recall    | 00.00%       | 00.00%    |
| F1 Score  | 00.00%       | 00.00%    |
| Loss      | 0.0000       | 0.0000    |

```bash
# PENDING
```

## Usage

### Datasets

[AG News](https://huggingface.co/datasets/fancyzhx/ag_news)

[DBpedia14](https://huggingface.co/datasets/fancyzhx/dbpedia_14)

[IMDB](https://huggingface.co/datasets/stanfordnlp/imdb)

Assuming you have HF CLI installed and execute this in the repo root:

```bash
huggingface-cli download fancyzhx/ag_news --repo-type dataset --local-dir hf/ag_news
huggingface-cli download fancyzhx/dbpedia_14 --repo-type dataset --local-dir hf/dbpedia_14
huggingface-cli download stanfordnlp/imdb --repo-type dataset --local-dir hf/imdb
```

### Install

I'm running PyTorch 2.5.1 and CUDA 12.4 - YMMV.

```bash
pip install -r requirements.txt
```

### Train

All the `train*` files are the same besides having the datasets changed up. Laziness wins.

**NOTE**: This does have Weights & Biases; if that breaks everything for you, comment out all the `wandb.*` stuff.

```bash
python train_ag_news.py
python train_dbpedia.py
python train_imdb.py
```
