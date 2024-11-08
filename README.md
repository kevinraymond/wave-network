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

**Latest Run:**

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

**Best Run:**

SAME AS LATEST

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

**Latest Run:**

| Metric    | Wave Network | BERT base |
| --------- | ------------ | --------- |
| Accuracy  | 98.15%       | 99.26%    |
| Precision | 98.16%       | 99.26%    |
| Recall    | 98.15%       | 99.26%    |
| F1 Score  | 98.15%       | 99.26%    |
| Loss      | 0.0630       | 0.0454    |

```bash
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

```bash
# don't have the console output; started keeping it after this best run
```

### IMDB

_Still requires tuning - memory usage is much higher! This is using the last updated [train.py](train.py) script. Clarification: I have only been running the Wave tests, not re-running BERT; this only shows the last Wave update, so I will update them all at some point to be on the same page._

**Latest Run:**

| Metric    | Wave Network | BERT base |
| --------- | ------------ | --------- |
| Accuracy  | 87.22%       | 88.69%    |
| Precision | 87.26%       | 88.69%    |
| Recall    | 87.22%       | 88.69%    |
| F1 Score  | 87.21%       | 88.69%    |
| Loss      | 0.3014       | 0.2818    |

```bash
Final Test Results:
WAVE_NETWORK (batch_size=64):
Performance Metrics: {'loss': 0.3014372759653479, 'accuracy': 0.87216, 'precision': np.float64(0.8725766277036728), 'recall': np.float64(0.87216), 'f1': np.float64(0.8721242512602836)}
Resource Usage: {'parameters': 24625154, 'memory_peak': 2892.970703125}

BERT (batch_size=32):
Performance Metrics: {'loss': 0.28180405385720797, 'accuracy': 0.88688, 'precision': np.float64(0.8868966495556584), 'recall': np.float64(0.88688), 'f1': np.float64(0.8868787829966989)}
Resource Usage: {'parameters': 109483778, 'memory_peak': 4163.06396484375}
```

**Best Run:**

SAME AS LATEST

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

Protecting sanity at all costs! I made a single [train.py](train.py) script with all the config at the top. It could still be improved, but one is better than three in this case.

**NOTE**: This does have Weights & Biases; if that breaks everything for you, comment out all the `wandb.*` stuff.

```bash
# update config as needed, then
python train.py
```
