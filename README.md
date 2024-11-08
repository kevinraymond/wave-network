# Wave Network

I read the [Wave Network: An Ultra-Small Language Model](https://arxiv.org/pdf/2411.02674) paper and it sounds promising - here's a quick proof of life implementation based on the paper, for Wave Network versus BERT.

Not going to give tons of step-by-step; if you're here looking, you probably have an idea how to get your system running.

Someone smarter than me can check [wave_network.py](wave_network.py) to make sure it's sound.

Hopefully this leads somewhere amazing - enjoy!

## Results

These tests are all run locally on a single Nvidia 4090. Here are the results of the first runs of each dataset.

### AG News

| Metric    | Wave Network | BERT base |
| --------- | ------------ | --------- |
| Accuracy  | 91.64%       | 94.53%    |
| Precision | 91.63%       | 94.58%    |
| Recall    | 91.64%       | 94.53%    |
| F1 Score  | 91.64%       | 94.53%    |
| Loss      | 0.2495       | 0.1778    |

I asked Claude to compare the results of this one against the paper:

> Key observations:
> \
> The results closely match the paper's claims! The Wave Network achieved 91.64% accuracy compared to BERT's 94.53%, which is remarkably close considering it uses only 2.4M parameters versus BERT's 100M parameters.\
> \
> The consistent metrics across precision, recall, and F1 for both models suggest they're performing uniformly well across all classes (no class imbalance issues).
> BERT's lower loss (0.1778 vs 0.2495) indicates it's more confident in its correct predictions, which makes sense given its larger size and pre-training.\
> \
> The results are quite exciting as they validate the paper's claims about achieving near-BERT performance with a much smaller model!

### DBpedia14

IT'S RUNNING ...

### IMDB

IT'S RUNNING ...

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

# Usage

All the `train*` files are the same besides having the datasets changed up. Laziness wins.

**NOTE**: This does have Weights & Biases; if that breaks everything for you, comment out all the `wandb.*` stuff.

```bash
python train_ag_news.py
python train_dbpedia.py
python train_imdb.py
```
