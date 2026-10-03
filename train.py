import argparse
import os
import random
import time

import numpy as np
import psutil
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup

from models.bag_of_embeddings import BagOfEmbeddings
from prepare_dataset import prepare_text_classification_data
from wave_network import WaveNetwork

# Configuration settings
MODEL_CONFIGS = {
    "wave_network": {
        "name": "wave_network",
        "learning_rate": 1e-3,
        "batch_size": 64,
        "model_params": {
            "embedding_dim": 768,
        },
    },
    # Control: WaveNetwork without the wave operation
    "bag_of_embeddings": {
        "name": "bag_of_embeddings",
        "learning_rate": 1e-3,
        "batch_size": 64,
        "model_params": {
            "embedding_dim": 768,
        },
    },
    "bert": {
        "name": "bert_baseline",
        "learning_rate": 2e-5,
        "batch_size": 32,
        "model_params": {
            "pretrained_model": "bert-base-uncased",
        },
    },
}

BASE_CONFIG = {
    "project_name": "wave-network-vs-bert",
    "num_epochs": 4,
    "warmup_steps": 100,
}

# Per-dataset paths and hyperparameters
DATASET_CONFIGS = {
    "imdb": {
        "train": "hf/imdb/plain_text/train-00000-of-00001.parquet",
        "test": "hf/imdb/plain_text/test-00000-of-00001.parquet",
        "weight_decay": 1e-5,
        "max_grad_norm": 5.0,
        "max_length": 384,
    },
    "ag_news": {
        "train": "hf/ag_news/data/train-00000-of-00001.parquet",
        "test": "hf/ag_news/data/test-00000-of-00001.parquet",
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "max_length": 128,
    },
}


def get_model(model_type, vocab_size, num_classes, output_mode=None):
    """Factory function to create models"""
    if model_type == "wave_network":
        kwargs = {} if output_mode is None else {"output_mode": output_mode}
        return WaveNetwork(
            vocab_size=vocab_size,
            embedding_dim=MODEL_CONFIGS[model_type]["model_params"]["embedding_dim"],
            num_classes=num_classes,
            **kwargs,
        )
    elif model_type == "bag_of_embeddings":
        return BagOfEmbeddings(
            vocab_size=vocab_size,
            embedding_dim=MODEL_CONFIGS[model_type]["model_params"]["embedding_dim"],
            num_classes=num_classes,
        )
    elif model_type == "bert":
        return BertForSequenceClassification.from_pretrained(
            MODEL_CONFIGS[model_type]["model_params"]["pretrained_model"],
            num_labels=num_classes,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_config(model_type, dataset="imdb"):
    """Get configuration for specified model type and dataset"""
    config = BASE_CONFIG.copy()
    config.update(DATASET_CONFIGS[dataset])
    config.update(MODEL_CONFIGS[model_type])
    config["dataset"] = dataset
    return config


def prepare_data(model_type, dataset="imdb"):
    """Prepare data for specified model type and dataset"""
    return prepare_text_classification_data(
        train_path=DATASET_CONFIGS[dataset]["train"],
        test_path=DATASET_CONFIGS[dataset]["test"],
        batch_size=MODEL_CONFIGS[model_type]["batch_size"],
        max_length=DATASET_CONFIGS[dataset]["max_length"],
    )


class Trainer:
    def __init__(self, model, data, config):
        """Initialize trainer"""
        self.model = model
        self.data = data
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model.to(self.device)

        # Set up optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        # Set up learning rate scheduler
        num_training_steps = len(data["train_loader"]) * config["num_epochs"]
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config["warmup_steps"],
            num_training_steps=num_training_steps,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Initialize best metrics
        self.best_val_accuracy = 0
        self.best_model_path = None

    def check_gradients(self):
        """Check for NaN/Inf in gradients"""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"WARNING: NaN gradient in {name}")
                    return False
                if torch.isinf(param.grad).any():
                    print(f"WARNING: Inf gradient in {name}")
                    return False
        return True

    def train_epoch(self, epoch):
        """Run one epoch of training"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()

        progress_bar = tqdm(self.data["train_loader"], desc=f"Epoch {epoch}")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            if isinstance(self.model, BertForSequenceClassification):
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                logits = outputs.logits
            else:
                logits = self.model(input_ids, attention_mask=attention_mask)
                loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Check gradients
            if not self.check_gradients():
                print("Skipping update due to invalid gradients")
                self.optimizer.zero_grad()
                continue

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])

            # Update weights
            self.optimizer.step()
            self.scheduler.step()

            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})

        # Calculate epoch metrics
        epoch_loss = total_loss / len(self.data["train_loader"])
        epoch_accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted"
        )

        # Calculate resource usage
        end_mem = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = end_mem - start_mem
        time_taken = time.time() - start_time

        return {
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "memory_used": memory_used,
            "time_taken": time_taken,
        }

    def evaluate(self, dataloader):
        """Evaluate model on given dataloader"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                if isinstance(self.model, BertForSequenceClassification):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                    logits = outputs.logits
                else:
                    logits = self.model(input_ids, attention_mask=attention_mask)
                    loss = self.criterion(logits, labels)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted"
        )

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def train(self):
        """Complete training loop"""
        # Initialize wandb
        wandb.init(
            project=self.config["project_name"],
            name=self.config["name"],
            config=self.config,
            group=f"{self.config['dataset']}_comparison",
            reinit=True,
        )

        for epoch in range(self.config["num_epochs"]):
            # Training phase
            train_metrics = self.train_epoch(epoch)

            # Validation phase
            val_metrics = self.evaluate(self.data["val_loader"])

            # Log metrics
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "train_f1": train_metrics["f1"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1": val_metrics["f1"],
                    "memory_used_mb": train_metrics["memory_used"],
                    "epoch_time_seconds": train_metrics["time_taken"],
                }
            )

            # Save best model
            if val_metrics["accuracy"] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics["accuracy"]
                if self.best_model_path:
                    os.remove(self.best_model_path)
                self.best_model_path = f"models/model_{epoch}_{val_metrics['accuracy']:.4f}.pt"
                torch.save(self.model.state_dict(), self.best_model_path)

        # Final test evaluation
        self.model.load_state_dict(torch.load(self.best_model_path, weights_only=True))
        test_metrics = self.evaluate(self.data["test_loader"])

        wandb.log(
            {
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
            }
        )

        return test_metrics


def train_model(model_type, output_mode=None, dataset="imdb"):
    """Train and evaluate a specific model type"""
    # Prepare data and config
    data = prepare_data(model_type, dataset)
    config = get_config(model_type, dataset)
    if output_mode is not None and model_type == "wave_network":
        config["name"] = f"{config['name']}_{output_mode}"
        config["output_mode"] = output_mode

    # Initialize model
    model = get_model(model_type, data["vocab_size"], data["num_classes"], output_mode)

    # Train model
    trainer = Trainer(model, data, config)
    metrics = trainer.train()

    # Calculate resource usage
    resources = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "memory_peak": (
            torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        ),
    }

    return {"metrics": metrics, "resources": resources}


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train Wave Network vs BERT baselines")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dataset", default="imdb", choices=list(DATASET_CONFIGS), help="Dataset to train on"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["wave_network"],
        choices=list(MODEL_CONFIGS),
        help="Model(s) to train",
    )
    parser.add_argument(
        "--output-mode",
        default=None,
        choices=["abs", "real", "real_imag"],
        help="Complex-to-real conversion for wave models: abs (pre-v0.3.0), real, or real_imag (default: model default)",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    results = {}

    # Initialize wandb
    wandb.init(project=BASE_CONFIG["project_name"], config={"seed": args.seed})

    try:
        for model_type in args.models:
            results[model_type] = train_model(model_type, args.output_mode, args.dataset)
            wandb.finish()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        # Print results
        print("\nFinal Test Results:")
        for model_type, result in results.items():
            print(f"\n{model_type.upper()} (batch_size={MODEL_CONFIGS[model_type]['batch_size']}):")
            print("Performance Metrics:", result["metrics"])
            print("Resource Usage:", result["resources"])

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
