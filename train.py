import numpy as np
import os
import psutil
import time
import torch
import torch.nn as nn
import wandb

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from prepare_dataset import prepare_text_classification_data
from wave_network import WaveNetwork
from memory_efficient_wave_network import MemoryEfficientWaveNetwork

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
    # "weight_decay": 1e-3, # orig
    "weight_decay": 1e-5,
    "num_epochs": 4, # orig
    "warmup_steps": 100, # orig
    # "max_grad_norm": 1.0, # orig
    "max_grad_norm": 5.0,
    # "max_length": 64, # orig
    "max_length": 384,
}

DATA_PATHS = {
    "train": "hf/imdb/plain_text/train-00000-of-00001.parquet",
    "test": "hf/imdb/plain_text/test-00000-of-00001.parquet",
}


def get_model(model_type, vocab_size, num_classes):
    """Factory function to create models"""
    if model_type == "wave_network":
        return WaveNetwork(
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


def get_config(model_type):
    """Get configuration for specified model type"""
    config = BASE_CONFIG.copy()
    config.update(MODEL_CONFIGS[model_type])
    return config


def prepare_data(model_type):
    """Prepare data for specified model type"""
    return prepare_text_classification_data(
        train_path=DATA_PATHS["train"],
        test_path=DATA_PATHS["test"],
        batch_size=MODEL_CONFIGS[model_type]["batch_size"],
        max_length=BASE_CONFIG["max_length"],
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
                logits = self.model(input_ids)
                loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config["max_grad_norm"]
            )

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
                    logits = self.model(input_ids)
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
            group="imdb_comparison",
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
                self.best_model_path = (
                    f"models/model_{epoch}_{val_metrics['accuracy']:.4f}.pt"
                )
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


def train_model(model_type):
    """Train and evaluate a specific model type"""
    # Prepare data and config
    data = prepare_data(model_type)
    config = get_config(model_type)

    # Initialize model
    model = get_model(model_type, data["vocab_size"], data["num_classes"])

    # Train model
    trainer = Trainer(model, data, config)
    metrics = trainer.train()

    # Calculate resource usage
    resources = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "memory_peak": (
            torch.cuda.max_memory_allocated() / 1024 / 1024
            if torch.cuda.is_available()
            else 0
        ),
    }

    return {"metrics": metrics, "resources": resources}


def main():
    results = {}

    # Initialize wandb
    wandb.init(project=BASE_CONFIG["project_name"])

    try:
        # Train Wave Network
        results["wave_network"] = train_model("wave_network")
        wandb.finish()
        torch.cuda.reset_peak_memory_stats()

        # Train BERT
        # results["bert"] = train_model("bert")
        # wandb.finish()

        # Print results
        print("\nFinal Test Results:")
        for model_type, result in results.items():
            print(
                f"\n{model_type.upper()} (batch_size={MODEL_CONFIGS[model_type]['batch_size']}):"
            )
            print(f"Performance Metrics:", result["metrics"])
            print(f"Resource Usage:", result["resources"])

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
