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


class Trainer:
    def __init__(self, model, data, config):
        """
        Initialize trainer

        Args:
            model: Wave Network or BERT model
            data: Dictionary containing dataloaders and tokenizer
            config: Training configuration
        """
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
            name=self.config["run_name"],
            config=self.config,
            group="dbpedia_comparison",
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


def main():
    from prepare_dataset import prepare_text_classification_data
    from wave_network import WaveNetwork

    # Base configuration
    base_config = {
        "project_name": "wave-network-vs-bert-dbpedia",
        "weight_decay": 0.01,
        "num_epochs": 4,
        "warmup_steps": 100,
        "max_grad_norm": 1.0,
        "max_length": 128,
    }

    # Prepare data with two different batch sizes
    data_wave = prepare_text_classification_data(
        train_path="hf/dbpedia_14/dbpedia_14/train-00000-of-00001.parquet",
        test_path="hf/dbpedia_14/dbpedia_14/test-00000-of-00001.parquet",
        batch_size=64,  # Wave Network batch size
        max_length=base_config["max_length"],
        text_column="content",
    )

    data_bert = prepare_text_classification_data(
        train_path="hf/dbpedia_14/dbpedia_14/train-00000-of-00001.parquet",
        test_path="hf/dbpedia_14/dbpedia_14/test-00000-of-00001.parquet",
        batch_size=32,  # BERT batch size
        max_length=base_config["max_length"],
        text_column="content",
    )

    results = {
        "wave_network": {"metrics": None, "resources": None},
        "bert": {"metrics": None, "resources": None},
    }

    # Initialize wandb run for the experiment group
    wandb.init(project=base_config["project_name"])

    try:
        # Train Wave Network
        wave_config = base_config.copy()
        wave_config.update(
            {
                "run_name": "wave_network",
                "learning_rate": 1e-3,  # Wave Network learning rate
                "batch_size": 64,
            }
        )

        wave_model = WaveNetwork(
            vocab_size=data_wave["vocab_size"],
            embedding_dim=768,
            num_classes=data_wave["num_classes"],
        )
        wave_trainer = Trainer(wave_model, data_wave, wave_config)
        wave_metrics = wave_trainer.train()
        results["wave_network"]["metrics"] = wave_metrics
        results["wave_network"]["resources"] = {
            "parameters": sum(p.numel() for p in wave_model.parameters()),
            "memory_peak": (
                torch.cuda.max_memory_allocated() / 1024 / 1024
                if torch.cuda.is_available()
                else 0
            ),
        }
        wandb.finish()
        torch.cuda.reset_peak_memory_stats()  # Reset memory stats between runs

        # Train BERT baseline
        bert_config = base_config.copy()
        bert_config.update(
            {
                "run_name": "bert_baseline",
                "learning_rate": 2e-5,  # BERT learning rate
                "batch_size": 32,
            }
        )

        bert_model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=data_bert["num_classes"]
        )
        bert_trainer = Trainer(bert_model, data_bert, bert_config)
        bert_metrics = bert_trainer.train()
        results["bert"]["metrics"] = bert_metrics
        results["bert"]["resources"] = {
            "parameters": sum(p.numel() for p in bert_model.parameters()),
            "memory_peak": (
                torch.cuda.max_memory_allocated() / 1024 / 1024
                if torch.cuda.is_available()
                else 0
            ),
        }
        wandb.finish()

        # Log comparison metrics
        wandb.init(
            project=base_config["project_name"],
            name="model_comparison",
            job_type="analysis",
        )

        # Create comparison table
        comparison_table = wandb.Table(columns=["Metric", "Wave Network", "BERT"])

        # Performance metrics
        metrics = ["accuracy", "precision", "recall", "f1", "loss"]
        for metric in metrics:
            comparison_table.add_data(
                metric,
                results["wave_network"]["metrics"][metric],
                results["bert"]["metrics"][metric],
            )

        # Resource metrics
        comparison_table.add_data(
            "Parameters",
            results["wave_network"]["resources"]["parameters"],
            results["bert"]["resources"]["parameters"],
        )
        comparison_table.add_data(
            "Peak Memory (MB)",
            results["wave_network"]["resources"]["memory_peak"],
            results["bert"]["resources"]["memory_peak"],
        )

        wandb.log(
            {
                "model_comparison": comparison_table,
                "config_comparison": {
                    "wave_network": wave_config,
                    "bert": bert_config,
                },
            }
        )

        # Print comparison
        print("\nFinal Test Results:")
        print("Wave Network (batch_size=64):")
        print(f"Performance Metrics:", results["wave_network"]["metrics"])
        print(f"Resource Usage:", results["wave_network"]["resources"])
        print("\nBERT (batch_size=32):")
        print(f"Performance Metrics:", results["bert"]["metrics"])
        print(f"Resource Usage:", results["bert"]["resources"])

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
