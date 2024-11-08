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


def main():
    from prepare_dataset import prepare_text_classification_data
    from wave_network import WaveNetwork

    # Configuration
    config = {
        "project_name": "wave-network-vs-bert",
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "num_epochs": 4,
        "warmup_steps": 100,
        "max_grad_norm": 1.0,
        "batch_size": 64,
        "max_length": 128,
    }

    # prepare data
    data = prepare_text_classification_data(
        train_path="hf/imdb/data/train-00000-of-00001.parquet",
        test_path="hf/imdb/data/test-00000-of-00001.parquet",
        batch_size=config["batch_size"],
        max_length=config["max_length"],
    )

    wandb.init(project=config["project_name"])
    results = {}

    try:
        # Train Wave Network
        config["run_name"] = "wave_network"
        wave_model = WaveNetwork(
            vocab_size=data["vocab_size"],
            embedding_dim=768,
            num_classes=data["num_classes"],
        )
        wave_trainer = Trainer(wave_model, data, config)
        results["wave_network"] = wave_trainer.train()
        wandb.finish()  # Finish the Wave Network run

        # Train BERT baseline
        config["run_name"] = "bert_baseline"
        config["learning_rate"] = 2e-5  # BERT typically needs lower learning rate
        bert_model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=data["num_classes"]
        )
        bert_trainer = Trainer(bert_model, data, config)
        results["bert"] = bert_trainer.train()
        wandb.finish()  # Finish the BERT run

        # Log comparison metrics
        wandb.init(
            project=config["project_name"], name="model_comparison", job_type="analysis"
        )

        # Create comparison table
        comparison_table = wandb.Table(columns=["Metric", "Wave Network", "BERT"])

        metrics = ["accuracy", "precision", "recall", "f1", "loss"]
        for metric in metrics:
            comparison_table.add_data(
                metric, results["wave_network"][metric], results["bert"][metric]
            )

        wandb.log(
            {
                "model_comparison": comparison_table,
                "parameter_count": {
                    "wave_network": sum(p.numel() for p in wave_model.parameters()),
                    "bert": sum(p.numel() for p in bert_model.parameters()),
                },
                "memory_usage": {
                    "wave_network": results["wave_network"]["memory_used"],
                    "bert": results["bert"]["memory_used"],
                },
            }
        )

        # Print comparison
        print("\nFinal Test Results:")
        print("Wave Network:")
        print(results["wave_network"])
        print("\nBERT:")
        print(results["bert"])

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
