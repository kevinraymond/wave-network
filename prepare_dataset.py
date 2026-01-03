import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        texts: np.ndarray,
        labels: np.ndarray,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        text_column: str = "text",
        label_column: str = "label",
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


def prepare_text_classification_data(
    train_path: str,
    test_path: str,
    text_column: str = "text",
    label_column: str = "label",
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 128,
    val_size: float = 0.2,
    batch_size: int = 64,
    random_state: int = 42,
    num_workers: int = 2,
    label_mapping: dict[str | int, int] | None = None,
) -> dict:
    """
    Prepare any text classification dataset stored in parquet format for training.

    Args:
        train_path: Path to training parquet file
        test_path: Path to test parquet file
        text_column: Name of the column containing text data
        label_column: Name of the column containing labels
        tokenizer_name: Name of the tokenizer to use
        max_length: Maximum sequence length
        val_size: Validation set size (fraction of training data)
        batch_size: Batch size for DataLoader
        random_state: Random seed for reproducibility
        num_workers: Number of worker processes for DataLoader
        label_mapping: Optional dictionary to map original labels to numeric indices

    Returns:
        Dictionary containing DataLoaders, vocabulary size, tokenizer, and number of classes
    """
    # Load the data
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Add split size logging
    total_train_size = len(train_df)

    # Validate column names
    required_columns = [text_column, label_column]
    for df, name in [(train_df, "training"), (test_df, "test")]:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns {missing_cols} in {name} dataset")

    # Print dataset info
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    print("\nLabel distribution in training set:")
    print(train_df[label_column].value_counts(normalize=True))

    # Handle label mapping
    if label_mapping is not None:
        train_df[label_column] = train_df[label_column].map(label_mapping)
        test_df[label_column] = test_df[label_column].map(label_mapping)
    elif train_df[label_column].dtype in ["object", "string"]:
        # Auto-create label mapping for string labels
        unique_labels = sorted(train_df[label_column].unique())
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        train_df[label_column] = train_df[label_column].map(label_mapping)
        test_df[label_column] = test_df[label_column].map(label_mapping)
    elif train_df[label_column].min() == 1:
        # Convert 1-based indices to 0-based
        train_df[label_column] = train_df[label_column] - 1
        test_df[label_column] = test_df[label_column] - 1

    # Split training data into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df[text_column].values,
        train_df[label_column].values,
        test_size=val_size,
        random_state=random_state,
        stratify=train_df[label_column].values,
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Create datasets
    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer, max_length, text_column, label_column
    )
    val_dataset = TextClassificationDataset(
        val_texts, val_labels, tokenizer, max_length, text_column, label_column
    )
    test_dataset = TextClassificationDataset(
        test_df[text_column].values,
        test_df[label_column].values,
        tokenizer,
        max_length,
        text_column,
        label_column,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Add detailed split reporting
    train_size = len(train_texts)
    val_size_actual = len(val_texts)
    test_size = len(test_df)

    print("\nDataset Split Details:")
    print(f"Total training data: {total_train_size}")
    print(f"Training set: {train_size} ({train_size/total_train_size:.1%})")
    print(f"Validation set: {val_size_actual} ({val_size_actual/total_train_size:.1%})")
    print(f"Test set: {test_size}")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer": tokenizer,
        "num_classes": len(np.unique(train_labels)),
        "label_mapping": label_mapping,
        # Add split information to return value
        "split_info": {
            "train_size": train_size,
            "val_size": val_size_actual,
            "test_size": test_size,
            "train_ratio": train_size / total_train_size,
            "val_ratio": val_size_actual / total_train_size,
        },
    }


# Example usage
def main():
    # Example with AG News dataset
    train_path = "hf/ag_news/data/train-00000-of-00001.parquet"
    test_path = "hf/ag_news/data/test-00000-of-00001.parquet"

    # Example with custom label mapping (if needed)

    data = prepare_text_classification_data(
        train_path=train_path,
        test_path=test_path,
        text_column="text",
        label_column="label",
        # label_mapping=label_mapping,  # Optional
    )

    # Print information about the prepared data
    print("\nDataset preparation completed:")
    print(f"Vocabulary size: {data['vocab_size']}")
    print(f"Number of classes: {data['num_classes']}")
    if data["label_mapping"]:
        print("\nLabel mapping:")
        for original, idx in data["label_mapping"].items():
            print(f"{original}: {idx}")

    # Example of accessing a batch
    batch = next(iter(data["train_loader"]))
    print("\nSample batch shapes:")
    print(f"Input IDs: {batch['input_ids'].shape}")
    print(f"Attention Mask: {batch['attention_mask'].shape}")
    print(f"Labels: {batch['label'].shape}")

    return data


if __name__ == "__main__":
    data = main()
