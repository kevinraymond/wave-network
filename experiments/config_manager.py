"""
Experiment configuration management.

Provides dataclasses for experiment configuration and YAML serialization.
Used by ablation runner and sweep scripts for reproducible experiments.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    # Core architecture
    vocab_size: int = 30522  # BERT tokenizer vocab size
    embedding_dim: int = 768
    num_classes: int = 4

    # Architecture type
    model_type: Literal[
        "wave_network", "deep_wave_network", "wave_attention", "fnet", "fnet_lite"
    ] = "wave_network"

    # Wave Network specific
    mode: Literal["modulation", "interference", "learnable"] = "modulation"
    learnable_mode: bool = False
    num_layers: int = 1

    # FNet specific
    ffn_dim: int | None = None  # None = 4 * embedding_dim
    max_seq_len: int = 512

    # Shared
    dropout: float = 0.1
    eps: float = 1e-8

    def to_model_kwargs(self) -> dict:
        """Convert to kwargs for model constructor."""
        base = {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "num_classes": self.num_classes,
            "eps": self.eps,
        }

        if self.model_type == "wave_network":
            base["mode"] = self.mode
            base["learnable_mode"] = self.learnable_mode
        elif self.model_type == "deep_wave_network":
            base["mode"] = self.mode
            base["num_layers"] = self.num_layers
        elif self.model_type == "wave_attention":
            base["num_layers"] = self.num_layers
            base["num_heads"] = 8  # Default
        elif self.model_type in ["fnet", "fnet_lite"]:
            base["num_layers"] = self.num_layers
            base["dropout"] = self.dropout
            if self.ffn_dim is not None:
                base["ffn_dim"] = self.ffn_dim
            if self.model_type == "fnet":
                base["max_seq_len"] = self.max_seq_len

        return base


@dataclass
class TrainingConfig:
    """Configuration for training loop."""

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    num_epochs: int = 4

    # Scheduling
    warmup_steps: int = 100
    scheduler_type: Literal["linear", "cosine", "constant"] = "linear"

    # Regularization
    max_grad_norm: float = 5.0
    dropout: float = 0.1

    # Data
    max_length: int = 384
    train_split: float = 0.8
    val_split: float = 0.1

    # Reproducibility
    seed: int = 42


@dataclass
class LoggingConfig:
    """Configuration for experiment tracking."""

    # W&B
    use_wandb: bool = True
    wandb_project: str = "wave-network"
    wandb_entity: str | None = None

    # Local logging
    log_dir: str = "logs"
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"

    # Logging frequency
    log_every_n_steps: int = 100
    eval_every_n_epochs: int = 1


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Identification
    name: str = "experiment"
    description: str = ""
    tags: list[str] = field(default_factory=list)

    # Components
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Dataset
    dataset: str = "ag_news"
    train_path: str | None = None
    test_path: str | None = None

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """Create configuration from dictionary."""
        # Handle nested configs
        if "model" in data and isinstance(data["model"], dict):
            data["model"] = ModelConfig(**data["model"])
        if "training" in data and isinstance(data["training"], dict):
            data["training"] = TrainingConfig(**data["training"])
        if "logging" in data and isinstance(data["logging"], dict):
            data["logging"] = LoggingConfig(**data["logging"])

        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def copy(self, **updates) -> "ExperimentConfig":
        """Create a copy with optional updates."""
        data = self.to_dict()
        data.update(updates)
        return ExperimentConfig.from_dict(data)


@dataclass
class AblationConfig:
    """Configuration for ablation studies."""

    # Base configuration (applied to all runs)
    base: ExperimentConfig = field(default_factory=ExperimentConfig)

    # Ablation variations
    # Each item is a dict of updates to apply to base config
    variations: list[dict] = field(default_factory=list)

    # Execution settings
    num_seeds: int = 3
    seed_start: int = 42
    parallel: bool = False

    # Output
    output_dir: str = "ablation_results"
    group_name: str = "ablation"

    @classmethod
    def from_yaml(cls, path: str) -> "AblationConfig":
        """Load ablation config from YAML."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Parse base config
        if "base" in data and isinstance(data["base"], dict):
            data["base"] = ExperimentConfig.from_dict(data["base"])

        return cls(**data)

    def generate_experiments(self) -> list[ExperimentConfig]:
        """Generate all experiment configs for this ablation."""
        experiments = []

        for variation in self.variations:
            for seed_idx in range(self.num_seeds):
                seed = self.seed_start + seed_idx * 1000

                # Create experiment from base + variation
                exp_data = self.base.to_dict()

                # Apply variation (handles nested dicts)
                _deep_update(exp_data, variation)

                # Set seed
                exp_data["training"]["seed"] = seed

                # Set name with seed
                base_name = variation.get("name", "variation")
                exp_data["name"] = f"{base_name}_seed{seed}"

                # Add tags
                exp_data["tags"] = exp_data.get("tags", []) + variation.get("tags", [])

                experiments.append(ExperimentConfig.from_dict(exp_data))

        return experiments


def _deep_update(base: dict, updates: dict) -> dict:
    """Recursively update nested dictionary."""
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def create_model_from_config(config: ModelConfig):
    """
    Factory function to create model from config.

    Args:
        config: ModelConfig instance

    Returns:
        Instantiated model
    """
    from models.fnet import FNet, FNetLite
    from wave_attention import WaveAttentionNetwork
    from wave_network import WaveNetwork
    from wave_network_deep import DeepWaveNetwork

    model_classes = {
        "wave_network": WaveNetwork,
        "deep_wave_network": DeepWaveNetwork,
        "wave_attention": WaveAttentionNetwork,
        "fnet": FNet,
        "fnet_lite": FNetLite,
    }

    model_class = model_classes.get(config.model_type)
    if model_class is None:
        raise ValueError(f"Unknown model type: {config.model_type}")

    return model_class(**config.to_model_kwargs())


# Preset configurations for common experiments
PRESETS = {
    "wave_small": ExperimentConfig(
        name="wave_small",
        model=ModelConfig(model_type="wave_network", embedding_dim=256),
        training=TrainingConfig(learning_rate=1e-3, num_epochs=4),
    ),
    "wave_base": ExperimentConfig(
        name="wave_base",
        model=ModelConfig(model_type="wave_network", embedding_dim=768),
        training=TrainingConfig(learning_rate=1e-3, num_epochs=4),
    ),
    "deep_wave": ExperimentConfig(
        name="deep_wave",
        model=ModelConfig(model_type="deep_wave_network", num_layers=3),
        training=TrainingConfig(learning_rate=1e-3, num_epochs=4),
    ),
    "fnet_small": ExperimentConfig(
        name="fnet_small",
        model=ModelConfig(model_type="fnet", num_layers=3, embedding_dim=256),
        training=TrainingConfig(learning_rate=1e-4, num_epochs=4),
    ),
    "fnet_base": ExperimentConfig(
        name="fnet_base",
        model=ModelConfig(model_type="fnet", num_layers=6, embedding_dim=768),
        training=TrainingConfig(learning_rate=1e-4, num_epochs=4),
    ),
}


def get_preset(name: str) -> ExperimentConfig:
    """Get a preset configuration by name."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name].copy()
