"""
Inference Speed Benchmark: Wave Network vs FNet vs Transformer

Measures latency, throughput, and memory usage across different
sequence lengths to understand the O(n) vs O(n²) scaling behavior.

Results are logged to MLflow for tracking and visualization.
"""

import argparse
import gc
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Add parent directory to path for imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.fnet import FNet
from wave_network import WaveNetwork


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    model_name: str
    seq_len: int
    batch_size: int
    device: str
    latency_ms: float
    latency_std_ms: float
    throughput_tokens_per_sec: float
    memory_mb: float
    num_params: int


class TransformerBaseline(nn.Module):
    """
    Simple Transformer encoder for baseline comparison.

    Uses PyTorch's TransformerEncoder with self-attention.
    Matched to have similar parameter count to WaveNetwork/FNet.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 768,
        num_classes: int = 4,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = None,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        if ffn_dim is None:
            ffn_dim = 4 * embedding_dim

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)

        # Create attention mask for transformer (True = ignore)
        if attention_mask is not None:
            # PyTorch transformer expects True for positions to IGNORE
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        # Encode
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Pool and classify
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        else:
            pooled = x.mean(dim=1)

        return self.classifier(pooled)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_memory(device: str) -> float:
    """Get current GPU memory usage in MB."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_memory():
    """Reset memory stats and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def benchmark_model(
    model: nn.Module,
    model_name: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> BenchmarkResult:
    """
    Benchmark a model's inference speed.

    Args:
        model: The model to benchmark
        model_name: Name for reporting
        input_ids: Input tensor
        attention_mask: Attention mask tensor
        num_warmup: Number of warmup runs
        num_runs: Number of timed runs

    Returns:
        BenchmarkResult with timing statistics
    """
    device = str(input_ids.device)
    batch_size, seq_len = input_ids.shape

    model.eval()
    reset_memory()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_ids, attention_mask)

    # Synchronize before timing
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(input_ids, attention_mask)

            if device == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

    # Calculate statistics
    latency_ms = sum(latencies) / len(latencies)
    latency_std = (sum((x - latency_ms) ** 2 for x in latencies) / len(latencies)) ** 0.5

    # Throughput: tokens per second
    total_tokens = batch_size * seq_len
    throughput = total_tokens / (latency_ms / 1000)

    # Memory
    memory_mb = measure_memory(device)

    return BenchmarkResult(
        model_name=model_name,
        seq_len=seq_len,
        batch_size=batch_size,
        device=device,
        latency_ms=latency_ms,
        latency_std_ms=latency_std,
        throughput_tokens_per_sec=throughput,
        memory_mb=memory_mb,
        num_params=count_parameters(model),
    )


def create_models(
    vocab_size: int,
    embedding_dim: int,
    num_classes: int,
    num_layers: int,
    max_seq_len: int,
    device: str,
):
    """Create all models for benchmarking with matched configurations."""
    models = {}

    # Wave Network (single layer by default)
    models["WaveNetwork"] = WaveNetwork(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        mode="modulation",
    ).to(device)

    # FNet
    models["FNet"] = FNet(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
    ).to(device)

    # Transformer baseline
    models["Transformer"] = TransformerBaseline(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
    ).to(device)

    return models


def run_benchmarks(
    seq_lengths: list[int],
    batch_sizes: list[int],
    devices: list[str],
    vocab_size: int = 30522,
    embedding_dim: int = 256,
    num_classes: int = 4,
    num_layers: int = 4,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> list[BenchmarkResult]:
    """
    Run full benchmark suite.

    Args:
        seq_lengths: List of sequence lengths to test
        batch_sizes: List of batch sizes to test
        devices: List of devices to test ("cpu", "cuda")
        vocab_size: Vocabulary size
        embedding_dim: Model embedding dimension
        num_classes: Number of output classes
        num_layers: Number of layers for FNet/Transformer
        num_warmup: Warmup iterations
        num_runs: Timed iterations

    Returns:
        List of BenchmarkResults
    """
    results = []

    for device in devices:
        if device == "cuda" and not torch.cuda.is_available():
            print(f"Skipping {device} - not available")
            continue

        print(f"\n{'='*60}")
        print(f"Device: {device.upper()}")
        print(f"{'='*60}")

        # Create models once per device (max_seq_len = max of all seq_lengths we'll test)
        max_seq_len = max(seq_lengths)
        models = create_models(
            vocab_size, embedding_dim, num_classes, num_layers, max_seq_len, device
        )

        # Print parameter counts
        print("\nParameter counts:")
        for name, model in models.items():
            print(f"  {name}: {count_parameters(model):,}")

        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                print(f"\n--- Batch={batch_size}, SeqLen={seq_len} ---")

                # Create input data
                input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
                attention_mask = torch.ones(batch_size, seq_len, device=device)

                for name, model in models.items():
                    try:
                        result = benchmark_model(
                            model=model,
                            model_name=name,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            num_warmup=num_warmup,
                            num_runs=num_runs,
                        )
                        results.append(result)

                        print(
                            f"  {name:15s}: {result.latency_ms:8.2f}ms "
                            f"(±{result.latency_std_ms:.2f}) | "
                            f"{result.throughput_tokens_per_sec:,.0f} tok/s | "
                            f"{result.memory_mb:.1f}MB"
                        )
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"  {name:15s}: OOM")
                            reset_memory()
                        else:
                            raise

    return results


def print_summary_table(results: list[BenchmarkResult]):
    """Print a summary comparison table."""
    print("\n" + "=" * 80)
    print("SUMMARY: Relative Speed (vs Transformer baseline)")
    print("=" * 80)

    # Group by (device, batch_size, seq_len)
    from collections import defaultdict

    grouped = defaultdict(dict)
    for r in results:
        key = (r.device, r.batch_size, r.seq_len)
        grouped[key][r.model_name] = r.latency_ms

    print(f"\n{'Device':<6} {'Batch':<6} {'SeqLen':<8} {'Wave vs Trans':<15} {'FNet vs Trans':<15}")
    print("-" * 60)

    for (device, batch, seq_len), latencies in sorted(grouped.items()):
        if "Transformer" in latencies:
            trans_lat = latencies["Transformer"]
            wave_speedup = trans_lat / latencies.get("WaveNetwork", trans_lat)
            fnet_speedup = trans_lat / latencies.get("FNet", trans_lat)
            print(
                f"{device:<6} {batch:<6} {seq_len:<8} {wave_speedup:>6.2f}x faster   {fnet_speedup:>6.2f}x faster"
            )


def save_results(results: list[BenchmarkResult], output_path: str):
    """Save results to JSON file."""
    import json

    data = [
        {
            "model_name": r.model_name,
            "seq_len": r.seq_len,
            "batch_size": r.batch_size,
            "device": r.device,
            "latency_ms": r.latency_ms,
            "latency_std_ms": r.latency_std_ms,
            "throughput_tokens_per_sec": r.throughput_tokens_per_sec,
            "memory_mb": r.memory_mb,
            "num_params": r.num_params,
        }
        for r in results
    ]

    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_path}")


def log_to_mlflow(results: list[BenchmarkResult], config: dict):
    """Log benchmark results to MLflow."""
    if not MLFLOW_AVAILABLE:
        print("MLflow not available, skipping logging")
        return

    mlflow.set_experiment("wave-network-inference-benchmarks")

    with mlflow.start_run(run_name="inference_speed_comparison"):
        # Log configuration
        mlflow.log_params(
            {
                "embedding_dim": config["embedding_dim"],
                "num_layers": config["num_layers"],
                "seq_lengths": str(config["seq_lengths"]),
                "batch_sizes": str(config["batch_sizes"]),
                "devices": str(config["devices"]),
                "num_runs": config["num_runs"],
            }
        )

        # Log parameter counts (once per model)
        param_counts = {}
        for r in results:
            if r.model_name not in param_counts:
                param_counts[r.model_name] = r.num_params
                mlflow.log_metric(f"params_{r.model_name}", r.num_params)

        # Log metrics for each configuration
        for r in results:
            prefix = f"{r.model_name}_{r.device}_b{r.batch_size}_s{r.seq_len}"
            mlflow.log_metrics(
                {
                    f"{prefix}_latency_ms": r.latency_ms,
                    f"{prefix}_latency_std_ms": r.latency_std_ms,
                    f"{prefix}_throughput": r.throughput_tokens_per_sec,
                    f"{prefix}_memory_mb": r.memory_mb,
                }
            )

        # Log speedup comparisons
        from collections import defaultdict

        grouped = defaultdict(dict)
        for r in results:
            key = (r.device, r.batch_size, r.seq_len)
            grouped[key][r.model_name] = r.latency_ms

        for (device, batch, seq_len), latencies in grouped.items():
            if "Transformer" in latencies:
                trans_lat = latencies["Transformer"]
                for model_name, lat in latencies.items():
                    if model_name != "Transformer":
                        speedup = trans_lat / lat
                        mlflow.log_metric(
                            f"speedup_{model_name}_vs_Transformer_{device}_b{batch}_s{seq_len}",
                            speedup,
                        )

        print("\nResults logged to MLflow")


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference speed")
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024],
        help="Sequence lengths to test",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 32],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["cpu", "cuda"],
        help="Devices to test",
    )
    parser.add_argument("--embedding-dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument(
        "--num-layers", type=int, default=4, help="Number of layers for FNet/Transformer"
    )
    parser.add_argument("--num-warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--num-runs", type=int, default=100, help="Timed iterations")
    parser.add_argument(
        "--output", type=str, default="data/results/inference_benchmark.json", help="Output file"
    )
    parser.add_argument("--mlflow", action="store_true", help="Log results to MLflow")
    parser.add_argument(
        "--no-mlflow", dest="mlflow", action="store_false", help="Disable MLflow logging"
    )
    parser.set_defaults(mlflow=True)

    args = parser.parse_args()

    print("Inference Speed Benchmark")
    print("=" * 60)
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Devices: {args.devices}")
    print(f"Embedding dim: {args.embedding_dim}")
    print(f"Num layers: {args.num_layers}")
    print(f"MLflow logging: {args.mlflow and MLFLOW_AVAILABLE}")

    results = run_benchmarks(
        seq_lengths=args.seq_lengths,
        batch_sizes=args.batch_sizes,
        devices=args.devices,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
    )

    print_summary_table(results)
    save_results(results, args.output)

    # Log to MLflow
    if args.mlflow:
        config = {
            "embedding_dim": args.embedding_dim,
            "num_layers": args.num_layers,
            "seq_lengths": args.seq_lengths,
            "batch_sizes": args.batch_sizes,
            "devices": args.devices,
            "num_runs": args.num_runs,
        }
        log_to_mlflow(results, config)


if __name__ == "__main__":
    main()
