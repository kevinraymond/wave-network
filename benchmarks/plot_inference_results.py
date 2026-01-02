"""
Plot inference benchmark results.

Generates visualizations comparing WaveNetwork, FNet, and Transformer performance.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(filepath: str) -> list[dict]:
    """Load benchmark results from JSON."""
    with open(filepath) as f:
        return json.load(f)


def plot_throughput_scaling(results: list[dict], output_dir: Path):
    """Plot throughput vs sequence length for each device (higher is better)."""
    devices = sorted({r["device"] for r in results})
    batch_sizes = sorted({r["batch_size"] for r in results})
    models = ["WaveNetwork", "FNet", "Transformer"]
    colors = {"WaveNetwork": "#2ecc71", "FNet": "#3498db", "Transformer": "#e74c3c"}
    markers = {"WaveNetwork": "o", "FNet": "s", "Transformer": "^"}

    for device in devices:
        fig, axes = plt.subplots(1, len(batch_sizes), figsize=(5 * len(batch_sizes), 4))
        if len(batch_sizes) == 1:
            axes = [axes]

        for ax, batch_size in zip(axes, batch_sizes):
            for model in models:
                data = [
                    r
                    for r in results
                    if r["device"] == device
                    and r["batch_size"] == batch_size
                    and r["model_name"] == model
                ]
                if data:
                    seq_lens = [d["seq_len"] for d in sorted(data, key=lambda x: x["seq_len"])]
                    throughputs = [
                        d["throughput_tokens_per_sec"] / 1e6
                        for d in sorted(data, key=lambda x: x["seq_len"])
                    ]
                    ax.plot(
                        seq_lens,
                        throughputs,
                        marker=markers[model],
                        color=colors[model],
                        label=model,
                        linewidth=2,
                        markersize=6,
                    )

            ax.set_xlabel("Sequence Length")
            ax.set_ylabel("Throughput (M tokens/sec) ↑")
            ax.set_title(f"Batch Size = {batch_size}")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.legend()
            ax.grid(True, alpha=0.3)

        device_name = device.replace(":", "_")
        fig.suptitle(
            f"Throughput Scaling ({device.upper()}) — Higher is Better",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            output_dir / f"throughput_scaling_{device_name}.png", dpi=150, bbox_inches="tight"
        )
        plt.close()
        print(f"Saved: throughput_scaling_{device_name}.png")


def plot_speedup(results: list[dict], output_dir: Path):
    """Plot speedup vs Transformer baseline."""
    devices = sorted({r["device"] for r in results})

    for device in devices:
        device_results = [r for r in results if r["device"] == device]
        batch_sizes = sorted({r["batch_size"] for r in device_results})
        seq_lens = sorted({r["seq_len"] for r in device_results})

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(seq_lens))
        width = 0.35

        for i, batch_size in enumerate(batch_sizes):
            wave_speedups = []
            fnet_speedups = []

            for seq_len in seq_lens:
                trans_data = [
                    r
                    for r in device_results
                    if r["batch_size"] == batch_size
                    and r["seq_len"] == seq_len
                    and r["model_name"] == "Transformer"
                ]
                wave_data = [
                    r
                    for r in device_results
                    if r["batch_size"] == batch_size
                    and r["seq_len"] == seq_len
                    and r["model_name"] == "WaveNetwork"
                ]
                fnet_data = [
                    r
                    for r in device_results
                    if r["batch_size"] == batch_size
                    and r["seq_len"] == seq_len
                    and r["model_name"] == "FNet"
                ]

                if trans_data and wave_data:
                    wave_speedups.append(trans_data[0]["latency_ms"] / wave_data[0]["latency_ms"])
                else:
                    wave_speedups.append(0)

                if trans_data and fnet_data:
                    fnet_speedups.append(trans_data[0]["latency_ms"] / fnet_data[0]["latency_ms"])
                else:
                    fnet_speedups.append(0)

            offset = (i - len(batch_sizes) / 2 + 0.5) * width * 0.8
            ax.bar(
                x + offset - width / 4,
                wave_speedups,
                width / 2,
                label=f"Wave (batch={batch_size})",
                alpha=0.8,
            )
            ax.bar(
                x + offset + width / 4,
                fnet_speedups,
                width / 2,
                label=f"FNet (batch={batch_size})",
                alpha=0.6,
            )

        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Speedup vs Transformer ↑")
        ax.set_title(f"Speedup vs Transformer ({device.upper()}) — Higher is Better")
        ax.set_xticks(x)
        ax.set_xticklabels(seq_lens)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=1, color="black", linestyle="--", alpha=0.5)

        device_name = device.replace(":", "_")
        plt.tight_layout()
        plt.savefig(output_dir / f"speedup_{device_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: speedup_{device_name}.png")


def main():
    parser = argparse.ArgumentParser(description="Plot inference benchmark results")
    parser.add_argument(
        "--input",
        type=str,
        default="data/results/inference_benchmark.json",
        help="Input JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results/plots",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(args.input)
    print(f"Loaded {len(results)} results")

    plot_throughput_scaling(results, output_dir)
    plot_speedup(results, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
