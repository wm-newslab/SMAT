"""Run automation or coordination detection from start to finish."""

import argparse
from pathlib import Path

from src.ablation import ablate_all
from src.datasets import create_datasets
from src.training import train_all


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("automation", "coordination"))
    parser.add_argument("dataset", help="Dataset name, for example main or infoOps")
    parser.add_argument("input_dir", type=Path)
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error(f"input directory does not exist: {args.input_dir}")

    # Fixed experiment settings.
    args.results_dir = Path("results")
    args.min_segments = 10
    args.balance = True
    args.epochs = 25
    args.batch_size = 64
    args.learning_rate = 1e-3
    args.hidden_dim = 64
    args.attention_dim = 32
    args.seed = 42
    args.device = "auto"
    return args


def run(args):
    print("\n=== dataset creation ===", flush=True)
    experiments = create_datasets(args)
    print("\n=== MIL training ===", flush=True)
    train_all(args, experiments)
    print("\n=== ablation ===", flush=True)
    ablate_all(args, experiments)


if __name__ == "__main__":
    run(parse_arguments())
