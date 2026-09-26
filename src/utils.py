"""Generic runtime, CSV I/O, and result-path utilities."""

import csv
import random

import numpy as np
import torch

from src.config import SEED


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_csv(path, required=()):
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    missing = set(required).difference(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return rows


def write_csv(path, rows, fieldnames=None):
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def experiment_dataset_path(results_dir, task, dataset, experiment):
    return results_dir / "datasets" / task / dataset / experiment / "users.csv"


def experiment_vocabulary_path(results_dir, task, dataset, experiment):
    return (
        results_dir
        / "datasets"
        / task
        / dataset
        / experiment
        / "vocabularies.json"
    )


def experiment_model_dir(results_dir, task, dataset, experiment):
    return results_dir / "mil" / task / dataset / experiment


def experiment_ablation_dir(results_dir, task, dataset, experiment):
    return results_dir / "ablation" / task / dataset / experiment
