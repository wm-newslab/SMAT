"""Run frozen-model segment ablations separately from MIL training."""

import numpy as np

from src.config import DATASET_FIELDS, DIMENSIONS
from src.features import build_instances, count_actions, transform_with_checkpoint
from src.model import infer, load_checkpoint, restore_model
from src.utils import (
    choose_device,
    experiment_ablation_dir,
    experiment_model_dir,
    read_csv,
    seed_everything,
    write_csv,
)

PERCENTAGES = (10, 20, 30, 40, 50)
METHODS = ("important", "random", "non_important", "longest", "shortest")
PREDICTION_FIELDS = (
    *DATASET_FIELDS,
    "test_fold",
    "true_label",
    "predicted_label",
    "attention_weights",
)


def condition_seed(seed, method, percentage, user):
    return seed + METHODS.index(method) * 1_000_000 + percentage * 10_000 + user


def select_removed(
    users, offsets, baseline_attention, actions, method, percentage, seed
):
    removed = {}
    for user in users:
        start, end = offsets[user]
        count = end - start
        remove_count = min(round(percentage / 100.0 * count), count)
        attention = baseline_attention[user]
        if remove_count == 0:
            chosen = []
        elif method == "important":
            chosen = np.argsort(-attention, kind="stable")[:remove_count]
        elif method in ("longest", "shortest"):
            action_counts = np.asarray(
                [count_actions(actions[index]) for index in range(start, end)]
            )
            direction = -1 if method == "longest" else 1
            chosen = np.argsort(direction * action_counts, kind="stable")[:remove_count]
        elif method == "non_important":
            chosen = np.argsort(attention, kind="stable")[:remove_count]
        elif method == "random":
            candidates = np.arange(count)
            rng = np.random.default_rng(
                condition_seed(seed, method, percentage, int(user))
            )
            chosen = rng.choice(
                candidates, size=min(remove_count, len(candidates)), replace=False
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        removed[user] = np.asarray(chosen, dtype=np.int64)
    return removed


def prepare_experiment(rows, model_dir, device):
    first_checkpoint = load_checkpoint(model_dir / "models" / "fold_1.pt", device)
    classes = first_checkpoint["classes"]
    fields = list(DIMENSIONS.values())
    strings, offsets, labels = build_instances(rows, fields, classes)
    fold_values = sorted({int(row["test_fold"]) for row in rows})
    resources = []
    baseline_attention = [
        np.asarray(row["attention_weights"].split(" | "), dtype=np.float64)
        for row in rows
    ]
    for fold in fold_values:
        checkpoint = load_checkpoint(model_dir / "models" / f"fold_{fold}.pt", device)
        if checkpoint["classes"] != classes:
            raise ValueError(f"Class order mismatch in fold {fold}")
        features = transform_with_checkpoint(strings, checkpoint)
        model = restore_model(checkpoint, device)
        users = np.asarray(
            [index for index, row in enumerate(rows) if int(row["test_fold"]) == fold]
        )
        resources.append((users, features, model))
    return classes, strings, offsets, labels, resources, baseline_attention


def prediction_row(row, classes, prediction, attention):
    return {
        **{field: row[field] for field in DATASET_FIELDS},
        "test_fold": row["test_fold"],
        "true_label": row["user_class"],
        "predicted_label": classes[prediction],
        "attention_weights": " | ".join(f"{weight:.6f}" for weight in attention),
    }


def run_experiment(experiment, dataset_path, args, device):
    model_dir = experiment_model_dir(
        args.results_dir, args.task, args.dataset, experiment
    )
    rows = read_csv(dataset_path, DATASET_FIELDS)
    mil_predictions = read_csv(
        model_dir / "predictions.csv",
        ("user_id", "test_fold", "attention_weights"),
    )
    if len(rows) != len(mil_predictions):
        raise ValueError("Dataset and MIL predictions have different row counts")
    for row, prediction in zip(rows, mil_predictions):
        if row["user_id"] != prediction["user_id"]:
            raise ValueError("Dataset and MIL prediction user order does not match")
        row.update(
            test_fold=prediction["test_fold"],
            attention_weights=prediction["attention_weights"],
        )
    (
        classes,
        strings,
        offsets,
        labels,
        resources,
        baseline_attention,
    ) = prepare_experiment(rows, model_dir, device)
    actions = strings[DIMENSIONS["action"]]
    output_dir = experiment_ablation_dir(
        args.results_dir, args.task, args.dataset, experiment
    )
    for percentage in PERCENTAGES:
        for method in METHODS:
            collected = []
            for users, features, model in resources:
                removed = select_removed(
                    users,
                    offsets,
                    baseline_attention,
                    actions,
                    method,
                    percentage,
                    args.seed,
                )
                fold_predictions, fold_attention = infer(
                    model,
                    features,
                    offsets,
                    labels,
                    users,
                    args.batch_size,
                    device,
                    removed,
                )
                for local, user in enumerate(users):
                    collected.append(
                        (
                            int(user),
                            prediction_row(
                                rows[user],
                                classes,
                                fold_predictions[local],
                                fold_attention[local],
                            ),
                        )
                    )
            collected.sort(key=lambda item: item[0])
            prediction_rows = [row for _, row in collected]
            path = output_dir / "predictions" / f"{method}_{percentage}.csv"
            write_csv(path, prediction_rows, PREDICTION_FIELDS)
            print(f"  saved {path.name}", flush=True)
    print(f"  saved: {output_dir}", flush=True)


def ablate_all(config, experiments):
    """Ablate every selected experiment using its frozen fold models."""
    seed_everything(config.seed)
    device = choose_device(config.device)
    for experiment, dataset_path in experiments:
        print(
            f"{config.task}/{config.dataset}/{experiment}: device={device}",
            flush=True,
        )
        run_experiment(experiment, dataset_path, config, device)
