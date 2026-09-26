"""Train fold-specific gated-attention MIL models for prepared experiments."""

from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from src.bags import make_loader
from src.config import DATASET_FIELDS, DIMENSIONS, N_FOLDS
from src.features import (
    build_instances,
    read_vocabularies,
    transform_with_vocabularies,
)
from src.model import GatedAttentionMIL, infer
from src.utils import (
    choose_device,
    experiment_model_dir,
    experiment_vocabulary_path,
    read_csv,
    seed_everything,
    write_csv,
)

PREDICTION_FIELDS = (
    *DATASET_FIELDS,
    "test_fold",
    "predicted_label",
    "attention_weights",
)


def folds(labels, groups, seed):
    counts = Counter(labels.tolist())
    if min(counts.values()) < N_FOLDS:
        raise ValueError(f"Every class needs at least {N_FOLDS} rows: {counts}")
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    return splitter.split(np.zeros(len(labels)), labels, groups)


def train_model(model, features, offsets, labels, users, args, device):
    loader = make_loader(
        features, offsets, labels, users, args.batch_size, shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for instances, mask, targets in loader:
            optimizer.zero_grad()
            logits, _ = model(instances.to(device), mask.to(device))
            loss = loss_function(logits, targets.to(device))
            loss.backward()
            optimizer.step()
        print(f"    epoch {epoch}/{args.epochs}", flush=True)


def save_checkpoint(path, model, vocabularies, classes):
    checkpoint = {
        "classes": classes,
        "vocabularies": vocabularies,
        "input_dim": model.encoder[0].in_features,
        "hidden_dim": model.encoder[0].out_features,
        "attention_dim": model.attention_v.out_features,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def train_experiment(experiment, input_path, args, device):
    rows = read_csv(input_path, DATASET_FIELDS)
    classes = sorted({row["user_class"] for row in rows})
    if len(classes) < 2:
        raise ValueError(f"{experiment} has fewer than two classes: {classes}")
    fields = list(DIMENSIONS.values())
    strings, offsets, labels = build_instances(rows, fields, classes)
    vocabulary_path = experiment_vocabulary_path(
        args.results_dir, args.task, args.dataset, experiment
    )
    vocabularies = read_vocabularies(vocabulary_path)
    if set(vocabularies) != set(fields):
        raise ValueError(
            f"{vocabulary_path} fields do not match the model inputs: "
            f"expected {sorted(fields)}, found {sorted(vocabularies)}"
        )
    features = transform_with_vocabularies(strings, vocabularies)
    groups = np.asarray([row["user_id"] for row in rows])
    output_dir = experiment_model_dir(
        args.results_dir, args.task, args.dataset, experiment
    )
    all_predictions = np.full(len(rows), -1, dtype=np.int64)
    all_attention = [None] * len(rows)
    test_folds = np.zeros(len(rows), dtype=np.int64)
    print(
        f"{args.task}/{args.dataset}/{experiment}: rows={len(rows)}, "
        f"classes={dict(Counter(row['user_class'] for row in rows))}, device={device}",
        flush=True,
    )
    for fold, (train_users, test_users) in enumerate(
        folds(labels, groups, args.seed), start=1
    ):
        print(
            f"  fold {fold}/{N_FOLDS}: train={len(train_users)}, test={len(test_users)}",
            flush=True,
        )
        model = GatedAttentionMIL(
            features.shape[1], len(classes), args.hidden_dim, args.attention_dim
        ).to(device)
        train_model(model, features, offsets, labels, train_users, args, device)
        predictions, attention = infer(
            model,
            features,
            offsets,
            labels,
            test_users,
            args.batch_size,
            device,
        )
        all_predictions[test_users] = predictions
        test_folds[test_users] = fold
        for local_index, user in enumerate(test_users):
            all_attention[user] = attention[local_index]

        save_checkpoint(
            output_dir / "models" / f"fold_{fold}.pt",
            model,
            vocabularies,
            classes,
        )
        print(f"    saved fold_{fold}.pt", flush=True)

    if np.any(all_predictions < 0) or any(value is None for value in all_attention):
        raise RuntimeError("Not every user received an out-of-fold prediction")
    prediction_rows = []
    for index, row in enumerate(rows):
        prediction_rows.append(
            {
                **{field: row[field] for field in DATASET_FIELDS},
                "test_fold": int(test_folds[index]),
                "predicted_label": classes[all_predictions[index]],
                "attention_weights": " | ".join(
                    f"{weight:.6f}" for weight in all_attention[index]
                ),
            }
        )
    write_csv(output_dir / "predictions.csv", prediction_rows, PREDICTION_FIELDS)
    print(f"  saved: {output_dir}", flush=True)


def train_all(config, experiments):
    """Train every prepared experiment."""
    seed_everything(config.seed)
    device = choose_device(config.device)
    for experiment, dataset_path in experiments:
        train_experiment(experiment, dataset_path, config, device)
