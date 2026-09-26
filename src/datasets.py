"""Create balanced, pause-segmented datasets for either SMAT task."""

import json
import random
import re
from collections import Counter

from src.config import DATASET_FIELDS, DIMENSIONS
from src.features import (
    build_instances,
    fit_vocabularies,
    json_to_segmented_row,
    segment_count,
    write_vocabularies,
)
from src.utils import experiment_dataset_path, experiment_vocabulary_path, write_csv


def source_directories(root):
    directories = sorted(path for path in root.iterdir() if path.is_dir())
    if not directories:
        raise ValueError(f"No dataset/campaign directories found under {root}")
    return directories


def coordination_campaign_directories(root):
    """Return campaign directories, accepting a campaign itself as the root."""
    if any(root.glob("*.json")):
        return [root]

    directories = source_directories(root)
    if "YYYY" in root.name:
        partition_pattern = re.compile(
            re.escape(root.name).replace("YYYY", r"\d{4}")
        )
        if all(partition_pattern.fullmatch(path.name) for path in directories):
            return [root]

    if directories and all(re.fullmatch(r"\d{4}_\d{2}", path.name) for path in directories):
        campaigns = [
            campaign
            for month in directories
            for campaign in source_directories(month)
        ]
        if not campaigns:
            raise ValueError(f"No campaign directories found under {root}")
        return sorted(campaigns)
    return directories


def read_users(directory, dataset_name):
    rows = []
    for path in sorted(directory.rglob("*.json")):
        try:
            with path.open(encoding="utf-8") as source:
                data = json.load(source)
            row = json_to_segmented_row(data, dataset_name)
            if not row["user_id"] or not row["user_class"]:
                raise ValueError("missing user_id or user_class")
            if segment_count(row) == 0:
                raise ValueError("no tweet segments")
            rows.append(row)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            print(f"Skipping {path}: {error}")
    return rows


def balance_rows(rows, seed):
    """Randomly undersample every class to the minority-class size."""
    by_class = {}
    for row in rows:
        by_class.setdefault(row["user_class"], []).append(row)
    if len(by_class) < 2:
        counts = Counter(row["user_class"] for row in rows)
        raise ValueError(f"At least two classes are required: {counts}")
    target = min(map(len, by_class.values()))
    rng = random.Random(seed)
    balanced = []
    for label in sorted(by_class):
        balanced.extend(rng.sample(by_class[label], target))
    rng.shuffle(balanced)
    return balanced


def remove_duplicate_users(rows):
    counts = Counter(row["user_id"] for row in rows)
    repeated = {user_id for user_id, count in counts.items() if count > 1}
    return [row for row in rows if row["user_id"] not in repeated], repeated


def prepare_experiment(rows, min_segments, should_balance, seed):
    rows = [row for row in rows if segment_count(row) >= min_segments]
    rows, duplicate_users = remove_duplicate_users(rows)
    if not rows:
        raise ValueError("No users remain after filtering")
    if should_balance:
        rows = balance_rows(rows, seed)
    else:
        labels = {row["user_class"] for row in rows}
        if len(labels) < 2:
            raise ValueError(
                f"At least two classes are required; found {sorted(labels)}"
            )
    return rows, duplicate_users


def create_automation(args):
    rows = []
    for source_dir in source_directories(args.input_dir):
        rows.extend(read_users(source_dir, source_dir.name))
    prepared, duplicates = prepare_experiment(
        rows, args.min_segments, args.balance, args.seed
    )
    return [(args.dataset, prepared, duplicates)]


def create_coordination(args):
    experiments = []
    for campaign_dir in coordination_campaign_directories(args.input_dir):
        rows = read_users(campaign_dir, campaign_dir.name)
        if not rows:
            print(f"Skipping empty campaign: {campaign_dir.name}")
            continue
        try:
            prepared, duplicates = prepare_experiment(
                rows, args.min_segments, args.balance, args.seed
            )
        except ValueError as error:
            print(f"Skipping campaign {campaign_dir.name}: {error}")
            continue
        experiments.append((campaign_dir.name, prepared, duplicates))
    if not experiments:
        raise ValueError("No coordination campaign produced a usable dataset")
    return experiments


def create_datasets(config):
    if config.task == "automation":
        experiments = create_automation(config)
    else:
        experiments = create_coordination(config)
    saved = []
    for experiment, rows, duplicates in experiments:
        output = experiment_dataset_path(
            config.results_dir, config.task, config.dataset, experiment
        )
        vocabulary_output = experiment_vocabulary_path(
            config.results_dir, config.task, config.dataset, experiment
        )
        fields = list(DIMENSIONS.values())
        classes = sorted({row["user_class"] for row in rows})
        strings, _, _ = build_instances(rows, fields, classes)
        vocabularies = fit_vocabularies(strings, fields)
        write_csv(output, rows, DATASET_FIELDS)
        write_vocabularies(vocabulary_output, vocabularies)
        saved.append((experiment, output))
        feature_count = sum(len(vocabulary) for vocabulary in vocabularies.values())
        print(
            f"Created {output}: {len(rows)} users "
            f"({len(duplicates)} duplicates excluded, {feature_count} features)"
        )
    return saved
