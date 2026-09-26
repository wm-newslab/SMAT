"""Export tweet-ID segments ranked by MIL attention weight."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


INPUT_FIELDS = (
    "user_id",
    "dataset_name",
    "user_class",
    "tweet_ids_for_segments",
    "attention_weights",
)
OUTPUT_FIELDS = ("user_id", "dataset", "user_class", "tweet_ranks")


def allow_large_csv_fields():
    """Raise the CSV field limit for rows containing many tweet segments."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def split_segments(value):
    if not value.strip():
        return []
    return [segment.strip() for segment in value.split("|")]


def rank_segments(row, source_path, row_number):
    tweet_segments = split_segments(row["tweet_ids_for_segments"])
    weight_values = split_segments(row["attention_weights"])
    if len(tweet_segments) != len(weight_values):
        raise ValueError(
            f"{source_path}:{row_number}: user {row['user_id']!r} has "
            f"{len(tweet_segments)} tweet segments but {len(weight_values)} "
            "attention weights"
        )

    weighted_segments = []
    for segment_index, (tweet_segment, weight_value) in enumerate(
        zip(tweet_segments, weight_values)
    ):
        try:
            weight = float(weight_value)
        except ValueError as error:
            raise ValueError(
                f"{source_path}:{row_number}: user {row['user_id']!r} has "
                f"an invalid attention weight at segment {segment_index + 1}: "
                f"{weight_value!r}"
            ) from error
        if not math.isfinite(weight):
            raise ValueError(
                f"{source_path}:{row_number}: user {row['user_id']!r} has "
                f"a non-finite attention weight at segment {segment_index + 1}"
            )
        tweet_ids = [tweet_id.strip() for tweet_id in tweet_segment.split(",")]
        tweet_ids = [tweet_id for tweet_id in tweet_ids if tweet_id]
        weighted_segments.append((weight, tweet_ids))

    weighted_segments.sort(key=lambda item: -item[0])
    return [
        {
            "rank": rank,
            "attention_weight": weight,
            "tweet_ids": tweet_ids,
        }
        for rank, (weight, tweet_ids) in enumerate(weighted_segments, start=1)
    ]


def output_path_for(source_path, input_dir, output_dir):
    relative_parent = source_path.relative_to(input_dir).parent
    name = "__".join(relative_parent.parts) or source_path.stem
    return output_dir / f"{name}.csv"


def export_file(source_path, destination_path):
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_name(f".{destination_path.name}.tmp")
    row_count = 0
    try:
        with source_path.open(encoding="utf-8", newline="") as source, \
                temporary_path.open("w", encoding="utf-8", newline="") as destination:
            reader = csv.DictReader(source)
            missing = set(INPUT_FIELDS).difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{source_path} is missing columns: {sorted(missing)}"
                )
            writer = csv.DictWriter(destination, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            for row_number, row in enumerate(reader, start=2):
                rankings = rank_segments(row, source_path, row_number)
                writer.writerow(
                    {
                        "user_id": row["user_id"],
                        "dataset": row["dataset_name"],
                        "user_class": row["user_class"],
                        "tweet_ranks": json.dumps(
                            rankings, ensure_ascii=False, separators=(",", ":")
                        ),
                    }
                )
                row_count += 1
        if row_count == 0:
            raise ValueError(f"CSV is empty: {source_path}")
        temporary_path.replace(destination_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return row_count


def export_all(input_dir, output_dir):
    source_paths = sorted(input_dir.rglob("predictions.csv"))
    if not source_paths:
        raise FileNotFoundError(f"No predictions.csv files found under {input_dir}")

    for source_path in source_paths:
        destination_path = output_path_for(source_path, input_dir, output_dir)
        row_count = export_file(source_path, destination_path)
        print(f"saved {destination_path} ({row_count:,} users)", flush=True)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Rank each user's tweet-ID segments by descending MIL attention weight."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/mil"),
        help="MIL results directory (default: results/mil)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("segment_ranking_radataset"),
        help="ranking CSV directory (default: segment_ranking_radataset)",
    )
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error(f"input directory does not exist: {args.input_dir}")
    return args


def main():
    allow_large_csv_fields()
    args = parse_arguments()
    export_all(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
