"""BLOC segmentation and character unigram/bigram feature construction."""

import json

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from src.config import ACTION_SYMBOLS, DIMENSIONS, SEGMENT_START_MARKS


def tweet_sort_key(item):
    key, _ = item
    try:
        return (0, int(key))
    except (TypeError, ValueError):
        return (1, str(key))


def join_segments(segments, separator=""):
    return " | ".join(separator.join(segment) for segment in segments)


def json_to_segmented_row(data, dataset_name):
    values = {"action": [], "content": [], "topics": [], "tweet_ids": []}
    tweets = sorted(data.get("tweet_details", {}).items(), key=tweet_sort_key)
    for _, tweet in tweets:
        action = str(tweet.get("action", ""))
        if (
            action[:1] in SEGMENT_START_MARKS
            and values["action"]
            and values["action"][-1]
        ):
            for field_values in values.values():
                field_values.append([])
        if not values["action"]:
            for field_values in values.values():
                field_values.append([])
        values["action"][-1].append(action)
        values["content"][-1].append(str(tweet.get("content_syntactic", "")))
        values["topics"][-1].append(str(tweet.get("topic_symbol", "")))
        values["tweet_ids"][-1].append(str(tweet.get("tweet_id", "")))

    return {
        "user_id": str(data.get("user_id", "")),
        "user_class": str(data.get("user_class", "")),
        "dataset_name": dataset_name,
        "action_bloc_string": join_segments(values["action"]),
        "content_bloc_string": join_segments(values["content"]),
        "topics_bloc_string": join_segments(values["topics"]),
        "tweet_ids_for_segments": join_segments(values["tweet_ids"], ","),
    }


def segment_count(row):
    value = row[DIMENSIONS["action"]]
    return len(value.split(" | ")) if value else 0


def clean_segment(text):
    return text.replace("(", "").replace(")", "").replace("|", "").strip()


def count_actions(action_segment):
    """Count BLOC action symbols while excluding all pause symbols."""
    return sum(symbol in ACTION_SYMBOLS for symbol in action_segment)


def split_aligned_segments(row, fields):
    segments = {
        field: [clean_segment(value) for value in row[field].split("|")]
        for field in fields
    }
    lengths = {field: len(values) for field, values in segments.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"User {row['user_id']} has misaligned segments: {lengths}")
    if not next(iter(lengths.values())):
        raise ValueError(f"User {row['user_id']} has no segments")
    return segments


def build_instances(rows, fields, classes):
    class_to_index = {label: index for index, label in enumerate(classes)}
    strings = {field: [] for field in fields}
    offsets = []
    labels = np.empty(len(rows), dtype=np.int64)
    cursor = 0
    for row_index, row in enumerate(rows):
        segments = split_aligned_segments(row, fields)
        count = len(segments[fields[0]])
        for field in fields:
            strings[field].extend(segments[field])
        offsets.append((cursor, cursor + count))
        cursor += count
        labels[row_index] = class_to_index[row["user_class"]]
    return strings, offsets, labels


def normalize_features(matrices):
    features = sparse.hstack(matrices, format="csr", dtype=np.float32)
    sums = np.asarray(features.sum(axis=1)).ravel()
    sums[sums == 0] = 1.0
    return (sparse.diags(1.0 / sums) @ features).tocsr()


def fit_vocabularies(strings, fields):
    """Fit one fixed character unigram/bigram vocabulary per dimension."""
    vocabularies = {}
    for field in fields:
        vectorizer = CountVectorizer(
            analyzer="char", ngram_range=(1, 2), lowercase=False, dtype=np.float32
        )
        vectorizer.fit(strings[field])
        vocabularies[field] = vectorizer.vocabulary_
    return vocabularies


def write_vocabularies(path, vocabularies):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as destination:
        json.dump(
            vocabularies,
            destination,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def read_vocabularies(path):
    with path.open(encoding="utf-8") as source:
        vocabularies = json.load(source)
    if not isinstance(vocabularies, dict) or not vocabularies:
        raise ValueError(f"Invalid or empty vocabulary file: {path}")
    return vocabularies


def transform_with_vocabularies(strings, vocabularies):
    matrices = []
    for field, vocabulary in vocabularies.items():
        if field not in strings:
            raise ValueError(f"Vocabulary contains unknown field: {field}")
        vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=(1, 2),
            lowercase=False,
            dtype=np.float32,
            vocabulary=vocabulary,
        )
        matrices.append(vectorizer.transform(strings[field]))
    return normalize_features(matrices)


def transform_with_checkpoint(strings, checkpoint):
    features = transform_with_vocabularies(strings, checkpoint["vocabularies"])
    if features.shape[1] != checkpoint["input_dim"]:
        raise ValueError("Feature width does not match checkpoint")
    return features
