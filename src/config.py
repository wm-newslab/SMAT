"""Shared constants describing the SMAT data and experiment configuration."""

SEGMENT_START_MARKS = {"⚁", "⚂", "⚃", "⚄", "⚅"}
ACTION_SYMBOLS = {"T", "P", "p", "π", "R", "r", "ρ"}
DIMENSIONS = {
    "action": "action_bloc_string",
    "content": "content_bloc_string",
    "topics": "topics_bloc_string",
}
DATASET_FIELDS = (
    "user_id",
    "user_class",
    "dataset_name",
    *DIMENSIONS.values(),
    "tweet_ids_for_segments",
)
SEED = 42
N_FOLDS = 5
