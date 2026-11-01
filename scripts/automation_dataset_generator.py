import csv
import gzip
import os
import json
import spacy
import re
import logging

from pathlib import Path
from transformers import pipeline
from transformers.utils import logging as hf_logging
from textblob import TextBlob
from bloc.generator import add_bloc_sequences
from bloc.util import (
    datetime,
    get_default_symbols,
    getDictFromJson,
    genericErrorInfo
)

logger = logging.getLogger('bloc.bloc')

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "bloc_representation-general.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

# =========================
#CONFIG
# =========================
gen_bloc_params = {
    'blank_mark': 60,
    'bloc_alphabets': [
        'action',
        'content_syntactic',
        'change',
        'content_semantic_sentiment'
    ],
    'fold_start_count': 10,
    'keep_bloc_segments': True,
    'keep_tweets': True,
    'segmentation_type': 'segment_on_pauses',
    'segment_on_pauses': 3600,
    'sort_action_words': False,
    'tweet_order': 'sorted'
}


src = [
    "josh_political",
    "gregory_purchased",
    "varol-icwsm",
    "gilani-17",
    "rtbust",
    "botwiki",
    "astroturf",
    "kevin_feedback",
    "verified",
    "midterm-2018",
    "caverlee"
    # "cresci-17",
    # "stock",
    # "zoher-organization",
    # "pronbots",    
]

path_to_dataset = "automation_dataset/retraining_data"

all_bloc_symbols = get_default_symbols()


# =========================
# TOPIC + ENTITY SYMBOL MAPS
# =========================
LABELS = [
    "politics",
    "business",
    "finance",
    "technology",
    "science",
    "health",
    "sports",
    "entertainment",
    "crime",
    "weather",
    "advertising"
]

TOPIC_SYMBOL_MAP = {
    "politics": "P",
    "business": "B",
    "finance": "F",
    "technology": "T",
    "science": "C",
    "health": "H",
    "sports": "S",
    "entertainment": "E",
    "crime": "L",
    "weather": "W",
    "advertising": "A",
    "other": "O"
}

ENTITY_SYMBOL_MAP = {
    "PERSON": "⚇",
    "GPE": "⌖",
    "LOC": "⌖",
    "ORG": "⋈",
    "PRODUCT": "x",
    "FAC": "⌖"
}

SENTIMENT_SYMBOL_MAP = {
    "positive": "⋃",
    "neutral": "-",
    "negative": "⋂"
}

# =========================
# NLP MODELS
# =========================

hf_logging.set_verbosity_error()
NLP = spacy.load("en_core_web_sm")

try:
    import torch
    _CUDA_AVAILABLE = torch.cuda.is_available()
    _HF_DEVICE = 0 if _CUDA_AVAILABLE else -1
    if _CUDA_AVAILABLE:
        torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    torch = None
    _CUDA_AVAILABLE = False
    _HF_DEVICE = -1

_PIPELINE_KWARGS = {}
if _CUDA_AVAILABLE:
    _PIPELINE_KWARGS["torch_dtype"] = torch.float16

TOPIC_CLF = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=_HF_DEVICE,
    **_PIPELINE_KWARGS
)

TOPIC_THRESHOLD = 0.4
TOPIC_BATCH_SIZE = 32 if _CUDA_AVAILABLE else 8
SPACY_BATCH_SIZE = 128
USER_REPRESENTATION_SEGMENT_SIZE = 4


# =========================
# HELPERS
# =========================
def parse_time(t):
    return datetime.strptime(t, "%a %b %d %H:%M:%S %z %Y")

def get_user_id_class_map(f):
    user_id_class_map = {}
    all_classes = set()

    try:
        with open(f) as fd:
            
            rd = csv.reader(fd, delimiter='\t')
            for user_id, user_class in rd:
                user_id_class_map[user_id] = user_class
                all_classes.add(user_class)
    except:
        genericErrorInfo()

    return user_id_class_map, all_classes

def update_user_class(dataset_name, user_class):
    updated_user_class = user_class
    if dataset_name == 'astroturf':
        updated_user_class = 'bot'
    elif dataset_name == 'cresci-17' and (user_class == 'socialspam' or user_class == 'bot-traditionspam' or user_class == 'bot-fakefollower' or user_class == 'bot-socialspam'):
        updated_user_class = 'bot'
    elif dataset_name == 'zoher-organization' and user_class == 'organization':
        updated_user_class = 'human'
    return updated_user_class

def get_topics_batch(texts):
    non_empty_texts = [text if text else " " for text in texts]
    outputs = TOPIC_CLF(
        non_empty_texts,
        candidate_labels=LABELS,
        hypothesis_template="This tweet is about {}.",
        multi_label=True,
        batch_size=TOPIC_BATCH_SIZE,
        truncation=True
    )
    if isinstance(outputs, dict):
        outputs = [outputs]

    topics = []
    for text, out in zip(texts, outputs):
        if not text:
            topics.append(("other", "O"))
            continue

        top_label = out["labels"][0]
        top_score = out["scores"][0]

        if top_score < TOPIC_THRESHOLD:
            topics.append(("other", "O"))
        else:
            topics.append((top_label, TOPIC_SYMBOL_MAP.get(top_label, "O")))

    return topics

def get_topics(text):
    return get_topics_batch([text])[0]

def get_twt_text_exclusively(txt, entities):
    spans = []

    for val in entities.values():
        for ent in val:
            if 'indices' in ent:
                spans.append(ent['indices'])

    for st, en in sorted(spans, reverse=True):
        txt = txt[:st] + txt[en:]

    return ' '.join(txt.split())

def clean_tweet_text(tweet):
    twt = tweet.get('retweeted_status', tweet)
    try:
        txt = twt.get('full_text') or twt.get('text') or ''
        txt = re.sub(r'^\s*RT\s*:?\s*', '', txt, flags=re.IGNORECASE)
        txt = re.sub(r'https?://\S+', '', txt).strip()
    except Exception as e:
        logger.exception("Failed to clean tweet text")
        genericErrorInfo()
        txt = ''

    return twt, txt


# =========================
# SEGMENT ANALYSIS
# =========================
def tweet_semantic_analysis(tweet):
    return tweets_semantic_analysis([tweet])[0]

def tweets_semantic_analysis(tweets):
    cleaned_tweets = [clean_tweet_text(tweet) for tweet in tweets]
    texts = [txt for _, txt in cleaned_tweets]
    docs = list(NLP.pipe(texts, batch_size=SPACY_BATCH_SIZE))
    topics = get_topics_batch(texts)

    return [
        build_tweet_semantic_result(twt, txt, doc, topic)
        for (twt, txt), doc, topic in zip(cleaned_tweets, docs, topics)
    ]

def build_tweet_semantic_result(twt, txt, doc, topic):
    entities, entity_symbols = [], []
    for ent in doc.ents:
        entities.append(ent.text)
        entity_symbols.append(
            ENTITY_SYMBOL_MAP.get(ent.label_, "⊛")
        )

    polarity = TextBlob(txt).sentiment.polarity
    sentiment = SENTIMENT_SYMBOL_MAP["positive"] if polarity > 0 else SENTIMENT_SYMBOL_MAP["neutral"] if polarity == 0 else SENTIMENT_SYMBOL_MAP["negative"]

    topic_label, topic_symbol = topic

    return {
            'text': twt.get('text', ''),
            'exclusive_text': txt,
            'entities': entities,
            'entity_symbols': entity_symbols,
            'sentiments': sentiment,
            'topics': topic_label,
            'topic_symbol': topic_symbol
        }

def build_user_representation(tweet_details, segment_size=USER_REPRESENTATION_SEGMENT_SIZE):
    representation = {
        'action': [],
        'content_syntactic': [],
        'change': [],
        'content_semantic_sentiment': [],
        'topics': [],
        'entities': [],
        'complete_bloc': []
    }

    details = [tweet_details[i] for i in sorted(tweet_details)]
    for start in range(0, len(details), segment_size):
        segment = details[start:start + segment_size]

        action_bloc = ''.join(tweet['action'] for tweet in segment)
        content_bloc = ''.join(tweet['content_syntactic'] for tweet in segment)
        change_bloc = ''.join(tweet['change'] for tweet in segment)
        sentiment_bloc = ''.join(tweet['sentiments'] for tweet in segment)
        topic_bloc = ''.join(tweet['topic_symbol'] for tweet in segment if tweet['topic_symbol'])
        entity_bloc = ''.join(
            symbol
            for tweet in segment
            for symbol in tweet['entity_symbols']
        )

        representation['action'].append(action_bloc)
        representation['content_syntactic'].append(content_bloc)
        representation['change'].append(change_bloc)
        representation['content_semantic_sentiment'].append(sentiment_bloc)
        representation['topics'].append(topic_bloc)
        representation['entities'].append(entity_bloc)
        representation['complete_bloc'].append(
            f"({action_bloc}){content_bloc}({change_bloc})({sentiment_bloc})({topic_bloc})({entity_bloc})"
        )

    return {
        key: ' | '.join(value)
        for key, value in representation.items()
    }


# =========================
# MAIN PIPELINE
# =========================

def preprocess_data(src, path_to_dataset):

    for tweets_file in src:
        print(f"Processing dataset: {tweets_file}")

        tweets_path = f"{path_to_dataset}/{tweets_file}/tweets.jsons.gz"
        class_path = f"{path_to_dataset}/{tweets_file}/userIds.txt"

        user_id_class_map, _ = get_user_id_class_map(class_path)
        out_dir = BASE_DIR / "analyzed_users_v5" / tweets_file
        os.makedirs(out_dir, exist_ok=True)

        with gzip.open(tweets_path, 'rt', encoding='utf-8', errors='ignore') as infile:
            for raw_line in infile:
                try:
                    user_id, tweets_json = raw_line.rstrip('\n').split('\t', 1)
                    logger.info(f"Processing user_id: {user_id} from dataset: {tweets_file}")
                    tweets = getDictFromJson(tweets_json)

                    if len(tweets) < 20:
                        continue

                    if user_id not in user_id_class_map:
                        logger.warning("Skipping user_id %s from %s: missing class label", user_id, tweets_file)
                        continue

                    user_class = update_user_class(
                        tweets_file,
                        user_id_class_map[user_id]
                    )

                    tweets_sorted = sorted(tweets, key=lambda x: parse_time(x['created_at']))
                    tweets = tweets_sorted[-300:]

                    u_bloc = add_bloc_sequences(
                        tweets,
                        all_bloc_symbols=all_bloc_symbols,
                        **gen_bloc_params
                    )

                    user_obj = {
                        'user_id': user_id,
                        'user_class': user_class,
                        'user_representation': {},
                        'tweet_details': {}
                    }

                    semantic_analyses = tweets_semantic_analysis(u_bloc['tweets'])
                    for i, (tweet, semantic_analysis) in enumerate(zip(u_bloc['tweets'], semantic_analyses)):

                        user_obj['tweet_details'][i] = {
                            'action' : tweet['bloc']['bloc_sequences_short']['action'],
                            'content_syntactic' : tweet['bloc']['bloc_sequences_short']['content_syntactic'],
                            'change' : tweet['bloc']['bloc_sequences_short']['change'],
                            'topics' : semantic_analysis['topics'],
                            'topic_symbol' : semantic_analysis['topic_symbol'],
                            'entities' : semantic_analysis['entities'],
                            'entity_symbols' : semantic_analysis['entity_symbols'],
                            'sentiments' : semantic_analysis['sentiments'],
                            'text' : semantic_analysis['text'],
                            'exclusive_text' : semantic_analysis['exclusive_text'],
                            'created_at' : tweet['created_at'],
                            'tweet_id': tweet['id'],
                        }

                    user_obj['user_representation'] = build_user_representation(
                        user_obj['tweet_details']
                    )
                    
                    out_path = out_dir / f"{user_id}.json"
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(user_obj, f, indent=4, ensure_ascii=False)

                except Exception as e:
                    logger.exception("Failed processing a user from dataset %s", tweets_file)
                    genericErrorInfo()


if __name__ == "__main__":
    preprocess_data(src, path_to_dataset)
