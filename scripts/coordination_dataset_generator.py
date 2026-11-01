import gzip
import os
import json
import spacy
import re
import csv
import logging

from datetime import datetime, timedelta
from bloc.generator import add_bloc_sequences
from bloc.util import (
    get_default_symbols,
    getDictFromJson,
    genericErrorInfo
)
from info_ops_tk.util import get_bloc_lite_twt_frm_full_twt
from info_ops_tk.util import get_bloc_lite_twt
from transformers import pipeline
from transformers.utils import logging as hf_logging
from textblob import TextBlob

logger = logging.getLogger('bloc.bloc')

# =========================
# CONFIG
# =========================

name_mapper = {
    '2018_10/iranian' : 'Iran_1',
    '2019_01/bangladesh_201901_1' : 'Bangladesh',
    '2019_01/iran_201901_X/iran_201901_1' : 'Iran_2',
    '2019_01/russia_201901_1' : 'Russia_1',
    '2019_01/venezuela_201901_1' : 'Venezuela_1',
    '2019_01/venezuela_201901_2' : 'Venezuela_2',
    '2019_06/catalonia_201906_1' : 'Catalonia',
    '2019_06/iran_201906_1' : 'Iran_3',
    '2019_06/iran_201906_2' : 'Iran_4',
    '2019_06/iran_201906_3' : 'Iran_5',
    '2019_06/venezuela_201906_1' : 'Venezuela_3',
    '2019_08/china_082019_1' : 'China_1',
    '2019_08/china_082019_2' : 'China_2',
    '2019_08/ecuador_082019_1' : 'Ecuador',
    '2019_08/egypt_uae_082019_1' : 'Egypt_UAE',
    '2019_08/spain_082019_1' : 'Spain',
    '2019_08/uae_082019_1' : 'UAE',
    '2020_03/ghana_nigeria_032020' : 'Ghana_Nigeria',
    '2020_08/qatar_082020' : 'Qatar',
    '2020_09/ira_092020' : 'Russia_2',
    '2020_09/iran_092020' : 'Iran_6',
    '2020_09/thailand_092020' : 'Thailand',
    '2020_12/GRU_202012' : 'Russia_3',
    '2020_12/IRA_202012' : 'Russia_4',
    '2020_12/armenia_202012' : 'Armenia',
    '2020_12/iran_202012' : 'Iran_7',
    '2021_12/CNCC_0621_YYYY/CNCC_0621_2021' : 'China_3',
    '2021_12/CNHU_0621_YYYY/CNHU_0621_2020' : 'China_5',
    '2021_12/CNHU_0621_YYYY/CNHU_0621_2021' : 'China_4',
    '2021_12/MX_0621_YYYY/MX_0621_2019' : 'Mexico_1',
    '2021_12/Venezuela_0621_YYYY/Venezuela_0621_2020' : 'Venezuela_5',
    '2021_12/Venezuela_0621_YYYY/Venezuela_0621_2021' : 'Venezuela_4',
    '2021_12/uganda_0621_YYYY/uganda_0621_2019' : 'Uganda_1',
    '2021_12/uganda_0621_YYYY/uganda_0621_2020' : 'Uganda_2'
}

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

all_bloc_symbols = get_default_symbols()

src =  [
    "2019_01/iran_201901_X/iran_201901_1",
    "2019_06/iran_201906_1",
    "2019_08/china_082019_1",
    "2019_08/uae_082019_1",
    "2019_08/ecuador_082019_1",
    "2021_12/CNHU_0621_YYYY/CNHU_0621_2021",
    "2021_12/CNHU_0621_YYYY/CNHU_0621_2020"
]
path_to_dataset = "coordination_dataset/YYYY_MM"

maximum_tweets_per_driver_and_control = 1000
min_tweets = 20

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
        txt = get_twt_text_exclusively(txt, twt.get('entities', {}))
        txt = re.sub(r'^\s*RT\s*:?\s*', '', txt, flags=re.IGNORECASE)
        txt = re.sub(r'https?://\S+', '', txt).strip()
    except Exception:
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
            'text': twt.get('full_text') or twt.get('text', ''),
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

def convert_to_ISO_time(date_str):
    return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")

def get_driver_control_per_day_tweets(filename):
    all_drivers = set()
    all_tweets = []

    driver_control_tweets = filename.replace('control_driver_users.csv', 'control_driver_tweets.jsonl.gz')
    with gzip.open(driver_control_tweets, 'rb') as f:
        for tweet in f:
            
            tweet = getDictFromJson(tweet)
            tweet = get_bloc_lite_twt_frm_full_twt(tweet)

            user_id = tweet['user']['id']

            all_drivers.add(user_id)
            
            all_tweets.append( tweet )
            
    return {
        'total_control': len(all_drivers),
        'control_driver_posts': all_tweets
    }

def get_driver_per_day_tweets(filename):
    
    if( os.path.exists(filename) is False ):
        return {}

    all_drivers = set()
    all_tweets = []

    with gzip.open(filename, 'rt') as infile:
        csvobj = csv.reader(infile)
        header = next(csvobj)
        hmap = {}
        [ hmap.update({ h[1]: h[0] }) for h in enumerate(header) ]
    
        while True:
            row = ''
            try:
                row = next(csvobj)
            except StopIteration:
                break
            except:
                genericErrorInfo()

            if( len(row) == 0 ):
                continue

            user_id = row[ hmap['userid'] ]

            tweet_time = row[ hmap['tweet_time'] ]

            if(not tweet_time):
                continue

            all_drivers.add(user_id)

            tweet = get_bloc_lite_twt(row, header)

            tweet['tweet_time'] = tweet_time
            all_tweets.append( tweet )

    return {
        'driver_posts': all_tweets,
        'total_drivers': len(all_drivers)
    }

def get_all_per_user_tweets(all_tweets, max_tweets, min_tweets):
    users_tweets = {}
    final_users_tweets = {}
    
    for twt in all_tweets:
        uid = twt['user']['id']
        users_tweets.setdefault(uid, [])
        users_tweets[uid].append( twt )

    for uid in users_tweets:
        if( len(users_tweets[uid]) > max_tweets ):
            users_tweets[uid] = sorted(
                users_tweets[uid],
                key=lambda k: convert_to_ISO_time(k['created_at']),
                reverse=True
            )[:max_tweets]
            
    for uid in users_tweets:
        if( len(users_tweets[uid]) < min_tweets ):
            continue

        final_users_tweets[uid] = users_tweets[uid]

    return final_users_tweets

def get_info_ops_drivers_control_users_tweets(file_path, dataset_name, config):
    config['datasetName'] = dataset_name
    max_tweets = config.get('maximum_tweets_per_driver_and_control', 5000)
    min_tweets = config.get('min_tweets', 20)
    campaign = dataset_name

    # File paths
    control_path = os.path.join(file_path, dataset_name, 'DriversControl', 'control_driver_users.csv')
    driver_path = os.path.join(file_path, dataset_name, 'driver_tweets.csv.gz')

    # Load user posts
    driver_posts = get_driver_per_day_tweets(driver_path)
    control_posts = get_driver_control_per_day_tweets(control_path)
        

    control_users = get_all_per_user_tweets(control_posts['control_driver_posts'], max_tweets, min_tweets)
    driver_users = get_all_per_user_tweets(driver_posts['driver_posts'], max_tweets, min_tweets)

    for user_id in control_users:
        tweets = control_users[user_id]
        u_bloc = add_bloc_sequences(tweets, all_bloc_symbols=all_bloc_symbols, **gen_bloc_params) # u_bloc =>  (bloc, tweets, bloc_segments, created_at_utc, screen_name, user_id, bloc_symbols_version, more_details)
        user_class = 'control'

        user_obj = {
            'user_id': user_id,
            'user_class': user_class,
            'user_representation': {},
            'tweet_details': {}
        }

        # print(u_bloc['tweets'][0])
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
                'tweet_id': tweet['id']
            }

        user_obj['user_representation'] = build_user_representation(
            user_obj['tweet_details']
        )

        # dataset_to_save = name_mapper[dataset_name]
        out_path = f"analyzed_users_v1/{dataset_name}"
        os.makedirs(out_path, exist_ok=True)
        out_file = f"{out_path}/{user_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(user_obj, f, indent=4, ensure_ascii=False)

    for user_id in driver_users:
        tweets = driver_users[user_id]
        u_bloc = add_bloc_sequences(tweets, all_bloc_symbols=all_bloc_symbols, **gen_bloc_params) # u_bloc =>  (bloc, tweets, bloc_segments, created_at_utc, screen_name, user_id, bloc_symbols_version, more_details)
        user_class = 'driver'

        user_obj = {
            'user_id': user_id,
            'user_class': user_class,
            'user_representation': {},
            'tweet_details': {}
        }

        # print(u_bloc['tweets'][0])
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
                'tweet_id' : tweet['id']
            }

        user_obj['user_representation'] = build_user_representation(
            user_obj['tweet_details']
        )

        # dataset_to_save = name_mapper[dataset_name]
        out_path = f"analyzed_users_v1/{dataset_name}"
        os.makedirs(out_path, exist_ok=True)
        out_file = f"{out_path}/{user_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(user_obj, f, indent=4, ensure_ascii=False)

def get_bloc_infoOps(config):   
    path_to_dataset = config['path_to_dataset']
    src = config['src']
    
    for datasetName in src:
        print(f"Processing {name_mapper[datasetName]}...")
        file_location = os.path.join(path_to_dataset, datasetName)
        
        control = os.path.exists(os.path.join(file_location, 'DriversControl/control_driver_users.csv'))
        drivers = os.path.exists(os.path.join(file_location, 'driver_tweets.csv.gz'))
        
        get_info_ops_drivers_control_users_tweets(path_to_dataset, datasetName, config)


if __name__ == "__main__":
    change_prop = {
        'src': src,
        'path_to_dataset': path_to_dataset,
        'maximum_tweets_per_driver_and_control': maximum_tweets_per_driver_and_control,
        'min_tweets': min_tweets
    }
    get_bloc_infoOps(change_prop)
