from test_gemma_2_2b import bias_categories
from collections import defaultdict
from rapidfuzz import fuzz
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device_map="cuda", torch_dtype=torch.bfloat16)
import spacy
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nlp = spacy.load("en_core_web_sm")
import json


protect_attr_lookup = [term for terms in bias_categories.values() for term in terms]

def fuzzy_match_protect_attr(word, threshold=95):
    word = word.lower()
    for term in protect_attr_lookup:
        if len(term) <= 3 or len(word) <= 3: # for short words use exact match
            if word == term:
                return term  
        else:
            if fuzz.partial_ratio(word.lower(), term) >= threshold: # for longer words use fuzzy matching 
                return term
    return None

def count_protected_terms(text):
    text = text.lower()
    count = 0
    for term in protect_attr_lookup:
        count += text.count(term)
    return count

def text_sentiment_counts(text):
    protect_attr_counts=0
    result = sentiment_pipeline(text, top_k=3)
    for r in result:
        if r["label"] == "LABEL_2":
            sentiment_scores= r["score"]
    # for r in result:
    #     if r["label"] == "LABEL_2":
    #         sentiment_scores = r["score"]
    #     else:
    #         sentiment_scores = "error"
    # result = sentiment_pipeline(text)[0]
 
    # label = result["label"].upper()
    # score = result["score"]

    # if "LABEL_2" in label:
    #     sentiment_scores=score
    # elif "LABEL_0" in label:
    #     sentiment_scores=-score
    # else:
    #     sentiment_scores=0.0

    protect_attr_counts = count_protected_terms(text)
    # doc = nlp(text)
    # for token in doc:
    #     # if token.ent_type_ != "" or token.pos_ in ["PROPN", "NOUN", "PRON"]: # If named entity (e.g. person or org), if proper noun (names, places), nouns and pronouns
    #         # print("token", token.text)
    #     protect_attr = fuzzy_match_protect_attr(token.text.lower())
    #     # print("fuzzy", protect_attr)
    #     if protect_attr:
    #         protect_attr_counts += 1

    return sentiment_scores, protect_attr_counts


def append_senitment_count(input_file):
    with open(f"z_data/generated_data/{input_file}.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        sentiment, count = text_sentiment_counts(entry["completion"])
        entry["sentiment"] = sentiment
        entry["protect_attr_count"] = count
    

    with open(f"z_data/sentiment_counts_2/{input_file}_rule_based.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

append_senitment_count("generated_race_bias_data_updated")
append_senitment_count("generated_nationality_bias_data_updated")