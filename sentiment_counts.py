from test_gemma_2_2b import bias_categories
from collections import defaultdict
from rapidfuzz import fuzz
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
import spacy
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nlp = spacy.load("en_core_web_sm")
import json


protect_attr_lookup = [category for category, terms in bias_categories.items() for term in terms]

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

def text_sentiment_counts(text):
    protect_attr_counts=0
    result = sentiment_pipeline(text)[0]
 
    label = result["label"].upper()
    score = result["score"]

    if "LABEL_2" in label:
        sentiment_scores=score
    elif "LABEL_0" in label:
        sentiment_scores=-score
    else:
        sentiment_scores=0.0

    doc = nlp(text)
    for token in doc:
        if token.ent_type_ != "" or token.pos_ in ["PROPN", "NOUN", "PRON"]: # If named entity (e.g. person or org), if proper noun (names, places), nouns and pronouns
            # print("token", token.text)
            protect_attr = fuzzy_match_protect_attr(token.text.lower())
            # print("fuzzy", protect_attr)
            if not protect_attr:
                continue
            else:
                protect_attr_counts += 1
   
    return sentiment_scores, protect_attr_counts


def append_senitment_count(input_file, output_file):
    with open(f"{input_file}.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        sentiment, count = text_sentiment_counts(entry["completion"])
        entry["sentiment"] = sentiment
        entry["protect_attr_count"] = count

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(output_file, f, indent=2, ensure_ascii=False)

append_senitment_count("./generated_race_bias_data_updated.json")