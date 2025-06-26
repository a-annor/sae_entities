import os
os.environ["DISABLE_TORCH_COMPILE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch
if hasattr(torch, 'compile'):
    torch.compile = lambda model, *args, **kwargs: model
print(torch.__version__)

from transformers import  pipeline
# import torch
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")# device_map="cpu", torch_dtype=torch.float32)
import spacy
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nlp = spacy.load("en_core_web_sm")
import json


def text_sentiment(text, sentiment_pipeline):
    result = sentiment_pipeline(text.strip().lower(), top_k=3)
    
    for r in result:
        print(r["label"])
        if r["label"] == "LABEL_0":
            sentiment_score= -r["score"]

    return round(sentiment_score,2)

def append_sentiment(input_file):
    input_path = f"z_my_data/test_prompt_data_completions/{input_file}.jsonl"
    output_path = f"z_my_data/test_prompt_data_sentiment/{input_file}_sentiment.jsonl"

    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            entry = json.loads(line)
            sentiment = text_sentiment(entry["completion"], sentiment_pipeline)
            entry["sentiment"] = sentiment
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

append_sentiment("test_Race_ethnicity_completion")
# append_senitment_count("generated_nationality_bias_data_updated")