import os
os.environ["DISABLE_TORCH_COMPILE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch
if hasattr(torch, 'compile'):
    torch.compile = lambda model, *args, **kwargs: model
print(torch.__version__)

from transformers import  pipeline
# import torch
import spacy
nlp = spacy.load("en_core_web_sm")
import json
from tqdm import tqdm

from argparse import ArgumentParser, Namespace
import glob



def text_sentiment(text, sentiment_pipeline):
    result = sentiment_pipeline(text.strip().lower(), top_k=3)
    
    for r in result:
        # print(r["label"])
        if r["label"] == "LABEL_0":
            sentiment_score= -r["score"]

    return round(sentiment_score,2)

def append_sentiment(input_file):
    input_path = f"data/Race_ethnicity/completions/{input_file}.jsonl"
    output_path = f"data/Race_ethnicity/sentiment/{input_file}_sentiment.jsonl"

    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            entry = json.loads(line)
            sentiment = text_sentiment(entry["completion"], sentiment_pipeline)
            entry["sentiment"] = sentiment
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

def process_file(input_path, output_path, sentiment_pipeline):
    """
    Reads a JSONL file, appends a sentiment score to each entry,
    and writes the result to a new JSONL file.
    """
    print(f"Processing {os.path.basename(input_path)}...")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        # Using tqdm for a progress bar
        for line in tqdm(f_in, desc="Analyzing sentiment", unit=" lines"):
            entry = json.loads(line)
            # Pass the 'completion' field to the sentiment analyzer
            sentiment = text_sentiment(entry.get("completion", ""), sentiment_pipeline)
            entry["sentiment"] = sentiment
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Finished processing. Output saved to {output_path}")

# append_sentiment("Race_ethnicity_completion")
# append_senitment_count("generated_nationality_bias_data_updated")

def main(args: Namespace):
    """
    Main function to find all files in the input directory and process them.
    """
    # Initialize the model pipeline once
    sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")# device_map="cpu", torch_dtype=torch.float32)

    
    # Find all .jsonl files in the input directory
    input_files = glob.glob(os.path.join(args.input_dir, "*.jsonl"))

    if not input_files:
        print(f"Warning: No .jsonl files found in {args.input_dir}")
        return

    print(f"Found {len(input_files)} files to process.")

    for input_path in input_files:
        # Create the corresponding output file path
        file_name = os.path.basename(input_path)
        output_path = os.path.join(args.output_dir, file_name)
        
        process_file(input_path, output_path, sentiment_pipeline)

# --- Script Entry Point ---

if __name__ == "__main__":
    parser = ArgumentParser(description="Sentiment analysis script for text completions.")
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        default="/home/ana42/rds/hpc-work/sae_entities/data/Race_ethnicity/completions",
        help="Directory containing the input .jsonl files (e.g., completions)."
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        default="/home/ana42/rds/hpc-work/sae_entities/data/Race_ethnicity/sentiment",
        help="Directory where the output files with sentiment scores will be saved."
    )
    
    args = parser.parse_args()
    main(args)