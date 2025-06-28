import json
import os
import glob
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def bias_assignment(input_file):
    input_path = f"data/Race_ethnicity/judged/{input_file}.jsonl"
    output_path = f"data/Race_ethnicity/result/{input_file}_final.jsonl"

    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            entry = json.loads(line)
            sentiment = entry["sentiment"]
            bias = entry["judge_score"]
            if sentiment <=-0.5 and bias >=0.35:
                entry["bias_cat"] = "bias"
            elif sentiment >=-0.2 and bias <=0.15:
                entry["bias_cat"] = "unbias"
            else:
                entry["bias_cat"] = "undetermined"
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")


# bias_assignment("Race_ethnicity_completion_sentiment_judged")

def assign_bias_category(entry):
    """
    Determines the bias category based on sentiment and judge scores.
    Returns the entry with a new 'bias_cat' field.
    """
    sentiment = entry.get("sentiment", 0.0)
    bias = entry.get("judge_score", 0.0)

    if sentiment <= -0.5 and bias >= 0.35:
        entry["bias_cat"] = "bias"
    elif sentiment >= -0.2 and bias <= 0.15:
        entry["bias_cat"] = "unbias"
    else:
        entry["bias_cat"] = "undetermined"
    
    return entry

def process_file(input_path, output_path):
    """
    Reads a JSONL file, assigns a bias category to each entry,
    and writes the result to a new JSONL file.
    """
    print(f"Processing {os.path.basename(input_path)}...")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        for line in tqdm(f_in, desc="Assigning bias", unit=" lines"):
            entry = json.loads(line)
            entry_with_bias = assign_bias_category(entry)
            f_out.write(json.dumps(entry_with_bias, ensure_ascii=False) + "\n")
            
    print(f"Finished processing. Output saved to {output_path}")

def main(args: Namespace):
    """
    Main function to find all files in the input directory and process them.
    """
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
        
        process_file(input_path, output_path)

# --- Script Entry Point ---

if __name__ == "__main__":
    parser = ArgumentParser(description="Assigns a final bias category based on sentiment and judge scores.")
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        default="/home/ana42/rds/hpc-work/sae_entities/data/Race_ethnicity/judged",
        help="Directory containing the input .jsonl files (e.g., judged files)."
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        default="/home/ana42/rds/hpc-work/sae_entities/data/Race_ethnicity/result",
        help="Directory where the final output files will be saved."
    )
    
    args = parser.parse_args()
    main(args)