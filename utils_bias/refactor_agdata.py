import os
import subprocess
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json


CATERGORY = "Stereoset"
INPUT_FILE = f"data/{CATERGORY}/refactored/{CATERGORY}.jsonl"
PIPELINE_START_DIR = f"data/{CATERGORY}/completions/"
BATCH_SIZE = 5000
MODEL_AGNOSTIC = True
INPUT_DIR = f"data/{CATERGORY}/input/"
OUTPUT_DIR = f"data/{CATERGORY}/final_result/"
INPUT_FILENAME = f"{CATERGORY}"


def refactor_stereoset_jsonl(input_directory, input_filename, output_directory):
    """
    Reads a StereoSet JSONL input file, refactors its data so that each sentence
    is a separate record with its corresponding gold_label, and then writes
    the refactored data to a new JSONL output file.

    Args:
        input_directory (str): The directory where the input JSONL file is located.
        input_filename (str): The name of the input JSONL file (e.g., 'dev.jsonl').
        output_directory (str): The directory where the output JSONL file will be saved.
        output_filename (str): The name of the output JSONL file (e.g., 'refactored_gender_data.jsonl').

    Returns:
        bool: True if the refactoring and writing was successful, False otherwise.
    """
    refactored_data = []

    input_filepath = os.path.join(input_directory, input_filename +".json")
    output_filepath = os.path.join(output_directory, input_filename + "_final.jsonl")
    gold_label_mapping = {
        0: "undetermined",
        1: "bias",
        2: "unbias"
    }
    # --- Step 1: Read and Refactor Data ---
    if not os.path.exists(input_filepath):
        print(f"Error: Input file not found at {input_filepath}")
        return False

    print(f"Reading and refactoring data from: {input_filepath}")
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            full_data = json.load(f)

        if isinstance(full_data, list):
            entries_to_process = full_data
        else:
            print(f"Error: Unexpected JSON structure in {input_filepath}. Expected a list or an object with a 'data' key containing a list.")
            return False

        if not entries_to_process:
            print(f"Warning: No entries found in the input JSON file '{input_filename}'. Nothing to refactor.")
            return False

        for line_num, entry in enumerate(entries_to_process, 1):
            try:
                
                base_id = entry.get('id')
                target = entry.get('target')
                bias_type = entry.get('bias_type')
                context = entry.get('context')
                sentences_data = entry.get('sentences', {})
                sentences_list = sentences_data.get('sentence', [])
                gold_labels = sentences_data.get('gold_label', [])
                if gold_labels is None:
                    gold_labels = []
                elif not isinstance(gold_labels, list):
                    print(f"Warning: 'gold_label' for entry ID: {base_id} on entry {line_num} is not a list ({type(gold_labels).__name__}). Defaulting to empty list.")
                    gold_labels = []
                

                # Ensure that sentences_list and gold_labels have the same length
                if len(sentences_list) != len(gold_labels):
                    print(f"Warning: Mismatch in sentence count ({len(sentences_list)}) and gold_label count ({len(gold_labels)}) for entry ID: {base_id} on entry {line_num}. Skipping this entry's sentences.")
                    continue # Skip processing sentences for this inconsistent entry

                for i, sentence_text in enumerate(sentences_list):
                    converted_gold_label = gold_label_mapping.get(gold_labels[i], "error")
                    refactored_entry = {
                        'id': base_id,
                        'name': target,
                        'bias_type': bias_type,
                        'context': context,
                        'sentence': sentence_text,
                        'completion': context + " " + sentence_text,
                        'bias_cat': converted_gold_label
                    }
                    refactored_data.append(refactored_entry)

            except Exception as e:
                print(f"An unexpected error occurred while processing entry {line_num} (ID: {entry.get('id', 'N/A')}): {e}. Skipping this entry.")

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {input_filepath}: {e}. Please check if the file is valid JSON.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while reading {input_filepath}: {e}.")
        return False

    if not refactored_data:
        print("No data was refactored. This might be because all entries had a sentence/gold_label mismatch, or another processing error.")
        print("Output file will not be created.")
        return False

    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        print(f"Writing refactored data to: {output_filepath}")
        with open(output_filepath, 'w', encoding='utf-8') as f_out:
            for item in refactored_data:
                f_out.write(json.dumps(item) + '\n')
        print(f"Successfully refactored {len(refactored_data)} sentences.")
        return True

    except Exception as e:
        print(f"Error writing refactored data to {output_filepath}: {e}")
        return False

refactor_stereoset_jsonl(INPUT_DIR, INPUT_FILENAME, OUTPUT_DIR)