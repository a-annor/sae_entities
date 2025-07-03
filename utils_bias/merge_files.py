import glob
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

CATERGORY = "Race_ethnicity"

input_pattern = f"data/{CATERGORY}/result/*.jsonl"
output_file = f"data/final_result/{CATERGORY}_completion_sentiment_judged_final.jsonl"

with open(output_file, 'w', encoding='utf-8') as outfile:
    for fname in glob.glob(input_pattern):
        with open(fname, 'r', encoding='utf-8') as infile:
            for line in infile:
                outfile.write(line)

print(f"All JSONL files matching '{input_pattern}' have been joined into '{output_file}'.")
