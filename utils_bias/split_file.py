import os
import subprocess
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


CATERGORY = "Race_ethnicity"
INPUT_FILE = f"data/{CATERGORY}/input/{CATERGORY}.jsonl"
PIPELINE_START_DIR = f"data/{CATERGORY}/prompts/"
BATCH_SIZE = 5000


def split_input_file(input_file, output_dir, batch_size):
    """Splits the main input file into batch files in the target directory."""
    print(f"----- Splitting {input_file} into batches of {batch_size} lines -----")
    with open(input_file, 'r', encoding='utf-8') as f_in:
        batch_num = 1
        batch_lines = []
        for i, line in enumerate(f_in):
            batch_lines.append(line)
            if (i + 1) % batch_size == 0:
                batch_path = os.path.join(output_dir, f'batch_{batch_num}.jsonl')
                with open(batch_path, 'w', encoding='utf-8') as f_out:
                    f_out.writelines(batch_lines)
                batch_num += 1
                batch_lines = []
        
        if batch_lines: # Write the final, smaller batch
            batch_path = os.path.join(output_dir, f'batch_{batch_num}.jsonl')
            with open(batch_path, 'w', encoding='utf-8') as f_out:
                f_out.writelines(batch_lines)
    print(f"----- Created {batch_num} batch files in {output_dir} -----")

split_input_file(INPUT_FILE, PIPELINE_START_DIR, BATCH_SIZE)