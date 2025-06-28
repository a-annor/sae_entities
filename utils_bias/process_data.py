import os
os.environ["DISABLE_TORCH_COMPILE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch
if hasattr(torch, 'compile'):
    torch.compile = lambda model, *args, **kwargs: model
print(torch.__version__)

import subprocess
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

CATERGORY = "Race_ethnicity"
INPUT_FILE = f"data/{CATERGORY}/input/{CATERGORY}.jsonl"
PIPELINE_START_DIR = f"data/{CATERGORY}/prompts/"
BATCH_SIZE = 5000

PIPELINE_STAGES = [
    {
        "script": "utils_bias/generate_completions.py",
        "input_dir": f"data/{CATERGORY}/prompts/",
        "output_dir": f"data/{CATERGORY}/completions/"
    },
    {
        "script": "utils_bias/sentiment.py",
        "input_dir": f"data/{CATERGORY}/completions/",
        "output_dir": f"data/{CATERGORY}/sentiment/" 
    },
    {
        "script": "utils_bias/judge_bias.py",
        "input_dir": f"data/{CATERGORY}/sentiment/",
        "output_dir": f"data/{CATERGORY}/judged/" 
    },
    {
        "script": "utils_bias/bias_assignment.py",
        "input_dir": f"data/{CATERGORY}/judged/",
        "output_dir": f"data/{CATERGORY}/result/"
    }
]

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


# def run_script(script_name):
#     """
#     Runs a Python script using the same interpreter and captures its output.

#     Args:
#         script_name (str): The name of the script to run.

#     Raises:
#         RuntimeError: If the script returns a non-zero exit code.
#     """
#     print(f"\n----- Running {script_name} -----")
#     # Use sys.executable to ensure the same Python interpreter is used.
#     process = subprocess.run(
#         [sys.executable, script_name],
#         capture_output=True,
#         text=True,
#         check=False  # Do not raise CalledProcessError automatically
#     )

#     # Print stdout and stderr for debugging purposes
#     print("--- STDOUT ---")
#     print(process.stdout)
#     if process.stderr:
#         print("--- STDERR ---")
#         print(process.stderr, file=sys.stderr)

#     if process.returncode != 0:
#         raise RuntimeError(f"{script_name} failed with exit code {process.returncode}")

#     print(f"----- Finished {script_name} successfully -----")
def run_script(script_name, input_dir, output_dir):
    """Runs a Python script with specified input and output directories."""
    print(f"\n----- Running {script_name} -----")
    print(f"Input Dir: {input_dir}\nOutput Dir: {output_dir}")
    
    # Ensure the output directory for this step exists
    os.makedirs(output_dir, exist_ok=True)
    
    command = [
        sys.executable,
        script_name,
        "--input-dir",
        input_dir,
        "--output-dir",
        output_dir
    ]
    
    process = subprocess.run(
        command, # Pass the full command with arguments
         check=False
    )
    print("--- STDOUT ---")
    print(process.stdout)
    if process.stderr:
        print("--- STDERR ---", file=sys.stderr)
        print(process.stderr, file=sys.stderr)
    if process.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {process.returncode}")
    print(f"----- Finished {script_name} successfully -----")

# def main():
#     """
#     Main function to run the entire data processing pipeline.
#     """
#     try:
#         # Step 0 spli into batches
#         split_input_file(MAIN_INPUT_FILE, PIPELINE_START_DIR, BATCH_SIZE)

#         # Step 1: Generate completions from prompts
#         run_script("utils_bias/generate_completions.py")

#         # Step 2: Perform sentiment analysis on completions
#         run_script("utils_bias/sentiment.py")

#         # Step 3: Judge the bias of the completions
#         run_script("utils_bias/judge_bias.py")

#         # Step 4: Assign final bias category based on sentiment and judge score
#         run_script("utils_bias/bias_assignment.py")

#         print("\nPipeline completed successfully!")
#         print("Final results are located in the 'data/final_result' directory.")

#     except RuntimeError as e:
#         print(f"\nAn error occurred during the pipeline execution: {e}", file=sys.stderr)
#         sys.exit(1)
#     except FileNotFoundError as e:
#         print(f"\nFile not found: {e}. Make sure all scripts are in the same directory.", file=sys.stderr)
#         sys.exit(1)

def main():
    """Main function to run the entire data processing pipeline in batches."""
    try:
        split_input_file(INPUT_FILE, PIPELINE_START_DIR, BATCH_SIZE)

        #Loop through the pipeline stages
        for stage in PIPELINE_STAGES:
            run_script(
                script_name=stage["script"],
                input_dir=stage["input_dir"],
                output_dir=stage["output_dir"]
            )

        print("\nBatch processing completed successfully!")
        print(f"Processed {CATERGORY} batch files are now in ready for review.")
        print("Run the 'merge_results.py' script to combine them.")

    except (RuntimeError, FileNotFoundError) as e:
        print(f"\nAn error occurred during the pipeline execution: {e}", file=sys.stderr)
        sys.exit(1)



if __name__ == "__main__":
    main()