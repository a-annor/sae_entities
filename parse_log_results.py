
import matplotlib.pyplot as plt
import re
import json
import pandas as pd
import os
import datetime
from transformers import pipeline
from utils_bias.sentiment import text_sentiment
from utils_bias.judge_bias import judge_bias

import sys
# Ensure the script can find utility modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

plt.style.use('ggplot') # Set the style globally here

# def parse_experiment_output(file_path):
#     """
#     Parses the raw text output from the experiment to extract relevant data.
#     """
#     with open(file_path, 'r') as f:
#         content = f.read()

#     data = []
#     # Split content by prompt sections
#     prompt_sections_raw = re.split(r'================================================================================\n\[PROMPT\]: (.*?)\n================================================================================', content, flags=re.DOTALL)

#     for i in range(1, len(prompt_sections_raw), 2):
#         prompt_text = prompt_sections_raw[i].strip()
#         section_content = prompt_sections_raw[i+1]

#         original_score_match = re.search(r'\[ORIGINAL COMPLETION\] \(Score: ([\d.-]+)\):', section_content)
#         original_score = float(original_score_match.group(1)) if original_score_match else None

#         coeff_blocks = re.findall(r'---\sCoefficient: ([\d.-]+)\s---\n\[POS STEERED\] \(Score: ([\d.-]+)\):.*?\n\[NEG STEERED\] \(Score: ([\d.-]+)\):', section_content, re.DOTALL)

#         for coeff_str, pos_score_str, neg_score_str in coeff_blocks:
#             coeff = float(coeff_str)
#             pos_score = float(pos_score_str)
#             neg_score = float(neg_score_str)

#             data.append({
#                 'prompt': prompt_text,
#                 'coeff': coeff,
#                 'original_score': original_score,
#                 'pos_steered_score': pos_score,
#                 'neg_steered_score': neg_score
#             })
#     return pd.DataFrame(data)
sentiment_score=True
if sentiment_score:
    sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device ="cuda")

def parse_experiment_output(file_path,sentiment_score, filename, output_dir=None):
    """
    Parses the raw text output from the experiment to extract relevant data.
    """
    with open(file_path, 'r') as f:
        content = f.read()

    data = []
    # Split content by prompt sections
    prompt_sections_raw = re.split(r'================================================================================\n\[PROMPT\]: (.*?)\n================================================================================', content, flags=re.DOTALL)

    for i in range(1, len(prompt_sections_raw), 2):
        prompt_text = prompt_sections_raw[i].strip()
        print(prompt_text)
        section_content = prompt_sections_raw[i+1]

        # Extract ORIGINAL COMPLETION
        original_match = re.search(
            r'\[ORIGINAL COMPLETION\] \(Score: [\d.-]+\):\n(.*?)(?:\n\n|--- Coefficient:)', 
            section_content, re.DOTALL
        )
        original_completion = original_match.group(1).strip() if original_match else None
        original_completion = original_completion.replace('<bos>', '').replace('<eos>', '').replace('<end_of_turn>', '').strip()
        # Find all coefficient blocks
        coeff_blocks = re.findall(
            r'--- Coefficient: ([\d\.-]+) ---\s*'
            r'\[POS STEERED\] \(Score: ([\d\.-]+)\):\s*(.*?)\s*'
            r'\[NEG STEERED\] \(Score: ([\d\.-]+)\):\s*(.*?)(?=(?:--- Coefficient:|\Z))',
            section_content, re.DOTALL
        )
        if not coeff_blocks:
            coeff_blocks = re.findall(
                r'--- Coefficient: ([\d\.-]+) ---\s*'
                r'\[BIAS STEERED\] \(Score: ([\d\.-]+)\):\s*(.*?)\s*'
                r'\[UNBIAS STEERED\] \(Score: ([\d\.-]+)\):\s*(.*?)(?=(?:--- Coefficient:|\Z))',
                section_content, re.DOTALL
            )
            label_1= 'bias'
            label_2 = 'unbias'
            if not coeff_blocks:
                print(f"  Warning: No steered blocks found for prompt: '{prompt_text}'. Skipping.")
                continue # Skip to the next prompt section
        else:
            label_1= 'pos'
            label_2 = 'neg'

        # for coeff_str, bias_steered_text, unbias_steered_text in coeff_blocks:
        for coeff_str, _, bias_steered_text, _, unbias_steered_text in coeff_blocks:
            coeff = float(coeff_str)
            # Remove trailing whitespace and ensure text extraction is clean
            bias_steered_text = bias_steered_text.replace('<bos>', '').replace('<eos>', '').replace('<end_of_turn>', '').strip()
            unbias_steered_text = unbias_steered_text.replace('<bos>', '').replace('<eos>', '').replace('<end_of_turn>', '').strip()
            print("BIAS: ", bias_steered_text)
            print("UNBIAS: ", unbias_steered_text)

            # Calculate fresh sentiment scores
            if sentiment_score:
                original_score= text_sentiment(original_completion, sentiment_pipeline) 
                bias_score = text_sentiment(bias_steered_text, sentiment_pipeline)
                unbias_score = text_sentiment(unbias_steered_text, sentiment_pipeline)
                print("SCORE: ", bias_score)
            
            else:
                original_score= judge_bias(original_completion, sentiment_pipeline) 
                bias_score = judge_bias(bias_steered_text, sentiment_pipeline)
                unbias_score = judge_bias(unbias_steered_text, sentiment_pipeline)
                print("SCORE: ", bias_score)
                

            data.append({
                'prompt': prompt_text,
                'coeff': coeff,
                'original_completion': original_completion,
                'bias_steered_completion': bias_steered_text,
                'unbias_steered_completion': unbias_steered_text,
                'original_score': original_score,
                f'{label_1}_steered_score': bias_score,
                f'{label_2}_steered_score': unbias_score
            })

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        jsonl_file_path = os.path.join(output_dir, f'parsed_{filename}.jsonl')
        with open(jsonl_file_path, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry) + '\n')
        print(f"Parsed data saved to {jsonl_file_path}")

    return pd.DataFrame(data)



def plot_sentiment_vs_coeff(df, sentiment_score, latent_type, output_dir="plots"):
    """
    Plots the sentiment scores against coefficients for each prompt.
    Generates a separate plot for each prompt with enhanced aesthetics.
    """
    os.makedirs(output_dir, exist_ok=True)

    unique_prompts = df['prompt'].unique()

    # Define a color palette for better distinction (can still be defined here)
    if latent_type=='sentiment':
        label_1 = 'pos'
        label_2 = 'neg'
        Label_1 = 'Pos'
        Label_2 = 'Neg'
    elif latent_type=='bias':
        label_1 = 'bias'
        label_2 = 'unbias'
        Label_1 = 'Biased'
        Label_2 = 'Unbiased'

    if sentiment_score:
        score_type ='Sentiment'
        score_type_file = 'sentiment'
    else:
        score_type ='LLM Judge'
        score_type_file = 'llm_judge'


    # colors = {
    #     f'{Label_1} Steered {score_type}': '#2CA02C', # Green
    #     f'{Label_2} Steered {score_type}': '#D62728', # Red
    #     f'Original {score_type}': '#1F77B4'    # Blue
    # }

    colors = {
        f'{Label_1} Steered {score_type}': '#D62728', # Green
        f'{Label_2} Steered {score_type}': '#2CA02C', # Red
        f'Original {score_type}': '#1F77B4'    # Blue
    }


    for prompt in unique_prompts:
        plt.figure(figsize=(12, 7)) # Create a new figure for each plot

        prompt_df = df[df['prompt'] == prompt].sort_values(by='coeff')

        # Plot POS STEERED sentiment
        plt.plot(prompt_df['coeff'], prompt_df[f'{label_1}_steered_score'],
                 marker='o', linestyle='-', color=colors[f'{Label_1} Steered {score_type}'],
                 linewidth=2, markersize=8, label=f'{Label_1} Steered {score_type}')

        # Plot NEG STEERED sentiment
        plt.plot(prompt_df['coeff'], prompt_df[f'{label_2}_steered_score'],
                 marker='X', linestyle='--', color=colors[f'{Label_2} Steered {score_type}'],
                 linewidth=2, markersize=8, label=f'{Label_2} Steered {score_type}')

        # Plot ORIGINAL sentiment as a horizontal line
        original_score_for_plot = prompt_df['original_score'].iloc[0]
        if pd.notna(original_score_for_plot):
            plt.axhline(y=original_score_for_plot, color=colors[f'Original {score_type}'], linestyle=':',
                        linewidth=2, label=f'Original {score_type} (Score: {original_score_for_plot:.3f})')

        # Customize title and labels
        plt.title(f'{score_type} Score vs. Steering Coefficient\nPrompt: "{prompt}"', fontsize=16, pad=20)
        plt.xlabel('Steering Coefficient', fontsize=14)
        plt.ylabel(f'{score_type} Score', fontsize=14)

        # Enhance grid
        plt.grid(True, linestyle='-', alpha=0.6)

        # Improve legend
        plt.legend(fontsize=11, frameon=True, borderpad=1)

        # Improve tick labels
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.ylim(-1, 1)

        # Add padding
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Sanitize prompt text for filename
        filename = re.sub(r'[^\w\s-]', '', prompt).replace(' ', '_')[:50]
        plt.savefig(os.path.join(output_dir, f'{filename}_{score_type_file}_plot.png'), dpi=300)
        plt.close()

    print(f"Individual prompt plots saved to the '{output_dir}' directory.")

# --- New function for average plot ---
def plot_average_sentiment(df, sentiment_score, latent_type, output_dir="plots"):
    """
    Plots the average positive and negative steered sentiment scores across all prompts.
    """
    
    if latent_type=='sentiment':
        label_1 = 'pos'
        label_2 = 'neg'
        Label_1 = 'Pos'
        Label_2 = 'Neg'
        average_df = df.groupby('coeff').agg(
        avg_pos_steered=(f'{label_1}_steered_score', 'mean'),
        avg_neg_steered=(f'{label_2}_steered_score', 'mean')
    ).reset_index().sort_values(by='coeff')
    elif latent_type=='bias':
        label_1 = 'bias'
        label_2 = 'unbias'
        Label_1 = 'Biased'
        Label_2 = 'Unbiased'
        average_df = df.groupby('coeff').agg(
        avg_bias_steered=(f'{label_1}_steered_score', 'mean'),
        avg_unbias_steered=(f'{label_2}_steered_score', 'mean')
    ).reset_index().sort_values(by='coeff')

    if sentiment_score:
        score_type ='Sentiment'
        score_type_file = 'sentiment'
    else:
        score_type ='LLM Judge'
        score_type_file = 'llm_judge'

    os.makedirs(output_dir, exist_ok=True)


    

    print("AVG COL: ", average_df.columns)
    plt.figure(figsize=(12, 7)) # Create a new figure for this plot

    # colors = { 
    #     f'{Label_1} Steered {score_type} (Average)': '#2CA02C',
    #     f'{Label_2} Steered {score_type} (Average)': '#D62728',
    # }

    colors = { 
        f'{Label_1} Steered {score_type} (Average)': '#D62728',
        f'{Label_2} Steered {score_type} (Average)': '#2CA02C',
    }

    # Plot average POS STEERED sentiment
    plt.plot(average_df['coeff'], average_df[f'avg_{label_1}_steered'],
             marker='o', linestyle='-', color=colors[f'{Label_1} Steered {score_type} (Average)'],
             linewidth=2, markersize=8, label=f'{Label_1} Steered {score_type} (Average)')

    # Plot average NEG STEERED sentiment
    plt.plot(average_df['coeff'], average_df[f'avg_{label_2}_steered'],
             marker='X', linestyle='--', color=colors[f'{Label_2} Steered {score_type} (Average)'],
             linewidth=2, markersize=8, label=f'{Label_2} Steered {score_type} (Average)')

    # Customize title and labels
    plt.title(f'Average {score_type} Score vs. Steering Coefficient (All Sample Prompts)', fontsize=16, pad=20)
    plt.xlabel(f'{score_type} Coefficient', fontsize=14)
    plt.ylabel(f'Average {score_type} Score', fontsize=14)

    # Enhance grid
    plt.grid(True, linestyle='-', alpha=0.6)

    # Improve legend
    plt.legend(fontsize=11, frameon=True, borderpad=1)

    # Improve tick labels
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.ylim(-1, 1)

    # Add padding
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(os.path.join(output_dir, f'average_{score_type_file}_plot.png'), dpi=300)
    plt.close()

    print(f"Average sentiment plot saved to the '{output_dir}' directory as 'average_sentiment_plot.png'.")


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    filename = 'steering_race_2_10_v4_2_describe'
    results_file = os.path.join(script_dir, f'{filename}.log')
    sentiment_score = True
    latent_type = "bias"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_output_directory = os.path.join(script_dir, f"plots_{filename}_{timestamp}")

    if not os.path.exists(results_file):
        print(f"Error: Results file not found at '{results_file}'.")
        print("Please ensure your log file is in the same directory as this script and named 'steering_pos_neg_20_v2.log'.")
    else:
        print(f"Parsing data from: {results_file}")
        df_results = parse_experiment_output(results_file,sentiment_score, filename, output_dir=plot_output_directory)
        print("DF columns:", df_results.columns)
        print("Data parsing complete. Generating plots...")

        plot_sentiment_vs_coeff(df_results, sentiment_score,latent_type, output_dir=plot_output_directory)
        plot_average_sentiment(df_results,sentiment_score,latent_type,  output_dir=plot_output_directory)

        print("\nAll plotting complete. Check the newly created folder for your plots.")