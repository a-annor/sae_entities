
import matplotlib.pyplot as plt
import re
import json
import pandas as pd
import os
import datetime
import seaborn as sns
import glob


import sys
# Ensure the script can find utility modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

plt.style.use('ggplot') # Set the style globally here


def load_jsonl_to_df(jsonl_path: str) -> pd.DataFrame:
    """
    Loads a JSONL file (one JSON object per line) into a pandas DataFrame.
    Renames columns to match plotting function expectations.
    """
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    df = pd.DataFrame(data)

    return df


def plot_perplexity_vs_coeff(
    df: pd.DataFrame,
    latent_type: str,
    output_dir: str = "plots_ppl",
):
    """
    Plots (completion) perplexity against coefficients for each prompt.
    Generates a separate plot per prompt, mirroring `plot_score_vs_coeff`.

    Expects the following columns in `df`:
        - 'prompt_text', 'coeff'
        - 'orig_ppl'
        - '<label_1>_ppl' and '<label_2>_ppl'
          where:
              if latent_type == 'sentiment': label_1='neg',  label_2='pos'
              if latent_type == 'bias'     : label_1='unbias', label_2='bias'
    """
    os.makedirs(output_dir, exist_ok=True)

    unique_prompts = df["prompt_text"].unique()

    if latent_type == "sentiment":
        label_1, label_2 = "neg", "pos"
        Label_1, Label_2 = "Neg", "Pos"
    elif latent_type == "bias":
        label_1, label_2 = "unbias", "bias"
        Label_1, Label_2 = "Unbiased", "Biased"
    else:
        raise ValueError(f"Unknown latent_type: {latent_type}")

    
    latent_id = df['latent_id'].iloc[0] 

    colours = {
        f"{Label_1} Steered PPL": "#D62728",  # red-ish
        f"{Label_2} Steered PPL": "#2CA02C",  # green-ish
        "Original PPL": "#1F77B4",            # blue
    }

    for prompt in unique_prompts:
        prompt_df = df[df["prompt_text"] == prompt].sort_values(by="coeff")

        plt.figure(figsize=(12, 7))

        # Steered curves
        plt.plot(
            prompt_df["coeff"],
            prompt_df[f"{label_1}_ppl"],
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=8,
            color=colours[f"{Label_1} Steered PPL"],
            label=f"{Label_1} Steered Perplexity",
        )

        plt.plot(
            prompt_df["coeff"],
            prompt_df[f"{label_2}_ppl"],
            marker="X",
            linestyle="--",
            linewidth=2,
            markersize=8,
            color=colours[f"{Label_2} Steered PPL"],
            label=f"{Label_2} Steered Perplexity",
        )

        # Original as horizontal line
        orig_ppl = prompt_df["orig_ppl"].iloc[0]
        if pd.notna(orig_ppl):
            plt.axhline(
                y=orig_ppl,
                color=colours["Original PPL"],
                linestyle=":",
                linewidth=2,
                label=f"Original Perplexity (={orig_ppl:.3f})",
            )

        plt.title(f"Perplexity vs. Steering Coefficient - Latent {latent_id} \nPrompt: \"{prompt}\"",
                  fontsize=16, pad=20)
        plt.xlabel("Steering Coefficient", fontsize=14)
        plt.ylabel("Perplexity (lower is better)", fontsize=14)

        plt.ylim(-0.01, 1)

        plt.grid(True, linestyle="-", alpha=0.6)
        plt.legend(fontsize=11, frameon=True, borderpad=1)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Safe filename
        filename = re.sub(r"[^\w\s-]", "", prompt).replace(" ", "_")[:50]
        plt.savefig(os.path.join(output_dir, f"{filename}_ppl_plot.png"), dpi=300)
        plt.close()

    print(f"Per‑prompt perplexity plots saved to '{output_dir}'.")



def plot_box_by_coeff_ppl(df, latent_type, output_dir="plots_ppl", logy=False):
    os.makedirs(output_dir, exist_ok=True)

    if latent_type == 'sentiment':
        label_2, label_1 = 'pos', 'neg'
        Label_2, Label_1 = 'Pos', 'Neg'
    elif latent_type == 'bias':
        label_2, label_1 = 'bias', 'unbias'
        Label_2, Label_1 = 'Biased', 'Unbiased'

    score_type = "Perplexity"
    score_type_file = "ppl"

    latent_id = df['latent_id'].iloc[0] 

    # Melt to long-form for seaborn
    df_long = pd.melt(
        df,
        id_vars=['coeff'],
        value_vars=[f'{label_1}_ppl', f'{label_2}_ppl'],
        var_name='steering_type',
        value_name='ppl'
    )
    df_long['steering_type'] = df_long['steering_type'].map({
        f'{label_1}_ppl': f'{Label_1} Steered',
        f'{label_2}_ppl': f'{Label_2} Steered'
    })

    plt.figure(figsize=(12, 7))
    ax = sns.boxplot(
        data=df_long,
        x='coeff',
        y='ppl',
        hue='steering_type',
        palette={
            f'{Label_1} Steered': '#D62728',
            f'{Label_2} Steered': '#2CA02C'
        }
    )

    # Plot the mean original ppl per coeff as a dotted line
    prompt_scores = df[['prompt_text', 'orig_ppl']].drop_duplicates()
    coeffs_used = df[['prompt_text', 'coeff']].drop_duplicates()
    original_per_coeff = pd.merge(coeffs_used, prompt_scores, on='prompt_text')
    original_score_by_coeff = original_per_coeff.groupby('coeff')['orig_ppl'].mean().reset_index()

    unique_coeffs = sorted(df['coeff'].unique())
    coeff_to_xtick = {coeff: i for i, coeff in enumerate(unique_coeffs)}
    x_vals = [coeff_to_xtick[c] for c in original_score_by_coeff['coeff']]
    y_vals = original_score_by_coeff['orig_ppl']

    plt.plot(x_vals, y_vals, linestyle=':', linewidth=2, color='#1F77B4', label='Original Perplexity (Mean)')

    plt.title(f'{score_type} Distribution by Coefficient - Latent {latent_id} (All Sample Prompts)', fontsize=16)
    plt.xlabel('Steering Coefficient', fontsize=14)
    plt.ylabel(f'{score_type} (lower is better)', fontsize=14)
    plt.xticks(ticks=range(len(unique_coeffs)), labels=unique_coeffs)
    if logy:
        plt.yscale('log')
    plt.legend(title=None)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'boxplot_coeff_{score_type_file}.png'), dpi=300)
    plt.close()

def plot_mean_std_by_coeff_ppl(df, latent_type, output_dir="plots_ppl", logy=False):
    os.makedirs(output_dir, exist_ok=True)

    if latent_type == 'sentiment':
        label_2, label_1 = 'pos', 'neg'
        Label_2, Label_1 = 'Pos', 'Neg'
    elif latent_type == 'bias':
        label_2, label_1 = 'bias', 'unbias'
        Label_2, Label_1 = 'Biased', 'Unbiased'

    score_type = "Perplexity"
    score_type_file = "ppl"

    latent_id = df['latent_id'].iloc[0] 

    # Aggregates
    l1_agg = df.groupby('coeff')[f'{label_1}_ppl'].agg(['mean', 'std']).reset_index()
    l2_agg = df.groupby('coeff')[f'{label_2}_ppl'].agg(['mean', 'std']).reset_index()

    prompt_scores = df[['prompt_text', 'orig_ppl']].drop_duplicates()
    coeffs_used = df[['prompt_text', 'coeff']].drop_duplicates()
    original_per_coeff = pd.merge(coeffs_used, prompt_scores, on='prompt_text')
    original_agg = original_per_coeff.groupby('coeff')['orig_ppl'].agg(['mean', 'std']).reset_index()

    plt.figure(figsize=(12, 7))

    # Label_1
    plt.plot(l1_agg['coeff'], l1_agg['mean'], label=f'{Label_1} Steered', color='#D62728', marker='o', linestyle='-')
    plt.fill_between(l1_agg['coeff'], l1_agg['mean'] - l1_agg['std'], l1_agg['mean'] + l1_agg['std'],
                     alpha=0.2, color='#D62728')

    # Label_2
    plt.plot(l2_agg['coeff'], l2_agg['mean'], label=f'{Label_2} Steered', color='#2CA02C', marker='x', linestyle='--')
    plt.fill_between(l2_agg['coeff'], l2_agg['mean'] - l2_agg['std'], l2_agg['mean'] + l2_agg['std'],
                     alpha=0.2, color='#2CA02C')

    # Original
    plt.plot(original_agg['coeff'], original_agg['mean'], label='Original Perplexity (Mean)',
             linestyle=':', linewidth=2, color='#1F77B4')

    plt.title(f'{score_type} across Coefficients - Latent {latent_id} (All Sample Prompts)', fontsize=16)
    plt.xlabel('Steering Coefficient', fontsize=14)
    plt.ylabel(f'{score_type} (lower is better)', fontsize=14)
    if logy:
        plt.yscale('log')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'mean_std_coeff_{score_type_file}.png'), dpi=300)
    plt.close()


# if __name__ == "__main__":
#     script_dir = os.path.dirname(__file__)
#     filename = 'steer-pos_vs_neg-pos_vs_neg-9'
#     latent_type="sentiment"

#     results_file = os.path.join(script_dir, f'steering_outputs/{filename}.jsonl')
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     plot_output_directory = os.path.join(script_dir, f"steering_ppl_plots/plots_ppl_{filename}_{timestamp}")

#     if not os.path.exists(results_file):
#         print(f"Error: Results file not found at '{results_file}'.")
#     else:
#         df = load_jsonl_to_df(results_file)
#         plot_box_by_coeff_ppl(df, latent_type=latent_type, output_dir=plot_output_directory)
#         plot_mean_std_by_coeff_ppl(df, latent_type=latent_type, output_dir=plot_output_directory)

if __name__ == "__main__":
    script_dir  = os.path.dirname(__file__)
    input_dir   = os.path.join(script_dir, "steering_outputs")
    file_prefix    = "steer-pos_vs_neg-pos_vs_neg"   
    latent_type = "sentiment"

    # find every *.jsonl starting with the prefix
    pattern = os.path.join(input_dir, f"{file_prefix}*.jsonl")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No JSONL files starting with '{file_prefix}' found in {input_dir}")
        raise SystemExit(1)

    print(f"Found {len(files)} file(s):")
    for p in files:
        print("  -", os.path.basename(p))

    timestamp_root = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_plot_dir = os.path.join(script_dir, f"steering_ppl_plots/plots_ppl_{file_prefix}")
    os.makedirs(base_plot_dir, exist_ok=True)

    for file_path in files:
        base = os.path.splitext(os.path.basename(file_path))[0]
        plot_output_directory = os.path.join(base_plot_dir, base)
        os.makedirs(plot_output_directory, exist_ok=True)

        print(f"\nProcessing {base} …")
        df = load_jsonl_to_df(file_path)
        plot_box_by_coeff_ppl(df, latent_type=latent_type, output_dir=plot_output_directory)
        plot_mean_std_by_coeff_ppl(df, latent_type=latent_type, output_dir=plot_output_directory)

    print(f"\nAll plots written under: {base_plot_dir}")