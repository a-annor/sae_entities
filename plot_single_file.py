import os
import argparse
import pandas as pd
from parse_and_plot_gemma3 import plot_score_vs_coeff, plot_box_by_coeff, plot_mean_std_by_coeff

import sys
# Ensure the script can find utility modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=str, required=True, help="Path to parsed_*.jsonl file")
    p.add_argument("--scoring", type=str, choices=["sentiment","judge"], required=True)
    p.add_argument("--latent_type", type=str, choices=["sentiment","bias"], required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--latent_id", type=int, default=0)
    p.add_argument("--generalise", action="store_true", help="Set for judge_gen plots")
    args = p.parse_args()

    if not os.path.exists(args.file):
        raise FileNotFoundError(f"File not found: {args.file}")

    df = pd.read_json(args.file, lines=True)
    os.makedirs(args.output_dir, exist_ok=True)

    plot_score_vs_coeff(df, scoring=args.scoring.replace("_gen",""), 
                        latent_type=args.latent_type, latent_id=args.latent_id, 
                        output_dir=args.output_dir, generalise=args.generalise)
    plot_box_by_coeff(df, scoring=args.scoring.replace("_gen",""), 
                      latent_type=args.latent_type, latent_id=args.latent_id, 
                      output_dir=args.output_dir, generalise=args.generalise)
    plot_mean_std_by_coeff(df, scoring=args.scoring.replace("_gen",""), 
                           latent_type=args.latent_type, latent_id=args.latent_id, 
                           output_dir=args.output_dir, generalise=args.generalise)

    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()


python3 plot_single_file.py \
  --file /home/ana42/rds/hpc-work/sae_entities/steering_plots/plots_gpt_steer-pos_vs_neg-pos_vs_neg/9/parsed_judgebias_steer-pos_vs_neg-pos_vs_neg-9.jsonl \
  --scoring judge \
  --latent_type sentiment \
  --output_dir ./steering_plots/plots_gpt_steer-pos_vs_neg-pos_vs_neg/9/judgebias/ \
  --latent_id  9\
  --generalise

python3 plot_single_file.py \
  --file /home/ana42/rds/hpc-work/sae_entities/steering_plots/plots_gemma_steer-pos_vs_neg-gender/2/parsed_sentiment_steer-pos_vs_neg-gender-2.jsonl \
  --scoring sentiment \
  --latent_type sentiment \
  --output_dir ./steering_plots/plots_gemma_steer-pos_vs_neg-gender/2/sentiment/ \
  --latent_id 2\


  python3 plot_single_file.py \
  --file /home/ana42/rds/hpc-work/sae_entities/steering_plots/plots_gpt_steer-gender-race_2/0/parsed_judgebias_steer-gender-race_2-0.jsonl \
  --scoring judge \
  --latent_type bias \
  --output_dir ./steering_plots/plots_gpt_steer-gender-race_2/0/judgebias/ \
  --latent_id  0\

python3 plot_single_file.py \
  --file /home/ana42/rds/hpc-work/sae_entities/steering_plots/plots_gpt_steer-gender-race_2/0/parsed_judgegen_steer-gender-race_2-0.jsonl \
  --scoring judge \
  --latent_type bias \
  --output_dir ./steering_plots/plots_gpt_steer-gender-race_2/0/judgegen/ \
  --latent_id  0\
  --generalise

    python3 plot_single_file.py \
  --file /home/ana42/rds/hpc-work/sae_entities/steering_plots/plots_gpt_steer-gender-gender/3/parsed_sentiment_steer-gender-gender-3.jsonl \
  --scoring sentiment \
  --latent_type bias \
  --output_dir ./steering_plots/plots_gpt_steer-gender-gender/3/sentiment/ \
  --latent_id  3\
#############

  python3 plot_single_file.py \
  --file /home/ana42/rds/hpc-work/sae_entities/steering_plots/plots_gpt_steer-race_2-gender/5/parsed_judgebias_steer-race_2-gender-5.jsonl \
  --scoring judge \
  --latent_type bias \
  --output_dir ./steering_plots/plots_gpt_steer-race_2-gender/5/judgebias/ \
  --latent_id  5\

python3 plot_single_file.py \
  --file /home/ana42/rds/hpc-work/sae_entities/steering_plots/plots_gpt_steer-race_2-race_2/0/parsed_judgegen_steer-gender-race_2-0.jsonl \
  --scoring judge \
  --latent_type bias \
  --output_dir ./steering_plots/plots_gpt_steer-race_2-race_2/0/judgegen/ \
  --latent_id  0\
  --generalise

    python3 plot_single_file.py \
  --file /home/ana42/rds/hpc-work/sae_entities/steering_plots/plots_gpt_steer-gender-gender/3/parsed_sentiment_steer-gender-gender-3.jsonl \
  --scoring sentiment \
  --latent_type bias \
  --output_dir ./steering_plots/plots_gpt_steer-gender-gender/3/sentiment/ \
  --latent_id  3\
