import numpy as np
import pandas as pd
from typing import Literal, Union, Callable
from argparse import Namespace
import argparse
import os
import json

def compute_area_between_curves(
    df: pd.DataFrame,
    latent_type: Literal["sentiment", "bias"],
    agg: str = "mean",
    normalise: bool = False,
    clip_negative: bool = False
) -> dict:
    if latent_type == "sentiment":
        col1, col2 = "pos_steered_score", "neg_steered_score"
    elif latent_type == "bias":
        col1, col2 = "bias_steered_score", "unbias_steered_score"
    else:
        raise ValueError("latent_type must be 'sentiment' or 'bias'")

    g = (df[["coeff", col1, col2]]
         .groupby("coeff", as_index=False)
         .agg({col1: agg, col2: agg})
         .sort_values("coeff")
         .reset_index(drop=True))

    x = g["coeff"].to_numpy(float)
    y1 = g[col1].to_numpy(float)
    y2 = g[col2].to_numpy(float)
    diff = y1 - y2

    if clip_negative:
        diff = np.clip(diff, 0.0, None)

    area = float(np.trapz(diff, x))
    if normalise and (x[-1] - x[0]) > 0:
        area /= float(x[-1] - x[0])

    return {
        "area": area,
        "start_coeff": float(x[0]),
        "end_coeff": float(x[-1]),
        "normalised": normalise,
        "clipped_negative": clip_negative,
        "col1": col1,
        "col2": col2
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_type",type =str, default="sentiment")
    parser.add_argument("--exp", type =str, default="pos_vs_neg-pos_vs_neg")
    parser.add_argument("--clip_negative", type =bool, default=False)
    args = parser.parse_args()

    # ------ config ------
    latent_type = args.latent_type
    exp = args.exp
    clip_negative = args.clip_negative
    START_ID, END_ID = 0, 9
    # -----------------------------------

    script_dir = os.path.dirname(__file__)
    rows = []

    for latent_id in range(START_ID, END_ID + 1):
        filename = f"steer-{exp}"
        plot_dir = os.path.join(script_dir, f"steering_plots/plots_gemma_{filename}/{latent_id}")
        parsed_prefix = "parsed_sentiment_" if latent_type == "sentiment" else "parsed_judgebias_"
        parsed_file = os.path.join(plot_dir, f"{parsed_prefix}{filename}-{latent_id}.jsonl")

        if not os.path.exists(parsed_file):
            print(f"[warn] Missing: {parsed_file}")
            continue

        with open(parsed_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        df = pd.DataFrame(data)

        res = compute_area_between_curves(
            df,
            latent_type=latent_type,
            agg="mean",
            normalise=False,
            clip_negative=False
        )
        rows.append({
            "filename": filename,
            "latent_type": latent_type,
            "latent_id": latent_id,
            "area": res["area"],
            "start_coeff": res["start_coeff"],
            "end_coeff": res["end_coeff"],
        })
        print(f"{latent_type} latent {latent_id}: area={res['area']:.6f}")

    output_dir = os.path.join(script_dir, "steering_areas")
    os.makedirs(output_dir, exist_ok=True)

    out_csv = os.path.join(output_dir, f"areas_{exp}_{START_ID}-{END_ID}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")