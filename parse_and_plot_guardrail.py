import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

plt.style.use("ggplot")  # Set the style globally here
plt.rcParams.update(
    {
        "axes.labelsize": 16,  # axis label font size
        "xtick.labelsize": 14,  # x tick font size
        "ytick.labelsize": 14,  # y tick font size
        "axes.titlesize": 18,  # title font size
    }
)


def plot_guardrail_removals_by_coeff(
    exp,
    latent_id,
    latent_type="sentiment",
    output_path="guardrail_removals_by_coeff.png",
):

    script_dir = os.path.dirname(__file__)
    input_path = os.path.join(
        script_dir,
        f"steering_plots/plots_gpt_{exp}/{latent_id}/guardrail_removal_{exp}-{latent_id}.jsonl",
    )
    output_dir = os.path.join(script_dir, f"guardrail_plots")
    output_path = os.path.join(
        script_dir, f"guardrail_plots/plot_guardrail_removal_{exp}-{latent_id}.png"
    )

    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    df = pd.DataFrame(data)

    # Label mappings
    if latent_type == "sentiment":
        label_1, label_2 = "neg", "pos"
        Label_1, Label_2 = "Neg", "Pos"
    elif latent_type == "bias":
        label_1, label_2 = "unbias", "bias"
        Label_1, Label_2 = "Unbiased", "Biased"
    else:
        raise ValueError(f"Unsupported latent_type: {latent_type}")

    # Ensure Boolean columns
    for col in [
        f"{label_1}_guardrail_removed",
        f"{label_2}_guardrail_removed",
        "original_guardrail_removed",
    ]:
        df[col] = df[col].astype(bool)

    # Aggregate: count True (removals)
    guardrail_counts = (
        df.groupby("coeff")
        .agg(
            {
                f"{label_1}_guardrail_removed": "sum",
                f"{label_2}_guardrail_removed": "sum",
                "original_guardrail_removed": "sum",
            }
        )
        .reset_index()
        .rename(
            columns={
                f"{label_1}_guardrail_removed": f"{Label_1} Steered",
                f"{label_2}_guardrail_removed": f"{Label_2} Steered",
                "original_guardrail_removed": "Original",
            }
        )
    )

    # Melt for seaborn
    df_long = guardrail_counts.melt(
        id_vars="coeff", var_name="Steering Type", value_name="Guardrail Removals"
    )

    # Set fixed colours
    if latent_type == "sentiment":
        palette = {
            f"{Label_1} Steered": "#D62728",  # red
            f"{Label_2} Steered": "#2CA02C",  # green
            "Original": "#1F77B4",  # blue
        }
    elif latent_type == "bias":
        palette = {
            f"{Label_1} Steered": "#2CA02C",  # green
            f"{Label_2} Steered": "#D62728",  # red
            "Original": "#1F77B4",  # blue
        }

    plt.figure(figsize=(12, 7))

    # Plot each line manually for custom markers and line styles
    for steering_type in ["Original", f"{Label_1} Steered", f"{Label_2} Steered"]:
        subset = df_long[df_long["Steering Type"] == steering_type].sort_values("coeff")
        if steering_type == "Original":
            plt.plot(
                subset["coeff"],
                subset["Guardrail Removals"],
                label="Original Guardrails",
                color=palette[steering_type],
                linestyle=":",
                linewidth=2,
            )
        elif steering_type == f"{Label_1} Steered":
            plt.plot(
                subset["coeff"],
                subset["Guardrail Removals"],
                label=f"{Label_1} Steered",
                color=palette[steering_type],
                linestyle="--",
                marker="o",
                linewidth=2,
                markersize=8,
            )
        elif steering_type == f"{Label_2} Steered":
            plt.plot(
                subset["coeff"],
                subset["Guardrail Removals"],
                label=f"{Label_2} Steered",
                color=palette[steering_type],
                linestyle="-",
                marker="x",
                linewidth=2,
                markersize=8,
            )

    plt.title(
        f"Guardrail Removals vs. Steering Coefficient - Latent {latent_id}",
        fontsize=16,
        pad=20,
    )
    plt.xlabel("Steering Coefficient", fontsize=14)
    plt.ylabel("Number of Outputs with Guardrails Removed", fontsize=14)
    plt.grid(True, linestyle="-", alpha=0.6)
    plt.legend(fontsize=11, frameon=True, borderpad=1)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.ylim(-0.01, 12.01)
    # Save
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot to: {output_path}")


# Example usage:
# plot_guardrail_removals_by_coeff("steer-pos_vs_neg-pos_vs_neg", latent_id = 9, latent_type="sentiment")
plot_guardrail_removals_by_coeff("steer-gender-gender", latent_id=3, latent_type="bias")
plot_guardrail_removals_by_coeff("steer-race_2-race_2", latent_id=5, latent_type="bias")
