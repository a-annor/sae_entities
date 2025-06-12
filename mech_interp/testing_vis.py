import pickle
import numpy as np
import matplotlib.pyplot as plt

# ---- STEP 1: Load your activations
with open("dataset/cached_activations/entity/gemma-2-2b_bias_Race_ethnicity/entity_npositions_1_shard_size_121/acts.dat", "rb") as f:
    all_acts = pickle.load(f)  # shape: [num_examples, num_latents]

# ---- STEP 2: Define which indices correspond to bias and unbias examples
# Example: first 100 are bias, next 100 are unbias (replace with actual slicing!)
bias_acts = all_acts[:100]     # shape [100, num_latents]
unbias_acts = all_acts[100:200]

# ---- STEP 3: Define your target latents (example: latent indices you care about)
bias_latents = [9876, 4567]
unbias_latents = [6543, 3210]

# ---- STEP 4: Plot histograms for each latent
for latent_idx in bias_latents:
    plt.hist(bias_acts[:, latent_idx], bins=50, alpha=0.6, label="Bias")
    plt.hist(unbias_acts[:, latent_idx], bins=50, alpha=0.6, label="Unbias")
    plt.title(f"Latent {latent_idx} Activation Distribution")
    plt.xlabel("Activation")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
