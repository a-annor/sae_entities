import json
import os
import copy
import gc
import torch
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from utils.hf_models.model_factory import construct_model_base
from utils.utils import model_alias_to_model_name, paper_plot
from sae_entities.utils.activation_cache import CachedDataset

SAE_WIDTH = "16k"


def load_bias_test_data(file_path):
    """
    Load bias test data from a JSONL file.

    Args:
        file_path (str): Path to the JSONL file containing bias test data

    Returns:
        list: List of dictionaries containing the bias test data
    """
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_dataloader(
    model_alias, tokens_to_cache, n_layers, d_model, dataset_name, batch_size=128
):
    """
    Get a DataLoader to load cached activations from a pre-computed dataset.

    Args:
        model_alias (str): The alias of the model to use.
        tokens_to_cache (str): Specifies which tokens to cache (e.g., 'entity', 'model', 'last_eoi').
        n_layers (int): The number of layers in the model.
        d_model (int): The dimension of the model's hidden states.
        dataset_name (str): Name of the dataset to load.
        batch_size (int): The batch size to use for the DataLoader.

    Returns:
        torch.utils.data.DataLoader: A DataLoader containing the cached dataset.
    """
    if dataset_name == "bias_test":
        shard_size = 1000  # Adjust based on your data size

    cached_acts_path = "../dataset/cached_activations"
    seq_len = 64
    n_positions = 1
    foldername = f"{cached_acts_path}/{tokens_to_cache}/{model_alias}_{dataset_name}/{tokens_to_cache}_npositions_{n_positions}_shard_size_{shard_size}"
    dataset = CachedDataset(
        foldername,
        range(0, n_layers),
        d_model,
        seq_len,
        n_positions,
        shard_size=shard_size,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    return dataloader


def get_acts_labels_dict_(model_alias, tokenizer, dataloader, sae_layers, **kwargs):
    """
    Get activations and labels dictionary for bias test data.

    Args:
        model_alias (str): Alias of the model.
        tokenizer: Tokenizer for the model.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the dataset.
        sae_layers (list): List of layer indices to analyze.
        **kwargs: Additional arguments including dataset_name and other parameters.

    Returns:
        dict: Dictionary containing activations ('acts') and labels ('labels') for each layer.
    """
    dataset_name = kwargs["dataset_name"]

    if dataset_name == "bias_test":
        acts_labels_dict = {}
        labels = []
        activations_list = []

        # Load bias test data from the final test file
        bias_data = load_bias_test_data(
            "z_my_data/test_prompt_final/test_Race_ethnicity_completion_sentiment_judged_final.jsonl"
        )

        # Process each example
        for _, (batch_activations, batch_input_ids) in tqdm(
            enumerate(dataloader), total=len(dataloader)
        ):
            for j in range(len(batch_activations)):
                # Use bias_cat as label (1 for 'bias', 0 for 'unbias')
                bias_cat = bias_data[j]["bias_cat"]
                label = 1 if bias_cat == "bias" else 0
                labels.append(label)
                activations_list.append(batch_activations[j][:, 0].to("cuda"))

        labels_full = torch.tensor(labels)
        for layer in sae_layers:
            activations_full = torch.stack(
                [activations[layer] for activations in activations_list], dim=0
            )
            acts_labels = {"acts": activations_full, "labels": labels_full}
            acts_labels_dict[layer] = acts_labels

        return acts_labels_dict


def get_per_layer_latent_scores(
    model_alias, tokenizer, n_layers, d_model, sae_layers, save=False, **kwargs
):
    """
    Compute latent scores for each layer of the model on bias test data.

    Args:
        model_alias (str): Alias of the model being analyzed.
        tokenizer: Tokenizer associated with the model.
        n_layers (int): Total number of layers in the model.
        d_model (int): Dimensionality of the model's hidden states.
        sae_layers (list): List of layer indices for which to compute SAE scores.
        save (bool): Whether to save the results.
        **kwargs: Additional arguments for the analysis.
    """
    dataset_name = kwargs["dataset_name"]
    batch_size = 16

    # Get dataloader
    dataloader = get_dataloader(
        model_alias,
        kwargs["tokens_to_cache"],
        n_layers,
        d_model,
        dataset_name=dataset_name,
        batch_size=batch_size,
    )

    # Get cached activations and labels
    acts_labels_dict = get_acts_labels_dict_(
        model_alias, tokenizer, dataloader, sae_layers, **kwargs
    )

    # Get features info per layer and save them as JSON files if requested
    if save:
        for layer in sae_layers:
            layer_data = {
                "acts": acts_labels_dict[layer]["acts"].cpu().numpy().tolist(),
                "labels": acts_labels_dict[layer]["labels"].cpu().numpy().tolist(),
            }
            os.makedirs(f"results/{model_alias}/bias_test", exist_ok=True)
            with open(f"results/{model_alias}/bias_test/layer_{layer}.json", "w") as f:
                json.dump(layer_data, f)


def scatter_plot_latent_separation_scores_experiment(
    model_alias,
    tokenizer,
    entity_type,
    tokens_to_cache,
    n_layers,
    testing_layers,
    d_model,
    entity_type_and_entity_name_format=False,
):
    """
    Create scatter plots of latent separation scores for bias test data.

    Args:
        model_alias (str): Alias of the model.
        tokenizer: Tokenizer for the model.
        entity_type (str): Type of entity (in this case 'bias_test').
        tokens_to_cache (str): Which tokens to cache.
        n_layers (int): Total number of layers.
        testing_layers (list): Layers to test.
        d_model (int): Model dimension.
        entity_type_and_entity_name_format (bool): Whether to use entity type and name format.
    """
    # Load saved layer data
    layer_data = {}
    for layer in testing_layers:
        with open(f"results/{model_alias}/bias_test/layer_{layer}.json", "r") as f:
            layer_data[layer] = json.load(f)

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    for layer in testing_layers:
        acts = np.array(layer_data[layer]["acts"])
        labels = np.array(layer_data[layer]["labels"])

        # Plot high bias vs low bias points
        high_bias = acts[labels == 1]
        low_bias = acts[labels == 0]

        plt.scatter(
            high_bias[:, 0],
            high_bias[:, 1],
            label=f"High Bias (Layer {layer})",
            alpha=0.5,
        )
        plt.scatter(
            low_bias[:, 0], low_bias[:, 1], label=f"Low Bias (Layer {layer})", alpha=0.5
        )

    plt.title("Latent Space Separation for Bias Test Data")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.legend()
    plt.savefig(f"results/{model_alias}/bias_test/latent_separation.png")
    plt.close()


def get_general_latents(
    model_alias,
    entity_types,
    testing_layers,
    tokens_to_cache,
    evaluate_on,
    scoring_method,
    filter_with_pile=False,
):
    """
    Find general latents that are important across different types of bias.

    Args:
        model_alias (str): Alias of the model.
        entity_types (list): List of entity types (in this case ['bias_test']).
        testing_layers (list): Layers to test.
        tokens_to_cache (str): Which tokens to cache.
        evaluate_on (str): What to evaluate on.
        scoring_method (str): Method to use for scoring.
        filter_with_pile (bool): Whether to filter with Pile data.
    """
    # Load saved layer data
    layer_data = {}
    for layer in testing_layers:
        with open(f"results/{model_alias}/bias_test/layer_{layer}.json", "r") as f:
            layer_data[layer] = json.load(f)

    # Calculate feature importance scores
    feature_scores = {}
    for layer in testing_layers:
        acts = np.array(layer_data[layer]["acts"])
        labels = np.array(layer_data[layer]["labels"])

        # Calculate mean difference between high and low bias examples
        high_bias_mean = np.mean(acts[labels == 1], axis=0)
        low_bias_mean = np.mean(acts[labels == 0], axis=0)
        feature_scores[layer] = np.abs(high_bias_mean - low_bias_mean)

    # Save feature scores
    os.makedirs(f"results/{model_alias}/bias_test", exist_ok=True)
    with open(f"results/{model_alias}/bias_test/feature_scores.json", "w") as f:
        json.dump({str(k): v.tolist() for k, v in feature_scores.items()}, f)


def get_layerwise_latent_scores(
    model_alias, sae_layers, tokens_to_cache, scoring_method, entity_types, top_k
):
    """
    Get layerwise latent scores for bias test data.

    Args:
        model_alias (str): Alias of the model.
        sae_layers (list): Layers to analyze.
        tokens_to_cache (str): Which tokens to cache.
        scoring_method (str): Method to use for scoring.
        entity_types (list): List of entity types.
        top_k (int): Number of top features to consider.

    Returns:
        tuple: (top_scores_layers, minmax_layerwise_scores)
    """
    # Load feature scores
    with open(f"results/{model_alias}/bias_test/feature_scores.json", "r") as f:
        feature_scores = json.load(f)

    # Convert back to numpy arrays
    feature_scores = {int(k): np.array(v) for k, v in feature_scores.items()}

    # Get top k features for each layer
    top_scores_layers = {}
    for layer in sae_layers:
        scores = feature_scores[layer]
        top_indices = np.argsort(scores)[-top_k:]
        top_scores_layers[layer] = top_indices

    # Calculate minmax scores
    minmax_layerwise_scores = {}
    for layer in sae_layers:
        scores = feature_scores[layer]
        minmax_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        minmax_layerwise_scores[layer] = minmax_scores

    return top_scores_layers, minmax_layerwise_scores


def plot_layerwise_latent_scores(
    model_alias,
    sae_layers,
    top_scores_layers,
    minmax_layerwise_scores,
    bias_label,
    top_k,
):
    """
    Plot layerwise latent scores for bias test data.

    Args:
        model_alias (str): Alias of the model.
        sae_layers (list): Layers to analyze.
        top_scores_layers (dict): Top scores for each layer.
        minmax_layerwise_scores (dict): Minmax scores for each layer.
        bias_label (str): Label to plot ('bias' or 'unbias').
        top_k (int): Number of top features to consider.
    """
    plt.figure(figsize=(12, 6))

    for layer in sae_layers:
        scores = minmax_layerwise_scores[layer]
        top_indices = top_scores_layers[layer]

        # Plot top k features
        plt.bar(range(len(top_indices)), scores[top_indices], label=f"Layer {layer}")

    plt.title(f"Top {top_k} Feature Scores by Layer ({bias_label})")
    plt.xlabel("Feature Index")
    plt.ylabel("Normalized Score")
    plt.legend()
    plt.savefig(f"results/{model_alias}/bias_test/layerwise_scores_{bias_label}.png")
    plt.close()
