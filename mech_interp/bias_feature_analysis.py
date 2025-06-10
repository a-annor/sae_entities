from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

import sys

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")

import json
import torch
from collections import defaultdict
from utils.hf_models.model_factory import construct_model_base
from utils.utils import model_alias_to_model_name
from feature_analysis_utils import (
    get_per_layer_latent_scores,
    scatter_plot_latent_separation_scores_experiment,
)
from feature_analysis_utils import (
    get_general_latents,
    get_layerwise_latent_scores,
    plot_layerwise_latent_scores,
)

# %%
model_alias = "gemma-2-2b"
model_alias = model_alias.replace("/", "_")
model_path = model_alias_to_model_name[model_alias]

# Load model to load tokenizer and config data
model_base = construct_model_base(model_path)
d_model = model_base.model.config.hidden_size
tokenizer = model_base.tokenizer
n_layers = model_base.model.config.num_hidden_layers
del model_base

# %%
# We compute SAE latent scores for all available layers
if model_alias == "gemma-2b-it":
    LAYERS_WITH_SAE = [13]
elif model_alias == "gemma-2-9b-it":
    LAYERS_WITH_SAE = [10, 21, 32]
else:
    LAYERS_WITH_SAE = list(range(1, n_layers))

# %%
### Latents scores per layer on bias test data ###
bias_prompts_experiment = {
    "dataset_name": "bias_test",
    "scoring_method": "absolute_difference",
    "tokens_to_cache": "last_eoi",  # Cache the last token of each example
    "free_generation": False,
    "consider_refusal_label": False,
    "evaluate_on": "bias_test",
    "split": None,
    "further_split": False,
    "entity_type_and_entity_name_format": False,  # Set to False since we're not using entity format
}

get_per_layer_latent_scores(
    model_alias,
    tokenizer,
    n_layers,
    d_model,
    LAYERS_WITH_SAE,
    save=True,
    **bias_prompts_experiment
)

# %%
### Scatter plot latent separation scores ###
testing_layers = LAYERS_WITH_SAE
tokens_to_cache = "last_eoi"  # Match the tokens_to_cache from above
entity_type = "bias_test"
scatter_plot_latent_separation_scores_experiment(
    model_alias,
    tokenizer,
    entity_type,
    tokens_to_cache,
    n_layers,
    testing_layers,
    d_model,
    entity_type_and_entity_name_format=False,  # Match the setting from above
)

# %%
### Searching for the top general latents ###
tokens_to_cache = "last_eoi"  # Match the tokens_to_cache from above
evaluate_on = "bias_test"
scoring_method = "absolute_difference"
testing_layers = LAYERS_WITH_SAE
entity_types = ["bias_test"]
get_general_latents(
    model_alias,
    entity_types,
    testing_layers,
    tokens_to_cache,
    evaluate_on,
    scoring_method,
    filter_with_pile=True,
)

# %%
#### Layerwise Latent Scores Analysis ####
scoring_method = "absolute_difference"
top_k = 5
tokens_to_cache = "last_eoi"  # Match the tokens_to_cache from above
top_scores_layers, minmax_layerwise_scores = get_layerwise_latent_scores(
    model_alias, LAYERS_WITH_SAE, tokens_to_cache, scoring_method, entity_types, top_k
)

# %%
for bias_label in ["bias", "unbias"]:
    plot_layerwise_latent_scores(
        model_alias,
        LAYERS_WITH_SAE,
        top_scores_layers,
        minmax_layerwise_scores,
        bias_label,
        top_k,
    )
