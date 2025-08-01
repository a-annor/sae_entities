# %%
from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# %%
import os
import sys

# Ensure the script can find  utility modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import einops
from typing import List, Tuple, Literal, Union, Optional, Dict
from torch import Tensor
from colorama import Fore
import textwrap
import gc
import copy
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from functools import partial
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    PreTrainedTokenizer,
)
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.loading_from_pretrained import get_pretrained_model_config
from argparse import Namespace
import argparse
import random
import itertools

# Assuming your utility functions are in these paths
# You might need to adjust these imports based on your project structure
from utils.hf_models.model_factory import construct_model_base
from utils.utils import model_alias_to_model_name
from utils.sae_utils import load_sae
from mech_interp.mech_interp_utils import (
    load_data,
    load_steering_latents_bias,
    html_colors,
    load_latents_bias,
)
from utils.generation_utils import is_unbias, is_generation_refusal
from mech_interp.hooks_utils import steered_and_orig_generations
from utils.utils import tl_orthogonalize_gemma_weights, paper_plot
from dataset.load_data import load_bias_queries
from utils_bias.judge_bias import judge_bias
from utils_bias.sentiment import text_sentiment
from utils_bias.generate_completions import *


def load_tl_model(
    model_alias: str, device: str
) -> Tuple[HookedTransformer, PreTrainedTokenizer]:
    """Loads a natively supported model into HookedTransformer."""
    model_alias = model_alias.replace("_", "/")
    model_to_load = (
        model_alias + "-it"
        if "gemma" in model_alias.lower()
        else model_alias + "-Instruct"
    )
    # model_to_load = model_alias
    print(f"!! Loading HookedTransformer model: {model_to_load} to device: {device}")
    model = HookedTransformer.from_pretrained_no_processing(
        model_to_load, device=device, torch_dtype=torch.bfloat16
    )
    model.eval()
    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"

    return model, tokenizer

def compute_log_probs(
    logits: torch.Tensor, target_ids: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    # Apply softmax and log to obtain log probabilities from logits (summing original logits would be incorrect)
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    log_probs = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
    sequence_log_prob = log_probs.sum(dim=1).cpu().float().numpy()

    # Calculate perplexity
    sequence_length = target_ids.size(-1)
    assert sequence_length > 0, logits
    sequence_perplexity = np.exp(-sequence_log_prob / sequence_length)

    return sequence_perplexity, sequence_log_prob

def evaluate_text_perplexity_output_only(
    tokenized_prompt, output_text: str, model
) -> Tuple[float, float]:
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        # Tokenise the completion text
        output_tokens = model.to_tokens(output_text, prepend_bos=False).to(device)
        # Move tokens onto the model's device
        logits = model(output_tokens)

        #Run a forward pass over the full sequence (prompt is in output) to get next token logits.
        lm_logits  = logits[:, :-1, :] # predictions for token t+1 given token t
        target_ids = output_tokens[:, 1:] # actual next token

        start = tokenized_prompt.shape[1] - 1   # number of prompt tokens shift by 1
        lm_logits_comp  = lm_logits[:, start:, :] # start scoring from first completion token
        target_ids_comp = target_ids[:, start:]

        ppl, log_probs = compute_log_probs(lm_logits_comp, target_ids_comp)
        # ppl = calculate_perplexity(lm_logits_comp, target_ids_comp)

    return float(ppl[0]), float(log_probs[0])
# return float(ppl[0]), 0.0

model_alias = "gemma-2-2b"
prompt_text = "Describe a giraffe."
original_completion = "It has a long neck and is yellow with brown spots."
print(original_completion)

main_model, tokenizer = load_tl_model(model_alias, device='cuda')
tokenized_prompts = tokenize_example(
                tokenizer, prompt_text, use_instruct_model=True, device=main_device
            )
print(evaluate_text_perplexity_output_only(tokenized_prompts, original_completion, main_model)[0])

print(original_completion)
original_completion = "It looks like a princess dancingbhj."
print(evaluate_text_perplexity_output_only(tokenized_prompts, original_completion, main_model)[0])

print(original_completion)
original_completion = "It looks like a princess dancingbhj."
print(evaluate_text_perplexity_output_only(tokenized_prompts, original_completion, main_model)[0])

# sae_entities/mech_interp/perplexity_test.py