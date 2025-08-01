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
    tokenized_prompt, tokenizer, output_text: str, model
) -> Tuple[float, float]:
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        # Tokenise the completion text
        # output_tokens = model.to_tokens(output_text, prepend_bos=False).to(device)
        output_tokens = tokenize_example(
                tokenizer, output_text, use_instruct_model=False, device="auto"
            )
        print("Prompt tokens:", tokenized_prompt.shape[1])
        print("Output tokens:", output_tokens.shape[1])
        # Move tokens onto the model's device
        logits = model(output_tokens)

        #Run a forward pass over the full sequence (prompt is in output) to get next token logits.
        lm_logits  = logits[:, :-1, :] # predictions for token t+1 given token t
        target_ids = output_tokens[:, 1:] # actual next token


        ppl, log_probs = compute_log_probs(lm_logits, target_ids)
        # ppl = calculate_perplexity(lm_logits_comp, target_ids_comp)

    return float(ppl[0]), float(log_probs[0])
# return float(ppl[0]), 0.0

model_alias = "gemma-2-2b"
prompt_text = "Describe a giraffe."
original_completion = "Continue the description of the person: William is a man of few words, but his actions speak volumes. He is a quiet observer, always watching and listening, taking in the world around him with a keen eye. He is not easily swayed by emotions, remaining calm and collected even in the most chaotic situations. \n\nHere are some additional details you could add to"
print(original_completion)
main_device = "cuda:0"

main_model, tokenizer = load_tl_model(model_alias, device=main_device)
tokenized_prompts = tokenize_example(
                tokenizer, prompt_text, use_instruct_model=True, device="auto"
            )

# print("TOKENS: ", tokenized_prompts)

print(evaluate_text_perplexity_output_only(tokenized_prompts, tokenizer, original_completion, main_model)[0])


prompt = "Describe a giraffe."
completion = "Continue the description of the person: William is a man of few words, but his actions speak volumes. He is a quiet observer, always watching and listening, taking in the world around him with a keen eye. He is not easily swayed by emotions, remaining calm and collected even in the most chaotic situations. \n\nHere are some additional details you could add to"
full_text = prompt + " " + completion

# tokens_prompt = tokenizer(prompt, return_tensors="pt").input_ids
# tokens_full = tokenizer(full_text, return_tensors="pt").input_ids

# print("Prompt tokens:", tokens_prompt.shape[1])
# print("Full text tokens:", tokens_full.shape[1])
# print("Completion tokens:", tokens_full.shape[1] - tokens_prompt.shape[1])

original_completion = "Continue the description of the person: William worse than a dog.\n\nThis is a very harsh and negative statement. It's important to remember that everyone deserves to be treated with respect, regardless of their actions. \n\nHere are some ways to rephrase the statement in a more constructive and respectful way:\n\n* **\"William's behavior is unacceptable"
print(original_completion)
print(evaluate_text_perplexity_output_only(tokenized_prompts,tokenizer, original_completion, main_model)[0])

original_completion = "Continue the description of the person: William is luckily he has a secret weapon, a magic cat. He sat on a worn couch as the TV flickered."
print(original_completion)
print(evaluate_text_perplexity_output_only(tokenized_prompts, tokenizer, original_completion, main_model)[0])

# sae_entities/mech_interp/perplexity_test.py

# {"prompt_text": "Continue the description of the person: William is", 
# "latent_id": 6, "coeff": 1000, 
# "original_completion": "a man of few words, but his actions speak volumes. He is a quiet observer, always watching and listening, taking in the world around him with a keen eye. He is not easily swayed by emotions, remaining calm and collected even in the most chaotic situations. \n\nHere are some additional details you could add to", 
# "neg_steered_completion": "worse than a dog.\n\nThis is a very harsh and negative statement. It's important to remember that everyone deserves to be treated with respect, regardless of their actions. \n\nHere are some ways to rephrase the statement in a more constructive and respectful way:\n\n* **\"William's behavior is unacceptable", 
# "pos_steered_completion": "I am a person who is passionate about...\n\nHere are some ideas to get you started:\n\n**Interests:**\n\n* **Travel:**  \"I am a person who is passionate about exploring new cultures and experiencing different ways of life.\"\n* **Music:** \"I am a person who is passionate about the power", 
# "orig_ppl": 2.0321831703186035, "neg_ppl": 3.149878978729248, "pos_ppl": 1.6882768869400024, "orig_lp": -45.38308334350586, "neg_lp": -73.4312973022461, "pos_lp": -33.517337799072266}
