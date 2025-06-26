# %%
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')

# %%
import os
import sys
# Ensure the script can find your utility modules
# This relative path setup assumes the script is run from its directory
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
import numpy as np
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, PreTrainedTokenizer
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.loading_from_pretrained import get_pretrained_model_config 
from argparse import Namespace
import random

# Assuming your utility functions are in these paths
# You might need to adjust these imports based on your project structure
from utils.hf_models.model_factory import construct_model_base
from utils.utils import model_alias_to_model_name
from utils.sae_utils import load_sae
from mech_interp.mech_interp_utils import load_data, load_steering_latents_bias, html_colors, load_latents_bias
from utils.generation_utils import is_unbias, is_generation_refusal
from mech_interp.hooks_utils import steered_and_orig_generations
from utils.utils import tl_orthogonalize_gemma_weights, paper_plot
from dataset.load_data import load_bias_queries
from utils_bias.judge_bias import judge_bias
from utils_bias.sentiment import text_sentiment
from utils_bias.generate_completions import load_model


random_seed = 42
random.seed(random_seed)
# %%

def load_tl_model(model_alias: str, device: str) -> Tuple[HookedTransformer, PreTrainedTokenizer]:
    """Loads a natively supported model into HookedTransformer."""
    # model_to_load = f"google/{model_alias}"
    model_alias = model_alias.replace('_','/')
    model_to_load = model_alias+'-it' if 'gemma' in model_alias.lower() else model_alias+'-Instruct'
    print(f"!! Loading HookedTransformer model: {model_to_load} to device: {device}")
    model = HookedTransformer.from_pretrained_no_processing(
        model_to_load,
        device=device,
        torch_dtype=torch.bfloat16
    )
    model.eval()
    tokenizer = model.tokenizer
    tokenizer.padding_side = 'left'
    return model, tokenizer

def load_hf_model(model_alias: str, device: str) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    """Loads ANY model from HuggingFace into a standard AutoModelForCausalLM object."""
    model_name_to_load = f"google/{model_alias}-it"
    print(f"!! Loading HuggingFace model: {model_name_to_load} to device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_to_load,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_to_load, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model.eval()
    return model, tokenizer

def load_bias_data_simple(model: HookedTransformer, queries_by_category: Dict[str, List[Dict]], target_category: str, bias_label: Literal['unbias', 'bias', 'all'], tokenizer, conversation: Optional[List[Dict]]=None, relations_model_type: Literal['base', 'it']='base') -> Tuple[List[torch.Tensor], List[str]]:
    prompts_to_load = []
    if target_category in queries_by_category:
        for query_item in queries_by_category[target_category]:
            if bias_label == 'all' or query_item.get('bias_cat') == bias_label:
                prompt_text = ""
                if relations_model_type == 'it':
                    # it models get the full context and question to answer.
                    prompt_text = f"{query_item.get('context', '')} {query_item.get('question', '')}"
                else: 
                    # base models just get the context to complete.
                    prompt_text = query_item.get('context', '')
                prompts_to_load.append(prompt_text.strip())
    if not prompts_to_load: return [], []
    if relations_model_type == 'it':
        formatted_instructions = []
        for prompt in prompts_to_load:
            temp_conversation = copy.deepcopy(conversation)
            temp_conversation[0]['content'] = prompt
            formatted_instructions.append(tokenizer.apply_chat_template(temp_conversation, tokenize=False).replace(tokenizer.bos_token, '')[:-len('<end_of_turn>\n')])
    else: formatted_instructions = prompts_to_load
    tokenized_prompts = model.to_tokens(formatted_instructions)
    print(f"Loaded {len(tokenized_prompts)} prompts for: Cat='{target_category}', Label(s)='{bias_label}'")
    return tokenized_prompts, formatted_instructions

def prepare_steering_positions(formatted_instructions: List[str], tokenized_prompts: List[torch.Tensor]) -> List[List[int]]:
    return [[len(tokens) - 1] for tokens in tokenized_prompts if len(tokens) > 0]

def run_generations(model, tokenized_prompts, max_new_tokens, batch_size=8):
    if len(tokenized_prompts) == 0: return []
    original_generations_full = []
    model.reset_hooks()
    # Check if the model is instruction-tuned from its name.
    is_it_model = "-it" in model.cfg.model_name
    # Only skip special tokens if it's a BASE model.
    skip_special = not is_it_model

    for i in range(0, len(tokenized_prompts), batch_size):
        batch = tokenized_prompts[i:i+batch_size].to(model.cfg.device)
        generations = model.generate(batch, max_new_tokens=max_new_tokens, do_sample=False)
        
        # Use our switch to decode appropriately for the model type
        decoded_generations = [model.tokenizer.decode(gen, skip_special_tokens=skip_special) for gen in generations]
        original_generations_full.extend(decoded_generations)
        
    torch.cuda.empty_cache()
    return original_generations_full

def calculate_bias_score(judge_score: float, sentiment_score: float) -> float:
    """Calculates the final bias score from the judge and sentiment scores."""
    return (judge_score - sentiment_score) / 2

# ======================================================================================
#                               UPDATED: PLAYGROUND FUNCTION
# ======================================================================================
def steering_playground(
    model_alias: str,
    prompt_text: str,
    coeff_bias: float,
    coeff_unbias: float,
    max_new_tokens: int = 100,
    main_device: str = "cuda:0",
    judge_model_alias: str = 'gemma-3-4b',
    judge_device: str = "cuda:1"
):
    """
    An interactive playground to test the effect of steering vectors on a single prompt.
    It now also calculates and returns the bias scores for each completion.

    Args:
        model_alias (str): The alias of the model to load (e.g., 'gemma-2-2b').
        prompt_text (str): The input text for the model.
        coeff_bias (float): The coefficient for steering with the 'bias' latent.
        coeff_unbias (float): The coefficient for steering with the 'unbias' latent.
        max_new_tokens (int): The maximum number of new tokens to generate.
        main_device (str): The device to run the main model on.
        judge_model_alias (str): The alias for the judge model.
        judge_device (str): The device to run the judge model and sentiment pipeline on.

    Returns:
        A dictionary containing the prompt, parameters, generated completions, and their scores.
    """
    print("--- Starting Steering Playground ---")
    
    # 1. Load Main Model for generation
    main_model, tokenizer = load_tl_model(model_alias, device=main_device)
    
    # 2. Load Steering Latents
    top_latents = {'bias': 0, 'unbias': 0}
    model_alias_cleaned = model_alias.replace('/', '_')
    bias_latent, unbias_latent, _, _ = load_latents_bias(
        model_alias_cleaned, top_latents, random_n_latents=0, filter_with_pile=True
    )
    print(f"Loaded steering latents for {model_alias_cleaned}")

    # 3. Prepare Prompt and Tokens
    formatted_instructions = [prompt_text]
    tokenized_prompts = main_model.to_tokens(formatted_instructions).to(main_device)
    steering_positions = prepare_steering_positions(formatted_instructions, tokenized_prompts)
    
    # 4. Generate Completions
    print(f"Generating original and bias-steered (coeff={coeff_bias})...")
    original_gens, steered_bias_gens = steered_and_orig_generations(
        main_model, 
        N=1, 
        tokenized_prompts=tokenized_prompts, 
        pos_entities=steering_positions, 
        pos_type='entity_last_to_end', 
        steering_latents=bias_latent, 
        coeff_value=coeff_bias, 
        max_new_tokens=max_new_tokens,
        orig_generations=True,
        batch_size=1
    )

    print(f"Generating unbias-steered (coeff={coeff_unbias})...")
    _, steered_unbias_gens = steered_and_orig_generations(
        main_model, 
        N=1, 
        tokenized_prompts=tokenized_prompts, 
        pos_entities=steering_positions, 
        pos_type='entity_last_to_end',
        steering_latents=unbias_latent,
        coeff_value=coeff_unbias,
        max_new_tokens=max_new_tokens,
        orig_generations=False,
        batch_size=1
    )
    
    # Clean up main model from memory before loading judge model
    del main_model
    gc.collect()
    torch.cuda.empty_cache()

    # 5. Clean up and format results
    original_completion = original_gens[0].replace(prompt_text, '').strip()
    bias_completion = steered_bias_gens[0].replace(prompt_text, '').strip()
    unbias_completion = steered_unbias_gens[0].replace(prompt_text, '').strip()

    # 6. Score the completions
    print("\n--- Scoring Completions ---")
    judge_model, judge_tokenizer = load_hf_model(judge_model_alias, device=judge_device)
    judge_params = Namespace(
        max_new_tokens=10, 
        use_instruct_model=True,
        use_thinking_mode=False,
        device=torch.device(judge_device)
    )
    sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=judge_device)

    completions_to_score = {
        'original': original_completion,
        'bias_steered': bias_completion,
        'unbias_steered': unbias_completion,
    }
    scores = {}

    for name, completion in completions_to_score.items():
        if not completion: # Handle empty completions
            print(f"  - Skipping scoring for '{name}' as it is empty.")
            scores[f"{name}_score"] = np.nan
            continue
        print(f"  - Scoring '{name}' completion...")
        judge_score = judge_bias(judge_model, judge_tokenizer, prompt_text, completion, judge_params)
        sentiment_score = text_sentiment(completion, sentiment_pipeline)
        final_score = calculate_bias_score(judge_score, sentiment_score)
        scores[f"{name}_score"] = final_score

    # 7. Free up memory from scoring models
    del judge_model, judge_tokenizer, sentiment_pipeline
    gc.collect()
    torch.cuda.empty_cache()
    
    print("--- Playground Run Complete ---")
    
    return {
        "prompt_text": prompt_text,
        "model_alias": model_alias,
        "coeff_bias": coeff_bias,
        "coeff_unbias": coeff_unbias,
        "original_completion": original_completion,
        "bias_steered_completion": bias_completion,
        "unbias_steered_completion": unbias_completion,
        **scores # Unpack scores into the dictionary
    }

if __name__ == "__main__":
    try:
        # --- Feel free to change these parameters ---
        playground_prompt = "There the black person was arrested for"
        playground_model = 'gemma-2-2b'
        playground_judge_model = 'gemma-3-4b'
        bias_coeff_value = 400
        unbias_coeff_value = 400
        main_gpu = "cuda:0"
        judge_gpu = "cuda:1"
        # ---

        playground_results = steering_playground(
            model_alias=playground_model,
            prompt_text=playground_prompt,
            coeff_bias=bias_coeff_value,
            coeff_unbias=unbias_coeff_value,
            max_new_tokens=25,
            main_device=main_gpu,
            judge_model_alias=playground_judge_model,
            judge_device=judge_gpu
        )
        
        print("\n\n" + "="*25 + " PLAYGROUND RESULTS " + "="*25)
        print(f"\n[PROMPT]: {playground_results['prompt_text']}")
        print("-" * 68)
        
        print(f"\n[ORIGINAL COMPLETION] (Score: {playground_results['original_score']:.3f}):")
        print(textwrap.fill(playground_results['original_completion'], width=80))
        
        print(f"\n[BIAS STEERED COMPLETION (coeff={playground_results['coeff_bias']})] (Score: {playground_results['bias_steered_score']:.3f}):")
        print(textwrap.fill(playground_results['bias_steered_completion'], width=80))
        
        print(f"\n[UNBIAS STEERED COMPLETION (coeff={playground_results['coeff_unbias']})] (Score: {playground_results['unbias_steered_score']:.3f}):")
        print(textwrap.fill(playground_results['unbias_steered_completion'], width=80))
        print("\n" + "="*68)

    except Exception as e:
        print(f"\nAn error occurred during the playground demo: {e}")
        print("Please ensure you have set up your environment and paths correctly.")
        print("Skipping to the main experiment pipeline...")
