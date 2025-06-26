# %%
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')

# %%
# import sys
# sys.path.append("../..")
# sys.path.append("../dataset")
# sys.path.append("./")
# sys.path.append("../")
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import json
import torch
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
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
from utils.hf_models.model_factory import construct_model_base
from utils.utils import model_alias_to_model_name
from transformer_lens import HookedTransformer, ActivationCache
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens.loading_from_pretrained import get_pretrained_model_config 
from utils.sae_utils import load_sae
from mech_interp.mech_interp_utils import load_data, load_steering_latents_bias
from mech_interp.mech_interp_utils import html_colors, load_latents_bias
from utils.generation_utils import is_unbias, is_generation_refusal
from mech_interp.hooks_utils import steered_and_orig_generations
from utils.utils import tl_orthogonalize_gemma_weights, paper_plot
from dataset.load_data import load_bias_queries
from utils_bias.judge_bias import judge_bias
from utils_bias.sentiment import text_sentiment
from utils_bias.generate_completions import load_model
from transformers import pipeline, PreTrainedTokenizer
from argparse import Namespace



import random
random_seed = 42
random.seed(random_seed)
# %%

def load_tl_model(model_alias: str, device: str) -> Tuple[HookedTransformer, PreTrainedTokenizer]:
    """Loads a natively supported model into HookedTransformer."""
    model_name_to_load = f"google/{model_alias}"
    print(f"!! Loading HookedTransformer model: {model_name_to_load} to device: {device}")
    model = HookedTransformer.from_pretrained_no_processing(
        model_name_to_load,
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
    return (judge_score - sentiment_score) / 2

# Plotting Function 
def plot_steering_across_bias_types(all_scores_data: Dict[str, Dict[str, list]], save: bool, **kwargs):
    method_to_plot_label = {'original': 'Original generation', 'steered_unbias': 'Steering unbias latent', 'steered_unbias_random': 'Random steering\nUnbias latent setting', 'orthogonalized_unbias': 'Orthogonalized model\nUnbias latent', 'steered_bias': 'Steering bias latent', 'steered_bias_random': 'Random steering\nBias latent setting'}
    colors_dict = {'original': '#1f77b4', 'steered_unbias': '#8c564b', 'steered_unbias_random': '#8c564b', 'orthogonalized_unbias': '#ff7f0e', 'steered_bias': '#2ca02c', 'steered_bias_random': '#2ca02c'}
    hatch_dict = {'original': None, 'steered_unbias': None, 'steered_bias': None, 'orthogonalized_unbias': '///', 'steered_unbias_random': '..', 'steered_bias_random': '..'}
    x_axis_groups = list(all_scores_data.keys())
    if not x_axis_groups: print("No data to plot."); return
    methods_in_order = list(method_to_plot_label.keys())
    x = np.arange(len(x_axis_groups)); width = 0.12; num_methods = len(methods_in_order)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    offsets = np.linspace(-width * (num_methods / 2 - 0.5), width * (num_methods / 2 - 0.5), num_methods)
    for i, method in enumerate(methods_in_order):
        means = [np.mean(all_scores_data[group].get(method, [np.nan])) for group in x_axis_groups]
        stds = [np.std(all_scores_data[group].get(method, [np.nan])) for group in x_axis_groups]
        valid_indices = ~np.isnan(means)
        if not np.any(valid_indices): continue
        
        # Conditional plotting for error bars
        errors_to_plot = None
        if 'random' in method:
             errors_to_plot = np.array(stds)[valid_indices]
        
        ax.bar(x[valid_indices] + offsets[i], np.array(means)[valid_indices], width, 
               yerr=errors_to_plot, label=method_to_plot_label.get(method, method), 
               color=colors_dict.get(method), hatch=hatch_dict.get(method), 
               edgecolor='black', linewidth=1, capsize=3)

    ax.set_ylabel('Bias Score ((Judge Score - Sentiment) / 2)', fontsize=12)
    ax.set_title(f'Effect of Steering Methods Across Bias Categories', fontsize=16, weight='bold')
    ax.set_xticks(x); ax.set_xticklabels([g.replace("_", " ") for g in x_axis_groups], fontsize=12)
    ax.grid(True, linestyle='--', axis='y', alpha=0.7); ax.set_axisbelow(True)
    ax.set_ylim(0, 1.0) # Set Y-axis from 0 to 1
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    if save:
        folder_path = f'./plots/bias_analysis/{kwargs["main_model_alias"]}'; os.makedirs(folder_path, exist_ok=True)
        filename = f'{folder_path}/final_bias_steering_comparison.png'; plt.savefig(filename, dpi=300, transparent=True)
        pdf_filename = f'{folder_path}/final_bias_steering_comparison.pdf'; plt.savefig(pdf_filename, transparent=True)
        print(f"Plot saved to {filename} and {pdf_filename}")
    plt.show()

##### Main Experiments Flow #####
# %%
#  Config
MAIN_MODEL_ALIAS = 'gemma-2-2b'
JUDGE_MODEL_ALIAS = 'gemma-3-4b'
MAIN_MODEL_DEVICE = "cuda:0"
JUDGE_MODEL_DEVICE = "cuda:1" 
conversation = [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]
main_model_alias_cleaned = MAIN_MODEL_ALIAS.replace('/', '_')
N_prompts_per_category = 121
max_new_tokens = 100
batch_size = 10
top_latents = {'bias': 0, 'unbias': 0}
coeff_values = {'bias': 400, 'unbias': 400}
methods_to_evaluate = ['original', 'steered_unbias', 'steered_unbias_random', 'orthogonalized_unbias', 'steered_bias', 'steered_bias_random']
BIAS_TYPES_TO_ANALYZE = ['Race_ethnicity', 'Gender', 'Religion', 'Nationality']
all_bias_queries_data = load_bias_queries()

bias_latent, unbias_latent, random_latents_bias, random_latents_unbias = load_latents_bias(main_model_alias_cleaned, top_latents, random_n_latents=3, filter_with_pile=True)
all_generations = defaultdict(lambda: defaultdict(dict))

# %%
# ======================================================================================
#                                   PHASE 1: GENERATION & STEERING
# ======================================================================================
print("--- STARTING PHASE 1: GENERATION ---")
for bias_type in BIAS_TYPES_TO_ANALYZE:
    print(f"\n{'='*20} Processing Category: {bias_type} {'='*20}")
    
    print(f"  - Loading main model for {bias_type}...")
    # The model is loaded once per category. We will make a copy for the orthogonalize step later.
    main_model, _ = load_tl_model(MAIN_MODEL_ALIAS, device=MAIN_MODEL_DEVICE)
    tokenized_prompts, formatted_instructions = load_bias_data_simple(main_model, all_bias_queries_data, target_category=bias_type, bias_label='all', tokenizer=main_model.tokenizer, conversation=conversation, relations_model_type='base')

    if not formatted_instructions:
        print(f"    - No prompts found for {bias_type}. Skipping.")
        del main_model; gc.collect(); torch.cuda.empty_cache()
        continue
    
    tokenized_prompts_for_eval = tokenized_prompts
    formatted_instructions_for_eval = formatted_instructions
    print(f"  - Processing {N_prompts_per_category} prompts for this category...")
    steering_positions = prepare_steering_positions(formatted_instructions_for_eval, tokenized_prompts_for_eval)
    
    # --- This block now uses a series of 'if' statements, matching the reference code ---

    # 1. Handle 'original' and 'steered_bias' (known) together for efficiency
    if 'original' in methods_to_evaluate:
        print("    - Generating: original & steered_bias (known)")
        original_gens, steered_bias_gens = steered_and_orig_generations(
            main_model, N_prompts_per_category, tokenized_prompts_for_eval, steering_positions, 
            pos_type='entity_last_to_end', 
            steering_latents=bias_latent, 
            ablate_latents=None,
            coeff_value=coeff_values['bias'], 
            max_new_tokens=max_new_tokens,
            orig_generations=True, # This is the key optimization
            batch_size=batch_size
        )
        
        prompts_for_method = [p.split('<end_of_turn>\n<start_of_turn>model\n')[0] for p in formatted_instructions_for_eval]
        
        all_generations[bias_type]['original']['prompts'] = prompts_for_method
        all_generations[bias_type]['original']['completions'] = [g.replace(p, '').strip() for g, p in zip(original_gens, prompts_for_method)]
        print("      - Stored 'original' results.")

        if 'steered_bias' in methods_to_evaluate:
            all_generations[bias_type]['steered_bias']['prompts'] = prompts_for_method
            all_generations[bias_type]['steered_bias']['completions'] = [g.replace(p, '').strip() for g, p in zip(steered_bias_gens, prompts_for_method)]
            print("      - Stored 'steered_bias' results.")

    # 2. Handle 'steered_unbias' (unknown)
    if 'steered_unbias' in methods_to_evaluate:
        print("    - Generating: steered_unbias (unknown)")
        _, steered_unbias_gens = steered_and_orig_generations(
            main_model, N_prompts_per_category, tokenized_prompts_for_eval, steering_positions, 
            pos_type='entity_last_to_end',
            steering_latents=unbias_latent,
            ablate_latents=None,
            coeff_value=coeff_values['unbias'],
            max_new_tokens=max_new_tokens,
            orig_generations=False, # We don't need the original again
            batch_size=batch_size
        )
        prompts_for_method = [p.split('<end_of_turn>\n<start_of_turn>model\n')[0] for p in formatted_instructions_for_eval]
        all_generations[bias_type]['steered_unbias']['prompts'] = prompts_for_method
        all_generations[bias_type]['steered_unbias']['completions'] = [g.replace(p, '').strip() for g, p in zip(steered_unbias_gens, prompts_for_method)]
        print("      - Stored 'steered_unbias' results.")

    # 3. Handle random steering methods
    if 'steered_bias_random' in methods_to_evaluate:
        # (This logic remains the same)
        print("    - Generating: steered_bias_random")
        generations_list = []
        for r_latent in random_latents_bias:
            _, steered_gens = steered_and_orig_generations(main_model, N_prompts_per_category, tokenized_prompts_for_eval, steering_positions, 'entity_last_to_end', [r_latent], None, coeff_values['bias'], max_new_tokens, False, batch_size)
            generations_list.append(steered_gens)
        prompts_for_method = [p.split('<end_of_turn>\n<start_of_turn>model\n')[0] for p in formatted_instructions_for_eval]
        all_generations[bias_type]['steered_bias_random']['prompts'] = prompts_for_method
        all_generations[bias_type]['steered_bias_random']['completions_per_run'] = [[g.replace(p, '').strip() for g, p in zip(run, prompts_for_method)] for run in generations_list]
        print("      - Stored 'steered_bias_random' results.")

    if 'steered_unbias_random' in methods_to_evaluate:
        # (This logic also remains the same)
        print("    - Generating: steered_unbias_random")
        generations_list = []
        for r_latent in random_latents_unbias:
            _, steered_gens = steered_and_orig_generations(main_model, N_prompts_per_category, tokenized_prompts_for_eval, steering_positions, 'entity_last_to_end', [r_latent], None, coeff_values['unbias'], max_new_tokens, False, batch_size)
            generations_list.append(steered_gens)
        prompts_for_method = [p.split('<end_of_turn>\n<start_of_turn>model\n')[0] for p in formatted_instructions_for_eval]
        all_generations[bias_type]['steered_unbias_random']['prompts'] = prompts_for_method
        all_generations[bias_type]['steered_unbias_random']['completions_per_run'] = [[g.replace(p, '').strip() for g, p in zip(run, prompts_for_method)] for run in generations_list]
        print("      - Stored 'steered_unbias_random' results.")

    # 4. Handle orthogonalization
    if 'orthogonalized_unbias' in methods_to_evaluate:
        # (This logic also remains the same, using deepcopy)
        print("    - Generating: orthogonalized_unbias")
        print("      - Creating a deep copy of the model for orthogonalization...")
        model_for_this_run = copy.deepcopy(main_model)
        direction = unbias_latent[0][3]
        tl_orthogonalize_gemma_weights(model_for_this_run, direction=direction)
        generations_list = run_generations(model_for_this_run, tokenized_prompts_for_eval, max_new_tokens, batch_size)
        prompts_for_method = [p.split('<end_of_turn>\n<start_of_turn>model\n')[0] for p in formatted_instructions_for_eval]
        all_generations[bias_type]['orthogonalized_unbias']['prompts'] = prompts_for_method
        all_generations[bias_type]['orthogonalized_unbias']['completions'] = [g.replace(p, '').strip() for g, p in zip(generations_list, prompts_for_method)]
        print("      - Stored 'orthogonalized_unbias' results.")
    
    # Delete the main model AFTER all methods for this bias_type are complete
    del main_model; gc.collect(); torch.cuda.empty_cache()
    print(f"  - Unloaded main model for {bias_type}.")

print("\n--- PHASE 1: GENERATION COMPLETE ---")
with open('all_generations.json', 'w') as f:
    json.dump(all_generations, f, indent=2)

# %%
# ======================================================================================
#                                   PHASE 2: SCORING
# ======================================================================================
print("\n--- STARTING PHASE 2: SCORING ---")
judge_model, judge_tokenizer = load_hf_model(JUDGE_MODEL_ALIAS, device=JUDGE_MODEL_DEVICE)
judge_params = Namespace(
    max_new_tokens=10, 
    use_instruct_model=True,
    use_thinking_mode=False,
    device = torch.device(f"{JUDGE_MODEL_DEVICE}")
    )
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=JUDGE_MODEL_DEVICE)
all_results_for_plotting = defaultdict(lambda: defaultdict(list))
with open('all_generations.json', 'r') as f:
    all_generations = json.load(f)

for bias_type, methods_data in all_generations.items():
    print(f"\nScoring category: {bias_type}")
    for method, data in methods_data.items():
        prompts = data['prompts']
        if 'random' in method:
            avg_scores_for_random_runs = []
            for run_completions in data['completions_per_run']:
                scores_for_this_run = []
                for i, completion in enumerate(run_completions):
                    judge_score = judge_bias(judge_model, judge_tokenizer, prompts[i], completion, judge_params)
                    sentiment_score = text_sentiment(completion, sentiment_pipeline)
                    scores_for_this_run.append(calculate_bias_score(judge_score, sentiment_score))
                avg_scores_for_random_runs.append(np.mean(scores_for_this_run))
            all_results_for_plotting[bias_type][method] = avg_scores_for_random_runs
        else:
            scores_for_this_method = []
            completions = data['completions']
            for i, completion in enumerate(completions):
                judge_score = judge_bias(judge_model, judge_tokenizer, prompts[i], completion, judge_params)
                sentiment_score = text_sentiment(completion, sentiment_pipeline)
                scores_for_this_method.append(calculate_bias_score(judge_score, sentiment_score))
            all_results_for_plotting[bias_type][method] = scores_for_this_method
print("\n--- PHASE 2: SCORING COMPLETE ---")
del judge_model, judge_tokenizer, sentiment_pipeline; gc.collect(); torch.cuda.empty_cache()

# ======================================================================================
#                                   PHASE 3: PLOTTING
# ======================================================================================
print("\n--- STARTING PHASE 3: PLOTTING ---")
plot_params = {'main_model_alias': main_model_alias_cleaned}
plot_steering_across_bias_types(all_results_for_plotting, save=True, **plot_params)

print("\nAnalysis complete.")