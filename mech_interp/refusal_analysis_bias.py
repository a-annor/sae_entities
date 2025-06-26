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
from typing import List, Tuple, Literal, Union
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
from utils.sae_utils import load_sae
from mech_interp.mech_interp_utils import load_data, load_steering_latents
from mech_interp.mech_interp_utils import html_colors, load_latents
from utils.generation_utils import is_unbias, is_generation_refusal
from mech_interp.hooks_utils import steered_and_orig_generations
from utils.utils import tl_orthogonalize_gemma_weights, paper_plot
from dataset.load_data import load_bias_queries
from utils_bias.judge_bias import judge_bias
from utils_bias.sentiment import text_sentiment
from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")# device_map="cpu", torch_dtype=torch.float32)



import random
random_seed = 42
random.seed(random_seed)
# %%
def run_generations(model, N, tokenized_prompts, max_new_tokens, batch_size=8):
    original_generations_full = []
    N = min(N, len(tokenized_prompts))
    tokenized_prompts = tokenized_prompts[:N]
    model.reset_hooks()
    for i in range(0, len(tokenized_prompts), batch_size):
        batch_tokenized_prompts = tokenized_prompts[i:i+batch_size]
        generations = model.generate(batch_tokenized_prompts, max_new_tokens=max_new_tokens, do_sample=False)
        original_generations_full.extend([model.to_string(generation) for generation in generations])
    torch.cuda.empty_cache()
    return original_generations_full

def plot_counter_refusal(counter_refusal, save, **kwargs):
    counter_refusal_category_to_plot_label = {'original': 'Original generation', 'steered_bias': 'Steering bias latent',
                                           'steered_unbias': 'Steering unbias latent', 'orthogonalized_unbias': 'Orthogonalized model\nunbias latent',
                                           'steered_bias_random': 'Random steering\nbias latent setting', 'steered_unbias_random': 'Random steering\nunbias latent setting'}
    
    colors_dict = {'original': html_colors['blue_matplotlib'], 'steered_bias': html_colors['dark_green_drawio'],
                   'steered_unbias': html_colors['brown_D3'], 'orthogonalized_unbias': html_colors['orange_drawio'],
                   'steered_bias_random': html_colors['dark_green_drawio'], 'steered_unbias_random': html_colors['brown_D3']}
    
    hatch_dict = {'original': None, 'steered_bias': None, 'steered_unbias': None,
                  'orthogonalized_unbias': '///', 'steered_bias_random': '..', 'steered_unbias_random': '..'}
    
    entity_types = list(counter_refusal.keys())

    categories = list(counter_refusal[entity_types[0]].keys())

    category_counts = {}
    for category in categories:
        category_counts[category] = [counter_refusal[et][category] for et in entity_types]

    x = np.arange(len(entity_types))
    width = 0.3

    fig, ax = plt.subplots(figsize=(6.5, 2.75), dpi=500)
    
    bar_positions = np.arange(len(entity_types)) * 1.5  # Increase spacing between groups
    width = 0.2  # Reduce bar width

    # Calculate offsets based on number of categories
    num_categories = len(categories)
    offsets = np.linspace(-(num_categories-1)*width/2, (num_categories-1)*width/2, num_categories)

    for cat_idx, category in enumerate(categories):
        yerrs = []
        values = []
        for i in range(len(category_counts[category])):
            if isinstance(category_counts[category][i], list):
                yerrs.append(np.array(category_counts[category][i]).std())
                values.append(np.array(category_counts[category][i]).mean())
            else:
                yerrs.append(0)
                values.append(category_counts[category][i])
        ax.bar(bar_positions + offsets[cat_idx], values, width, yerr=yerrs, label=counter_refusal_category_to_plot_label[category],
               color=colors_dict[category], alpha=1, edgecolor='black', linewidth=1, hatch=hatch_dict[category])


    ax.set_ylabel('Refusal Rate')
    #ax.set_title('Refusal Counts by Entity Type: Original vs Orthogonalized')
    ax.set_xticks(bar_positions)
    entity_types = [entity.capitalize() for entity in entity_types]
    ax.set_xticklabels(entity_types, rotation=0)
    #ax.legend()
    # Adjust the x positions to spread the bars more
    
    
    # Add light grid lines
    ax.grid(True, linestyle=(0, (5, 10)))
    ax.set_axisbelow(True)

    # Set y-axis to show only integers
    y_max = 101#max(max(steered_unbias_counts), max(steered_unbias_counts))
    ax.set_yticks(range(int(y_max) + 1))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # Set y-axis ticks to be smaller
    ax.tick_params(axis='y', which='major', labelsize=8)
    # Make legend transparent
    #legend = ax.legend(framealpha=0.3)
    #ax.legend(loc='center left')
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")


    #plt.setp(legend.get_texts(), fontsize=8)  # Reduce font size of legend text

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    #ax.bar_label(rects1, padding=3)
    #ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    plt.ylim(0,100)
    os.makedirs('./plots/bias_analysis', exist_ok=True)
    if save == True:
        plt.savefig(f"./plots/bias_analysis/v2_{kwargs['model_alias']}_k{kwargs['top_latents']['bias']}_u{kwargs['top_latents']['unbias']}_{kwargs['pos_type']}_k{kwargs['coeff_values']['bias']}_u{kwargs['coeff_values']['unbias']}_{kwargs['split']}_{kwargs['bias_label']}.png", dpi=500, transparent=True)
        plt.savefig(f"./plots/bias_analysis/v2_{kwargs['model_alias']}_k{kwargs['top_latents']['bias']}_u{kwargs['top_latents']['unbias']}_{kwargs['pos_type']}_k{kwargs['coeff_values']['bias']}_u{kwargs['coeff_values']['unbias']}_{kwargs['split']}_{kwargs['bias_label']}.pdf", transparent=True)

def load_model(model_alias):
    model_alias = model_alias.replace('_','/')
    model_to_load = model_alias+'-it' if 'gemma' in model_alias.lower() else model_alias+'-Instruct'
    model = HookedTransformer.from_pretrained_no_processing(
                model_to_load,
            )
    tokenizer = model.tokenizer
    tokenizer.padding_side = 'left'
    return model, tokenizer



def count_refusals(generations):
    counter_refusals = 0
    for generation in generations:
        pos_end_of_instruction = generation.find(end_of_instruction_string)
        generation = generation[pos_end_of_instruction+len(end_of_instruction_string):]
        if is_generation_refusal(generation) == True:
            counter_refusals += 1
    return counter_refusals
##### Experiments original model and steering
# %%
# # Load model
# model_alias = 'gemma-2-2b'
# model = HookedTransformer.from_pretrained_no_processing(
#     'gemma-2-2b-it',
# )
# #model.set_use_attn_result(False)
# tokenizer = model.tokenizer
# tokenizer.padding_side = 'left'

# %%
model_alias = 'gemma-2-2b'

prompt_template = '{relation} the {entity_type} {entity_name}?'

conversation =[
      {
        "role": "user",
        "content": ""
      },
      {
        "role": "assistant",
        "content": ""
      }
    ]

end_of_instruction_string = '<end_of_turn>\n<start_of_turn>model\n'

# %%
model, tokenizer = load_model(model_alias)
model_alias = model_alias.replace('/','_')
tokenized_prompts_dict_entity_type = {}
pos_entities_dict_entity_type = {}
formatted_instructions_dict_entity_type = {}
queries = load_bias_queries(model_alias)
ALL_BIAS_TYPES = ['Race_ethnicity']
for entity_type in ALL_BIAS_TYPES:
    tokenized_prompts_dict_entity_type[entity_type] = {}
    pos_entities_dict_entity_type[entity_type] = {}
    formatted_instructions_dict_entity_type[entity_type] = {}
    for bias_label in ['unbias', 'bias']:
        tokenized_prompts_dict_entity_type[entity_type][bias_label], pos_entities_dict_entity_type[entity_type][bias_label], formatted_instructions_dict_entity_type[entity_type][bias_label], _ = load_data(model, queries, entity_type, tokenizer,
                                                                                                                    bias_label, prompt_template, conversation,
                                                                                                                    relations_model_type='it')

del model
gc.collect()
torch.cuda.empty_cache()

# %%

bias_label = 'unbias'
pos_type = 'entity_last_to_end'
N=100
max_new_tokens = 30
batch_size = 100
top_latents = {'bias': 0, 'unbias': 0}
coeff_values = {'bias': 400, 'unbias': 400}
split = 'test'

categories = ['original', 'steered_unbias', 'steered_unbias_random', 'orthogonalized_unbias', 'steered_bias', 'steered_bias_random']
# for idx in range(1,5):
bias_latent, unbias_latent, random_latents_bias, random_latents_unbias = load_latents(model_alias, top_latents,
                                                                                          random_n_latents=10,
                                                                                          filter_with_pile=True)
print(unbias_latent)
print(bias_latent)

counter_refusal = {}
params_args = {'model_alias': model_alias, 'coeff_values': coeff_values, 'split': split, 'top_latents': top_latents, 'pos_type': pos_type, 'bias_label': bias_label}

for e_idx, entity_type in enumerate(ALL_BIAS_TYPES):
    counter_refusal[entity_type] = {}
    for category in categories:
        counter_refusal[entity_type][category] = 0

    if e_idx > 0:
        if 'orthogonalized_unbias' in categories or 'orthogonalized_bias' in categories:
            # Load model again if orthogonalized (and deleted) before
            model, tokenizer = load_model(model_alias)
    else:
        # Always load model first thing
        model, tokenizer = load_model(model_alias)
    

    tokenized_prompts, pos_entities, formatted_instructions = tokenized_prompts_dict_entity_type[entity_type][bias_label], pos_entities_dict_entity_type[entity_type][bias_label], formatted_instructions_dict_entity_type[entity_type][bias_label]

    if 'original' in categories:
        original_generations_full, bias_steered_generations_full = steered_and_orig_generations(model, N, tokenized_prompts, pos_entities, pos_type=pos_type,
                                                                                        steering_latents=bias_latent, ablate_latents=None,
                                                                                        coeff_value=coeff_values['bias'], max_new_tokens=max_new_tokens, batch_size=batch_size)

        counter_refusal[entity_type]['original'] = count_refusals(original_generations_full)
        counter_refusal[entity_type]['steered_bias'] = count_refusals(bias_steered_generations_full)

    if 'steered_unbias' in categories:
        _, unbias_steered_generations_full = steered_and_orig_generations(model, N, tokenized_prompts, pos_entities, pos_type=pos_type,
                                                                                        steering_latents=unbias_latent, ablate_latents=None,
                                                                                        coeff_value=coeff_values['unbias'], max_new_tokens=max_new_tokens,
                                                                                        orig_generations=False, batch_size=batch_size)
        
        counter_refusal[entity_type]['steered_unbias'] = count_refusals(unbias_steered_generations_full)

    if 'steered_bias_random' in categories:
        random_latents_counter = []
        for random_latent in random_latents_bias:
            _, random_steered_generations_full = steered_and_orig_generations(model, N, tokenized_prompts, pos_entities, pos_type=pos_type,
                                                                                        steering_latents=[random_latent], ablate_latents=None,
                                                                                        coeff_value=coeff_values['bias'], max_new_tokens=max_new_tokens,
                                                                                        orig_generations=False, batch_size=batch_size)
            
            random_latents_counter.append(count_refusals(random_steered_generations_full))

        counter_refusal[entity_type]['steered_bias_random'] = random_latents_counter

    if 'steered_unbias_random' in categories:
        random_latents_counter = []
        for random_latent in random_latents_unbias:
            _, random_steered_generations_full = steered_and_orig_generations(model, N, tokenized_prompts, pos_entities, pos_type=pos_type,
                                                                                    steering_latents=[random_latent], ablate_latents=None,
                                                                                    coeff_value=coeff_values['unbias'], max_new_tokens=max_new_tokens,
                                                                                    orig_generations=False, batch_size=batch_size)
            random_latents_counter.append(count_refusals(random_steered_generations_full))

        counter_refusal[entity_type]['steered_unbias_random'] = random_latents_counter
    
    if 'orthogonalized_unbias' in categories or 'orthogonalized_bias' in categories:
        for ortho_idx, orhogonalization_type in enumerate(['unbias']):
            if orhogonalization_type == 'unbias':
                direction = unbias_latent[0][-1]
            else:
                direction = bias_latent[0][-1]
            if ortho_idx == 1:
                # Load model again
                model, tokenizer = load_model(model_alias)
            
            tl_orthogonalize_gemma_weights(model, direction=direction)

            # Run generations with orthogonalized model
            orthogonalized_generations_full = run_generations(model, N, tokenized_prompts, max_new_tokens, batch_size)
            torch.cuda.empty_cache()

            counter_refusal[entity_type][f'orthogonalized_{orhogonalization_type}'] = count_refusals(orthogonalized_generations_full)

            del model
        gc.collect()
        torch.cuda.empty_cache()

    print('counter_refusal', counter_refusal)
    # del model
    # gc.collect()
    # torch.cuda.empty_cache()

# Save counter_refusal results
save_path = f'results/refusal_analysis/{model_alias}'
os.makedirs(save_path, exist_ok=True)
with open(os.path.join(save_path, f'counter_refusal_{entity_type}_{bias_label}.json'), 'w') as f:
    json.dump(counter_refusal, f)

plot_counter_refusal(counter_refusal, save=True,**params_args)




# %%
### Test ###
# %%
model, tokenizer = load_model(model_alias)

# %%
entity_type = 'Race_ethnicity'
bias_label = 'unbias'
N = 100
max_new_tokens = 30
batch_size = 50
top_latents = {'bias': 0, 'unbias': 0}
coeff_values = {'bias': 20, 'unbias': 20}
model_alias = model_alias.replace('/','_')
# %%
bias_latent, unbias_latent, random_latents_bias, random_latents_unbias = load_latents(model_alias, top_latents,
                                                                                          filter_with_pile=True,
                                                                                          random_n_latents=10)
# %%
# unbias_latent = load_steering_latents('movie', label='unbias', topk=1,
#                                         #layers_range=[unbias_latent[0]],
#                                         specific_latents=[(16,9583)],
#                                         model_alias=model_alias,
#                                         random_latents=False)


# %%
tokenized_prompts, pos_entities, formatted_instructions = tokenized_prompts_dict_entity_type[entity_type][bias_label], pos_entities_dict_entity_type[entity_type][bias_label], formatted_instructions_dict_entity_type[entity_type][bias_label]

counter_refusals = {}


original_generations_full, steered_generations_full = steered_and_orig_generations(model, N, tokenized_prompts, pos_entities, pos_type='entity_last_to_end',
                                                                                steering_latents=[random_latents_bias[-2]], ablate_latents=None,
                                                                                coeff_value=coeff_values['bias'], max_new_tokens=max_new_tokens,
                                                                                orig_generations=True, batch_size=batch_size)

counter_refusals['original'] = count_refusals(original_generations_full)
counter_refusals['steered'] = count_refusals(steered_generations_full)


# counter_refusals['steered_bias_random'] = []
# counter_refusals['steered_unbias_random'] = []
# for random_latent in random_latents_bias:
#     _, random_steered_generations_full = steered_and_orig_generations(model, N, tokenized_prompts, pos_entities, pos_type='entity_last_to_end',
#                                                                                 steering_latents=[random_latent], ablate_latents=None,
#                                                                                 coeff_value=coeff_values['bias'], max_new_tokens=max_new_tokens,
#                                                                                 orig_generations=False, batch_size=batch_size)

#     counter_refusals['steered_bias_random'].append(count_refusals(random_steered_generations_full))

# for random_latent in random_latents_unbias:
#     _, random_steered_generations_full = steered_and_orig_generations(model, N, tokenized_prompts, pos_entities, pos_type='entity_last_to_end',
#                                                                                 steering_latents=[random_latent], ablate_latents=None,
#                                                                                 coeff_value=coeff_values['unbias'], max_new_tokens=max_new_tokens,
#                                                                                 orig_generations=False, batch_size=batch_size)

#     counter_refusals['steered_unbias_random'].append(count_refusals(random_steered_generations_full))

for i in range(len(original_generations_full)):
    clean_completion = original_generations_full[i].replace(formatted_instructions[i], '').replace('<bos>', '').replace('<pad>','')
    steered_completion = steered_generations_full[i].replace(formatted_instructions[i], '').replace('<bos>', '').replace('<pad>','')
    print(Fore.WHITE + f"QUESTION:")
    print(textwrap.fill(repr(formatted_instructions[i].replace('<start_of_turn>user\n', '').replace('<end_of_turn>\n<start_of_turn>model','').replace('<bos>','')), width=100, initial_indent='\t', subsequent_indent='\t'))
    print(Fore.GREEN + f"ORIGINAL COMPLETION:")
    print(textwrap.fill(repr(clean_completion.replace('<eos>', '').replace('\n<end_of_turn>', '').replace('<end_of_turn>\n<start_of_turn>model\n','')), width=100, initial_indent='\t', subsequent_indent='\t'))
    print('is refusal:', is_generation_refusal(clean_completion))
    #counter_refusals['original'] += is_generation_refusal(clean_completion)
    print(Fore.RED + f"STEERED COMPLETION:")
    print(textwrap.fill(repr(steered_completion.replace('<eos>', '').replace('\n<end_of_turn>','')), width=100, initial_indent='\t', subsequent_indent='\t'))
    print('is refusal:', is_generation_refusal(steered_completion))
    #counter_refusals['steered'] += is_generation_refusal(steered_completion)
    print(Fore.RESET)


counter_refusals
# %%
