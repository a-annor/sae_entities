# import sys
# sys.path.append('./')
# sys.path.append('../')
# sys.path.append('../..')
# sys.path.append('../../..')

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import os
import torch
from typing import List, Callable, Dict, Callable, Union, Tuple
import os
import gc
import time
import json
from tqdm import tqdm
import glob
from jaxtyping import Int, Float
from torch import Tensor
import torch
import numpy as np
import random
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.utils.data import Dataset as TorchDataset
import json
\
from utils.hf_patching_utils import add_hooks
from utils.hf_models.model_base import ModelBase
from utils.hf_models.model_factory import construct_model_base
from utils.utils import clear_memory, model_alias_to_model_name, find_string_in_tokens
from typing import List, Callable, Any
random.seed(10)

DEFAULT_CACHE_DIR = os.path.join("dataset/cached_activations")

def _get_activations_pre_hook(cache: Float[Tensor, "pos d_model"]):
    def hook_fn(module, input):
        nonlocal cache
        activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone().to(cache)
        cache[:, :] += activation[:, :].to(cache)
    return hook_fn

@torch.no_grad()
def _get_activations_fixed_seq_len(model, tokenizer, prompts: List[str], block_modules: List[torch.nn.Module], seq_len: int = 512, layers: List[int]=None, batch_size=32, save_device: Union[torch.device, str] = "cuda", verbose=True) -> Tuple[Float[Tensor, 'n seq_len'], Float[Tensor, 'n layer seq_len d_model']]:
    torch.cuda.empty_cache()

    if layers is None:
        layers = range(model.config.num_hidden_layers)

    n_layers = len(layers)
    d_model = model.config.hidden_size

    # we store the activations in high-precision to avoid numerical issues
    activations = torch.zeros((len(prompts), n_layers, seq_len, d_model), device=save_device)
    all_input_ids = torch.zeros((len(prompts), seq_len), dtype=torch.long, device=save_device)

    # Fill all_input_ids with tokenizer.pad_token_id
    all_input_ids.fill_(tokenizer.pad_token_id)

    for i in tqdm(range(0, len(prompts), batch_size), disable=not verbose):
        inputs = tokenizer(prompts[i:i+batch_size], return_tensors="pt", padding=True, truncation=True, max_length=seq_len)

        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        inputs_len = len(input_ids)
        num_input_toks = input_ids.shape[-1]

        fwd_pre_hooks = [
            (block_modules[layer], _get_activations_pre_hook(cache=activations[i:i+inputs_len, layer_idx, -num_input_toks:, :])) 
            for layer_idx, layer in enumerate(layers)
        ]

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        all_input_ids[i:i+inputs_len, -num_input_toks:] = input_ids

    return all_input_ids, activations

# %%

def get_compute_activation_fn(model_base: ModelBase, layers: List[int]=None, seq_len: int = 128, batch_size: int = 32):

    # print(f"WARNING: Saving only activations at the last 6 token positions")

    def compute_activation_fn(prompts: List[str]):
        nonlocal model_base, layers, seq_len, batch_size
        n_layers = model_base.model.config.num_hidden_layers
        if layers is None:
            layers = range(n_layers)

        input_ids, activations  = _get_activations_fixed_seq_len(
            model_base.model,
            model_base.tokenizer,
            prompts=prompts,
            block_modules=model_base.model_block_modules,
            seq_len=seq_len,
            layers=layers,
            batch_size=batch_size,
            save_device='cuda',
            verbose=False
        )
        activations: Float[Tensor, 'n n_layers seq_len d_model'] = activations.cpu()
        input_ids: Int[Tensor, 'n seq_len'] = input_ids.cpu()

        n_layers = activations.shape[1]
        assert n_layers == len(layers), f"Expected {len(layers)} layers, but got {n_layers}"

        # save only the activations at the last 6 token positions
        # activations = activations[:, :, -6:, :]

        return input_ids, activations

    return compute_activation_fn


# def cache_activations(model_base: ModelBase, prompts: List[str], compute_activation_fn: Callable,
#                       tokens_to_cache: str, layers: List[int],foldername: str,
#                       batch_size: int = 32, seq_len: int = 128, shard_size: int = 1000,
#                       substrings: List[str] = None, n_positions: int = 1) -> None:

#     d_model = model_base.model.config.hidden_size
#     if layers is None:
#         n_layers = model_base.model.config.num_hidden_layers
#     else:
#         n_layers = len(layers)

#     # full_input_ids = model_base.tokenizer(prompts, return_tensors="pt", padding=True).input_ids[:,:seq_len]    
#     # slices_to_cache = [find_string_in_tokens(substring, full_input_ids[j], model_base.tokenizer) for j, substring in enumerate(substrings)]
#     # discarded_indexes = [i for i, s in enumerate(slices_to_cache) if s is None]

#     print('shape shard size', (shard_size, n_layers, n_positions, d_model))
#     # Create memmap files

#     memmap_file_acts = np.memmap(f"{foldername}/acts.dat", dtype='float32', mode='w+', 
#                             shape=(shard_size, n_layers, n_positions, d_model))
#     memmap_file_ids = np.memmap(f"{foldername}/ids.dat", dtype='float32', mode='w+', 
#                             shape=(shard_size, seq_len))

#     total_n = 0
#     for i in tqdm(range(0, len(prompts), batch_size), desc="Computing activations"):

#         batch_prompts = prompts[i:i+batch_size]
#         batch_substrings_tmp = substrings[i:i+batch_size] if substrings is not None else None
#         # remove bos
#         batch_prompts_tmp = [prompt.replace('<bos>', '') for prompt in batch_prompts]

#         ### Check if the substrings are in the input ids
#         inputs_ = model_base.tokenizer(batch_prompts_tmp, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
#         batch_prompts = []
#         batch_substrings = []
#         for p in range(len(batch_prompts_tmp)):
#             ids = inputs_.input_ids[p]
#             prompt_til_seq_len = model_base.tokenizer.decode(ids)
#             if batch_substrings_tmp[p] in prompt_til_seq_len:
#                 batch_prompts.append(batch_prompts_tmp[p])
#                 batch_substrings.append(batch_substrings_tmp[p])
        
#         # Get input ids and activations
#         input_ids, activations = compute_activation_fn(prompts=batch_prompts)
        
#         # Ensure activations are on CPU for numpy conversion later
#         activations = activations.cpu()
#         input_ids = input_ids.cpu()

#         # Initialize list to store sliced activations for this batch
#         current_batch_activations_sliced = []
#         # current_batch_input_ids_full = [] # If you want to store the full input_ids for the cached items

#         # Iterate through each item in the batch
#         for j in range(activations.shape[0]): # activations.shape[0] is the actual batch size after compute_activation_fn
            
#             # Determine the actual sequence length for this specific example (ignoring padding)
#             # Find the index of the first padding token, or use the full seq_len if no padding.
#             actual_len_idx = (input_ids[j] == model_base.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
#             if actual_len_idx.numel() > 0: # If padding token found
#                 actual_len = actual_len_idx[0].item()
#             else: # No padding token found, use full seq_len
#                 actual_len = input_ids[j].shape[0] # This should be `seq_len`

#             if actual_len == 0:
#                 # Skip empty sequences resulting from tokenization
#                 print(f"Warning: Skipping example {i+j} due to empty tokenized sequence.")
#                 continue

#             # Calculate the start position for slicing the last n_positions tokens
#             # Ensure start_pos is not negative
#             start_pos = max(0, actual_len - n_positions)

#             # Perform the slicing: [batch_idx, layer_idx, token_idx, d_model_idx]
#             current_activations_for_item = activations[j, :, start_pos:actual_len, :]

#             # IMPORTANT: Pad if the actual sequence length is less than n_positions
#             # This ensures consistent shape (n_layers, n_positions, d_model) for each item
#             if current_activations_for_item.shape[1] < n_positions:
#                 num_missing_positions = n_positions - current_activations_for_item.shape[1]
#                 # Create a zero-padded tensor of the correct target shape
#                 padded_activations = torch.zeros(
#                     n_layers, n_positions, d_model,
#                     dtype=current_activations_for_item.dtype,
#                     device=current_activations_for_item.device
#                 )
#                 # Copy the actual activations into the END of the padded tensor
#                 padded_activations[:, num_missing_positions:, :] = current_activations_for_item
#                 current_activations_for_item = padded_activations
            
#             current_batch_activations_sliced.append(current_activations_for_item)
#             # current_batch_input_ids_full.append(input_ids[j]) # If you need this
            
#             # For debugging, print the tokens being cached for the first example
#             if i == 0 and j == 0:
#                 print('--- Debugging Slicing ---')
#                 print('Original Prompt:', batch_prompts_tmp[j])
#                 print('Full Token IDs (first 20):', input_ids[j][:20].tolist(), '...')
#                 print('Decoded Full Input (truncated to seq_len):', model_base.tokenizer.decode(input_ids[j], skip_special_tokens=False))
#                 print('Actual sequence length (non-padded):', actual_len)
#                 print(f'Slicing from index {start_pos} to {actual_len} (exclusive). Taking {n_positions} tokens.')
#                 print('Token(s) that will be cached:', model_base.tokenizer.decode(input_ids[j, start_pos:actual_len], skip_special_tokens=False))
#                 print(f"Shape of activations for this item: {current_activations_for_item.shape}")
#                 print(f"Min/Max/Mean of activations for this item: {current_activations_for_item.min().item():.4f}/{current_activations_for_item.max().item():.4f}/{current_activations_for_item.mean().item():.4f}")
#                 print('-------------------------')

#         if not current_batch_activations_sliced: # If no valid activations were found in this batch
#             print(f"Warning: No valid activations to cache for batch starting at index {i}. Skipping batch.")
#             continue

#         # Stack all valid activations for this batch
#         activations_to_save = torch.stack(current_batch_activations_sliced, dim=0)
        
#         # Calculate how many examples from this batch to add to memmap
#         num_to_add = min(activations_to_save.shape[0], shard_size - total_n_cached)

#         if num_to_add <= 0:
#             print("No more space in shard for this batch. Stopping.")
#             break

#         # Save to memmap files
#         memmap_file_acts[total_n_cached : total_n_cached + num_to_add] = activations_to_save[:num_to_add].float().numpy()
#         memmap_file_ids[total_n_cached : total_n_cached + num_to_add] = input_ids[:num_to_add].numpy() # Save full input_ids for context

#         total_n_cached += num_to_add
        
#         if total_n_cached >= shard_size:
#             print('Shard full. Files loaded.')
#             print('Total examples cached:', total_n_cached)
#             break

#         # Flush changes to disk after all batches (or shard is full)
#         memmap_file_acts.flush()
#         memmap_file_ids.flush()
#         print(f"Activation caching complete. Total examples cached: {total_n_cached}")
    #     # Filter activations based on substrings
    #     if tokens_to_cache is not None:
    #         # Get slices of activations to cache
    #         slices_to_cache = [find_string_in_tokens(substring, input_ids[j], model_base.tokenizer) for j, substring in enumerate(batch_substrings)]
    #         activations_sliced_list = []
    #         input_ids_list = []
    #         for j, s in enumerate(slices_to_cache):
    #             # TODO: this may fail if seq_len is short
    #             if s is None:
    #                 continue
    #             left_pos = s.stop-n_positions
    #             if i == 0 and j == 0:
    #                 print('tokens', model_base.tokenizer.convert_ids_to_tokens(input_ids[j]))
    #                 print('token(s) cached:', model_base.tokenizer.decode(input_ids[j,left_pos:s.stop]))
    #                 print('left_pos', left_pos)
    #             activations_sliced = activations[j, :, left_pos:s.stop, :]
    #             activations_sliced_list.append(activations_sliced)
    #             input_ids_list.append(input_ids[j])
    #         activations = torch.stack(activations_sliced_list, dim=0)
    #         input_ids = torch.stack(input_ids_list, dim=0)

    #     added_batch_size = activations.shape[0]
    #     if added_batch_size == 0:
    #         continue
    #     end_idx = min(shard_size,i+added_batch_size)
    #     if end_idx == shard_size:
    #         # if reaching the end of the shard, take the remaining examples
    #         n = end_idx - i
    #     else:
    #         # add the full batch
    #         n = added_batch_size
    #     total_n += n
    #     # TODO: avoid upcast to float?
    #     memmap_file_acts[i:end_idx] = activations[:n].float().cpu().numpy()
    #     memmap_file_ids[i:end_idx] = input_ids[:n].cpu().numpy()
    
    #     if end_idx == shard_size:
    #         print('Files loaded')
    #         print('total_n', total_n)
    #         break
    # # Flush changes to disk
    # memmap_file_acts.flush()
    # memmap_file_ids.flush()


def cache_activations(model_base: Any, prompts: List[str], compute_activation_fn: Callable,
                      tokens_to_cache: str, layers: List[int], foldername: str,
                      batch_size: int = 32, seq_len: int = 128, shard_size: int = 1000,
                      substrings: List[str] = None, # This parameter is no longer used for filtering
                      n_positions: int = 1) -> None:

    d_model = model_base.model.config.hidden_size
    if layers is None:
        n_layers = model_base.model.config.num_hidden_layers
    else:
        n_layers = len(layers)

    print('shape shard size', (shard_size, n_layers, n_positions, d_model))
    
    # Create memmap files
    memmap_file_acts = np.memmap(f"{foldername}/acts.dat", dtype='float32', mode='w+', 
                            shape=(shard_size, n_layers, n_positions, d_model))
    memmap_file_ids = np.memmap(f"{foldername}/ids.dat", dtype='float32', mode='w+', 
                            shape=(shard_size, seq_len))

    total_n_cached = 0 # Track how many samples are successfully cached

    for i in tqdm(range(0, len(prompts), batch_size), desc="Computing activations"):
        if total_n_cached >= shard_size: # Stop if shard is full
            print("Reached shard size limit. Stopping.")
            break

        # Prepare the batch of prompts by removing BOS token.
        # This is the actual list of strings that will be tokenized and fed to the model.
        # This list of prompts will be passed directly to compute_activation_fn.
        batch_prompts_for_model = [prompt.replace('<bos>', '') for prompt in prompts[i:i+batch_size]]

        
        # Get input ids and activations for ALL prompts in the batch_prompts_for_model list.
        # compute_activation_fn will handle tokenization, truncation, and padding to seq_len.
        input_ids, activations = compute_activation_fn(prompts=batch_prompts_for_model)
        
        # Ensure activations and input_ids are on CPU
        # activations = activations.cpu()
        # input_ids = input_ids.cpu()

        current_batch_activations_sliced = []
        
        # Iterate through each item that compute_activation_fn successfully processed
        for j in range(activations.shape[0]): 
            
            # Determine the actual sequence length (number of non-padding tokens)
            actual_len_idx = (input_ids[j] == model_base.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
            actual_len = (input_ids[j] != model_base.tokenizer.pad_token_id).sum().item()

            if actual_len == 0:
                # This warning means the prompt tokenized to an empty sequence.
                # This should be very rare now, only for truly empty or invalid input strings.
                print(f"Warning: Skipping example {i+j} due to empty tokenized sequence (actual_len is 0).")
                continue

            # Calculate the start position for slicing the last n_positions tokens
            # start_pos = max(0, actual_len - n_positions)
            start_pos = seq_len - n_positions


            # Perform the slicing
            # current_activations_for_item = activations[j, :, start_pos:actual_len, :]
            current_activations_for_item = activations[j, :, start_pos:seq_len, :]


            # Pad if the actual slice is shorter than n_positions
            if current_activations_for_item.shape[1] < n_positions:
                num_missing_positions = n_positions - current_activations_for_item.shape[1]
                padded_activations = torch.zeros(
                    n_layers, n_positions, d_model,
                    dtype=current_activations_for_item.dtype,
                    device=current_activations_for_item.device
                )
                padded_activations[:, num_missing_positions:, :] = current_activations_for_item
                current_activations_for_item = padded_activations
            
            current_batch_activations_sliced.append(current_activations_for_item)
            
            # Debug prints for the first cached example
            if i == 0 and j == 0:
                print('--- DEBUG: Caching Slicing Details ---')
                print('Original Prompt (cleaned):', batch_prompts_for_model[j])
                print('Full Token IDs (first 20):', input_ids[j][:20].tolist(), '...')
                print('Decoded Full Input (truncated to seq_len):', model_base.tokenizer.decode(input_ids[j], skip_special_tokens=False))
                print('Actual sequence length (non-padded):', actual_len)
                # --- DEFINITIVE FIX FOR DEBUG PRINTS ---
                print(f'Slicing from index {start_pos} to {seq_len} (exclusive). Taking {n_positions} tokens.')
                print('Token(s) that will be cached:', model_base.tokenizer.decode(input_ids[j, start_pos:seq_len], skip_special_tokens=False))
                # --- END FIX ---
                print(f"Shape of activations for this item: {current_activations_for_item.shape}")
                print(f"Min/Max/Mean of activations for this item: {current_activations_for_item.min().item():.4f}/{current_activations_for_item.max().item():.4f}/{current_activations_for_item.mean().item():.4f}")
                print('------------------------------------')

        if not current_batch_activations_sliced:
            # This should be very rare now, only if compute_activation_fn returns empty for all prompts in a batch
            print(f"Warning: No valid activations sliced for batch starting at index {i}. Skipping batch.")
            continue

        # Stack all valid activations for this batch
        activations_to_save = torch.stack(current_batch_activations_sliced, dim=0)
        
        # Calculate how many examples from this batch to add to memmap
        num_to_add = min(activations_to_save.shape[0], shard_size - total_n_cached)

        if num_to_add <= 0:
            print("No more space in shard for this batch. Stopping.")
            break

        # Save to memmap files
        memmap_file_acts[total_n_cached : total_n_cached + num_to_add] = activations_to_save[:num_to_add].float().numpy()
        memmap_file_ids[total_n_cached : total_n_cached + num_to_add] = input_ids[:num_to_add].numpy() 

        total_n_cached += num_to_add
        
        if total_n_cached >= shard_size:
            print('Shard full. Files loaded.')
            break

    memmap_file_acts.flush()
    memmap_file_ids.flush()
    print(f"Activation caching complete. Total examples cached: {total_n_cached}")

# %%
def load_pile_data(tokens_to_cache: str):
    from datasets import load_dataset

    assert tokens_to_cache == 'random'
    prompts = load_dataset("NeelNanda/pile-10k")['train']['text']
    substrings = []
    for prompt in prompts:
        substrings.append(random.choices(prompt.split(), k=1)[0])

    return prompts, substrings

def load_triviaqa_queries_to_cache(model_alias: str):
    from dataset.load_data import load_triviaqa_queries, balance_data
    
    queries = load_triviaqa_queries(model_alias=model_alias, strict_filtering=False)
    print(f"Initial number of loaded queries is {len(queries)}")

    for query in queries:
        if query['label'] != 'correct':
            query['label'] = 'incorrect'

    if balance_data:
        queries = balance_data(queries, labels=['correct', 'incorrect'], shuffle=True)
    prompts = [query["prompt"]+query['greedy_completion'] for query in queries]
    prompts = list(set(prompts)) # deduplicate just in case

    print(f"Number of deduplicated and balanced queries is {len(prompts)}")

    return prompts

def load_triviaqa_instructions_to_cache(model_alias: str, tokens_to_cache: str, balanced_data: bool = False):
    from dataset.load_data import load_triviaqa_queries, balance_data
    
    queries = load_triviaqa_queries(model_alias=model_alias, split=None)
    print(f"Initial number of loaded queries is {len(queries)}")

    for query in queries:
        if query['label'] != 'correct':
            query['label'] = 'incorrect'
    if balanced_data:
        queries = balance_data(queries, labels=['correct', 'incorrect'], shuffle=True)
    prompts = [query["prompt"] for query in queries]
    prompts = list(set(prompts)) # deduplicate just in case

    print(f"Number of deduplicated and balanced queries is {len(prompts)}")
    if tokens_to_cache == "last_eoi":
        substrings = ["<end_of_turn>\n<start_of_turn>model\n" for query in queries]
    else:
        substrings = [tokens_to_cache for query in queries]

    return prompts, substrings

def load_wikidata_instructions_to_cache(model_alias: str, tokens_to_cache: str, entity_type: str = None, balance_data: bool = True, entity_type_and_entity_name_format: bool = False):
    from dataset.load_data import load_wikidata_queries, balance_data
    
    queries = load_wikidata_queries(model_alias=model_alias, strict_filtering=False, entity_type_and_entity_name_format=entity_type_and_entity_name_format)
    if entity_type is not None:
        queries = [query for query in queries if query['entity_type'] == entity_type]

    print(f"Initial number of loaded queries is {len(queries)}")

    for query in queries:
        if query['label'] != 'correct':
            query['label'] = 'incorrect'

    if balance_data == True:
        queries = balance_data(queries, labels=['correct', 'incorrect'], shuffle=True)
    
    if tokens_to_cache == "entity":
        substrings = [query['entity'] for query in queries]
    elif tokens_to_cache == "last_eoi":
        substrings = ["<end_of_turn>\n<start_of_turn>model\n" for query in queries]
    else:
        substrings = [tokens_to_cache for query in queries]

    prompts = [query["prompt"] for query in queries]

    print(f"Number of deduplicated and balanced queries is {len(prompts)}")

    return prompts, substrings

def load_bias_instructions_to_cache(
    # model_alias: str, to be updated with load_data func
    tokens_to_cache: str,
    category: str,
    # balance_data: bool = True,
):
    """
    Load bias-related prompts for caching activations.
    Filter out undetermined cases and convert bias_cat to binary labels.
    """
    
    assert tokens_to_cache == 'bias'

    # data_path = f"z_my_data/test_prompt_final/test_{category}_completion_sentiment_judged_final.jsonl" # TO UPDATE AND MOVE TO laod_data.py

    # prompts = []
    # substrings = []

    # try:
    #     with open(data_path, "r") as f:
    #         for line in f:
    #             data = json.loads(line)
    #             # Only include if bias_cat is not undetermined
    #             if data["bias_cat"] != "undetermined":
    #                 prompts.append(data["context"])
    #                 substrings.append(data["name"])   

    #     print(
    #         f"Loaded {len(prompts)} prompts for category {category} (excluding undetermined cases)"
    #     )
    # except FileNotFoundError:
    #     print(f"Warning: No data file found for category {category}, skipping...")
    #     return [], []
    from dataset.load_data import load_bias_queries

    all_bias_data = load_bias_queries()

    if category not in all_bias_data:
        print(f"Warning: No data found for category {category}, skipping...")
        return [], []

    queries = all_bias_data[category]
    prompts = [q["completion"] for q in queries if q["bias_cat"] != "undetermined"]
    substrings = [q["completion"][-1] for q in queries if q["bias_cat"] != "undetermined"]

    print(f"Loaded {len(prompts)} prompts for category {category} (excluding undetermined cases)")
    
    return prompts, substrings

class CachedDataset(TorchDataset):
    """Torch Dataset backed by a memory-mapped numpy array."""
    def __init__(
        self,
        path_name: str,
        layers: list,
        d_model: int,
        seq_len: int,
        n_positions: int = 1,
        shard_size: int | None = None,
        num_examples: int | None = None,
        dtype = 'float32',
    ):

        acts_data_path = f"{path_name}/acts.dat"
        input_ids_path = f"{path_name}/ids.dat"
        acts = np.memmap(acts_data_path, dtype=dtype, mode="r", shape=(shard_size, len(layers), n_positions, d_model))
        input_ids = np.memmap(input_ids_path, dtype=dtype, mode="r", shape=(shard_size, seq_len))

        if num_examples is not None:
            acts = acts[:num_examples]
            input_ids = input_ids[:num_examples]
        self.mmap = np.array(acts)
        self.input_ids = np.array(input_ids)

    def __len__(self):
        return len(self.mmap)

    def __getitem__(self, idx):
        return (torch.from_numpy(np.asarray(self.mmap[idx].copy())), torch.from_numpy(np.asarray(self.input_ids[idx].copy())).int())
    
    def shard(self, num_shards: int, shard_id: int) -> "CachedDataset":
        mmap = CachedDataset.__new__(CachedDataset)

        # Split the mmap array into `num_shards` and return the `shard_id`-th shard
        shards = np.array_split(self.mmap, num_shards)
        mmap.mmap = shards[shard_id]
        return mmap

def main(args):
    
    torch.set_grad_enabled(False)

    model_alias = args.model_alias
    batch_size = args.batch_size
    tokens_to_cache = args.tokens_to_cache # "entity" / "last_eoi" / "?"
    entity_type_and_entity_name_format = args.entity_type_and_entity_name_format
    dataset = args.dataset
    model_path = model_alias_to_model_name[model_alias]
    
    layers = None # None means all layers
    n_positions = 1 # starting from the tokens_to_cache, how many positions to take to the left
    #layers_str = "_".join([str(l) for l in layers])

    model_base = construct_model_base(model_path)
    shard_size = None

      # --- START: RECOMMENDED FIX ---
    # Set a dedicated pad token to avoid ambiguity with the EOS token.
    if model_base.tokenizer.pad_token is None:
        print("Tokenizer does not have a pad token. Adding a new one.")
        model_base.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model_base.model.resize_token_embeddings(len(model_base.tokenizer))
        if hasattr(model_base.model.config, 'pad_token_id'):
            model_base.model.config.pad_token_id = model_base.tokenizer.pad_token_id
    # --- END: RECOMMENDED FIX ---


    # model_base.tokenizer.pad_token = model_base.tokenizer.eos_token

    def get_activations(model_base, prompts, substrings, seq_len, batch_size, tokens_to_cache, n_positions, dataset_name, shard_size):
        print('len prompts', len(prompts))
        print('prompt:', prompts[0])
        print('substring:', substrings[0])

        compute_activation_fn = get_compute_activation_fn(model_base, seq_len=seq_len, batch_size=batch_size)

        # Update shard size to be the minimum of the number of prompts and the shard size
        if shard_size is None:
            shard_size = len(prompts)
        else:
            shard_size = min(shard_size, len(prompts))

        print('shard_size', shard_size)

        foldername = f"{DEFAULT_CACHE_DIR}/{tokens_to_cache}/{model_alias}_{dataset_name}/{tokens_to_cache}_npositions_{n_positions}_shard_size_{shard_size}"
        os.makedirs(foldername, exist_ok=True)

        cache_activations(model_base, prompts, compute_activation_fn, tokens_to_cache, layers, foldername, batch_size, seq_len, shard_size, substrings, n_positions)



    if dataset == "wikidata":
        seq_len = 64
        for entity_type in ['player', 'movie', 'city', 'song']:
            
            dataset_name = f"wikidata_{entity_type}"
            entity_type = dataset_name.split('_')[1]
            prompts, substrings = load_wikidata_instructions_to_cache(model_alias=model_alias, tokens_to_cache=tokens_to_cache,
                                                                        entity_type=entity_type, balance_data=False,
                                                                        entity_type_and_entity_name_format=entity_type_and_entity_name_format)
            get_activations(model_base, prompts, substrings, seq_len, batch_size, tokens_to_cache, n_positions, dataset_name, shard_size)
    elif dataset == "bias":
            # ALL_BIAS_TYPES = ['Race_ethnicity', 'Nationality', 'Religion', 'Gender_identity']
            ALL_BIAS_TYPES = ["Race_1", "Race_2", "Gender", "Stereoset", "Synth", "Pos_vs_Neg"]
            seq_len = 64
            for bias_type in ALL_BIAS_TYPES:
                dataset_name = f"bias_{bias_type}"
                prompts, substrings = load_bias_instructions_to_cache(tokens_to_cache,category=bias_type)
                if len(prompts) == 0:
                    print(f"Skipping category {bias_type} — no prompts found.")
                    continue
                get_activations(model_base, prompts, substrings, seq_len, batch_size, tokens_to_cache, n_positions, dataset_name, shard_size)

    elif dataset == 'triviaqa':
        seq_len = 128
        dataset_name = dataset
        prompts, substrings = load_triviaqa_instructions_to_cache(model_alias, tokens_to_cache)
        get_activations(model_base, prompts, substrings, seq_len, batch_size, tokens_to_cache, n_positions, dataset_name, shard_size)

    elif dataset == 'pile':
        seq_len = 128
        dataset_name = dataset
        prompts, substrings = load_pile_data(tokens_to_cache)
        get_activations(model_base, prompts, substrings, seq_len, batch_size, tokens_to_cache, n_positions, dataset_name, shard_size)

    else:
        raise ValueError(f"Unknown dataset name: {dataset}")

    

if __name__ == '__main__':
    '''
    Pile activations: python -m utils.activation_cache --model_alias gemma-2-2b --tokens_to_cache random --batch_size 128 --dataset pile
    Wikidata activations: python -m utils.activation_cache --model_alias gemma-2-2b --tokens_to_cache entity --batch_size 128 --entity_type_and_entity_name_format --dataset wikidata
    '''

    parser = argparse.ArgumentParser(description='Cache activations for a given model')
    parser.add_argument('--model_alias', type=str, default="gemma-2-2b", help='Alias of the model to use')
    parser.add_argument('--tokens_to_cache', type=str, default="entity", help='How to find the position to cache. Options: "entity", "last_eoi", "?", "random"')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for processing')
    parser.add_argument('--entity_type_and_entity_name_format', action='store_true', help='Whether to format the prompt as: The {entity_type} "{entity}"')
    parser.add_argument('--dataset', type=str, default="wikidata", help='Dataset to use. Options: "wikidata", "triviaqa", "pile", "bias"')
    args = parser.parse_args()
    
    main(args)

