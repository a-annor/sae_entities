import os
os.environ["DISABLE_TORCH_COMPILE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch
if hasattr(torch, 'compile'):
    torch.compile = lambda model, *args, **kwargs: model

import re
import copy
import time
import random
from tqdm import tqdm
from typing import Tuple, List, Dict, Union
from argparse import ArgumentParser, Namespace
import json
from pathlib import Path

import torch
import numpy as np

from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM


def set_process_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed=seed)
    random.seed(seed)
    print("Setting process seed:", seed)


def load_model(args: Namespace) -> Tuple[torch.nn.Module, PreTrainedTokenizer]:
    if args.model_name == "llama-3.2":
        assert args.model_size in ["1b", "3b"], args.model_size
        model_name = f"meta-llama/Llama-3.2-{args.model_size.upper()}{'-Instruct' if args.use_instruct_model else ''}"
    elif args.model_name == "gemma-2":
        assert args.model_size in ["2b", "9b", "27b"], args.model_size
        model_name = f"google/gemma-2-{args.model_size.lower()}{'-it' if args.use_instruct_model else ''}"
    elif args.model_name == "gemma-3":
        assert args.model_size in ["1b", "4b", "12b", "27b"], args.model_size
        model_name = f"google/gemma-3-{args.model_size.lower()}-{'it' if args.use_instruct_model else 'pt'}"
    elif args.model_name == "qwen3":
        assert args.model_size in ["4b", "8b", "14b", "32b"], args.model_size
        model_name = f"Qwen/Qwen3-{args.model_size.upper()}{'' if args.use_instruct_model else '-Base'}"
    else:
        raise RuntimeError(f"Unsupported model: {args.model_name}")
    print("!! Loading model:", model_name)

    kwargs = dict(torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **kwargs)

    return model, tokenizer


def convert_chat_list_to_dialogues(
    msg_list: List[str], system_prompt: str = None
) -> List[Dict[str, str]]:
    chat = []
    if system_prompt is not None:
        chat.append({"role": "system", "content": system_prompt})
    for i, content in enumerate(msg_list):
        role = "user" if i % 2 == 0 else "assistant"
        chat.append({"role": role, "content": content})
    return chat


def tokenize_example(
    tokenizer: PreTrainedTokenizer,
    example: Union[str, List[str]],
    system_prompt: str = None,
    remove_batch_dim: bool = False,
    use_instruct_model: bool = False,
    thinking_mode: bool = False,
    device: torch.device = None,
) -> torch.Tensor:
    # Convert into standardized chat format
    if not isinstance(example, list):
        assert isinstance(example, str), example
        example = [example]
    if not isinstance(example[0], dict):
        example = convert_chat_list_to_dialogues(
            example, system_prompt=system_prompt
        )  # expects a list of alternating messages
    assert all(["role" in x and "content" in x for x in example]), example

    if use_instruct_model:
        input_ids = tokenizer.apply_chat_template(
            example,
            add_generation_prompt=True,
            enable_thinking=thinking_mode,
            return_tensors="pt",
        )
    else:
        example = "\n".join([x["content"] for x in example])  # linearize the sequence
        input_ids = tokenizer(example, return_tensors="pt").input_ids

    if remove_batch_dim:
        assert len(input_ids.shape) == 2, input_ids.shape
        assert input_ids.shape[0] == 1, input_ids.shape
        input_ids = input_ids[0]  # remove batch dim

    if device is not None:
        if isinstance(device, torch.device):
            input_ids = input_ids.to(device)
        else:
            assert device == "auto"
            input_ids = input_ids.to("cuda")

    return input_ids


def extract_response(tokenizer: PreTrainedTokenizer, outputs: torch.Tensor, tokenized_input: torch.Tensor,
                     skip_special_tok: bool = True) -> str:
    assert len(outputs) == 1, outputs
    assert len(tokenized_input.shape) == 2 and tokenized_input.shape[0] == 1, tokenized_input.shape
    prompt_len = tokenized_input.shape[1]
    decoded_text = [tokenizer.decode(sequence, skip_special_tokens=skip_special_tok) for sequence in outputs]
    cropped_response = [tokenizer.decode(sequence[prompt_len:], skip_special_tokens=skip_special_tok) for sequence in outputs]
    assert len(cropped_response) == 1, cropped_response
    decoded_text = decoded_text[0]
    cropped_response = cropped_response[0]
    return decoded_text, cropped_response


def remove_role_from_response(response: str) -> str:
    # Remove role indicators using regex
    pattern = r"\b(?:assistant|user|system|model)\b\s*\n"
    clean_text = re.sub(pattern, "", response, flags=re.IGNORECASE)
    clean_text = clean_text.split("/ target:")[0].strip()
    return clean_text


@torch.no_grad()
def generate_model_response(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    args: Namespace,
    prompt: Union[str, List[str]],
    system_prompt: str = None,
) -> Tuple[str, str]:
    input_ids = tokenize_example(
        tokenizer,
        prompt,
        system_prompt,
        use_instruct_model=args.use_instruct_model,
        thinking_mode=args.use_thinking_mode,
        device=args.device,
    )
    if args.use_instruct_model:
        # Use sampling for instruction models
        outputs = model.generate(
            input_ids,
            do_sample=True,
            top_p=0.9,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    else:
        # Use greedy decoding for base models
        outputs = model.generate(
            input_ids,
            do_sample=False,  # greedy decoding
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_response, cropped_response = extract_response(tokenizer, outputs, input_ids)
    return full_response, cropped_response


# def generate_completion(
#     model, tokenizer, prompt: str, max_new_tokens: int = 256
# ) -> str:
#     """Generate a completion for the given prompt."""
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(
#             inputs.input_ids,
#             do_sample=False,  # greedy decoding
#             max_new_tokens=max_new_tokens,
#             pad_token_id=tokenizer.eos_token_id,
#         )
#     completion = tokenizer.decode(
#         outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
#     )
#     return completion


def process_file(input_file: Path, output_file: Path, model, tokenizer, args):
    """Process a single JSONL file and generate completions."""
    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f]
        ambig_entries = [item for item in data if (item.get("context_condition") == "ambig") and (item.get("question_polarity") == "neg")]


    for item in tqdm(ambig_entries, desc=f"Processing {input_file.name}"):
        if "completion" not in item:
            prompt = item["context"]
            print(f"\nProcessing prompt: {prompt}")
            _, completion = generate_model_response(model, tokenizer, args, prompt)
            print(f"Generated completion: {completion}")
            item["completion"] = completion

    with open(output_file, "w") as f:
        for item in ambig_entries:
            f.write(json.dumps(item,  ensure_ascii=False) + "\n")
        print(f"Successfully wrote {len(ambig_entries)} entries to {output_file}")
        if output_file.exists():
            print(f"Verified output file exists at: {output_file}")
        else:
            print(f"Error: Output file was not created at {output_file}")



def main(args: Namespace) -> None:
    # Setup seed
    set_process_seed(args.seed)

    # Add additional args
    args.use_thinking_mode = False

    # Load the Gemma-2 2b base model
    args_gemma2_pt = copy.deepcopy(args)
    args_gemma2_pt.model_name = "gemma-2"
    args_gemma2_pt.model_size = "2b"
    args_gemma2_pt.use_instruct_model = False
    args_gemma2_pt.max_new_tokens = 64
    args_gemma2_pt.device = torch.device("cuda:0")
    model_gemma2_pt, tokenizer_gemma2_pt = load_model(args_gemma2_pt)
    model_gemma2_pt = model_gemma2_pt.to(args_gemma2_pt.device)  # move to device
    model_gemma2_pt.eval()

    # Setup paths
    data_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    print(f"Input dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    output_dir.mkdir(exist_ok=True)

    # Process all JSONL files
    input_files = list(data_dir.glob("*.jsonl"))
    print(f"Found {len(input_files)} JSONL files to process")

    if not input_files:
        print(f"No JSONL files found in {data_dir}")
        return
    
    for input_file in input_files:
        output_file = output_dir / f"{input_file.stem}_completion.jsonl"
        process_file(
            input_file,
            output_file,
            model_gemma2_pt,
            tokenizer_gemma2_pt,
            args_gemma2_pt,
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Argument parser for LLM completion generation")
    parser.add_argument("--seed", type=int, default=43, help="Seed (default: 43)")
    parser.add_argument(
        "--input-dir",
        type=str,
        # default="/home/ana42/rds/hpc-work/sae_entities/z_my_data/prompt_data",
        default="/home/ana42/rds/hpc-work/sae_entities/z_my_data/test_prompt_data",
        help="Input directory containing JSONL files (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        # default="/home/ana42/rds/hpc-work/sae_entities/z_my_data/prompt_data_completions",
        default="/home/ana42/rds/hpc-work/sae_entities/z_my_data/test_prompt_data_completions",
        help="Output directory for completion files (default: data_completions)",
    )
    args = parser.parse_args()
    main(args)
