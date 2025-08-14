# Evaluating Bias Awareness in Large Language Models
This codebase is adapted from the implementation accompanying _'Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models_' by Ferrando et al. (https://github.com/javiferran/sae_entities), and extended to investigate bias awareness in Gemma-2-2B.

## Setup
Setup a virtual environment and install all requirements (this will ask for your HuggingFace token):
```bash
git clone https://github.com/a-annor/sae_entities.git
cd sae_entities
source setup.sh
```

For installing [SAE-Lens](https://github.com/jbloomAus/SAELens/tree/main):
```bash
pip install sae-lens
```

## Codebase structure
The `/dataset` folder contains the necessary code to create the dataset and run the model generations. It also includes the generations at `/dataset/processed`.

The `/mech_interp` folder contains the code to perform the analysis of the SAE latents.

## Get Activations
To cache residual stream activations on entity tokens, for instance of Gemma 2 2B run:
```bash
cd sae_entities
python -m utils.activation_cache --model_alias gemma-2-2b --tokens_to_cache bias --batch_size 128 --dataset bias
```

To ensure specificity to entity tokens, we exclude latents that activate frequently (>2%) on random
tokens sampled from the Pile dataset. So, for extracting activations of random tokens of the Pile dataset, run:
```bash
cd sae_entities
python -m utils.activation_cache --model_alias gemma-2-2b --tokens_to_cache random --batch_size 128 --dataset pile
```


## SAE Latent Analysis
In `mech_interp/feature_analysis.py` we compute the SAE latent scores for all layers as well as run metrics to find the most relevant latents.


```
