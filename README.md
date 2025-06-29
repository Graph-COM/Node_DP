# Differentially Private Relational Learning with Entity-level Privacy Guarantees

<p align="center">
    <a href="https://arxiv.org/abs/2506.08347"><img src="https://img.shields.io/badge/-Paper-grey?logo=read%20the%20docs&logoColor=green" alt="Paper"></a>
    <a href="https://github.com/Graph-COM/Node_DP"><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a>
    <a href="https://github.com/Graph-COM/PvGaLM/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-red.svg"></a>
</p>

This repository provides tools for fine-tuning large language models over sensitive graph-structured data for relational learning with entity-level differential privacy. It supports privacy accounting for coupled sampling in relational learning through [node_dp_accounting module](https://github.com/Graph-COM/Node_DP/tree/master/node_dp_accounting) and prediction tasks on text-attributed graphs with privacy guarantees through [private fine-tuning module](https://github.com/Graph-COM/Node_DP/tree/master/private_finetuning).

## Repository Structure
- node_dp_accounting module
  - pos_neg_amplification.py: Main script for running the RDP accountant on positive-negative sampling with adaptive gradient clipping.
- private fine-tuning module
  - Lora_SeqLP_prv.py: Main script for private fine-tuning with LoRA and relation prediction tasks.
  - arguments.py: Configurable arguments and CLI interface.
  - dataset.py: Dataset preprocessing and loading utilities.
  - model.py: Model wrapper with support for HuggingFace transformers.
  - trainer.py: Training loop with privacy tracking and evaluation.
  - transformers_support.py: Patches to adapt HuggingFace modules for compatibility with Opacus.
  - utils.py: Miscellaneous utility functions (e.g., logging, checkpointing).

## Environment Setup
```
conda create -n pyvacy python=3.11
conda activate pyvacy
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.39.3
pip install sentencepiece
pip install peft
pip install datasets
pip install evaluate
pip install opacus
pip install wandb
pip install pandas
pip install scikit-learn
```

[Optional] Use with Jupyter Notebook

```
conda install ipykernel
ipython kernel install --user --name=pyvacy
```

## Training Commands
To launch training, run:
```
bash lp_train_llama.sh <CUDA_ID> <DATASET_NAME> <EPSILON> <NOISE_SCALE> <CLIP_NORM> <BATCH_SIZE> <MODEL_NAME>
```
-  MODEL_NAME options:
	- base -> bert-base-uncased
	- large -> bert-large-uncased
	- default -> meta-llama/Llama-2-7b-hf 

The script wraps calls to Lora_SeqLP_prv.py using parameters defined in `arguments.py`.

### Reproducing Table 1 Results
```
cd private_finetuning/script/
bash run_reproduce.sh
```

**Notes**
- This repository integrates Opacus with efficient per-loss-term gradient clipping and privacy accounting, compatible with HuggingFace’s transformer models.
- The noise scale used in training is computed based on a target privacy budget (ε), using the scripts provided in the node_dp_accounting/ directory.
- Make sure your wandb account is configured properly if logging is enabled.

## Citation
Please cite our paper if you are interested in our work on differentially private relational learning.
```
Node/Entity-level DP
@article{huang2025differentially,
  title={Differentially Private Relational Learning with Entity-level Privacy Guarantees},
  author={Huang, Yinan and Yin, Haoteng and Chien, Eli and Wei, Rongzhe and Li, Pan},
  journal={arXiv preprint arXiv:2506.08347},
  year={2025}
}

Edge/Relation-level DP
@inproceedings{
    yin2024privately,
    title={Privately Learning from Graphs with Applications in Fine-tuning Large Language Models},
    author={Haoteng Yin and Rongzhe Wei and Eli Chien and Pan Li},
    booktitle={Statistical Foundations of LLMs and Foundation Models (NeurIPS 2024 Workshop)},
    year={2024},
}
```

