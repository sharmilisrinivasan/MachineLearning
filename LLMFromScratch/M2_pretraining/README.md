# M2_pretraining

This module is part of the **LLMFromScratch** project and focuses on the pretraining of simple GPT (Generative Pre-trained Transformer) model we architected from scratch in module `M1_simple_gpt_model`. Please refer to [`LLMFROMSCRATCH/M1_simple_gpt_model/README.md`](../M1_simple_gpt_model/README.md) for details.

## Contents
The following structure is organized in the order of implementation and usage of the components.

```
M2_pretraining/
├── README.md                  # Documentation for the M2 Pretraining module.
└── trial_notebooks/           # Contains Jupyter notebooks for step-wise and end-to-end testing.
    ├── intermediates          # PyTorch (PT) files and other outputs generated during the execution of trial notebooks.
    ├── 1_model_setup.ipynb    # Notebook to load the model, set up datasets, and evaluate initial losses.
    ├── 2_model_training.ipynb # Notebook for pretraining the model, analyzing losses, and testing outputs.
├── helpers.py                 # Helper functions for the M2 pretraining process like plot generator and sample output generator
├── evaluate.py                # Model evaluation methods used during training in `train.py`
├── train.py                   # Script for model pretraining, loss calculation, and output testing.
├── test.py                    # Script to test the pretraining module by loading datasets, running pretraining, and validating outputs.
```

## Prerequisites

1. Ensure virtual environment is set and all required dependencies are installed. Please refer to [`LLMFROMSCRATCH/README.md`](../README.md) for Python version and setup details.

2. Able to run module `M1_simple_gpt_model` successfully to load a `TrialGPTModel` Pytorch module.

## Usage

Script to test the Pretraining module by loading train and validation sets, pretraining the model, and validating model outputs.
After execution, the script will plot the training and validation loss values, and save the pretrained model for further use.

Use the following command from the parent directory `LLMFROMSCRATCH` to run this file:
```bash
python -m M2_pretraining.test
```

## Notes
- `Trial Notebooks`:
  - Pretraining is split into 2 steps - `1_model_setup` and `2_model_training`
  - Useful for debugging or rerunning the pretraining step by step, as they allow you to isolate and analyze specific parts of the process.
  - Helps in understanding each step in detail by providing intermediate outputs and visualizations.
- `test.py`: Provides a sample of how to use the classes in this directory to load data, pretrain the simple GPT model, and evaluate the same.
