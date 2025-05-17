# M1: Simple GPT Model

This module implements a simple GPT (Generative Pre-trained Transformer) model from scratch. It builds upon the tokenizers created in the `M0_data` module (refer to the `M0_data` directory for tokenizer implementation details) and focuses on creating a foundational transformer-based language model. This module is designed for educational purposes and research, providing a step-by-step understanding of transformer architectures.

## Contents
The following structure is organized in the order of implementation and usage of the components.

```
M1_simple_gpt_model/
├── README.md               # Overview and usage instructions for the Simple GPT Model module.
└── trial_notebooks/        # Contains Jupyter notebooks for step-wise and end-to-end testing.
    ├── step_wise/          # Notebooks for step-by-step implementation of the transformer architecture.
    └── e2e_gpt_model.ipynb # Notebook for complete end-to-end GPT model implementation and testing.
├── config.py               # Configuration file for trial model hyperparameters.
├── helpers.py              # Helper functions and classes for creating an E2E GPT model.
├── multi_head_attention.py # A PyTorch implementation of the Multi-Head Attention mechanism used in transformer architectures.
├── transformer.py          # A PyTorch implementation of a Transformer Block, which consists of a multi-head attention layer followed by a feed-forward neural network.
├── trial_gpt_model.py      # A PyTorch implementation of a simplified GPT model using other building block modules in this directory.
├── generate.py             # Utilties to generate text using the provided model and input data.
├── test.py                 # Script to test the GPT model by generating text samples and validating model outputs.
```

## Usage

1. Notebook to check End to end Architecture and test run: `trial_notebooks/e2e_gpt_model.ipynb`
    
2. Test file to generate text and test end-to-end:
    Use the following command from the parent directory `LLMFROMSCRATCH` to run this file:
    ```bash
    python -m M1_simple_gpt_model.test

    *Note*: Ensure virtual environment is set and all required dependencies are installed. Please refer to `LLMFROMSCRATCH/README.md` for Python version and setup details.

## Notes
- `Trial Notebooks`:
  - The whole transformer architecture is split into multiple steps and step-wise execution is available in `trial_notebooks/step_wise`
  - Complete end-to-end architecture with all components and test codes is available in `trial_notebooks/e2e_gpt_model.ipynb`
  - Useful for debugging or recreating the end-to-end system step by step.
  - Helps in understanding each step in detail.
- `test.py`: Provides a sample of how to use the classes in this directory to generate text using the simple GPT model we architected.
