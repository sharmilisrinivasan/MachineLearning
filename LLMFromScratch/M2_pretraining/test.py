"""
Use following command from the parent directory `LLMFROMSCRATCH` to run this file:
python -m M2_pretraining.test
"""

import io
import sys

import torch

from M0_data.helpers import GPT2Tokenizer, create_data_loader
from M1_simple_gpt_model.trial_gpt_model import TrialGPTModel
from M2_pretraining.train import train_model
from M2_pretraining.helpers import generate_print_sample

GPT_TEST_CONFIG_124M = {
    "context_length": 256, # Context Length supported by the model; Using reduced size to reduce compute resources; Actual value: 1024
    "drop_rate" : 0.1,  # Tring with smaller prob number, Can be kept zero.
    "emb_dim": 768,  # Dimension of Embedding to be created
    "n_heads": 12,  # Number of heads in Multi-Head Attention
    "n_layers": 12,  # Number of times Transformer block is repeated
    "qvbias": False,  # Skipping bias terms to make the transformers training faster
    "vocab_size": 50257,  # Size of gpt2 tokenizer used
}

def test_trial_gpt_model_training():
    #  -1. Set seed for tests consistency
    torch.manual_seed(123)  # For consistent reproducibility

    #  0. Load Dataset
    with open("M0_data/ponniyinselvan.txt", "r", encoding="utf-8") as read_file:
        raw_text = read_file.read()

    #  1. Tokenize
    tokenizer = GPT2Tokenizer()
    tokenized_raw_text = tokenizer.tokenize_batch(raw_text)
    assert tokenized_raw_text.shape == (1, 5330), f"In Tokens shape not as expected: {tokenized_raw_text.shape}"
    print("========= Input Tokens shape Asserted =========")

    #  2. Train and Validation Split
    batch_size = 2
    train_ratio = 0.90  # 90% Train set
    split_idx = int(len(raw_text) * train_ratio)

    #  2.1. Train Set
    train_set = raw_text[:split_idx]
    train_loader = create_data_loader(train_set,
                                    max_length=GPT_TEST_CONFIG_124M["context_length"],
                                    stride=GPT_TEST_CONFIG_124M["context_length"],
                                    batch_size=batch_size,
                                    shuffle=True,  # For training shuffling is ok
                                    drop_last=True,
                                    num_workers=0)
    
    train_tokens = 0
    for in_data, out_data in train_loader:
        assert in_data.shape == (2, 256), f"Batched Train In Tokens shape not as expected: {in_data.shape}"
        assert out_data.shape == (2, 256), f"Batched Train Out Tokens shape not as expected: {out_data.shape}"
        train_tokens += in_data.numel()
    assert train_tokens == 4608, f"Total Train Tokens not as expected: {train_tokens}"
    print("========= Train Tokens Asserted =========")

    # 2.2. Validation Set
    validation_set = raw_text[split_idx:]
    validation_loader = create_data_loader(validation_set,
                                        max_length=GPT_TEST_CONFIG_124M["context_length"],
                                        stride=GPT_TEST_CONFIG_124M["context_length"],
                                        batch_size=batch_size,
                                        shuffle=False,  # For validation shuffling is turned off
                                        drop_last=False,  # Need to infer on last short batch as well
                                        num_workers=0)
    validation_tokens = 0
    for in_data, out_data in validation_loader:
        assert in_data.shape == (2, 256), f"Batched Validation In Tokens shape not as expected: {in_data.shape}"
        assert out_data.shape == (2, 256), f"Batched Validation Out Tokens shape not as expected: {out_data.shape}"
        validation_tokens += in_data.numel()
    assert validation_tokens == 512, f"Total Train Tokens not as expected: {validation_tokens}"
    print("========= Validation Tokens Asserted =========")

    #  3. Training
    
    #  3.1. Load Model
    model = TrialGPTModel(GPT_TEST_CONFIG_124M, print_interims=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #  3.2. Initialie Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    #  3.3 Train
    print("========= Starting Test Training =========")
    num_epochs = 1
    train_losses, validation_losses, tokens_seen_tracker = train_model(model, device,
                                                                       num_epochs, optimizer,
                                                                       train_loader, validation_loader, tokenizer,
                                                                       "The story revolves around ", 5, 5)
    assert len(train_losses) == 2, f"Number of Train Losses not as expected: {train_losses}"
    print("========= # Train Losses Asserted =========")
    assert len(validation_losses) == 2, f"Number of Validation Losses not as expected: {validation_losses}"
    print("========= # Validation Losses Asserted =========")
    assert tokens_seen_tracker == [512, 3072], f"Tokens Processed not as expected: {tokens_seen_tracker}"
    print("========= Tokens Processed Asserted =========")

    #  4. Test Trained Model
    test_in_string = "The story revolves around "
    tokens_to_generate = 1
    capturedOutput = io.StringIO()

    sys.stdout = capturedOutput  # Redirect Stdout
    generate_print_sample(model, test_in_string, tokenizer, max_output_tokens=tokens_to_generate)
    sys.stdout = sys.__stdout__  # Reset Stdout

    print_output = capturedOutput.getvalue()
    assert print_output.startswith(test_in_string), f"Generated string does NOT start with actual input test string: {print_output}"
    assert(len(print_output) > len(test_in_string)), f"NO new tokens generated: {print_output}"
    print("========= Generation usind trained model Asserted =========")


if __name__ == "__main__":
    test_trial_gpt_model_training()
