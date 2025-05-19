"""
This file contains helper functions for the M2 pretraining process like 
- Generating samples to print during training intermediate steps 
- Generating plots of training and validation losses
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch

from M1_simple_gpt_model.generate import generate_text

def generate_print_sample(model, test_string, tokenizer, max_output_tokens = 10):
    """
    Generate a sample text using the model and print it.
    Args:
        model: The model to use for text generation.
        test_string: The input string to generate text from.
        tokenizer: The tokenizer to use for tokenizing the input string.
        max_output_tokens: The maximum number of tokens to generate.
    """

    # 1. Tokenize
    tokenized_in_string = tokenizer.tokenize_batch([test_string])
    context_length = model.pos_emb.weight.shape[0]

    # 2. Model to Eval mode and generate text
    model.eval()  # Set Model to Evaluation
    with torch.no_grad():
        output_tokens = generate_text(model, tokenized_in_string, max_output_tokens, context_length, print_interims=False)
    model.train()  # Reset Model to train mode

    # 3. Detokenize and Print
    output = tokenizer.detokenize_batch(output_tokens)
    print(output[0])

def plot_losses(epochs_seen, train_losses, validation_losses, tokens_seen, path_to_save=None):
    """
    Plot the training and validation losses over epochs and tokens seen.
    Args:
        epochs_seen: List of epochs seen.
        train_losses: List of training losses.
        validation_losses: List of validation losses.
        tokens_seen: List of tokens seen.
        path_to_save: Path to save the plot. If None, the plot is displayed but not saved.
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, validation_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    if path_to_save:
        plt.savefig(path_to_save)
    plt.show()
