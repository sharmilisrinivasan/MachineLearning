import torch

from .helpers import print_shape

def generate_text(model, in_data, max_tokens_to_generate, context_size, print_interims = False):
    """
    Generates text using the provided model and input data.
    This function performs the following steps:
    1. Truncates the input data to the context size supported by the model.
    2. Uses the model to predict the next token based on the input data.
    3. Appends the predicted token to the input data for the next iteration.
    4. Repeats the process for the specified number of tokens to generate.
    Args:
        model (torch.nn.Module): The model to use for text generation.
        in_data (torch.Tensor): The tokenized input data, expected shape is [batch_size, in_seq_len].
        Note: in_seq_len might be greater than context_size. If greater, it will be truncated.
        max_tokens_to_generate (int): The number of tokens to generate.
        context_size (int): The maximum context size supported by the model.
        print_interims (bool): Flag to enable or disable printing of intermediate steps.
    Returns:
        torch.Tensor: The generated text, shape is [batch_size, in_seq_len + max_tokens_to_generate].
    """

    for iter_cnt in range(max_tokens_to_generate):

        print("\n") if print_interims else None
        print(f"*********************** Generating {iter_cnt+1} word ***********************") if print_interims else None

        #  1. Truncate incoming batch to context length as model cannot process beyond this
        #  Note: Last X tokens are taken: *NOT* first X words
        #  Out Shape: batch_size * context_length
        in_data = in_data[:, -context_size:]
        print_shape("Truncated input", in_data.shape, is_print=print_interims)

        #  2. Get predictions from model
        #  Out Shape: batch_size * context_length * vocab_size
        with torch.no_grad():  # No gradient descent and back prop during inferencing
            logits = model(in_data)
        print_shape("Inferencing from model", logits.shape, is_print=print_interims)

        #  3. Consider only the last step
        #  Out Shape: batch_size * vocab_size
        logits = logits[:, -1, :]
        print_shape("Filtering for last step", logits.shape, is_print=print_interims)

        #  4. Softmax for probabilities
        #  Out Shape: batch_size * vocab_size
        probas = torch.softmax(logits, dim=-1)  # On vocab dimension (dim=-1)
        print_shape("Softmax", logits.shape, is_print=print_interims)

        #   5. Get index of highest probability
        #   Out Shape: batch_size * 1
        index_to_add = torch.argmax(probas, dim=-1, keepdim=True)
        print_shape("Pick Highest Prob Index", index_to_add.shape, is_print=print_interims)

        #  6. Append to data for next batch iteration
        #  Out Shape: batch_size * in_seq_len + 1
        in_data = torch.cat((in_data, index_to_add), dim=1)  # Add to the column of 2D => dim=1
        print_shape("Next batch Size", in_data.shape, is_print=print_interims)

    return in_data
