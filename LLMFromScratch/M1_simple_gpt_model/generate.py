import torch

from .helpers import print_shape

def generate_text(model, in_data, max_tokens_to_generate, context_size):
    #  model: Model to use
    #  in_data: Tokenized data on shape batch_size * in_seq_len (Note: in_seq_len might be > context_size)
    #  max_tokens_to_generate: Number of tokens/words to generate
    #  context_size: Max context size supported by the model

    for iter_cnt in range(max_tokens_to_generate):

        print("\n")
        print(f"*********************** Generating {iter_cnt+1} word ***********************")

        #  1. Truncate incoming batch to context length as model cannot process beyond this
        #  Note: Last X tokens are taken: *NOT* first X words
        #  Out Shape: batch_size * context_length
        in_data = in_data[:, -context_size:]
        print_shape("Truncated input", in_data.shape)

        #  2. Get predictions from model
        #  Out Shape: batch_size * context_length * vocab_size
        with torch.no_grad():  # No gradient descent and back prop during inferencing
            logits = model(in_data)
        print_shape("Inferencing from model", logits.shape)

        #  3. Consider only the last step
        #  Out Shape: batch_size * vocab_size
        logits = logits[:, -1, :]
        print_shape("Filtering for last step", logits.shape)

        #  4. Softmax for probabilities
        #  Out Shape: batch_size * vocab_size
        probas = torch.softmax(logits, dim=-1)  # On vocab dimension (dim=-1)
        print_shape("Softmax", logits.shape)

        #   5. Get index of highest probability
        #   Out Shape: batch_size * 1
        index_to_add = torch.argmax(probas, dim=-1, keepdim=True)
        print_shape("Pick Highest Prob Index", index_to_add.shape)

        #  6. Append to data for next batch iteration
        #  Out Shape: batch_size * in_seq_len + 1
        in_data = torch.cat((in_data, index_to_add), dim=1)  # Add to the column of 2D => dim=1
        print_shape("Next batch Size", in_data.shape)

    return in_data
