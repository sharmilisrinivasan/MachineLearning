import tiktoken
import torch

class GPT2Tokenizer():
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def tokenize_batch(self, in_batch, max_in_seq_len):
        #  in_batch is list of strings
        #  out_batch is Tensor list of Tensors of token_ids

        out_batch = []

        for in_str in in_batch:
            # For Each String,
            # 1. Encode using tokenizer
            # 2. Convert to Tensor
            # 3. Resize to max_in_seq_len: All inputs to be of same length = Either truncated or padded with 0s
            out_batch.append(torch.tensor(self.tokenizer.encode(in_str)).resize_(max_in_seq_len))

        return torch.stack(out_batch, dim=0)

    def detokenize_batch(self, in_batch):
        #  in_batch is Tensor list of Tensors of token_ids
        #  out_batch is list of strings
        out_batch = []

        for data in in_batch:
            out_batch.append(self.tokenizer.decode(data.tolist()))

        return out_batch