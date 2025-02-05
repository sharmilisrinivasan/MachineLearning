import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPT2Tokenizer():
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def tokenize_batch(self, in_batch, max_in_seq_len=-1):
        #  in_batch is list of strings or simple string
        #  max_in_seq_len is used to resie out strings in case of multi length strings. Can be skipped for single string input.
        #  out_batch is Tensor list of Tensors of token_ids

        if isinstance(in_batch, str):
            in_batch = [in_batch]

        out_batch = []

        for in_str in in_batch:
            to_append = torch.tensor(self.tokenizer.encode(in_str))  # Encode using tokenizer and Convert to Tensor
            if max_in_seq_len != -1:
                to_append = to_append.resize_(max_in_seq_len)  # 3. Resize to max_in_seq_len: All inputs to be of same length = Either truncated or padded with 0s
            out_batch.append(to_append)

        return torch.stack(out_batch, dim=0)

    def detokenize_batch(self, in_batch):
        #  in_batch is Tensor list of Tensors of token_ids
        #  out_batch is list of strings
        out_batch = []

        for data in in_batch:
            out_batch.append(self.tokenizer.decode(data.tolist()))

        return out_batch

class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids)-max_length, stride):
            input_ids_chunk = token_ids[i: i+max_length]
            target_ids_chunk = token_ids[i+1: i+max_length+1]

            self.input_ids.append(torch.tensor(input_ids_chunk))
            self.target_ids.append(torch.tensor(target_ids_chunk))

    def __len__(self):
        return(len(self.input_ids))

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# Method to load the Dataset class
def create_data_loader(text, max_length=256, stride=128, batch_size=4, shuffle=True, drop_last = True, num_workers=0):

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDataset(text, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader
