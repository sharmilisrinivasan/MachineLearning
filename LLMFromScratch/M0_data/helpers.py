import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPT2Tokenizer():
    """
    A simple wrapper around the tiktoken tokenizer for GPT-2.
    This class provides methods to tokenize and detokenize text.
    """
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def tokenize_batch(self, in_batch):
        #  in_batch is list of strings or simple string
        #  out_batch is Tensor list of Tensors of token_ids

        if isinstance(in_batch, str):
            in_batch = [in_batch]

        return torch.nn.utils.rnn.pad_sequence([torch.tensor(self.tokenizer.encode(in_str)) for in_str in in_batch],
                                               batch_first=True,
                                               padding_side='left',
                                               padding_value=self.tokenizer.eot_token)

    def detokenize_batch(self, in_batch):
        #  in_batch is Tensor list of Tensors of token_ids
        #  out_batch is list of strings
        out_batch = []

        to_strip = self.tokenizer.decode([self.tokenizer.eot_token])
        for data in in_batch:
            out_batch.append(self.tokenizer.decode(data.tolist()).strip(to_strip))

        return out_batch

class GPTDataset(Dataset):
    """
    A PyTorch Dataset class for GPT-2 tokenized text data.
    This class takes a text string and tokenizes it into input and target sequences.
    """
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
    """
    Create a DataLoader for the GPTDataset.
    Args:
        text (str): The input text to be tokenized.
        max_length (int): The maximum length of the sequences.
        stride (int): The stride for creating overlapping sequences.
        batch_size (int): The batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.
        drop_last (bool): Whether to drop the last incomplete batch.
        num_workers (int): Number of workers for data loading.
    Returns:
        DataLoader: A PyTorch DataLoader for the GPTDataset.
    """

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
