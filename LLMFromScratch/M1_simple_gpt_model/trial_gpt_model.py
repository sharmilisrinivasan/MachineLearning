import torch
import torch.nn as nn

from .helpers import print_shape, LayerNorm
from .transformer import TransformerBlock

class TrialGPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])  # Size of vocab dictionary, Size of output vector
        self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"])  # Size of context length (max supported), Size of output vector
        self.drop_emb = nn.Dropout(config["drop_rate"])
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(config) for i in range(config["n_layers"])])  # Repeat Transformer block sequentially n_layer times
        self.final_norm = LayerNorm(config["emb_dim"], print_level=2)
        self.linear_out = nn.Linear(config["emb_dim"], config["vocab_size"])

    def embedding_layers(self, batch, in_seq_len):
       #  0. For formating prints
        print_level = 2
        print(f"{"  "*print_level}======= Beginning Embedding =======")

        # 1. Token Embedding
        # Maps each token of each data in input batch to size of output vector
        # Out Shape: batch_size * in_seq_len * emb_dim
        tok_embeds = self.tok_emb(batch)
        print_shape("Token Embedding Layer", tok_embeds.shape, print_level)

        # 2. Positional Embedding
        # Maps each position of each data in input batch to size of output vector
        # Out Shape: in_seq_len * emb_dim
        pos_ids = torch.arange(in_seq_len, device=batch.device)  # Creating tensor of positional ids
        pos_embeds = self.pos_emb(pos_ids)
        print_shape("Positional Embedding Layer", pos_embeds.shape, print_level)

        # 3. Final Embedding
        # Token Embedding + Positional Embedding
        # Out Shape: batch_size * in_seq_len * emb_dim
        final_embedding = tok_embeds + pos_embeds
        print_shape("Final Embedding Layer", final_embedding.shape, print_level)

        print(f"{"  "*print_level}======= Completed Embedding =======")

        return final_embedding

    
    def forward(self, batch_tokenized_data):

        #  -1. For formating prints
        print_level = 1

        #  0. Input
        #  Shape of Incoming batch data: batch_size * in_seq_len
        batch_size, in_seq_len = batch_tokenized_data.shape

        #  1. Embedding Layers (Token and position)
        #  Out Shape: batch_size * in_seq_len * emb_dim
        processed_data = self.embedding_layers(batch_tokenized_data, in_seq_len)
        print_shape("Embedding Layers", processed_data.shape, print_level)

        #   2. Dropout
        #   Out Shape: batch_size * in_seq_len * emb_dim
        processed_data = self.drop_emb(processed_data)
        print_shape("Dropout Layer on Embedding", processed_data.shape, print_level)

        #  3. Transformer Block
        #  Out Shape: batch_size * in_seq_len * emb_dim
        processed_data = self.transformer_blocks(processed_data)
        print_shape("Sequential Transformer Blocks", processed_data.shape, print_level)
        
        #  4. Final Norm Layer
        #  Out Shape: batch_size * in_seq_len * emb_dim
        processed_data = self.final_norm(processed_data)
        print_shape("Final Norm Layer", processed_data.shape, print_level)

        #  5. Linear Output Layer
        #  Embeds every token vector of "emb_dim" length to a dimensional space of "vocab_size"
        #  Out Shape: batch_size * in_seq_len * vocab_size
        logits = self.linear_out(processed_data)
        print_shape("Final Logits", logits.shape, print_level)

        return logits