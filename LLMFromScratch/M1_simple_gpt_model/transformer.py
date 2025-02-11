import torch.nn as nn

from .helpers import print_shape, LayerNorm, FeedForward
from .multi_head_attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, config, print_interims = False):
        super().__init__()
        self.norm_1 = LayerNorm(config["emb_dim"], print_level=3, print_interims=print_interims)
        self.multi_head_attention = MultiHeadAttention(config, print_interims=print_interims)
        self.drop_out = nn.Dropout(config["drop_rate"])
        self.norm_2 = LayerNorm(config["emb_dim"], print_level=3, print_interims=print_interims)
        self.feed_forward = FeedForward(config, print_interims=print_interims)
        self.print_interims = print_interims  # Turn on print if need to evaluate interim steps


    def forward(self, in_data):
        #  In Shape: batch_size * in_seq_len * emb_dim

        #  -1. For formating prints
        print_level = 2
        print(f"{"  "*print_level}@@@@@@@@@@@@ Beginning Transformer Block @@@@@@@@@@@@") if self.print_interims else None

        #  0.0 Save shortcut to add later
        shortcut = in_data

        #  1. First Norm Layer
        #  Out Shape: batch_size * in_seq_len * emb_dim
        processed_data = self.norm_1(in_data)
        print_shape("First Layer Norm", processed_data.shape, print_level, self.print_interims)

        #  2. Maksed Multi Head Attention
        #  Out Shape: batch_size * in_seq_len * emb_dim
        processed_data = self.multi_head_attention(processed_data)
        print_shape("Masked Multi Head Attention", processed_data.shape, print_level, self.print_interims)

        #  3. First Dropout
        #  Out Shape: batch_size * in_seq_len * emb_dim
        processed_data = self.drop_out(processed_data)
        print_shape("First Dropout", processed_data.shape, print_level, self.print_interims)

        #  4. Add Shortcut (Original data back)
        #  Out Shape: batch_size * in_seq_len * emb_dim
        processed_data = processed_data + shortcut
        print_shape("Shortcut Addition - Original data", processed_data.shape, print_level, self.print_interims)

        #  0.1 Update shortcut to add later
        shortcut = processed_data

        #  5. Second Norm Layer
        #  Out Shape: batch_size * in_seq_len * emb_dim
        processed_data = self.norm_2(processed_data)
        print_shape("Second Layer Norm", processed_data.shape, print_level, self.print_interims)

        #  6. Feed Forward
        #  Out Shape: batch_size * in_seq_len * emb_dim
        processed_data = self.feed_forward(processed_data)
        print_shape("Feed Forward", processed_data.shape, print_level, self.print_interims)

        #  7. Second Dropout
        #  Out Shape: batch_size * in_seq_len * emb_dim
        processed_data = self.drop_out(processed_data)
        print_shape("Second Dropout", processed_data.shape, print_level, self.print_interims)

        #  8. Add Shortcut (Intermediate data back)
        #  Out Shape: batch_size * in_seq_len * emb_dim
        processed_data = processed_data + shortcut
        print_shape("Shortcut Addition - Intermediate data", processed_data.shape, print_level, self.print_interims)

        print(f"{"  "*print_level}@@@@@@@@@@@@ Completed Transformer Block @@@@@@@@@@@@") if self.print_interims else None

        return processed_data
