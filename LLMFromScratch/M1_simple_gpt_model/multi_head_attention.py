import torch
import torch.nn as nn

from .helpers import print_shape

class MultiHeadAttention(nn.Module):
    def __init__(self, config,print_interims=False):
        super().__init__()

        #  Extract required fiedls from the config
        context_length = config["context_length"]
        self.emb_dim = config["emb_dim"]
        self.n_heads = config["n_heads"]
        qvbias = config["qvbias"]

        #  Linear Transformations with Weight Matrices
        #  Shapes for all Weight Matrices: emb_dim * emb_dim
        self.linear_query = nn.Linear(self.emb_dim, self.emb_dim, qvbias)
        self.linear_key = nn.Linear(self.emb_dim, self.emb_dim, qvbias)
        self.linear_value = nn.Linear(self.emb_dim, self.emb_dim, qvbias)

        #  Multi Head Prep
        assert self.emb_dim % self.n_heads == 0, "emb_dim must be divisible by n_heads"
        self.head_dim = self.emb_dim // self.n_heads  # Splitting emb_dim across all heads equally

        #  Mask Prep
        #  Upper Triangle with ones of context length is created; 
        #  Primary diagonal and below are marked zeros. 
        #  Note: 
        #  - diagonal=1 implies consider from one above primary diagonal; 
        #  - 0 will imply to include primary diagonal; 
        #  - -X will imply to consider X diagonals below primary diagonal
        #  Shape of Mask: context_length * context_length
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        #  Note regarding register buffer: 
        #  If you have parameters in your model, which should be saved and restored in the state_dict,
        #  but not trained by the optimizer, you should register them as buffers.
        #  Buffers won’t be returned in model.parameters(),
        # so that the optimizer won’t have a change to update them.
        self.register_buffer("mask", mask)

        # Dropout
        self.dropout = nn.Dropout(config["drop_rate"])

        # Linear Layer for out projection (optional)
        self.out_proj_linear = nn.Linear(self.emb_dim, self.emb_dim)

        # Turn on print if need to evaluate interim steps
        self.print_interims = print_interims


    def forward(self, in_data):
        #  In Shape: batch_size * in_seq_len * emb_dim

        #  -1. For formating prints
        print_level = 3
        print(f"{'  '*print_level}!!!!!!!!!!!!!!!! Beginning Multi Head Attention !!!!!!!!!!!!!!!!") if self.print_interims else None

        #  0. Extracting dimension shapes from incoming data
        batch_size, in_seq_len, _ = in_data.shape

        #  1. Linear Layers
        #  Out Shape for all 3: batch_size * in_seq_len * emb_dim
        queries = self.linear_query(in_data)
        keys= self.linear_key(in_data)
        values = self.linear_value(in_data)
        print_shape("Linear Transformation of Queries", queries.shape, print_level, self.print_interims)
        print_shape("Linear Transformation of Keys", keys.shape, print_level, self.print_interims)
        print_shape("Linear Transformation of Values", values.shape, print_level, self.print_interims)

        #  2. Split Embeddings into Multi-Heads
        #  We implicitly split the matrix by adding a `n_heads` dimension
        #  Unroll last dim: 
        #  (batch_size, in_seq_len, emb_dim) -> (batch_size, in_seq_len, n_heads, head_dim)
        #  Out Shape for all 3: batch_size * in_seq_len * n_heads * head_dim
        queries = queries.view(batch_size, in_seq_len, self.n_heads, self.head_dim)
        keys = keys.view(batch_size, in_seq_len, self.n_heads, self.head_dim)
        values = values.view(batch_size, in_seq_len, self.n_heads, self.head_dim)
        print_shape("Queries - Split to Multi Head", queries.shape, print_level, self.print_interims)
        print_shape("Keys - Split to Multi Head", keys.shape, print_level, self.print_interims)
        print_shape("Values - Split to Multi Head", values.shape, print_level, self.print_interims)

        #  3. Transpose for further computation
        #  Out Shape for all 3: batch_size * n_heads * in_seq_len * head_dim
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)
        print_shape("Queries - Transposed", queries.shape, print_level, self.print_interims)
        print_shape("Keys - Transposed", keys.shape, print_level, self.print_interims)
        print_shape("Values - Transposed", values.shape, print_level, self.print_interims)

        #  4. Scaled Self Attention with mask
        #  Out Shape: batch_size * num_heads * in_seq_len * in_seq_len

        ##  4.1. Self Attention (dot-product) at each head
        ##  Output = Q . KT
        ##  Out Shape: batch_size * num_heads * in_seq_len * in_seq_len
        keys_transpose = keys.transpose(2, 3)  # Transposing in_seq_len & head_dim => For output to be in_seq_len by in_seq_len per head
        self_attention_scores = queries @ keys_transpose
        print_shape("Self Attention", self_attention_scores.shape, print_level, self.print_interims)

        ##  4.2. Preping the mask & Masking
        ##  Masking is used in "decoder" architecture to stop model from seeing into future information during training.
        mask_bool = self.mask.bool()[:in_seq_len, :in_seq_len]  # Original mask truncated to the in_seq_len and converted to boolean
        print(f"{'  '*print_level}Mask Matrix is: {mask_bool}") if self.print_interims else None
        self_attention_scores.masked_fill_(mask_bool, -torch.inf)  # Use Mask and fill masked areas with "-inf"
        print_shape("Masked Self Attention", self_attention_scores.shape, print_level, self.print_interims)  # Shape is maintained - Only Upper triangle is masked to inf

        ##  4.3. Scaling the Self Attention
        ##  https://paperswithcode.com/method/scaled
        ##  Scaled dot-product attention is an attention mechanism where the dot products are scaled down by sqrt(dk)
        denominator = keys.shape[-1] ** 0.5  # Get number of elements per head and take square root for denominator of scaling
        scaled_self_attention_scores = self_attention_scores / denominator  # scaled value = self_attention_scores / sq_root(#dim per head)
        print_shape("Scaled Masked Self Attention", scaled_self_attention_scores.shape, print_level, self.print_interims)

        #  5. Attention Weights
        #  dropout(softmax(Scaled Attention Scores)) * values
        #  Out Shape: batch_size * num_heads * in_seq_len * head_dim
        softmax_scaled_self_attention_scores = torch.softmax(scaled_self_attention_scores, dim=-1)  # Softmax
        dropout_softmax_scaled_self_attention_scores = self.dropout(softmax_scaled_self_attention_scores)  # Dropout
        attention_weights = dropout_softmax_scaled_self_attention_scores @ values  # Dot product
        print_shape("Attention Weights", attention_weights.shape, print_level, self.print_interims)

        #  6. Reshapes
        #  Out Shape: batch_size * in_seq_len * emb_dim
        context_vector = attention_weights.transpose(1,2)  # Reverse Swap of heads and in_seq_len
        print_shape("Reverse Swap", context_vector.shape, print_level, self.print_interims)
        ## Note for context_vector.contiguous:
        ## Makes a copy of the tensor such that the order of its elements in memory is the same as if it had been created from scratch with the same data.
        context_vector = context_vector.contiguous().view(batch_size, in_seq_len, self.emb_dim)  # Merge heads To form batch_size * in_seq_len * original emb_dim
        print_shape("Final Reshapes", context_vector.shape, print_level, self.print_interims)

        #  7. Optional Linear projection
        #  Out Shape: batch_size * in_seq_len * emb_dim
        context_vector_to_return = self.out_proj_linear(context_vector)
        print_shape("Masked Multi Head Attention", context_vector_to_return.shape, print_level, self.print_interims)

        context_vector_to_return = self.dropout(context_vector_to_return)  # Dropout

        print(f"{'  '*print_level}!!!!!!!!!!!!!!!! Completed Multi Head Attention !!!!!!!!!!!!!!!!") if self.print_interims else None
        return context_vector_to_return
