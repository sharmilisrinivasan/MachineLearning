import torch
import torch.nn as nn

#  =========================== 1. General Helpers ===========================
def print_shape(step_name, shape, level=0, is_print=False):
    # If turned_off then print is skipped
    if not is_print:
        return
    print(f"{'  '*level}Shape of output from {step_name} is {shape}")

#  =========================== 2. Layer Normalization ===========================
class LayerNorm(nn.Module):
    def __init__(self, out_dim, print_level=0, print_interims=False):
        super().__init__()
        self.epsilon = 1e-5
        self.scale = nn.Parameter(torch.ones(out_dim))  # Of type torch.Size([out_dim])
        self.shift = nn.Parameter(torch.zeros(out_dim)) # Of type torch.Size([out_dim])
        self.print_level = print_level
        self.print_interims = print_interims  # Turn on print if need to evaluate interim steps

    def forward(self, in_data):
        print(f"{'  '*self.print_level}======= Beginning Layer Normalisation =======") if self.print_interims else None

        # 1. Mean
        # Out Shape: <in_dim...> * 1
        mean = in_data.mean(dim=-1, keepdim=True)  # Dim=-1 => Along last layer; keep_dim retains dimension. Else, reduces #dim by 1
        print_shape("Mean", mean.shape, self.print_level, self.print_interims)

        # 2. Variance
        # Out Shape: <in_dim...> * 1
        var = in_data.var(dim=-1, keepdim=True, unbiased=False)  # Turning off Unbiased avoids division by zero
        print_shape("Variance", var.shape, self.print_level, self.print_interims)

        # 3. Normalisation
        # Out Shape: <same as in_data>
        norm_in_data = (in_data - mean)/torch.sqrt(var + self.epsilon)
        print_shape("Normalisation", norm_in_data.shape, self.print_level, self.print_interims)

        # 4. Final Output
        # Out Shape: <same as in_data>
        final_norm = (self.scale * norm_in_data) + self.shift
        print_shape("Final Normalised output", final_norm.shape, self.print_level, self.print_interims)

        print(f"{'  '*self.print_level}======= Completed Layer Normalisation =======") if self.print_interims else None

        return final_norm

#  =========================== 3. Feed Forward Layer Helpers ===========================
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, in_tensor):
        gelu = lambda x : (0.5 * x * (
            1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
                (x + 0.044715 * torch.pow(x, 3)))))
        return gelu(in_tensor)

class FeedForward(nn.Module):
    def __init__(self, config, print_interims=False):
        super().__init__()
        emb_dim = config["emb_dim"]
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4*emb_dim),
            GELU(),
            nn.Linear(4*emb_dim, emb_dim),
            nn.Dropout(config["drop_rate"])
        )
        self.print_interims = print_interims  # Turn on print if need to evaluate interim steps

    def forward(self, in_data):
        #  In Shape: batch_size * in_seq_len * emb_dim

        #  -1. For formating prints
        print_level = 3
        print(f"{'  '*print_level}!!!!!!!!!!!!!!!! Beginning Feed Forward !!!!!!!!!!!!!!!!") if self.print_interims else None
        
        #  1. Feed Forward
        #  Linear -> GELU -> Linear
        #  Out Shape: batch_size * in_seq_len * emb_dim 
        output = self.layers(in_data)
        print_shape("Feed Forward", output.shape, print_level, self.print_interims)
        
        print(f"{'  '*print_level}!!!!!!!!!!!!!!!! Completed Feed Forward !!!!!!!!!!!!!!!!") if self.print_interims else None
        return output
