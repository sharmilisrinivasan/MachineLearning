"""
This module contains helper functions and classes for creating an E2E GPT model.
"""
import torch
import torch.nn as nn

#  =========================== 1. General Helpers ===========================
def print_shape(step_name, shape, level=0, is_print=False):
    """
    Print the shape of the tensor at each step of the model.
    Args:
        step_name (str): Name of the step in the model.
        shape (torch.Size): Shape of the tensor.
        level (int): Indentation level for printing.
        is_print (bool): Flag to control printing - If turned off then print is skipped
    """

    if not is_print:
        return
    print(f"{'  '*level}Shape of output from {step_name} is {shape}")

#  =========================== 2. Layer Normalization ===========================
class LayerNorm(nn.Module):
    """
    A PyTorch implementation of Layer Normalization.

    Layer Normalization is a technique to normalize the inputs across the features 
    of a layer, improving training stability and convergence by handling the 
    problems of Internal covariate shift. This implementation includes optional 
    debugging features to print intermediate steps.

    Attributes:
        epsilon (float): A small constant added to the variance to prevent division 
                         by zero during normalization. Default is 1e-5.
        scale (torch.nn.Parameter): A learnable parameter to scale the normalized 
                                    input. Initialized to ones with shape [out_dim].
        shift (torch.nn.Parameter): A learnable parameter to shift the normalized 
                                    input. Initialized to zeros with shape [out_dim].
        print_level (int): An integer indicating the indentation level for debug 
                           print statements. Default is 0.
        print_interims (bool): A flag to enable or disable printing of intermediate 
                               steps for debugging purposes. Default is False.

    Methods:
        forward(in_data):
            Performs the forward pass of layer normalization on the input data.

            Args:
                in_data (torch.Tensor): The input tensor to be normalized. Expected 
                                        shape is [..., out_dim], where `out_dim` is 
                                        the last dimension.

            Returns:
                torch.Tensor: The normalized tensor with the same shape as `in_data`.

            Intermediate Steps (if `print_interims` is True):
                - Computes the mean of the input tensor along the last dimension.
                - Computes the variance of the input tensor along the last dimension.
                - Normalizes the input tensor using the computed mean and variance.
                - Applies the learnable scale and shift parameters to the normalized 
                  tensor.
    """

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
    """
    A PyTorch implementation of the Gaussian Error Linear Unit (GELU) activation function.
    GELU is thought to be a smoother alternative to the ReLU activation function.
    It is defined as:
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))
    This implementation includes optional debugging features to print intermediate steps.
    Attributes:
        None
    Methods:
        forward(in_tensor):
            Applies the GELU activation function to the input tensor.
            Args:
                in_tensor (torch.Tensor): The input tensor to be activated.
            Returns:
                torch.Tensor: The activated tensor with the same shape as `in_tensor`.
    """
    def __init__(self):
        super().__init__()

    def forward(self, in_tensor):
        gelu = lambda x : (0.5 * x * (
            1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
                (x + 0.044715 * torch.pow(x, 3)))))
        return gelu(in_tensor)

class FeedForward(nn.Module):
    """
    A PyTorch implementation of a Feed Forward layer with two linear transformations
    and a GELU activation function in between. This layer is typically used in
    transformer architectures.
    Attributes:
        layers (torch.nn.Sequential): A sequential container that holds the linear 
                                      layers and activation function.
        print_interims (bool): A flag to enable or disable printing of intermediate 
                               steps for debugging purposes. Default is False.
    Methods:
        forward(in_data):
            Performs the forward pass of the feed forward layer.
            Args:
                in_data (torch.Tensor): The input tensor to be processed. Expected 
                                        shape is [batch_size, in_seq_len, emb_dim].
            Returns:
                torch.Tensor: The output tensor after applying the feed forward 
                              transformations. Shape is [batch_size, in_seq_len, emb_dim].
    """
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
