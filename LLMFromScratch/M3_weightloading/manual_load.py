import numpy as np
import torch

def return_value_to_assign(left_parameter, right_parameter):
    if left_parameter.shape != right_parameter.shape:
        raise ValueError(f"Shape mismatch. Left: {left_parameter.shape}, Right: {right_parameter.shape}")
    
    # If Shape matches,
    return torch.nn.Parameter(torch.tensor(right_parameter))

class TransformerBlockWeightLoading:
    def __init__(self, model):
        self.model = model

    # --------------------- Helpers ---------------------
    def assign_norm_1(self, layer_number, layer_norm_params):
        """
        Inputs: layer_norm_params: gpt2_params["blocks"][layer_number]["ln_1"]
        To update following model parameters:
        (2) norm_1
        * scale
        * shift
        """
        # 1. Assign Scale
        self.model.transformer_blocks[layer_number].norm_1.scale = return_value_to_assign(
            self.model.transformer_blocks[layer_number].norm_1.scale, layer_norm_params["g"])
        
        # 2. Assign Shift
        self.model.transformer_blocks[layer_number].norm_1.shift = return_value_to_assign(
            self.model.transformer_blocks[layer_number].norm_1.shift, layer_norm_params["b"])
        
        print(f"Updated Layer Norm layer 1 of Transformer Block {layer_number} with scale and shift from incoming params.")

    def assign_multi_head_attention(self, layer_number, layer_mha_params):
        """
        Inputs: layer_mha_params: gpt2_params["blocks"][layer_number]["attn"]
        To update following model parameters:
        (9) multi_head_attention
        * (1) mask
        * (2) linear_query
            * weight
            * bias
        * (2) linear_key
            * weight
            * bias
        * (2) linear_value
            * weight
            * bias
        * (2) out_proj_linear
            * weight
            * bias
        """
        # 1. Mask - No change


        # 2. QKV
        incoming_qkv = layer_mha_params["c_attn"]
        
        # 2.1. Weights of query, key, value
        # print(f"Incoming QKV Params shape: {incoming_qkv["w"].shape}") #  - To Check Uncomment
        # 2.1.1. Split weights into 3 parts
        query_weight, key_weight, value_weight = np.split(incoming_qkv["w"], 3, axis=-1)
        # 2.1.2. Assign QKV weights to model
        # Note: Transposition is required as per directions in https://github.com/rasbt/LLM-workshop-2024/blob/main/05_weightloading/05_part-1.ipynb
        self.model.transformer_blocks[layer_number].multi_head_attention.linear_query.weight = return_value_to_assign(
            self.model.transformer_blocks[layer_number].multi_head_attention.linear_query.weight, query_weight.T)
        self.model.transformer_blocks[layer_number].multi_head_attention.linear_key.weight = return_value_to_assign(
                self.model.transformer_blocks[layer_number].multi_head_attention.linear_key.weight, key_weight.T)
        self.model.transformer_blocks[layer_number].multi_head_attention.linear_value.weight = return_value_to_assign(
                self.model.transformer_blocks[layer_number].multi_head_attention.linear_value.weight, value_weight.T)
    
        # 2.2. Biases of query, key, value
        # 2.2.1. Split biases into 3 parts
        query_bias, key_bias, value_bias = np.split(incoming_qkv["b"], 3, axis=-1)
        # 2.2.2. Assign QKV biases to model
        self.model.transformer_blocks[layer_number].multi_head_attention.linear_query.bias = return_value_to_assign(
            self.model.transformer_blocks[layer_number].multi_head_attention.linear_query.bias, query_bias)
        self.model.transformer_blocks[layer_number].multi_head_attention.linear_key.bias = return_value_to_assign(
            self.model.transformer_blocks[layer_number].multi_head_attention.linear_key.bias, key_bias)
        self.model.transformer_blocks[layer_number].multi_head_attention.linear_value.bias = return_value_to_assign(
            self.model.transformer_blocks[layer_number].multi_head_attention.linear_value.bias, value_bias)
        

        # 3. out_proj_linear
        incoming_out_proj = layer_mha_params["c_proj"]
        
        # 3.1. Assign out_proj weights to model
        # Note: Transposition is required as per directions in https://github.com/rasbt/LLM-workshop-2024/blob/main/05_weightloading/05_part-1.ipynb
        self.model.transformer_blocks[layer_number].multi_head_attention.out_proj_linear.weight = return_value_to_assign(
            self.model.transformer_blocks[layer_number].multi_head_attention.out_proj_linear.weight, incoming_out_proj["w"].T)

        # 3.2. Assign out_proj biases to model
        self.model.transformer_blocks[layer_number].multi_head_attention.out_proj_linear.bias = return_value_to_assign(
            self.model.transformer_blocks[layer_number].multi_head_attention.out_proj_linear.bias, incoming_out_proj["b"])
        
        print(f"Updated MHA of Transformer Block {layer_number} with weights and biases from incoming params.")

    def assign_norm_2(self, layer_number, layer_norm_params):
        """
        Inputs: layer_norm_params: gpt2_params["blocks"][layer_number]["ln_2"]
        To update following model parameters:
        (2) norm_2
        * scale
        * shift
        """
        # 1. Assign Scale
        self.model.transformer_blocks[layer_number].norm_2.scale = return_value_to_assign(
            self.model.transformer_blocks[layer_number].norm_2.scale, layer_norm_params["g"])
        
        # 2. Assign Shift
        self.model.transformer_blocks[layer_number].norm_2.shift = return_value_to_assign(
            self.model.transformer_blocks[layer_number].norm_2.shift, layer_norm_params["b"])
        
        print(f"Updated Layer Norm layer 2 of Transformer Block {layer_number} with scale and shift from incoming params.")
    
    def assign_feed_forward(self, layer_number, layer_ff_params):
        """
        Inputs: layer_ff_params: gpt2_params["blocks"][layer_number]["mlp"]
        To update following model parameters:
        (4) feed_forward
        * (2) layers.0 (Linear Layer-1 - Pre-GeLU)
            * weight
            * bias
        * (2) layers.2 (Linear Layer-2 - Post-GeLU)
            * weight
            * bias
        """

        # 1. Linear Layer-1 - Pre-GeLU
        pre_gelu = layer_ff_params["c_fc"]
        
        # 1.1. Assign pre_gelu weights
        # print(f"Incoming Pre GELU shape: {pre_gelu['w'].shape}") #  - To Check Uncomment
        # Note: Transposition is required as incoming weights are in shape (emb_dim, 4*emb_dim) and our model requires (4*emb_dim, emb_dim)
        self.model.transformer_blocks[layer_number].feed_forward.layers[0].weight = return_value_to_assign(
            self.model.transformer_blocks[layer_number].feed_forward.layers[0].weight, pre_gelu["w"].T)
        
        # 1.2. Assign pre_gelu biases
        self.model.transformer_blocks[layer_number].feed_forward.layers[0].bias = return_value_to_assign(
            self.model.transformer_blocks[layer_number].feed_forward.layers[0].bias, pre_gelu["b"])


        # 2. Linear Layer-2 - Post-GeLU
        post_gelu = layer_ff_params["c_proj"]

        # 2.1. Assign post_gelu weights
        # print(f"Incoming Post GELU shape: {post_gelu['w'].shape}") #  - To Check Uncomment
        # Note: Transposition is required as incoming weights are in shape (4*emb_dim, 4*emb_dim) and our model requires (emb_dim, emb_dim)
        self.model.transformer_blocks[layer_number].feed_forward.layers[2].weight = return_value_to_assign(
            self.model.transformer_blocks[layer_number].feed_forward.layers[2].weight, post_gelu["w"].T)
        
        # 2.2. Assign post_gelu biases
        self.model.transformer_blocks[layer_number].feed_forward.layers[2].bias = return_value_to_assign(
            self.model.transformer_blocks[layer_number].feed_forward.layers[2].bias, post_gelu["b"])
        
        print(f"Updated FF of Transformer Block {layer_number} with weights and biases from incoming params.")

    
    # ------------------- Main Driver -------------------
    def assign(self, in_gpt2_block_params):
        """
        Inputs: in_gpt2_block_params: gpt2_params["blocks"]
        17 Parameters
        * (2) norm_1
        * (9) multi_head_attention
        * (2) norm_2
        * (4) feed_forward
        """
        num_layers = len(in_gpt2_block_params)

        for layer_num in range(num_layers):
            print(f"============ Transformer Block {layer_num} ============")
            # 1. Layer Norm 1
            layer_norm_1_params = in_gpt2_block_params[layer_num]["ln_1"]
            self.assign_norm_1(layer_num, layer_norm_1_params)

            # 2. Multi Head Attention
            layer_mha_params = in_gpt2_block_params[layer_num]["attn"]
            self.assign_multi_head_attention(layer_num, layer_mha_params)

            # 3. Layer Norm 2
            layer_norm_2_params = in_gpt2_block_params[layer_num]["ln_2"]
            self.assign_norm_2(layer_num, layer_norm_2_params)

            # 4. Feed Forward
            layer_ff_params = in_gpt2_block_params[layer_num]["mlp"]
            self.assign_feed_forward(layer_num, layer_ff_params)
        
        print("================================================")
        print("All Transformer layers updated with incoming parameters.")


class ManualWeightLoading:
    def __init__(self, model):
        self.model = model
        self.transformer_block_weight_loader = TransformerBlockWeightLoading(self.model)

    # --------------------- Helpers ---------------------
    def assign_final_norm(self, layer_norm_params):
        """
        Inputs: layer_norm_params: gpt2_params
        To update following model parameters:
        (2) final_norm
        * scale
        * shift
        """
        # 1. Assign Scale
        self.model.final_norm.scale = return_value_to_assign(self.model.final_norm.scale, layer_norm_params["g"])
        
        # 2. Assign Shift
        self.model.final_norm.shift = return_value_to_assign(self.model.final_norm.shift, layer_norm_params["b"])
        
        print("Updated Final Norm with scale and shift from incoming params.")

    def assign_linear_out(self, linear_out_params):
        """
        Inputs: linear_out_params: gpt2_params
        To update following model parameters:
        (2) linear_out
            * weight
            * bias
        """
        # 1. Assign Weights
        self.model.linear_out.weight = return_value_to_assign(self.model.linear_out.weight, linear_out_params["wte"])
        # 2. Assign Biases
        # Skip - No particular Assignment
       
        print("Updated Linear Out with weights and biases from incoming params.")


    # ------------------- Main Driver -------------------
    def assign(self, gpt2_params):
        """
        Inputs: gpt2_params: gpt2_params
        To update following model parameters:
        210 parameters
        * (1) tok_emb.weight
        * (1) pos_emb.weight
        * (12* 17 = 204) For 12 Transformer blocks
        * (2) final_norm
        * (2) linear_out
        """
        # 1. Assign Token Embedding Weights
        self.model.tok_emb.weight = return_value_to_assign(self.model.tok_emb.weight, gpt2_params["wte"])

        # 2. Assign Position Embedding Weights
        self.model.pos_emb.weight = return_value_to_assign(self.model.pos_emb.weight, gpt2_params["wpe"])

        print("Updated Token and Position Embedding weights with incoming parameters.")

        # 3. Assign Transformer Block Weights
        self.transformer_block_weight_loader.assign(gpt2_params["blocks"])
        
        # 4. Assign Final Norm Weights
        self.assign_final_norm(gpt2_params)

        # 5. Assign Linear Out Weights
        self.assign_linear_out(gpt2_params)

        print("All weights updated with incoming parameters.")
