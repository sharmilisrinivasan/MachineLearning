# Actual Values of GPT-2 "Large" provided
TRIAL_CONFIG_PARAMS = {
    "context_length": 9, # Context Length supported by the model; Actual value: 1024
    "drop_rate" : 0.0,  # Keeping it zero. Can be tried with smaller prob number
    "emb_dim": 10,  # Dimension of Embedding to be created; Actual value: 1280
    "n_heads": 2,  # Number of heads in Multi-Head Attention; Actual value: 20
    "n_layers": 3,  # Number of times Transformer block is repeated; Actual value: 36
    "qvbias": False,  # Skipping bias terms to make the transformers training faster
    "vocab_size": 50257,  # Size of gpt2 tokenizer used
}
