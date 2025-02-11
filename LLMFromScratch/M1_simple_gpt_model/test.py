"""
Use following command from the parent directory `LLMFROMSCRATCH` to run this file:
python -m M1_simple_gpt_model.test
"""

from M0_data.helpers import GPT2Tokenizer
from M1_simple_gpt_model.config import TRIAL_CONFIG_PARAMS
from M1_simple_gpt_model.generate import generate_text
from M1_simple_gpt_model.trial_gpt_model import TrialGPTModel

def test_trail_gpt_model():
    #  0.1 Sample Inputs
    in_seq_len = 7
    txt1 = "Every effort moves you towards your goal"  # Text with token_len=in_seq_len
    txt2 = "Every day holds a"  # Text with token_len<in_seq_len
    txt3 = "This statement is going to have token length greater than input sequence length"  # Text with token_len>in_seq_len
    in_strings = [txt1, txt2, txt3]

    #  0.2. Initialize Tokenize
    tokenizer = GPT2Tokenizer()

    #  1. Tokenize - Convert words to token IDs
    tokenized_in_strings = tokenizer.tokenize_batch(in_strings, in_seq_len)
    assert tokenized_in_strings.shape == (3, 7), f"In Tokens shape not as expected: {tokenized_in_strings.shape}"
    print("========= Input Tokens shape Asserted =========")

    #  2. Generate output with the model
    trial_gpt_model = TrialGPTModel(TRIAL_CONFIG_PARAMS, print_interims=True)
    output_tokens = generate_text(trial_gpt_model, tokenized_in_strings, 1, TRIAL_CONFIG_PARAMS["context_length"], print_interims=True)
    assert output_tokens.shape == (3, 8), f"Out Tokens shape not as expected: {output_tokens.shape}"
    print("========= Output Tokens shape Asserted =========")

    #  3. Detokenize - Convert Token IDs to Words
    output = tokenizer.detokenize_batch(output_tokens)
    assert len(output) == 3, f"Number of output elements not as expected: {len(output)}"
    print("========= Output Asserted =========")
    print("========= Sample Output =========")
    print("\n".join(output))

if __name__ == "__main__":
    test_trail_gpt_model()
