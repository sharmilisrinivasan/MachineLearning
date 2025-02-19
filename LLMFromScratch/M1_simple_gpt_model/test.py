"""
Use following command from the parent directory `LLMFROMSCRATCH` to run this file:
python -m M1_simple_gpt_model.test

Note: As the steps are to be executed sequentially, writing as one test
"""

import unittest

from M0_data.helpers import GPT2Tokenizer
from M1_simple_gpt_model.config import TRIAL_CONFIG_PARAMS
from M1_simple_gpt_model.generate import generate_text
from M1_simple_gpt_model.trial_gpt_model import TrialGPTModel

class TestTrialGPTModel(unittest.TestCase):
    def setUp(self):
        print("========= Setting Up =========")

        #  1. Initialize Tokenizer
        self.tokenizer = GPT2Tokenizer()

        #  2. Initialize Model
        self.trial_gpt_model = TrialGPTModel(TRIAL_CONFIG_PARAMS, print_interims=False)

        # 3. Sample Inputs
        txt1 = "Every effort moves you towards your goal"  # Text with token_len=in_seq_len
        txt2 = "Every day holds a"  # Text with token_len<in_seq_len
        txt3 = "This statement is going to have token length greater than input sequence length"  # Text with token_len>in_seq_len
        self.test_strings = [txt1, txt2, txt3]

        # 3.1. Inputs Parameters
        self.in_seq_len = 7
        self.batch_size = len(self.test_strings)
        self.max_tokens_to_generate = 1

    def test_trial_gpt_model(self):

        #  1. Tokenize - Convert words to token IDs
        print("========= Asserting Input Tokens shape =========")
        tokenized_in_strings = self.tokenizer.tokenize_batch(self.test_strings, self.in_seq_len)
        self.assertTupleEqual(tokenized_in_strings.shape, (self.batch_size, self.in_seq_len), "In Tokens shape not as expected")

        #  2. Generate output with the model
        print("========= Asserting Output Tokens shape =========")
        output_tokens = generate_text(self.trial_gpt_model, tokenized_in_strings, self.max_tokens_to_generate, TRIAL_CONFIG_PARAMS["context_length"], print_interims=False)
        self.assertTupleEqual(output_tokens.shape, (self.batch_size, self.in_seq_len+self.max_tokens_to_generate), "Out Tokens shape not as expected")

        #  3. Detokenize - Convert Token IDs to Words
        output = self.tokenizer.detokenize_batch(output_tokens)

        print("========= Asserting Number of outputs from model =========")
        self.assertEqual(len(output), self.batch_size, "Number of output elements not as expected")

        print("========= Asserting actual outputs from model =========")
        for idx,out_str in enumerate(output):
            print(f"    Asserting {idx+1} / {self.batch_size} output element")
            in_str = self.test_strings[idx]
            print(f"    Input: {in_str}")
            print(f"    Output: {out_str}")
            self.assertTrue(out_str.startswith(in_str[:self.in_seq_len]), "Generated string does NOT start with actual input test string")
            print("    ----------------------------------")

if __name__ == "__main__":
    unittest.main()
