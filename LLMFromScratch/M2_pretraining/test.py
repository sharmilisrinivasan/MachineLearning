"""
Use following command from the parent directory `LLMFROMSCRATCH` to run this file:
python -m M2_pretraining.test

Note: As the steps are to be executed sequentially, writing as one test
"""

import io
import sys
import unittest

import torch

from M0_data.helpers import GPT2Tokenizer, create_data_loader
from M1_simple_gpt_model.trial_gpt_model import TrialGPTModel
from M2_pretraining.train import train_model
from M2_pretraining.helpers import generate_print_sample

class TestTrain(unittest.TestCase):

    def setUp(self):
        print("========= Setting Up =========")

        #  -1. Set seed for tests consistency
        torch.manual_seed(123)  # For consistent reproducibility

        #  0. Load Dataset
        with open("M0_data/ponniyinselvan.txt", "r", encoding="utf-8") as read_file:
            self.raw_text = read_file.read()

        #  1. Model config
        self.GPT_TEST_CONFIG_124M = {
            "context_length": 256, # Context Length supported by the model; Using reduced size to reduce compute resources; Actual value: 1024
            "drop_rate" : 0.1,  # Tring with smaller prob number, Can be kept zero.
            "emb_dim": 768,  # Dimension of Embedding to be created
            "n_heads": 12,  # Number of heads in Multi-Head Attention
            "n_layers": 12,  # Number of times Transformer block is repeated
            "qvbias": False,  # Skipping bias terms to make the transformers training faster
            "vocab_size": 50257,  # Size of gpt2 tokenizer used
        }

        #  2. Train Params
        self.batch_size = 2
        self.train_ratio = 0.90  # 90% Train set
        self.num_epochs = 1

        #  3. Tokenizer
        self.tokenizer = GPT2Tokenizer()

        #  4. Load Model
        self.model = TrialGPTModel(self.GPT_TEST_CONFIG_124M, print_interims=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        #  5. Initialie Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0004, weight_decay=0.1)

        #  6. Train Test Parameters
        self.test_string = "The story revolves around "
        self.test_tokens_to_generate = 1

    def  test_train(self):

        #  1. Tokenize
        print("========= Asserting Input Tokens shape =========")
        tokenized_raw_text = self.tokenizer.tokenize_batch(self.raw_text)
        self.assertTupleEqual(tokenized_raw_text.shape, (1, 5330), "In Tokens shape not as expected")

        #  2. Train and Validation Split
        split_idx = int(len(self.raw_text) * self.train_ratio)

        #  2.1. Train Set
        print("========= Asserting Train Tokens =========")
        train_set = self.raw_text[:split_idx]
        train_loader = create_data_loader(train_set,
                                        max_length=self.GPT_TEST_CONFIG_124M["context_length"],
                                        stride=self.GPT_TEST_CONFIG_124M["context_length"],
                                        batch_size=self.batch_size,
                                        shuffle=True,  # For training shuffling is ok
                                        drop_last=True,
                                        num_workers=0)

        train_tokens = 0
        for in_data, out_data in train_loader:
            self.assertTupleEqual(in_data.shape, (self.batch_size, self.GPT_TEST_CONFIG_124M["context_length"]), "Batched Train In Tokens shape not as expected")
            self.assertTupleEqual(out_data.shape, (self.batch_size, self.GPT_TEST_CONFIG_124M["context_length"]), "Batched Train Out Tokens shape not as expected")
            train_tokens += in_data.numel()
        self.assertEqual(train_tokens, 4608, "Total Train Tokens not as expected")

        # 2.2. Validation Set
        print("========= Asserting Validation Tokens =========")
        validation_set = self.raw_text[split_idx:]
        validation_loader = create_data_loader(validation_set,
                                            max_length=self.GPT_TEST_CONFIG_124M["context_length"],
                                            stride=self.GPT_TEST_CONFIG_124M["context_length"],
                                            batch_size=self.batch_size,
                                            shuffle=False,  # For validation shuffling is turned off
                                            drop_last=False,  # Need to infer on last short batch as well
                                            num_workers=0)
        validation_tokens = 0
        for in_data, out_data in validation_loader:
            self.assertTupleEqual(in_data.shape, (self.batch_size, self.GPT_TEST_CONFIG_124M["context_length"]), "Batched Validation In Tokens shape not as expected")
            self.assertTupleEqual(out_data.shape, (self.batch_size, self.GPT_TEST_CONFIG_124M["context_length"]), "Batched Validation Out Tokens shape not as expected")
            validation_tokens += in_data.numel()
        self.assertEqual(validation_tokens, 512, "Total Validation Tokens not as expected")

        #  3. Training
        print("========= Starting Test Training =========")
        train_losses, validation_losses, tokens_seen_tracker = train_model(self.model, self.device,
                                                                        self.num_epochs, self.optimizer,
                                                                        train_loader, validation_loader, self.tokenizer,
                                                                        self.test_string, 5, 5)

        print("========= Asserting Train Losses =========")
        self.assertEqual(len(train_losses), 2, "Number of Train Losses not as expected")

        print("========= Asserting Validation Losses =========")
        self.assertEqual(len(validation_losses), 2, "Number of Validation Losses not as expected")

        print("========= Asserting Tokens Processed =========")
        self.assertListEqual(tokens_seen_tracker, [512, 3072], "Tokens Processed not as expected")

        #  4. Test Trained Model
        print("========= Asserting Generation using trained model =========")
        capturedOutput = io.StringIO()

        sys.stdout = capturedOutput  # Redirect Stdout
        generate_print_sample(self.model, self.test_string, self.tokenizer, max_output_tokens=self.test_tokens_to_generate)
        sys.stdout = sys.__stdout__  # Reset Stdout

        print_output = capturedOutput.getvalue()
        self.assertTrue(print_output.startswith(self.test_string), f"Generated string does NOT start with actual input test string: {print_output}")
        self.assertGreater(len(print_output), len(self.test_string), "NO new tokens generated")


if __name__ == "__main__":
    unittest.main()
