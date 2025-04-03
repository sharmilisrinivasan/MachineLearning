"""
Scripts here are direct replica from the source:
https://github.com/rasbt/LLM-workshop-2024/blob/main/05_weightloading/gpt_download.py
"""

import json
import os
import urllib.request

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# ================= Helpers =================

def download_file(source_url, destination):
    # Send a GET request to download the file
    with urllib.request.urlopen(source_url) as response:
        file_size = int(response.headers.get("Content-Length", 0))

        # Check if file exists and has the same size
        if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"File already exists and is up-to-date: {destination}")
                    return

        # Define the block size for reading the file
        block_size = 1024  # 1 Kilobyte

        progress_bar_description = os.path.basename(source_url)  # Extract filename from URL
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                # Open the destination file in binary write mode
                with open(destination, "wb") as file:
                    # Read the file in chunks and write to destination
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))  # Update progress bar

def load_gpt2_params_from_tf_ckpt(checkpoint_path, n_layers):
    # 1. Create params dictionary to return and Initialize with empty blocks for each layer
    params = {"blocks": [{} for _ in range(n_layers)]}

    # 2. Iterate over each variable in the checkpoint and add to params
    for variable_name, _ in tf.train.list_variables(checkpoint_path):
        # 2.1. Get the value and remove one Dim layers
        variable_value = np.squeeze(tf.train.load_variable(checkpoint_path, variable_name))  # np.squeeze removes dim with len=1

        # 2.2. Parse Variable name to get dict keys
        variable_name_parts = variable_name.split("/")[1:]  # Skip the 'model/' prefix

        # 2.3. Initialise and access the right key in the params dictionary
        target_dict = params  # Pointer to right key in params dictionary - Starts at root

        # 2.3.1. Move pointer to right list element in case of layer variables
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])  # E.g. "0" of "h0"
            target_dict = params["blocks"][layer_number]

        # 2.3.2. Go from 2nd to last but one element in the variable name and create if key does not exist; Move the pointer to new key
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # 2.4. Assign value to the key in the params dictionary
        destination_key = variable_name_parts[-1]  # Get last key name and assign value
        target_dict[destination_key] = variable_value
    
    return params

# ================= Main Download Utility Driver =================

def download_and_load_gpt2_params(model_size, destination_folder):
    # 1. Download the GPT-2 model files

    # 1.1. Source URL and filenames
    base_url = f"https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # 1.2. Destination Folder
    destination_folder = os.path.join(destination_folder, model_size)
    os.makedirs(destination_folder, exist_ok=True)

    # 1.3. Download each file
    for filename in filenames:
        source_file_url = os.path.join(base_url, model_size, filename)
        destination_file_path = os.path.join(destination_folder, filename)
        download_file(source_file_url, destination_file_path)

    # 2. Load the GPT-2 model settings
    settings = json.load(open(os.path.join(destination_folder, "hparams.json")))

    # 3. Load the GPT-2 model params
    tf_ckpt_path = tf.train.latest_checkpoint(destination_folder)
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings["n_layer"])
   
    return settings, params
