This project involves building GPT-2 from the ground up. It explores the model’s architecture, implements it manually using PyTorch, and covers the following key steps:

* Constructing GPT-2 from scratch to gain a deeper understanding of its components.

* Pretraining the model and manually loading pretrained weights from the original GPT-2.

* Applying LoRA-based fine-tuning using LitGPT.

* Evaluating the fine-tuned model using public datasets and leveraging LLM-as-Judge for assessment.

This work was completed while following a workshop led by Sebastian Raschka. As such, the structure and portions of the code are inspired by or directly adapted from his workshop materials.

* Sebastian Raschka's Workshop on YouTube: [Watch here](https://youtu.be/quh7z1q7-uc)

* Workshop GitHub Repository: [Visit here](https://github.com/rasbt/LLM-workshop-2024)

### Steps to run scripts in this project

* Python Version: **3.10** ( Mandatory for running `finetune_lora` from litgpt - https://github.com/Lightning-AI/litgpt/issues/1931 )
* Script to create, activate and intall requirements in conda environment

```
conda create -n "llm_from_scratch_py310" python=3.10
conda activate llm_from_scratch_py310
pip install -r requirements.txt
```
