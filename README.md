# **Instruction Fine-Tuning of GPT-2 with LoRA**

## Introduction
This project demonstrates how a base language model like GPT-2 can be adapted into an instruction-following assistant using LoRA. It focuses on improving response quality, structure, and controllability through efficient fine-tuning.

## Goal
To transform a base GPT-2 model into an instruction-following assistant using parameter-efficient fine-tuning (LoRA).

## Motivation
Base models like GPT-2 are powerful at generating text but **struggle to follow instructions** reliably. They often:
- Repeat phrases or go in loops  
- Mix correct and incorrect information  
- Mimic text patterns instead of reasoning  
- Are sensitive to how prompts are phrased  

This project explores these limitations and improves GPT-2’s behavior using **instruction fine-tuning**.

## Architecture
```text
Instruction → Tokenizer → GPT-2 (Frozen)
                          ↓
               LoRA Adapters (Trainable)
                          ↓
                  Response Output
````

## How It Works

* A base GPT-2 model is used, which is trained for next-token prediction but does not reliably follow instructions.
* The training data is structured as **instruction–response pairs**, enabling the model to learn how to map user instructions to appropriate outputs.
* During fine-tuning, the base model weights are **frozen**, and LoRA adapters are applied to the **attention projection layers**.
* A masking strategy is used where **instruction tokens are ignored in the loss**, and only response tokens are learned.
* Only the LoRA parameters are updated, resulting in more structured and instruction-aligned outputs.

## Results

### Before Fine-Tuning (Base GPT-2)

```text
Prompt: what is sun?
GPT-2 Output: what is sun? No, I'm not sure, but I'm sure it's a large sun," said the man,
who asked not to be named...

Prompt: Once upon a time
GPT-2 Output: Once upon a time of great struggle, we did not understand...
```

### After Fine-Tuning (LoRA Model)

```text
Ask: what is water?
Model: A chemical compound of hydrogen and oxygen.

Ask: what is earth?
Model: The third planet from the Sun that supports life.

Ask: what is the Moon?
Model: A natural satellite that orbits the Earth.
```

## Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* PEFT (LoRA)
* JSON (dataset format)

## How to Run

```bash
git clone https://github.com/tulasinnd/llm-instruction-finetuning-lora.git
cd llm-instruction-finetuning-lora

pip install -r requirements.txt

# Train the model
python main.py

# Run inference
python generate.py
```

## What I Learned

* Base language models are not naturally instruction-following
* Masking plays a key role in controlling what the model learns
* LoRA enables efficient fine-tuning without updating full model weights
* Prompt structure significantly affects model behavior

