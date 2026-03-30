# Instruction Fine-Tuning with LoRA

## Goal
To transform a base language model into an instruction-following assistant using parameter-efficient fine-tuning (LoRA).

## Motivation
Base models like GPT-2 are powerful at generating text but **struggle to follow instructions** reliably. They often:
- Repeat phrases or go in loops  
- Mix correct and incorrect information  
- Mimic text patterns instead of reasoning  
- Are sensitive to how prompts are phrased 
This project aims to study these limitations and improve GPT-2’s behavior using **instruction fine-tuning**.

### Here are a few examples showing GPT-2’s behavior on simple instructions:
```text
Prompt: what is sun?
GPT-2 Output: what is sun? No, I'm not sure, but I'm sure it's a large sun," said the man,
who asked not to be named. In an interview Friday in the Capitol, the conservative senator 
slammed the IRS's targeting

Prompt: Once upon a time
GPT-2 Output: Once upon a time of great struggle, we did not understand. We did not know
the meaning of a word, the meaning of each other's names, our own needs, our own aspirations, 
or any of the human rights that had been brought to 
```
## Plan
- Test base model behavior (done)
- Create instruction-response dataset
- Apply LoRA fine-tuning
- Compare before vs after results