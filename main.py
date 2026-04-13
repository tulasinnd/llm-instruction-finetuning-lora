from data import get_dataloader
import config
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token # whenever I batch variable-length sequences → I must pad them, pad token is none so use eos
model = AutoModelForCausalLM.from_pretrained(config.model_name)

# apply LoRA to the model
lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    target_modules=config.lora_target_modules,  # GPT-2 attention
    lora_dropout=config.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM")

model = get_peft_model(model, lora_config) # Original weights are frozen and LoRA matrices are trainable 
model.print_trainable_parameters()
model.to(device)

# sample data for training (replace later)
with open(config.dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# training Loop
dataloader = get_dataloader(data, tokenizer, config.batch_size)
optimizer = AdamW(model.parameters(), lr=config.learning_rate)
model.train()

for epoch in range(config.epochs):
    print(f"\nEpoch {epoch+1}")

    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Step {step}, Loss: {loss.item()}")

# save LoRA Weights
model.save_pretrained(config.output_dir)

# inference
model.eval()
prompt = """### Instruction:
What is water?

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_length=config.max_length,
        do_sample=True,
        temperature=config.temperature,
        eos_token_id=tokenizer.eos_token_id
    )

print("\nGenerated Output:\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))