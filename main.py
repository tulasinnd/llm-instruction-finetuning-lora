from data import get_dataloader
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load tokenizer and model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # whenever I batch variable-length sequences → I must pad them, pad token is none so use eos
model = AutoModelForCausalLM.from_pretrained(model_name)

# apply LoRA to the model
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],  # GPT-2 attention
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM")

model = get_peft_model(model, lora_config) # Original weights are frozen and LoRA matrices are trainable 
model.print_trainable_parameters()
model.to(device)

# sample data for training (replace later)
with open("dataset/instruction_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# training Loop
dataloader = get_dataloader(data, tokenizer, batch_size=6)
optimizer = AdamW(model.parameters(), lr=5e-5)
model.train()

for epoch in range(10):
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
model.save_pretrained("lora-gpt2")

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
        max_length=100,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )

print("\nGenerated Output:\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))