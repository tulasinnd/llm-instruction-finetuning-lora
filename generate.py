import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load Tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Load Base Model + LoRA
# -----------------------------
base_model = AutoModelForCausalLM.from_pretrained(config.model_name)
model = PeftModel.from_pretrained(base_model, config.output_dir)

model.to(device)
model.eval()

print("\nYour LoRA Model is Ready! Ask or Type 'quit' to exit.\n")

# -----------------------------
# Interactive Loop
# -----------------------------
while True:
    user_input = input("Ask anything: ").strip()

    if user_input.lower() in ["quit", "exit", "q"]:
        print("Exiting...")
        break

    # Prompt Template
    prompt = f"""### Instruction:
{user_input}

### Response:
"""

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=config.max_length,           # control response length
            do_sample=config.sampling,
            temperature=config.temperature,             # more focused output
            top_p=config.top_p,                         # nucleus sampling
            repetition_penalty=1.2,                     # reduce repetition
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Clean Response
    response = decoded.split("### Response:")[-1].strip()
    response = response.split("\n")[0]                  # keep only first line

    print(f"Model: {response}\n")