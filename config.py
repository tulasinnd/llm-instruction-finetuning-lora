# MODEL
model_name = "gpt2"

# TRAINING
batch_size = 10
epochs = 20
learning_rate = 5e-5

# GENERATION
max_length = 50
temperature = 0.5
top_p = 0.9

# LORA
lora_r = 8
lora_alpha = 32
lora_dropout = 0.1
lora_target_modules = ["c_attn"]

# PATHS
dataset_path = "dataset/instruction_dataset.json"
output_dir = "lora-gpt2"