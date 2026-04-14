from torch.utils.data import Dataset, DataLoader
import torch

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Split instruction and response
        instruction_text = f"""### Instruction:
{item['instruction']}

### Response:
"""
        response_text = f"""{item['response']}{self.tokenizer.eos_token}"""

        # Tokenize separately
        instruction_enc = self.tokenizer(
            instruction_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False
        )

        response_enc = self.tokenizer(
            response_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False
        )

        # Combine
        input_ids = instruction_enc["input_ids"] + response_enc["input_ids"]

        # Labels (ignore instruction, learn response)
        labels = [-100] * len(instruction_enc["input_ids"]) + response_enc["input_ids"]

        # Attention mask (all real tokens initially)
        attention_mask = [1] * len(input_ids)

        # Truncate
        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]
        attention_mask = attention_mask[:self.max_length]

        # Padding
        padding_length = self.max_length - len(input_ids)

        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        labels += [-100] * padding_length

        # Convert to tensors and return
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels)
        }


# Helper function
def get_dataloader(data, tokenizer, batch_size):
    dataset = InstructionDataset(data, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)