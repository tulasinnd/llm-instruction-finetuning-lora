from torch.utils.data import Dataset, DataLoader

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        text = f"""### Instruction:
        {item['instruction']}

        ### Response:
        {item['response']}{self.tokenizer.eos_token}"""

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }

# Helper function
def get_dataloader(data, tokenizer, batch_size=5):
    dataset = InstructionDataset(data, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)