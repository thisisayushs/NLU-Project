# utils.py

import torch
import torch.utils.data as data
from functools import partial
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output


class PennTreeBank(data.Dataset):
    def __init__(self, corpus):
        self.sents = [sent for sent in corpus]

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        return self.sents[idx]


def get_tokenizer(model_name="openai-community/gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def collate_fn(batch, tokenizer, device):
    tokenized = tokenizer(batch, padding=True, return_tensors="pt")
    input_ids = tokenized.input_ids[:, :-1].detach().clone().to(device)
    # shifted labels kept for parity with Part 1.A; the HF loop re-derives its
    # own labels from input_ids (the model shifts internally), so this is unused
    labels = tokenized.input_ids[:, 1:].detach().clone().to(device)
    n_tokens = torch.sum(input_ids != tokenizer.pad_token_id)
    return input_ids, labels, n_tokens


def get_dataloaders(train_path, dev_path, test_path, tokenizer, device,
                    train_bs=8, eval_bs=16):
    train_dataset = PennTreeBank(read_file(train_path))
    dev_dataset = PennTreeBank(read_file(dev_path))
    test_dataset = PennTreeBank(read_file(test_path))
    cf = partial(collate_fn, tokenizer=tokenizer, device=device)
    train_loader = DataLoader(train_dataset, batch_size=train_bs, collate_fn=cf, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=eval_bs, collate_fn=cf)
    test_loader = DataLoader(test_dataset, batch_size=eval_bs, collate_fn=cf)
    return train_loader, dev_loader, test_loader