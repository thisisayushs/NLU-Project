# functions.py

import math
import torch


def param_stats(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total params: {total:,}")
    print(f"trainable params: {trainable:,}")
    print(f"frozen params: {total - trainable:,}")


def train_loop(data, optimizer, model, tokenizer, clip=1.0):
    model.train()
    loss_array = []
    number_of_tokens = []
    for input_ids, _, n_tokens in data:
        optimizer.zero_grad()
        # the HF model shifts labels internally, so we do NOT shift here
        labels = input_ids.clone().detach()
        labels[labels == tokenizer.pad_token_id] = -100  # -100 is ignored by the loss
        output = model(input_ids, labels=labels)
        n = n_tokens.item()
        loss_array.append(output.loss.item() * n)   # weight each loss by its token count
        number_of_tokens.append(n)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # standard fine-tuning safeguard
        optimizer.step()
    return sum(loss_array) / sum(number_of_tokens)


def eval_loop(data, model, tokenizer):
    model.eval()
    loss_array = []
    number_of_tokens = []
    with torch.no_grad():
        for input_ids, _, n_tokens in data:
            labels = input_ids.clone().detach()
            labels[labels == tokenizer.pad_token_id] = -100
            output = model(input_ids, labels=labels)
            n = n_tokens.item()
            loss_array.append(output.loss.item() * n)
            number_of_tokens.append(n)
    loss = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(loss)
    return ppl, loss