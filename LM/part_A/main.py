# main.py
import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from model import GPT2
from utils import get_tokenizer, get_dataloaders
from functions import init_weights, train_loop, eval_loop

# device: CUDA on Azure, MPS on the Mac, CPU as fallback
if torch.cuda.is_available():
    DEVICE = "cuda:0"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print("Using device:", DEVICE)

DATA = "dataset/PennTreeBank"
tokenizer = get_tokenizer()
vocab_len = len(tokenizer)
train_loader, dev_loader, test_loader = get_dataloaders(
    f"{DATA}/ptb.train.txt", f"{DATA}/ptb.valid.txt", f"{DATA}/ptb.test.txt",
    tokenizer, DEVICE)

lr = 1e-3  # This is now good for AdamW
model = GPT2(vocab_len, pos_emb_size=1024, d_model=256, n_heads=4,
             num_layers=1, ff_dim=1024).to(DEVICE)
model.apply(init_weights)
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion_train = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
criterion_eval = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

n_epochs = 100
patience = 3
best_ppl = math.inf
best_state = None
for epoch in range(n_epochs):
    train_loop(train_loader, optimizer, criterion_train, model)
    train_loss = train_loop(train_loader, optimizer, criterion_train, model)
    train_ppl = math.exp(train_loss)
    ppl_dev, _ = eval_loop(dev_loader, criterion_eval, model)
    print(f"epoch {epoch:3d} | train PPL {train_ppl:7.2f} | dev PPL {ppl_dev:7.2f}")
    if ppl_dev < best_ppl:
        best_ppl = ppl_dev
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        patience = 3
    else:
        patience -= 1
    if patience <= 0:
        break
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()

model.load_state_dict(best_state)
final_ppl, _ = eval_loop(test_loader, criterion_eval, model)
print("Test PPL:", final_ppl)