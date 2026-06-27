import os
import math
import torch
import torch.optim as optim

from model import GPT2_LoRA, mark_only_lora_as_trainable
from utils import get_tokenizer, get_dataloaders
from functions import param_stats, train_loop, eval_loop

# device: CUDA on Azure, MPS on the Mac, CPU as fallback
if torch.cuda.is_available():
    DEVICE = "cuda:0"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print("Using device:", DEVICE)

# LoRA hyperparameters
RANK = 16
ALPHA = 32
lr = 3e-4

# we save a full, self-contained checkpoint to bin/model.pt at the end of this run.
# False while sweeping, set True only for the final/best config and re-run.
# that one config once to write the checkpoint.
SAVE_MODEL = True

MODEL_NAME = "openai-community/gpt2"
DATA = "dataset/PennTreeBank"

tokenizer = get_tokenizer(MODEL_NAME)
train_loader, dev_loader, test_loader = get_dataloaders(
    f"{DATA}/ptb.train.txt", f"{DATA}/ptb.valid.txt", f"{DATA}/ptb.test.txt",
    tokenizer, DEVICE)

# we load pretrained GPT-2, attach LoRA adapters, and train ONLY the adapters.
model = GPT2_LoRA.from_pretrained(MODEL_NAME, rank=RANK, alpha=ALPHA).to(DEVICE)
mark_only_lora_as_trainable(model)
param_stats(model)

optimizer = optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)

n_epochs = 100
patience = 3
best_ppl = math.inf
best_state = None
for epoch in range(n_epochs):
    train_loss = train_loop(train_loader, optimizer, model, tokenizer)
    train_ppl = math.exp(train_loss)
    ppl_dev, _ = eval_loop(dev_loader, model, tokenizer)
    print(f"epoch {epoch:3d} | train PPL {train_ppl:7.2f} | dev PPL {ppl_dev:7.2f}")
    if ppl_dev < best_ppl:
        best_ppl = ppl_dev
        # only the adapters change (the base is frozen), so we snapshot just those.
        best_state = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items() if "lora" in k}
        patience = 3
    else:
        patience -= 1
    if patience <= 0:
        break
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()

# we restore the best adapters (base is unchanged) - model is now the best model.
model.load_state_dict(best_state, strict=False)
final_ppl, _ = eval_loop(test_loader, model, tokenizer)
print("Test PPL:", final_ppl)

# we save the full, self-contained checkpoint (base + adapters) only when requested.
if SAVE_MODEL:
    os.makedirs("bin", exist_ok=True)
    torch.save(model.state_dict(), "bin/model.pt")
    n = sum(v.numel() for v in model.state_dict().values())
    print(f"Saved bin/model.pt (full model, {n:,} params) | reported test PPL: {final_ppl}")
else:
    print("(SAVE_MODEL=False -> no checkpoint written; set True for the final run)")