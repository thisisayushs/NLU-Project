# run_partB_v2.py -- bulletproof + self-diagnosing Part 1.B trainer.
#  * prints model dtype (fp16 would let AdamW grad^2 overflow)
#  * prints max gradient BEFORE and AFTER the clip every 50 batches
#    (if "after" is ever > 1.00, clip_grad_value_ is not working in your env)
#  * HARD-clamps adapter weights to [-5, 5] after every step, so a weight
#    can never reach 1e30 no matter what the explosion mechanism is
#  * LR warmup for early stability
# The training loop is INLINE (no functions.py) so it cannot be stale.

import os, math, torch
import torch.optim as optim
import transformers
from model import GPT2_LoRA, mark_only_lora_as_trainable
from utils import get_tokenizer, get_dataloaders

DEVICE = ("cuda:0" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
print("device:", DEVICE, "| torch:", torch.__version__,
      "| transformers:", transformers.__version__, "(want 4.38.0)")

RANK, ALPHA   = 8, 16
TARGET_LR     = 1e-4
WARMUP        = 200      # ramp LR 0 -> TARGET_LR over this many steps
GRAD_CLIP     = 1.0      # clamp grad entries to [-1, 1]
WEIGHT_CLAMP  = 5.0      # hard clamp adapter weights to [-5, 5] each step
SAVE_MODEL    = False
MODEL_NAME, DATA = "openai-community/gpt2", "dataset/PennTreeBank"

tok = get_tokenizer(MODEL_NAME)
train_loader, dev_loader, test_loader = get_dataloaders(
    f"{DATA}/ptb.train.txt", f"{DATA}/ptb.valid.txt", f"{DATA}/ptb.test.txt", tok, DEVICE)

model = GPT2_LoRA.from_pretrained(MODEL_NAME, rank=RANK, alpha=ALPHA).to(DEVICE)
mark_only_lora_as_trainable(model)
print("model dtype:", next(model.parameters()).dtype, "(want torch.float32)")
trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("trainable:", f"{trn:,}")

trainable = [p for p in model.parameters() if p.requires_grad]
opt = optim.AdamW(trainable, lr=TARGET_LR)

def maxw():
    return max(p.abs().max().item() for p in trainable)
def maxg():
    gs = [p.grad.abs().max().item() for p in trainable if p.grad is not None]
    return max(gs) if gs else 0.0

global_step = 0
def set_lr(step):
    lr = TARGET_LR * min(1.0, step / max(1, WARMUP))
    for grp in opt.param_groups:
        grp["lr"] = lr
    return lr

def train_epoch(loader, verbose=False):
    global global_step
    model.train()
    tot_loss, tot_tok = 0.0, 0
    for i, (input_ids, _, n_tokens) in enumerate(loader):
        lr_now = set_lr(global_step)
        opt.zero_grad()
        labels = input_ids.clone().detach()
        labels[labels == tok.pad_token_id] = -100
        out = model(input_ids, labels=labels)
        if not torch.isfinite(out.loss):
            if verbose and i % 50 == 0:
                print(f"  [SKIP] batch {i:5d} | max_w {maxw():.3e}", flush=True)
            global_step += 1
            continue
        n = n_tokens.item()
        tot_loss += out.loss.item() * n; tot_tok += n
        out.loss.backward()
        g_before = maxg()
        for p in trainable:
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_value_(trainable, GRAD_CLIP)
        g_after = maxg()
        opt.step()
        with torch.no_grad():                       # HARD weight clamp
            for p in trainable:
                p.clamp_(-WEIGHT_CLAMP, WEIGHT_CLAMP)
        if verbose and i % 50 == 0:
            print(f"  batch {i:5d} | lr {lr_now:.2e} | loss {out.loss.item():7.4f} "
                  f"| grad {g_before:.2e}->{g_after:.2e} | max_w {maxw():.3e}", flush=True)
        global_step += 1
    return tot_loss / tot_tok if tot_tok else float("nan")

@torch.no_grad()
def evaluate(loader):
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    for input_ids, _, n_tokens in loader:
        labels = input_ids.clone().detach()
        labels[labels == tok.pad_token_id] = -100
        out = model(input_ids, labels=labels)
        n = n_tokens.item()
        tot_loss += out.loss.item() * n; tot_tok += n
    loss = tot_loss / tot_tok
    return math.exp(loss), loss

print("=== epoch 0 (verbose) ===", flush=True)
tl = train_epoch(train_loader, verbose=True)
dppl, _ = evaluate(dev_loader)
print(f"epoch   0 | train PPL {math.exp(tl):7.2f} | dev PPL {dppl:7.2f}", flush=True)

best_ppl = dppl
best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora" in k}
patience = 3
for epoch in range(1, 100):
    tl = train_epoch(train_loader)
    dppl, _ = evaluate(dev_loader)
    print(f"epoch {epoch:3d} | train PPL {math.exp(tl):7.2f} | dev PPL {dppl:7.2f}", flush=True)
    if dppl < best_ppl:
        best_ppl = dppl
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora" in k}
        patience = 3
    else:
        patience -= 1
        if patience <= 0:
            break
    if DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()

model.load_state_dict(best_state, strict=False)
tppl, _ = evaluate(test_loader)
print("Test PPL:", tppl, flush=True)
if SAVE_MODEL:
    os.makedirs("bin", exist_ok=True)
    torch.save(model.state_dict(), "bin/model.pt")
    print(f"saved bin/model.pt | test PPL {tppl}", flush=True)