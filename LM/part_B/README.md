# LM Part 1.B: LoRA Fine-Tuning of Pretrained GPT-2

The pretrained GPT-2 (124M) is fine-tuned on Penn Treebank with hand-rolled LoRA
adapters and no PEFT library. The base model is frozen, and only the low-rank adapters
are trained.

## Result

Test perplexity 19.05 (dev minimum 20.94 at epoch 18), with `r=16, α=32, lr=3e-4`. About
884K parameters are trained, which is 0.35% of the 124M total. This improves on the
26.8M-parameter from-scratch model in Part 1.A (29.49) by more than 10 PPL.

## Run

```bash
python -u main.py 2>&1 | tee run.log   # first run downloads GPT-2 weights (~500 MB)
```

`SAVE_MODEL=True` writes a full, self-contained checkpoint to `bin/model.pt`.

## Load the saved model

```python
model = GPT2_LoRA.from_pretrained("openai-community/gpt2", rank=16, alpha=32)
model.load_state_dict(torch.load("bin/model.pt", map_location="cpu"))
model.eval()
```

## Files

`model.py` (LoRA-wrapped GPT-2), `utils.py`, `functions.py`, `main.py`, `bin/model.pt`.

## Notes

One bug is worth recording, because it cost real time and gives no error at construction.
Hugging Face's `from_pretrained` builds the model inside a `no_init_weights()` context
that replaces every `torch.nn.init.*` function with a no-op, since the weights are about
to be overwritten by the checkpoint. A LoRA adapter built during that construction is not
in the checkpoint, so if its `A` matrix is initialized with `nn.init.normal_` or
`kaiming_`, the call does nothing and `A` holds uninitialized memory. The loss is then
NaN on the first step. The model constructs cleanly, so the failure only appears during
training, and the cause sits inside a Hugging Face context manager that is patching
`torch.nn.init`.

The fix is to assign the tensors directly and skip `nn.init`:

```python
self.A = nn.Parameter(torch.randn(rank, in_dim) / math.sqrt(in_dim))  # avoids nn.init.*
self.B = nn.Parameter(torch.zeros(out_dim, rank))                     # zeros: ΔW starts at 0
```

Rank shows diminishing returns. Holding `α = 2·rank` fixed keeps the LoRA scaling
`α/rank` constant, so capacity is the only thing changing:

| rank / α | trainable | test PPL |
|---|---|---|
| 4 / 8 | 221K | 20.91 |
| 8 / 16 | 442K | 19.85 |
| 16 / 32 | 884K | 19.27 |

The gain per doubling falls from 1.06 to 0.58, and overfitting starts to appear at r=16.
The reported 19.05 comes from the re-run that saved the checkpoint, within
nondeterminism of 19.27.

On the checkpoint format: we save the full fine-tuned model (frozen base plus adapters,
about 500 MB). The adapters-only file would be about 2 MB and is the more common LoRA
practice. The full checkpoint reloads in two lines with `strict=True`, which is why we
chose it. This trades disk space for a simpler reload.

## Environment

Single NVIDIA Tesla V100-PCIE-16GB (Azure Lab VM). Python 3.8, PyTorch 2.2.0+cu121,
transformers 4.38.0.