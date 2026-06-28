# LM Part 1.A: GPT-2 Language Model from Scratch

A decoder-only GPT-2 implemented from scratch in PyTorch and trained as a word-level
language model on Penn Treebank. The architecture was improved one lever at a time,
starting from a deliberately weak baseline.

## Result

Test perplexity 29.49 (dev minimum 32.96 at epoch 31). This number comes from the run
that wrote `bin/model.pt`, so the saved weights match the reported figure.

Final config: `d_model=256, n_heads=8, num_layers=4, ff_dim=1024, dropout=0.3,
weight_tying=True, pos_emb_size=1024`.

## Run

```bash
# venv (Python 3.8): torch 2.2.0+cu121, transformers 4.38.0, tqdm
python -u main.py 2>&1 | tee run.log
```

Config lives as constants at the top of `main.py`. `SAVE_MODEL=True` writes
`bin/model.pt` and prints the test PPL of those weights.

## Load the saved model

```python
model = GPT2(vocab_len, pos_emb_size=1024, d_model=256, n_heads=8,
             num_layers=4, ff_dim=1024, dropout=0.3, weight_tying=True)
model.load_state_dict(torch.load("bin/model.pt", map_location="cpu"))
model.eval()
```

## Files

`model.py` (GPT-2 modules), `utils.py` (data and Penn Treebank loader), `functions.py`
(train and eval loops), `main.py` (config and run), `bin/model.pt` (saved weights).

## Notes

The architecture was changed one lever at a time. Each row below changes a single thing
from the row above it:

| lever | change | test PPL |
|---|---|---|
| 1 | learning rate to 1e-3 | 45.5 |
| 2a | width (d_model 256) | 37.49 |
| 2b | depth (4 layers) | 35.40 |
| 2c | heads (8) | 34.64 |
| 3 | dropout 0.3 | 31.51 |
| 4 | weight tying | 29.49 |

Weight tying gave the largest single improvement and was the only change that pushed the
dev minimum below about 40. Capacity and dropout act on the activations and the depth,
while roughly half of the parameters sit in the token-embedding and output matrices.
Tying merges those two matrices into one, and on a corpus of about a million tokens that
regularizes more strongly than the other levers.

Dropout helped, though its rate had little effect. Sweeping 0.1, 0.3, and 0.5 left the
dev minimum within about 0.3 PPL across all three. What changed with dropout was the
training behavior. Without it the dev perplexity bottomed at epoch 1 and the model
overfit immediately. With it the model trained for about ten epochs before early
stopping.

Two runs of the identical final config gave 29.44 and 29.49, both bottoming at epoch 31.
The difference is CUDA reduction nondeterminism. We report 29.49 because those are the
saved weights.

A practical note on Git: a trained `.pt` is large, and once it is committed GitHub
rejects the push for exceeding 100 MB. Removing it afterward means rewriting history with
`git filter-repo`. Add `bin/` to `.gitignore` before the first `git add`.

## Environment

Single NVIDIA Tesla V100-PCIE-16GB (Azure Lab VM). Python 3.8, PyTorch 2.2.0+cu121,
transformers 4.38.0.