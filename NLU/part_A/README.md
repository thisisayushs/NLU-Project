# NLU Part 2.A: Joint Intent and Slot Filling, GPT-2 from Scratch

The from-scratch GPT-2 decoder from Experiment 1 is reused with two heads for joint
intent classification (sentence-level) and slot filling (token-level) on ATIS. The model
is causal, so a `cls` token is appended to the end of each utterance. The final position
is the only one that has seen the whole sentence, and it feeds the intent head. The
architecture is improved one lever at a time.

## Result (5 seeds)

| metric | mean ± std |
|---|---|
| Slot F1 (conll) | 87.40 ± 0.42 |
| Intent accuracy | 89.81 ± 1.00 |

Final config: `d_model=200, n_heads=4, num_layers=1, ff_dim=512, lr=3e-4, dropout=0.0`.
`bin/model.pt` is the single best-dev run (seed 42, test slot F1 87.03).

## Run

```bash
python -u main.py 2>&1 | tee run.log   # multi-run main: 5 seeds, reports mean ± std, saves best-dev
```

## Load the saved model

```python
model = GPT2(vocab_len, slots_len, n_intents, pos_emb_size=1024,
             d_model=200, n_heads=4, num_layers=1, ff_dim=512, dropout=0.0)
model.load_state_dict(torch.load("bin/model.pt", map_location="cpu"))
model.eval()
```

## Files

`model.py` (two-head GPT-2), `utils.py` (ATIS, Lang, dataset), `functions.py` (train and
eval, conll), `main.py` (config and multi-run), `conll.py`, `bin/model.pt`.

## Notes

The main difficulty was a collapse to a trivial solution, where the model predicts `O`
for every token and `flight` for every intent. It is easy to spot, because slot F1 drops
to 0 while intent accuracy freezes at exactly 73.69, the base rate of `flight` in the dev
set. When intent sits frozen at that value the model has collapsed. When intent is still
climbing the model is only learning slowly, which calls for a different response.

The fix was to raise the learning rate to 3e-4. The lower rates (1e-4, 3e-5, 1e-5) could
not take steps large enough to leave the all-`O` region, so slot F1 stayed at 0 and early
stopping ended the run before it recovered. At 3e-4 the model leaves that region within a
couple of epochs and then climbs steadily.

Learning-rate warmup is the usual stabilizer for from-scratch transformers, and here it
made things worse. The slow ramp let the all-`O` solution settle in before the rate was
high enough to learn real spans. Dropping warmup and using a higher constant rate
resolved it.

Intent accuracy carries more variance than slot F1 across seeds (±1.00 against ±0.42).
Early stopping selects the checkpoint on slot F1, so intent accuracy is read at whatever
epoch slot F1 happens to peak, and it picks up extra noise from that.

Which levers helped:

| lever | dev slot F1 | verdict |
|---|---|---|
| d_model 20 to 200 | 90.18 | keep (decisive) |
| n_heads to 4 | 91.91 | keep (marginal) |
| num_layers to 2 | 91.67 | revert (no dev gain) |
| ff_dim 20 to 512 | 93.96 | keep (real) |
| dropout {0.1,0.3,0.5} | ≤ 93.04 | revert (hurt) |

Width and feed-forward width were the two levers that mattered. The default `ff_dim=20`
forces every 200-dim token through a 200 → 20 → 200 projection, and widening it to 512
gave the second clear improvement. Depth and output-only dropout did not help, and were
reverted on the dev metric.

The architecture has a ceiling built in. A causal model cannot see tokens to the right,
yet slots such as `toloc.city_name` depend on words that follow them. Appending the `cls`
token recovers intent, since one position sees the whole sentence, while slots still
suffer from the left-only context. Part 2.B's bidirectional BERT addresses this directly.

The standalone save run scored 85.81 while the 5-seed mean is 87.40. That run sat at the
low end of the spread, which is why the result is reported as a mean with standard
deviation.

## Environment

Single NVIDIA Tesla V100-PCIE-16GB (Azure Lab VM). Python 3.8, PyTorch 2.2.0+cu121,
transformers 4.38.0, scikit-learn.