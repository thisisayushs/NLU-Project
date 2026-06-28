# NLU Part 2.B: Joint Intent and Slot Filling, Fine-Tuned GPT-2 and BERT

Two pretrained models are fine-tuned for joint intent and slot filling on ATIS: BERT-base
(encoder, bidirectional) and GPT-2 (decoder, causal). The main engineering problem is
sub-tokenization, since a pretrained tokenizer splits a word into several word-pieces and
breaks the one-word-to-one-slot alignment.

## Result (5 seeds each)

| model | architecture | slot F1 (conll) | intent accuracy |
|---|---|---|---|
| GPT-2 fine-tuned | causal | 92.73 ± 0.26 | 96.35 ± 0.45 |
| BERT-base fine-tuned | bidirectional | 95.75 ± 0.21 | 97.54 ± 0.28 |

`bin/model_bert.pt` (best-dev seed 42, test 95.88) and `bin/model_gpt2.pt` (best-dev seed
43, test 92.79) are the saved checkpoints.

## Run

One switch, `MODEL_TYPE`, selects the model. Run it twice:

```bash
# main.py: MODEL_TYPE = "bert"
python -u main.py 2>&1 | tee run_bert.log
# main.py: MODEL_TYPE = "gpt2"
python -u main.py 2>&1 | tee run_gpt2.log
```

## Load a saved model

```python
model = JointBERT("bert-base-uncased", num_slots=129, num_intents=26, dropout=0.1)
model.load_state_dict(torch.load("bin/model_bert.pt", map_location="cpu"))
model.eval()
# GPT-2: JointGPT2("gpt2", 129, 26, dropout=0.1) and load model_gpt2.pt.
# At inference the GPT-2 tokenizer still needs add_prefix_space=True and pad_token=eos.
```

## Files

`model.py` (`JointBERT` and `JointGPT2`), `utils.py` (sub-token alignment and collate),
`functions.py` (train and eval, conll), `main.py` (`MODEL_TYPE` switch, multi-run),
`conll.py`, `bin/model_bert.pt`, `bin/model_gpt2.pt`.

## Notes

Sub-token alignment is the part that needs care, and a wrong scheme fails quietly.
Word-pieces break the one-word-to-one-slot mapping. Following the JointBERT paper, each
word's slot label goes on its first sub-token, and the continuation pieces and special
tokens are set to -100 so the loss ignores them. At evaluation, the prediction at each
first-sub-token position is read in word order to rebuild one slot per word for conll. A
misaligned scheme does not raise an error. It either caps slot F1 well below the ceiling
or produces a mismatch between the label count and the word count, which is why this part
gets verified carefully.

GPT-2 needs three adjustments that BERT does not:
1. The fast tokenizer requires `add_prefix_space=True` to accept pre-split input
   (`is_split_into_words=True`), otherwise it raises on the first batch.
2. GPT-2 has no pad token, so set `tokenizer.pad_token = tokenizer.eos_token`.
3. GPT-2 has no `[CLS]`, so intent is read from the last real token of each sequence,
   located with `attention_mask.sum(1) - 1`. This mirrors the appended-`cls` approach
   from Part 2.A.

The same alignment code serves both models. Every Hugging Face fast tokenizer exposes
`word_ids()` regardless of whether it uses WordPiece or byte-level BPE, so `align_labels`
and the eval reconstruction are identical for BERT and GPT-2. Only the intent-pooling
line differs, reading the front `[CLS]` for BERT and the last token for GPT-2.

The three-regime comparison is in the report. In short: with the causal architecture held
fixed, pretraining adds about 5.3 slot F1 (from-scratch to fine-tuned GPT-2), and with
pretraining held fixed, bidirectionality adds about 3.0 slot F1 (GPT-2 to BERT).
Bidirectionality lifts slots by about 3 points and intent by about 1. Intent is a
sentence-level decision that GPT-2's last token already sees, while slots are per-token
and a left-only token cannot anticipate a label such as `toloc.city_name`.

Each saved checkpoint is the full fine-tuned model (about 440 MB for BERT, about 500 MB
for GPT-2), self-contained for a two-line reload. The multi-run saves only the best-dev
run for each model.

## Environment

Single NVIDIA Tesla V100-PCIE-16GB (Azure Lab VM). Python 3.8, PyTorch 2.2.0+cu121,
transformers 4.38.0, scikit-learn. The `resume_download` FutureWarning from
`huggingface_hub` during downloads is harmless.