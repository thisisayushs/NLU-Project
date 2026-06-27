import os

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from transformers import AutoTokenizer

from utils import prepare_data, DEVICE
from model import JointBERT, JointGPT2
from functions import train_loop, eval_loop

MODEL_TYPE = "gpt2"        # "bert" or "gpt2"  - we run once each.
LR         = 5e-5
DROPOUT    = 0.1
TRAIN_BS   = 32
EVAL_BS    = 64
N_EPOCHS   = 20
PATIENCE   = 5
CLIP       = 1.0
RUNS       = 5             # independent runs (different seeds)
SEED_BASE  = 42            # run r uses seed SEED_BASE + r
SAVE_MODEL = True          # saves the FULL best-dev model -> bin/model_<type>.pt

MODEL_NAME = {"bert": "bert-base-uncased", "gpt2": "gpt2"}[MODEL_TYPE]
MODEL_CLS  = {"bert": JointBERT,           "gpt2": JointGPT2}[MODEL_TYPE]

TRAIN_PATH = os.path.join('dataset', 'ATIS', 'train.json')
TEST_PATH  = os.path.join('dataset', 'ATIS', 'test.json')

def build_tokenizer():
    if MODEL_TYPE == "gpt2":
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
        tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        return tok
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def train_one_run(seed, num_slots, num_intents, train_loader, dev_loader, test_loader, vocab):
    torch.manual_seed(seed)
    model = MODEL_CLS(MODEL_NAME, num_slots, num_intents, dropout=DROPOUT).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_intents = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_state = None
    patience = PATIENCE

    for epoch in range(N_EPOCHS):
        train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=CLIP)
        results_dev, intent_dev, _ = eval_loop(dev_loader, criterion_slots, criterion_intents, model, vocab)
        slot_f1 = results_dev['total']['f']
        intent_acc = intent_dev['accuracy']
        print(f"  epoch {epoch:3d} | slot F1 {slot_f1*100:6.2f} | intent acc {intent_acc*100:6.2f}")

        if slot_f1 > best_f1:
            best_f1 = slot_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = PATIENCE
        else:
            patience -= 1
        if patience <= 0:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, vocab)
    test_f1 = results_test['total']['f']
    test_acc = intent_test['accuracy']

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return test_f1, test_acc, best_f1, best_state


def main():
    print("Using device:", DEVICE, "| model:", MODEL_NAME, f"({MODEL_TYPE})")
    tokenizer = build_tokenizer()
    vocab, train_loader, dev_loader, test_loader = prepare_data(
        TRAIN_PATH, TEST_PATH, tokenizer, TRAIN_BS, EVAL_BS)
    num_slots = len(vocab.slot2id)
    num_intents = len(vocab.intent2id)
    print(f"slots={num_slots}  intents={num_intents} | runs={RUNS}  lr={LR}")

    slot_f1s, intent_accs = [], []
    best_dev = -1.0
    best_dev_state = None
    best_run_info = None

    for r in range(RUNS):
        seed = SEED_BASE + r
        print(f"\n===== RUN {r + 1}/{RUNS} (seed {seed}) =====")
        test_f1, test_acc, dev_f1, state = train_one_run(
            seed, num_slots, num_intents, train_loader, dev_loader, test_loader, vocab)
        slot_f1s.append(test_f1 * 100)
        intent_accs.append(test_acc * 100)
        print(f"RUN {r + 1} result: test slot F1 {test_f1*100:.2f} | "
              f"test intent acc {test_acc*100:.2f} | best dev F1 {dev_f1*100:.2f}")
        if dev_f1 > best_dev:
            best_dev = dev_f1
            best_dev_state = state
            best_run_info = (seed, dev_f1, test_f1, test_acc)

    slot_f1s = np.asarray(slot_f1s)
    intent_accs = np.asarray(intent_accs)
    print("\n" + "=" * 60)
    print(f"[{MODEL_TYPE}] Slot F1    : {slot_f1s.mean():.2f} +- {slot_f1s.std():.2f}   runs {[round(x, 2) for x in slot_f1s.tolist()]}")
    print(f"[{MODEL_TYPE}] Intent Acc : {intent_accs.mean():.2f} +- {intent_accs.std():.2f}   runs {[round(x, 2) for x in intent_accs.tolist()]}")

    if SAVE_MODEL and best_dev_state is not None:
        os.makedirs('bin', exist_ok=True)
        path = os.path.join('bin', f'model_{MODEL_TYPE}.pt')
        torch.save(best_dev_state, path)
        seed, dev_f1, test_f1, test_acc = best_run_info
        print(f"Saved {path} from best-dev run (seed {seed}): "
              f"dev F1 {dev_f1*100:.2f} | test slot F1 {test_f1*100:.2f} | intent acc {test_acc*100:.2f}")


if __name__ == '__main__':
    main()