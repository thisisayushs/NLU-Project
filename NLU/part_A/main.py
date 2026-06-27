import os
import copy

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from utils import prepare_data, DEVICE, PAD_TOKEN
from model import GPT2
from functions import init_weights, train_loop, eval_loop

# ============================== CONFIG ====================================
# Part 2.A FINAL config (locked from the lever sweep).
LR         = 3e-4
D_MODEL    = 200
N_HEADS    = 4
NUM_LAYERS = 1
FF_DIM     = 512
DROPOUT    = 0.0

RUNS       = 5         # number of independent runs (different seeds)
SEED_BASE  = 42        # run r uses seed SEED_BASE + r
N_EPOCHS   = 200
PATIENCE   = 5
CLIP       = 5.0
SAVE_MODEL = True      # saves the checkpoint of the BEST-DEV run to bin/model.pt
# =========================================================================

TRAIN_PATH = os.path.join('dataset', 'ATIS', 'train.json')
TEST_PATH  = os.path.join('dataset', 'ATIS', 'test.json')


def train_one_run(seed, vocab_len, slots_len, n_intents, train_loader, dev_loader, test_loader, lang):
    torch.manual_seed(seed)
    model = GPT2(
        vocab_len, slots_len, n_intents,
        pos_emb_size=1024,
        d_model=D_MODEL, n_heads=N_HEADS, num_layers=NUM_LAYERS,
        ff_dim=FF_DIM, dropout=DROPOUT,
    ).to(DEVICE)
    model.apply(init_weights)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_state = None
    patience = PATIENCE

    for epoch in range(N_EPOCHS):
        train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=CLIP)
        results_dev, intent_dev, _ = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
        slot_f1 = results_dev['total']['f']
        intent_acc = intent_dev['accuracy']
        print(f"  epoch {epoch:3d} | slot F1 {slot_f1*100:6.2f} | intent acc {intent_acc*100:6.2f}")

        if slot_f1 > best_f1:
            best_f1 = slot_f1
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            patience = PATIENCE
        else:
            patience -= 1
        if patience <= 0:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
    return results_test['total']['f'], intent_test['accuracy'], best_f1, best_state


def main():
    print("Using device:", DEVICE)
    lang, train_loader, dev_loader, test_loader = prepare_data(TRAIN_PATH, TEST_PATH)

    vocab_len = len(lang.word2id)
    slots_len = len(lang.id2slot)
    n_intents = len(lang.intent2id)
    print(f"vocab={vocab_len}  slots={slots_len}  intents={n_intents}")
    print(f"config: lr={LR} d_model={D_MODEL} n_heads={N_HEADS} "
          f"num_layers={NUM_LAYERS} ff_dim={FF_DIM} dropout={DROPOUT} | runs={RUNS}")

    slot_f1s, intent_accs = [], []
    best_dev = -1.0
    best_dev_state = None
    best_run_info = None

    for r in range(RUNS):
        seed = SEED_BASE + r
        print(f"\n===== RUN {r + 1}/{RUNS} (seed {seed}) =====")
        test_f1, test_acc, dev_f1, state = train_one_run(
            seed, vocab_len, slots_len, n_intents, train_loader, dev_loader, test_loader, lang)
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
    print(f"Slot F1    : {slot_f1s.mean():.2f} +- {slot_f1s.std():.2f}   runs {[round(x, 2) for x in slot_f1s.tolist()]}")
    print(f"Intent Acc : {intent_accs.mean():.2f} +- {intent_accs.std():.2f}   runs {[round(x, 2) for x in intent_accs.tolist()]}")

    if SAVE_MODEL and best_dev_state is not None:
        os.makedirs('bin', exist_ok=True)
        torch.save(best_dev_state, os.path.join('bin', 'model.pt'))
        seed, dev_f1, test_f1, test_acc = best_run_info
        print(f"Saved bin/model.pt from best-dev run (seed {seed}): "
              f"dev F1 {dev_f1*100:.2f} | test slot F1 {test_f1*100:.2f} | intent acc {test_acc*100:.2f}")


if __name__ == '__main__':
    main()