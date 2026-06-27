import numpy as np
import torch
import torch.nn as nn

from conll import evaluate
from sklearn.metrics import classification_report

IGNORE_INDEX = -100


def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=1.0):
    model.train()
    loss_array = []
    for batch in data:
        optimizer.zero_grad()
        slots, intent = model(batch['input_ids'], batch['attention_mask'])

        # slot loss over all token positions; -100 labels are ignored internally
        loss_slot = criterion_slots(slots.reshape(-1, slots.shape[-1]),
                                    batch['slot_labels'].reshape(-1))
        loss_intent = criterion_intents(intent, batch['intents'])
        loss = loss_intent + loss_slot

        loss_array.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    return loss_array


def eval_loop(data, criterion_slots, criterion_intents, model, vocab):
    model.eval()
    loss_array = []
    ref_intents, hyp_intents = [], []
    ref_slots, hyp_slots = [], []

    with torch.no_grad():
        for batch in data:
            slots, intent = model(batch['input_ids'], batch['attention_mask'])
            loss_slot = criterion_slots(slots.reshape(-1, slots.shape[-1]),
                                        batch['slot_labels'].reshape(-1))
            loss_intent = criterion_intents(intent, batch['intents'])
            loss_array.append((loss_intent + loss_slot).item())

            # intent
            hyp_intents += [vocab.id2intent[x] for x in intent.argmax(1).tolist()]
            ref_intents += [vocab.id2intent[x] for x in batch['intents'].tolist()]

            # slots: take the prediction at each word's FIRST sub-token (where
            # slot_labels != -100), in word order -> one slot per original word
            preds = slots.argmax(-1)            # (B, L)
            labels = batch['slot_labels']       # (B, L), -100 at ignored positions
            for i in range(preds.shape[0]):
                words = batch['words'][i]
                valid = (labels[i] != IGNORE_INDEX).nonzero(as_tuple=True)[0].tolist()
                n = min(len(valid), len(words))
                gt = [vocab.id2slot[labels[i, valid[k]].item()] for k in range(n)]
                pr = [vocab.id2slot[preds[i, valid[k]].item()] for k in range(n)]
                ref_slots.append([(words[k], gt[k]) for k in range(n)])
                hyp_slots.append([(words[k], pr[k]) for k in range(n)])

    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        print("Warning:", ex)
        results = {"total": {"f": 0}}

    report_intent = classification_report(ref_intents, hyp_intents,
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array
