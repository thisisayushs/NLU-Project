import os
import json
from collections import Counter

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
IGNORE_INDEX = -100   # slot loss ignores these positions (sub-token continuations, specials)


def load_data(path):
    with open(path) as f:
        return json.loads(f.read())


def create_dev_set(tmp_train_raw, portion=0.10, seed=42):
    """Stratified 10% dev split on intent (same as Part 2.A)."""
    intents = [x['intent'] for x in tmp_train_raw]
    count_y = Counter(intents)
    inputs, labels, mini_train = [], [], []
    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(tmp_train_raw[id_y]); labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    X_train, X_dev, _, _ = train_test_split(
        inputs, labels, test_size=portion, random_state=seed, shuffle=True, stratify=labels)
    X_train.extend(mini_train)
    return X_train, X_dev


class Vocab:
    """Label vocabularies from the whole corpus. No word vocab needed — the
    pretrained tokenizer handles words. No pad/cls slot id — we mask with -100."""

    def __init__(self, corpus):
        slots = sorted(set(s for ex in corpus for s in ex['slots'].split()))
        intents = sorted(set(ex['intent'] for ex in corpus))
        self.slot2id = {s: i for i, s in enumerate(slots)}
        self.intent2id = {t: i for i, t in enumerate(intents)}
        self.id2slot = {i: s for s, i in self.slot2id.items()}
        self.id2intent = {i: t for t, i in self.intent2id.items()}


class ATISDataset(data.Dataset):
    def __init__(self, dataset, vocab):
        self.words = [ex['utterance'].split() for ex in dataset]
        self.slot_names = [ex['slots'].split() for ex in dataset]
        self.slot_ids = [[vocab.slot2id[s] for s in sl] for sl in self.slot_names]
        self.intent_ids = [vocab.intent2id[ex['intent']] for ex in dataset]

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        return {
            'words': self.words[idx],
            'slot_ids': self.slot_ids[idx],
            'slot_names': self.slot_names[idx],
            'intent_id': self.intent_ids[idx],
        }


def align_labels(word_ids, word_slot_ids, ignore=IGNORE_INDEX):
    """JointBERT alignment: each ORIGINAL word's slot goes on its FIRST sub-token;
    continuation sub-tokens and special tokens ([CLS]/[SEP]/[PAD]) get `ignore`.
    `word_ids` is the HF fast-tokenizer mapping token->original-word-index (None for specials)."""
    labels, prev = [], None
    for wid in word_ids:
        if wid is None:
            labels.append(ignore)
        elif wid != prev:          # first sub-token of this word
            labels.append(word_slot_ids[wid])
        else:                      # continuation sub-token (e.g. ##aware)
            labels.append(ignore)
        prev = wid
    return labels


def make_collate(tokenizer):
    def collate(batch):
        words = [ex['words'] for ex in batch]
        enc = tokenizer(words, is_split_into_words=True, padding=True,
                        truncation=True, return_tensors='pt')
        B, L = enc['input_ids'].shape
        slot_labels = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
        for i, ex in enumerate(batch):
            word_ids = enc.word_ids(batch_index=i)
            slot_labels[i] = torch.tensor(align_labels(word_ids, ex['slot_ids']), dtype=torch.long)
        intents = torch.tensor([ex['intent_id'] for ex in batch], dtype=torch.long)
        return {
            'input_ids': enc['input_ids'].to(DEVICE),
            'attention_mask': enc['attention_mask'].to(DEVICE),
            'slot_labels': slot_labels.to(DEVICE),
            'intents': intents.to(DEVICE),
            'words': words,                                         # for conll eval
            'slot_names': [ex['slot_names'] for ex in batch],       # gt for conll eval
        }
    return collate


def prepare_data(train_path, test_path, tokenizer, train_bs=32, eval_bs=64):
    tmp_train_raw = load_data(train_path)
    test_raw = load_data(test_path)
    train_raw, dev_raw = create_dev_set(tmp_train_raw)
    corpus = train_raw + dev_raw + test_raw
    vocab = Vocab(corpus)
    collate = make_collate(tokenizer)
    train_loader = DataLoader(ATISDataset(train_raw, vocab), batch_size=train_bs, collate_fn=collate, shuffle=True)
    dev_loader = DataLoader(ATISDataset(dev_raw, vocab), batch_size=eval_bs, collate_fn=collate)
    test_loader = DataLoader(ATISDataset(test_raw, vocab), batch_size=eval_bs, collate_fn=collate)
    return vocab, train_loader, dev_loader, test_loader
