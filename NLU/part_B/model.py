import torch
import torch.nn as nn
from transformers import AutoModel


class JointBERT(nn.Module):
    """Fine-tune a pretrained ENCODER (BERT) for joint intent + slot filling.

    BERT is bidirectional -> slots see both left AND right context. Intent comes
    from the [CLS] token, which BERT places at position 0 (the front).
    """

    def __init__(self, model_name, num_slots, num_intents, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.slot_out = nn.Linear(hidden, num_slots)
        self.intent_out = nn.Linear(hidden, num_intents)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = self.dropout(out.last_hidden_state)   # (B, L, hidden)
        slots = self.slot_out(h)                  # (B, L, num_slots)
        intent = self.intent_out(h[:, 0])         # [CLS] at the front for BERT
        return slots, intent


class JointGPT2(nn.Module):
    """Fine-tune a pretrained DECODER (GPT-2) for joint intent + slot filling.

    GPT-2 is causal -> a token only sees its left context (the same structural
    limit as Part 2.A's from-scratch model, now on pretrained weights). There is
    no [CLS] token, so intent comes from the LAST REAL token of each sequence
    (located via attention_mask), which under causal attention has seen the whole
    input. Slots are still per-token.
    """

    def __init__(self, model_name, num_slots, num_intents, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.slot_out = nn.Linear(hidden, num_slots)
        self.intent_out = nn.Linear(hidden, num_intents)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = self.dropout(out.last_hidden_state)   # (B, L, hidden)
        slots = self.slot_out(h)                  # (B, L, num_slots)

        # intent from the last NON-pad token (right padding): index = #real_tokens - 1
        last_idx = attention_mask.sum(dim=1) - 1          # (B,)
        batch_idx = torch.arange(h.size(0), device=h.device)
        cls_h = h[batch_idx, last_idx]                    # (B, hidden)
        intent = self.intent_out(cls_h)
        return slots, intent