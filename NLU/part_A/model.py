import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.h_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        B, L, d_model = x.size()

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)

        similarity = q @ k.transpose(-2, -1)
        similarity = similarity * (1.0 / torch.sqrt(torch.tensor(self.h_dim, device=x.device)))
        similarity = similarity.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(similarity, dim=-1)

        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, L, d_model)
        y = self.out_proj(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_dim, dropout)

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class GPT2(nn.Module):
    """Decoder-only transformer adapted for joint intent classification + slot filling.

    Two heads instead of an LM head:
      - slot_out:   per-token Linear(d_model -> slots_size)   (sequence labeling)
      - intent_out: Linear(d_model -> n_intents) on the CLS token's hidden state
                    (the CLS is appended to the END of the sequence, so under the
                    causal mask it is the only position that has seen the full input)

    Lever 2 ("dropout before the final output layers") is implemented as a single
    out_dropout applied after ln_f, before BOTH heads. dropout=0.0 -> faithful baseline.
    """

    def __init__(
        self,
        vocab_size,
        slots_size,
        n_intents,
        pos_emb_size=1024,
        d_model=20,
        n_heads=1,
        num_layers=1,
        ff_dim=20,
        dropout=0.0,
    ):
        super().__init__()
        self.pos_emb_size = pos_emb_size

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(pos_emb_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        # dropout before the output heads (Part 2.A, lever 2)
        self.out_dropout = nn.Dropout(dropout)

        # two task heads
        self.slot_out = nn.Linear(d_model, slots_size)
        self.intent_out = nn.Linear(d_model, n_intents)

        # causal mask: token i attends only to tokens j <= i
        mask = torch.tril(torch.ones(pos_emb_size, pos_emb_size)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, idx, seq_lens):
        B, L = idx.shape
        assert L <= self.pos_emb_size

        pos = torch.arange(L, device=idx.device)
        x = self.token_embed(idx) + self.pos_embed(pos)

        mask = self.mask[:, :, :L, :L]
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        x = self.out_dropout(x)  # lever 2: dropout before both output heads

        # slots: one prediction per token (CLS position is predicted too but ignored,
        # since its ground-truth slot is the pad id)
        slots = self.slot_out(x)

        # intent: read the CLS token (last real position) of each sequence
        tmp = []
        for i in range(x.shape[0]):
            tmp.append(x[i, seq_lens[i] - 1])
        cls_tokens = torch.stack(tmp)
        intent = self.intent_out(cls_tokens)

        return slots, intent  # (B, L, slots_size), (B, n_intents)
