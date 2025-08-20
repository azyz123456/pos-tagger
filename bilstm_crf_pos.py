import math
import random
from collections import Counter
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import nltk
try:
    from nltk.corpus import treebank
except LookupError:
    nltk.download("treebank")
    from nltk.corpus import treebank


# -----------------------------
# Repro & device
# -----------------------------
SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Data utilities
# -----------------------------
PAD = "<PAD>"
UNK = "<UNK>"

TaggedSent = List[Tuple[str, str]]

def train_dev_test_split(data, seed=SEED, ratios=(0.8, 0.1, 0.1)):
    idx = list(range(len(data)))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    n = len(data)
    n_train = int(ratios[0] * n)
    n_dev   = int(ratios[1] * n)
    train = [data[i] for i in idx[:n_train]]
    dev   = [data[i] for i in idx[n_train:n_train+n_dev]]
    test  = [data[i] for i in idx[n_train+n_dev:]]
    return train, dev, test

def build_vocabs(train_tagged: List[TaggedSent], min_freq: int = 1):
    wc = Counter(w for s in train_tagged for (w, _) in s)
    words = [w for w, c in wc.items() if c > min_freq]
    word2id = {PAD:0, UNK:1}
    for w in sorted(words):
        word2id[w] = len(word2id)

    tags = sorted({t for s in train_tagged for (_, t) in s})
    tag2id = {t:i for i, t in enumerate(tags)}
    id2tag = {i:t for t,i in tag2id.items()}
    return word2id, tag2id, id2tag

def map_oov(sent: TaggedSent, word2id: Dict[str,int]) -> Tuple[List[int], List[int]]:
    w_ids = [word2id.get(w, word2id[UNK]) for (w, _) in sent]
    # tag ids filled later since tag2id only for labels
    return w_ids, None

class POSTagDataset(Dataset):
    def __init__(self, tagged_sents: List[TaggedSent], word2id, tag2id, min_freq_unk=1):
        self.word2id = word2id
        self.tag2id = tag2id
        self.samples = []
        for s in tagged_sents:
            w_ids = [word2id.get(w, word2id[UNK]) for (w, _) in s]
            t_ids = [tag2id[t] for (_, t) in s]
            self.samples.append((w_ids, t_ids))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_batch(batch, pad_id=0):
    # batch: list of (w_ids, t_ids)
    max_len = max(len(w) for (w, _) in batch)
    ws, ts, lens = [], [], []
    for (w, t) in batch:
        pad_w = w + [pad_id]*(max_len - len(w))
        pad_t = t + [-1]*(max_len - len(t))   # -1 will be ignored in loss
        ws.append(pad_w); ts.append(pad_t); lens.append(len(w))
    return (torch.tensor(ws, dtype=torch.long),
            torch.tensor(ts, dtype=torch.long),
            torch.tensor(lens, dtype=torch.long))


# -----------------------------
# CRF layer
# -----------------------------
class CRF(nn.Module):
    """
    Linear-chain CRF with start/end transitions and Viterbi decoding.
    Inputs: emissions [B, T, num_tags] (unnormalized scores)
            mask     [B, T] bool (True for real token, False for pad)
    """
    def __init__(self, num_tags: int, start_tag_id: int = None, end_tag_id: int = None):
        super().__init__()
        self.num_tags = num_tags
        # Transition scores: trans[i, j] = score of i -> j
        self.trans = nn.Parameter(torch.empty(num_tags, num_tags))
        nn.init.xavier_uniform_(self.trans)
        # Separate start/end vectors
        self.start = nn.Parameter(torch.empty(num_tags))
        self.end   = nn.Parameter(torch.empty(num_tags))
        nn.init.normal_(self.start, std=0.1)
        nn.init.normal_(self.end,   std=0.1)

    def _log_sum_exp(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        m, _ = x.max(dim=dim, keepdim=True)
        return m.squeeze(dim) + torch.log(torch.sum(torch.exp(x - m), dim=dim))

    def neg_log_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # emissions: [B,T,K], tags: [B,T], mask: [B,T]
        logZ = self._compute_log_partition(emissions, mask)          # [B]
        score = self._score_sentence(emissions, tags, mask)          # [B]
        return torch.mean(logZ - score)

    def _compute_log_partition(self, emissions, mask):
        B, T, K = emissions.shape
        # alpha for time 0: start + emission of tag y0
        alpha = self.start + emissions[:, 0]  # [B,K]
        for t in range(1, T):
            emit = emissions[:, t].unsqueeze(1)  # [B,1,K]
            # broadcast: alpha[:,:,None] + trans[None,:,:] + emit
            scores = alpha.unsqueeze(2) + self.trans.unsqueeze(0) + emit  # [B,K,K]
            alpha_next = self._log_sum_exp(scores, dim=1)  # sum over prev tags -> [B,K]
            alpha = torch.where(mask[:, t].unsqueeze(1), alpha_next, alpha)  # keep when masked
        alpha = alpha + self.end
        return self._log_sum_exp(alpha, dim=1)  # [B]

    def _score_sentence(self, emissions, tags, mask):
        B, T, K = emissions.shape
        # start + first emission
        score = self.start.gather(0, tags[:,0]) + emissions.gather(2, tags[:,0].unsqueeze(-1)).squeeze(-1)[:,0]
        for t in range(1, T):
            emit_t = emissions.gather(2, tags[:,t].unsqueeze(-1)).squeeze(-1)[:,t]    # [B]
            trans_t = self.trans[tags[:,t-1], tags[:,t]]                               # [B]
            step = emit_t + trans_t
            score = torch.where(mask[:, t], score + step, score)
        score = score + self.end.gather(0, tags.gather(1, (mask.long().sum(1)-1).unsqueeze(1)).squeeze(1))
        return score  # [B]

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        B, T, K = emissions.shape
        # time 0
        scores = self.start + emissions[:,0]           # [B,K]
        backpointers = []

        for t in range(1, T):
            broadcast_scores = scores.unsqueeze(2) + self.trans.unsqueeze(0)  # [B,K,K]
            best_scores, best_tags = broadcast_scores.max(dim=1)              # [B,K], [B,K]
            scores = torch.where(mask[:, t].unsqueeze(1), best_scores + emissions[:,t], scores)
            backpointers.append(best_tags)

        scores = scores + self.end
        best_last_tags = scores.argmax(dim=1)  # [B]
        paths = []
        for b in range(B):
            seq_len = int(mask[b].sum().item())
            bp = backpointers[:seq_len-1]
            tag = best_last_tags[b].item()
            path = [tag]
            for t in reversed(range(seq_len-1)):
                tag = bp[t][b, tag].item()
                path.append(tag)
            path.reverse()
            paths.append(path)
        return paths


# -----------------------------
# BiLSTM-CRF model
# -----------------------------
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size: int, num_tags: int, emb_dim=100, lstm_hidden=256, lstm_layers=1, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, lstm_hidden//2, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True, dropout=0.0 if lstm_layers==1 else dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, num_tags)
        self.crf = CRF(num_tags)

    def emissions(self, x, lengths):
        emb = self.emb(x)                    # [B,T,E]
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = self.dropout(out)
        logits = self.fc(out)                # [B,T,K]
        return logits

    def loss(self, x, y, lengths):
        mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)  # [B,T] bool
        emissions = self.emissions(x, lengths)
        return self.crf.neg_log_likelihood(emissions, y, mask)

    @torch.no_grad()
    def predict(self, x, lengths):
        mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        emissions = self.emissions(x, lengths)
        paths = self.crf.decode(emissions, mask)
        return paths


# -----------------------------
# Training / Evaluation
# -----------------------------
def accuracy(model, loader, id2tag):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, Y, L in loader:
            X, Y, L = X.to(DEVICE), Y.to(DEVICE), L.to(DEVICE)
            preds = model.predict(X, L)
            for b in range(X.size(0)):
                n = int(L[b].item())
                gold = Y[b, :n].tolist()
                pred = preds[b]
                # align lengths just in case
                m = min(n, len(pred))
                correct += sum(1 for i in range(m) if pred[i] == gold[i])
                total += m
    return correct / total if total else 0.0

def main():
    # Load PTB (45-tag PTB tagset)
    all_tagged: List[TaggedSent] = treebank.tagged_sents()
    train_raw, dev_raw, test_raw = train_dev_test_split(all_tagged, ratios=(0.85, 0.075, 0.075))

    # Build vocabs from training
    word2id, tag2id, id2tag = build_vocabs(train_raw, min_freq=1)

    # Datasets & loaders
    train_ds = POSTagDataset(train_raw, word2id, tag2id)
    dev_ds   = POSTagDataset(dev_raw,   word2id, tag2id)
    test_ds  = POSTagDataset(test_raw,  word2id, tag2id)

    BATCH = 32
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate_batch)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH, shuffle=False, collate_fn=collate_batch)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, collate_fn=collate_batch)

    # Model
    model = BiLSTM_CRF(
        vocab_size=len(word2id),
        num_tags=len(tag2id),
        emb_dim=100, lstm_hidden=256, lstm_layers=1, dropout=0.3
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)

    # Train
    EPOCHS = 6
    best_dev = 0.0
    for ep in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        for X, Y, L in train_loader:
            X, Y, L = X.to(DEVICE), Y.to(DEVICE), L.to(DEVICE)
            loss = model.loss(X, Y, L)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()
        dev_acc = accuracy(model, dev_loader, id2tag)
        test_acc = accuracy(model, test_loader, id2tag)
        print(f"Epoch {ep:02d} | loss {total_loss/len(train_loader):.4f} | dev {dev_acc:.4f} | test {test_acc:.4f}")
        if dev_acc > best_dev:
            best_dev = dev_acc
            torch.save(model.state_dict(), "bilstm_crf_pos.pt")

    print("\nBest dev acc:", best_dev)
    print("Reloading best and reporting final test…")
    model.load_state_dict(torch.load("bilstm_crf_pos.pt", map_location=DEVICE))
    print("Final test acc:", accuracy(model, test_loader, id2tag))

    demo = "The quick brown fox jumps over the lazy dog .".split()
    demo_ids = [word2id.get(w, word2id[UNK]) for w in demo]
    X = torch.tensor([demo_ids], dtype=torch.long).to(DEVICE)
    L = torch.tensor([len(demo)], dtype=torch.long).to(DEVICE)
    pred = model.predict(X, L)[0]
    inv = {i:t for t,i in tag2id.items()}
    print("\nDemo prediction:")
    print(list(zip(demo, [inv[i] for i in pred])))

if __name__ == "__main__":
    main()
