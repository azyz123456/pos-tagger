# ------------------------------------------------------------
# Part-of-speech tagging with Bigram/Trigram HMMs on Penn Treebank
# - Trains on NLTK's treebank tagged sentences
# - Handles OOV with <UNK> using a frequency cutoff on training
# - Implements Viterbi decoding with optional beam search
# - Evaluates on a held-out test set
# ------------------------------------------------------------

import math
import random
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional
import nltk

# Ensure the Penn Treebank is available
try:
    from nltk.corpus import treebank
except LookupError:
    nltk.download("treebank")
    from nltk.corpus import treebank


START = "<s>"
START2 = "<s2>"   # second start symbol (for trigram)
END = "</s>"
UNK = "<UNK>"

TaggedSent = List[Tuple[str, str]]
Sent = List[str]


def train_dev_test_split(data, seed=1337, ratios=(0.8, 0.1, 0.1)):
    """Split list into train/dev/test according to ratios."""
    assert abs(sum(ratios) - 1.0) < 1e-9
    rnd = random.Random(seed)
    idx = list(range(len(data)))
    rnd.shuffle(idx)
    n = len(data)
    n_train = int(ratios[0] * n)
    n_dev = int(ratios[1] * n)
    train = [data[i] for i in idx[:n_train]]
    dev = [data[i] for i in idx[n_train:n_train + n_dev]]
    test = [data[i] for i in idx[n_train + n_dev:]]
    return train, dev, test


def replace_rare_words(tagged_sents: List[TaggedSent], min_freq: int = 1):
    """Replace words with <UNK> if their global frequency <= min_freq in the training set."""
    word_counts = Counter(w for sent in tagged_sents for (w, t) in sent)
    def norm_word(w):
        return w if word_counts[w] > min_freq else UNK
    normalized = [[(norm_word(w), t) for (w, t) in sent] for sent in tagged_sents]
    vocab = set(w for sent in normalized for (w, _) in sent)
    return normalized, vocab


# -------------------------------
# Base HMM Tagger
# -------------------------------

class BaseHMMTagger:
    def __init__(self, alpha_emiss: float = 1.0, alpha_trans: float = 0.1):
        """
        alpha_emiss: Laplace smoothing for emissions
        alpha_trans: Laplace smoothing for transitions
        """
        self.alpha_emiss = alpha_emiss
        self.alpha_trans = alpha_trans
        self.tags: List[str] = []
        self.vocab: set = set([UNK])
        # Emission counts: (tag, word) -> count
        self.emiss_counts: Dict[Tuple[str, str], int] = Counter()
        # Tag counts for emission normalization
        self.tag_token_counts: Dict[str, int] = Counter()

    # ---- Emissions ----
    def _emiss_logp(self, tag: str, word: str) -> float:
        """log P(word | tag) with Laplace smoothing."""
        if word not in self.vocab:
            word = UNK
        V = len(self.vocab)
        num = self.emiss_counts[(tag, word)] + self.alpha_emiss
        den = self.tag_token_counts[tag] + self.alpha_emiss * V
        return math.log(num) - math.log(den)

    # ---- Interface to implement in subclasses ----
    def fit(self, train_sents: List[TaggedSent]):
        raise NotImplementedError

    def tag(self, sent: Sent, beam_k: Optional[int] = None) -> List[Tuple[str, str]]:
        raise NotImplementedError

    # ---- Evaluation ----
    def accuracy(self, tagged_sents: List[TaggedSent], beam_k: Optional[int] = None) -> float:
        correct = 0
        total = 0
        for sent in tagged_sents:
            words = [w for (w, t) in sent]
            gold = [t for (w, t) in sent]
            pred_pairs = self.tag(words, beam_k=beam_k)
            pred = [t for (_, t) in pred_pairs]
            for g, p in zip(gold, pred):
                correct += int(g == p)
                total += 1
        return correct / total if total else 0.0


# Bigram HMM Tagger

class BigramHMMTagger(BaseHMMTagger):
    def __init__(self, alpha_emiss: float = 1.0, alpha_trans: float = 0.1):
        super().__init__(alpha_emiss, alpha_trans)
        # Transition counts: (prev_tag, curr_tag) -> count
        self.trans_counts: Dict[Tuple[str, str], int] = Counter()
        # Counts for prev_tag occurrences (for denom)
        self.prev_tag_counts: Dict[str, int] = Counter()

    def fit(self, train_sents: List[TaggedSent]):
        # Collect tags and vocab from training (assumes <UNK> already in vocab)
        self.tags = sorted(list({t for sent in train_sents for (_, t) in sent}))
        # Emission counts
        for sent in train_sents:
            prev = START
            for (w, t) in sent:
                self.vocab.add(w)
                self.emiss_counts[(t, w)] += 1
                self.tag_token_counts[t] += 1
                self.trans_counts[(prev, t)] += 1
                self.prev_tag_counts[prev] += 1
                prev = t
            # transition to END
            self.trans_counts[(prev, END)] += 1
            self.prev_tag_counts[prev] += 1

        # Add transitions from START to all tags with small prior to avoid zeros
        for t in self.tags:
            _ = self.trans_counts[(START, t)]
        # Add transitions to END to avoid zeros
        for t in self.tags:
            _ = self.trans_counts[(t, END)]

    def _trans_logp(self, prev_tag: str, curr_tag: str) -> float:
        """log P(curr_tag | prev_tag) with Laplace smoothing."""
        T = len(self.tags) + 1  # + END
        num = self.trans_counts[(prev_tag, curr_tag)] + self.alpha_trans
        den = self.prev_tag_counts[prev_tag] + self.alpha_trans * (T + (1 if prev_tag == START else 0))
        return math.log(num) - math.log(den)

    def tag(self, sent: Sent, beam_k: Optional[int] = None) -> List[Tuple[str, str]]:
        if not sent:
            return []

        # Viterbi with optional beam pruning
        # dp[t][tag] = (log_prob, backpointer_tag)
        dp: List[Dict[str, Tuple[float, Optional[str]]]] = []
        # Init (transition from START)
        first_word = sent[0] if sent[0] in self.vocab else UNK
        curr = {}
        for tag in self.tags:
            score = self._trans_logp(START, tag) + self._emiss_logp(tag, first_word)
            curr[tag] = (score, START)
        # Beam prune
        if beam_k is not None and beam_k > 0:
            curr = dict(sorted(curr.items(), key=lambda x: x[1][0], reverse=True)[:beam_k])
        dp.append(curr)

        # Recursion
        for i in range(1, len(sent)):
            w = sent[i] if sent[i] in self.vocab else UNK
            prev_states = dp[-1]
            # Optionally, restrict prev_tags to beam top-K
            candidates_prev = list(prev_states.keys())
            nxt: Dict[str, Tuple[float, Optional[str]]] = {}
            for tag in self.tags:
                best_score = -1e100
                best_prev = None
                # consider transitions from beam-limited prev tags
                for ptag in candidates_prev:
                    score_prev = prev_states[ptag][0]
                    score = score_prev + self._trans_logp(ptag, tag) + self._emiss_logp(tag, w)
                    if score > best_score:
                        best_score = score
                        best_prev = ptag
                nxt[tag] = (best_score, best_prev)
            if beam_k is not None and beam_k > 0:
                nxt = dict(sorted(nxt.items(), key=lambda x: x[1][0], reverse=True)[:beam_k])
            dp.append(nxt)

        # Termination (add transition to END)
        last_states = dp[-1]
        best_final_score = -1e100
        best_final_tag = None
        for tag, (score, prev) in last_states.items():
            end_score = score + self._trans_logp(tag, END)
            if end_score > best_final_score:
                best_final_score = end_score
                best_final_tag = tag

        # Backtrace
        tags_out = [best_final_tag]
        for i in range(len(sent) - 1, 0, -1):
            tags_out.append(dp[i][tags_out[-1]][1])  # previous tag
        tags_out.reverse()
        return list(zip(sent, tags_out))


# Trigram HMM Tagger (2nd-order)

class TrigramHMMTagger(BaseHMMTagger):
    def __init__(self, alpha_emiss: float = 1.0, alpha_trans: float = 0.1):
        super().__init__(alpha_emiss, alpha_trans)
        # trigram counts: (t_{i-2}, t_{i-1}, t_i) -> count
        self.tri_counts: Dict[Tuple[str, str, str], int] = Counter()
        # bigram counts for denominator: (t_{i-2}, t_{i-1}) -> count
        self.bi_counts: Dict[Tuple[str, str], int] = Counter()

    def fit(self, train_sents: List[TaggedSent]):
        self.tags = sorted(list({t for sent in train_sents for (_, t) in sent}))

        for sent in train_sents:
            prev2 = START2
            prev1 = START
            for (w, t) in sent:
                self.vocab.add(w)
                self.emiss_counts[(t, w)] += 1
                self.tag_token_counts[t] += 1
                self.tri_counts[(prev2, prev1, t)] += 1
                self.bi_counts[(prev2, prev1)] += 1
                prev2, prev1 = prev1, t
            # transition to END as the final state
            self.tri_counts[(prev2, prev1, END)] += 1
            self.bi_counts[(prev2, prev1)] += 1

        # Ensure transitions from starts exist (sparse smoothing guard)
        for t in self.tags:
            _ = self.tri_counts[(START2, START, t)]

    def _trans_logp(self, t2: str, t1: str, t: str) -> float:
        """
        log P(t | t2, t1) with Laplace smoothing over tag vocabulary + END.
        """
        T = len(self.tags) + 1  # + END
        num = self.tri_counts[(t2, t1, t)] + self.alpha_trans
        den = self.bi_counts[(t2, t1)] + self.alpha_trans * T
        # Guard zero denominator (if unseen bigram context) by backing off to uniform
        if den == 0:
            return -math.log(T)
        return math.log(num) - math.log(den)

    def tag(self, sent: Sent, beam_k: Optional[int] = None) -> List[Tuple[str, str]]:
        if not sent:
            return []

        # Viterbi for 2nd-order HMM using states as pairs (t_{i-1}, t_i)
        # dp[i]: dict mapping (u, v) -> (log_prob, backptr_tag_for_u)
        dp: List[Dict[Tuple[str, str], Tuple[float, Optional[str]]]] = []

        # Initialize for position 0 (word1): transition (START2, START) -> t1
        w1 = sent[0] if sent[0] in self.vocab else UNK
        curr: Dict[Tuple[str, str], Tuple[float, Optional[str]]] = {}
        for t1 in self.tags:
            score = self._trans_logp(START2, START, t1) + self._emiss_logp(t1, w1)
            curr[(START, t1)] = (score, START2)  # backpointer is START2 but not used directly here
        if beam_k is not None and beam_k > 0:
            curr = dict(sorted(curr.items(), key=lambda x: x[1][0], reverse=True)[:beam_k])
        dp.append(curr)

        if len(sent) == 1:
            # Add transition to END
            best, best_pair = -1e100, None
            for (u, v), (score, _) in dp[-1].items():
                end_score = score + self._trans_logp(u, v, END)
                if end_score > best:
                    best, best_pair = end_score, (u, v)
            # Recover single tag v
            return [(sent[0], best_pair[1])]

        # Position 1 (word2) uses transitions (START, t1) -> t2
        w2 = sent[1] if sent[1] in self.vocab else UNK
        nxt: Dict[Tuple[str, str], Tuple[float, Optional[str]]] = {}
        for (u, v), (score_uv, _) in dp[-1].items():
            for t2 in self.tags:
                score = score_uv + self._trans_logp(u, v, t2) + self._emiss_logp(t2, w2)
                key = (v, t2)
                if (key not in nxt) or (score > nxt[key][0]):
                    nxt[key] = (score, u)  # backpointer: best t_{i-2}
        if beam_k is not None and beam_k > 0:
            nxt = dict(sorted(nxt.items(), key=lambda x: x[1][0], reverse=True)[:beam_k])
        dp.append(nxt)

        # Recursion for positions i >= 2
        for i in range(2, len(sent)):
            w = sent[i] if sent[i] in self.vocab else UNK
            prev_layer = dp[-1]
            # Candidate previous pairs limited by beam
            prev_pairs = list(prev_layer.keys())
            curr: Dict[Tuple[str, str], Tuple[float, Optional[str]]] = {}
            # We need to choose new tag t_i and consider all prev pairs (u, v) where u=t_{i-2}, v=t_{i-1}
            # Then state key is (v, t_i)
            # For beam pruning, we limit prev_pairs AND we'll prune current states to top-K after.
            for (u, v) in prev_pairs:
                score_uv, _ = prev_layer[(u, v)]
                for t in self.tags:
                    score = score_uv + self._trans_logp(u, v, t) + self._emiss_logp(t, w)
                    key = (v, t)
                    if (key not in curr) or (score > curr[key][0]):
                        curr[key] = (score, u)
            if beam_k is not None and beam_k > 0:
                curr = dict(sorted(curr.items(), key=lambda x: x[1][0], reverse=True)[:beam_k])
            dp.append(curr)

        # Termination: add transition to END from each pair (u, v)
        best_final_score = -1e100
        best_pair = None
        for (u, v), (score, back_u) in dp[-1].items():
            end_score = score + self._trans_logp(u, v, END)
            if end_score > best_final_score:
                best_final_score = end_score
                best_pair = (u, v)

        # Backtrace: we know last two tags (u, v) = best_pair
        n = len(sent)
        tags_out = [None] * n
        tags_out[-1] = best_pair[1]   # v = t_n
        tags_out[-2] = best_pair[0]   # u = t_{n-1}

        # Backtrack remaining positions
        # For dp[i], keys are (t_{i-1}, t_i).
        # We stored backpointer = best u (i-2) for each key.
        for i in range(n - 3, -1, -1):
            # We need backpointer for key (tags_out[i+1], tags_out[i+2])
            key = (tags_out[i+1], tags_out[i+2])
            back_u = dp[i + 2][key][1]  # this is t_{i}
            tags_out[i] = back_u

        return list(zip(sent, tags_out))


# -------------------------------
# Data preparation and training
# -------------------------------

def build_pipeline(min_freq_unk: int = 1,
                   alpha_emiss: float = 1.0,
                   alpha_trans: float = 0.1):
    """
    Build train/dev/test sets using NLTK Penn Treebank.
    Replaces rare words with <UNK> in training and applies same mapping to dev/test.
    """
    # Load PTB tagged sentences (default PTB tagset)
    all_tagged: List[TaggedSent] = treebank.tagged_sents()

    # Split
    train_raw, dev_raw, test_raw = train_dev_test_split(all_tagged, seed=1337, ratios=(0.8, 0.1, 0.1))

    # Replace rare words by <UNK> *in training only*
    train_norm, vocab = replace_rare_words(train_raw, min_freq=min_freq_unk)

    def map_oov(sent: TaggedSent):
        return [(w if w in vocab else UNK, t) for (w, t) in sent]

    dev_norm = [map_oov(s) for s in dev_raw]
    test_norm = [map_oov(s) for s in test_raw]

    # Instantiate models
    bigram = BigramHMMTagger(alpha_emiss=alpha_emiss, alpha_trans=alpha_trans)
    trigram = TrigramHMMTagger(alpha_emiss=alpha_emiss, alpha_trans=alpha_trans)

    # Fit
    bigram.fit(train_norm)
    trigram.fit(train_norm)

    return (bigram, trigram), (train_norm, dev_norm, test_norm)


def evaluate_models():
    (bigram, trigram), (train, dev, test) = build_pipeline(
        min_freq_unk=1,   # words seen once become <UNK>
        alpha_emiss=1.0,  # Laplace for emissions
        alpha_trans=0.1   # mild smoothing for transitions
    )

    # Evaluate without beam and with beam
    for beam in [None, 10, 5]:
        b_dev_acc = bigram.accuracy(dev, beam_k=beam)
        t_dev_acc = trigram.accuracy(dev, beam_k=beam)
        b_test_acc = bigram.accuracy(test, beam_k=beam)
        t_test_acc = trigram.accuracy(test, beam_k=beam)
        label = "no-beam" if beam is None else f"beam={beam}"
        print(f"[Bigram | {label}]  Dev Acc: {b_dev_acc:.4f} | Test Acc: {b_test_acc:.4f}")
        print(f"[Trigram| {label}]  Dev Acc: {t_dev_acc:.4f} | Test Acc: {t_test_acc:.4f}")

    # Demo tagging
    demo_sent = "The quick brown fox jumps over the lazy dog .".split()
    print("\nDemo sentence:", " ".join(demo_sent))
    print("Bigram tags:", bigram.tag(demo_sent, beam_k=10))
    print("Trigram tags:", trigram.tag(demo_sent, beam_k=10))


if __name__ == "__main__":
    evaluate_models()
