"""Markov melody model: counts, smoothing, back-off and surprisal.

The model counts, for every context length k = 0..order, which pitch
followed each context of the k previous pitches (melody starts padded
with START). Probabilities are made at lookup (DECISIONS #2–#6):

  * back off to the longest context that was seen in training
  * add-α smoothing over the vocabulary (config.PITCH_RANGE)

Surprisal = −log₂ P(pitch | context) under this untempered model
(CLAUDE.md invariant 1).

Usage (from the repo root):
    python -m composer.markov --evaluate   # held-out surprisal for orders/alphas
    python -m composer.markov --train      # fit on all melodies, save to models/
"""

import argparse
import pickle
from collections import Counter, defaultdict

import numpy as np

import config
from composer.corpus import load_soprano_lines

START = None  # padding before a melody's first note (DECISIONS #4)
MODEL_PATH = config.MODELS_DIR / "markov.pkl"


def train(melodies, order=config.MARKOV_ORDER, alpha=config.SMOOTHING_ALPHA):
    """Count continuations for every context length 0..order."""
    lo, hi = config.PITCH_RANGE
    counts = [defaultdict(Counter) for _ in range(order + 1)]
    for melody in melodies:
        padded = [START] * order + list(melody)
        for t in range(order, len(padded)):
            for k in range(order + 1):
                counts[k][tuple(padded[t - k:t])][padded[t]] += 1
    return {"order": order, "alpha": alpha, "vocab": np.arange(lo, hi + 1),
            "counts": [dict(c) for c in counts]}


def next_distribution(model, context):
    """P(next pitch | context) over model['vocab']; sums to 1, no zeros.

    context: the previous pitches, any length (START-padded if short).
    """
    order = model["order"]
    context = ([START] * order + list(context))[len(context):] if order else []
    for k in range(order, -1, -1):  # longest context seen in training wins
        seen = model["counts"][k].get(tuple(context[order - k:]))
        if seen:
            break
    vocab, alpha = model["vocab"], model["alpha"]
    c = np.array([seen.get(int(p), 0) for p in vocab], dtype=float)
    return (c + alpha) / (c.sum() + alpha * len(vocab))


def surprisal(model, context, pitch):
    """−log₂ P(pitch | context) in bits, untempered."""
    probs = next_distribution(model, context)
    return float(-np.log2(probs[int(pitch) - model["vocab"][0]]))


def melody_surprisal(model, melody):
    """Surprisal of every note, each given the notes before it."""
    return [surprisal(model, melody[:t], p) for t, p in enumerate(melody)]


def mean_surprisal(model, melodies):
    """Average bits per note over melodies (cross-entropy)."""
    return float(np.mean([s for m in melodies for s in melody_surprisal(model, m)]))


def split_by_melody(melodies, test_frac=0.2, seed=0):
    """Train/held-out split by whole melody, never by note."""
    idx = np.random.default_rng(seed).permutation(len(melodies))
    test = set(idx[:int(round(test_frac * len(melodies)))].tolist())
    return ([m for i, m in enumerate(melodies) if i not in test],
            [m for i, m in enumerate(melodies) if i in test])


def save(model, path=MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load(path=MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate(melodies, orders=(1, 2, 3), alphas=(0.01, 0.1, 1.0)):
    """Print training vs. held-out mean surprisal for each order and alpha."""
    train_set, test_set = split_by_melody(melodies)
    print(f"{len(train_set)} training / {len(test_set)} held-out melodies\n")
    print(f"{'order':>5} {'alpha':>6} {'train bits':>11} {'held-out bits':>14}")
    for order in orders:
        for alpha in alphas:
            m = train(train_set, order, alpha)
            print(f"{order:>5} {alpha:>6} {mean_surprisal(m, train_set):>11.3f} "
                  f"{mean_surprisal(m, test_set):>14.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true", help="fit on all melodies and save")
    ap.add_argument("--evaluate", action="store_true", help="compare orders and alphas")
    args = ap.parse_args()
    if not (args.train or args.evaluate):
        ap.error("give --train and/or --evaluate")

    melodies = load_soprano_lines()
    if args.evaluate:
        evaluate(melodies)
    if args.train:
        model = train(melodies)
        save(model)
        print(f"saved {MODEL_PATH} (order {model['order']}, alpha {model['alpha']}, "
              f"{len(melodies)} melodies)")


if __name__ == "__main__":
    main()
