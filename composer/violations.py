"""Plant improbable notes ("violations") in generated melodies.

At a violation position, the generated note is replaced by the pitch the
untempered model finds least probable, among pitches whose leap from the
previous note is within VIOLATION_INTERVAL_TOL of the original leap. That
makes violations improbable but not bigger leaps, so improbability isn't
confounded with interval size (analysis/VIOLATION_PLAN.md).

Usage (from the repo root):
    python -m composer.violations      # statistics of planted violations
"""

import argparse

import numpy as np

import config
from composer import markov
from composer.sample import sample_melody


def violation_positions(n, rng, rate=config.VIOLATION_RATE):
    """Positions for violations in a melody of n notes, respecting lead,
    tail and minimum gap. The per-note probability is raised so that,
    despite the blocked positions after each violation, about `rate` of
    notes become violations."""
    p = 1.0 / max(1.0 / rate - config.VIOLATION_MIN_GAP, 1.0)
    positions, last = [], -10 ** 6
    for i in range(config.VIOLATION_MIN_LEAD, n - config.VIOLATION_MIN_TAIL):
        if i - last > config.VIOLATION_MIN_GAP and rng.random() < p:
            positions.append(i)
            last = i
    return positions


def violation_pitch(model, context, original):
    """Least probable pitch with a leap within tolerance of the original's.

    Only steps: the original's leap and the violation's must both be
    1..VIOLATION_MAX_LEAP semitones (None if the original isn't a step).
    """
    prev = context[-1]
    leap = abs(original - prev)
    if not 1 <= leap <= config.VIOLATION_MAX_LEAP:
        return None
    probs = markov.next_distribution(model, context)
    best, best_p = None, np.inf
    for p, prob in zip(model["vocab"], probs):
        p = int(p)
        new_leap = abs(p - prev)
        if (p in (original, prev) or abs(new_leap - leap) > config.VIOLATION_INTERVAL_TOL
                or not 1 <= new_leap <= config.VIOLATION_MAX_LEAP):
            continue
        if prob < best_p:
            best, best_p = p, prob
    return best


def plant_violations(model, pitches, rng):
    """Returns (pitches, surprisals, violations).

    violations: one dict per planted note with its position, the original
    pitch and the original's surprisal. Surprisals are recomputed on the
    melody as played, under the untempered model.
    """
    pitches = list(pitches)
    violations = []
    for i in violation_positions(len(pitches), rng):
        new = violation_pitch(model, pitches[:i], pitches[i])
        if new is None:
            continue
        violations.append({"position": i, "original_pitch": pitches[i],
                           "original_surprisal": markov.surprisal(model, pitches[:i], pitches[i])})
        pitches[i] = new
    return pitches, markov.melody_surprisal(model, pitches), violations


def violation_melody(model, rng, temperature=config.VIOLATION_TEMPERATURE):
    """A predictable melody with violations planted."""
    pitches, _ = sample_melody(model, temperature=temperature, rng=rng)
    return plant_violations(model, pitches, rng)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200, help="melodies to generate")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = markov.load()
    rng = np.random.default_rng(args.seed)
    n_notes, viol_bits, norm_bits, viol_leap, orig_leap, norm_leap = 0, [], [], [], [], []
    for _ in range(args.n):
        pitches, bits, viols = violation_melody(model, rng)
        n_notes += len(pitches)
        pos = {v["position"] for v in viols}
        after = {p + 1 for p in pos}
        for i in range(1, len(pitches)):
            leap = abs(pitches[i] - pitches[i - 1])
            if i in pos:
                viol_bits.append(bits[i])
                viol_leap.append(leap)
            elif i not in after:
                norm_bits.append(bits[i])
                norm_leap.append(leap)
        orig_leap += [abs(v["original_pitch"] - pitches[v["position"] - 1]) for v in viols]
    print(f"{args.n} melodies, {n_notes} notes, {len(viol_bits)} violations "
          f"({100 * len(viol_bits) / n_notes:.1f}%)")
    print(f"surprisal: violations {np.mean(viol_bits):.2f} bits, normal {np.mean(norm_bits):.2f} bits")
    print(f"|interval|: violations {np.mean(viol_leap):.2f}, originals they replaced "
          f"{np.mean(orig_leap):.2f}, normal {np.mean(norm_leap):.2f} semitones")


if __name__ == "__main__":
    main()
