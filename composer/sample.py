"""Generate melodies by temperature sampling from the Markov model.

Temperature reshapes the distribution for sampling only: P^(1/T),
renormalized, computed in log space. Surprisal of each generated note is
always taken from the untempered model (CLAUDE.md invariant 1), so it
stays comparable across temperatures.

Usage (from the repo root; needs python -m composer.markov --train first):
    python -m composer.sample                 # mean surprisal per temperature
    python -m composer.sample --play 1.0      # listen to one melody at T = 1.0
"""

import argparse

import numpy as np

import config
from composer import markov


def tempered(probs, temperature):
    """probs^(1/T), renormalized, without underflow."""
    logp = np.log(probs) / temperature
    p = np.exp(logp - logp.max())
    return p / p.sum()


def sample_melody(model, length=config.MELODY_LENGTH,
                  temperature=config.DEFAULT_TEMPERATURE, rng=None,
                  end_on_tonic=config.END_ON_TONIC):
    """Returns (pitches, surprisals): MIDI ints, and untempered bits per note.

    With end_on_tonic, sampling continues past `length` until the model
    itself produces the tonic (up to config.MAX_EXTRA_NOTES more), so
    melodies sound finished. Nothing is forced: every note is still
    sampled, and its surprisal is unchanged (DECISIONS #21).
    """
    rng = rng or np.random.default_rng()
    pitches, surprisals = [], []
    max_len = length + (config.MAX_EXTRA_NOTES if end_on_tonic else 0)
    while len(pitches) < max_len:
        probs = markov.next_distribution(model, pitches)
        i = rng.choice(len(probs), p=tempered(probs, temperature))
        pitches.append(int(model["vocab"][i]))
        surprisals.append(float(-np.log2(probs[i])))
        if len(pitches) >= length and (not end_on_tonic or pitches[-1] % 12 == config.TONIC_PC):
            break
    return pitches, surprisals


def random_key_shift(rng=None):
    """Semitones to move a melody from C to a random key (0 if RANDOM_KEY is off).

    Apply to the pitches that are *played* only. Surprisal and intervals
    are computed in C and don't change (DECISIONS #18).
    """
    if not config.RANDOM_KEY:
        return 0
    rng = rng or np.random.default_rng()
    lo, hi = config.KEY_SHIFT_RANGE
    return int(rng.integers(lo, hi + 1))


def intervals(pitches):
    """Signed semitones from the previous note; NaN for the first."""
    return [np.nan] + list(np.diff(pitches).astype(float))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50, help="melodies per temperature")
    ap.add_argument("--play", type=float, metavar="T", help="play one melody at temperature T")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    model = markov.load()
    rng = np.random.default_rng(args.seed)

    if args.play is not None:
        from audio.playback import midi_to_hz, play_with_onsets, render_sequence
        pitches, bits = sample_melody(model, temperature=args.play, rng=rng)
        shift = random_key_shift(rng)
        played = [p + shift for p in pitches]
        names = "C C# D Eb E F F# G Ab A Bb B".split()
        print(f"T = {args.play}, key {names[(config.TONIC_PC + shift) % 12]}: {played}\n"
              f"mean surprisal {np.mean(bits):.2f} bits")
        audio, onsets = render_sequence(list(midi_to_hz(played)))
        play_with_onsets(audio, onsets, lambda k: None)
        return

    # Check: mean untempered surprisal should rise with temperature
    print(f"{'T':>5} {'mean bits':>10} {'mean |interval|':>16}")
    for t in config.CALIBRATION_TEMPERATURES:
        bits, leaps = [], []
        for _ in range(args.n):
            pitches, s = sample_melody(model, temperature=t, rng=rng)
            bits += s
            leaps += list(np.abs(np.diff(pitches)))
        print(f"{t:>5} {np.mean(bits):>10.2f} {np.mean(leaps):>16.2f}")


if __name__ == "__main__":
    main()
