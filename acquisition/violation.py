"""Violation recording: predictable melodies with planted improbable notes.

Melodies at VIOLATION_TEMPERATURE (0.5); about 10% of notes are replaced
by the pitch the Bach model finds least probable with the same leap size
(±1 semitone) as the original (composer/violations.py). The analysis is
fixed in advance: analysis/VIOLATION_PLAN.md.

Blocks, breaks, IDs, sync checks and resume work exactly like calibration
(acquisition/record.py). Each block also saves which notes were
violations, beside the note log (whose columns stay the contract's 7).

Usage (from the repo root; close view_stream first):
    python -m acquisition.violation --dry-run --fake-eeg     # no hardware, silent
    python -m acquisition.violation --dry-run                 # headset, silent
    python -m acquisition.violation --subject 1               # 6 blocks, ~16 min
    python -m acquisition.violation --subject 1 --start-block 4   # resume

Output (data/raw/), per block:
    YYYY-MM-DD_subjNN_violation_<block>.h5 / _notes.csv / _violations.csv
"""

import numpy as np
import pandas as pd

import config
from acquisition.record import MELODY_ID_STRIDE, run_session
from audio.playback import midi_to_hz
from composer.sample import intervals, random_key_shift
from composer.violations import violation_melody


def build_violation_block(model, block, duration, rng):
    """Like record.build_block, plus the list of violations (by note slot)."""
    freqs, notes, violations = [], [], []
    k = 0
    while len(freqs) * config.IOI < duration:
        k += 1
        pitches, bits, viols = violation_melody(model, rng)
        shift = random_key_shift(rng)
        played = [p + shift for p in pitches]
        base = len(notes)
        for pitch, iv, s in zip(played, intervals(pitches), bits):
            notes.append({"melody_id": block * MELODY_ID_STRIDE + k, "pitch": pitch,
                          "interval": iv, "surprisal": s,
                          "temperature": config.VIOLATION_TEMPERATURE})
        for v in viols:
            violations.append({"slot": base + v["position"],
                               "original_pitch": v["original_pitch"] + shift,
                               "original_surprisal": v["original_surprisal"]})
        freqs += list(midi_to_hz(played)) + [None] * config.MELODY_GAP_SLOTS
    return freqs, notes, violations


def violations_path(eeg_path):
    return eeg_path.with_name(eeg_path.stem + "_violations.csv")


def save_violations(eeg_path, log, first_index, violations):
    """Save the violations that were actually played; print a quick summary."""
    v = pd.DataFrame(violations, columns=["slot", "original_pitch", "original_surprisal"])
    v = v[v["slot"] < len(log)]
    v.insert(0, "note_index", first_index + v.pop("slot"))
    v.to_csv(violations_path(eeg_path), index=False)
    print(f"saved {violations_path(eeg_path)}")

    is_v = log["note_index"].isin(v["note_index"])
    print(f"\n{is_v.sum()} violations in {len(log)} notes ({100 * is_v.mean():.1f}%): "
          f"surprisal {log.loc[is_v, 'surprisal'].mean():.1f} vs. {log.loc[~is_v, 'surprisal'].mean():.1f} bits, "
          f"|interval| {log.loc[is_v, 'interval'].abs().mean():.2f} vs. "
          f"{log.loc[~is_v, 'interval'].abs().mean():.2f} semitones")


def main():
    run_session(__doc__, "violation", config.VIOLATION_BLOCKS,
                build_violation_block, after_block=save_violations)


if __name__ == "__main__":
    main()
