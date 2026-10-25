"""Load Bach chorale soprano lines as MIDI pitch lists, transposed to C.

See composer/DECISIONS.md (#1, #9, #11, #17, #19). Parsing the corpus is
slow (~1–2 min), so the full result is cached in data/processed/.

Usage (from the repo root):
    python -m composer.corpus            # load (or read cache) and summarize
    python -m composer.corpus --refresh  # rebuild the cache
"""

import argparse
import json

import numpy as np
from music21 import corpus

import config

CACHE = config.DATA_PROCESSED / "soprano_melodies.json"


def find_tonic(chorale):
    return chorale.analyze("key").tonic

def merge_repeats(melody):
    """Collapse runs of the same pitch into one note (DECISIONS #20)."""
    return [p for i, p in enumerate(melody) if i == 0 or p != melody[i - 1]]

def load_soprano_lines(limit=None, refresh=False, merge=config.MERGE_REPEATS):
    melodies = load_raw_soprano_lines(limit, refresh)
    return [merge_repeats(m) for m in melodies] if merge else melodies

def load_raw_soprano_lines(limit=None, refresh=False):
    if CACHE.exists() and not refresh and limit is None:
        return json.loads(CACHE.read_text())

    # loading Bach chorale soprano parts
    soprano_parts = []

    for i, chorale in enumerate(corpus.chorales.Iterator()):
        if limit is not None and i >= limit:
            break
        for part in chorale.parts:
            if part.partName and part.partName.strip().lower().startswith("soprano"):
                soprano_part = []
                # transpose to C
                tonic = find_tonic(chorale).pitchClass
                offset = 12 - tonic if 12 - tonic < 6 else -tonic

                for note in part.flatten().notes:
                    # skip tied notes
                    if note.tie and (note.tie.type == "start" or note.tie.type == "continue"):
                        continue
                    soprano_part.append(note.pitch.midi + offset)

                soprano_parts.append(soprano_part)
                break

    if limit is None:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(soprano_parts))
    return soprano_parts

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="only the first N chorales (not cached)")
    ap.add_argument("--refresh", action="store_true", help="rebuild the cache")
    args = ap.parse_args()

    melodies = load_soprano_lines(limit=args.limit, refresh=args.refresh)
    pitches = np.concatenate(melodies)
    lengths = [len(m) for m in melodies]
    lo, hi = config.PITCH_RANGE
    print(f"{len(melodies)} melodies, {len(pitches)} notes "
          f"(per melody {min(lengths)}-{max(lengths)}, median {int(np.median(lengths))})")
    print(f"pitch range {pitches.min()}-{pitches.max()}, median {int(np.median(pitches))}; "
          f"outside PITCH_RANGE {lo}-{hi}: {((pitches < lo) | (pitches > hi)).sum()}")

if __name__ == "__main__":
    main()
