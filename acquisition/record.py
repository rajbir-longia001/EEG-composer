"""Calibration recording: generated melodies at a mix of temperatures.

Each block (~2.5 min) is its own recording, saved as soon as it ends, with
a break between blocks. Within a block, melodies follow each other with a
short silent gap; inside a melody, notes are strictly isochronous. Each
melody gets a temperature from CALIBRATION_TEMPERATURES (shuffled,
balanced) and a random key (DECISIONS #18), giving a wide range of
surprisal values for the ERP/TRF analysis.

IDs are unique across the session and derived from the block number, so a
crashed session can resume with --start-block:
    note_index = (block − 1) × 1000 + 1, 2, ...
    melody_id  = block × 100 + 1, 2, ...

Usage (from the repo root; train the composer first with
`python -m composer.markov --train`, and close view_stream):
    python -m acquisition.record --dry-run --fake-eeg      # no hardware, silent
    python -m acquisition.record --dry-run                  # headset, silent
    python -m acquisition.record --subject 1                # all 5 blocks
    python -m acquisition.record --subject 1 --start-block 3   # resume

Output (data/raw/), one pair per block:
    YYYY-MM-DD_subjNN_calibration_<block>.h5          EEG + marker channel
    YYYY-MM-DD_subjNN_calibration_<block>_notes.csv   note log
Dry runs use run "dry<block>" and may overwrite each other.
"""

import argparse

import numpy as np

import config
from acquisition.session import check_sync, record_block, recording_paths
from analysis.preprocess import load_recording
from audio.playback import midi_to_hz
from composer import markov
from composer.sample import intervals, random_key_shift, sample_melody

NOTE_INDEX_STRIDE = 1000  # note_index offset per block (a block has < 400 notes)
MELODY_ID_STRIDE = 100    # melody_id offset per block


def build_block(model, block, duration, rng):
    """Generate melodies until the block lasts `duration` seconds.

    Returns (freqs, notes): Hz per IOI slot (None = silent gap) and one
    note-log row per sounding note (without note_index/onset_time).
    """
    temps = []
    freqs, notes = [], []
    k = 0
    while len(freqs) * config.IOI < duration:
        if not temps:  # refill: every temperature equally often, random order
            temps = list(rng.permutation(config.CALIBRATION_TEMPERATURES))
        temperature = float(temps.pop())
        k += 1
        pitches, bits = sample_melody(model, temperature=temperature, rng=rng)
        shift = random_key_shift(rng)
        played = [p + shift for p in pitches]
        for pitch, iv, s in zip(played, intervals(pitches), bits):
            notes.append({"melody_id": block * MELODY_ID_STRIDE + k, "pitch": pitch,
                          "interval": iv, "surprisal": s, "temperature": temperature})
        freqs += list(midi_to_hz(played)) + [None] * config.MELODY_GAP_SLOTS
    return freqs, notes


def summarize(notes):
    """Per-temperature surprisal summary of a recorded block."""
    print(f"\n{notes['melody_id'].nunique()} melodies, {len(notes)} notes")
    print(f"{'T':>5} {'notes':>6} {'mean bits':>10} {'range':>12}")
    for t, g in notes.groupby("temperature"):
        print(f"{t:>5} {len(g):>6} {g['surprisal'].mean():>10.2f} "
              f"{g['surprisal'].min():>5.1f}–{g['surprisal'].max():<5.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", type=int, help="subject number (required unless --dry-run)")
    ap.add_argument("--blocks", type=int, help=f"number of blocks (default "
                    f"{config.CALIBRATION_BLOCKS}; 1 with --dry-run)")
    ap.add_argument("--start-block", type=int, default=1, help="resume from this block")
    ap.add_argument("--dry-run", action="store_true", help="silent, short block to test the setup")
    ap.add_argument("--fake-eeg", action="store_true", help="signal generator instead of headset")
    ap.add_argument("--serial", default=config.SERIAL_NUMBER)
    ap.add_argument("--seed", type=int, help="random seed for melodies, temperatures and keys")
    ap.add_argument("--overwrite", action="store_true", help="replace existing block files")
    ap.add_argument("--no-pause", action="store_true", help="don't wait for Enter between blocks")
    args = ap.parse_args()

    if args.dry_run:
        subject = args.subject or 0
        n_blocks = args.blocks or 1
        duration = config.DRY_RUN_BLOCK_DURATION
    else:
        if args.subject is None:
            ap.error("--subject is required for a real recording")
        subject = args.subject
        n_blocks = args.blocks or config.CALIBRATION_BLOCKS
        duration = config.BLOCK_DURATION

    try:
        model = markov.load()
    except FileNotFoundError:
        ap.error("no trained model; run: python -m composer.markov --train")

    blocks = range(args.start_block, n_blocks + 1)
    paths = {b: recording_paths(subject, "calibration", f"dry{b}" if args.dry_run else str(b))
             for b in blocks}
    existing = [e.name for e, _ in paths.values() if e.exists()]
    if existing and not (args.dry_run or args.overwrite):
        ap.error(f"{', '.join(existing)} exist; use --start-block or --overwrite")
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    all_ok = True
    for b in blocks:
        eeg_path, notes_path = paths[b]
        freqs, notes = build_block(model, b, duration, rng)
        print(f"\n=== Block {b}/{n_blocks}: {len(notes)} notes, ~{len(freqs) * config.IOI / 60:.1f} min"
              f"{' [DRY RUN: silent]' if args.dry_run else ''}"
              f"{' [FAKE EEG]' if args.fake_eeg else ''} ===")
        log, spacing = record_block(eeg_path, notes_path, freqs, notes,
                                    first_index=(b - 1) * NOTE_INDEX_STRIDE + 1,
                                    silent=args.dry_run, fake_eeg=args.fake_eeg,
                                    serial=args.serial)
        if log.empty:
            print("no notes were played; stopping")
            return
        all_ok &= check_sync(load_recording(eeg_path), log, spacing)
        summarize(log)
        if len(log) < len(notes):
            print(f"\nblock {b} was cut short; resume with --start-block {b} --overwrite")
            return
        if b < n_blocks and not args.no_pause:
            input(f"\nBreak. Press Enter to start block {b + 1} (Ctrl+C to stop) ")

    print("\ncalibration done; all sync checks passed" if all_ok
          else "\ncalibration done, but a sync check FAILED: see above")


if __name__ == "__main__":
    main()
