"""Auditory oddball: the sanity check before any real experiment.

Plays a stream of identical standard tones (1000 Hz) with rare, slightly
higher deviants (1200 Hz) while recording EEG, with a marker at every
tone onset. Then it checks that the markers line up with the tones and
plots the ERP to standards vs. deviants at Fz/Cz.

What a working setup looks like:
  * sync check: one marker per tone, in order, spaced exactly IOI apart
  * ERP: a clear response to every tone (N1 ~100 ms, P2 ~200 ms), and a
    more negative deviant response ~100–250 ms (mismatch negativity)

Modes:
  --dry-run   Silent: writes silence through the audio stream (same timing,
              nothing heard), 30 tones by default. Checks the audio device,
              headset, markers, file saving and analysis end to end.
  --fake-eeg  Use g.Pype's signal generator instead of the headset, so the
              script can be tested with no hardware. The ERP is then noise.

Usage (from the repo root; close view_stream first, the headset allows
one connection):
    python -m acquisition.oddball --dry-run --fake-eeg   # anywhere, no hardware
    python -m acquisition.oddball --dry-run               # headset, no sound
    python -m acquisition.oddball --subject 1 --run 1     # the real thing (~3 min)

Output (data/raw/, figures/):
    YYYY-MM-DD_subjNN_oddball_<run>.h5          EEG + marker channel
    YYYY-MM-DD_subjNN_oddball_<run>_notes.csv   note log (one row per tone)
    YYYY-MM-DD_subjNN_oddball_<run>_erp.png     ERP figure
Dry runs use run "dry" and may overwrite each other.
"""

import argparse
import math

import numpy as np

import config
from acquisition.session import check_sync, record_block, recording_paths
from analysis.erp import oddball_epochs, plot_oddball_erp
from analysis.preprocess import load_recording
from audio.playback import hz_to_midi

def make_sequence(n, p_deviant=config.ODDBALL_DEVIANT_PROB,
                  n_lead=config.ODDBALL_LEAD_STANDARDS, seed=None):
    """Boolean array, True = deviant. Starts with n_lead standards and never
    has two deviants in a row (each deviant needs a standard to deviate from)."""
    rng = np.random.default_rng(seed)
    seq = np.zeros(n, dtype=bool)
    for i in range(n_lead, n):
        seq[i] = (not seq[i - 1]) and rng.random() < p_deviant
    return seq


def oddball_notes(sequence, standard_hz, deviant_hz, block_id=1):
    """Note-log rows for the tone sequence: pitch = nearest MIDI number;
    surprisal, interval and temperature don't apply (NaN)."""
    pitch = {False: int(round(float(hz_to_midi(standard_hz)))),
             True: int(round(float(hz_to_midi(deviant_hz))))}
    return [{"melody_id": block_id, "pitch": pitch[bool(d)], "interval": math.nan,
             "surprisal": math.nan, "temperature": math.nan} for d in sequence]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", type=int, help="subject number (required unless --dry-run)")
    ap.add_argument("--run", default="1")
    ap.add_argument("--dry-run", action="store_true", help="silent run to test the setup")
    ap.add_argument("--fake-eeg", action="store_true", help="signal generator instead of headset")
    ap.add_argument("--n-tones", type=int, help=f"default {config.ODDBALL_N_TONES} "
                    f"({config.ODDBALL_DRY_RUN_TONES} with --dry-run)")
    ap.add_argument("--standard-hz", type=float, default=config.ODDBALL_STANDARD_HZ)
    ap.add_argument("--deviant-hz", type=float, default=config.ODDBALL_DEVIANT_HZ)
    ap.add_argument("--serial", default=config.SERIAL_NUMBER)
    ap.add_argument("--seed", type=int, help="random seed for the tone order")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing recording")
    ap.add_argument("--no-show", action="store_true", help="save the ERP figure without opening it")
    args = ap.parse_args()

    if args.dry_run:
        subject, run = (args.subject or 0), "dry"
        n_tones = args.n_tones or config.ODDBALL_DRY_RUN_TONES
    else:
        if args.subject is None:
            ap.error("--subject is required for a real recording")
        subject, run = args.subject, args.run
        n_tones = args.n_tones or config.ODDBALL_N_TONES

    eeg_path, notes_path = recording_paths(subject, "oddball", run)
    if eeg_path.exists() and not (args.dry_run or args.overwrite):
        ap.error(f"{eeg_path.name} exists; pick another --run or pass --overwrite")
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sequence = make_sequence(n_tones, seed=args.seed)
    print(f"{n_tones} tones ({sequence.sum()} deviants), IOI {config.IOI * 1000:.0f} ms, "
          f"~{n_tones * config.IOI / 60:.1f} min"
          f"{' [DRY RUN: silent]' if args.dry_run else ''}"
          f"{' [FAKE EEG]' if args.fake_eeg else ''}")

    freqs = [args.deviant_hz if d else args.standard_hz for d in sequence]
    notes, spacing = record_block(eeg_path, notes_path, freqs,
                                  oddball_notes(sequence, args.standard_hz, args.deviant_hz),
                                  silent=args.dry_run, fake_eeg=args.fake_eeg,
                                  serial=args.serial)
    if notes.empty:
        print("no tones were played; nothing to check")
        return

    raw = load_recording(eeg_path)
    sync_ok = check_sync(raw, notes, spacing)

    epochs = oddball_epochs(raw, notes)
    kept = {c: len(epochs[c]) for c in ("standard", "deviant")}
    played = {"standard": int((~sequence[:len(notes)]).sum()),
              "deviant": int(sequence[:len(notes)].sum())}
    print(f"\nEpochs kept after artifact rejection: "
          f"standard {kept['standard']}/{played['standard']}, "
          f"deviant {kept['deviant']}/{played['deviant']}")
    if kept["standard"] < 0.5 * played["standard"]:
        print("  many epochs rejected: check electrode contact with view_stream")

    plot_oddball_erp(epochs, title=eeg_path.stem + (" (fake EEG)" if args.fake_eeg else ""),
                     out_path=config.FIGURES_DIR / f"{eeg_path.stem}_erp.png",
                     show=not args.no_show)
    print("\nsetup OK" if sync_ok else "\nsync check FAILED: see above")


if __name__ == "__main__":
    main()
