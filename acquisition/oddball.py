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
import time

import numpy as np

import config
from acquisition.session import build_pipeline, finalize_eeg_file, recording_paths, write_note_log
from analysis.erp import oddball_epochs, plot_oddball_erp
from analysis.preprocess import load_recording, marker_events
from audio.playback import hz_to_midi, play_with_onsets, render_sequence

SYNC_MEAN_TOL = 0.005  # s, allowed error in the mean marker spacing
# s, allowed marker jitter: 1 sample at 250 Hz. Rounding onsets to 4 ms samples
# alone gives ~2 ms std when IOI isn't a whole number of samples.
SYNC_STD_TOL = 0.004


def make_sequence(n, p_deviant=config.ODDBALL_DEVIANT_PROB,
                  n_lead=config.ODDBALL_LEAD_STANDARDS, seed=None):
    """Boolean array, True = deviant. Starts with n_lead standards and never
    has two deviants in a row (each deviant needs a standard to deviate from)."""
    rng = np.random.default_rng(seed)
    seq = np.zeros(n, dtype=bool)
    for i in range(n_lead, n):
        seq[i] = (not seq[i - 1]) and rng.random() < p_deviant
    return seq


def record(eeg_path, notes_path, sequence, standard_hz, deviant_hz,
           silent=False, fake_eeg=False, serial=None, run_id=1):
    """Play the sequence while recording. Saves the EEG file and the note log
    (also on Ctrl+C or a crash). Returns the note log DataFrame."""
    p, marker = build_pipeline(eeg_path, serial=serial, fake_eeg=fake_eeg)
    rows = []
    try:
        p.start()
        print(f"recording; settling for {config.RECORD_SETTLE:.0f} s ...")
        time.sleep(config.RECORD_SETTLE)

        freqs = [deviant_hz if d else standard_hz for d in sequence]
        audio, onsets = render_sequence(freqs, silent=silent)
        pitches = {False: int(round(float(hz_to_midi(standard_hz)))),
                   True: int(round(float(hz_to_midi(deviant_hz))))}

        def on_onset(k):
            # Called as tone k reaches the speaker (see audio/playback.py)
            i = k + config.FIRST_NOTE_INDEX
            marker.emit(i)
            rows.append({"note_index": i, "melody_id": run_id,
                         "onset_time": time.time(), "pitch": pitches[bool(sequence[k])],
                         "interval": math.nan, "surprisal": math.nan,
                         "temperature": math.nan})
            if i % 25 == 0 or k == len(sequence) - 1:
                print(f"\r  tone {i}/{len(sequence)}", end="", flush=True)

        try:
            underflows = play_with_onsets(audio, onsets, on_onset)
            print()
            if underflows:
                print(f"warning: {underflows} audio underflows (glitches); close other programs")
        except KeyboardInterrupt:
            print("\nstopped early; saving what was recorded")
        time.sleep(config.RECORD_TAIL)  # let the last epoch finish recording
    finally:
        p.stop()
        p.close()
        notes = write_note_log(rows, notes_path)
        if rows:
            finalize_eeg_file(eeg_path)
    print(f"saved {eeg_path}\nsaved {notes_path}")
    return notes


def check_sync(raw, notes, ioi=config.IOI):
    """Compare recorded markers with the note log. Prints a report; returns
    True if every check passes."""
    events = marker_events(raw)
    sfreq = raw.info["sfreq"]
    expected = notes["note_index"].to_numpy(dtype=int)
    checks = []

    checks.append(("one marker per tone", len(events) == len(expected),
                   f"{len(events)} markers, {len(expected)} tones"))
    checks.append(("marker values match note log",
                   np.array_equal(events[:, 2], expected), "values = note_index, in order"))

    if len(events) > 2:
        spacing = np.diff(events[:, 0]) / sfreq
        mean_err = spacing.mean() - ioi
        checks.append(("marker spacing = IOI", abs(mean_err) < SYNC_MEAN_TOL,
                       f"mean {spacing.mean() * 1000:.1f} ms vs IOI {ioi * 1000:.0f} ms"))
        checks.append(("marker jitter small", spacing.std() < SYNC_STD_TOL,
                       f"std {spacing.std() * 1000:.1f} ms, "
                       f"worst {np.abs(spacing - ioi).max() * 1000:.1f} ms "
                       f"(1 sample = {1000 / sfreq:.1f} ms)"))
        # The host clock should agree; a mismatch means EEG samples were dropped
        host = np.diff(notes["onset_time"].to_numpy()[:len(events)])
        drift = (spacing[:len(host)] - host).sum()
        checks.append(("EEG clock agrees with host clock", abs(drift) < 0.05,
                       f"total difference {drift * 1000:.0f} ms over {spacing.sum():.0f} s"))

    print("\nSync check")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return all(ok for _, ok, _ in checks)


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

    notes = record(eeg_path, notes_path, sequence, args.standard_hz, args.deviant_hz,
                   silent=args.dry_run, fake_eeg=args.fake_eeg, serial=args.serial,
                   run_id=1)
    if notes.empty:
        print("no tones were played; nothing to check")
        return

    raw = load_recording(eeg_path)
    sync_ok = check_sync(raw, notes)

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
