"""Shared pieces of a recording session: file names, the g.Pype recording
pipeline, and the note log.

Recording pipeline (one per block):

    BCICore (or a fake Generator) --+
                                    +--> Router --> HDF5Writer
    Marker  ------------------------+

marker.emit(note_index) puts a one-sample pulse with value note_index on
the marker channel, on the same timeline as the EEG samples.
"""

import datetime as dt
import glob
import os
import time

import gpype as gp
import numpy as np
import pandas as pd

import config
from analysis.preprocess import marker_events
from audio.playback import play_with_onsets, render_sequence

SYNC_MEAN_TOL = 0.005  # s, allowed mean error in marker spacing
# s, allowed marker jitter: 1 sample at 250 Hz. Rounding onsets to 4 ms samples
# alone gives ~2 ms std when IOI isn't a whole number of samples.
SYNC_STD_TOL = 0.004


def recording_paths(subject, session, run, date=None):
    """(eeg_path, notes_path) following YYYY-MM-DD_subjNN_<session>_<run>."""
    if session not in config.SESSIONS:
        raise ValueError(f"session must be one of {config.SESSIONS}")
    date = date or dt.date.today().isoformat()
    stem = f"{date}_subj{int(subject):02d}_{session}_{run}"
    return (config.DATA_RAW / f"{stem}{config.EEG_EXT}",
            config.DATA_RAW / f"{stem}_notes.csv")


def build_pipeline(eeg_path, serial=None, fake_eeg=False, extra_sinks=()):
    """Build (but don't start) the recording pipeline.

    Returns (pipeline, marker). fake_eeg=True uses g.Pype's signal generator
    instead of the headset, so the whole session can be tested without
    hardware. extra_sinks get the same stream as the writer (EEG + marker),
    e.g. the closed loop's live buffer.
    """
    p = gp.Pipeline()
    marker = gp.Marker()
    both = gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL])
    p.connect(marker, both["in2"])

    if fake_eeg:
        source = gp.Generator(sampling_rate=250, channel_count=len(config.CH_NAMES),
                              signal_frequency=10, signal_amplitude=10,
                              noise_amplitude=10)
        p.connect(source, both["in1"])
        # A generator has no channel names or units; give it the Core-8's
        labeled = gp.ChannelLabeler(labels=config.CH_NAMES + ["marker"],
                                    units=["uV"] * len(config.CH_NAMES) + [None])
        p.connect(both, labeled)
        last = labeled
    else:
        source = gp.BCICore(serial=serial)
        p.connect(source, both["in1"])
        last = both

    # The writer appends _YYYYMMDD_HHMMSS_mmm to this name; see finalize_eeg_file
    p.connect(last, gp.HDF5Writer(file_name=str(eeg_path)))
    for sink in extra_sinks:
        p.connect(last, sink)
    return p, marker


def finalize_eeg_file(eeg_path):
    """Rename the writer's timestamped file to eeg_path. Returns eeg_path."""
    stem, ext = os.path.splitext(str(eeg_path))
    written = sorted(glob.glob(f"{glob.escape(stem)}_*{ext}"))
    if not written:
        raise FileNotFoundError(f"g.Pype wrote no file matching {stem}_*{ext}")
    os.replace(written[-1], eeg_path)
    return eeg_path


def write_note_log(rows, notes_path):
    """Save the note log with exactly config.NOTE_LOG_COLUMNS."""
    df = pd.DataFrame(rows, columns=config.NOTE_LOG_COLUMNS)
    df.to_csv(notes_path, index=False)
    return df


def record_block(eeg_path, notes_path, freqs, notes, first_index=config.FIRST_NOTE_INDEX,
                 silent=False, fake_eeg=False, serial=None):
    """Play one block while recording EEG, with a marker at every tone.

    freqs: Hz per IOI slot; None = silent slot (no tone, no marker).
    notes: one dict per sounding slot with melody_id, pitch, interval,
        surprisal, temperature. note_index (first_index, first_index + 1, ...)
        and onset_time are filled in here.

    Saves the EEG file and the note log, also on Ctrl+C or a crash.
    Returns (note log DataFrame, expected seconds between consecutive onsets).
    """
    audio, onsets = render_sequence(freqs, silent=silent)
    if len(onsets) != len(notes):
        raise ValueError(f"{len(onsets)} sounding slots but {len(notes)} note rows")

    p, marker = build_pipeline(eeg_path, serial=serial, fake_eeg=fake_eeg)
    rows = []
    try:
        p.start()
        print(f"recording; settling for {config.RECORD_SETTLE:.0f} s ...")
        time.sleep(config.RECORD_SETTLE)

        def on_onset(k):
            # Called as tone k reaches the speaker (see audio/playback.py)
            i = first_index + k
            marker.emit(i)
            rows.append({**notes[k], "note_index": i, "onset_time": time.time()})
            if (k + 1) % 25 == 0 or k == len(notes) - 1:
                print(f"\r  tone {k + 1}/{len(notes)}", end="", flush=True)

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
        df = write_note_log(rows, notes_path)
        if rows:
            finalize_eeg_file(eeg_path)
    print(f"saved {eeg_path}\nsaved {notes_path}")
    return df, np.diff(onsets) / config.AUDIO_SFREQ


def check_sync(raw, notes, expected_spacing):
    """Compare recorded markers with the note log and the audio schedule.

    expected_spacing: seconds between consecutive onsets as scheduled
    (from record_block). Prints a report; returns True if every check passes.
    """
    events = marker_events(raw)
    sfreq = raw.info["sfreq"]
    expected = notes["note_index"].to_numpy(dtype=int)
    checks = [("one marker per tone", len(events) == len(expected),
               f"{len(events)} markers, {len(expected)} tones"),
              ("marker values match note log", np.array_equal(events[:, 2], expected),
               "values = note_index, in order")]

    n = min(len(events), len(expected_spacing) + 1)
    if n > 2:
        spacing = np.diff(events[:n, 0]) / sfreq
        err = spacing - np.asarray(expected_spacing[:n - 1])
        checks.append(("marker spacing matches audio", abs(err.mean()) < SYNC_MEAN_TOL,
                       f"mean error {err.mean() * 1000:+.1f} ms"))
        checks.append(("marker jitter small", err.std() < SYNC_STD_TOL,
                       f"std {err.std() * 1000:.1f} ms, worst {np.abs(err).max() * 1000:.1f} ms "
                       f"(1 sample = {1000 / sfreq:.1f} ms)"))
        # The host clock should agree; a mismatch means EEG samples were dropped
        host = np.diff(notes["onset_time"].to_numpy()[:n])
        drift = (spacing - host).sum()
        checks.append(("EEG clock agrees with host clock", abs(drift) < 0.05,
                       f"total difference {drift * 1000:.0f} ms over {spacing.sum():.0f} s"))

    print("\nSync check")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return all(ok for _, ok, _ in checks)
