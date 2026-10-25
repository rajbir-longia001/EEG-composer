"""Load g.Pype recordings into MNE.

Channels are picked by role, not position: "signal" channels become EEG
(scaled to volts by their recorded unit) and "trigger" channels become a
stim channel named "marker". Anything else (e.g. BCICore's timeline
channels) is dropped.
"""

import argparse

import gpype as gp
import mne
import numpy as np

import config

MARKER_CH = "marker"


def load_recording(eeg_path):
    """Read an HDF5 recording into an mne.io.RawArray (EEG in volts)."""
    with gp.Pipeline() as p:
        reader = gp.HDF5Reader(file_name=str(eeg_path), mode="batch",
                               variable_name="data", has_time_row=True)
        p.connect(reader, gp.Collector())
        result = p.run()

    roles = [c.role for c in result.channels]
    eeg_idx = [i for i, r in enumerate(roles) if r == "signal"]
    trig_idx = [i for i, r in enumerate(roles) if r == "trigger"]
    if not eeg_idx or len(trig_idx) != 1:
        raise ValueError(f"expected signal channels and one trigger channel, got roles {roles}")

    labels = result.labels
    if labels is None or any(labels[i] is None for i in eeg_idx):
        names = list(config.CH_NAMES)  # fall back to the Core-8 montage
    else:
        names = [labels[i] for i in eeg_idx]

    scales = [result.channels[i].si_scale for i in eeg_idx]
    if any(s is None for s in scales):
        raise ValueError("recording has no units for its EEG channels; record to HDF5")

    eeg = result.data[:, eeg_idx].T.astype(float) * np.asarray(scales)[:, None]
    stim = result.data[:, trig_idx].T.astype(float)
    info = mne.create_info(names + [MARKER_CH], result.rate,
                           ["eeg"] * len(names) + ["stim"])
    raw = mne.io.RawArray(np.vstack([eeg, stim]), info, verbose=False)
    raw.set_montage(config.MONTAGE, on_missing="warn")
    return raw


def marker_events(raw):
    """MNE-style events array (sample, 0, note_index) from the marker channel."""
    m = raw.get_data(picks=MARKER_CH)[0]
    idx = np.flatnonzero(m)
    return np.column_stack([idx, np.zeros_like(idx), m[idx].round().astype(int)])


def main():
    ap = argparse.ArgumentParser(description="Print a summary of a recording.")
    ap.add_argument("eeg_path")
    args = ap.parse_args()
    raw = load_recording(args.eeg_path)
    events = marker_events(raw)
    print(raw)
    print(f"{len(events)} markers, values {events[:, 2].min() if len(events) else '-'}"
          f"..{events[:, 2].max() if len(events) else '-'}")


if __name__ == "__main__":
    main()
