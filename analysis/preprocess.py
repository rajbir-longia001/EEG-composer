"""Load g.Pype recordings into MNE and prepare them for ERPs and TRFs.

Channels are picked by role, not position: "signal" channels become EEG
(scaled to volts by their recorded unit) and "trigger" channels become a
stim channel named "marker". Anything else (e.g. BCICore's timeline
channels) is dropped.

Filtering:
  * ERPs: MNE zero-phase 1–30 Hz (offline only).
  * TRFs: causal Butterworth 1–8 Hz + keep every TRF_DECIM-th sample. The
    same causal pipeline is used for the offline comparison, the template
    and the live loop (CLAUDE.md invariant 7), so their timing matches.

Usage (from the repo root):
    python -m analysis.preprocess data/raw/<file>.h5     # summary of one recording
"""

import argparse
import glob
from pathlib import Path

import gpype as gp
import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt

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


def find_session_files(subject, session, date=None, dry=False):
    """(eeg_path, notes_path) for every block of a session, in run order.

    dry=False skips dry runs; dry=True returns only dry runs.
    """
    pattern = f"{date or '*'}_subj{int(subject):02d}_{session}_*{config.EEG_EXT}"
    pairs = []
    for eeg in sorted(glob.glob(str(config.DATA_RAW / pattern))):
        eeg = Path(eeg)
        run = eeg.stem.split(f"_{session}_", 1)[1]
        if run.startswith("dry") != dry:
            continue
        notes = eeg.with_name(eeg.stem + "_notes.csv")
        if notes.exists():
            pairs.append((eeg, notes))
    return pairs


def load_session(subject, session, date=None, dry=False):
    """List of (raw, notes DataFrame), one per block."""
    pairs = find_session_files(subject, session, date, dry)
    if not pairs:
        raise FileNotFoundError(f"no {session} recordings for subject {subject} in {config.DATA_RAW}"
                                f"{' (dry runs)' if dry else ''}")
    print(f"loading {len(pairs)} {session} block(s): {', '.join(e.name for e, _ in pairs)}")
    return [(load_recording(e), pd.read_csv(n)) for e, n in pairs]


def find_bad_channels(raw):
    """EEG channels that look flat or much noisier than the rest (1–30 Hz)."""
    filtered = raw.copy().filter(*config.ERP_BAND, picks="eeg", verbose=False)
    std_uv = filtered.get_data(picks="eeg").std(axis=1) / config.UV_TO_V
    names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
    med = np.median(std_uv)
    return [n for n, s in zip(names, std_uv)
            if s < config.BAD_FLAT_UV or s > config.BAD_NOISY_FACTOR * med]


def mark_bads(blocks):
    """Mark the same bad channels (union over blocks) in every block; returns them."""
    bads = sorted({ch for raw, _ in blocks for ch in find_bad_channels(raw)})
    for raw, _ in blocks:
        raw.info["bads"] = list(bads)
    if bads:
        print(f"bad channels (excluded): {', '.join(bads)}")
    return bads


def erp_filter(raw):
    """Zero-phase ERP band filter (offline only)."""
    return raw.copy().filter(*config.ERP_BAND, picks="eeg", verbose=False)


def causal_bandpass(data, sfreq, band=config.TRF_BAND, order=config.CAUSAL_FILTER_ORDER):
    """Causal Butterworth bandpass along the last axis (same filter as the live loop)."""
    sos = butter(order, band, btype="bandpass", fs=sfreq, output="sos")
    return sosfilt(sos, data, axis=-1)


def trf_preprocess(raw):
    """Good EEG channels, causal TRF band, decimated.

    Returns (eeg, sfreq, ch_names, good): eeg in µV with shape
    (n_samples, n_channels) at sfreq = raw sfreq / TRF_DECIM, and a boolean
    mask that is False within ARTIFACT_PAD of any sample over ERP_REJECT.
    """
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    ch_names = [raw.ch_names[i] for i in picks]
    data = causal_bandpass(raw.get_data(picks=picks), raw.info["sfreq"])
    data = data[:, ::config.TRF_DECIM].T / config.UV_TO_V
    sfreq = raw.info["sfreq"] / config.TRF_DECIM

    over = np.abs(data).max(axis=1) > config.ERP_REJECT["eeg"] / config.UV_TO_V
    pad = int(round(config.ARTIFACT_PAD * sfreq))
    good = ~np.convolve(over, np.ones(2 * pad + 1), mode="same").astype(bool)
    return data, sfreq, ch_names, good


def main():
    ap = argparse.ArgumentParser(description="Print a summary of a recording.")
    ap.add_argument("eeg_path")
    args = ap.parse_args()
    raw = load_recording(args.eeg_path)
    events = marker_events(raw)
    print(raw)
    print(f"{len(events)} markers, values {events[:, 2].min() if len(events) else '-'}"
          f"..{events[:, 2].max() if len(events) else '-'}")
    bads = find_bad_channels(raw)
    print(f"bad channels: {', '.join(bads) if bads else 'none'}")


if __name__ == "__main__":
    main()
