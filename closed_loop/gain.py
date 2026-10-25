"""Surprise response gain: how strongly the EEG follows surprisal right now.

The calibrated template (analysis/trf.py --save-template) is held fixed;
only its strength is estimated per window (CLAUDE.md "Closed-loop gain"):

  1. residual = EEG − prediction from the acoustic features (onset, pitch,
     |interval|, melody start) using the template's weights
  2. predicted surprise = surprisal impulses at note onsets convolved with
     the template's surprisal TRF
  3. gain = dot(predicted surprise, residual) / dot(predicted surprise, itself)

gain ≈ 1: responding as strongly as in calibration; gain ≈ 0: surprising
notes aren't standing out. One window's gain is noisy; the loop averages
GAIN_SMOOTHING_WINDOWS of them before each temperature decision.

Usage (from the repo root):
    python -m closed_loop.gain --simulate      # check: gain tracks a known effect size
    python -m closed_loop.gain --subject 1     # gain per window on calibration data (≈ 1)
"""

import argparse

import numpy as np
import pandas as pd

import config
from analysis.preprocess import load_session, marker_events, trf_preprocess
from analysis.trf import FEATURES_A, FEATURES_AM, design_matrix, note_features


def load_template(path):
    """Template saved by analysis/trf.py, with stats and channel names unpacked."""
    d = np.load(path)
    t = {k: d[k] for k in d.files}
    t["ch_names"] = [str(c) for c in t["ch_names"]]
    t["stats"] = dict(zip(("pitch", "abs_interval", "surprisal"),
                          [tuple(row) for row in t["stats"]]))
    return t


def predictions(n_samples, onsets, rows, template):
    """(acoustic, surprise) predicted EEG, each (n_samples, n_channels) in µV.

    onsets: sample index (at the template's rate) of each note in rows.
    """
    lags = template["lags"]
    X = design_matrix(n_samples, onsets, note_features(rows, template["stats"]).to_numpy(), lags)
    w = template["weights"].reshape(len(FEATURES_AM) * len(lags), -1)
    n_a = len(FEATURES_A) * len(lags)
    acoustic = X[:, :n_a] @ w[:n_a] + template["intercept"]
    surprise = X[:, n_a:] @ w[n_a:]
    return acoustic, surprise


def window_gain(eeg, acoustic, surprise, mask):
    """Least-squares strength of the surprise prediction in the residual."""
    residual = (eeg - acoustic)[mask]
    s = surprise[mask]
    denom = float((s * s).sum())
    return float((s * residual).sum()) / denom if denom > 0 else float("nan")


def next_temperature(temperature, direction, smoothed, prev_smoothed):
    """Trial-and-error step: keep going if the smoothed gain rose, reverse if
    it fell; bounce off TEMPERATURE_BOUNDS. Returns (new T, new direction)."""
    if prev_smoothed is not None and smoothed < prev_smoothed:
        direction = -direction
    lo, hi = config.TEMPERATURE_BOUNDS
    new = temperature + direction * config.TEMPERATURE_STEP
    if not lo <= new <= hi:
        direction = -direction
        new = temperature + direction * config.TEMPERATURE_STEP
    return round(min(max(new, lo), hi), 3), direction


def block_window_gains(raw, notes, template, window=config.WINDOW_DURATION):
    """Gain for consecutive windows of one recorded block (offline check)."""
    raw = raw.copy()
    raw.info["bads"] = [c for c in raw.ch_names if c not in template["ch_names"] + ["marker"]]
    eeg, sfreq, ch_names, good = trf_preprocess(raw)
    if ch_names != template["ch_names"]:
        raise ValueError(f"channels {ch_names} don't match the template's {template['ch_names']}")

    rows = notes.set_index("note_index")
    events = marker_events(raw)
    events = events[np.isin(events[:, 2], rows.index)]
    rows = rows.loc[events[:, 2]].reset_index()
    onsets = np.round(events[:, 0] / config.TRF_DECIM).astype(int)
    acoustic, surprise = predictions(len(eeg), onsets, rows, template)

    lags = template["lags"]
    start, stop = onsets[0] + lags[0], onsets[-1] + lags[-1]
    step = int(round(window * sfreq))
    out = []
    for w0 in range(start, stop, step):
        span = np.zeros(len(eeg), dtype=bool)
        span[max(w0, 0):min(w0 + step, len(eeg))] = True
        in_win = (onsets >= w0) & (onsets < w0 + step)
        if in_win.sum() < 10:
            continue
        out.append({"gain": window_gain(eeg, acoustic, surprise, span & good),
                    "n_notes": int(in_win.sum()),
                    "temperature": float(rows.loc[in_win, "temperature"].mean())})
    return out


def session_gains(blocks, template):
    rows = []
    for b, (raw, notes) in enumerate(blocks, start=1):
        for w, g in enumerate(block_window_gains(raw, notes, template), start=1):
            rows.append({"block": b, "window": w, **g})
    return pd.DataFrame(rows)


def simulate_check():
    """Template from one simulated session (effect 1.5 µV), gains on fresh
    sessions with effects 0–3 µV: the mean gain should be ≈ effect / 1.5."""
    from analysis.trf import choose_alpha, fit_template, session_dataset, simulate_session
    calib = simulate_session(effect_uv=1.5, seed=0)
    X, y, groups, sfreq, ch_names, stats = session_dataset(calib)
    weights, intercept, alpha = fit_template(X, y, choose_alpha(X, y, groups))
    from analysis.trf import lag_samples
    template = {"weights": weights, "intercept": intercept, "lags": lag_samples(sfreq),
                "ch_names": ch_names, "stats": stats}

    print(f"\n{'true effect':>12} {'expected gain':>14} {'mean gain':>10} {'SD/window':>10} {'windows':>8}")
    for i, effect in enumerate((0.0, 0.75, 1.5, 3.0)):
        gains = session_gains(simulate_session(n_blocks=2, effect_uv=effect, seed=100 + i), template)
        print(f"{effect:>10.2f}µV {effect / 1.5:>14.2f} {gains['gain'].mean():>10.2f} "
              f"{gains['gain'].std():>10.2f} {len(gains):>8}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simulate", action="store_true", help="check on synthetic sessions")
    ap.add_argument("--subject", type=int)
    ap.add_argument("--template", help="template .npz (default: the subject's calibration template)")
    ap.add_argument("--session", default="calibration", choices=["calibration", "live"])
    ap.add_argument("--date")
    ap.add_argument("--dry", action="store_true", help="use dry-run recordings")
    args = ap.parse_args()

    if args.simulate:
        simulate_check()
        return
    if args.subject is None:
        ap.error("give --subject (or --simulate)")
    path = args.template or config.MODELS_DIR / f"subj{args.subject:02d}_calibration_template.npz"
    gains = session_gains(load_session(args.subject, args.session, args.date, args.dry),
                          load_template(path))
    print(gains.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nmean gain {gains['gain'].mean():.2f} ± {gains['gain'].std():.2f} (SD across windows)")


if __name__ == "__main__":
    main()
