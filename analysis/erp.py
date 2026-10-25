"""ERPs around note onsets.

Oddball: standard vs. deviant tones at Fz/Cz, plus the difference wave
(deviant − standard). A working setup shows a clear auditory response
(N1 ~100 ms, P2 ~200 ms) to every tone, and a more negative deviant
response ~100–250 ms (mismatch negativity, MMN).

Calibration: notes split at the session's median surprisal, high vs. low.
Each melody's first note is left out (it follows a gap and a key change).
Caveat: high-surprisal notes tend to be bigger leaps, so this comparison
is confounded with interval size; it's a first look. The TRF comparison
(analysis/trf.py) is what controls for acoustics.

Usage (from the repo root):
    python -m analysis.erp --session oddball --subject 1 --run 1
    python -m analysis.erp --session calibration --subject 1
    python -m analysis.erp --eeg data/raw/<oddball file>.h5
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

import config
from analysis.preprocess import (erp_filter, load_recording, load_session, mark_bads,
                                 marker_events)
from audio.playback import hz_to_midi

STANDARD, DEVIANT = 1, 2
LOW, HIGH = 1, 2

# Categorical slots 1-2 for the two conditions; neutral ink for the difference
COLOR_A = "#2a78d6"
COLOR_B = "#eb6834"
COLOR_DIFF = "#52514e"
COLOR_GUIDE = "#c3c2b7"
RESPONSE_WINDOW = (0.10, 0.25)  # s, shaded on the plots as a guide


def _epochs(raw, events, event_id):
    return mne.Epochs(erp_filter(raw), events, event_id=event_id,
                      tmin=config.ERP_TMIN, tmax=config.ERP_TMAX,
                      baseline=config.ERP_BASELINE, reject=config.ERP_REJECT,
                      picks="eeg", preload=True, on_missing="warn", verbose=False)


def _recode(raw, notes, label_of):
    """Marker events with values replaced by label_of(note_row) (None = drop)."""
    rows = notes.set_index("note_index")
    events = marker_events(raw)
    keep = []
    for ev in events:
        row = rows.loc[ev[2]] if ev[2] in rows.index else None
        code = label_of(row) if row is not None else None
        if code is not None:
            keep.append([ev[0], 0, code])
    if len(keep) < len(events):
        print(f"  {len(events) - len(keep)} of {len(events)} markers not used")
    return np.array(keep, dtype=int).reshape(-1, 3)


def oddball_epochs(raw, notes):
    """Epochs labelled 'standard'/'deviant' from markers + note log pitches."""
    std_pitch = int(round(float(hz_to_midi(config.ODDBALL_STANDARD_HZ))))
    events = _recode(raw, notes, lambda r: STANDARD if int(r["pitch"]) == std_pitch else DEVIANT)
    return _epochs(raw, events, {"standard": STANDARD, "deviant": DEVIANT})


def calibration_epochs(blocks):
    """Epochs labelled 'low'/'high' surprisal (median split over the session)."""
    mark_bads(blocks)
    median = np.median(pd.concat([n for _, n in blocks])["surprisal"].dropna())
    print(f"median surprisal {median:.2f} bits")

    def label(row):
        if pd.isna(row["interval"]):  # first note of a melody
            return None
        return HIGH if row["surprisal"] >= median else LOW

    epochs = [_epochs(raw, _recode(raw, notes, label), {"low": LOW, "high": HIGH})
              for raw, notes in blocks]
    return mne.concatenate_epochs(epochs, verbose=False)


def plot_erp(epochs, cond_a, cond_b, label_a, label_b, title="", out_path=None, show=True):
    """Two conditions and their difference (b − a) at the central channels."""
    channels = [c for c in config.CENTRAL_CHANNELS if c in epochs.ch_names]
    n_a, n_b = len(epochs[cond_a]), len(epochs[cond_b])
    t_ms = epochs.times * 1000

    fig, axes = plt.subplots(1, len(channels), figsize=(5 * len(channels), 4),
                             sharey=True, squeeze=False)
    for ax, ch in zip(axes[0], channels):
        ax.axvspan(*(np.array(RESPONSE_WINDOW) * 1000), color=COLOR_GUIDE, alpha=0.25, lw=0)
        ax.axhline(0, color=COLOR_GUIDE, lw=1)
        ax.axvline(0, color=COLOR_GUIDE, lw=1)
        a = b = None
        if n_a:
            a = epochs[cond_a].average(picks=ch).data[0] * 1e6
            ax.plot(t_ms, a, color=COLOR_A, lw=2, label=f"{label_a} (n={n_a})")
        if n_b:
            b = epochs[cond_b].average(picks=ch).data[0] * 1e6
            ax.plot(t_ms, b, color=COLOR_B, lw=2, label=f"{label_b} (n={n_b})")
        if a is not None and b is not None:
            ax.plot(t_ms, b - a, color=COLOR_DIFF, lw=2, ls="--",
                    label=f"{label_b} − {label_a}")
        ax.set_title(ch)
        ax.set_xlabel("time from note onset (ms)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(np.mean(RESPONSE_WINDOW) * 1000, 1.0, "100–250 ms",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8, color="#6b6a64")
    axes[0][0].set_ylabel("amplitude (µV, positive up)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"saved {out_path}")
    if show:
        plt.show()
    return fig


def plot_oddball_erp(epochs, title="", out_path=None, show=True):
    return plot_erp(epochs, "standard", "deviant", "standard", "deviant",
                    title=title or "Oddball ERP", out_path=out_path, show=show)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eeg", type=Path, help="one oddball recording .h5 (notes CSV beside it)")
    ap.add_argument("--session", default="oddball", choices=["oddball", "calibration"])
    ap.add_argument("--subject", type=int)
    ap.add_argument("--run", default="1", help="oddball run")
    ap.add_argument("--date", help="YYYY-MM-DD (default: oddball today, calibration any)")
    ap.add_argument("--dry", action="store_true", help="use dry-run recordings (testing)")
    ap.add_argument("--no-show", action="store_true", help="save the figure without opening it")
    args = ap.parse_args()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.session == "calibration" and not args.eeg:
        if args.subject is None:
            raise SystemExit("give --subject")
        epochs = calibration_epochs(load_session(args.subject, "calibration", args.date, args.dry))
        name = f"subj{args.subject:02d}_calibration{'_dry' if args.dry else ''}"
        print(f"epochs kept: low {len(epochs['low'])}, high {len(epochs['high'])}")
        plot_erp(epochs, "low", "high", "low surprisal", "high surprisal",
                 title=f"{name}: high vs. low surprisal (confounded with interval size)",
                 out_path=config.FIGURES_DIR / f"{name}_erp.png", show=not args.no_show)
        return

    if args.eeg:
        eeg_path = args.eeg
        notes_path = eeg_path.with_name(eeg_path.stem + "_notes.csv")
    else:
        if args.subject is None:
            raise SystemExit("give --eeg or --subject")
        from acquisition.session import recording_paths
        eeg_path, notes_path = recording_paths(args.subject, "oddball", args.run, args.date)

    epochs = oddball_epochs(load_recording(eeg_path), pd.read_csv(notes_path))
    plot_oddball_erp(epochs, title=eeg_path.stem,
                     out_path=config.FIGURES_DIR / f"{eeg_path.stem}_erp.png",
                     show=not args.no_show)


if __name__ == "__main__":
    main()
