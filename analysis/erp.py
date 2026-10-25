"""ERPs around note onsets.

Oddball: standard vs. deviant tones at Fz/Cz, plus the difference wave
(deviant − standard). A working setup shows a clear auditory response
(N1 ~100 ms, P2 ~200 ms) to every tone, and a more negative deviant
response ~100–250 ms (mismatch negativity, MMN). Expect it to be noisy
with few deviants; this is a sanity check, not a result.

Usage (from the repo root):
    python -m analysis.erp --session oddball --subject 1 --run 1
    python -m analysis.erp --eeg data/raw/<file>.h5
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

import config
from analysis.preprocess import load_recording, marker_events
from audio.playback import hz_to_midi

STANDARD, DEVIANT = 1, 2

# Categorical slots 1-2 for the two conditions; neutral ink for the difference
COLOR_STANDARD = "#2a78d6"
COLOR_DEVIANT = "#eb6834"
COLOR_DIFF = "#52514e"
COLOR_GUIDE = "#c3c2b7"
MMN_WINDOW = (0.10, 0.25)  # s, shaded on the plot as a guide


def oddball_epochs(raw, notes):
    """Epochs labelled 'standard'/'deviant' from markers + note log pitches."""
    std_pitch = int(round(float(hz_to_midi(config.ODDBALL_STANDARD_HZ))))
    pitch_of = dict(zip(notes["note_index"].astype(int), notes["pitch"].astype(int)))

    events = marker_events(raw)
    known = np.array([v in pitch_of for v in events[:, 2]], dtype=bool)
    if not known.all():
        print(f"warning: {(~known).sum()} markers have no row in the note log; ignored")
    events = events[known].copy()
    events[:, 2] = [STANDARD if pitch_of[v] == std_pitch else DEVIANT for v in events[:, 2]]

    filtered = raw.copy().filter(*config.ERP_BAND, picks="eeg", verbose=False)
    return mne.Epochs(filtered, events, event_id={"standard": STANDARD, "deviant": DEVIANT},
                      tmin=config.ERP_TMIN, tmax=config.ERP_TMAX,
                      baseline=config.ERP_BASELINE, reject=config.ERP_REJECT,
                      picks="eeg", preload=True, on_missing="warn", verbose=False)


def plot_oddball_erp(epochs, title="", out_path=None, show=True):
    """Standard, deviant and difference waves at the central channels."""
    channels = [c for c in config.CENTRAL_CHANNELS if c in epochs.ch_names]
    n_std, n_dev = len(epochs["standard"]), len(epochs["deviant"])
    t_ms = epochs.times * 1000

    fig, axes = plt.subplots(1, len(channels), figsize=(5 * len(channels), 4),
                             sharey=True, squeeze=False)
    for ax, ch in zip(axes[0], channels):
        ax.axvspan(*(np.array(MMN_WINDOW) * 1000), color=COLOR_GUIDE, alpha=0.25, lw=0)
        ax.axhline(0, color=COLOR_GUIDE, lw=1)
        ax.axvline(0, color=COLOR_GUIDE, lw=1)
        std = dev = None
        if n_std:
            std = epochs["standard"].average(picks=ch).data[0] * 1e6
            ax.plot(t_ms, std, color=COLOR_STANDARD, lw=2, label=f"standard (n={n_std})")
        if n_dev:
            dev = epochs["deviant"].average(picks=ch).data[0] * 1e6
            ax.plot(t_ms, dev, color=COLOR_DEVIANT, lw=2, label=f"deviant (n={n_dev})")
        if std is not None and dev is not None:
            ax.plot(t_ms, dev - std, color=COLOR_DIFF, lw=2, ls="--",
                    label="deviant − standard")
        ax.set_title(ch)
        ax.set_xlabel("time from tone onset (ms)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(np.mean(MMN_WINDOW) * 1000, 1.0, "MMN window", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color="#6b6a64")
    axes[0][0].set_ylabel("amplitude (µV, positive up)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9)
    fig.suptitle(title or "Oddball ERP")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"saved {out_path}")
    if show:
        plt.show()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eeg", type=Path, help="recording .h5 (notes CSV found beside it)")
    ap.add_argument("--session", default="oddball", choices=config.SESSIONS)
    ap.add_argument("--subject", type=int)
    ap.add_argument("--run", default="1")
    ap.add_argument("--date", help="YYYY-MM-DD (default today)")
    ap.add_argument("--no-show", action="store_true", help="save the figure without opening it")
    args = ap.parse_args()

    if args.session != "oddball":
        raise SystemExit("only --session oddball is implemented so far")
    if args.eeg:
        eeg_path = args.eeg
        notes_path = eeg_path.with_name(eeg_path.stem + "_notes.csv")
    else:
        if args.subject is None:
            raise SystemExit("give --eeg or --subject")
        from acquisition.session import recording_paths
        eeg_path, notes_path = recording_paths(args.subject, args.session, args.run, args.date)

    raw = load_recording(eeg_path)
    notes = pd.read_csv(notes_path)
    epochs = oddball_epochs(raw, notes)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_oddball_erp(epochs, title=eeg_path.stem,
                     out_path=config.FIGURES_DIR / f"{eeg_path.stem}_erp.png",
                     show=not args.no_show)


if __name__ == "__main__":
    main()
