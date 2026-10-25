"""Live closed loop: play melodies, estimate the surprise response gain,
adjust temperature.

Each window (~WINDOW_DURATION s) plays melodies at the current
temperature while EEG is recorded. Afterwards the window's EEG goes
through the same causal filter and decimation as the template
(analysis/preprocess.trf_arrays), and closed_loop/gain.py estimates the
gain. Every GAIN_SMOOTHING_WINDOWS windows, the mean gain at the current
temperature is compared with the previous one: if it rose, T keeps moving
the same way; if it fell, it reverses (closed_loop/gain.next_temperature).

Everything is also recorded like any other session (EEG .h5 + note log),
and a per-window loop log and figure are saved.

Usage (from the repo root; close view_stream first):
    python -m closed_loop.run_live --dry-run --fake-eeg --template models/simulated_effect1.5_template.npz
    python -m closed_loop.run_live --subject 1                  # 15 min, subject's template
    python -m closed_loop.run_live --subject 1 --minutes 20 --start-temperature 1.0

Output: data/raw/YYYY-MM-DD_subjNN_live_<run>.h5, _notes.csv, _loop.csv;
figures/..._live_<run>_loop.png. Dry runs use run "dry".
"""

import argparse
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gpype as gp

import config
from acquisition.record import MELODY_ID_STRIDE, build_block
from acquisition.session import build_pipeline, finalize_eeg_file, recording_paths, write_note_log
from analysis.preprocess import trf_arrays
from audio.playback import play_with_onsets, render_sequence
from closed_loop.gain import load_template, next_temperature, predictions, window_gain
from composer import markov

KEYS = gp.Constants.Keys
ROLES = gp.Constants.ChannelRoles
PORT_IN = gp.Constants.Defaults.PORT_IN

COLOR_T = "#2a78d6"
COLOR_GAIN = "#eb6834"
COLOR_RAW = "#c3c2b7"


class LiveBuffer(gp.Sink):
    """Keeps every frame of the recording stream in memory for the loop.

    step() runs on g.Pype's thread, snapshot() on ours, hence the lock.
    """

    def __init__(self, **kwargs):
        input_ports = kwargs.pop(KEYS.INPUT_PORTS, [gp.IPort.Configuration()])
        super().__init__(input_ports=input_ports, **kwargs)
        self._lock = threading.Lock()
        self._frames = []
        self.sfreq = self.eeg_cols = self.marker_col = self.labels = None

    def setup(self, data, port_context_in):
        ctx = port_context_in[PORT_IN]
        roles = ctx.get(KEYS.CHANNEL_ROLES) or []
        self.sfreq = float(ctx[KEYS.SAMPLING_RATE])
        self.eeg_cols = [i for i, r in enumerate(roles) if r == ROLES.SIGNAL]
        triggers = [i for i, r in enumerate(roles) if r == ROLES.TRIGGER]
        if not self.eeg_cols or not triggers:
            raise ValueError(f"live buffer needs signal and trigger channels, got roles {roles}")
        self.marker_col = triggers[-1]
        labels = ctx.get(KEYS.CHANNEL_LABELS)
        if labels and all(labels[i] for i in self.eeg_cols):
            self.labels = [labels[i] for i in self.eeg_cols]
        else:
            self.labels = list(config.CH_NAMES[:len(self.eeg_cols)])
        return super().setup(data, port_context_in)

    def step(self, data):
        frame = np.array(data[PORT_IN], dtype=float, copy=True)
        with self._lock:
            self._frames.append(frame)
        return {}

    def snapshot(self):
        with self._lock:
            return np.vstack(self._frames) if self._frames else np.empty((0, 0))


def live_gain(buffer, rows, template):
    """Gain for the notes in rows, from everything recorded so far."""
    data = buffer.snapshot()
    if data.size == 0:
        return float("nan"), 0
    cols = [buffer.eeg_cols[buffer.labels.index(c)] for c in template["ch_names"]]
    eeg, _, good = trf_arrays(data[:, cols].T, buffer.sfreq)  # causal: past data only

    marker = data[:, buffer.marker_col]
    idx = np.flatnonzero(marker)
    values = marker[idx].round().astype(int)
    rows = pd.DataFrame(rows).set_index("note_index")
    found = np.isin(values, rows.index)
    if found.sum() < 10:
        return float("nan"), int(found.sum())
    rows = rows.loc[values[found]].reset_index()
    onsets = np.round(idx[found] / config.TRF_DECIM).astype(int)

    acoustic, surprise = predictions(len(eeg), onsets, rows, template)
    lags = template["lags"]
    span = np.zeros(len(eeg), dtype=bool)
    span[max(onsets[0] + lags[0], 0):onsets[-1] + lags[-1] + 1] = True
    return window_gain(eeg, acoustic, surprise, span & good), int(found.sum())


def plot_loop(log, title, out_path=None, show=True):
    """Temperature and gain over the session."""
    t = log["end_min"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.step(t, log["temperature"], where="post", color=COLOR_T, lw=2)
    ax1.set_ylabel("temperature")
    ax1.set_title("Sampling temperature")
    ax2.axhline(0, color=COLOR_RAW, lw=1)
    ax2.axhline(1, color=COLOR_RAW, lw=1, ls=":")
    ax2.scatter(t, log["gain"], s=64, color=COLOR_RAW, label="per window", zorder=2)
    ax2.plot(t, log["smoothed"], color=COLOR_GAIN, lw=2, label="smoothed", zorder=3)
    ax2.set_ylabel("surprise response gain")
    ax2.set_xlabel("time (min)")
    ax2.set_title("Surprise response gain (1 = as strong as in calibration)")
    ax2.legend(frameon=False, fontsize=9)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"saved {out_path}")
    if show:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", type=int, help="subject number (required unless --dry-run)")
    ap.add_argument("--run", default="1")
    ap.add_argument("--template", help="template .npz (default: the subject's calibration template)")
    ap.add_argument("--minutes", type=float, help="session length (default 15; 3 with --dry-run)")
    ap.add_argument("--window", type=float, default=config.WINDOW_DURATION, help="seconds per window")
    ap.add_argument("--start-temperature", type=float, default=config.DEFAULT_TEMPERATURE)
    ap.add_argument("--dry-run", action="store_true", help="silent run to test the setup")
    ap.add_argument("--fake-eeg", action="store_true", help="signal generator instead of headset")
    ap.add_argument("--serial", default=config.SERIAL_NUMBER)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    if not args.dry_run and args.subject is None:
        ap.error("--subject is required for a real session")
    subject = args.subject if args.subject is not None else 0
    run = "dry" if args.dry_run else args.run
    minutes = args.minutes or (3 if args.dry_run else 15)
    template_path = args.template or config.MODELS_DIR / f"subj{subject:02d}_calibration_template.npz"
    try:
        template = load_template(template_path)
    except FileNotFoundError:
        ap.error(f"no template at {template_path}; run analysis.trf --save-template (or pass --template)")
    model = markov.load()

    eeg_path, notes_path = recording_paths(subject, "live", run)
    loop_path = eeg_path.with_name(eeg_path.stem + "_loop.csv")
    if eeg_path.exists() and not (args.dry_run or args.overwrite):
        ap.error(f"{eeg_path.name} exists; pick another --run or pass --overwrite")
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    buffer = LiveBuffer()
    p, marker = build_pipeline(eeg_path, serial=args.serial, fake_eeg=args.fake_eeg,
                               extra_sinks=[buffer])
    print(f"closed loop: {minutes:g} min, {args.window:g} s windows, decision every "
          f"{config.GAIN_SMOOTHING_WINDOWS} windows, start T = {args.start_temperature}"
          f"{' [DRY RUN: silent]' if args.dry_run else ''}{' [FAKE EEG]' if args.fake_eeg else ''}")

    temperature, direction, prev_smoothed = args.start_temperature, +1, None
    all_rows, loop, recent = [], [], []
    next_index = config.FIRST_NOTE_INDEX
    t0 = None
    try:
        p.start()
        time.sleep(config.RECORD_SETTLE)
        t0 = time.time()
        window = 0
        while time.time() - t0 < minutes * 60:
            window += 1
            freqs, notes = build_block(model, window, args.window, rng, temperature=temperature)
            audio, onsets = render_sequence(freqs, silent=args.dry_run)
            rows = []

            def on_onset(k, first=next_index, notes=notes, rows=rows):
                i = first + k
                marker.emit(i)
                rows.append({**notes[k], "note_index": i, "onset_time": time.time()})

            try:
                play_with_onsets(audio, onsets, on_onset)
            finally:
                all_rows += rows
                write_note_log(all_rows, notes_path)  # saved after every window
            next_index += len(notes)
            time.sleep(config.TRF_TMAX + 0.3)  # responses to the last notes

            gain, n_used = live_gain(buffer, rows, template)
            recent.append(gain)
            smoothed = float(np.nanmean(recent)) if np.isfinite(recent).any() else float("nan")
            entry = {"window": window, "end_min": (time.time() - t0) / 60,
                     "temperature": temperature, "gain": gain, "smoothed": smoothed,
                     "n_notes": n_used, "decision": ""}
            if len(recent) == config.GAIN_SMOOTHING_WINDOWS:
                new_t, direction = next_temperature(temperature, direction, smoothed, prev_smoothed)
                trend = "first" if prev_smoothed is None else ("rose" if smoothed >= prev_smoothed else "fell")
                entry["decision"] = f"{trend}: T {temperature} -> {new_t}"
                prev_smoothed, temperature, recent = smoothed, new_t, []
            loop.append(entry)
            pd.DataFrame(loop).to_csv(loop_path, index=False)
            print(f"  window {window:>2} | T {entry['temperature']:<5} | gain {gain:+.2f} "
                  f"(smoothed {smoothed:+.2f}, {n_used} notes) {entry['decision']}")
    except KeyboardInterrupt:
        print("\nstopped; saving what was recorded")
    finally:
        p.stop()
        p.close()
        write_note_log(all_rows, notes_path)
        if all_rows:
            finalize_eeg_file(eeg_path)
    print(f"saved {eeg_path}\nsaved {notes_path}\nsaved {loop_path}")

    if loop:
        plot_loop(pd.DataFrame(loop), title=eeg_path.stem + (" (fake EEG)" if args.fake_eeg else ""),
                  out_path=config.FIGURES_DIR / f"{eeg_path.stem}_loop.png", show=not args.no_show)


if __name__ == "__main__":
    main()
