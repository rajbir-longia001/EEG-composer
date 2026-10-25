"""Temporal response functions: does surprisal predict EEG beyond acoustics?

Ridge regression from time-lagged note features to EEG (CLAUDE.md
invariants 5–7):

  Model A    onset, pitch height, |interval|, melody start  (acoustic controls)
  Model A+M  A + surprisal

Each feature is an impulse at the note onset scaled by the feature's
(z-scored) value, lagged TRF_TMIN..TRF_TMAX. EEG is causally filtered to
the TRF band and decimated (analysis/preprocess.trf_preprocess). Scores
are Pearson r between predicted and actual held-out EEG, per channel.

Cross-validation splits by melody_id (GroupKFold), and ridge alpha is
chosen inside each training fold (nested CV). The claim is "A+M beats A
on held-out melodies".

--save-template fits A+M on all data and saves the weights: the surprisal
TRF is the listener's surprise template for the closed loop.

--simulate runs everything on a synthetic session (composer melodies,
EEG with a known surprisal response + noise), to check the code finds an
effect that is there and recovers its shape. --simulate-effect 0 checks
that it finds none when there is none.

Usage (from the repo root):
    python -m analysis.trf --simulate                         # test the code
    python -m analysis.trf --subject 1                        # real data
    python -m analysis.trf --subject 1 --save-template
"""

import argparse

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

import config
from analysis.preprocess import load_session, mark_bads, marker_events, trf_preprocess

FEATURES_A = ["onset", "pitch", "abs_interval", "melody_start"]
FEATURES_AM = FEATURES_A + ["surprisal"]
INNER_FOLDS = 3

COLOR_A = "#8a8984"    # model A: neutral
COLOR_AM = "#2a78d6"   # model A+M
COLOR_CH = {"Fz": "#2a78d6", "Cz": "#eb6834"}


# --- Features and design matrix --------------------------------------------

def feature_stats(notes):
    """Means/stds used to z-score features (saved with the template)."""
    iv = notes["interval"].abs().dropna()
    return {"pitch": (notes["pitch"].mean(), notes["pitch"].std()),
            "abs_interval": (iv.mean(), iv.std()),
            "surprisal": (notes["surprisal"].mean(), notes["surprisal"].std())}


def note_features(notes, stats):
    """One row per note, one column per feature in FEATURES_AM."""
    def z(col, values):
        mean, std = stats[col]
        return (values - mean) / std
    first = notes["interval"].isna()
    return pd.DataFrame({
        "onset": 1.0,
        "pitch": z("pitch", notes["pitch"]),
        "abs_interval": z("abs_interval", notes["interval"].abs()).where(~first, 0.0),
        "melody_start": first.astype(float),
        "surprisal": z("surprisal", notes["surprisal"]),
    }, index=notes.index)[FEATURES_AM]


def lag_samples(sfreq):
    return np.arange(int(round(config.TRF_TMIN * sfreq)), int(round(config.TRF_TMAX * sfreq)) + 1)


def design_matrix(n_samples, onsets, values, lags):
    """Lagged impulse features: column f * n_lags + j is feature f delayed by lags[j]."""
    n_feat = values.shape[1]
    z = np.zeros((n_samples, n_feat))
    np.add.at(z, onsets, values)
    X = np.zeros((n_samples, n_feat * len(lags)))
    for j, lag in enumerate(lags):
        cols = np.arange(n_feat) * len(lags) + j
        if lag >= 0:
            X[lag:, cols] = z[:n_samples - lag]
        else:
            X[:lag, cols] = z[-lag:]
    return X


def block_dataset(raw, notes, stats):
    """(X with FEATURES_AM, y in µV, melody group per sample, sfreq, ch_names)."""
    y, sfreq, ch_names, good = trf_preprocess(raw)
    rows = notes.set_index("note_index")
    events = marker_events(raw)
    events = events[np.isin(events[:, 2], rows.index)]
    rows = rows.loc[events[:, 2]]
    onsets = np.round(events[:, 0] / config.TRF_DECIM).astype(int)
    lags = lag_samples(sfreq)
    X = design_matrix(len(y), onsets, note_features(rows, stats).to_numpy(), lags)

    # Each sample belongs to the melody whose notes it can respond to
    groups = np.full(len(y), -1)
    for mid in np.unique(rows["melody_id"]):
        idx = onsets[rows["melody_id"].to_numpy() == mid]
        groups[max(idx.min() + lags[0], 0):idx.max() + lags[-1] + 1] = mid
    keep = good & (groups >= 0)
    return X[keep], y[keep], groups[keep], sfreq, ch_names


def session_dataset(blocks):
    """Stack all blocks (lagging is done per block, so nothing crosses a boundary)."""
    stats = feature_stats(pd.concat([n for _, n in blocks]))
    parts = [block_dataset(raw, notes, stats) for raw, notes in blocks]
    X = np.vstack([p[0] for p in parts])
    y = np.vstack([p[1] for p in parts])
    groups = np.concatenate([p[2] for p in parts])
    return X, y, groups, parts[0][3], parts[0][4], stats


# --- Fitting and cross-validation -------------------------------------------

def pearson(pred, y):
    """Pearson r per channel."""
    a, b = pred - pred.mean(0), y - y.mean(0)
    return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))


def choose_alpha(X, y, groups):
    """Ridge alpha with the best mean r in inner GroupKFold (training data only)."""
    cv = GroupKFold(n_splits=min(INNER_FOLDS, len(np.unique(groups))))
    scores = []
    for alpha in config.RIDGE_ALPHAS:
        r = [pearson(Ridge(alpha=alpha).fit(X[tr], y[tr]).predict(X[te]), y[te]).mean()
             for tr, te in cv.split(X, y, groups)]
        scores.append(np.mean(r))
    return config.RIDGE_ALPHAS[int(np.argmax(scores))]


def cross_validate(X, y, groups):
    """Nested CV of A vs. A+M. Returns one dict per outer fold."""
    n_a = len(FEATURES_A) * (X.shape[1] // len(FEATURES_AM))
    outer = GroupKFold(n_splits=min(config.CV_FOLDS, len(np.unique(groups))))
    folds = []
    for k, (tr, te) in enumerate(outer.split(X, y, groups), start=1):
        fold = {}
        for name, cols in (("A", slice(0, n_a)), ("A+M", slice(None))):
            Xtr, Xte = X[tr][:, cols], X[te][:, cols]
            alpha = choose_alpha(Xtr, y[tr], groups[tr])
            fold[name] = pearson(Ridge(alpha=alpha).fit(Xtr, y[tr]).predict(Xte), y[te])
            fold[f"{name} alpha"] = alpha
        folds.append(fold)
        print(f"  fold {k}: r A {fold['A'].mean():.4f} (α={fold['A alpha']:g}), "
              f"A+M {fold['A+M'].mean():.4f} (α={fold['A+M alpha']:g})")
    return folds


def template_alpha(folds):
    """Geometric median of the alphas the A+M model chose across folds."""
    return float(np.exp(np.median(np.log([f["A+M alpha"] for f in folds]))))


def fit_template(X, y, alpha):
    """Fit A+M on all data; weights shaped (feature, lag, channel)."""
    model = Ridge(alpha=alpha).fit(X, y)
    n_lags = X.shape[1] // len(FEATURES_AM)
    weights = model.coef_.T.reshape(len(FEATURES_AM), n_lags, -1)
    return weights, model.intercept_, alpha


def save_template(path, weights, intercept, alpha, sfreq, ch_names, stats):
    np.savez(path, weights=weights, intercept=intercept, alpha=alpha, sfreq=sfreq,
             lags=lag_samples(sfreq), features=np.array(FEATURES_AM), ch_names=np.array(ch_names),
             stats=np.array([[*stats[f]] for f in ("pitch", "abs_interval", "surprisal")]),
             band=np.array(config.TRF_BAND), order=config.CAUSAL_FILTER_ORDER,
             decim=config.TRF_DECIM, causal=True)
    print(f"saved template {path}")


# --- Report --------------------------------------------------------------

def report(folds, ch_names):
    """Print the A vs. A+M comparison; returns (mean Δr, folds won)."""
    central = [ch_names.index(c) for c in config.CENTRAL_CHANNELS if c in ch_names]
    d_all = np.array([(f["A+M"] - f["A"]).mean() for f in folds])
    d_cen = np.array([(f["A+M"] - f["A"])[central].mean() for f in folds])
    won = int((d_all > 0).sum())
    print(f"\nHeld-out prediction (Pearson r, mean over folds)")
    print(f"{'channel':>8} {'A':>8} {'A+M':>8} {'Δr':>8}")
    for i, ch in enumerate(ch_names):
        a = np.mean([f["A"][i] for f in folds])
        am = np.mean([f["A+M"][i] for f in folds])
        print(f"{ch:>8} {a:>8.4f} {am:>8.4f} {am - a:>+8.4f}")
    print(f"\nA+M beats A on held-out melodies in {won}/{len(folds)} folds; "
          f"mean Δr {d_all.mean():+.4f} (all channels), {d_cen.mean():+.4f} (Fz/Cz)")
    return d_all.mean(), won


def plot_trf(folds, weights, sfreq, ch_names, title, out_path=None, show=True):
    """Fold scores, the surprisal TRF (template) and the onset TRF at Fz/Cz."""
    lags_ms = lag_samples(sfreq) / sfreq * 1000
    central = [c for c in config.CENTRAL_CHANNELS if c in ch_names]
    cidx = [ch_names.index(c) for c in central]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    a = [f["A"][cidx].mean() for f in folds]
    am = [f["A+M"][cidx].mean() for f in folds]
    for k in range(len(folds)):
        ax.plot([0, 1], [a[k], am[k]], color="#c3c2b7", lw=1.5, zorder=1)
    ax.scatter(np.zeros(len(a)), a, s=64, color=COLOR_A, zorder=2, label="A (acoustic)")
    ax.scatter(np.ones(len(am)), am, s=64, color=COLOR_AM, zorder=2, label="A+M (+ surprisal)")
    ax.set_xticks([0, 1], ["A", "A+M"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("held-out r, Fz/Cz mean")
    ax.set_title("Prediction per fold")
    ax.legend(frameon=False, fontsize=9, loc="best")

    for ax, feat, name in ((axes[1], "surprisal", "Surprisal TRF (template)"),
                           (axes[2], "onset", "Onset TRF")):
        w = weights[FEATURES_AM.index(feat)]
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.axvline(0, color="#c3c2b7", lw=1)
        for c, i in zip(central, cidx):
            ax.plot(lags_ms, w[:, i], color=COLOR_CH.get(c, "#52514e"), lw=2, label=c)
        ax.set_xlabel("lag (ms)")
        ax.set_ylabel("weight (µV per unit feature)")
        ax.set_title(name)
        ax.legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"saved {out_path}")
    if show:
        plt.show()
    return fig


# --- Simulation ----------------------------------------------------------

def _bump(t, center, width):
    return np.exp(-0.5 * ((t - center) / width) ** 2)


def simulate_session(n_blocks=config.CALIBRATION_BLOCKS, effect_uv=1.5, noise_uv=20.0, seed=0):
    """Synthetic calibration session: composer melodies, EEG = onset response
    + effect_uv × (z-scored surprisal) × surprise kernel + noise, at the
    scheduled onsets. Returns blocks like load_session."""
    from acquisition.record import MELODY_ID_STRIDE, NOTE_INDEX_STRIDE, build_block
    from audio.playback import render_sequence
    from composer import markov

    rng = np.random.default_rng(seed)
    model = markov.load()
    sfreq = 250.0
    t = np.arange(0, 0.6, 1 / sfreq)
    onset_kernel = -4 * _bump(t, 0.10, 0.02) + 3 * _bump(t, 0.20, 0.04)
    surprise_kernel = -_bump(t, 0.20, 0.04)
    spread = np.array([1.0 if c in config.CENTRAL_CHANNELS else 0.5 for c in config.CH_NAMES])

    blocks = []
    for b in range(1, n_blocks + 1):
        freqs, notes = build_block(model, b, config.BLOCK_DURATION, rng)
        _, onsets = render_sequence(freqs, silent=True)
        notes = pd.DataFrame(notes)
        notes.insert(0, "note_index", (b - 1) * NOTE_INDEX_STRIDE + 1 + np.arange(len(notes)))
        notes["onset_time"] = onsets / config.AUDIO_SFREQ
        notes = notes[config.NOTE_LOG_COLUMNS]

        samples = (config.RECORD_SETTLE * sfreq + onsets / config.AUDIO_SFREQ * sfreq).astype(int)
        n = samples[-1] + int(sfreq * (1 + config.RECORD_TAIL))
        s = notes["surprisal"].to_numpy()
        drive_on, drive_s = np.zeros(n), np.zeros(n)
        drive_on[samples] = 1
        drive_s[samples] = (s - s.mean()) / s.std()
        signal = (np.convolve(drive_on, onset_kernel)[:n]
                  + effect_uv * np.convolve(drive_s, surprise_kernel)[:n])
        eeg = spread[:, None] * signal + rng.normal(0, noise_uv, (len(spread), n))
        marker = np.zeros(n)
        marker[samples] = notes["note_index"].to_numpy()
        info = mne.create_info(config.CH_NAMES + ["marker"], sfreq,
                               ["eeg"] * len(config.CH_NAMES) + ["stim"])
        raw = mne.io.RawArray(np.vstack([eeg * config.UV_TO_V, marker]), info, verbose=False)
        blocks.append((raw, notes))
    print(f"simulated {n_blocks} blocks, {sum(len(n) for _, n in blocks)} notes, "
          f"surprisal effect {effect_uv} µV per SD, noise {noise_uv} µV")
    return blocks


# --- Main ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", type=int)
    ap.add_argument("--date", help="YYYY-MM-DD (default: any)")
    ap.add_argument("--dry", action="store_true", help="use dry-run recordings (testing)")
    ap.add_argument("--simulate", action="store_true", help="synthetic session with a known effect")
    ap.add_argument("--simulate-effect", type=float, default=1.5,
                    help="µV per SD of surprisal in --simulate (0 = no effect)")
    ap.add_argument("--save-template", action="store_true", help="fit A+M on all data and save")
    ap.add_argument("--no-show", action="store_true", help="save the figure without opening it")
    args = ap.parse_args()

    if args.simulate:
        blocks = simulate_session(effect_uv=args.simulate_effect)
        name = f"simulated_effect{args.simulate_effect:g}"
    else:
        if args.subject is None:
            ap.error("give --subject (or --simulate)")
        blocks = load_session(args.subject, "calibration", args.date, args.dry)
        name = f"subj{args.subject:02d}_calibration{'_dry' if args.dry else ''}"
    mark_bads(blocks)

    X, y, groups, sfreq, ch_names, stats = session_dataset(blocks)
    print(f"{len(y)} samples at {sfreq:g} Hz, {len(np.unique(groups))} melodies, "
          f"{len(ch_names)} channels, {X.shape[1]} lagged features")
    folds = cross_validate(X, y, groups)
    report(folds, ch_names)

    weights, intercept, alpha = fit_template(X, y, template_alpha(folds))
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_trf(folds, weights, sfreq, ch_names, title=f"{name}: TRF model comparison",
             out_path=config.FIGURES_DIR / f"{name}_trf.png", show=not args.no_show)
    if args.save_template:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        save_template(config.MODELS_DIR / f"{name}_template.npz",
                      weights, intercept, alpha, sfreq, ch_names, stats)


if __name__ == "__main__":
    main()
