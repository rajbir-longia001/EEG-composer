# CLAUDE.md

Context for AI coding assistants working in this repo. See `README.md` for the full project description.

## Project in one paragraph

Hackathon project (neurotech, time-limited). A Markov melody model trained on Bach chorale soprano lines generates melodies and computes per-note surprisal. Notes are played as sine tones while EEG is recorded from a g.tec Unicorn (8 channels) via BrainFlow, with a marker at every note onset. Analysis shows EEG tracks surprisal (ERPs + ridge-regression TRFs). A closed loop then adjusts sampling temperature to maximize the listener's "surprise response gain." **Prioritize working, simple code over elegance. The team has limited ML experience: prefer numpy/scikit-learn/MNE; do not introduce PyTorch or deep learning.**

## Pipeline and module ownership

| Step | Module | Responsibility |
|------|--------|----------------|
| 1 | `composer/` | corpus loading, Markov model, surprisal, temperature sampling |
| 2 | `audio/` | tone synthesis and playback |
| 3–4 | `acquisition/` | BrainFlow streaming, markers, oddball sanity check, recordings |
| 5–6 | `analysis/` | MNE preprocessing, ERPs, TRFs, per-listener template |
| 7 | `closed_loop/` | live gain estimate and temperature updates |

Shared constants live in `config.py`. Never hard-code channel names, sampling rates, tempo, filter cutoffs, lag ranges or paths in modules; import them from `config.py`.

## Invariants (do not break these)

1. **Surprisal uses the untempered model.** Temperature affects sampling only. Surprisal = −log₂ P(note | context) under the original trained probabilities, so values are comparable across temperatures.
2. **Marker value = `note_index`.** Every note gets a nonzero marker inserted via `board.insert_marker(note_index)` immediately before playback. `note_index` starts at 1 (0 means "no marker" in BrainFlow).
3. **Every recording writes a note log** with exactly these columns: `note_index, melody_id, onset_time, pitch, interval, surprisal, temperature`. This CSV is the contract between modules; don't rename or drop columns.
4. **Units:** BrainFlow returns µV; multiply by `1e-6` before creating MNE objects.
5. **Cross-validation splits by `melody_id`** (`GroupKFold`). Never shuffle individual samples: adjacent EEG samples are correlated and this inflates scores.
6. **Both TRF models include acoustic controls.** Model A = onset, pitch height, |interval|. Model A+M = A + surprisal. The claim is always "A+M beats A on held-out melodies."
7. **Real-time code uses causal filters only** (`scipy.signal.lfilter` / BrainFlow `DataFilter`), never MNE's zero-phase `raw.filter`, which looks at future samples. Offline analysis may use MNE filters.
8. **Stimulus acoustics are constant:** same loudness, timbre, duration and inter-onset interval for every note. Only pitch varies. Tones use ~5 ms onset/offset ramps.

## Hardware and platform

- Unicorn via BrainFlow: `BoardIds.UNICORN_BOARD`. Supported on **Windows and Ubuntu only (not macOS)**; device must be paired through its USB dongle.
- Assumed channel order: `Fz, C3, Cz, C4, Pz, PO7, Oz, PO8` (from `config.py`; verify against device docs). Central channels Fz/Cz matter most.
- Get sampling rate and channel indices from `BoardShim.get_sampling_rate`, `get_eeg_channels`, `get_marker_channel`; don't hard-code them.
- Always release the session (`board.stop_stream(); board.release_session()`) in a `finally` block so a crash doesn't leave the device locked.
- Save data after every block, not only at the end of a session.

## Default parameters (mirror config.py)

- Inter-onset interval: 0.4–0.5 s, isochronous
- Markov context: 2–3 previous pitches, with smoothing
- ERP filter: 1–30 Hz; TRF filter: 1–8 Hz; TRF sampling rate: 64 Hz
- TRF lags: −0.1 to 0.5 s; ridge alpha chosen by cross-validation
- ERP epochs: −0.1 to 0.5 s, baseline (None, 0), reject ~100–150 µV
- Closed-loop window: 45–60 s, gain smoothed over several windows

## Closed-loop gain (step 7)

Keep the calibrated surprisal TRF shape fixed; estimate only its strength per window:
1. residual = EEG − prediction from onset/acoustic features (calibrated model)
2. predicted_surprise = surprisal-weighted impulses at note onsets convolved with the surprisal template
3. `gain = dot(predicted_surprise, residual) / dot(predicted_surprise, predicted_surprise)`
Temperature update is trial and error: step T; continue if smoothed gain rose, reverse if it fell.

## Data and git

- Never commit `data/`, `models/`, `*.fif`, or recording CSVs. Large datasets (e.g. the Di Liberto Bach EEG set, ~6 GB) stay outside the repo.
- File naming: `YYYY-MM-DD_subjNN_<session>_<run>.csv` and `..._notes.csv`; sessions are `oddball`, `calibration`, `live`.
- Small commits, one branch per person, merge often.

## Libraries

`numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`, `mne`, `brainflow`, `music21`, `sounddevice`. Optional: `pretty_midi`, `pylsl`, `jupyter`. Ask before adding anything else.

## Coding style

- Plain functions and small modules; minimal classes.
- Every script runnable standalone with `argparse` and a `main()`.
- Prefer clear names and short docstrings over clever code.
- When in doubt, add a quick plot: plots are how we verify EEG code works.

## Language and claims

In docs, comments and demo text, say "neural surprise response" or "prediction-error response," not "the listener was surprised." Don't claim the system measures enjoyment.
