# CLAUDE.md

Context for AI coding assistants working in this repo. See `README.md` for the full project description.

## Project in one paragraph

Hackathon project (neurotech, time-limited). A Markov melody model trained on Bach chorale soprano lines generates melodies and computes per-note surprisal. Notes are played as sine tones while EEG is recorded from a g.tec Unicorn BCI Core-8 (8 channels) via g.tec's g.Pype SDK, with a marker at every note onset. Analysis shows EEG tracks surprisal (ERPs + ridge-regression TRFs). A closed loop then adjusts sampling temperature to maximize the listener's "surprise response gain." **Prioritize working, simple code over elegance. The team has limited ML experience: prefer numpy/scikit-learn/MNE; do not introduce PyTorch or deep learning.**

## Pipeline and module ownership

| Step | Module | Responsibility |
|------|--------|----------------|
| 1 | `composer/` | corpus loading, Markov model, surprisal, temperature sampling |
| 2 | `audio/` | tone synthesis and playback |
| 3–4 | `acquisition/` | g.Pype pipeline, markers, oddball sanity check, recordings |
| 5–6 | `analysis/` | MNE preprocessing, ERPs, TRFs, per-listener template |
| 7 | `closed_loop/` | live gain estimate and temperature updates |

Shared constants live in `config.py`. Never hard-code channel names, sampling rates, tempo, filter cutoffs, lag ranges or paths in modules; import them from `config.py`.

## Invariants (do not break these)

1. **Surprisal uses the untempered model.** Temperature affects sampling only. Surprisal = −log₂ P(note | context) under the original trained probabilities, so values are comparable across temperatures.
2. **Marker value = `note_index`.** Every note gets a nonzero marker via `marker.emit(note_index)` (a `gp.Marker` routed beside the EEG) immediately before playback. `emit` timestamps on the calling thread, so call it from the playback thread, right before `sounddevice` starts the tone. `note_index` starts at 1 (0 means "no marker" in the marker channel).
3. **Every recording writes a note log** with exactly these columns: `note_index, melody_id, onset_time, pitch, interval, surprisal, temperature` (`config.NOTE_LOG_COLUMNS`). This CSV is the contract between modules; don't rename or drop columns. `onset_time` is `time.time()` at `emit` (for logging and cross-checks; the marker channel is the timing that counts). `interval` is NaN for a melody's first note. Oddball logs leave `surprisal`, `interval`, `temperature` as NaN.
4. **Units:** `gp.BCICore` outputs µV; multiply by `1e-6` (`config.UV_TO_V`) before creating MNE objects. MNE rejection thresholds are in volts too (`reject=dict(eeg=120e-6)`).
5. **Cross-validation splits by `melody_id`** (`GroupKFold`). Never shuffle individual samples: adjacent EEG samples are correlated and this inflates scores. Choose ridge alpha inside each training fold (nested CV).
6. **Both TRF models include acoustic controls.** Model A = onset, pitch height, |interval|. Model A+M = A + surprisal. The claim is always "A+M beats A on held-out melodies."
7. **Real-time code uses causal filters only** (`gp.Bandpass` with its default `phase="causal"`, or `scipy.signal.sosfilt`), never MNE's zero-phase `raw.filter`, which looks at future samples. Offline analysis may use MNE filters, **except** the surprise template used by the closed loop: fit it on calibration data filtered with the same causal filter as the live loop, or its timing won't match live EEG.
8. **Stimulus acoustics are constant:** same loudness, timbre, duration and inter-onset interval for every note. Only pitch varies. Tones use ~5 ms onset/offset ramps.
9. **Keep the hardware reference** (Core-8 GND/REF on the right mastoid). Don't re-reference to the average: with 8 mostly central channels it cancels much of the Fz/Cz response.

## Hardware and platform

- Device: **g.tec Unicorn BCI Core-8** (not the Hybrid Black): 8 channels, 250 Hz, Bluetooth 5, no USB dongle. One GND/REF electrode (shorted internally) on the right mastoid.
- Acquisition uses **g.Pype** (`pip install "gpype[all]"`, Python 3.10–3.14, Windows/macOS/Linux x86_64; g.tec non-commercial license). The repo's `examples/` folder (github.com/gtec-medical-engineering/gpype) is the best reference; `s3e1_bci_core.py`, `s4e9_into_mne.py` and `s5e4_oddball_paradigms.py` are closest to what we do.
- Recording pipeline: `gp.BCICore()` and `gp.Marker()` → `gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL])` → `gp.HDF5Writer`. HDF5 keeps units and channel roles (CSV doesn't), so it loads into MNE without guessing scale. The BCICore output also carries timeline channels (`master_index`, `valid`, `host_time`); drop them before MNE.
- Live data into our own code: a small `gp.Sink` subclass whose `step()` copies each frame into a buffer (see `s7e3_two_ends.py`). `step()` runs on the source's thread, so guard the buffer with a lock.
- Channel order: `Fz, C3, Cz, C4, Pz, PO7, POz, PO8` (BCICore's default montage, mirrored in `config.py`; verify on the cap). Central channels Fz/Cz matter most.
- Read the sampling rate and channel labels from the recording/port context; don't hard-code them.
- Always `p.stop(); p.close()` in a `finally` block so a crash doesn't leave the device connected.
- Writers append the date and time to `file_name`; rename to our naming convention after `p.close()`.
- Save data after every block (one pipeline run per block), not only at the end of a session.

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

- Never commit `data/`, `models/`, `*.fif`, `*.h5`, or recording CSVs. Large datasets (e.g. the Di Liberto Bach EEG set, ~6 GB) stay outside the repo.
- File naming: `YYYY-MM-DD_subjNN_<session>_<run>.h5` (EEG) and `..._notes.csv`; sessions are `oddball`, `calibration`, `live`.
- Small commits, one branch per person, merge often.

## Libraries

`numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`, `mne`, `gpype[all]`, `music21`, `sounddevice`, `jupyter`. Optional: `pretty_midi`, `pylsl`. Ask before adding anything else.

## Coding style

- Plain functions and small modules; minimal classes.
- Every script runnable standalone with `argparse` and a `main()`, from the repo root as `python -m <package>.<module>` (each module folder has an `__init__.py`).
- Prefer clear names and short docstrings over clever code.
- When in doubt, add a quick plot: plots are how we verify EEG code works.

## Language and claims

In docs, comments and demo text, say "neural surprise response" or "prediction-error response," not "the listener was surprised." Don't claim the system measures enjoyment.
