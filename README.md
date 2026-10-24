# EEG Surprise Composer

A closed-loop music composer that listens to your brain. A statistical melody model generates music and estimates how surprising each note is; EEG measures how strongly the listener's brain responds to those surprises; the composer then adjusts how predictable its music is to keep the listener in the zone where their brain is most sensitive to surprise.

> **Core idea:** Music works by setting up expectations and then playing with them. Too predictable and it's boring; too random and nothing stands out. This project uses the brain's own prediction-error response to find the balance for each listener.

---

## Table of contents

1. [Background](#background)
2. [System overview](#system-overview)
3. [Components](#components)
4. [Repository structure](#repository-structure)
5. [Setup](#setup)
6. [Data formats](#data-formats)
7. [Running the pipeline](#running-the-pipeline)
8. [Analysis plan](#analysis-plan)
9. [Limitations](#limitations)
10. [Future work](#future-work)
11. [References](#references)

---

## Background

When we listen to music, the brain continuously predicts what comes next. Notes that violate those predictions evoke larger cortical responses roughly 100–350 ms after onset. Di Liberto et al. (2020) showed that these responses can be measured with scalp EEG during naturalistic listening by relating EEG to per-note **surprisal** values from a statistical (Markov) model of melody, using temporal response functions (ridge regression on time-lagged stimulus features).

This project turns that finding into an interactive system:

- **Measure:** calibrate a per-listener model of the neural surprise response.
- **Control:** use the strength of that response, in real time, to steer a generative composer.

**Terminology.** We measure a *neural surprise (prediction-error) response*, not conscious, felt surprise. Surprisal comes from our model, which approximates, but is not identical to, the listener's expectations.

---

## System overview

```
 Bach chorales ──► Markov model ──► note + surprisal ──► audio playback ──► listener
                        ▲                                      │              │
                        │                                 note markers       EEG
                   temperature                                 ▼              ▼
                        │                              synchronized recording
                        │                                      │
                        └──── surprise response gain ◄── analysis (TRF template)
```

| Step | Component | Purpose |
|------|-----------|---------|
| 1 | Composer | Generate melodies; compute per-note surprisal |
| 2 | Audio playback | Play notes with controlled acoustics |
| 3 | Synchronization | Mark each note onset in the EEG stream |
| 4 | EEG acquisition | Record brain activity (calibration + live) |
| 5 | Preprocessing | Clean the EEG signal |
| 6 | Analysis | Show EEG tracks surprisal; build a per-listener template |
| 7 | Closed loop | Adjust temperature to maximize surprise sensitivity |

---

## Components

### 1. Composer

- **Training data:** soprano lines extracted from the Bach chorales in the `music21` corpus.
- **Model:** variable-context Markov model predicting the next pitch from the previous 2–3 pitches, with smoothing for unseen transitions.
- **Output per note:** pitch, probability, and **surprisal** = −log₂ P(note | context).
- **Temperature:** controls sampling only: probabilities are raised to the power 1/T and renormalized. Low T → predictable; high T → unusual.
- **Important:** surprisal is always computed from the **untempered** model, which stands in for the listener's expectations, so values are comparable across temperatures.
- *(Optional)* Short-term learning: also update on the current melody, mimicking how listeners learn patterns within a session.

### 2. Audio playback

- Tones synthesized in `numpy` (sine tones with 5 ms fades to avoid clicks).
- **Held constant:** loudness, timbre, note duration, tempo (isochronous, one note every ~400–500 ms). Pitch is the only thing that varies.
- Remaining acoustic variation (pitch height, interval size) is handled as control features in the analysis.
- **Wired headphones required.** Bluetooth adds variable delay (jitter), which smears averaged responses.

### 3. Synchronization

- A marker is inserted into the EEG stream (`BoardShim.insert_marker`) at every note onset; the marker value encodes the note index.
- Every recording saves a **note log** CSV (see [Data formats](#data-formats)) linking marker values to pitch and surprisal.
- A constant audio-output delay shifts responses in time but is tolerable; variable delay is the main risk.

### 4. EEG acquisition

- **Hardware:** g.tec Unicorn, 8 channels (Fz, C3, Cz, C4, Pz, PO7, Oz, PO8; verify order against device docs). Central channels (Fz, Cz) carry the auditory responses of interest.
- **Software:** BrainFlow. Unicorn support requires **Windows or Ubuntu** and pairing through the supplied USB dongle.
- **Listener protocol:** seated, relaxed jaw, eyes on a fixation point, minimal movement. Blocks of 2–3 minutes with breaks.
- **Calibration session:** 10–15 minutes of generated melodies at a mix of temperatures, giving a wide range of surprisal values.
- **Sanity check first:** an auditory oddball (standard 1000 Hz / deviant 1200 Hz tones) to verify signal quality, markers and pipeline.

### 5. Preprocessing

- Load into MNE (BrainFlow µV → volts), apply the standard 10-20 montage.
- Mark bad channels; re-reference to the average.
- Bandpass: ~1–8 Hz for TRF analysis, ~1–30 Hz for ERP plots.
- Downsample to 64 Hz for TRFs.
- Reject epochs/segments with large artifacts.
- Live loop: **causal** filters only (`scipy.signal` / BrainFlow `DataFilter`).

### 6. Analysis

**a. ERP comparison.** Epoch around note onsets, split notes into high- and low-surprisal halves, and compare averaged responses at Fz/Cz. Expected: larger response to high-surprisal notes ~100–300 ms after onset.

**b. TRF model comparison.** Fit ridge-regression TRFs (`mne.decoding.ReceptiveField`, lags ≈ −100 to 500 ms):

| Model | Features |
|-------|----------|
| A (acoustic) | note onset, pitch height, absolute interval size |
| A + M | A + surprisal |

- Cross-validate **by melody** (`GroupKFold`), never by random samples.
- **Key result:** if A + M predicts held-out EEG better than A, the brain tracks the model's expectations beyond what acoustics explain.
- *(Optional)* Compare surprisal from different context lengths; the one that best predicts EEG best approximates this listener's expectations.
- *(Optional)* Behavioral check: listener rates melody surprise (1–5) or presses a key on "wrong" notes; compare with surprisal and EEG.

The fitted surprisal TRF is this listener's **surprise template** (timing, shape, channel distribution), used in step 7.

### 7. Closed loop

**Goal:** find the temperature where the listener's brain is most sensitive to surprise.

- Too predictable → few violations to respond to.
- Too random → no stable expectations, so surprising notes stop standing out.

**Every 45–60 seconds:**

1. Predict EEG from onsets and acoustic controls using the calibrated model; subtract from the actual EEG.
2. Build the predicted surprise response: each note's surprisal × template, placed at its onset.
3. Fit a single scalar, the **surprise response gain**: `gain = (predicted · residual) / (predicted · predicted)`.
   - gain ≈ 1 → responding as strongly as during calibration
   - gain ≈ 0 → surprising notes aren't standing out
4. Smooth gain over several windows, then adjust temperature by trial and error: step T; if gain rose, continue in that direction; if it fell, reverse.

This is a matched filter: the template's shape is fixed from calibration, and only its strength is estimated live, so short windows can support the estimate.

---

## Repository structure

```
eeg-surprise-composer/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py               # shared constants
├── composer/
│   ├── corpus.py           # load chorales, extract soprano lines
│   ├── markov.py           # train model, probabilities, surprisal
│   └── sample.py           # temperature sampling
├── audio/
│   └── playback.py         # tone synthesis + playback
├── acquisition/
│   ├── oddball.py          # sanity-check experiment
│   └── record.py           # BrainFlow streaming + per-note markers
├── analysis/
│   ├── preprocess.py       # MNE loading, filtering, rejection
│   ├── erp.py              # high vs. low surprisal ERPs
│   └── trf.py              # ridge TRFs, model comparison, template
├── closed_loop/
│   ├── gain.py             # surprise response gain
│   └── run_live.py         # live loop: play, record, update temperature
├── notebooks/              # exploration and demo plots
├── data/                   # recordings (gitignored)
│   ├── raw/
│   └── processed/
├── models/                 # trained Markov model, templates (gitignored)
└── figures/                # final plots
```

**`config.py`** holds everything shared: channel names/order, sampling rate, note duration, tempo, filter cutoffs, TRF lag range, window length, temperature step, file paths.

**`.gitignore`** should include `data/`, `models/`, `*.fif`, recording CSVs, `__pycache__/`, `.ipynb_checkpoints/`.

---

## Setup

```bash
git clone <repo-url>
cd eeg-surprise-composer
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**Dependencies:** `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`, `mne`, `brainflow`, `music21`, `sounddevice`, `jupyter` (optional: `pretty_midi`, `pylsl`).

After confirming everything works on the recording laptop, pin versions with `pip freeze > requirements.txt`.

**Hardware checklist**
- [ ] Windows or Ubuntu laptop for recording
- [ ] Unicorn USB dongle plugged in; headset paired through it
- [ ] Wired headphones
- [ ] Electrode gel (optional, improves signal)

---

## Data formats

### Note log (one CSV per recording)

The handoff between composer, acquisition and analysis. Every recording saves one.

| Column | Type | Description |
|--------|------|-------------|
| `note_index` | int | Unique index; equals the EEG marker value |
| `melody_id` | int | Melody the note belongs to (used for CV splits) |
| `onset_time` | float | Wall-clock time the note was triggered (s) |
| `pitch` | int | MIDI pitch number |
| `interval` | int | Semitones from previous note (signed) |
| `surprisal` | float | −log₂ P under the untempered model (bits) |
| `temperature` | float | Sampling temperature in effect |

### EEG recordings

BrainFlow CSV via `DataFilter.write_file`; channel indices from `BoardShim.get_eeg_channels` / `get_marker_channel`.

### File naming

```
YYYY-MM-DD_subjNN_<session>_<run>.csv          # EEG
YYYY-MM-DD_subjNN_<session>_<run>_notes.csv    # note log
```
Sessions: `oddball`, `calibration`, `live`.

---

## Running the pipeline

```bash
# 0. Sanity check (signal + markers + ERP)
python acquisition/oddball.py
python analysis/erp.py --session oddball

# 1. Train the composer
python composer/markov.py --train

# 2. Calibration recording (10–15 min)
python acquisition/record.py --session calibration

# 3. Analysis: ERPs, TRF comparison, save template
python analysis/erp.py --session calibration
python analysis/trf.py --session calibration --save-template

# 4. Closed loop
python closed_loop/run_live.py --template models/<subject>_template.npz
```

*(Command-line arguments are the intended interface; update as scripts are implemented.)*

---

## Analysis plan

| Tier | Deliverable | Status |
|------|-------------|--------|
| Sanity check | Oddball ERP: deviant vs. standard | ☐ |
| Guaranteed result | ERP: high vs. low surprisal notes | ☐ |
| Core result | TRF: A + M beats A on held-out melodies | ☐ |
| Stretch | Live closed loop adjusting temperature | ☐ |
| Fallback | Closed loop run on recorded data / Di Liberto dataset | ☐ |

**Pipeline validation:** before relying on our own data, run the TRF analysis on the Di Liberto et al. Bach EEG dataset to confirm the code reproduces a surprisal effect.

---

## Limitations

- **Model ≠ listener.** Surprisal approximates the listener's expectations; musicianship, culture and within-session learning cause mismatches. A low gain can mean weak surprise responses *or* model mismatch.
- **Hardware.** 8 channels and short recordings give noisy single-window estimates; the closed loop is a proof of concept.
- **Timing.** Audio output delay is not measured precisely.
- **Simple stimuli.** Monophonic, isochronous sine-tone melodies trade musical richness for experimental control.
- **Not "enjoyment."** We do not claim to measure pleasure, only whether the brain is tracking the music's structure.

---

## Future work

- **Harmony:** chord-level Markov model (Roman-numeral reduction via `music21`); harmonic surprise has its own neural signature (ERAN).
- **Performance mode:** melody over a fixed, temperature-independent accompaniment for a more musical demo.
- **Richer models:** neural sequence models vs. Markov surprisal as predictors of EEG.
- **Behavioral validation:** combine ratings, EEG and model surprisal across multiple listeners.

---

## References

- Di Liberto, G. M., Pelofi, C., Bianco, R., et al. (2020). Cortical encoding of melodic expectations in human temporal cortex. *eLife*, 9, e51784. https://doi.org/10.7554/eLife.51784
- Kern, P., Heilbron, M., de Lange, F. P., & Spaak, E. (2022). Cortical activity during naturalistic music listening reflects short-range predictions based on long-term experience. *eLife*. https://doi.org/10.7554/eLife.80935
- Di Liberto et al. Bach EEG dataset (Dryad): https://doi.org/10.5061/dryad.g1jwstqmh
- MNE-Python: https://mne.tools
- BrainFlow: https://brainflow.readthedocs.io
- music21: https://web.mit.edu/music21/
