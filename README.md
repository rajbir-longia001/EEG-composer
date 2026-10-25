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
- **Model:** fixed-order Markov model predicting the next pitch from the previous 2–3 pitches (`MARKOV_ORDER` in `config.py`), with smoothing for unseen transitions. The first notes of a melody have a shorter context; pad it with a start symbol.
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

- A g.Pype `Marker` source is routed beside the EEG. g.Pype timestamps each marker on the calling thread and places it on the same timeline as the EEG samples.
- A block of notes is rendered into one audio array, so onsets fall on exact samples and the sound card's clock sets the rhythm. An audio callback records when each onset will reach the DAC, and `marker.emit(note_index)` fires at that moment (`audio.playback.play_with_onsets`). Measured jitter is ~0.4 ms. The obvious alternative, emitting before each blocking `stream.write()`, jittered ~8 ms because Windows audio returns in ~20 ms blocks.
- Every recording saves a **note log** CSV (see [Data formats](#data-formats)) linking marker values to pitch and surprisal.
- A constant audio-output delay shifts responses in time but is tolerable; variable delay is the main risk.

### 4. EEG acquisition

- **Hardware:** g.tec Unicorn BCI Core-8: 8 channels at 250 Hz over Bluetooth 5, with hybrid dry/wet (g.SAHARA) electrodes. Standard layout Fz, C3, Cz, C4, Pz, PO7, POz, PO8 (verify on the cap). One combined GND/REF electrode (shorted internally), placed on the right mastoid. Central channels (Fz, Cz) carry the auditory responses of interest.
- **Software:** [g.Pype](https://gpype.gtec.at/), g.tec's Python SDK. A recording is one pipeline:

  ```python
  import gpype as gp

  p = gp.Pipeline()
  amp = gp.BCICore()                     # first Core-8 in range; µV; default montage
  marker = gp.Marker()                   # marker.emit(note_index) before each tone
  both = gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL])
  p.connect(amp, both["in1"])
  p.connect(marker, both["in2"])
  p.connect(both, gp.HDF5Writer(file_name="recording.h5"))
  try:
      p.start()
      ...                                # play notes, emitting a marker for each
  finally:
      p.stop()
      p.close()
  ```

  For the live loop, a small custom `gp.Sink` hands each frame of filtered EEG to our own code (see g.Pype's `examples/s7e3_two_ends.py`). g.Pype's `examples/` folder is the best reference. The closest examples are `s3e1_bci_core.py`, `s4e9_into_mne.py` and `s5e4_oddball_paradigms.py`.
- **Listener protocol:** seated, relaxed jaw, eyes on a fixation point, minimal movement. Blocks of 2–3 minutes with breaks.
- **Calibration session:** 10–15 minutes of generated melodies at a mix of temperatures, giving a wide range of surprisal values.
- **Sanity check first:** an auditory oddball (standard 1000 Hz / deviant 1200 Hz tones) to verify signal quality, markers and pipeline.

### 5. Preprocessing

- Load the HDF5 recording into MNE (µV → volts; drop the `master_index`, `valid`, `host_time` timeline channels; the marker channel becomes a stim channel), then apply the standard 10-20 montage.
- Mark bad channels. Keep the hardware reference (the Core-8's GND/REF electrode on the right mastoid). Do **not** re-reference to the average: with 8 mostly central channels, the average contains much of the Fz/Cz response and would cancel it.
- Bandpass: ~1–8 Hz for TRF analysis, ~1–30 Hz for ERP plots.
- Downsample to 64 Hz for TRFs.
- Reject epochs/segments with large artifacts. MNE thresholds are in volts: `reject=dict(eeg=120e-6)` means 120 µV.
- Live loop: **causal** filters only (`gp.Bandpass`, which is causal by default, or `scipy.signal.sosfilt`).
- The surprise template used live must be fitted on data filtered with the **same causal filter**. Zero-phase and causal filters give responses with different timing, so a template fitted on zero-phase data would not line up with live EEG and would underestimate the gain.

### 6. Analysis

**a. ERP comparison.** Epoch around note onsets, split notes into high- and low-surprisal halves, and compare averaged responses at Fz/Cz. Expected: larger response to high-surprisal notes ~100–300 ms after onset. *Caveat:* high-surprisal notes tend to be large leaps, so this comparison is confounded with interval size. It is a quick visual check; the TRF comparison below is what controls for acoustics.

**b. TRF model comparison.** Fit ridge-regression TRFs (`mne.decoding.ReceptiveField`, lags ≈ −100 to 500 ms):

| Model | Features |
|-------|----------|
| A (acoustic) | note onset, pitch height, absolute interval size |
| A + M | A + surprisal |

- Cross-validate **by melody** (`GroupKFold`), never by random samples.
- Choose the ridge alpha **inside** each training fold (nested CV), so held-out melodies never influence the fit.
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

**Time budget:** a 50 s window holds ~110 notes, and each temperature decision averages ~3 windows, so expect about one temperature step every 2.5–5 minutes. Plan the demo length around this.

---

## Repository structure

Each module folder is a Python package (has an `__init__.py`); run scripts from the repo root with `python -m <package>.<module>`.

```
NeurotechHacks/
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
│   └── record.py           # g.Pype recording pipeline + per-note markers
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

**`.gitignore`** covers `data/`, `models/`, `*.fif`, `*.npz`/`*.npy`, `*.h5`, recording CSVs, `__pycache__/`, `.ipynb_checkpoints/`.

---

## Setup

```bash
git clone <repo-url>
cd NeurotechHacks
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**Python 3.10–3.14** (required by g.Pype).

**Dependencies:** `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`, `mne`, `gpype[all]`, `music21`, `sounddevice`, `jupyter` (for `notebooks/`). Optional: `pretty_midi`, `pylsl`. On macOS's zsh, quote the extra: `pip install "gpype[all]"`. g.Pype is under g.tec's non-commercial license.

`requirements.txt` lists what we depend on; `requirements-lock.txt` pins the exact versions known to work together (Python 3.14, Windows). To match them exactly: `pip install -r requirements-lock.txt`. After changing dependencies, regenerate it with `pip freeze > requirements-lock.txt`.

Quick check that the install works (no headset needed):

```bash
python -c "import numpy, scipy, pandas, sklearn, mne, music21, sounddevice, gpype; print('ok')"
```

**Hardware checklist**
- [ ] Recording laptop: Windows, macOS, or x86_64 Linux, with Bluetooth
- [ ] Unicorn BCI Core-8 charged and paired over Bluetooth; GND/REF electrode on the right mastoid
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
| `onset_time` | float | `time.time()` when the marker was emitted (s); for logging and cross-checks (the EEG marker channel is the timing that counts) |
| `pitch` | int | MIDI pitch number |
| `interval` | int | Semitones from previous note (signed); empty (NaN) for the first note of each melody |
| `surprisal` | float | −log₂ P under the untempered model (bits) |
| `temperature` | float | Sampling temperature in effect |

**Oddball recordings** use the same columns: `melody_id` = block number, `pitch` = nearest MIDI pitch to the tone, and `surprisal`, `interval`, `temperature` left empty (NaN). Whether a tone is standard or deviant is derived from `pitch`.

### EEG recordings

HDF5 via `gp.HDF5Writer`. HDF5 keeps channel units and roles (a CSV doesn't, so its scale is ambiguous). Columns are the 8 EEG channels (µV), BCICore's timeline channels (`master_index`, `valid`, `host_time`) and the marker channel (0 = no marker, otherwise `note_index`). Read the file back with `gp.HDF5Reader` in batch mode, then convert it to MNE as in g.Pype's `examples/s4e9_into_mne.py`. g.Pype writers append the date and time to the file name, so the recording script renames the file to the convention below once the pipeline is closed.

### File naming

```
YYYY-MM-DD_subjNN_<session>_<run>.h5           # EEG
YYYY-MM-DD_subjNN_<session>_<run>_notes.csv    # note log
```
Sessions: `oddball`, `calibration`, `live`.

---

## Running the pipeline

```bash
# Run everything from the repo root with -m so `import config` works.

# 0. Sanity check (signal + markers + ERP)
python -m acquisition.view_stream                    # signal quality; close before recording
python -m acquisition.oddball --dry-run --fake-eeg   # no hardware, silent: tests the code
python -m acquisition.oddball --dry-run              # headset, silent: tests headset + markers
python -m acquisition.oddball --subject 1 --run 1    # real oddball (~3 min), sync check + ERP plot
python -m analysis.erp --session oddball --subject 1 --run 1   # re-plot later

# 1. Train the composer
python -m composer.markov --train

# 2. Calibration recording (10–15 min)
python -m acquisition.record --session calibration

# 3. Analysis: ERPs, TRF comparison, save template
python -m analysis.erp --session calibration
python -m analysis.trf --session calibration --save-template

# 4. Closed loop
python -m closed_loop.run_live --template models/<subject>_template.npz
```

*(Command-line arguments are the intended interface; update as scripts are implemented. `python acquisition/oddball.py` would fail, because it puts `acquisition/` rather than the repo root on the import path.)*

---

## Analysis plan

| Tier | Deliverable | Status |
|------|-------------|--------|
| Sanity check | Oddball ERP: deviant vs. standard | ☐ |
| First look | ERP: high vs. low surprisal notes (confounded with interval size) | ☐ |
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
- g.Pype (g.tec): https://gpype.gtec.at/ · https://github.com/gtec-medical-engineering/gpype
- music21: https://web.mit.edu/music21/
