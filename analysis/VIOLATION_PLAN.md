# Violation experiment: analysis plan (written before recording)

Written 2026-10-25, before any violation data was recorded. Settings marked `config.` live in `config.py` and must not be changed after recording starts.

## Why

The continuous calibration analysis (TRF, A+M vs. A) found no surprisal effect for subject 1: 0/5 folds, Δr ≈ 0. Surprisal there correlated r = 0.69 with |interval|, so most of its variance was already explained by acoustic model A. This experiment plants improbable notes whose leap size matches the note they replace, so improbability and interval size are separated by design.

## Stimuli

- Melodies from the Bach Markov model at a low, predictable temperature (`config.VIOLATION_TEMPERATURE` = 0.5), random key per melody, same timbre, IOI and gaps as calibration.
- **Violations:** about 10% of notes (`config.VIOLATION_RATE`). At a violation position, the generated note is replaced by the pitch with the **lowest probability under the untempered model**, given the preceding notes, among pitches whose leap from the previous note is within ±1 semitone (`config.VIOLATION_INTERVAL_TOL`) of the original leap. Excluded: the original pitch, and a repeat of the previous note.
- **Steps only** (*amended 2026-10-25 after a fake-EEG dry run, before any real violation data*): a violation is planted only where the original note is a step (1–2 semitones from the previous note), and the violation must also be a step (`config.VIOLATION_MAX_LEAP` = 2). Reason: in the dry run, the ±1 rule often produced 3+ semitone leaps, which have almost no interval-matched controls at T = 0.5. That left violations at a mean |interval| of 2.0 vs. 1.0 for controls, the confound this design exists to remove.
- **Spacing:** no violation in a melody's first 3 notes (`config.VIOLATION_MIN_LEAD`) or last 2 (`config.VIOLATION_MIN_TAIL`), and at least 4 normal notes between violations (`config.VIOLATION_MIN_GAP`).
- Notes after a violation keep their originally generated pitches. All surprisal values are recomputed under the untempered model on the melody as played (CLAUDE.md invariant 1).
- The note log keeps its 7 contract columns. Violations are listed in a separate `<recording>_violations.csv` (note_index, original pitch, original surprisal).

## Recording

Session `violation`, 6 blocks of ~2.7 min (`config.VIOLATION_BLOCKS`), about 16 min of listening, same listener instructions as calibration.

## Pre-specified test

- **Epochs:** −100 to 500 ms around each note, baseline −100–0 ms, ERP filter 1–30 Hz, reject >120 µV. Bad channels excluded first (same rules as the oddball). All settings are from `config.py`, as for the oddball.
- **Violation epochs:** every kept violation note.
- **Control epochs:** kept normal notes, excluding each melody's first note and the note right after a violation. Drawn at random with fixed seed `config.CONTROL_SEED` = 2026 so that their |interval| distribution matches the kept violations': for each |interval| value v with n_v kept violations, draw min(available, `config.CONTROL_RATIO` × n_v) controls, without replacement, with `CONTROL_RATIO` = 3.
- **Primary test:** at **Fz**, mean amplitude over **100–250 ms**, violation vs. control, **Welch's t-test, two-sided, α = 0.05**. Expected direction: violations **more negative** (a mismatch-type response). Report the difference (violation − control) in µV, t, p and both n.
- **Secondary (reported, not the claim):** the same test at Cz; the ERP figure; checks that the matching worked (mean |interval| and mean surprisal for violations vs. controls).
- **If Fz is marked bad,** the primary test moves to Cz. Decide this before looking at any ERP.

## What counts as the result

- **Support:** Fz difference negative and p < 0.05: "notes the Bach model finds improbable evoke a larger negative response, with leap size held constant."
- **No support:** anything else, reported as is. No changing the window, channel, matching or test after seeing the data. Anything else we try is labelled **exploratory**.
