# Day 2 runbook

Step-by-step plan for the last ~12 hours: record data, analyze it, build the closed loop, and prepare the demo. Run every command from the repo root with `.venv` active (`.venv\Scripts\Activate.ps1`).

**What already works:** the composer, the oddball and calibration recording scripts, ERP and TRF analysis, live viewing, and the closed loop (gain estimate + live loop). All of it is tested with fake EEG, simulated data and silent audio. **Not yet run:** anything with the real headset and sound.

**Suggested split:** one person runs the recordings (steps 1–5) while another prepares the demo (step 8) and, if there's time, writes the Di Liberto adapter (fallback). The time freed by the finished closed loop is best spent on **more data**: extra calibration blocks, or a second listener.

| Step | What | Time |
|---|---|---|
| 0 | Commit and branch | 10 min |
| 1 | Headset on, signal check | 30–45 min |
| 2 | Audio check | 10 min |
| 3 | Oddball sanity check | 20–30 min |
| 4 | Calibration recording | ~25 min |
| 5 | Analysis: ERP, TRF, template | 30 min |
| 6 | Decide: closed loop live, or on recorded data | 5 min |
| 7 | Closed loop (built; check + run) | ~40 min |
| 8 | Demo prep | 1–2 h |

---

## 0. Commit and branch (10 min)

**Nothing from day 1 is committed yet.** Do this first.

```powershell
git status                                  # data/, models/, *.h5 must NOT appear
git checkout -b setup-day1
git add -A
git commit -m "Composer, acquisition and analysis pipeline"
git push -u origin setup-day1               # if there's a remote
```

Don't commit the test figures in `figures/` (`*_dry_*`, `simulated_*`). Then make one branch per person (CLAUDE.md).

---

## 1. Headset on, signal check (30–45 min)

**Setup**
- Charge the headset fully before you start.
- Put the **GND/REF electrode on the right mastoid**, the bone behind the ear.
- Seat each electrode on the scalp, parting hair under it. The hybrid electrodes can be used dry or slightly wet; wetting improves contact.
- **Turn off Bluetooth headphones and phones nearby.** They compete for the radio, and Windows may send audio to them.

**View the signal:**

```powershell
python -m acquisition.view_stream --raw --range-uv 200    # contact check first
python -m acquisition.view_stream                          # then the filtered view, ±50 µV
```

**Checklist** (details in the day-1 conversation):

| Test | Expected | If not |
|---|---|---|
| 8 labeled traces | Fz C3 Cz C4 Pz PO7 POz PO8 | see Troubleshooting below |
| Raw, still | steady traces | flat = disconnected, pinned = no contact: re-seat it |
| Hum (raw vs. filtered) | small 60 Hz hum, removed by the filter | heavy hum on **all** channels = poor mastoid GND/REF contact |
| Sitting still, filtered | about ±10–30 µV | one noisy channel: re-seat or wet it |
| Hard blinks | big slow spikes, largest at **Fz** | if mostly elsewhere, channel order is wrong |
| Jaw clench | fuzz burst | normal; listener must keep jaw relaxed |
| **Eyes closed 10 s** | **~10 Hz waves at PO7/POz/PO8** | best proof you're recording brain activity |
| Tap one electrode | spike on that channel only | checks channel mapping |

**Close the viewer window before recording.** The headset accepts only one connection.

### Troubleshooting the connection

| Message | Meaning | Fix |
|---|---|---|
| `GDS library not found` (first 2 lines) | Harmless: GDS is for g.tec's wired amplifiers | Ignore |
| `heard no Bluetooth peripheral at all` | Headset isn't advertising | Power-cycle it; make sure it's charged and not connected to another laptop or app; move closer |
| Not found, but other devices heard | Wrong serial | Check the label against `SERIAL_NUMBER` in `config.py`, or pass `--serial U8-...` |
| Was connected yesterday, now fails | Another process still holds it | Close other Python windows and the Core Recorder; power-cycle the headset |
| Don't | Pair it in Windows Settings | g.Pype connects directly; a Windows pairing can block it. Remove it if someone paired it |

---

## 2. Audio check (10 min)

Use **wired headphones**. Bluetooth audio adds variable delay, which smears the ERPs.

```powershell
python -m audio.playback                      # 6 test tones; prints the output device
python -m composer.sample --play 1.0          # one generated melody
```

- **Check the printed `Output device`.** It must be the wired headphones, not the Bluetooth ones (WH-CH520, Bose QC) or the laptop speakers. If it's wrong, change the Windows default output device and rerun.
- **Timbre:** notes use a piano-like tone (`TIMBRE = "piano"` in `config.py`; `"sine"` for pure tones). Pick one now and **keep it for calibration and the live loop**, since the template is fitted on that sound. Compare: `python -m audio.playback`, then set `TIMBRE = "sine"` and run it again.
- Set a comfortable, moderate volume, then **don't change it** for the rest of the session. Volume isn't recorded. The piano tone peaks higher than the sine at the same average loudness, so set the volume with the timbre you'll actually use.
- If you see `underflow` warnings, close other programs (browser, Zoom, Teams).

---

## 3. Oddball sanity check (20–30 min)

```powershell
python -m acquisition.oddball --dry-run                   # headset on, silent, ~30 s
python -m acquisition.oddball --subject 1 --run 1         # real: 400 tones, ~3 min
```

**Listener instructions:** sit relaxed, jaw loose, eyes on a fixed point, minimal movement. Blinking is fine, but try not to blink in bursts.

**Read the output:**
1. **Sync check:** all 5 lines should say `PASS`.
2. **Epochs kept:** should be most of them (over 80%).
3. **ERP figure** (`figures/..._oddball_1_erp.png`): look for a dip around 100 ms (N1) and a rise around 200 ms (P2) at Fz/Cz on **both** curves. A bonus is a more negative deviant curve at 100–250 ms (mismatch negativity). With ~60 deviants it may be noisy.

| Problem | Likely cause | Fix |
|---|---|---|
| `marker jitter` FAIL | CPU busy | Close other apps; rerun |
| `EEG clock agrees with host clock` FAIL | Bluetooth dropped EEG samples | Move the laptop closer; turn off other Bluetooth devices |
| `one marker per tone` FAIL | Recording stopped early, or a marker was lost | Check the warnings; rerun |
| Many epochs rejected | Poor contact or movement | Back to step 1; re-seat noisy channels |
| No N1/P2 at all | Volume too low, wrong audio device, or bad Fz/Cz contact | Check step 2; check the Fz/Cz traces in the viewer |
| Unclear ERP | Too few trials | `--run 2`, then compare. Both runs together are a stronger check |

To re-plot later: `python -m analysis.erp --session oddball --subject 1 --run 1 --date YYYY-MM-DD`. Without `--date` it looks for today's file.

**Don't continue to calibration until the oddball shows an N1/P2.** If the basic auditory response isn't visible, the surprisal analysis won't work either.

---

## 4. Calibration recording (~25 min including breaks)

```powershell
python -m composer.markov --train          # only if models/markov.pkl is missing
python -m acquisition.record --dry-run     # headset, silent, one 30 s block
python -m acquisition.record --subject 1   # real: 5 blocks × ~2.7 min, break between blocks
```

- After each block: the sync check should be all `PASS`, and the surprisal summary should rise with temperature (about 1.2 bits at T = 0.5 up to about 5 at T = 2.0).
- **Breaks:** the script waits for Enter. Use the break to let the listener blink and stretch, and to check electrodes if a block looked noisy.
- **Listener instructions:** same as the oddball. Just listen; no task.
- **Interrupted or crashed:** the finished blocks are saved. The script prints the resume command, e.g. `python -m acquisition.record --subject 1 --start-block 3 --overwrite`. `--overwrite` replaces only the cut-short block.
- **More data helps** the TRF analysis. If time allows, add blocks: `python -m acquisition.record --subject 1 --start-block 6 --blocks 8`.

---

## 5. Analysis (30 min)

```powershell
python -m analysis.trf --simulate --no-show          # 20 s: confirm the code still finds a known effect
python -m analysis.erp --session calibration --subject 1
python -m analysis.trf --subject 1 --save-template
```

**What "it worked" looks like:**
- **ERP:** a difference between high- and low-surprisal notes around 100–250 ms at Fz/Cz. It's confounded with interval size, so treat it as a first look.
- **TRF:** `A+M beats A on held-out melodies in k/5 folds`. **4/5 or 5/5 is the result.** Expect a small Δr, about +0.01; even the simulated effect gave only +0.0095. Consistency across folds is what matters.
- **Template:** `models/subj01_calibration_template.npz`. The surprisal TRF in the figure is the listener's template. It will look delayed by ~40 ms, with a rebound after the main peak; that's the causal filter and it's expected.

| Problem | What to do |
|---|---|
| `bad channels: Fz` (or Cz) | Analysis runs without it, but the main channels matter most. Re-seat and record more blocks if there's time |
| A+M doesn't beat A | Record more blocks (step 4). Check the ERP and the oddball. Fall back to analyzing the Di Liberto dataset (README "Fallback") |
| `no calibration recordings for subject 1` | Wrong subject or date; check `data/raw/`. `--date` filters by date |

---

## 6. Decide (5 min)

- **The template looks good (≥4/5 folds):** run the live closed loop (step 7).
- **Weak or unclear:** record more calibration blocks first (step 4). If it's still weak, demo the loop with `run_live --dry-run` on a listener anyway, and present the offline gain check and the simulation as the method demo. The README lists this as the fallback tier.

---

## 7. Closed loop (~40 min)

Already built: `closed_loop/gain.py` (gain estimate, temperature rule, offline checks) and `closed_loop/run_live.py` (live loop).

```powershell
python -m closed_loop.gain --simulate                    # 10 s: gain tracks a known effect (expect ≈ 0, 0.5, 1, 2)
python -m closed_loop.gain --subject 1                   # gain per window on calibration data: mean should be ≈ 1
python -m closed_loop.run_live --dry-run                 # headset, silent, 3 min, subject 0 → needs --template
python -m closed_loop.run_live --subject 1               # real: 15 min, uses models/subj01_calibration_template.npz
```

- For the dry run with the real template: `python -m closed_loop.run_live --dry-run --template models/subj01_calibration_template.npz`.
- **Each window prints** its temperature, the gain, the smoothed gain and the number of notes used. Every 3rd window it prints the decision, e.g. `rose: T 1.25 -> 1.5`.
- **Outputs:** EEG and note log like any session, plus `..._live_1_loop.csv` (one row per window) and the figure `figures/..._live_1_loop.png` (temperature and gain over time).
- **Options:** `--minutes 20` for longer, `--start-temperature 0.5` to start low, `--window 50` for the window length in seconds.
- **Expect noise.** A single 50 s window's gain has an SD of about 0.25–0.3, even on clean simulated data, which is why decisions average 3 windows. On real data, a smoothed gain that stays clearly above 0 shows the template is being picked up live. Whether the temperature settles anywhere in 15 minutes is a bonus.
- **Time budget:** each decision takes about 2.5 minutes (3 × 50 s windows), so 15 minutes gives about 6 temperature steps.
- **If gains are all near 0:** check `closed_loop.gain --subject 1` on the calibration data. If that's also near 0, the template is weak (back to step 5). If calibration gives ≈ 1 but live gives ≈ 0, check contact quality, and check that `TIMBRE` and the volume are the same as during calibration.

---

## 8. Demo prep (1–2 h)

- **Figures to show:** oddball ERP, calibration ERP, TRF comparison (fold plot + template), and the closed-loop temperature and gain over time.
- **Wording** (CLAUDE.md): say "neural surprise response" or "prediction-error response", not "the listener was surprised". Don't claim it measures enjoyment.
- **Limitations** to state: model ≠ listener; 8 channels and short windows are noisy; simple sine-tone melodies; audio delay not measured.
- Update the README "Analysis plan" status table with what worked.
- Commit, without the data.

---

## Quick reference

| Command | What |
|---|---|
| `python -m acquisition.view_stream [--raw]` | live signal check |
| `python -m audio.playback` | test tones and output device |
| `python -m composer.sample --play T` | hear a melody at temperature T |
| `python -m acquisition.oddball [--dry-run] --subject N --run R` | oddball + sync check + ERP |
| `python -m acquisition.record [--dry-run] --subject N [--start-block B]` | calibration blocks |
| `python -m analysis.erp --session oddball\|calibration --subject N` | ERPs |
| `python -m analysis.trf --subject N [--save-template]` | A vs. A+M, template |
| `python -m analysis.trf --simulate` | check the TRF code |
| `python -m acquisition.violation [--dry-run] --subject N` | planted-violation blocks |
| `python -m analysis.erp --session violation --subject N` | pre-registered violation test |
| `python -m closed_loop.gain --simulate` / `--subject N` | check the gain estimate |
| `python -m closed_loop.run_live [--dry-run] --subject N` | live closed loop |
| Add `--fake-eeg` (recording scripts) | no headset needed |

**Files:** recordings go to `data/raw/` (`YYYY-MM-DD_subjNN_<session>_<run>.h5` + `_notes.csv`), figures to `figures/`, models and templates to `models/`. None of `data/` or `models/` is committed.
