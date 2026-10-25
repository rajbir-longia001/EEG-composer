"""Shared constants for every module. Import from here; never hard-code these elsewhere."""

from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"

SESSIONS = ("oddball", "calibration", "live")

# --- Hardware (g.tec Unicorn BCI Core-8 via g.Pype, Bluetooth 5, 250 Hz) ---
# Acquisition uses gp.BCICore(). Read the sampling rate from the port context or
# recording at runtime rather than hard-coding it.
DEVICE = "Unicorn BCI Core-8"
SERIAL_NUMBER = "U8-2025.10.15"
# gp.BCICore's default 8-channel montage. Verify on the cap.
CH_NAMES = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "POz", "PO8"]
CENTRAL_CHANNELS = ["Fz", "Cz"]
# Extra channels BCICore appends after the EEG; drop them before building MNE objects.
TIMELINE_CHANNELS = ["master_index", "valid", "host_time"]
MONTAGE = "colin27_1020"  # MNE's replacement for the deprecated "standard_1020"
UV_TO_V = 1e-6  # gp.BCICore outputs µV; MNE expects volts
EEG_EXT = ".h5"  # gp.HDF5Writer keeps units and channel roles; CSV does not

# --- Note log contract (do not rename or drop columns) ---
NOTE_LOG_COLUMNS = [
    "note_index", "melody_id", "onset_time", "pitch",
    "interval", "surprisal", "temperature",
]
FIRST_NOTE_INDEX = 1  # marker 0 means "no marker" in the marker channel

# --- Audio / stimulus (constant acoustics: only pitch varies) ---
AUDIO_SFREQ = 44100
IOI = 0.45            # inter-onset interval, s (isochronous)
NOTE_DURATION = 0.35  # tone length, s (must be < IOI)
RAMP_DURATION = 0.005  # onset/offset ramps, s
TONE_AMPLITUDE = 0.2   # fixed loudness, 0..1 (sine: peak; piano: same RMS as that sine)
TIMBRE = "piano"       # "sine" or "piano"; must match between calibration and live
PIANO_HARMONICS = 6    # partials in the piano-like tone
PIANO_DECAY = 0.25     # s, decay time constant of the fundamental (higher partials decay faster)

# --- Composer ---
MARKOV_ORDER = 2        # previous pitches used as context
SMOOTHING_ALPHA = 0.1   # additive smoothing for unseen transitions
PITCH_RANGE = (55, 84)  # MIDI pitches the model can emit (G3..C6)
MELODY_LENGTH = 32      # notes per generated melody (before the tonic ending)
MERGE_REPEATS = True    # train on melodies with repeated notes merged (DECISIONS #20)
TONIC_PC = 0            # every melody is transposed so its tonic is C
END_ON_TONIC = True     # keep sampling past MELODY_LENGTH until a tonic (DECISIONS #21)
MAX_EXTRA_NOTES = 8     # give up on the tonic ending after this many extra notes
RANDOM_KEY = True       # transpose each melody to a random key at playback (DECISIONS #18)
KEY_SHIFT_RANGE = (-5, 6)  # semitones, inclusive: all 12 keys, smallest register moves
DEFAULT_TEMPERATURE = 1.0

# --- Oddball sanity check ---
ODDBALL_STANDARD_HZ = 1000.0
ODDBALL_DEVIANT_HZ = 1200.0
ODDBALL_DEVIANT_PROB = 0.15
ODDBALL_N_TONES = 400
ODDBALL_DRY_RUN_TONES = 30   # default length of a --dry-run
ODDBALL_LEAD_STANDARDS = 5   # standards before the first deviant possible

# --- Recording timing ---
RECORD_SETTLE = 2.0  # s of EEG before the first tone, so the stream is steady
RECORD_TAIL = 1.0    # s of EEG after the last tone, so its epoch is complete

# --- Recording protocol ---
BLOCK_DURATION = 150.0  # s, 2–3 minute blocks
CALIBRATION_TEMPERATURES = [0.5, 1.0, 1.5, 2.0]
CALIBRATION_BLOCKS = 5     # 5 × 150 s ≈ 12.5 min of melodies
DRY_RUN_BLOCK_DURATION = 30.0  # s, one short block for --dry-run
MELODY_GAP_SLOTS = 3       # silent IOI slots between melodies (3 × 0.45 s = 1.35 s)

# --- Preprocessing ---
# Keep the hardware reference: the Core-8 has one GND/REF electrode (shorted internally);
# put it on the right mastoid. An average reference over 8 mostly central channels
# would cancel much of the Fz/Cz response.
ERP_BAND = (1.0, 30.0)  # Hz
TRF_BAND = (1.0, 8.0)   # Hz
TRF_DECIM = 4           # keep every 4th sample after the causal lowpass: 250 → 62.5 Hz.
                        # (64 Hz isn't an integer ratio of 250, and resampling to it
                        # would need a non-causal filter the live loop can't use.)
BAD_FLAT_UV = 0.5       # channel std (1–30 Hz) below this = flat / disconnected
BAD_NOISY_FACTOR = 4.0  # channel std above this × median std = noisy
ARTIFACT_PAD = 0.25     # s masked either side of a TRF sample over ERP_REJECT
NOTCH_BANDS = [(48.0, 52.0), (58.0, 62.0)]  # Hz, 50 and 60 Hz mains (US is 60)
CAUSAL_FILTER_ORDER = 4  # Butterworth order for gp.Bandpass / scipy sosfilt (live + template fit)

# --- ERPs ---
ERP_TMIN, ERP_TMAX = -0.1, 0.5  # s
ERP_BASELINE = (None, 0)
ERP_REJECT = {"eeg": 120e-6}  # volts (= 120 µV); MNE thresholds are in volts

# --- TRFs ---
TRF_TMIN, TRF_TMAX = -0.1, 0.5  # lags, s
RIDGE_ALPHAS = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]  # chosen by nested CV
CV_FOLDS = 5  # GroupKFold over melody_id

# --- Closed loop ---
WINDOW_DURATION = 50.0     # s per gain estimate
GAIN_SMOOTHING_WINDOWS = 3  # windows averaged before each temperature decision
TEMPERATURE_STEP = 0.25
TEMPERATURE_BOUNDS = (0.3, 3.0)
