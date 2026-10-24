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
# gp.BCICore's default 8-channel montage. Verify on the cap.
CH_NAMES = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "POz", "PO8"]
CENTRAL_CHANNELS = ["Fz", "Cz"]
# Extra channels BCICore appends after the EEG; drop them before building MNE objects.
TIMELINE_CHANNELS = ["master_index", "valid", "host_time"]
MONTAGE = "standard_1020"
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
TONE_AMPLITUDE = 0.2   # fixed loudness, 0..1

# --- Composer ---
MARKOV_ORDER = 2        # previous pitches used as context
SMOOTHING_ALPHA = 0.1   # additive smoothing for unseen transitions
PITCH_RANGE = (55, 84)  # MIDI pitches the model can emit (G3..C6)
MELODY_LENGTH = 32      # notes per generated melody
DEFAULT_TEMPERATURE = 1.0

# --- Oddball sanity check ---
ODDBALL_STANDARD_HZ = 1000.0
ODDBALL_DEVIANT_HZ = 1200.0
ODDBALL_DEVIANT_PROB = 0.15
ODDBALL_N_TONES = 400

# --- Recording protocol ---
BLOCK_DURATION = 150.0  # s, 2–3 minute blocks
CALIBRATION_TEMPERATURES = [0.5, 1.0, 1.5, 2.0]

# --- Preprocessing ---
# Keep the hardware reference: the Core-8 has one GND/REF electrode (shorted internally);
# put it on the right mastoid. An average reference over 8 mostly central channels
# would cancel much of the Fz/Cz response.
ERP_BAND = (1.0, 30.0)  # Hz
TRF_BAND = (1.0, 8.0)   # Hz
TRF_SFREQ = 64          # Hz after downsampling
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
