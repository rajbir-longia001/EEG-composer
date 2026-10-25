"""Shared pieces of a recording session: file names, the g.Pype recording
pipeline, and the note log.

Recording pipeline (one per block):

    BCICore (or a fake Generator) --+
                                    +--> Router --> HDF5Writer
    Marker  ------------------------+

marker.emit(note_index) puts a one-sample pulse with value note_index on
the marker channel, on the same timeline as the EEG samples.
"""

import datetime as dt
import glob
import os

import gpype as gp
import pandas as pd

import config


def recording_paths(subject, session, run, date=None):
    """(eeg_path, notes_path) following YYYY-MM-DD_subjNN_<session>_<run>."""
    if session not in config.SESSIONS:
        raise ValueError(f"session must be one of {config.SESSIONS}")
    date = date or dt.date.today().isoformat()
    stem = f"{date}_subj{int(subject):02d}_{session}_{run}"
    return (config.DATA_RAW / f"{stem}{config.EEG_EXT}",
            config.DATA_RAW / f"{stem}_notes.csv")


def build_pipeline(eeg_path, serial=None, fake_eeg=False):
    """Build (but don't start) the recording pipeline.

    Returns (pipeline, marker). fake_eeg=True uses g.Pype's signal generator
    instead of the headset, so the whole session can be tested without
    hardware.
    """
    p = gp.Pipeline()
    marker = gp.Marker()
    both = gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL])
    p.connect(marker, both["in2"])

    if fake_eeg:
        source = gp.Generator(sampling_rate=250, channel_count=len(config.CH_NAMES),
                              signal_frequency=10, signal_amplitude=10,
                              noise_amplitude=10)
        p.connect(source, both["in1"])
        # A generator has no channel names or units; give it the Core-8's
        labeled = gp.ChannelLabeler(labels=config.CH_NAMES + ["marker"],
                                    units=["uV"] * len(config.CH_NAMES) + [None])
        p.connect(both, labeled)
        last = labeled
    else:
        source = gp.BCICore(serial=serial)
        p.connect(source, both["in1"])
        last = both

    # The writer appends _YYYYMMDD_HHMMSS_mmm to this name; see finalize_eeg_file
    p.connect(last, gp.HDF5Writer(file_name=str(eeg_path)))
    return p, marker


def finalize_eeg_file(eeg_path):
    """Rename the writer's timestamped file to eeg_path. Returns eeg_path."""
    stem, ext = os.path.splitext(str(eeg_path))
    written = sorted(glob.glob(f"{glob.escape(stem)}_*{ext}"))
    if not written:
        raise FileNotFoundError(f"g.Pype wrote no file matching {stem}_*{ext}")
    os.replace(written[-1], eeg_path)
    return eeg_path


def write_note_log(rows, notes_path):
    """Save the note log with exactly config.NOTE_LOG_COLUMNS."""
    df = pd.DataFrame(rows, columns=config.NOTE_LOG_COLUMNS)
    df.to_csv(notes_path, index=False)
    return df
