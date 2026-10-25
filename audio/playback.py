"""Tone synthesis and sample-accurate isochronous playback.

Every note has the same loudness, timbre, duration and onset ramps; only
pitch varies.

Timing: a whole block of notes is rendered into one array, so onsets sit
on exact sample positions and the sound card's clock keeps the rhythm.
An audio callback feeds the array to the device and, for each onset,
records PortAudio's estimate of when that sample reaches the DAC.
play_with_onsets() waits for that moment and calls on_onset(k), so a
marker emitted there lines up with the sound.

Measured on a Windows laptop (default MME backend): marker jitter
~0.4 ms std. The earlier blocking-write approach jittered ~8 ms because
writes return in ~20 ms driver blocks. Any remaining offset (driver
latency estimate vs. true speaker time) is constant and only shifts ERPs.
"""

import argparse
import threading
import time

import numpy as np
import sounddevice as sd

import config

LEAD_SILENCE = 0.5  # s before the first onset, so the stream is running steadily


def midi_to_hz(midi):
    """MIDI pitch number(s) to frequency in Hz (A4 = 69 = 440 Hz)."""
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=float) - 69.0) / 12.0)


def hz_to_midi(hz):
    """Frequency in Hz to (fractional) MIDI pitch number."""
    return 69.0 + 12.0 * np.log2(np.asarray(hz, dtype=float) / 440.0)


def make_tone(freq_hz, duration=config.NOTE_DURATION, sfreq=config.AUDIO_SFREQ,
              amplitude=config.TONE_AMPLITUDE, ramp=config.RAMP_DURATION):
    """Sine tone with raised-cosine onset/offset ramps, as float32."""
    n = int(round(duration * sfreq))
    t = np.arange(n) / sfreq
    tone = amplitude * np.sin(2 * np.pi * freq_hz * t)
    n_ramp = int(round(ramp * sfreq))
    if n_ramp > 0:
        rise = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_ramp)))
        tone[:n_ramp] *= rise
        tone[-n_ramp:] *= rise[::-1]
    return tone.astype(np.float32)


def render_sequence(freqs_hz, ioi=config.IOI, sfreq=config.AUDIO_SFREQ, silent=False):
    """Render notes back to back, one every IOI.

    Returns (audio, onsets): a float32 array and the sample index of each
    note onset. silent=True renders zeros with identical timing (dry runs).
    """
    if config.NOTE_DURATION >= ioi:
        raise ValueError("NOTE_DURATION must be shorter than IOI")
    step = int(round(ioi * sfreq))
    lead = int(round(LEAD_SILENCE * sfreq))
    n_tone = int(round(config.NOTE_DURATION * sfreq))
    onsets = lead + step * np.arange(len(freqs_hz))
    audio = np.zeros(lead + step * len(freqs_hz), dtype=np.float32)
    if not silent:
        tones = {f: make_tone(f, sfreq=sfreq) for f in set(freqs_hz)}
        for f, start in zip(freqs_hz, onsets):
            audio[start:start + n_tone] = tones[f]
    return audio, onsets


def play_with_onsets(audio, onsets, on_onset, sfreq=config.AUDIO_SFREQ):
    """Play audio, calling on_onset(k) (k = 0, 1, ...) as each onset reaches the DAC.

    on_onset runs on the calling thread, so g.Pype's Marker.emit timestamps
    it at the right moment. Returns the number of output underflows
    (should be 0; more means the computer couldn't keep up).
    """
    dac_times = [None] * len(onsets)
    state = {"pos": 0, "underflows": 0}
    lock = threading.Lock()

    def callback(outdata, frames, time_info, status):
        if status.output_underflow:
            state["underflows"] += 1
        pos = state["pos"]
        seg = audio[pos:pos + frames]
        outdata[:len(seg), 0] = seg
        outdata[len(seg):, 0] = 0
        dac = time_info.outputBufferDacTime or time_info.currentTime
        first, last = np.searchsorted(onsets, [pos, pos + frames])
        with lock:
            for k in range(first, last):
                dac_times[k] = dac + (onsets[k] - pos) / sfreq
        state["pos"] = pos + frames

    with sd.OutputStream(samplerate=sfreq, channels=1, dtype="float32",
                         latency="low", callback=callback) as stream:
        for k in range(len(onsets)):
            while True:
                with lock:
                    due = dac_times[k]
                if due is not None:
                    break
                time.sleep(0.001)
            while stream.time < due:
                time.sleep(0.0005)
            on_onset(k)
        # let the rest of the audio (the last tone) finish
        end = due + (len(audio) - onsets[-1]) / sfreq
        while stream.time < end:
            time.sleep(0.01)
    return state["underflows"]


def main():
    ap = argparse.ArgumentParser(description="Play test tones and report onset timing.")
    ap.add_argument("--hz", type=float, nargs="+", default=[1000.0, 1200.0])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    print(f"Output device: {sd.query_devices(kind='output')['name']}")
    audio, onsets = render_sequence(args.hz * args.repeats)
    times = []
    underflows = play_with_onsets(audio, onsets, lambda k: times.append(time.perf_counter()))
    spacing = np.diff(times) * 1000
    print(f"onset spacing {spacing.mean():.1f} ± {spacing.std():.2f} ms "
          f"(IOI {config.IOI * 1000:.0f} ms), underflows: {underflows}")


if __name__ == "__main__":
    main()
