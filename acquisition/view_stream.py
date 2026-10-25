"""Live view of the Unicorn BCI Core-8 for checking signal quality.

Adapted from g.Pype's training example (Season 3, Episode 1): the
amplifier streams into the ERP bandpass and mains notches from config.py,
shown in a scrolling scope. Nothing is recorded.

Close this window before running a recording script: the amplifier
accepts only one connection at a time.

Usage (from the repo root):
    python -m acquisition.view_stream
    python -m acquisition.view_stream --serial UN-XXXX.XX.XX   # another amplifier
    python -m acquisition.view_stream --raw                    # unfiltered view

Requires: pip install "gpype[all]"
"""

import argparse

import gpype as gp

import config


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serial", default=config.SERIAL_NUMBER,
                    help="amplifier serial (default: config.SERIAL_NUMBER)")
    ap.add_argument("--range-uv", type=float, default=50,
                    help="scope amplitude range in µV (default ±50)")
    ap.add_argument("--raw", action="store_true", help="show the unfiltered signal")
    args = ap.parse_args()

    app = gp.MainApp()
    p = gp.Pipeline()

    # The given amplifier, at the rate it reports.
    # Core-8 channels are named Fz C3 Cz C4 Pz PO7 POz PO8 by default.
    source = gp.BCICore(serial=args.serial)

    scope = gp.TimeSeriesScope(amplitude_limit=args.range_uv, time_window=10)

    if args.raw:
        p.connect(source, scope)
    else:
        # ERP band keeps brain rhythms, drops drift and muscle noise;
        # notches cover 50 and 60 Hz mains
        f_lo, f_hi = config.ERP_BAND
        node = gp.Bandpass(f_lo=f_lo, f_hi=f_hi)
        p.connect(source, node)
        for lo, hi in config.NOTCH_BANDS:
            notch = gp.Bandstop(f_lo=lo, f_hi=hi)
            p.connect(node, notch)
            node = notch
        p.connect(node, scope)

    app.add_widget(scope)
    try:
        p.start()
        app.run()  # blocks until the window is closed
    finally:
        p.stop()
        p.close()


if __name__ == "__main__":
    main()
