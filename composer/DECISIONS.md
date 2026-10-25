# Composer decisions

What the Markov composer (`composer/`) assumes, and why. **Settled** means build on it. **Tentative** means use it now, then confirm with the held-out surprisal test. **Open** means it still needs an answer. Update this file when a decision changes.

**What "melody" means.** In training, a melody is the **whole soprano line of one chorale**, first note to last: 368 melodies, median 46 notes (22–185). Rests don't split it (see #11). In generation and the experiment, a melody is one generated sequence of `config.MELODY_LENGTH` notes. Each has a `melody_id` in the note log and is the unit that cross-validation splits by.

**The held-out surprisal test:** train on 80% of the chorales, split by whole chorale, never by note. Then compare the average surprisal per note (bits) on the other 20%. Lower is better: that model predicts unseen Bach better, which makes it a better stand-in for a listener's expectations.

## Settled

| # | Decision | Why |
|---|---|---|
| 1 | Training data: the soprano part of each music21 Bach chorale, chosen **by part name**, not position: the first part whose name starts with "soprano" (case-insensitive). 368 melodies, 18,511 notes. The 3 chorales with unnamed parts are skipped. | In 21 chorales the first part is an instrument, not the soprano. `startswith` also catches "Soprano 1" (#149) and "Soprano Oboe 1 Violin1" (#10, the soprano voice with its doubling instruments named). |
| 2 | Model: nested dictionary, context tuple → counts of the next pitch. | Sampling and normalizing need all continuations of one context. |
| 3 | Store **counts**; turn them into probabilities only at lookup. | Smoothing and back-off both need raw counts. |
| 4 | Pad the start of each melody with a `START` symbol (`None`). | Gives the first notes a context, and the model learns how melodies begin. |
| 5 | Add-α smoothing over the vocabulary: P = (c + α) / (N + α·V), with V = the pitches in `config.PITCH_RANGE`. | No zero probabilities: surprisal stays finite, and generation can produce transitions Bach never used. |
| 6 | Back off to a shorter context when a context was never seen. | Smoothing an unseen context gives a uniform guess, which wastes what the model knows. |
| 7 | Surprisal = −log₂ P(note \| context) under the **untempered** model. | CLAUDE.md invariant 1: values stay comparable across temperatures. |
| 8 | Temperature: raise probabilities to the power 1/T and renormalize. Used for sampling only. | Controls how predictable the music is. Work in log space to avoid underflow. |
| 9 | A tied note counts as **one** note; grace notes are kept (only 3 in the corpus). | Matches what a listener hears as separate notes. |
| 10 | Rhythm is ignored: every generated note has the same duration and inter-onset interval. | CLAUDE.md invariant 8: only pitch varies. Chorale sopranos are mostly steady quarter notes anyway. |
| 20 | **Repeated notes are merged** before training: a run of the same pitch becomes one note (`config.MERGE_REPEATS`). Costs 2,777 of 18,511 notes (15%); 15,734 remain, median 40 per melody. The cache keeps the raw melodies, so this is a switch. | In Bach, a repeat carries a new syllable, rhythm and chord. Played as identical isochronous sine tones it just sounds static, and the generator repeated notes as often as Bach did (15.8% vs 15.3%). After merging, generated repeats drop to 0–1.4%. |
| 21 | **Melodies end on the tonic** when possible: after `MELODY_LENGTH` notes, sampling continues until the model produces a C, up to `MAX_EXTRA_NOTES` (8) more (`config.END_ON_TONIC`). Nothing is forced, and surprisal is unchanged. | Melodies sound finished instead of stopping mid-phrase. Reaches the tonic in 90% of melodies at T = 0.5 and 71% at T = 2.0. Melodies are 32–40 notes, so block durations vary slightly. |
| 18 | **Random key per melody at playback:** the composer works in C; each melody is shifted by a random −5 to +6 semitones (all 12 keys, smallest register moves) just before it's played (`config.RANDOM_KEY`, `KEY_SHIFT_RANGE`; `sample.random_key_shift`). The note log's `pitch` is the **played** pitch; surprisal and interval are unaffected by the shift. Played range: MIDI 50–90 (147–1480 Hz). | Decorrelates absolute pitch height (a TRF control feature) from surprisal, and sounds less samey. Cost: a new register at each melody start can enlarge the first notes' response, so the analysis may drop each melody's first note or two. |
| 11 | **Phrases are ignored** for now: no fermata boundaries, each chorale's soprano line is one continuous sequence. Rests are skipped, not used as split points: the notes on either side of a rest become adjacent. | Simplest option that touches no other module. Cost: phrase-boundary jumps are learned as ordinary transitions. See "Deferred" for the alternative. |
| 17 | Transpose each melody so its tonic (from music21's key estimate) is C, moving in **whichever direction is shorter** (at most 6 semitones). Keep the resulting register; don't shift octaves. | Keeps the register close to the original. Result: MIDI 55–84 (median 72, middle 90% 62–79), so **no note falls outside `PITCH_RANGE`** and nothing needs folding or dropping. Register doesn't affect what the model learns (shifting everything by 12 only relabels the counts); it only sets the vocabulary bounds and what's heard at playback (196–1047 Hz). |

## Tentative (confirm with held-out surprisal)

| # | Current choice | Alternatives to compare |
|---|---|---|
| 12 | Order 2 (`config.MARKOV_ORDER`). **Confirmed 2026-10-25:** best held-out surprisal, 2.272 bits (order 1: 2.442, order 3: 2.277, with α = 0.1). Order 3 overfits: 1.73 bits on training data vs. 2.34 held-out at α = 0.01. | Order 1, order 3 |
| 13 | α = 0.1 (`config.SMOOTHING_ALPHA`). **Confirmed 2026-10-25:** best for orders 2 and 3 (`python -m composer.markov --evaluate`). Re-checked after merging repeats (#20): order 2 / α = 0.1 is still best (1.965 bits held-out; order 3: 1.981). Bits before and after #20 aren't comparable, since the data changed. | 0.01, 1 |
| 14 | Major and minor transposed to the **same tonic** (e.g. C major and C minor). The corpus is 194 major / 174 minor by music21's estimate. |
| 19 | Tonic from `chorale.analyze('key')`. It agrees with the final bass note in 306/363 chorales and works on all 368. | The key stored in the file (363 of 368 have one; it agrees with the final bass note in only 282), or the final bass note itself as the tonic. None is perfect: modal chorales (e.g. Phrygian *Ach Gott, vom Himmel sieh' darein*) legitimately end off the analyzed tonic. | Relative keys (C major / A minor), or separate major and minor models |

## Open

| # | Question | Notes |
|---|---|---|
| 16 | Pitch representation: **semitones from the tonic, keeping the octave**, or scale degrees? | Scale degrees can't represent chromatic notes, which Bach uses often. Pitch class (no octave) can't tell a step up from a seventh down. Semitones-with-octave keeps both. |

## Deferred: fermata phrase boundaries (#15)

The idea: replace each fermata (2,236 in the sopranos, about every 8 notes) with a boundary token. The generator then outputs boundaries as **silent beats** of one inter-onset interval, so phrases become audible while every onset stays on the beat grid.

Corpus facts: 216 rests occur mid-melody, 178 of them right after a fermata. Only 3 are eighth rests.

Before adopting it, decide:
- **Markers and the note log:** a silent beat has no note, so it gets no marker and no note-log row. What is `interval` for the first note after a silence?
- **EEG confound:** the note after a silence produces a larger auditory response whatever its surprisal, and phrase starts tend to be unusual. You'd need a "follows a silence" feature among TRF model A's acoustic controls (an amendment to CLAUDE.md invariant 6).
- **Generation:** does `MELODY_LENGTH` count silent beats? Can two boundaries occur in a row?

Adopt it only if it gives clearly lower held-out surprisal than decision 11, and the changes above are made at the same time.
