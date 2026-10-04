# MelodyMorph

**A Transformer-based melody continuation and variation generator.**

Give it a short musical motif (5–15 notes) and MelodyMorph generates several
*continuations* (what comes next) or *variations* (the motif re-imagined with
its rhythm kept and new pitches). It's a decoder-only Transformer trained on
monophonic folk melodies.

This is a Lab 3 (Unit II: Autoregressive Models + Transformers) case-study
prototype. It works on symbolic MIDI, not raw audio, and it does one narrow
generation task rather than end-to-end song generation.

## How it works

```
seed -> key-normalise -> REMI tokens -> decoder-only Transformer -> grammar-masked
sampling (batched, KV-cached) -> k candidates -> transpose back -> MIDI / piano roll / audio
```

- **Data**: melodies from the `music21` package's bundled corpus (Essen
  folksongs + Bach chorale soprano lines), so there's nothing to download.
  - Duple and triple metres are kept: 2/4, 4/4, 2/2, 3/4, 6/8, 3/8, plus 4/2,
    3/2 and 6/4, which are rescaled to half note values. That's 7.4k tunes; tunes
    with changing or irregular metres are skipped.
  - Pickups stay at the end of bar 0, so every `BAR` token is a real barline.
    About 90% of true downbeats land on one, against 28% before the rework.
  - Tunes are transposed to C major / A minor and exact duplicates are dropped.
  - The train/val split hashes each tune's opening, so near-duplicates can't
    straddle it.
- **Tokenizer**: REMI-style, 75 tokens: a time-signature token (`TS_16` for
  duple bars, `TS_12` for triple), then `BAR`, in-bar `POS` (16th grid),
  `PITCH` (C3–C6) and `DUR` (1–16 sixteenths). The time-signature tokens come
  last, so 73-token checkpoints from before them still load; they handle 4/4
  seeds only.
- **Model**: a from-scratch decoder-only Transformer: 4 layers, 4 heads,
  d_model 256, 3.2M parameters. Training windows start at a tune's `BOS` and
  never mix two tunes, which matches how the model is prompted. A KV cache makes
  sampling incremental.
- **Sampling**: temperature / top-p under a grammar mask.
  - The mask makes ill-formed token streams and overlapping notes impossible,
    and stops each candidate exactly on the target bar.
  - The optional repetition penalty applies only to pitches.
- **Variation mode**: the model reads the whole seed, then the seed's rhythm
  (BAR/POS/DUR) is replayed while every pitch is sampled. Candidates are ranked
  by interval-contour similarity, preferring a "related but not a copy" band.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/), which provisions its own Python 3.12.

```bash
uv sync --extra dev    # dev = tests, ruff, and the Streamlit app
```

On Linux and Windows this installs the CUDA build of PyTorch. On macOS it
installs the standard build (CPU / MPS). Streamlit is an optional extra
(`--extra app`), so the CLI alone stays lean.

## Usage

```bash
# 1. Build the corpus (data/melodies.jsonl; about 1.5 min with all cores)
uv run melodymorph prepare-data

# 2. Train (early stopping; configs/smoke.yaml is a sanity check in under a minute)
uv run melodymorph train --config configs/base.yaml          # add --amp on tensor-core GPUs
uv run melodymorph train --config configs/base.yaml --set dropout=0.2 --set seed=7
# Resume from the *.last.pt file (it has the optimizer state; best.pt is inference-only)
uv run melodymorph train --config configs/base.yaml --resume checkpoints/best.last.pt

# 3. Generate
uv run melodymorph generate --seed-preset "Ode to Joy (opening)" --mode continuation -k 4 --bars 4
uv run melodymorph generate --seed-text "C4/q E4/q G4/h A4/e G4/e" --mode variation -k 4
uv run melodymorph generate --seed-preset "Folk phrase" --meter 3/4      # triple metre
uv run melodymorph generate --seed-midi motif.mid --out out/

# 4. Evaluate (held-out pitch perplexity, Kneser-Ney baselines, generation metrics)
uv run melodymorph evaluate --generation --report reports/eval.json
```

`generate` writes `<mode>_<i>.mid` and `<mode>_<i>.png` into `--out`.

### Web UI

```bash
uv run streamlit run app/streamlit_app.py
```

You can enter a seed as a preset, as text notation, or as a MIDI upload. Text
and preset seeds have a 4/4 or 3/4 switch; a MIDI seed takes its metre from the
file. Pick a mode and sampling settings to get piano rolls, in-browser audio and MIDI
downloads. Ratings go to `out/ratings.csv` along with the seed, the output, the
sampling settings and the checkpoint.

### Text seed notation

Write each note as `PITCH+OCTAVE/DURATION`, separated by spaces, e.g. `C4/q E4/q G4/h`.

- **Durations**: `w` `h` `q` `e` `s` (whole to sixteenth). Append `.` for a
  dotted note, or give a plain number of sixteenths (`C4/5`). A missing
  duration means a quarter.
- **Accidentals**: `#` for sharp, `b` for flat.
- **Rests**: `R/q` is a quarter rest. A leading rest makes a pickup.

### Limits

- Supported metres are those with 16- or 12-step bars. A MIDI file in 5/4, 7/8
  or similar is read as 4/4, with a warning. A MIDI file in 3/2 or 6/4 is also
  read as 4/4: MIDI ticks aren't rescaled the way the corpus is.
- Seeds longer than 32 notes are cut to 32: the last ones for continuation, the
  first ones for variation.
- The pitch range is C3–C6. A seed outside it is octave-shifted to fit.

## Evaluation

`evaluate` reports negative log-likelihood per token family. The headline is
**pitch perplexity**: overall token perplexity is flattering, because `POS`
tokens are nearly free under the grammar.

The baseline is an interpolated Kneser-Ney n-gram model (orders 3, 6 and 9) on
the same tokens. Both models score each validation tune on its own.

The generation metrics score only the generated notes, from 10-note validation
seeds with 4 candidates each:
- completion rate
- in-key ratio
- pitch-class Jensen-Shannon divergence against the corpus
- repetition, within-seed distinctness and rhythm entropy
- for variations, motif score and the in-band rate

## Tests

```bash
uv run pytest && uv run ruff check .
```

The tests cover:
- metre alignment and pickups in duple and triple time, deduplication, and the leak-proof split
- time-signature tokens, and backward compatibility with 73-token checkpoints
- melody-aligned windows (every target scored once, padding ignored)
- causality and KV-cache equivalence with the full forward pass
- the grammar mask (no overlaps, stopping on the target bar, the re-prefill path)
- key normalisation, rhythm-locked variations that never copy the seed, and seed trimming
- lossless text notation, MIDI round-trips and time-signature detection
- Kneser-Ney normalisation, and rating context

CI runs the same commands on every push (`.github/workflows/ci.yml`).

## Project layout

```
melodymorph/
├── corpus.py      # music21 -> metre-aligned, key-normalised, deduplicated melodies; split
├── tokenizer.py   # REMI vocabulary, encode/decode, shared pitch helpers
├── keys.py        # Krumhansl-Schmuckler key estimation for seeds
├── dataset.py     # melody-aligned windows, transposition augmentation
├── model.py       # decoder-only Transformer + KV cache
├── train.py       # training loop, early stopping, checkpoints
├── sample.py      # grammar-masked, batched, KV-cached sampling
├── generate.py    # continuation / variation modes
├── evaluate.py    # per-family NLL, Kneser-Ney baseline, generation metrics, ratings
├── midi_io.py     # MIDI read/write, text-seed parsing
├── viz.py         # piano roll + contour strip (matplotlib Figure API)
├── audio.py       # numpy synth -> WAV
└── cli.py         # `melodymorph` entry point
app/streamlit_app.py   # web UI
configs/               # base.yaml (full run), smoke.yaml (fast check)
scripts/               # ablation and comparison scripts (scripts/legacy = pre-audit code)
```
