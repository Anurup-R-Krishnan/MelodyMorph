# MelodyMorph

**A Transformer-based melody continuation and variation generator.**

Give it a short musical motif (5–15 notes), and MelodyMorph generates several
plausible *continuations* (what comes next) or *variations* (a re-imagining of
the motif itself) — a decoder-only Transformer trained on monophonic melodies,
sampled multiple times, with a motif-preservation filter so the results stay
recognisably related to your seed instead of wandering off.

This is a Lab 3 (Unit II: Autoregressive Models + Transformers) case-study
prototype: symbolic MIDI, not raw audio; a narrow, well-scoped generation task
rather than end-to-end song generation.

## How it works

```
seed melody -> REMI tokenizer -> decoder-only Transformer -> sampling (+ grammar
mask + motif filter) -> k candidate melodies -> MIDI / piano-roll / audio
```

- **Data**: melodies extracted straight from the `music21` package's bundled
  corpus (Essen folksong collection + Bach chorale soprano lines) — no external
  downloads. Quantized to a 16th-note grid, transposed to a common key.
- **Tokenizer**: a REMI-style vocabulary (~73 tokens: `BAR`, in-bar `POS`,
  `PITCH`, `DUR`) that encodes metre explicitly.
- **Model**: a from-scratch decoder-only Transformer (causal self-attention,
  ~3–4M parameters) — small enough to train on a single consumer GPU in minutes.
- **Sampling**: temperature / top-k / top-p decoding under a grammar mask that
  makes structurally invalid output impossible, plus a motif-similarity filter
  (interval-contour edit distance) for the "variation" mode.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) (system Python doesn't need to be
compatible — `uv` provisions its own 3.12 interpreter).

```bash
uv sync --extra dev
```

This installs PyTorch (CUDA build, if you have an NVIDIA GPU), `music21`,
`mido`, `matplotlib`, `streamlit`, and friends into a local `.venv`.

## Usage

```bash
# 1. Build the training corpus (cached to data/melodies.jsonl, run once)
uv run melodymorph prepare-data

# 2. Train (a few minutes on a GPU; configs/smoke.yaml for a fast sanity check)
uv run melodymorph train --config configs/base.yaml

# 3. Generate continuations or variations from a seed
uv run melodymorph generate --seed-preset "Ode to Joy (opening)" \
    --mode continuation -k 4 --bars 4 --out out/

uv run melodymorph generate --seed-text "C4/q E4/q G4/h A4/e G4/e" \
    --mode variation -k 4 --out out/

# 4. Evaluate: perplexity + comparison against an n-gram Markov baseline
uv run melodymorph evaluate --checkpoint checkpoints/best.pt
```

Each `generate` call writes `<mode>_<i>.mid` and `<mode>_<i>.png` (piano roll,
seed region shaded) into `--out`.

### Web UI

```bash
uv run streamlit run app/streamlit_app.py
```

Enter a seed (preset, typed note notation, or an uploaded MIDI file), pick a
mode and sampling settings, and get back piano rolls, in-browser audio
playback, and MIDI downloads for every candidate.

### Text seed notation

`PITCH+OCTAVE/DURATION`, space-separated — e.g. `C4/q E4/q G4/h`.
Durations: `w` `h` `q` `e` `s` (whole/half/quarter/eighth/sixteenth), append
`.` for dotted. `#`/`b` for sharps/flats. `R/q` is a quarter-note rest.

## Tests

```bash
uv run pytest
```

Covers tokenizer round-tripping, model shapes/causality, the grammar mask, and
MIDI I/O.

## Project layout

```
melodymorph/
├── corpus.py      # music21 corpus -> normalized monophonic melodies
├── tokenizer.py    # REMI-style event vocabulary, encode/decode
├── dataset.py      # windowed dataset, transposition augmentation
├── model.py        # decoder-only Transformer
├── train.py        # training loop + checkpointing
├── sample.py        # temperature/top-k/top-p + grammar mask
├── generate.py     # continuation / variation modes, motif filter
├── midi_io.py      # MIDI read/write, text-seed parsing
├── viz.py          # piano-roll rendering
├── audio.py        # numpy synth -> WAV playback
├── evaluate.py     # perplexity, musical metrics, Markov baseline
└── cli.py           # `melodymorph` command-line entry point
app/streamlit_app.py  # web UI
configs/              # base.yaml (full run), smoke.yaml (fast sanity check)
```
