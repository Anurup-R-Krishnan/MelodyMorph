# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack
Existing: Streamlit app (`app/streamlit_app.py`, multipage via `st.navigation`), Python 3.12, matplotlib Figure API for piano rolls, numpy WAV synthesis. Streamlit >= 1.52. Custom visuals ship as inline SVG/CSS/JS injected with `st.markdown(unsafe_allow_html=True)` or `st.components.v1.html`.

## Users
Course demo / showcase audience: an instructor and classmates seeing a Lab 3 (Generative AI, Unit II: autoregressive models + Transformers) case study on a laptop or projector. Secondary: a musician sketching ideas who pastes a motif, auditions takes and exports MIDI. (Confirmed by the user: demo/showcase.)

## Product Purpose
MelodyMorph takes a short seed melody (5-15 notes; typed notation, a preset, or an uploaded MIDI file) and generates several continuations (what comes next) or variations (same rhythm, new pitches) with a from-scratch 3.2M-parameter decoder-only Transformer, so the audience can hear and see an autoregressive model at work. Success: the first impression is memorable and every control demonstrably works.

## Positioning
A Transformer you can watch compose: grammar-masked sampling, metre-aware REMI tokens (4/4 and 3/4), key normalisation, and a motif-similarity filter, all auditionable in the browser with piano roll, audio and MIDI export. Not a text-to-audio black box; symbolic and inspectable.

## Operating Context
Seed input (preset / text notation like `C4/q E4/q G4/h` / MIDI upload) -> controls (mode, bars, takes, temperature, top-p, pitch repetition penalty, BPM, metre) -> RUN -> takes ledger with piano roll, contour strip, in-browser audio, `.mid` download, 1-5 musicality and motif ratings logged to `out/ratings.csv`. Spec page documents tokenizer, model, sampling. GPU (GTX 1650) or CPU inference; takes appear in seconds.

## Capabilities and Constraints
- Pages: Console, Takes, Spec (keep).
- Metres: 4/4 and 3/4 (6/8 MIDI maps to the 12-step bar); seeds over 32 notes are trimmed.
- Pitch range C3-C6; vocabulary 75 tokens; checkpoint `checkpoints/best.pt`.
- Evaluation on held-out tunes: pitch perplexity 3.87 vs 6.36 for the best Kneser-Ney baseline; 100% completion, 99% of variations in the similarity band (real measurements, reportable).
- Must keep every existing function working; presentation is free to change.

## Brand Commitments
Name "MelodyMorph". Everything else (identity, palette, type, logo, artwork) is open: the user chose a fully free redesign. The current "MM-01 console" skin is evidence, not authority.

## Evidence on Hand
Real: trained checkpoint, eval report (`reports/final_eval.json`), five preset motifs, generated takes. Absent and not to be invented: user testimonials, benchmarks beyond the measured ones, claims of audio-model capability.

## Product Principles
- Show the model's mechanism (tokens, grammar mask, sampling) rather than hiding it.
- The melody is the hero: every screen puts notes, rhythm and contour first.
- Every control does exactly what it says; state (fresh/stale, metre, mode) is always visible.
- Impressive on first load, trustworthy after twenty minutes.

## Accessibility & Inclusion
Contrast >= 4.5:1 body text; honour `prefers-reduced-motion` for all animation; keyboard focus visible; audio never autoplays.
