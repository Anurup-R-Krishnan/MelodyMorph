"""MelodyMorph — a Transformer you can watch compose (multipage).

Pages: Console (seed, dials, RUN) / Takes (every generated take, playable) /
Spec (what the model is and how well it does). Navigation is Streamlit-native,
so every link keeps the session. The look is a Musica Viva concert poster:
one vermilion field, ink, paper, and a visible 16th-step construction grid.
"""

from __future__ import annotations

import base64
import json
import os
import random
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from melodymorph.arrange import (
    LEAD_INSTRUMENTS,
    STYLES,
    arrange,
    chord_chart,
    lead_only,
    styles_for,
    to_midi_bytes,
)
from melodymorph.artwork import (
    arc_poster_img,
    contour_img,
    logo_svg,
    roll_figure_html,
    svg_img,
)
from melodymorph.audio import melody_to_wav_bytes
from melodymorph.evaluate import log_human_rating
from melodymorph.generate import MAX_SEED_NOTES, generate_continuations, generate_variations
from melodymorph.harmony import infer_chords
from melodymorph.midi_io import (
    PRESET_SEEDS,
    melody_to_note_string,
    midi_bar,
    midi_time_signature,
    parse_note_string,
    read_midi,
    write_midi,
)
from melodymorph.render import available as render_available
from melodymorph.render import find_soundfont, render_mix, render_stems
from melodymorph.tokenizer import Note, melody_duration, pitch_name
from melodymorph.train import load_checkpoint

APP_DIR = Path(__file__).resolve().parent
st.set_page_config(
    page_title="MelodyMorph",
    page_icon=str(APP_DIR / "static" / "icon.png"),
    layout="wide",
    initial_sidebar_state="collapsed",
)

CHECKPOINT_PATH = os.environ.get("MELODYMORPH_CHECKPOINT", "checkpoints/best.pt")
DEFAULT_PRESET = "Ode to Joy (opening)"
PLAYER_SR = 22050  # plenty for a 4-partial synth below C6, and half the payload
EVAL_SUMMARY = Path("runs/eval_summary.json")


# --------------------------------------------------------------------------
# cached work
# --------------------------------------------------------------------------
@st.cache_resource
def get_model(checkpoint_path: str):
    return load_checkpoint(checkpoint_path)


@st.cache_data(show_spinner=False)
def get_audio_b64(melody_triples: tuple, tempo: int) -> str:
    """The take as a base64 WAV for the roll's own player."""
    melody = [Note(*t) for t in melody_triples]
    return base64.b64encode(melody_to_wav_bytes(melody, tempo_bpm=tempo, sr=PLAYER_SR)).decode("ascii")


@st.cache_data(show_spinner=False)
def get_cached_midi_bytes(melody_triples: tuple, tempo: int, bar: int = 16) -> bytes:
    import tempfile

    melody = [Note(*t) for t in melody_triples]
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        write_midi(melody, tmp_path, tempo_bpm=tempo, bar=bar)
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


@st.cache_data(show_spinner=False, max_entries=48)
def get_band(triples: tuple, bar: int, style: str, lead: str, tempo: int, humanize: bool, sf: str) -> dict:
    """Arrange a melody and render the band: one MP3 stem per instrument (when a
    soundfont is available), the chord chart, a type-1 MIDI file and an MP3 mixdown.
    ``style == "solo"`` plays just the lead, which is how a seed is auditioned."""
    melody = [Note(*t) for t in triples]
    program = LEAD_INSTRUMENTS[lead]
    if style == "solo":
        arr = lead_only(melody, program=program, bar=bar, tempo=tempo)
    else:
        arr = arrange(melody, style=style, bar=bar, tempo=tempo, lead_program=program, humanize=humanize)
    out = {"chords": chord_chart(arr), "midi": to_midi_bytes(arr), "key": arr.key, "stems": [], "mp3": None}
    if sf:
        stems = render_stems(arr, Path(sf), bitrate=80)
        for t in arr.tracks:
            label = lead if t.key == "lead" else t.label.capitalize()
            out["stems"].append({"key": t.key, "label": label, "volume": t.volume,
                                 "b64": base64.b64encode(stems[t.key]).decode("ascii")})
        out["mp3"] = render_mix(arr, Path(sf))
    return out


def _triples(melody) -> tuple:
    return tuple((n.onset, n.pitch, n.dur) for n in melody)


def brand_mark(size: int = 40) -> str:
    return svg_img(logo_svg(size), "MelodyMorph mark")


def H(markup: str) -> None:
    """Render trusted, generated markup (all dynamic text is escaped upstream)."""
    st.html(markup, unsafe_allow_javascript=True)


# --------------------------------------------------------------------------
# theme + session
# --------------------------------------------------------------------------
st.markdown(f"<style>{(APP_DIR / 'static' / 'theme.css').read_text()}</style>", unsafe_allow_html=True)
H(f"<script>{(APP_DIR / 'static' / 'player.js').read_text()}</script>")

st.session_state.setdefault("mm_candidates", [])
st.session_state.setdefault("mm_mode", "continuation")
st.session_state.setdefault("mm_sig", None)
st.session_state.setdefault("mm_bar", 16)  # bar length of the seed: 16 = duple, 12 = triple
st.session_state.setdefault("mm_metre", "4/4")
st.session_state.setdefault("mm_tempo", 100)  # plain state: widget keys vanish on other pages
st.session_state.setdefault("mm_style", "ballad")
st.session_state.setdefault("mm_lead", "Violin")
st.session_state.setdefault("mm_human", True)
if "mm_seed" not in st.session_state:
    st.session_state.mm_seed = parse_note_string(PRESET_SEEDS[DEFAULT_PRESET])


def _shuffle_dials():
    """Shuffle callback: runs before widgets instantiate, so writing widget
    keys here is legal (writing them in the script body after render is not)."""
    m = st.session_state.get("ctl_mode", "continuation")
    st.session_state["src_choice"] = "Preset"
    st.session_state["preset_name"] = random.choice(list(PRESET_SEEDS))
    st.session_state[f"temp_{m}"] = round(random.uniform(0.8, 1.3) * 20) / 20
    st.session_state["top_p"] = round(random.uniform(0.85, 1.0), 2)


def takes_are_stale() -> bool:
    """True when the takes on show were made from a different seed or metre."""
    sig = st.session_state.mm_sig
    seed = st.session_state.mm_seed
    return sig is None or (
        seed is not None and sig[:2] != (melody_to_note_string(seed), st.session_state.mm_bar))


def _metre_label() -> str:
    """The seed's time signature, for the readout.

    The metre radio's own widget state is filled in before the rerun, so on the
    console it is right for the click being handled; it is dropped on pages that
    do not render the radio, and for MIDI seeds there is no radio at all, so fall
    back to the stored value there."""
    if st.session_state.get("src_choice") != "MIDI":
        live = st.session_state.get("seed_meter")
        if live:
            return live
    return st.session_state.get("mm_metre", "4/4")


SOUNDFONT = find_soundfont()
CAN_RENDER = render_available(SOUNDFONT)


def _sound_state(bar: int) -> tuple[str, str, bool]:
    """The chosen style (valid for this metre), lead instrument and human-feel flag."""
    keys = [s.key for s in styles_for(bar)]
    style = st.session_state.mm_style if st.session_state.mm_style in keys else keys[0]
    return style, st.session_state.mm_lead, st.session_state.mm_human


def band_for(melody, bar: int, tempo: int, style: str | None = None) -> dict:
    """Cached band for ``melody`` in the current sound settings (``style="solo"`` = lead only)."""
    s, lead, human = _sound_state(bar)
    return get_band(_triples(melody), bar, style or s, lead, tempo, human, str(SOUNDFONT) if CAN_RENDER else "")


def _sync_sound(src: str, dst: str) -> None:
    st.session_state[dst] = st.session_state[src]


def sound_picker(prefix: str, bar: int) -> None:
    """Style and lead-instrument pickers. The Console and Takes pages each have a
    set; both write the plain ``mm_*`` state, so the choice survives navigation."""
    styles = {s.key: s.label for s in styles_for(bar)}
    style, lead, _ = _sound_state(bar)
    st.session_state[f"{prefix}_style"] = style
    st.session_state[f"{prefix}_lead"] = lead
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.selectbox("Band style", list(styles), format_func=styles.get, key=f"{prefix}_style",
                     on_change=_sync_sound, args=(f"{prefix}_style", "mm_style"),
                     help="the groove under the melody: chords, bass and drums are written for you")
    with c2:
        st.selectbox("Lead instrument", list(LEAD_INSTRUMENTS), key=f"{prefix}_lead",
                     on_change=_sync_sound, args=(f"{prefix}_lead", "mm_lead"),
                     help="who plays the melody")


# --------------------------------------------------------------------------
# chrome
# --------------------------------------------------------------------------
def topbar(active: int = 0) -> None:
    """Brand + working page navigation. Rendered first on every page.

    ``active`` is the index of the current page (0 console, 1 takes, 2 spec):
    Streamlit does not expose the current page on its links, so the page marks it."""
    if not Path(CHECKPOINT_PATH).exists():
        H(f"<div class='brand'>{brand_mark()}<b>melodymorph</b></div>")
        st.error("No checkpoint found. Train one, then reload.")
        st.code("uv run melodymorph prepare-data\nuv run melodymorph train --config configs/base.yaml")
        st.stop()
    n = len(st.session_state["mm_candidates"])
    st.markdown(
        f"<style>.st-key-topbar [data-testid='stColumn']:nth-child({active + 2}) a[data-testid='stPageLink-NavLink']"
        "{background:var(--ink)!important;padding:0 .7rem!important;box-shadow:none}"
        f".st-key-topbar [data-testid='stColumn']:nth-child({active + 2}) a[data-testid='stPageLink-NavLink'] *"
        "{color:var(--field)!important}</style>",
        unsafe_allow_html=True,
    )
    with st.container(key="topbar"):
        c_brand, c1, c2, c3, c4 = st.columns([3.6, 1, 1, 0.9, 1.1], gap="small", vertical_alignment="center")
        with c_brand:
            H(f"<div class='brand'>{brand_mark()}<div><b>melodymorph</b>"
              "<i>a transformer that writes the next notes</i></div></div>")
        with c1:
            st.page_link(studio_pg, label="console")
        with c2:
            st.page_link(takes_pg, label="takes")
        with c3:
            st.page_link(spec_pg, label="spec")
        with c4:
            if n:
                tone = "stale" if takes_are_stale() else "fresh"
                H(f"<span class='stamp mini {tone}'>{n} take{'s' if n != 1 else ''}</span>")


def page_heading(title: str, blurb: str, stamp: str = "") -> None:
    H(f"<div class='page-h'><h1>{escape(title)}</h1><p>{escape(blurb)}</p>{stamp}</div>")


def _engine_readout(model, tokenizer, metre: str) -> str:
    device = str(next(model.parameters()).device)
    params = model.num_parameters()
    ps = f"{params / 1e6:.1f}M" if params >= 1e6 else f"{params / 1e3:.0f}K"
    items = [("model", f"{ps} parameters"), ("vocabulary", f"{tokenizer.vocab_size} tokens"),
             ("grid", f"16th · {escape(metre)}"), ("device", escape(device))]
    return "<div class='readout'>" + "".join(f"<span>{k} <b>{v}</b></span>" for k, v in items) + "</div>"


def _token_class(tokenizer, tid: int) -> str:
    if tid == tokenizer.bar_id:
        return "t-bar"
    if tokenizer.is_pitch(tid):
        return "t-pitch"
    if tokenizer.is_dur(tid):
        return "t-dur"
    if tokenizer.is_pos(tid):
        return "t-pos"
    if tokenizer.is_ts(tid):
        return "t-ts"
    return "t-sp"


def token_spans(tokenizer, melody, bar: int) -> str:
    ids = tokenizer.encode(melody, add_special=True, bar=bar)
    return "".join(f"<span class='{_token_class(tokenizer, t)}' style='--i:{i}'>{escape(tokenizer.token(t))}</span>"
                   for i, t in enumerate(ids))


# --------------------------------------------------------------------------
# console
# --------------------------------------------------------------------------
def render_seed_editor():
    """Parses the current input and stores it; returns the seed (or None)."""
    H("<h2 class='sheet-h'>seed</h2>")
    source = st.radio("Input", ["Preset", "Text", "MIDI"], horizontal=True, key="src_choice")
    seed = None
    bar, metre = 16, "4/4"
    if source != "MIDI":
        metre = st.radio("Metre", ["4/4", "3/4"], horizontal=True, key="seed_meter",
                         help="MIDI seeds take their metre from the file")
        bar = 16 if metre == "4/4" else 12
    if source == "Preset":
        name = st.selectbox("Motif", list(PRESET_SEEDS), key="preset_name")
        st.code(PRESET_SEEDS[name], language=None)
        seed = parse_note_string(PRESET_SEEDS[name])
    elif source == "Text":
        text = st.text_area("Notation", value=PRESET_SEEDS[DEFAULT_PRESET],
                            help="w h q e s + '.' · #/b · R = rest", key="seed_text", height=110)
        if text.strip():
            try:
                seed = parse_note_string(text)
            except ValueError as exc:
                st.error(str(exc))
        else:
            st.warning("Notation is empty. Enter notes or pick a preset.")
    else:
        upload = st.file_uploader("MIDI seed", type=["mid", "midi"], key="midi_upload")
        if upload is not None:
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                tmp_path.write_bytes(upload.read())
                seed = read_midi(tmp_path)
                ts = midi_time_signature(tmp_path)
                bar = midi_bar(ts) or 16
                metre = ts if midi_bar(ts) else "4/4"
                if midi_bar(ts) is None:
                    st.warning(f"This file is in {ts}, which the model does not support "
                               "(it knows 2/4, 4/4, 3/4, 6/8). Reading it as 4/4.")
            except ValueError as exc:
                st.error(str(exc))
            finally:
                tmp_path.unlink(missing_ok=True)
        else:
            H("<p class='hint'>Drop a .mid file with a single melody line.</p>")
    if seed is not None and len(seed) > MAX_SEED_NOTES:
        st.info(f"Seed has {len(seed)} notes. Generation uses {MAX_SEED_NOTES} of them "
                "(the last ones to continue, the first ones to vary).")
    # Invalid input clears the monitor; takes stay visible (and are flagged stale).
    st.session_state.mm_seed = seed
    st.session_state.mm_bar = bar
    st.session_state.mm_metre = metre
    return seed


def render_control_deck():
    """Returns (mode, bars, k, temperature, top_p, rep_penalty, tempo, go)."""
    H("<h2 class='sheet-h'>generate</h2>")
    mode = st.radio("Mode", ["continuation", "variation"],
                    format_func=lambda m: "continue" if m == "continuation" else "vary",
                    key="ctl_mode", horizontal=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        bars = st.slider("Bars to add", 1, 8, 4, key="ctl_bars",
                         help="continuation only; a variation keeps the seed's length and rhythm")
    with c2:
        k = st.slider("Takes", 1, 8, 4, key="ctl_k")
    c3, c4 = st.columns(2, gap="medium")
    with c3:
        # Per-mode key: otherwise Streamlit keeps the old mode's value and the
        # variation default never applies after switching modes.
        temperature = st.slider("Temperature", 0.5, 1.5, 0.95 if mode == "continuation" else 1.05, 0.05,
                                key=f"temp_{mode}", help="higher is bolder and less predictable")
    with c4:
        top_p = st.slider("Top-p", 0.5, 1.0, 0.95, 0.01, key="top_p")
    c5, c6 = st.columns(2, gap="medium")
    with c5:
        rep_penalty = st.slider("Repeat penalty", 1.0, 2.0, 1.0, 0.05, key="ctl_rep_penalty",
                                help="penalty on recently used pitches; continuation only; 1.0 = off")
    with c6:
        st.session_state.setdefault("bpm", st.session_state.mm_tempo)  # survives leaving the page
        tempo = st.slider("Tempo (BPM)", 60, 160, step=5, key="bpm")
    st.session_state.mm_tempo = tempo
    H("<h2 class='sheet-h'>sound</h2>")
    sound_picker("con", st.session_state.mm_bar)
    st.toggle("Hear the seed with the band", key="seed_band",
              help="otherwise the seed plays on the lead instrument alone")
    if not CAN_RENDER:
        H("<p class='hint'>Real instruments need a soundfont. Run <b>python -m scripts.get_soundfont</b> "
          "and reload. Until then the lead plays on the built-in synth.</p>")
    go = st.button("generate", type="primary", key="run_btn")
    st.button("shuffle seed and dials", type="secondary", on_click=_shuffle_dials, key="shuffle_btn")
    H(f"<p class='hint'>checkpoint {escape(CHECKPOINT_PATH)}</p>")
    return mode, bars, k, temperature, top_p, rep_penalty, tempo, go


def run_generation(model, tokenizer, seed, sig, mode, bars, k, temperature, top_p, rep_penalty=1.0):
    try:
        with st.spinner("composing takes"):
            common = dict(k=k, temperature=float(temperature), top_p=float(top_p),
                          bar=st.session_state.mm_bar)
            if mode == "continuation":
                st.session_state.mm_candidates = generate_continuations(
                    model, tokenizer, seed, n_bars=bars,
                    repetition_penalty=float(rep_penalty), **common)
            else:
                st.session_state.mm_candidates = generate_variations(model, tokenizer, seed, **common)
            st.session_state.mm_sig = sig
            st.session_state.mm_mode = mode
    except Exception as exc:  # never blank-crash the console
        st.error(f"Generation failed: {exc}")
        return False
    return True


def studio_view():
    topbar(0)
    model, tokenizer = get_model(CHECKPOINT_PATH)

    col_left, col_right = st.columns([1.55, 1], gap="large")
    # Fill the control sheet first: the monitor on the left must show this
    # interaction's seed, not the previous one.
    with col_right:
        with st.container(key="sheet", border=True):
            render_seed_editor()
            mode, bars, k, temperature, top_p, rep_penalty, tempo, go = render_control_deck()

    seed = st.session_state.mm_seed
    bar = st.session_state.mm_bar
    with col_left:
        art = arc_poster_img(seed or [], seed_len=10**9, bar=bar)
        H("<section class='hero'>"
          f"<div class='hero-art'>{art}</div>"
          "<div class='hero-body'><div>"
          "<h1 class='wordmark'><span><em>melody</em></span><span><em>morph</em></span></h1>"
          "<p class='hero-sub'>Give it five to fifteen notes. A small Transformer writes what comes next, "
          "or the same rhythm with new pitches.</p>"
          "<button type='button' class='hero-go'>jump to generate</button></div>"
          f"{_engine_readout(model, tokenizer, _metre_label())}</div></section>")
        if seed is not None:
            triples = _triples(seed)
            chords = [(c.start, c.dur, c.name) for c in infer_chords(seed, bar)]
            heard = band_for(seed, bar, tempo, None if st.session_state.get("seed_band") else "solo")
            H(roll_figure_html(seed, uid="seed-roll", seed_len=10**9, bar=bar, tempo=tempo,
                               stems=heard["stems"] or None, chords=chords,
                               audio_b64=None if heard["stems"] else get_audio_b64(triples, tempo)))
            lo, hi = min(n.pitch for n in seed), max(n.pitch for n in seed)
            H(f"<p class='note-line'>{len(seed)} notes · {melody_duration(seed)} steps · "
              f"{pitch_name(lo)}–{pitch_name(hi)} · {escape(melody_to_note_string(seed))}</p>")
            ids = token_spans(tokenizer, seed, bar)
            H("<div class='cap'>what the model reads</div>"
              f"<div class='ticker' aria-hidden='true'><div>{ids}{ids}</div></div>"
              "<dl class='key'>"
              "<div><dt>ring</dt><dd>pitch. Low notes sit inside, high notes outside.</dd></div>"
              "<div><dt>angle</dt><dd>when the note starts, clockwise from the left.</dd></div>"
              "<div><dt>length</dt><dd>how long the note lasts.</dd></div>"
              "<div><dt>colour</dt><dd>paper is your seed, ink is what the model wrote.</dd></div></dl>")
        else:
            H("<p class='note-line'>No signal. Load a valid seed in the sheet on the right.</p>")

    sig = (melody_to_note_string(seed), bar, mode, bars, k, round(float(temperature), 3),
           round(float(top_p), 3), round(float(rep_penalty), 3)) if seed is not None else None
    if go:
        if seed is None:
            st.warning("Nothing to run. Load a valid seed first.")
        elif run_generation(model, tokenizer, seed, sig, mode, bars, k, temperature, top_p, rep_penalty):
            st.switch_page(takes_pg)


# --------------------------------------------------------------------------
# takes
# --------------------------------------------------------------------------
def band_panel(bar: int, tempo: int) -> None:
    """Style, lead instrument and feel for every take on the page."""
    sty_key = _sound_state(bar)[0]
    sty = STYLES[sty_key]
    with st.container(key="band", border=True):
        H("<h2 class='sheet-h'>band</h2>")
        c1, c2 = st.columns([2.4, 1], gap="large", vertical_alignment="bottom")
        with c1:
            sound_picker("tk", bar)
        with c2:
            st.session_state["tk_human"] = st.session_state.mm_human
            st.toggle("Human feel", key="tk_human", on_change=_sync_sound, args=("tk_human", "mm_human"),
                      help="phrase-shaped dynamics and slight timing drift, instead of a machine grid")
        H(f"<p class='hint'>{escape(sty.blurb)} Chords are inferred from each take's own melody. "
          f"Suggested tempo: {sty.suggested_bpm} BPM (yours is {tempo}).</p>")
        if tempo != sty.suggested_bpm:
            st.button(f"use {sty.suggested_bpm} BPM", key="tempo_btn", type="secondary",
                      on_click=_set_tempo, args=(sty.suggested_bpm,))
        if not CAN_RENDER:
            st.warning("No soundfont found, so there are no real instruments yet. Run "
                       "`python -m scripts.get_soundfont`, then reload. The lead plays on the built-in synth meanwhile.")


def _set_tempo(bpm: int) -> None:
    st.session_state.mm_tempo = bpm
    st.session_state["bpm"] = bpm


def take_card(i, cand, tempo, mm_mode, tokenizer):
    triples = _triples(cand.melody)
    band = band_for(cand.melody, cand.bar, tempo)
    lo, hi = min(n.pitch for n in cand.melody), max(n.pitch for n in cand.melody)
    score = (f"<div class='take-score'><b>{cand.motif_score:.2f}</b>motif kept</div>"
             if cand.motif_score is not None else "")
    chord_names = " ".join(name for _, _, name in band["chords"][:8])
    with st.container(key=f"take{i}", border=True):
        H(f"<div class='take-head'><span class='take-n'>{i + 1:02d}</span>"
          f"<div class='take-meta'>{escape(mm_mode)}"
          f"<span>{len(cand.melody)} notes · {melody_duration(cand.melody)} steps · "
          f"{pitch_name(lo)}–{pitch_name(hi)}</span>"
          f"<span>{escape(chord_names)}</span></div>{score}</div>")
        c_art, c_roll = st.columns([1, 2.1], gap="medium")
        with c_art:
            H("<div class='art-frame'>"
              f"{arc_poster_img(cand.melody, seed_len=cand.seed_len_steps, bar=cand.bar, width=600, height=600, center=(0.5, 0.5), radius=0.47)}"
              "</div><div class='cap'>interval contour: each bar is one step up or down</div>"
              f"{contour_img(cand.melody)}")
        with c_roll:
            H(roll_figure_html(cand.melody, uid=f"take-roll-{i}", seed_len=cand.seed_len_steps,
                               bar=cand.bar, tempo=tempo, stems=band["stems"] or None, chords=band["chords"],
                               audio_b64=None if band["stems"] else get_audio_b64(triples, tempo)))
        d1, d2, d3, d4, d5 = st.columns(5, gap="small")
        with d1:
            st.download_button("melody .mid", data=get_cached_midi_bytes(triples, tempo=tempo, bar=cand.bar),
                               file_name=f"{mm_mode}_{i + 1}.mid", mime="audio/midi", key=f"dl_{i}")
        with d2:
            st.download_button("band .mid", data=band["midi"], file_name=f"{mm_mode}_{i + 1}_band.mid",
                               mime="audio/midi", key=f"dlb_{i}")
        with d3:
            if band["mp3"]:
                st.download_button("mixdown .mp3", data=band["mp3"], file_name=f"{mm_mode}_{i + 1}.mp3",
                                   mime="audio/mpeg", key=f"dlm_{i}")
        with d4:
            with st.expander("tokens"):
                H(f"<div class='tokens'>{token_spans(tokenizer, cand.melody, cand.bar)}</div>")
        with d5:
            with st.expander("rate this take"):
                mus = st.slider("Musicality", 1, 5, 3, key=f"mus_{i}")
                mot = st.slider("Motif lock", 1, 5, 3, key=f"mot_{i}")
                if st.button("log rating", key=f"rate_{i}"):
                    sig = st.session_state.mm_sig or ("",)
                    log_human_rating("out/ratings.csv", {
                        "checkpoint": CHECKPOINT_PATH, "mode": mm_mode, "take": i + 1,
                        "seed": sig[0], "melody": melody_to_note_string(cand.melody),
                        "params": repr(sig[1:] + (st.session_state.mm_style, st.session_state.mm_lead)),
                        "musicality_1_5": mus, "motif_preservation_1_5": mot,
                    })
                    st.toast(f"Take {i + 1} logged: {mus}/5 musicality, {mot}/5 motif.")


def takes_view():
    topbar(1)
    _, tokenizer = get_model(CHECKPOINT_PATH)
    candidates = st.session_state.mm_candidates
    tempo = st.session_state.mm_tempo
    if not candidates:
        page_heading("no takes yet", "Build a seed on the console and press generate. You will land back here.")
        H("<div class='hero' style='aspect-ratio:3/1'><div class='hero-art'>"
          f"{arc_poster_img([], animate=False)}</div></div>")
        st.page_link(studio_pg, label="open the console")
        return
    stale = takes_are_stale()
    stamp = ("<span class='stamp stale'>stale</span>" if stale else "<span class='stamp fresh'>fresh</span>")
    n = len(candidates)
    page_heading(f"{n} take{'s' if n != 1 else ''}",
                 "The seed changed since these were made. Press generate on the console to refresh them."
                 if stale else "Press play to hear the band. Mute or solo any instrument, click the roll to seek.",
                 stamp)
    Path("out").mkdir(exist_ok=True)
    mm_mode = st.session_state.mm_mode
    band_panel(candidates[0].bar, tempo)
    with st.spinner("rehearsing the band"):
        for i, cand in enumerate(candidates):
            take_card(i, cand, tempo, mm_mode, tokenizer)


# --------------------------------------------------------------------------
# spec
# --------------------------------------------------------------------------
def _eval_table() -> str:
    if not EVAL_SUMMARY.exists():
        return ""
    try:
        s = json.loads(EVAL_SUMMARY.read_text())
        rows = [("This model", s["transformer"]["pitch_ppl"], True)]
        for order in ("9", "6", "3"):
            rows.append((f"Kneser-Ney, order {order}", s["kneser_ney"][order]["pitch_ppl"], False))
    except (KeyError, ValueError, TypeError):
        return ""
    body = "".join(f"<tr class='{'win' if win else ''}'><td>{escape(name)}</td><td>{ppl:.2f}</td></tr>"
                   for name, ppl, win in rows)
    cont, var = s.get("continuation", {}), s.get("variation", {})
    extra = (f"<p class='hint' style='margin-top:.9rem'>Held-out pitch perplexity, lower is better "
             f"({s.get('n_val', '?')} validation tunes). Sampled continuations finish "
             f"{cont.get('completion_rate', 0):.0%} of the time and land {cont.get('in_key', 0):.0%} of "
             f"their notes in key; {var.get('in_band_rate', 0):.0%} of variations fall in the motif band.</p>")
    return f"<table class='bench'>{body}</table>{extra}"


def spec_view():
    topbar(2)
    _, tokenizer = get_model(CHECKPOINT_PATH)
    page_heading("how it writes", "Symbolic MIDI only: a decoder-only Transformer trained from scratch on folk melodies.")
    sample = parse_note_string("C4/q E4/q G4/h A4/e G4/e F4/q E4/q D4/h")
    art = arc_poster_img(sample, bar=16, width=600, height=300, center=(0.5, 0.62), radius=0.7)
    H("<div class='spec-grid'>"
      "<article><h2>tokens</h2>"
      f"{art}<p>Each note is three events: where it starts in the bar, its pitch, its length. "
      f"The vocabulary is {tokenizer.vocab_size} tokens on a 16th grid, with a time-signature token for 4/4 or 3/4 bars. "
      "Tunes come from the Essen folksong collection and Bach chorale soprano lines in 2/4, 4/4, 3/4 and 6/8, "
      "aligned to real barlines and moved to C major or A minor.</p></article>"
      "<article><h2>model</h2>"
      "<p>A causal decoder-only Transformer: 4 layers, 4 heads, width 256, 3.2 million parameters, written out "
      "explicitly so the attention is readable. Training windows start at a tune's first token and never mix two "
      "tunes, which matches how it is prompted. Sampling is cached, so each new note costs one step.</p>"
      "<p style='margin-top:.8rem'>Sampling runs under a grammar mask: a note can never start before the last one "
      "ends, and every take stops exactly on the target bar.</p></article>"
      "<article><h2>band</h2>"
      "<p>The Transformer writes the tune only. Everything under it is written by rules: a chord for every half "
      "bar, chosen by dynamic programming over the diatonic chords of the estimated key so that chords contain the "
      "melody's strong-beat notes and move the way pop harmony moves, then voiced with smooth voice-leading. "
      "A style turns the chords into bass, comping and drums, with phrase-shaped dynamics and slight timing drift.</p>"
      "<p style='margin-top:.8rem'>Each instrument is rendered through a General MIDI soundfont as its own stem, "
      "which is what the mixer on every take mutes, solos and balances.</p></article>"
      f"<article><h2>how good</h2>{_eval_table() or '<p>No evaluation summary found. Run melodymorph evaluate.</p>'}"
      "</article></div>")
    rp = Path("out/ratings.csv")
    n_ratings = max(sum(1 for _ in rp.open()) - 1, 0) if rp.exists() else 0
    H(f"<p class='note-line' style='margin-top:1.4rem'>{n_ratings} human rating{'s' if n_ratings != 1 else ''} "
      "logged to out/ratings.csv. Rate takes on the takes page.</p>")


studio_pg = st.Page(studio_view, title="Console", url_path="console", default=True)
takes_pg = st.Page(takes_view, title="Takes", url_path="takes")
spec_pg = st.Page(spec_view, title="Spec", url_path="spec")
pg = st.navigation([studio_pg, takes_pg, spec_pg], position="hidden")
pg.run()
