"""MelodyMorph MM-01 — studio console (multipage).

Pages: Console (big seed monitor + input + dials) / Takes (big take graphs) /
Spec (what the box does). Navigation is Streamlit-native so every nav item
actually works. No italics, 22px max type.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import matplotlib.pyplot as plt

from melodymorph.audio import melody_to_wav_bytes
from melodymorph.evaluate import log_human_rating
from melodymorph.generate import generate_continuations, generate_variations
from melodymorph.midi_io import (
    PRESET_SEEDS,
    melody_to_note_string,
    parse_note_string,
    read_midi,
    write_midi,
)
from melodymorph.tokenizer import melody_duration
from melodymorph.train import load_checkpoint
from melodymorph.viz import plot_piano_roll

st.set_page_config(page_title="MelodyMorph MM-01", page_icon="🎛", layout="wide")

CHECKPOINT_PATH = os.environ.get("MELODYMORPH_CHECKPOINT", "checkpoints/best.pt")
DEFAULT_PRESET = "Ode to Joy (opening)"
_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def pitch_name(p: int) -> str:
    return f"{_NAMES[p % 12]}{p // 12 - 1}"


@st.cache_resource
def get_model(checkpoint_path: str):
    return load_checkpoint(checkpoint_path)


def dark_roll(melody, seed_len=0):
    """Dark re-skin of viz.plot_piano_roll, sized to fill the page width.

    Recolouring is driven by note onset vs seed_len (patches are added in
    melody order), never by sniffing the light-theme palette.
    """
    fig = plot_piano_roll(melody, seed_len=seed_len, title="")
    fig.set_size_inches(12.5, 4.4)
    fig.patch.set_facecolor("#0C0C0E")
    ax = fig.axes[0]
    ax.set_facecolor("#0C0C0E")
    for spine in ax.spines.values():
        spine.set_color((1, 1, 1, 0.10))
    ax.tick_params(colors="#85858C", labelsize=8)
    ax.xaxis.label.set_color("#85858C")
    ax.xaxis.label.set_fontsize(9)
    for line in ax.get_lines():
        if line.get_linestyle() == "--":
            line.set_color("#D8C79A")  # seed divider
            line.set_alpha(0.9)
        else:
            line.set_color((1, 1, 1, 0.13))
            line.set_alpha(0.6)
    for patch, note in zip(ax.patches, melody):
        if note.onset < seed_len:
            patch.set_facecolor("#D8C79A")  # dry amber = seed
            patch.set_edgecolor("#2A2415")
        else:
            patch.set_facecolor("#7EE8B8")  # mint = generated
            patch.set_edgecolor("#0B2E22")
        patch.set_linewidth(0.6)
        patch.set_alpha(0.95)
    fig.tight_layout(pad=1.0)
    return fig


def contour_strip(melody, seed_len=0):
    """Pitch-contour strip: the motif's shape at a glance."""
    notes = sorted(melody, key=lambda n: n.onset)
    fig, ax = plt.subplots(figsize=(4.5, 1.6))
    fig.patch.set_facecolor("#0C0C0E")
    ax.set_facecolor("#0C0C0E")
    if notes:
        xs = [n.onset for n in notes]
        ys = [n.pitch for n in notes]
        if seed_len:
            ax.axvspan(0, seed_len, color="#D8C79A", alpha=0.08)
        ax.step(xs, ys, where="post", color="#7EE8B8", linewidth=1.4)
        ax.plot(xs, ys, "o", color="#7EE8B8", markersize=3)
        ax.set_xlim(0, max(xs) + 4)
    for spine in ax.spines.values():
        spine.set_color((1, 1, 1, 0.10))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout(pad=0.4)
    return fig


THEME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=Plus+Jakarta+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

em, i { font-style: normal !important; font-family: inherit !important; }

.stApp { background: #080808; color: #E8E8EA; font-family: 'Plus Jakarta Sans', system-ui, sans-serif; }
.stApp::before {
  content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background:
    radial-gradient(700px 320px at 50% 0%, rgba(126,232,184,0.05), transparent 65%),
    linear-gradient(rgba(255,255,255,0.022) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.022) 1px, transparent 1px);
  background-size: auto, 40px 40px, 40px 40px;
}
[data-testid="stAppViewContainer"] > .main { background: transparent; }
.block-container { max-width: 1320px; padding-top: 2rem; padding-bottom: 3rem; }
.grain { position: fixed; inset: 0; z-index: 40; pointer-events: none; opacity: .04;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='3'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E"); }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; height: 0; }
[data-testid="stSidebar"] { display: none; }
header[data-testid="stHeader"] { background: transparent; }

/* top status strip */
.topbar { display: flex; align-items: center; justify-content: space-between; gap: 12px;
  padding: 12px 16px; margin-bottom: 10px;
  background: rgba(14,14,16,0.82); border: 1px solid rgba(255,255,255,0.09); border-radius: 12px;
  box-shadow: 0 18px 44px -18px rgba(0,0,0,0.9), inset 0 1px 0 rgba(255,255,255,0.08); }
.topbar .mk { font-family: 'JetBrains Mono'; font-size: 11px; font-weight: 500; letter-spacing: 0.08em; color: #fff; display: flex; align-items: center; gap: 8px; white-space: nowrap; }
.topbar .mk .led { width: 7px; height: 7px; border-radius: 50%; background: #7EE8B8; box-shadow: 0 0 10px #7EE8B8; flex: none; animation: blink 2.2s cubic-bezier(0.32,0.72,0,1) infinite; }
.topbar .tstatus { font-family: 'JetBrains Mono'; font-size: 10px; letter-spacing: 0.1em; color: #7EE8B8; border: 1px solid rgba(126,232,184,0.3); background: rgba(126,232,184,0.07); border-radius: 8px; padding: 7px 11px; white-space: nowrap; }

/* working page nav — real Streamlit page_links, themed as console tabs.
   NOTE: the data-testid sits ON the anchor itself, not a wrapper. */
a[data-testid="stPageLink-NavLink"] { font-family: 'JetBrains Mono' !important; font-size: 11px !important; letter-spacing: 0.08em !important; color: #9A9AA1 !important; text-decoration: none !important; border: 1px solid rgba(255,255,255,0.09) !important; border-radius: 9px !important; padding: 9px 6px !important; display: block !important; width: 100% !important; text-align: center !important; background: rgba(255,255,255,0.02) !important; transition: all 450ms cubic-bezier(0.32,0.72,0,1) !important; }
a[data-testid="stPageLink-NavLink"]:hover { color: #fff !important; background: rgba(255,255,255,0.07) !important; }
a[data-testid="stPageLink-NavLink"][aria-current="page"] { color: #080808 !important; background: #E8E8EA !important; border-color: #E8E8EA !important; font-weight: 500 !important; }

/* console masthead — compact, 22px max */
.mast { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; padding: 18px 4px 14px; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 16px; flex-wrap: wrap; }
.mast h1 { font-family: 'Space Grotesk'; font-size: 22px; font-weight: 600; letter-spacing: -0.02em; margin: 0; line-height: 1.2; }
.mast h1 span { color: #7A7A82; font-weight: 400; }
.mast p { font-family: 'JetBrains Mono'; font-size: 11px; color: #85858C; margin: 6px 0 0; letter-spacing: 0.04em; }
.readout { display: flex; gap: 0; border: 1px solid rgba(255,255,255,0.09); border-radius: 10px; overflow: hidden; background: #0C0C0E; }
.readout div { padding: 8px 14px; border-left: 1px solid rgba(255,255,255,0.07); }
.readout div:first-child { border-left: none; }
.readout .k { font-family: 'JetBrains Mono'; font-size: 9px; letter-spacing: 0.14em; color: #63636B; display: block; }
.readout .v { font-family: 'JetBrains Mono'; font-size: 12px; color: #E8E8EA; }

/* tight double-bezel hardware modules */
div[data-testid="stContainer"] { background: rgba(255,255,255,0.035) !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 20px !important; padding: 4px !important; box-shadow: 0 20px 50px -28px rgba(0,0,0,0.9) !important; }
div[data-testid="stContainer"] > div { background: #0C0C0E; border-radius: 16px; box-shadow: inset 0 1px 0 rgba(255,255,255,0.07); padding: 16px; border: 1px solid rgba(255,255,255,0.04); }
.mod-label { font-family: 'JetBrains Mono'; font-size: 10px; letter-spacing: 0.18em; color: #7EE8B8; display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
.mod-label::after { content: ""; flex: 1; height: 1px; background: rgba(255,255,255,0.08); }
.mod-label.amber { color: #D8C79A; }

/* widgets — compact technical */
[data-testid="stVerticalBlock"] { gap: 0.9rem; }
section[data-testid="stColumns"] { gap: 1rem !important; }
label, [data-testid="stWidgetLabel"] p { color: #B9B9BF !important; font-size: 10px !important; letter-spacing: 0.14em !important; text-transform: uppercase; font-weight: 500 !important; font-family: 'JetBrains Mono' !important; }
div[data-testid="stRadio"] div[role="radiogroup"] { gap: 6px; }
div[data-testid="stRadio"] label[data-baseweb="radio"] { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.09); border-radius: 10px; padding: 7px 12px; font-family: 'JetBrains Mono' !important; font-size: 12px !important; transition: all 450ms cubic-bezier(0.32,0.72,0,1); }
div[data-testid="stRadio"] label[data-baseweb="radio"]:hover { background: rgba(255,255,255,0.08); }
[data-testid="stSlider"] [data-baseweb="slider"] div div div { background: #7EE8B8 !important; height: 2px !important; }
[data-testid="stSlider"] div[data-baseweb="thumb"] { background: #E8E8EA !important; border: 2px solid #0C0C0E !important; box-shadow: 0 0 0 1px rgba(255,255,255,0.35) !important; height: 14px !important; width: 14px !important; }
[data-testid="stSelectbox"] > div > div, [data-testid="stTextArea"] textarea, [data-testid="stFileUploader"] section { background: #101013 !important; border: 1px solid rgba(255,255,255,0.09) !important; border-radius: 10px !important; color: #fff !important; }
[data-testid="stTextArea"] textarea { font-family: 'JetBrains Mono' !important; font-size: 12px !important; line-height: 1.6 !important; }
code { background: rgba(255,255,255,0.05) !important; color: #D8C79A !important; border: 1px solid rgba(255,255,255,0.09); border-radius: 6px; font-size: 11px; font-family: 'JetBrains Mono' !important; }

/* hardware keys */
div[data-testid="stButton"] > button[kind="primary"] { background: #E8E8EA !important; color: #080808 !important; border: none !important; border-bottom: 3px solid #7EE8B8 !important; border-radius: 10px !important; padding: 12px !important; font-family: 'JetBrains Mono' !important; font-weight: 500 !important; font-size: 12px !important; letter-spacing: 0.1em !important; width: 100%; transition: all 450ms cubic-bezier(0.32,0.72,0,1) !important; }
div[data-testid="stButton"] > button[kind="primary"]:hover { transform: translateY(-1px) !important; background: #fff !important; }
div[data-testid="stButton"] > button[kind="primary"]:active { transform: translateY(1px) scale(0.99) !important; border-bottom-width: 1px !important; }
div[data-testid="stButton"] > button[kind="secondary"] { background: transparent !important; color: #85858C !important; border: 1px dashed rgba(255,255,255,0.18) !important; border-radius: 10px !important; padding: 9px !important; font-family: 'JetBrains Mono' !important; font-size: 11px !important; letter-spacing: 0.1em !important; width: 100%; transition: all 450ms cubic-bezier(0.32,0.72,0,1) !important; }
div[data-testid="stButton"] > button[kind="secondary"]:hover { color: #fff !important; border-color: rgba(255,255,255,0.4) !important; }
div[data-testid="stDownloadButton"] > button { background: transparent !important; color: #E8E8EA !important; border: 1px solid rgba(255,255,255,0.14) !important; border-radius: 8px !important; padding: 8px 12px !important; font-family: 'JetBrains Mono' !important; font-size: 11px !important; width: 100%; transition: all 450ms cubic-bezier(0.32,0.72,0,1) !important; }
div[data-testid="stDownloadButton"] > button:hover { border-color: #7EE8B8 !important; color: #7EE8B8 !important; }
audio { width: 100%; height: 32px; border-radius: 999px; background: #101013; }
audio::-webkit-media-controls-panel { background: #101013 !important; }
audio::-webkit-media-controls-current-time-display, audio::-webkit-media-controls-time-remaining-display { color: #E8E8EA !important; }
audio::-webkit-media-controls-play-button, audio::-webkit-media-controls-mute-button { filter: invert(0.85); }
[data-testid="stCodeBlock"] { background: #101013 !important; border: 1px solid rgba(255,255,255,0.09) !important; border-radius: 10px !important; }
[data-testid="stCodeBlock"] pre { background: transparent !important; }
[data-testid="stCodeBlock"] code { background: transparent !important; border: none !important; color: #D8C79A !important; }
[data-testid="stCodeBlock"] button { color: #85858C !important; }
/* hard font fallbacks: if the webfont fails, never fall to a default serif */
.mast h1 { font-family: 'Space Grotesk', 'Plus Jakarta Sans', system-ui, sans-serif !important; }
.mono, .mod-label, .ledger-head, .strip-lbl, .spec, .topbar .mk, .topbar .tstatus, code { font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace !important; }
[data-testid="stPyplot"] { background: #0C0C0E; border-radius: 10px; overflow: hidden; border: 1px solid rgba(255,255,255,0.06); }
.stAlert { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.09) !important; border-radius: 10px !important; }
[data-testid="stExpander"] { border: 1px solid rgba(255,255,255,0.09) !important; border-radius: 10px !important; background: #0C0C0E !important; }

/* ledger rows */
.ledger-head { display: flex; align-items: center; gap: 10px; font-family: 'JetBrains Mono'; font-size: 11px; margin: 2px 0 10px; flex-wrap: wrap; }
.ledger-head .idx { background: #E8E8EA; color: #080808; border-radius: 6px; padding: 2px 8px; font-weight: 500; }
.ledger-head .score { color: #7EE8B8; border: 1px solid rgba(126,232,184,0.3); background: rgba(126,232,184,0.07); border-radius: 6px; padding: 2px 8px; }
.ledger-head .stale { color: #D8C79A; border: 1px solid rgba(216,199,154,0.35); background: rgba(216,199,154,0.08); border-radius: 6px; padding: 2px 8px; }
.ledger-head .fresh { color: #7EE8B8; border: 1px solid rgba(126,232,184,0.3); border-radius: 6px; padding: 2px 8px; }
.ledger-head .meta { color: #63636B; margin-left: auto; }
.spec { width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono'; font-size: 11px; }
.spec td { padding: 9px 4px; border-top: 1px solid rgba(255,255,255,0.07); color: #9A9AA1; vertical-align: top; }
.spec td:first-child { color: #E8E8EA; width: 148px; }
.mono { font-family: 'JetBrains Mono'; font-size: 11px; color: #85858C; }
.strip-lbl { font-family: 'JetBrains Mono'; font-size: 9px; letter-spacing: 0.14em; color: #63636B; margin: 0 0 6px; }

/* entrance — pure CSS keyframes (Streamlit strips <script>, so no JS reveals) */
@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
div[data-testid="stContainer"] { animation: rise 0.45s cubic-bezier(0.32,0.72,0,1) both; }
section[data-testid="stColumns"] > div:nth-child(2) div[data-testid="stContainer"] { animation-delay: 0.06s; }
section[data-testid="stColumns"] > div:nth-child(3) div[data-testid="stContainer"] { animation-delay: 0.12s; }

@media (max-width: 900px) {
  .mast h1 { font-size: 19px; }
  .readout { width: 100%; overflow-x: auto; }
  section[data-testid="stColumns"] { flex-direction: column !important; }
  section[data-testid="stColumns"] > div { width: 100% !important; flex: 1 1 100% !important; }
  .block-container { padding-left: 1rem; padding-right: 1rem; }
}
"""

st.markdown(f"<style>{THEME_CSS}</style>", unsafe_allow_html=True)
st.markdown('<div class="grain"></div>', unsafe_allow_html=True)

# Session defaults (setdefault: safe for widget keys only before first render)
st.session_state.setdefault("mm_candidates", [])
st.session_state.setdefault("mm_mode", "continuation")
st.session_state.setdefault("mm_sig", None)
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


def topbar():
    """Status strip + working page nav. Rendered first on every page."""
    if not Path(CHECKPOINT_PATH).exists():
        with st.container():
            st.markdown('<div class="mod-label amber">MM-01 // NO TAPE LOADED</div>', unsafe_allow_html=True)
            st.markdown("**Checkpoint missing.** Train once, then reload.")
            st.code("uv run melodymorph prepare-data\nuv run melodymorph train --config configs/base.yaml")
        st.stop()
    n = len(st.session_state["mm_candidates"])
    abbr = {"continuation": "CONT", "variation": "VAR"}.get(st.session_state["mm_mode"], "—")
    st.markdown(
        f"<div class='topbar'><div class='mk'><span class='led'></span>MM-01 · MELODY MORPH</div>"
        f"<div class='tstatus'>TAKES {n:02d} · {abbr}</div></div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        st.page_link(studio_pg, label="01 / CONSOLE")
    with c2:
        st.page_link(takes_pg, label=f"02 / TAKES ({n})")
    with c3:
        st.page_link(spec_pg, label="03 / SPEC")


def masthead(title: str, sub: str, hint: str):
    model, _ = get_model(CHECKPOINT_PATH)
    device = str(next(model.parameters()).device)
    st.markdown(
        f"<div class='mast'><div><h1>{title}</h1><p>{hint}</p></div>"
        "<div class='readout'>"
        "<div><span class='k'>ENGINE</span><span class='v'>decoder · 3.4M</span></div>"
        "<div><span class='k'>VOCAB</span><span class='v'>73 REMI</span></div>"
        "<div><span class='k'>GRID</span><span class='v'>16th · 4/4</span></div>"
        f"<div><span class='k'>DEVICE</span><span class='v'>{device}</span></div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def render_seed_editor():
    """SRC module. Parses the current input and stores it; returns (seed, tempo)."""
    with st.container():
        st.markdown('<div class="mod-label amber">SRC // SEED INPUT</div>', unsafe_allow_html=True)
        source = st.radio("Input", ["Preset", "Text", "MIDI"], horizontal=True, key="src_choice")
        seed = None
        if source == "Preset":
            name = st.selectbox("Motif", list(PRESET_SEEDS), key="preset_name")
            st.code(PRESET_SEEDS[name])
            seed = parse_note_string(PRESET_SEEDS[name])
        elif source == "Text":
            text = st.text_area("Notation", value=PRESET_SEEDS[DEFAULT_PRESET],
                                help="w h q e s + '.' · #/b · R = rest", key="seed_text")
            if text.strip():
                try:
                    seed = parse_note_string(text)
                except ValueError as exc:
                    st.error(str(exc))
            else:
                st.warning("Notation is empty — enter notes or pick a preset.")
        else:
            upload = st.file_uploader("MIDI seed", type=["mid", "midi"], key="midi_upload")
            if upload is not None:
                tmp_path = Path("out/_uploaded_seed.mid")
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path.write_bytes(upload.read())
                try:
                    seed = read_midi(tmp_path)
                except ValueError as exc:
                    st.error(str(exc))
            else:
                st.markdown("<div class='mono'>— awaiting file.</div>", unsafe_allow_html=True)
        # Invalid input clears the monitor; takes stay visible (ledger flags stale).
        st.session_state.mm_seed = seed
    return seed


def render_control_deck():
    """CTL module. Returns (mode, bars, k, temperature, top_p, tempo, go)."""
    with st.container():
        st.markdown('<div class="mod-label">CTL // GENERATION</div>', unsafe_allow_html=True)
        mode = st.radio("Mode", ["continuation", "variation"],
                        format_func=lambda m: "CONT — extend" if m == "continuation" else "VAR — re-imagine",
                        key="ctl_mode")
        bars = st.slider("Bars", 1, 8, 4)
        k = st.slider("Takes", 1, 8, 4)
        # Per-mode key: otherwise Streamlit keeps the old mode's value and the
        # variation default (1.15) never applies after switching modes.
        temperature = st.slider("Temp", 0.5, 1.5,
                                0.95 if mode == "continuation" else 1.15, 0.05,
                                key=f"temp_{mode}")
        top_p = st.slider("Top-p", 0.5, 1.0, 0.95, 0.01, key="top_p")
        tempo = st.slider("BPM", 60, 160, 100, 5, key="bpm")
        go = st.button("RUN  ●", type="primary", use_container_width=True)
        st.button("SHUFFLE SEED + DIALS", type="secondary", use_container_width=True,
                  on_click=_shuffle_dials)
        st.markdown(f"<div class='mono'>ckpt · {CHECKPOINT_PATH}</div>", unsafe_allow_html=True)
    return mode, bars, k, temperature, top_p, tempo, go


def run_generation(model, tokenizer, device, seed, sig, mode, bars, k, temperature, top_p):
    try:
        with st.spinner("Sampling takes…"):
            fn = generate_continuations if mode == "continuation" else generate_variations
            st.session_state.mm_candidates = fn(
                model, tokenizer, seed, n_bars=bars, k=k,
                temperature=float(temperature), top_p=float(top_p), device=device)
            st.session_state.mm_sig = sig
            st.session_state.mm_mode = mode
    except Exception as exc:  # never blank-crash the console
        st.error(f"Generation failed: {exc}")
        return False
    return True


def take_card(i, cand, tempo, mm_mode):
    with st.container():
        score = f"{cand.motif_score:.2f}" if cand.motif_score is not None else "—"
        lo, hi = min(n.pitch for n in cand.melody), max(n.pitch for n in cand.melody)
        st.markdown(
            f"<div class='ledger-head'><span class='idx'>TAKE {i + 1:02d}</span>"
            f"<span class='score'>MOTIF {score}</span>"
            f"<span class='meta'>{len(cand.melody)} notes · "
            f"{melody_duration(cand.melody)} steps · {pitch_name(lo)}–{pitch_name(hi)}</span></div>",
            unsafe_allow_html=True,
        )
        fig = dark_roll(cand.melody, seed_len=cand.seed_len_steps)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        a_col, s_col = st.columns([1.7, 1])
        with a_col:
            st.audio(melody_to_wav_bytes(cand.melody, tempo_bpm=tempo), format="audio/wav")
            midi_path = Path("out") / f"{mm_mode}_{i + 1}.mid"
            write_midi(cand.melody, midi_path, tempo_bpm=tempo)
            st.download_button("SAVE .MID", data=midi_path.read_bytes(),
                               file_name=midi_path.name, mime="audio/midi",
                               key=f"dl_{i}", use_container_width=True)
        with s_col:
            st.markdown("<div class='strip-lbl'>CONTOUR</div>", unsafe_allow_html=True)
            sfig = contour_strip(cand.melody, seed_len=cand.seed_len_steps)
            st.pyplot(sfig, use_container_width=True)
            plt.close(sfig)
            with st.expander("RATE TAKE"):
                mus = st.slider("Musicality", 1, 5, 3, key=f"mus_{i}")
                mot = st.slider("Motif lock", 1, 5, 3, key=f"mot_{i}")
                if st.button("LOG RATING", key=f"rate_{i}", use_container_width=True):
                    log_human_rating("out/ratings.csv", f"{mm_mode}_{i + 1}", mus, mot)
                    st.toast(f"Take {i + 1}: {mus}/5 musicality, {mot}/5 motif — logged.")


def studio_view():
    topbar()
    model, tokenizer = get_model(CHECKPOINT_PATH)
    device = str(next(model.parameters()).device)
    masthead("MelodyMorph <span>MM-01 · console</span>", "",
             "SEED 5–15 NOTES → k SAMPLED TAKES · GRAMMAR-MASKED · MOTIF-FILTERED")

    # Big seed monitor first — the graph is the point of the page.
    with st.container():
        st.markdown('<div class="mod-label">VIEW // SEED MONITOR</div>', unsafe_allow_html=True)
        seed = st.session_state.mm_seed
        tempo = st.session_state.get("bpm", 100)
        if seed is not None:
            fig = dark_roll(seed, seed_len=melody_duration(seed) + 1)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            v1, v2 = st.columns([1.6, 1])
            with v1:
                st.audio(melody_to_wav_bytes(seed, tempo_bpm=tempo), format="audio/wav")
            with v2:
                lo, hi = min(n.pitch for n in seed), max(n.pitch for n in seed)
                st.markdown(
                    f"<div class='mono'>{len(seed)} notes · {melody_duration(seed)} steps<br>"
                    f"range {pitch_name(lo)}–{pitch_name(hi)}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(f"<div class='mono'>{melody_to_note_string(seed)}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='mono'>— no signal. Load a seed in SRC below.</div>",
                        unsafe_allow_html=True)

    col_src, col_ctl = st.columns(2, gap="medium")
    with col_src:
        seed = render_seed_editor()
    with col_ctl:
        mode, bars, k, temperature, top_p, tempo, go = render_control_deck()

    seed = st.session_state.mm_seed
    sig = (melody_to_note_string(seed), mode, bars, k,
           round(float(temperature), 3), round(float(top_p), 3)) if seed is not None else None
    if go:
        if seed is None:
            st.warning("Nothing to run — load a valid seed in SRC first.")
        elif run_generation(model, tokenizer, device, seed, sig, mode, bars, k,
                            temperature, top_p):
            st.switch_page(takes_pg)


def takes_view():
    topbar()
    masthead("Tape ledger <span>— generated takes</span>", "",
             "MINT = GENERATED · AMBER = SEED REFERENCE")
    candidates = st.session_state.mm_candidates
    tempo = st.session_state.get("bpm", 100)
    if not candidates:
        with st.container():
            st.markdown('<div class="mod-label">LEDGER // IDLE</div>', unsafe_allow_html=True)
            st.markdown("<div class='mono'>No takes yet. Build a seed on the Console and press RUN — "
                        "you will land back here automatically.</div>", unsafe_allow_html=True)
            st.page_link(studio_pg, label="→ OPEN CONSOLE")
        return
    seed = st.session_state.mm_seed
    mode, bars, k = st.session_state.get("ctl_mode", "continuation"), 4, len(candidates)
    sig = st.session_state.mm_sig
    stale = sig is None or (seed is not None and sig[0] != melody_to_note_string(seed))
    badge = ("<span class='stale'>STALE — seed changed, press RUN on Console</span>" if stale
             else "<span class='fresh'>FRESH</span>")
    st.markdown(
        f"<div class='ledger-head'><span class='idx'>{len(candidates):02d} TAKES</span>{badge}</div>",
        unsafe_allow_html=True,
    )
    Path("out").mkdir(exist_ok=True)
    mm_mode = "variation" if any(c.motif_score is not None for c in candidates) else "continuation"
    for i, cand in enumerate(candidates):
        take_card(i, cand, tempo, mm_mode)


def spec_view():
    topbar()
    masthead("Spec sheet <span>— what the box does</span>", "", "MM-01 · LAB 3 · SYMBOLIC MIDI ONLY")
    with st.container():
        st.markdown(
            """<table class="spec">
<tr><td>01 · Tokenizer</td><td>REMI — BAR / POS / PITCH / DUR on a 16th grid. Essen folksongs + Bach soprano lines, transposed to a common key.</td></tr>
<tr><td>02 · Model</td><td>Causal decoder-only transformer, ~3–4M params, trained from scratch on windowed melodies with transposition augmentation.</td></tr>
<tr><td>03 · Sampling</td><td>Temperature / top-p under a grammar mask — invalid sequences are impossible. Variation mode regenerates bar one hot and keeps the close-but-not-identical band.</td></tr>
</table>""",
            unsafe_allow_html=True,
        )
    rp = Path("out/ratings.csv")
    n_ratings = max(sum(1 for _ in rp.open()) - 1, 0) if rp.exists() else 0
    with st.container():
        st.markdown('<div class="mod-label">EVAL // HUMAN RATINGS</div>', unsafe_allow_html=True)
        st.markdown(f"<div class='mono'>{n_ratings} rating{'s' if n_ratings != 1 else ''} logged · "
                    "out/ratings.csv — rate takes on the Takes page.</div>", unsafe_allow_html=True)


studio_pg = st.Page(studio_view, title="Console", url_path="console")
takes_pg = st.Page(takes_view, title="Takes", url_path="takes")
spec_pg = st.Page(spec_view, title="Spec", url_path="spec")
pg = st.navigation([studio_pg, takes_pg, spec_pg], position="hidden")
pg.run()
