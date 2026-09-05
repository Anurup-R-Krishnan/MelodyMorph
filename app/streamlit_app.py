"""MelodyMorph MM-01 — compact studio console.

No hero, no italics, no oversized type. Dense instrument layout:
source rail / viewer / control deck + ledger-style candidates.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import matplotlib.pyplot as plt

from melodymorph.audio import melody_to_wav_bytes
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

st.set_page_config(
    page_title="MelodyMorph MM-01",
    page_icon="◍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CHECKPOINT_PATH = os.environ.get("MELODYMORPH_CHECKPOINT", "checkpoints/best.pt")


@st.cache_resource
def get_model(checkpoint_path: str):
    return load_checkpoint(checkpoint_path)


def dark_roll(melody, seed_len=0):
    fig = plot_piano_roll(melody, seed_len=seed_len, title="")
    fig.set_size_inches(10, 2.9)
    fig.patch.set_facecolor("#0C0C0E")
    ax = fig.axes[0]
    ax.set_facecolor("#0C0C0E")
    for spine in ax.spines.values():
        spine.set_color("rgba(255,255,255,0.10)")
    ax.tick_params(colors="#85858C", labelsize=7)
    ax.xaxis.label.set_color("#85858C")
    ax.xaxis.label.set_fontsize(8)
    for line in ax.get_lines():
        line.set_color("rgba(255,255,255,0.13)")
        line.set_alpha(0.6)
    for patch in list(ax.patches):
        fc = patch.get_facecolor()
        is_seed = fc[0] < 0.5
        if is_seed:
            patch.set_facecolor("#D8C79A")  # dry amber for seed
            patch.set_edgecolor("#2A2415")
        else:
            patch.set_facecolor("#7EE8B8")  # single mint accent for generated
            patch.set_edgecolor("#0B2E22")
        patch.set_linewidth(0.6)
        patch.set_alpha(0.95)
    fig.tight_layout(pad=1.0)
    return fig


THEME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=Plus+Jakarta+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* kill italics everywhere — hard rule */
em, i, .hero-h1 em { font-style: normal !important; font-family: inherit !important; }

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
.block-container { max-width: 1320px; padding-top: 5.2rem; padding-bottom: 3rem; }
.grain { position: fixed; inset: 0; z-index: 40; pointer-events: none; opacity: .04;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='3'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E"); }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; height: 0; }
[data-testid="stSidebar"] { display: none; }
header[data-testid="stHeader"] { background: transparent; }

/* slim transport bar — fixed, only blurred fixed element */
.transport {
  position: fixed; top: 12px; left: 50%; transform: translateX(-50%); z-index: 30;
  display: flex; align-items: center; gap: 16px;
  height: 46px; padding: 0 8px 0 14px; max-width: min(1320px, calc(100% - 2rem)); width: max-content;
  background: rgba(14,14,16,0.82); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
  border: 1px solid rgba(255,255,255,0.09); border-radius: 12px;
  box-shadow: 0 18px 44px -18px rgba(0,0,0,0.9), inset 0 1px 0 rgba(255,255,255,0.08);
}
.transport .mk { font-family: 'JetBrains Mono'; font-size: 11px; font-weight: 500; letter-spacing: 0.08em; color: #fff; display:flex; align-items:center; gap:8px; }
.transport .mk .led { width: 7px; height: 7px; border-radius: 50%; background: #7EE8B8; box-shadow: 0 0 10px #7EE8B8; }
.transport .sep { width: 1px; height: 22px; background: rgba(255,255,255,0.10); }
.transport .tlink { font-family: 'JetBrains Mono'; font-size: 11px; color: #9A9AA1; text-decoration: none; padding: 7px 10px; border-radius: 8px; transition: all 500ms cubic-bezier(0.32,0.72,0,1); }
.transport .tlink:hover { color: #fff; background: rgba(255,255,255,0.07); }
.transport .rec { font-family: 'JetBrains Mono'; font-size: 11px; font-weight: 500; color: #080808; background: #7EE8B8; border-radius: 8px; padding: 8px 14px; text-decoration: none; transition: all 500ms cubic-bezier(0.32,0.72,0,1); }
.transport .rec:hover { transform: translateY(-1px); background: #A5F0CC; }
.burger { display: none; width: 32px; height: 32px; border-radius: 8px; background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.10); position: relative; cursor: pointer; }
.burger span { position: absolute; left: 8px; width: 16px; height: 1.5px; background: #fff; transition: all 500ms cubic-bezier(0.32,0.72,0,1); }
.burger span:nth-child(1){ top: 11px; } .burger span:nth-child(2){ top: 16px; } .burger span:nth-child(3){ top: 21px; }
.burger.open span:nth-child(1){ top: 16px; transform: rotate(45deg); }
.burger.open span:nth-child(2){ opacity: 0; }
.burger.open span:nth-child(3){ top: 16px; transform: rotate(-45deg); }
.menu-overlay { position: fixed; inset: 0; z-index: 25; background: rgba(8,8,8,0.9); backdrop-filter: blur(24px); display: flex; flex-direction: column; align-items: flex-start; justify-content: center; padding: 0 8vw; gap: 4px; opacity: 0; pointer-events: none; transition: opacity 600ms cubic-bezier(0.32,0.72,0,1); }
.menu-overlay.open { opacity: 1; pointer-events: auto; }
.menu-overlay a { font-family: 'JetBrains Mono'; font-size: 15px; color: #fff; text-decoration: none; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.08); width: 100%; opacity: 0; transform: translateY(16px); transition: all 600ms cubic-bezier(0.32,0.72,0,1); }
.menu-overlay.open a { opacity: 1; transform: translateY(0); }
.menu-overlay.open a:nth-child(2){ transition-delay: .07s; } .menu-overlay.open a:nth-child(3){ transition-delay: .14s; }

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

/* generate key — rectangular hardware key, not a pill */
div[data-testid="stButton"] > button[kind="primary"] { background: #E8E8EA !important; color: #080808 !important; border: none !important; border-bottom: 3px solid #7EE8B8 !important; border-radius: 10px !important; padding: 12px !important; font-family: 'JetBrains Mono' !important; font-weight: 500 !important; font-size: 12px !important; letter-spacing: 0.1em !important; width: 100%; transition: all 450ms cubic-bezier(0.32,0.72,0,1) !important; }
div[data-testid="stButton"] > button[kind="primary"]:hover { transform: translateY(-1px) !important; background: #fff !important; }
div[data-testid="stButton"] > button[kind="primary"]:active { transform: translateY(1px) scale(0.99) !important; border-bottom-width: 1px !important; }
div[data-testid="stDownloadButton"] > button { background: transparent !important; color: #E8E8EA !important; border: 1px solid rgba(255,255,255,0.14) !important; border-radius: 8px !important; padding: 8px 12px !important; font-family: 'JetBrains Mono' !important; font-size: 11px !important; width: 100%; transition: all 450ms cubic-bezier(0.32,0.72,0,1) !important; }
div[data-testid="stDownloadButton"] > button:hover { border-color: #7EE8B8 !important; color: #7EE8B8 !important; }
audio { width: 100%; height: 32px; }
[data-testid="stPyplot"] { background: #0C0C0E; border-radius: 10px; overflow: hidden; border: 1px solid rgba(255,255,255,0.06); }
.stAlert { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.09) !important; border-radius: 10px !important; }

/* ledger rows */
.ledger-head { display: flex; align-items: center; gap: 10px; font-family: 'JetBrains Mono'; font-size: 11px; margin: 2px 0 10px; }
.ledger-head .idx { background: #E8E8EA; color: #080808; border-radius: 6px; padding: 2px 8px; font-weight: 500; }
.ledger-head .score { color: #7EE8B8; border: 1px solid rgba(126,232,184,0.3); background: rgba(126,232,184,0.07); border-radius: 6px; padding: 2px 8px; }
.ledger-head .meta { color: #63636B; margin-left: auto; }
.spec { width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono'; font-size: 11px; }
.spec td { padding: 9px 4px; border-top: 1px solid rgba(255,255,255,0.07); color: #9A9AA1; vertical-align: top; }
.spec td:first-child { color: #E8E8EA; width: 148px; }
.mono { font-family: 'JetBrains Mono'; font-size: 11px; color: #85858C; }

/* subtle reveal — 16px only */
.reveal { opacity: 0; transform: translateY(16px); transition: opacity 700ms cubic-bezier(0.32,0.72,0,1), transform 700ms cubic-bezier(0.32,0.72,0,1); will-change: transform; }
.reveal.in { opacity: 1; transform: translateY(0); }

@media (max-width: 900px) {
  .transport .tlink, .transport .sep { display: none; }
  .burger { display: block; }
  .mast h1 { font-size: 19px; }
  .readout { width: 100%; overflow-x: auto; }
  section[data-testid="stColumns"] { flex-direction: column !important; }
  section[data-testid="stColumns"] > div { width: 100% !important; flex: 1 1 100% !important; }
  .block-container { padding-left: 1rem; padding-right: 1rem; }
}
"""

REVEAL_JS = """
<script>
(function(){
  const io = new IntersectionObserver((es)=>{
    es.forEach(e=>{ if(e.isIntersecting){ e.target.classList.add('in'); io.unobserve(e.target);} });
  },{threshold:0.08});
  const hook = ()=>{
    document.querySelectorAll('div[data-testid="stContainer"]').forEach(el=>{
      if(!el.classList.contains('reveal')){ el.classList.add('reveal'); io.observe(el); }
    });
  };
  hook();
  new MutationObserver(hook).observe(document.body,{childList:true,subtree:true});
  window.toggleMenu = function(){
    document.getElementById('burger').classList.toggle('open');
    document.getElementById('menuOverlay').classList.toggle('open');
  };
})();
</script>
"""

st.markdown(f"<style>{THEME_CSS}</style>", unsafe_allow_html=True)
st.markdown('<div class="grain"></div>', unsafe_allow_html=True)
st.markdown(
    """
<div class="transport">
  <div class="mk"><span class="led"></span> MM-01 · MELODY MORPH</div>
  <div class="sep"></div>
  <a class="tlink" href="#console">CONSOLE</a>
  <a class="tlink" href="#ledger">LEDGER</a>
  <a class="tlink" href="#spec">SPEC</a>
  <a class="rec" href="#console">● RUN</a>
  <div class="burger" id="burger" onclick="toggleMenu()"><span></span><span></span><span></span></div>
</div>
<div class="menu-overlay" id="menuOverlay">
  <a href="#console">01 / Console</a>
  <a href="#ledger">02 / Ledger</a>
  <a href="#spec">03 / Spec</a>
</div>
""",
    unsafe_allow_html=True,
)

if not Path(CHECKPOINT_PATH).exists():
    with st.container():
        st.markdown('<div class="mod-label amber">MM-01 // NO TAPE LOADED</div>', unsafe_allow_html=True)
        st.markdown("**Checkpoint missing.** Train once, then reload.")
        st.code("uv run melodymorph prepare-data\nuv run melodymorph train --config configs/base.yaml")
    st.stop()

model, tokenizer = get_model(CHECKPOINT_PATH)
device = next(model.parameters()).device.type

if "mm_candidates" not in st.session_state:
    st.session_state.mm_candidates = []
if "mm_seed" not in st.session_state:
    st.session_state.mm_seed = None

# compact masthead — 22px title, mono subline, hardware readout
st.markdown(
    f"""
<div class="mast" id="console">
  <div>
    <h1>MelodyMorph <span>MM-01 · motif console</span></h1>
    <p>SEED 5–15 NOTES → k SAMPLED CONTINUATIONS / VARIATIONS · GRAMMAR-MASKED · MOTIF-FILTERED</p>
  </div>
  <div class="readout">
    <div><span class="k">ENGINE</span><span class="v">decoder · 3.4M</span></div>
    <div><span class="k">VOCAB</span><span class="v">73 REMI</span></div>
    <div><span class="k">GRID</span><span class="v">16th · 4/4</span></div>
    <div><span class="k">DEVICE</span><span class="v">{device}</span></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# three-pane console: source rail / viewer / control deck
col_src, col_view, col_ctl = st.columns([1, 1.7, 1], gap="medium")

with col_ctl:
    with st.container():
        st.markdown('<div class="mod-label">CTL // GENERATION</div>', unsafe_allow_html=True)
        mode = st.radio("Mode", ["continuation", "variation"],
                        format_func=lambda m: "CONT — extend" if m == "continuation" else "VAR — re-imagine")
        bars = st.slider("Bars", 1, 8, 4)
        k = st.slider("Takes", 1, 8, 4)
        temperature = st.slider("Temp", 0.5, 1.5, 0.95 if mode == "continuation" else 1.15, 0.05)
        top_p = st.slider("Top-p", 0.5, 1.0, 0.95, 0.01)
        tempo = st.slider("BPM", 60, 160, 100, 5)
        go = st.button("RUN  ●", type="primary", use_container_width=True)
        st.markdown(f"<div class='mono'>ckpt · {CHECKPOINT_PATH}</div>", unsafe_allow_html=True)

with col_src:
    with st.container():
        st.markdown('<div class="mod-label amber">SRC // SEED INPUT</div>', unsafe_allow_html=True)
        source = st.radio("Input", ["Preset", "Text", "MIDI"], horizontal=True)
        seed = None
        if source == "Preset":
            name = st.selectbox("Motif", list(PRESET_SEEDS))
            st.code(PRESET_SEEDS[name])
            seed = parse_note_string(PRESET_SEEDS[name])
        elif source == "Text":
            text = st.text_area("Notation", value=PRESET_SEEDS["Ode to Joy (opening)"],
                                help="w h q e s + '.' · #/b · R = rest")
            if text.strip():
                try:
                    seed = parse_note_string(text)
                except ValueError as exc:
                    st.error(str(exc))
        else:
            upload = st.file_uploader("MIDI seed", type=["mid", "midi"])
            if upload is not None:
                tmp_path = Path("out/_uploaded_seed.mid")
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path.write_bytes(upload.read())
                try:
                    seed = read_midi(tmp_path)
                except ValueError as exc:
                    st.error(str(exc))
        if seed is not None:
            st.session_state.mm_seed = seed
            st.session_state.mm_tempo = tempo
        else:
            st.info("Select a seed input.")

with col_view:
    with st.container():
        st.markdown('<div class="mod-label">VIEW // SEED MONITOR</div>', unsafe_allow_html=True)
        seed = st.session_state.mm_seed
        tempo = st.session_state.get("mm_tempo", 100)
        if seed is not None:
            fig = dark_roll(seed, seed_len=melody_duration(seed) + 1)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.audio(melody_to_wav_bytes(seed, tempo_bpm=tempo), format="audio/wav")
            st.markdown(
                f"<div class='mono'>amber = seed · {len(seed)} notes · "
                f"{melody_duration(seed)} steps · <br>{melody_to_note_string(seed)}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='mono'>— no signal. Load a seed in SRC.</div>", unsafe_allow_html=True)

seed = st.session_state.mm_seed
tempo = st.session_state.get("mm_tempo", 100)

if go and seed is not None:
    with st.spinner("Sampling takes…"):
        fn = generate_continuations if mode == "continuation" else generate_variations
        st.session_state.mm_candidates = fn(
            model, tokenizer, seed, n_bars=bars, k=k,
            temperature=temperature, top_p=top_p, device=device)
        st.session_state.mm_mode = mode
        st.session_state.mm_tempo = tempo

candidates = st.session_state.mm_candidates

st.markdown(
    f"<div class='mast' id='ledger' style='margin-top:26px'>"
    f"<div><h1>Tape ledger <span>— {len(candidates)} take{'s' if len(candidates) != 1 else ''}</span></h1>"
    f"<p>MINT = GENERATED · AMBER = SEED REFERENCE</p></div></div>",
    unsafe_allow_html=True,
)

if not candidates:
    with st.container():
        st.markdown('<div class="mod-label">LEDGER // IDLE</div>', unsafe_allow_html=True)
        st.markdown("<div class='mono'>Press RUN in CTL. Takes land here as numbered ledger rows — deduped, motif-scored.</div>", unsafe_allow_html=True)
else:
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    mm_mode = st.session_state.get("mm_mode", "continuation")
    mm_tempo = st.session_state.get("mm_tempo", 100)
    for i, cand in enumerate(candidates):
        with st.container():
            score = f"{cand.motif_score:.2f}" if cand.motif_score is not None else "—"
            st.markdown(
                f"<div class='ledger-head'><span class='idx'>TAKE {i + 1:02d}</span>"
                f"<span class='score'>MOTIF {score}</span>"
                f"<span class='meta'>{len(cand.melody)} notes · {melody_duration(cand.melody)} steps</span></div>",
                unsafe_allow_html=True,
            )
            fig = dark_roll(cand.melody, seed_len=cand.seed_len_steps)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            a_col, d_col = st.columns([2.2, 1])
            with a_col:
                st.audio(melody_to_wav_bytes(cand.melody, tempo_bpm=mm_tempo), format="audio/wav")
            with d_col:
                midi_path = out_dir / f"{mm_mode}_{i + 1}.mid"
                write_midi(cand.melody, midi_path, tempo_bpm=mm_tempo)
                st.download_button("SAVE .MID", data=midi_path.read_bytes(),
                                   file_name=midi_path.name, mime="audio/midi",
                                   key=f"dl_{i}", use_container_width=True)

st.markdown(
    "<div class='mast' id='spec' style='margin-top:26px'>"
    "<div><h1>Spec sheet <span>— what the box does</span></h1></div></div>",
    unsafe_allow_html=True,
)
with st.container():
    st.markdown(
        """<table class="spec">
<tr><td>01 · Tokenizer</td><td>REMI — BAR / POS / PITCH / DUR on a 16th grid. Essen folksongs + Bach soprano lines, transposed to a common key.</td></tr>
<tr><td>02 · Model</td><td>Causal decoder-only transformer, ~3–4M params, trained from scratch on windowed melodies with transposition augmentation.</td></tr>
<tr><td>03 · Sampling</td><td>Temperature / top-p under a grammar mask — invalid sequences are impossible. Variation mode regenerates bar one hot and keeps the close-but-not-identical band.</td></tr>
</table>""",
        unsafe_allow_html=True,
    )

st.markdown(
    "<div class='mono' style='text-align:center; padding:26px 0 6px'>MELODY MORPH MM-01 · LAB 3 · SYMBOLIC MIDI ONLY</div>",
    unsafe_allow_html=True,
)
st.markdown(REVEAL_JS, unsafe_allow_html=True)
