"""Run the Streamlit app headlessly: catches import errors and script exceptions that
unit tests of the library never see."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")
CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "best.pt"

pytestmark = pytest.mark.skipif(not CHECKPOINT.exists(), reason="needs checkpoints/best.pt")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def test_console_runs_without_an_exception():
    at = _app()
    assert not at.exception, [e.value for e in at.exception]


def test_console_offers_the_sound_controls():
    at = _app()
    labels = {s.label for s in at.selectbox}
    assert {"Band style", "Lead instrument"} <= labels


def test_style_choice_is_remembered_in_session_state():
    at = _app()
    style = next(s for s in at.selectbox if s.label == "Band style")
    style.select("rock").run()
    assert not at.exception
    assert at.session_state["mm_style"] == "rock"


def test_triple_metre_only_offers_styles_that_have_a_pattern():
    at = _app()
    next(r for r in at.radio if r.label == "Metre").set_value("3/4").run()
    assert not at.exception
    style = next(s for s in at.selectbox if s.label == "Band style")
    assert set(style.options) == {"pop ballad", "waltz"}


def test_generate_lands_on_takes_with_a_band_for_each_take():
    at = _app()
    next(b for b in at.button if b.key == "run_btn").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.session_state["mm_candidates"]) == 4
