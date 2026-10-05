import base64
import re
import xml.dom.minidom

from melodymorph.artwork import (
    arc_poster_img,
    arc_poster_svg,
    contour_img,
    contour_svg,
    logo_svg,
    roll_figure_html,
    roll_geometry,
    roll_html,
    svg_img,
)
from melodymorph.midi_io import parse_note_string
from melodymorph.tokenizer import Note


def _melody():
    return parse_note_string("C4/q E4/q G4/h A4/e G4/e F4/q")


def _valid_xml(svg: str) -> None:
    xml.dom.minidom.parseString(svg)  # raises on malformed markup


def test_every_svg_is_well_formed_xml():
    m = _melody()
    for svg in (logo_svg(), arc_poster_svg(m, seed_len=16), arc_poster_svg(m, animate=False), contour_svg(m)):
        _valid_xml(svg)


def test_one_arc_and_one_roll_note_per_melody_note():
    m = _melody()
    assert len(re.findall(r'<path class="a ', arc_poster_svg(m))) == len(m)
    assert len(re.findall(r'<i class="n ', roll_html(m))) == len(m)


def test_seed_and_generated_notes_are_told_apart():
    m = [Note(0, 60, 4), Note(16, 64, 4)]
    arcs = arc_poster_svg(m, seed_len=16)
    assert 'class="a s' in arcs and 'class="a g' in arcs
    roll = roll_html(m, seed_len=16)
    assert 'class="n s' in roll and 'class="n g' in roll


def test_higher_pitch_sits_on_a_larger_ring():
    m = [Note(0, 60, 4), Note(4, 72, 4)]
    radii = [float(x) for x in re.findall(r"A([\d.]+) [\d.]+ 0", arc_poster_svg(m))]
    assert radii[0] < radii[1]


def test_higher_pitch_sits_higher_on_the_roll():
    m = [Note(0, 60, 4), Note(4, 72, 4)]
    tops = [float(x) for x in re.findall(r'top:([\d.]+)%;width', roll_html(m))]
    assert tops[1] < tops[0]  # smaller top = higher up


def test_empty_melody_still_renders():
    for svg in (arc_poster_svg([]), contour_svg([])):
        _valid_xml(svg)
    assert 'class="rl"' in roll_html([])


def test_roll_geometry_is_bar_aligned():
    m = [Note(0, 60, 4), Note(20, 65, 4)]
    g = roll_geometry(m, bar=12)
    assert g["steps"] % 12 == 0 and g["steps"] >= 24
    assert f'--steps:{g["steps"]}' in roll_html(m, bar=12)


def test_note_positions_are_percentages_that_fit_the_roll():
    html = roll_html([Note(0, 60, 4), Note(12, 64, 4)], min_steps=16)
    for left, width in re.findall(r"left:([\d.]+)%;top:[\d.]+%;width:([\d.]+)%", html):
        assert 0 <= float(left) and float(left) + float(width) <= 100.001


def test_figure_has_player_only_when_audio_is_given():
    m = _melody()
    assert "mm-play" not in roll_figure_html(m, uid="a")
    html = roll_figure_html(m, uid="b", audio_b64="AAAA", tempo=120)
    assert "mm-play" in html and 'data-sps="0.12500"' in html


def test_svg_ships_as_a_decodable_image_with_escaped_alt():
    img = svg_img(logo_svg(), 'say "hi" <b>')
    assert img.startswith("<img") and "<b>" not in img and "&quot;" in img
    payload = re.search(r"base64,([^\"]+)", img).group(1)
    _valid_xml(base64.b64decode(payload).decode())


def test_images_describe_the_melody_for_assistive_tech():
    assert "6 notes" in arc_poster_img(_melody()) and "Interval contour" in contour_img(_melody())
