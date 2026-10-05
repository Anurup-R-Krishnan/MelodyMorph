---
name: MelodyMorph
description: A Musica Viva concert poster that is also an instrument; the melody draws the artwork.
colors:
  field-vermilion: "#F2512B"
  ink: "#12100E"
  paper: "#F5F4EF"
  ultramarine: "#1F3BE8"
  stale-red: "#B3200A"
  muted-ink: "#5C5852"
  hairline: "rgba(18, 16, 14, 0.16)"
typography:
  wordmark:
    fontFamily: "Archivo, Helvetica Neue, Arial, system-ui, sans-serif"
    fontSize: "clamp(3rem, 8vw, 6rem)"
    fontWeight: 800
    lineHeight: 0.86
    letterSpacing: "-0.04em"
    fontVariation: "'wdth' 112"
  page-title:
    fontFamily: "Archivo, Helvetica Neue, Arial, system-ui, sans-serif"
    fontSize: "clamp(2.6rem, 6.4vw, 6rem)"
    fontWeight: 800
    lineHeight: 0.88
    letterSpacing: "-0.04em"
    fontVariation: "'wdth' 112"
  sheet-heading:
    fontFamily: "Archivo, Helvetica Neue, Arial, system-ui, sans-serif"
    fontSize: "1.7rem"
    fontWeight: 800
    lineHeight: 1
    letterSpacing: "-0.035em"
    fontVariation: "'wdth' 108"
  action:
    fontFamily: "Archivo, Helvetica Neue, Arial, system-ui, sans-serif"
    fontSize: "1.6rem"
    fontWeight: 800
    lineHeight: 1
    letterSpacing: "-0.035em"
    fontVariation: "'wdth' 110"
  body:
    fontFamily: "Archivo, Helvetica Neue, Arial, system-ui, sans-serif"
    fontSize: "0.95rem"
    fontWeight: 500
    lineHeight: 1.5
  data:
    fontFamily: "Archivo, Helvetica Neue, Arial, system-ui, sans-serif"
    fontSize: "0.8rem"
    fontWeight: 700
    lineHeight: 1.4
    fontFeature: "'tnum'"
  label:
    fontFamily: "Archivo, Helvetica Neue, Arial, system-ui, sans-serif"
    fontSize: "0.72rem"
    fontWeight: 700
    letterSpacing: "0.06em"
rounded:
  none: "0"
spacing:
  xs: "4px"
  sm: "0.55rem"
  md: "1rem"
  lg: "1.4rem"
  xl: "2rem"
components:
  button-run:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    typography: "{typography.action}"
    rounded: "{rounded.none}"
    padding: "0 1.2rem"
    height: "4rem"
  button-run-hover:
    backgroundColor: "{colors.field-vermilion}"
    textColor: "{colors.ink}"
  button-secondary:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "0 0.9rem"
    height: "2.8rem"
  button-secondary-hover:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
  sheet:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "1.3rem 1.4rem"
  segment-selected:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    rounded: "{rounded.none}"
    padding: "0.6rem 0.7rem"
  stamp-stale:
    textColor: "{colors.stale-red}"
    rounded: "{rounded.none}"
    padding: "0.25rem 0.7rem 0.2rem"
  ticker:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    rounded: "{rounded.none}"
---

# Design System: MelodyMorph

## Overview

**Creative North Star: "The Concert Poster That Plays"**

A Musica Viva / Josef Mueller-Brockmann poster world, replacing the earlier dark hardware console. One flat saturated vermilion field carries everything; ink-black type and rules and warm paper sheets sit on it as flat planes. The melody itself draws the artwork: concentric arc fragments whose ring is pitch, angle is onset and sweep is duration, so the poster is evidence, not decoration. The logo mark is the same arc grammar drawn once. Recognisable with content removed: arcs on a flat colour field.

The system is code-led. There is no generated imagery; all artwork is computed from the melody and shipped as SVG or DOM/CSS. Density is poster-scale for display (wordmark and page titles at up to 6rem) against small, tabular, neutral grotesk for data. Edges are square, borders are ink rules, and there are no shadows.

Divergence from the first contract: a decorative page-level construction grid was drawn and then removed after review. No page grid exists. The only grids left are functional: the roll's 16th-step grid and the arc poster's construction rings and barline ticks.

**Key Characteristics:**
- One vermilion ground, ink rules, paper sheets; square corners everywhere.
- Archivo variable, using the width axis (wdth 105-112) for display and lowercase for the wordmark and action words.
- Artwork is generated from pitch, onset and duration, never decorative.
- Ultramarine is a signal colour for the playhead, the live note and keyboard focus.
- Every motion honours prefers-reduced-motion.

## Colors

A three-plane palette (field, ink, paper) with one cool signal and one warning red.

### Primary
- **Field Vermilion** (`{colors.field-vermilion}`): the page ground, hero ground, art frames, sung-note fill in the roll, and hover fill of the RUN and play buttons.

### Secondary
- **Ultramarine** (`{colors.ultramarine}`): playhead, the currently sounding note, keyboard focus outline, and the grid-reveal step lines in the roll.

### Tertiary
- **Stale Red** (`{colors.stale-red}`): only the STALE stamp.

### Neutral
- **Ink** (`{colors.ink}`): text, every border and rule, RUN button, selected segment, ticker, ghost notes.
- **Warm Paper** (`{colors.paper}`): sheets, take cards, roll ground, inputs, text on ink.
- **Muted Ink** (`{colors.muted-ink}`): hints and secondary meta only.
- **Hairline** (`{colors.hairline}`): faint dividers.

### Named Rules
**The Signal Blue Rule.** Ultramarine marks what is happening now (playhead, live note) and where the keyboard is. It never fills a surface or decorates.
**The One Field Rule.** There is exactly one ground colour. Panels are paper or ink, never a second hue.
**The Stamp Red Rule.** Stale red is reserved for the stale stamp; state words are stamped, not coloured pills.

## Typography

**Display Font:** Archivo variable (self-hosted woff2, OFL; weight 100-900, width 62-125), with Helvetica Neue, Arial, system-ui.
**Body Font:** the same Archivo. A single family; hierarchy comes from weight, width and size.
**Label/Mono Font:** none; tabular numerals (`tnum`) are enabled globally for measurements.

**Character:** A wide, heavy, tightly tracked grotesk at poster scale against a small neutral setting for data. Tight negative tracking on display; light positive tracking on uppercase labels.

### Hierarchy
- **Wordmark** (800, wdth 112, clamp(3rem, 8vw, 6rem), 0.86): "melodymorph" in lowercase, two lines, hero only.
- **Page title** (800, wdth 112, clamp(2.6rem, 6.4vw, 6rem), 0.88): Takes and Spec headings.
- **Sheet heading** (800, wdth 108, 1.7rem, 1): sheet titles with a 2px ink rule beneath. Take numerals use 3.4rem, spec headings 2.1rem, same recipe.
- **Action** (800, wdth 110, 1.6rem, lowercase): the RUN button and spinner text.
- **Body** (500, 0.95rem, 1.5, max 62ch): spec prose; hero sub 1.05rem at 30ch.
- **Data** (700, 0.8-0.88rem): readouts, note lines, clocks, benchmark values.
- **Label** (700, 0.72rem, 0.06em, uppercase): widget labels and captions.

### Named Rules
**The Width Axis Rule.** Display sizes widen (wdth 108-112); data stays at default width. Do not substitute a system display face.
**The Tabular Rule.** Numbers are tabular everywhere so readouts do not jitter during playback.

## Layout

Single column canvas, max 1360px, padding 1.4rem 2rem (1rem 1rem under 900px). A top bar (brand left, three plain-text page links right, 2px ink rule beneath) precedes every page. The Console hero is a 1.8:1 poster with the wordmark occupying the left 56% over the arc artwork; the control sheet and roll sit in Streamlit columns. Spec is a 3-column article grid with 1.2rem gaps. Take sheets stack with staggered entry. Vertical rhythm is 1rem between blocks, 1.2-1.6rem before major sections.

Below 900px: columns stack full width, the hero becomes a column (text first, a 15rem art band below with a 2px rule, object-position 74%), a hero-level RUN shortcut button appears, the spec grid is one column.

No page-level construction grid exists. Grids appear only where functional: the roll (below) and the arc poster.

## Elevation & Depth

Flat. There are no shadows; depth is conveyed by ink borders (2px for sheets and hero, 1.5px for controls) and by paper planes on the field. The one exception is the slider thumb, which uses a paper-then-ink double ring (not a drop shadow). Stamps use multiply blending so they read as ink pressed onto paper.

### Named Rules
**The Flat Plane Rule.** Layer by border and plane colour. No blur, no soft or offset shadows.

## Shapes

Square. Every radius is 0, including widgets, alerts, toasts and uploader. Borders are solid ink at 2px (structure) or 1.5px (controls). Segmented radios share borders by -1.5px overlap. Stamps are the single rotated element (-4deg, double 3px border). The spinner ring and the arcs are the only round forms, and the arcs are artwork.

## Components

### Buttons
- **Shape:** square, ink border.
- **Primary (RUN):** full-width, 4rem tall, ink ground, paper lowercase action text with a hard arrow mask at right. One per surface.
- **Hover / Focus:** RUN flips to vermilion with ink text and the arrow slides 8px; press scales to 0.985. Focus is a 3px ultramarine outline, 2px offset.
- **Secondary / Download:** transparent, 1.5px ink border, 2.8rem tall; fills ink on hover.

### Segmented Choice (radio)
Joined ink-bordered cells, no radio dots; the selected cell is ink with paper text.

### Sheets / Containers
Paper, 2px ink border, square, 1.3rem x 1.4rem padding, no shadow. Sheets enter with sheet-in (from the right) or sheet-up with staggered delays.

### Navigation
Top bar of plain text links (700, 1.05rem, wdth 105), no pill. Hover draws a 3px ink underline via inset shadow; the brand mark and name sit left.

### Piano Roll (signature)
DOM/CSS, percent-positioned notes on a 16th-step grid: bar lines in solid ink, beat lines in 34% ink, bar shading alternating. A CSS-only checkbox "reveal grid" adds ultramarine step lines. Seed notes are vermilion with ink border, generated notes ink, the sounding note ultramarine and scaled 1.45 vertically, the playhead a 3px ultramarine bar. A square ink play button sits in the roll bar with a clip-path morph from triangle to pause; clock is tabular at the right.

### Arc Poster and Contour (signature)
Self-contained SVGs: rings are pitch, angle is onset, sweep is duration. Arcs draw in note order and the ring set slowly rotates (160s). The interval contour is one bar per interval, up or down. The logo mark is the same system.

### Stamps
Uppercase 800, wdth 112, double border, rotated -4deg, slam-in animation. Fresh is ink, stale is stale red; a mini size exists for dense rows.

### Token Ribbon and Ticker
Small bordered chips (0.7rem): bar tokens ink-filled, pitch tokens vermilion, duration paper, position dashed. The ticker is an ink strip with a 60s marquee.

### Motion
Ease `cubic-bezier(.16, 1, .3, 1)` throughout. Vocabulary: arcs draw in note order, wordmark letter-rise (1.1s), sheet-in / sheet-up, stamp slam, token pop, note grow, ticker marquee, arc rotation. Under prefers-reduced-motion CSS durations collapse to ~0, the marquee stops, and the SVG artwork carries its own reduced-motion rule.

### Platform constraints
Streamlit's sanitiser strips inline `<svg>`. Static artwork therefore ships as `<img>` data-URI SVGs with their own embedded style and animation; the interactive roll must be DOM/CSS so a delegated script (player.js) can drive its playhead. The only raster is `icon.png` (favicon), a rasterisation of logo-on-field.svg, not generated. `.streamlit/config.toml` mirrors the palette (ink primary, vermilion background, paper secondary, Archivo, no radius). Palette constants in `melodymorph/artwork.py` mirror the CSS tokens and must change together.

## Do's and Don'ts

### Do:
- **Do** keep one vermilion ground and make every raised surface paper with a 2px ink border.
- **Do** keep every corner square and every shadow absent.
- **Do** derive new artwork from the melody (pitch to ring, onset to angle, duration to sweep).
- **Do** use ultramarine only for playhead, live note, step lines in the roll grid, and focus.
- **Do** ship new static artwork as data-URI `<img>` SVG with an embedded prefers-reduced-motion rule; use DOM/CSS for anything interactive.
- **Do** use stamped words for state, and one ink RUN per surface.

### Don't:
- **Don't** return to a dark hardware console with neon accents.
- **Don't** add a page-level decorative grid; it was tried and removed.
- **Don't** fill surfaces or chips with ultramarine, or introduce a second ground colour.
- **Don't** use blurred or offset shadows, rounded corners, or gradient fills.
- **Don't** introduce a second typeface or inline `<svg>` in Streamlit markdown.
