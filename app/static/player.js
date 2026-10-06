// One delegated listener drives every roll on the page: play/pause for a band of
// synchronised stems, a moving playhead, notes lighting up as they sound,
// click-to-seek, and the mixer (mute / solo / volume per instrument).
(function () {
  if (window.__mmPlayer) return;
  window.__mmPlayer = true;

  var raf = null;
  var current = null;
  var DRIFT = 0.05; // seconds a stem may stray from the lead before it is re-synced

  function fmt(t) {
    t = Math.max(0, t);
    return Math.floor(t / 60) + ":" + String(Math.floor(t % 60)).padStart(2, "0");
  }

  function parts(fig) {
    var all = fig.querySelectorAll("audio");
    return {
      all: all,
      audio: all[0],
      grid: fig.querySelector(".rl-grid"),
      ph: fig.querySelector(".ph"),
      clock: fig.querySelector(".mm-clock"),
      btn: fig.querySelector(".mm-play"),
      notes: fig.querySelectorAll(".n"),
    };
  }

  function sync(p) {
    var t = p.audio.currentTime;
    for (var i = 1; i < p.all.length; i++) {
      if (Math.abs(p.all[i].currentTime - t) > DRIFT) p.all[i].currentTime = t;
    }
  }

  function paint(fig) {
    var p = parts(fig);
    if (!p.audio || !p.grid) return;
    var sps = parseFloat(fig.dataset.sps);
    var steps = +p.grid.dataset.steps;
    var t = p.audio.currentTime;
    var step = t / sps;
    p.ph.style.left = Math.min(step / steps, 1) * 100 + "%";
    p.ph.style.opacity = t > 0 || !p.audio.paused ? 1 : 0;
    p.clock.textContent = fmt(t) + (isFinite(p.audio.duration) ? " / " + fmt(p.audio.duration) : "");
    p.notes.forEach(function (n) {
      var on = +n.dataset.on;
      var du = +n.dataset.du;
      n.classList.toggle("live", step >= on && step < on + du);
    });
  }

  function loop() {
    if (!current) return;
    var p = parts(current);
    paint(current);
    if (!p.audio.paused) {
      sync(p);
      raf = requestAnimationFrame(loop);
    }
  }

  function setPlaying(fig, on) {
    fig.classList.toggle("playing", on);
    var b = parts(fig).btn;
    if (b) b.setAttribute("aria-label", on ? "Pause" : "Play");
  }

  function pauseAll(fig) {
    parts(fig).all.forEach(function (a) { a.pause(); });
  }

  function stopOthers(except) {
    document.querySelectorAll("figure.mm-roll.playing").forEach(function (f) {
      if (f !== except) {
        pauseAll(f);
        setPlaying(f, false);
      }
    });
  }

  // Mixer: a soloed stem silences every other; otherwise a muted stem is silent.
  function applyMix(fig) {
    var rows = fig.querySelectorAll(".mx");
    var anySolo = fig.querySelector('.mx-s[aria-pressed="true"]') !== null;
    rows.forEach(function (row) {
      var audio = fig.querySelector('audio[data-stem="' + row.dataset.stem + '"]');
      if (!audio) return;
      var muted = row.querySelector(".mx-m").getAttribute("aria-pressed") === "true";
      var solo = row.querySelector(".mx-s").getAttribute("aria-pressed") === "true";
      var silent = anySolo ? !solo : muted;
      audio.muted = silent;
      audio.volume = row.querySelector(".mx-v").value / 100;
      row.classList.toggle("is-off", silent);
    });
  }

  document.addEventListener("click", function (e) {
    var go = e.target.closest && e.target.closest(".hero-go");
    if (go) {
      var sheet = document.querySelector(".st-key-sheet");
      if (sheet) sheet.scrollIntoView({ behavior: "smooth", block: "start" });
      return;
    }
    var fig = e.target.closest && e.target.closest("figure.mm-roll");
    if (!fig) return;
    var p = parts(fig);
    if (!p.audio) return;

    var mute = e.target.closest(".mx-m");
    var solo = e.target.closest(".mx-s");
    if (mute || solo) {
      var btn = mute || solo;
      btn.setAttribute("aria-pressed", btn.getAttribute("aria-pressed") === "true" ? "false" : "true");
      applyMix(fig);
      return;
    }

    if (e.target.closest(".mm-play")) {
      if (p.audio.paused) {
        stopOthers(fig);
        current = fig;
        applyMix(fig);
        if (p.audio.ended || p.audio.currentTime >= (p.audio.duration || 1e9) - 0.05) {
          p.all.forEach(function (a) { a.currentTime = 0; });
        }
        sync(p);
        p.all.forEach(function (a) {
          var pr = a.play();
          if (pr && pr.catch) pr.catch(function () {});
        });
      } else {
        pauseAll(fig);
      }
      return;
    }

    if (e.target.closest(".rl-grid")) {
      var r = p.grid.getBoundingClientRect();
      var frac = Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1);
      var t = frac * +p.grid.dataset.steps * parseFloat(fig.dataset.sps);
      if (isFinite(p.audio.duration)) t = Math.min(t, p.audio.duration);
      p.all.forEach(function (a) { a.currentTime = t; });
      paint(fig);
    }
  });

  document.addEventListener("input", function (e) {
    var v = e.target.closest && e.target.closest(".mx-v");
    if (!v) return;
    var fig = v.closest("figure.mm-roll");
    if (fig) applyMix(fig);
  });

  // Only the first audio element (the lead) drives the clock and the playhead.
  ["play", "pause", "ended", "timeupdate", "loadedmetadata"].forEach(function (ev) {
    document.addEventListener(
      ev,
      function (e) {
        var a = e.target;
        if (!a || a.tagName !== "AUDIO") return;
        var fig = a.closest("figure.mm-roll");
        if (!fig || a !== fig.querySelector("audio")) return;
        if (ev === "play") {
          setPlaying(fig, true);
          current = fig;
          cancelAnimationFrame(raf);
          raf = requestAnimationFrame(loop);
        } else if (ev === "pause" || ev === "ended") {
          if (ev === "ended") pauseAll(fig);
          setPlaying(fig, false);
          paint(fig);
        } else {
          paint(fig);
        }
      },
      true
    );
  });
})();
