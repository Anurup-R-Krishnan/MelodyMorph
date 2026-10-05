// One delegated listener drives every roll on the page: play/pause, a moving
// playhead, notes lighting up as they sound, and click-to-seek.
(function () {
  if (window.__mmPlayer) return;
  window.__mmPlayer = true;

  var raf = null;
  var current = null;

  function fmt(t) {
    t = Math.max(0, t);
    return Math.floor(t / 60) + ":" + String(Math.floor(t % 60)).padStart(2, "0");
  }

  function parts(fig) {
    return {
      audio: fig.querySelector("audio"),
      grid: fig.querySelector(".rl-grid"),
      ph: fig.querySelector(".ph"),
      clock: fig.querySelector(".mm-clock"),
      btn: fig.querySelector(".mm-play"),
      notes: fig.querySelectorAll(".n"),
    };
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
    paint(current);
    if (!parts(current).audio.paused) raf = requestAnimationFrame(loop);
  }

  function setPlaying(fig, on) {
    fig.classList.toggle("playing", on);
    var b = parts(fig).btn;
    if (b) b.setAttribute("aria-label", on ? "Pause" : "Play");
  }

  function stopOthers(except) {
    document.querySelectorAll("figure.mm-roll.playing").forEach(function (f) {
      if (f !== except) {
        parts(f).audio.pause();
        setPlaying(f, false);
      }
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

    if (e.target.closest(".mm-play")) {
      if (p.audio.paused) {
        stopOthers(fig);
        current = fig;
        if (p.audio.ended || p.audio.currentTime >= (p.audio.duration || 1e9) - 0.05) p.audio.currentTime = 0;
        var pr = p.audio.play();
        if (pr && pr.catch) pr.catch(function () {});
      } else {
        p.audio.pause();
      }
      return;
    }

    if (e.target.closest(".rl-grid")) {
      var r = p.grid.getBoundingClientRect();
      var frac = Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1);
      var t = frac * +p.grid.dataset.steps * parseFloat(fig.dataset.sps);
      if (isFinite(p.audio.duration)) t = Math.min(t, p.audio.duration);
      p.audio.currentTime = t;
      paint(fig);
    }
  });

  ["play", "pause", "ended", "timeupdate", "loadedmetadata"].forEach(function (ev) {
    document.addEventListener(
      ev,
      function (e) {
        var a = e.target;
        if (!a || a.tagName !== "AUDIO") return;
        var fig = a.closest("figure.mm-roll");
        if (!fig) return;
        if (ev === "play") {
          setPlaying(fig, true);
          current = fig;
          cancelAnimationFrame(raf);
          raf = requestAnimationFrame(loop);
        } else if (ev === "pause" || ev === "ended") {
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
