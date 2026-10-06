// deck.js: the slide deck.
// Scales every 1600 x 900 slide to fit the window next to the rail, marks the current page in the
// rail, and tells the page's own code when the current page or the scale changes ("framechange", "slidescale").
(() => {
  const W = 1600,
    H = 900;
  const root = document.documentElement;
  const rail = document.querySelector(".rail");

  // ---- fit: one scale for every slide, the largest that fits beside the rail ----
  let scale = 0;
  function fit() {
    const w = root.clientWidth - (rail ? rail.offsetWidth : 0),
      h = window.innerHeight;
    const s = Math.min(w / W, h / H);
    if (Math.abs(s - scale) < 1e-4) return;
    scale = s;
    root.style.setProperty("--s", s.toFixed(5));
    document.dispatchEvent(
      new CustomEvent("slidescale", { detail: { scale: s } }),
    );
  }
  fit();
  addEventListener("resize", fit);

  // ---- current page: the one crossing the middle of the window ----
  const frames = [...document.querySelectorAll(".frame[id]")];
  const links = [...document.querySelectorAll(".rail a[href^='#']")];
  let current = -1;
  /** Move between pages while leaving keyboard input to focused controls. */
  function move(direction) {
    const at = frames.reduce(
      (best, frame, i) =>
        Math.abs(frame.getBoundingClientRect().top) <
        Math.abs(frames[best].getBoundingClientRect().top)
          ? i
          : best,
      0,
    );
    frames[
      Math.max(0, Math.min(frames.length - 1, at + direction))
    ].scrollIntoView();
  }
  document.addEventListener("keydown", (event) => {
    if (
      event.defaultPrevented ||
      event.target.closest("input,select,button,summary,canvas,[role=button]")
    )
      return;
    if (["ArrowLeft", "ArrowRight", "PageUp", "PageDown"].includes(event.key)) {
      event.preventDefault();
      move(["ArrowLeft", "PageUp"].includes(event.key) ? -1 : 1);
    }
  });
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        const i = frames.indexOf(e.target);
        if (i === current) continue;
        current = i;
        links.forEach((a) =>
          a.hash === `#${e.target.id}`
            ? a.setAttribute("aria-current", "true")
            : a.removeAttribute("aria-current"),
        );
        document.dispatchEvent(
          new CustomEvent("framechange", { detail: { id: e.target.id } }),
        );
      }
    },
    { rootMargin: "-45% 0px -45% 0px" },
  );
  frames.forEach((f) => io.observe(f));
})();
