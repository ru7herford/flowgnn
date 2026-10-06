// The deck's flow fields, what each network is told, the training replay, the results and the comparison
// of any test shape. The One round, How far and Influence slides live in graph.js. Every number shown
// is read from docs/data/, written by scripts/export_page.py from the saved runs.
import { createGraphExplorer } from "./graph.js";
import { drawField, fieldColour, scales } from "./fields.js";

const $ = (id) => document.getElementById(id);
// The four networks differ only in what each point is told; the code names them M0-M3.
const names = { M0: "mesh only", M1: "mesh + distance", M2: "mesh + shape", M3: "mesh + distance + shape" };
const colours = { M0: "#B5520B", M1: "#91A8BA", M2: "#3D546A", M3: "#087F8C" };
const label = (s) => `${s}: ${names[s]}`;
// Windows onto the channel for each kind of picture, in flow units (the obstacle is about 1 long).
const VIEWS = {
  hero: [-2.2, -1.98, 8.8, 1.98],
  tile: [-1.3, -2.25, 3.9, 2.25],
  map: [-1.2, -1.5, 4.6, 1.5],
  inputs: [-0.75, -0.95, 1.75, 1.15],
  learn: [-1.5, -1.4, 6.5, 1.4],
};
const pct = (x) => `${(100 * x).toFixed(1)}%`;
// The velocity error ε_u and pressure error ε_p (defined on the Results slide), in HTML and in SVG text.
const eps = (q) => `ε<sub>${q}</sub>`;
const epsSvg = (q) => `ε<tspan baseline-shift="sub" font-size="75%">${q}</tspan>`;
/** Seconds as a short readable duration: "0.02 s", "78 s", "4 minutes". */
const duration = (t) => (t < 1 ? `${t.toFixed(2)} s` : t < 120 ? `${Math.round(t)} s` : `${Math.round(t / 60)} minutes`);
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;

const state = { summary: null, cache: new Map(), caseId: "triangle_05", distance: false, shape: false, field: 0 };
const training = { at: 0, timer: null, played: false, view: "prediction", bins: new Map() };
const model = () => (state.distance ? (state.shape ? "M3" : "M1") : state.shape ? "M2" : "M0");
const record = (id) => state.summary.cases.find((c) => c.id === id);

/** Show a failure on the page and stop every control. */
function fail(error) {
  stopTraining();
  document.querySelectorAll("button").forEach((b) => (b.disabled = true));
  $("load-error").hidden = false;
  $("load-error").textContent = `Could not load recorded data: ${error.message}. Serve docs/ over HTTP and check the export files.`;
  console.error(error);
}

async function fetchData(url, json = false) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return json ? r.json() : r.arrayBuffer();
}

/** One shape's binary file: positions, triangles, the solver fields and each network's prediction. */
async function caseData(rec) {
  if (state.cache.has(rec.id)) return state.cache.get(rec.id);
  const b = await fetchData(`data/${rec.id}.bin`),
    n = rec.nodes;
  let offset = 8 * n + 12 * rec.triangles;
  const take = () => {
    const a = new Float32Array(b, offset, n);
    offset += 4 * n;
    return a;
  };
  const data = {
    pos: new Float32Array(b, 0, 2 * n),
    tri: new Uint32Array(b, 8 * n, 3 * rec.triangles),
    reference: [take(), take(), take()],
    predictions: {},
  };
  for (const s of Object.keys(state.summary.input_sets)) data.predictions[s] = [take(), take(), take()];
  offset += 4 * n; // node ids in the complete mesh (used by the tests, not drawn)
  if (offset !== b.byteLength) throw new Error(`Unexpected field layout: ${rec.id}`);
  state.cache.set(rec.id, data);
  return data;
}

/** Mean of one shape's error over the three trained networks of an input choice. */
const shapeError = (rec, s, metric) => mean(rec.errors[s].map((r) => r.all[metric]));

/** Velocity error of one input choice on one type of test shape, per trained network: the mean over those
 *  shapes. The Results slide draws the mean and range of these; the opening line quotes the mean. */
function groupErrors(shape, s) {
  const cases = state.summary.cases.filter((c) => c.split === "test" && c.shape === shape);
  return cases[0].errors[s].map((_, r) => mean(cases.map((c) => c.errors[s][r].all.velocity)));
}

// ---- Start and the task ---------------------------------------------------------------------------

/** The opening comparison and the four task tiles, all from the same unseen triangle. */
async function drawIntroduction() {
  const rec = record("triangle_05"),
    data = await caseData(rec),
    colour = fieldColour(rec, 0);
  // The opening line states the main result on the test triangles (a shape type not in training): what the
  // six shape inputs change. The picture is one of those triangles, predicted by M3.
  $("hero-lead").textContent = "I trained one graph neural network (GNN) four times, each time with different inputs " +
    "at each mesh point. Six numbers that describe the obstacle cut its error on unseen triangles from " +
    `${pct(mean(groupErrors("triangle", "M0")))} to ${pct(mean(groupErrors("triangle", "M2")))}.`;
  $("hero-tag").innerHTML = `GNN: ${names.M3} · ${eps("u")} = ${(100 * shapeError(rec, "M3", "velocity")).toFixed(0)}%`;
  drawField($("hero-reference"), rec, data, data.reference[0], colour, VIEWS.hero);
  drawField($("hero-prediction"), rec, data, data.predictions.M3[0], colour, VIEWS.hero);
  $("hero-scale").innerHTML = scales(rec, 0, "field");
  drawField($("task-reference"), rec, data, data.reference[0], colour, VIEWS.tile);
  drawField($("task-prediction"), rec, data, data.predictions.M3[0], colour, VIEWS.tile);

  const [x0, y0, x1, y1] = VIEWS.tile;
  const X = (x) => ((x - x0) / (x1 - x0)) * 300,
    Y = (y) => ((y1 - y) / (y1 - y0)) * 260;
  const outline = `<path d="${rec.outline.map(([x, y], i) => `${i ? "L" : "M"}${X(x)},${Y(y)}`).join(" ")}Z" class="shape"/>`;
  $("task-shape").innerHTML = `<rect width="300" height="260" class="tile-bg"/>${outline}`;
  let links = "",
    points = "";
  const inside = (i) => data.pos[2 * i] > x0 && data.pos[2 * i] < x1 && data.pos[2 * i + 1] > y0 && data.pos[2 * i + 1] < y1;
  for (let t = 0; t < data.tri.length; t += 3) {
    const ids = Array.from(data.tri.subarray(t, t + 3));
    if (!ids.every(inside)) continue;
    links += `<path d="${ids.map((i, k) => `${k ? "L" : "M"}${X(data.pos[2 * i])},${Y(data.pos[2 * i + 1])}`).join(" ")}Z"/>`;
  }
  for (let i = 0; i < rec.nodes; i++)
    if (inside(i)) points += `<circle cx="${X(data.pos[2 * i])}" cy="${Y(data.pos[2 * i + 1])}" r="1.6"/>`;
  $("task-graph").innerHTML = `<rect width="300" height="260" class="tile-bg"/><g class="mesh-links">${links}</g><g class="mesh-points">${points}</g>${outline}`;
}

/** Drag (or arrow keys) moves the line between the solver on the left and the network on the right. */
function connectCompare() {
  const box = $("compare"),
    divider = $("divider");
  const set = (percent) => {
    const p = Math.max(0, Math.min(100, percent));
    $("hero-prediction").style.clipPath = `inset(0 0 0 ${p}%)`;
    divider.style.left = `${p}%`;
    divider.setAttribute("aria-valuenow", Math.round(p));
  };
  const follow = (event) => {
    const r = box.getBoundingClientRect();
    set(((event.clientX - r.left) / r.width) * 100);
  };
  box.addEventListener("pointerdown", (event) => {
    box.setPointerCapture(event.pointerId);
    follow(event);
  });
  box.addEventListener("pointermove", (event) => {
    if (box.hasPointerCapture(event.pointerId)) follow(event);
  });
  divider.addEventListener("keydown", (event) => {
    const step = { ArrowLeft: -5, ArrowRight: 5 }[event.key];
    if (!step) return;
    event.preventDefault();
    set(Number(divider.getAttribute("aria-valuenow")) + step);
  });
  set(50);
}

// ---- Four networks: what each network is told -----------------------------------------------------

/** The traced point on its triangle, its distance and direction to the obstacle, the obstacle's size, and
 *  the inputs of M0 and M3 with what each one means. Values from graph.example_inputs. */
async function drawInputs(graph) {
  const rec = record(graph.case),
    data = await caseData(rec),
    ex = graph.example_inputs;
  const [x0, y0, x1, y1] = VIEWS.inputs;
  const X = (x) => ((x - x0) / (x1 - x0)) * 620,
    Y = (y) => ((y1 - y) / (y1 - y0)) * 520;
  const inside = (i) => data.pos[2 * i] > x0 && data.pos[2 * i] < x1 && data.pos[2 * i + 1] > y0 && data.pos[2 * i + 1] < y1;
  let svg = `<rect width="620" height="520" class="tile-bg"/><g class="mesh-links">`;
  for (let t = 0; t < data.tri.length; t += 3) {
    const ids = Array.from(data.tri.subarray(t, t + 3));
    if (ids.every(inside)) svg += `<path d="${ids.map((i, k) => `${k ? "L" : "M"}${X(data.pos[2 * i])},${Y(data.pos[2 * i + 1])}`).join(" ")}Z"/>`;
  }
  const xs = rec.outline.map((p) => p[0]),
    ys = rec.outline.map((p) => p[1]);
  const [bx0, bx1, by0, by1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  svg += `</g><path d="${rec.outline.map(([x, y], i) => `${i ? "L" : "M"}${X(x)},${Y(y)}`).join(" ")}Z" class="shape"/>`;
  // The obstacle's length (along the flow) and height (across it): two of the six shape numbers.
  svg += `<rect x="${X(bx0)}" y="${Y(by1)}" width="${X(bx1) - X(bx0)}" height="${Y(by0) - Y(by1)}" class="extent"/>` +
    `<text x="${(X(bx0) + X(bx1)) / 2}" y="${Y(by0) + 30}" class="dim">length ${ex.shape[0].toFixed(2)}</text>` +
    `<text x="${X(bx0) - 12}" y="${(Y(by0) + Y(by1)) / 2 + 8}" class="dim halo" style="text-anchor:end">height ${ex.shape[1].toFixed(2)}</text>`;
  const px = X(ex.x),
    py = Y(ex.y),
    qx = X(ex.x + ex.distance * ex.direction[0]),
    qy = Y(ex.y + ex.distance * ex.direction[1]);
  svg += `<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" class="arrowhead"/></marker></defs>` +
    `<line x1="${px}" y1="${py}" x2="${qx}" y2="${qy}" class="distance" marker-end="url(#head)"/>` +
    `<text x="${(px + qx) / 2 + 14}" y="${(py + qy) / 2 - 10}" class="dim accent">distance ${ex.distance.toFixed(2)}</text>` +
    `<circle cx="${px}" cy="${py}" r="11" class="this-point"/><text x="${px + 18}" y="${py - 14}" class="dim strong">this point</text>`;
  $("inputs-drawing").innerHTML = svg;

  $("model-key").innerHTML = Object.keys(names)
    .map((s) => `<span class="model"><i style="background:${colours[s]}"></i><b>${s}:</b>${names[s]}</span>`)
    .join("");
  $("column-M0").textContent = label("M0");
  $("column-M3").textContent = label("M3");
  const f = (v) => v.toFixed(2).replace("-", "−");
  // [name, value, what it means]; the first three are what every network gets.
  const mesh = [["x", f(ex.x), "along the channel"], ["y", f(ex.y), "across it"], ["type", ex.type, "fluid, wall, inlet, outlet, obstacle"]];
  const distance = [
    ["distance", f(ex.distance), "to the nearest point of the obstacle"],
    ["direction", `${f(ex.direction[0])}, ${f(ex.direction[1])}`, "towards that point"],
  ];
  const shape = ["length", "height", "area", "perimeter", "centre x", "centre y"].map((n, i) => [n, f(ex.shape[i])]);
  shape[0][2] = "obstacle, along the flow";
  shape[1][2] = "across the flow";
  shape[4][2] = "where the obstacle sits";
  const rows = (cells, cls) => cells.map(([n, v, meaning = ""]) => `<div class="input ${cls}"><dt>${n}</dt><dd>${v}</dd><small>${meaning}</small></div>`).join("");
  $("row-mesh").innerHTML = rows(mesh.map(([n, v]) => [n, v]), "");
  $("row-both").innerHTML = rows(mesh, "") + `<div class="divide">+ distance</div>` + rows(distance, "added") +
    `<div class="divide">+ shape</div>` + rows(shape, "added");
}

// ---- Explore any test shape -------------------------------------------------------------------------

/** The shape buttons: a small outline of each unseen shape. */
function drawShapePicker() {
  const tests = state.summary.cases.filter((c) => c.split === "test");
  $("shape-picker").innerHTML = tests
    .map((c) => {
      const xs = c.outline.map((p) => p[0]),
        ys = c.outline.map((p) => p[1]);
      const s = Math.min(48 / (Math.max(...xs) - Math.min(...xs)), 30 / (Math.max(...ys) - Math.min(...ys)));
      const cx = (Math.max(...xs) + Math.min(...xs)) / 2,
        cy = (Math.max(...ys) + Math.min(...ys)) / 2;
      const title = c.shape === "triangle" ? `Triangle, ${c.params.angle_deg}°` : `Airfoil NACA ${c.params.naca}, ${c.params.aoa_deg}°`;
      return `<button class="thumb" data-case="${c.id}" aria-pressed="${c.id === state.caseId}" title="${title}" aria-label="${title}"><svg viewBox="0 0 60 40"><path d="${c.outline.map(([x, y], i) => `${i ? "L" : "M"}${30 + (x - cx) * s},${20 - (y - cy) * s}`).join(" ")}Z"/></svg></button>`;
    })
    .join("");
}

/** Solver, prediction and error for the chosen shape, input choice and field, plus all four scores. */
async function render() {
  const rec = record(state.caseId),
    data = await caseData(rec),
    s = model(),
    k = state.field;
  const errors = Float32Array.from(data.predictions[s][k], (v, i) => Math.abs(v - data.reference[k][i]));
  drawField($("reference"), rec, data, data.reference[k], fieldColour(rec, k), VIEWS.map);
  drawField($("prediction"), rec, data, data.predictions[s][k], fieldColour(rec, k), VIEWS.map);
  drawField($("error"), rec, data, errors, fieldColour(rec, k, true), VIEWS.map);
  const metric = k === 2 ? "pressure" : "velocity";
  $("prediction-title").textContent = `GNN · ${label(s)}`;
  $("score-label").innerHTML = `${eps(metric === "pressure" ? "p" : "u")} on this shape (mean of 3 trained networks)`;
  $("score-definition").innerHTML = "Error map: the difference between GNN and SU2 at each point. " + (metric === "pressure"
    ? `${eps("p")} = ‖p<sub>GNN</sub> − p<sub>SU2</sub>‖ / ‖p<sub>SU2</sub>‖ over all mesh points.`
    : `${eps("u")}: as defined on the Results slide.`);
  const scores = Object.fromEntries(Object.keys(names).map((m) => [m, shapeError(rec, m, metric)]));
  const top = Math.max(...Object.values(scores));
  $("score-bars").innerHTML = Object.keys(names)
    .map((m) => `<div class="bar${m === s ? " current" : ""}" data-model="${m}"><span><b>${m}:</b> ${names[m]}</span><i style="width:${(100 * scores[m]) / top}%"></i><b>${pct(scores[m])}</b></div>`)
    .join("");
  $("field-scales").innerHTML = scales(rec, k);
  $("add-distance").setAttribute("aria-pressed", state.distance);
  $("add-shape").setAttribute("aria-pressed", state.shape);
  for (const b of $("field-choice").children) b.setAttribute("aria-pressed", Number(b.dataset.field) === k);
  for (const b of $("shape-picker").children) b.setAttribute("aria-pressed", b.dataset.case === state.caseId);
  Object.assign($("maps").dataset, { case: rec.id, model: s, field: k });
}

// ---- Training: the replay ------------------------------------------------------------------------------

/** Recorded horizontal velocity of one network at every snapshot epoch (16-bit over a shared range). */
async function progressData(s) {
  if (!training.bins.has(s)) {
    const b = await fetchData(`data/progress_${s}.bin`);
    const p = state.summary.progress,
      n = record(p.case).nodes,
      [low, high] = p.range;
    if (b.byteLength !== p.epochs.length * n * 2) throw new Error(`Unexpected snapshot layout: ${s}`);
    training.bins.set(s, Float32Array.from(new Uint16Array(b), (q) => low + (q / 65535) * (high - low)));
  }
  return training.bins.get(s);
}

/** The recorded epoch from which a network's velocity error stays below a level for good, or null. */
const settledBelow = (s, level) => {
  const p = state.summary.progress,
    i = p.runs[s].velocity.findLastIndex((v) => v >= level);
  return i === p.epochs.length - 1 ? null : p.epochs[i + 1];
};

/** SU2's answer, both networks' predictions at the current snapshot, and the error curves up to it. */
async function drawTraining() {
  const p = state.summary.progress,
    rec = record(p.case),
    at = training.at,
    error = training.view === "error",
    n = rec.nodes;
  const [data, ...frames] = await Promise.all([caseData(rec), progressData("M0"), progressData("M3")]);
  drawField($("training-reference"), rec, data, data.reference[0], fieldColour(rec, 0), VIEWS.learn);
  ["M0", "M3"].forEach((s, k) => {
    const values = frames[k].subarray(at * n, (at + 1) * n);
    const drawn = error ? Float32Array.from(values, (v, i) => Math.abs(v - data.reference[0][i])) : values;
    drawField($(`training-${s}`), rec, data, drawn, fieldColour(rec, 0, error), VIEWS.learn);
    $(`training-name-${s}`).textContent = label(s);
    $(`training-error-${s}`).innerHTML = `${eps("u")} = ${pct(p.runs[s].velocity[at])}`;
  });
  $("training-epoch").textContent = `Epoch ${p.epochs[at]} of ${state.summary.epochs}`;
  $("training-scrub").value = at;
  $("training-scales").innerHTML = scales(rec, 0, error ? "error" : "field");
  for (const b of $("training-view").children) b.setAttribute("aria-pressed", b.dataset.view === training.view);
  // The curves grow with the replay: only the epochs recorded so far, and each mark once it is reached.
  const series = {};
  for (const s of ["M0", "M3"]) series[s] = p.epochs.slice(0, at + 1).map((e, i) => [e, p.runs[s].velocity[i]]);
  const settled = ["M3", "M0"].map((s) => [s, settledBelow(s, 0.5)]).filter(([, e]) => e !== null);
  $("training-finding").innerHTML = settled.length === 2
    ? `M3 stays below ${eps("u")} = 50% from epoch ${settled[0][1]}, M0 only from epoch ${settled[1][1]}.`
    : "";
  const marks = settled.filter(([, e]) => e <= p.epochs[at]);
  chart($("training-chart"), series, {
    xmax: state.summary.epochs,
    ymax: 1.2,
    xlabel: "epoch (one pass through all training shapes)",
    ylabel: `${epsSvg("u")} on this triangle`,
    levels: [[0.5, "50%"], [1, "100%: same as predicting the inflow everywhere"]],
    marks,
    cursor: p.epochs[at],
  });
  $("training-chart").dataset.cursor = p.epochs[at];
}

function setTraining(at) {
  stopTraining();
  training.at = Math.max(0, Math.min(state.summary.progress.epochs.length - 1, at));
  drawTraining().catch(fail);
}
function stopTraining() {
  clearInterval(training.timer);
  training.timer = null;
  $("training-play").textContent = "Play";
}
function playTraining() {
  if (training.timer) return stopTraining();
  const last = state.summary.progress.epochs.length - 1;
  if (training.at === last || !training.played) training.at = 0;
  training.played = true;
  $("training-play").textContent = "Pause";
  drawTraining().catch(fail);
  // About 12 s for the 152 recorded snapshots: each frame shows 2 epochs of training.
  training.timer = setInterval(() => {
    if (training.at === last) return stopTraining();
    training.at++;
    drawTraining().catch(fail);
  }, 80);
}

// ---- Charts, results, limits ------------------------------------------------------------------------

/** Line chart drawn from saved points. Lines are labelled at their ends, spaced so labels never overlap. */
function chart(svg, series, { xmax, ymax, xlabel, xsteps = 5, ylabel = "", levels = [], marks = [], cursor = null, vline = null }) {
  const [W, H] = [svg.viewBox.baseVal.width, svg.viewBox.baseVal.height];
  const m = { l: ylabel ? 124 : 82, r: 60, t: 22, b: 58 };
  const X = (x) => m.l + (x / xmax) * (W - m.l - m.r),
    Y = (y) => H - m.b - (Math.min(y, ymax) / ymax) * (H - m.t - m.b);
  let html = "";
  for (let i = 0; i <= 4; i++) {
    const y = (ymax * i) / 4;
    html += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(y)}" y2="${Y(y)}" class="grid"/><text x="${m.l - 12}" y="${Y(y) + 7}" class="tick" text-anchor="end">${Math.round(100 * y)}%</text>`;
  }
  for (let i = 0; i <= xsteps; i++) {
    const x = (xmax * i) / xsteps;
    html += `<text x="${X(x)}" y="${H - m.b + 30}" class="tick" text-anchor="middle">${Math.round(x)}</text>`;
  }
  // Reference levels are labelled just below their line, at the right, clear of the mark labels above.
  for (const [y, text] of levels)
    html += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(y)}" y2="${Y(y)}" class="level"/><text x="${W - m.r}" y="${Y(y) + 26}" class="tick" text-anchor="end">${text}</text>`;
  if (vline !== null) html += `<line x1="${X(vline[0])}" x2="${X(vline[0])}" y1="${m.t}" y2="${H - m.b}" class="level"/><text x="${X(vline[0]) + 8}" y="${m.t + 22}" class="tick">${vline[1]}</text>`;
  const ends = [];
  for (const [s, points] of Object.entries(series)) {
    const d = points.map(([x, y], i) => `${i ? "L" : "M"}${X(x)},${Y(y)}`).join(" ");
    html += `<path d="${d}" fill="none" stroke="${colours[s]}" stroke-width="${s === "M3" ? 4 : 3}" ${s === "M0" ? 'stroke-dasharray="9 6"' : ""}/>`;
    const [lx, ly] = points.at(-1);
    ends.push({ s, x: X(lx), y: Y(ly) });
  }
  ends.sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 28);
  for (const e of ends) html += `<text x="${e.x + 12}" y="${e.y + 8}" class="key" fill="${colours[e.s]}">${e.s}</text>`;
  // Each mark is labelled in its own row above the lines, joined to its point by a thin leader.
  marks.forEach(([s, epoch], row) => {
    const y = series[s].find(([x]) => x === epoch)[1],
      ty = m.t + 18 + 28 * row;
    html += `<line x1="${X(epoch)}" x2="${X(epoch)}" y1="${Y(y)}" y2="${ty + 6}" stroke="${colours[s]}" stroke-width="1.5"/>` +
      `<circle cx="${X(epoch)}" cy="${Y(y)}" r="7" fill="${colours[s]}"/>` +
      `<text x="${X(epoch) + 8}" y="${ty}" class="mark halo" fill="${colours[s]}">${s} below 50% from epoch ${epoch}</text>`;
  });
  if (cursor !== null) html += `<line x1="${X(cursor)}" x2="${X(cursor)}" y1="${m.t}" y2="${H - m.b}" class="cursor"/>`;
  html += `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" class="axis" text-anchor="middle">${xlabel}</text>`;
  // The y label runs along the axis, left of the tick values.
  if (ylabel) html += `<text transform="translate(26 ${(m.t + H - m.b) / 2}) rotate(-90)" class="axis" text-anchor="middle">${ylabel}</text>`;
  svg.innerHTML = html;
}

/** Results: the four networks' velocity error against SU2, grouped by test shape type. Each bar is the mean over
 *  the three trained networks; the thin line spans the lowest and highest of the three. An arrow from M0 to M2
 *  in each group shows what the six shape inputs change. */
function resultsChart(svg) {
  const [W, H] = [svg.viewBox.baseVal.width, svg.viewBox.baseVal.height];
  const m = { l: 124, r: 10, t: 130, b: 96 }, // room above the bars for the brackets, and left for the y label
    ymax = 0.25;
  const Y = (y) => H - m.b - (y / ymax) * (H - m.t - m.b);
  const groups = [["triangle", "6 test triangles", "shape type not in training"], ["naca", "4 test airfoils", "new profiles and angles"]];
  const groupWidth = (W - m.l - m.r) / groups.length,
    slot = (groupWidth - 40) / 4;
  let html = `<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0L10 5L0 10Z" class="arrowhead"/></marker></defs>`;
  for (let i = 0; i <= 5; i++) {
    const y = (ymax * i) / 5;
    html += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(y)}" y2="${Y(y)}" class="grid"/><text x="${m.l - 12}" y="${Y(y) + 7}" class="tick" text-anchor="end">${Math.round(100 * y)}%</text>`;
  }
  html += `<text transform="translate(26 ${(m.t + H - m.b) / 2}) rotate(-90)" class="axis" text-anchor="middle">${epsSvg("u")} (lower is better)</text>`;
  groups.forEach(([shape, title, note], g) => {
    const x0 = m.l + g * groupWidth + 20;
    html += `<text x="${x0 + 2 * slot}" y="${H - 34}" class="key" text-anchor="middle">${title}</text>` +
      `<text x="${x0 + 2 * slot}" y="${H - 8}" class="tick" text-anchor="middle">${note}</text>`;
    const tops = {};
    Object.keys(names).forEach((s, k) => {
      const seeds = groupErrors(shape, s);
      const v = mean(seeds),
        x = x0 + k * slot + 40,
        w = slot - 80;
      tops[s] = { x: x + w / 2, y: Y(Math.max(...seeds)) - 34, v };
      html += `<rect x="${x}" y="${Y(v)}" width="${w}" height="${Y(0) - Y(v)}" rx="4" fill="${colours[s]}" data-model="${s}" data-shape="${shape}" data-value="${v}"/>` +
        `<line x1="${x + w / 2}" x2="${x + w / 2}" y1="${Y(Math.min(...seeds))}" y2="${Y(Math.max(...seeds))}" class="range"/>` +
        `<text x="${x + w / 2}" y="${Y(Math.max(...seeds)) - 10}" class="value" text-anchor="middle">${pct(v)}</text>` +
        `<text x="${x + w / 2}" y="${Y(0) + 28}" class="key" text-anchor="middle" fill="${colours[s]}">${s}</text>`;
    });
    // A bracket over the bars from M0 down to M2, clear of the value labels in between.
    const [from, to] = [tops.M0, tops.M2],
      over = Math.min(from.y, tops.M1.y, to.y) - 30;
    html += `<path d="M${from.x},${from.y} V${over} H${to.x} V${to.y}" fill="none" class="effect" marker-end="url(#arrow)"/>` +
      `<text x="${(from.x + to.x) / 2}" y="${over - 12}" class="key" text-anchor="middle">shape inputs: ${pct(from.v)} → ${pct(to.v)}</text>`;
  });
  svg.innerHTML = html;
}

/** The speed panel on the limits slide: one network's prediction time against SU2's solve time, per test shape. */
function speedPanel(speed) {
  const [fastest, median, slowest] = speed.su2_seconds;
  const minutes = (t) => `${Math.round(t / 60)} min`;
  const gpu = speed.gpu
    ? `<div class="speed-row"><b>${duration(speed.gpu.warm_seconds)}</b><span>GNN, on a laptop GPU (${duration(speed.cpu.warm_seconds)} without a GPU)</span></div>`
    : "";
  return `<h3>Time per test shape</h3>${gpu}` +
    `<div class="speed-row"><b>${minutes(median)}</b><span>SU2, median (${duration(fastest)} to ${minutes(slowest)}), on 9× more points</span></div>`;
}

/** Results, where the error is, and limits: the bars, error by links from the obstacle, the rounds experiment,
 *  speed and the checks. */
function evidence() {
  const summary = state.summary,
    series = {};
  resultsChart($("results-chart"));
  $("speed").hidden = !summary.timing;
  if (summary.timing) $("speed").innerHTML = speedPanel(summary.timing);
  const bins = summary.hop_bins;
  for (const [s, values] of Object.entries(summary.hop_error)) series[s] = values.map((v, i) => [(bins[i] + bins[i + 1]) / 2, v]);
  chart($("hop-chart"), series, { xmax: bins.at(-1), xsteps: 4, ymax: 0.6, xlabel: "links from the obstacle",
    ylabel: `${epsSvg("u")} in each band`, vline: [12, "12 rounds"] });
  // Same network with 6, 12 and 18 rounds (seed 0, 150 epochs, validation error): does a larger reach help?
  const rx = summary.rounds_experiment;
  if (rx) {
    const table = $("rounds-table");
    table.hidden = false;
    table.querySelector("thead tr").innerHTML = `<th>Rounds</th>${rx.rounds.map((r) => `<th>${r}</th>`).join("")}`;
    table.querySelector("tbody").innerHTML = ["M0", "M3"]
      .map((s) => `<tr><th><i style="background:${colours[s]}"></i>${label(s)}</th>${rx[s].val.map((v) => `<td>${pct(v)}</td>`).join("")}</tr>`)
      .join("");
  }
  $("checks").innerHTML = [
    ["Checked", ["SU2 circle: drag and wake length inside the published spread; separation angle 0.4° above it",
      "The messages on the One round slide equal the saved network's messages",
      "The influence values agree with finite differences", "Tests for the graph, model and data"]],
    ["Limits", ["One flow type: steady, Reynolds number 40", "34 training shapes; no triangles in training",
      "Half the training shapes: 2 to 3 times the validation error. More data should help."]],
  ].map(([h, lines]) => `<div><h3>${h}</h3><ul>${lines.map((l) => `<li>${l}</li>`).join("")}</ul></div>`).join("");
}

// ---- Start ---------------------------------------------------------------------------------------

async function main() {
  const summary = await fetchData("data/summary.json", true);
  if (summary.version !== 3) throw new Error("Page data needs regeneration with scripts/export_page.py");
  state.summary = summary;
  training.at = summary.progress.epochs.length - 1;

  drawShapePicker();
  $("shape-picker").addEventListener("click", (event) => {
    const b = event.target.closest("[data-case]");
    if (!b) return;
    state.caseId = b.dataset.case;
    render().catch(fail);
  });
  $("add-distance").addEventListener("click", () => {
    state.distance = !state.distance;
    render().catch(fail);
  });
  $("add-shape").addEventListener("click", () => {
    state.shape = !state.shape;
    render().catch(fail);
  });
  $("field-choice").addEventListener("click", (event) => {
    const b = event.target.closest("[data-field]");
    if (!b) return;
    state.field = Number(b.dataset.field);
    render().catch(fail);
  });
  $("training-view").addEventListener("click", (event) => {
    const b = event.target.closest("[data-view]");
    if (!b) return;
    training.view = b.dataset.view;
    drawTraining().catch(fail);
  });
  $("training-scrub").max = summary.progress.epochs.length - 1;
  $("training-scrub").addEventListener("input", (event) => setTraining(Number(event.target.value)));
  $("training-play").addEventListener("click", playTraining);
  // The replay starts the first time the training slide is reached and pauses whenever it is left.
  document.addEventListener("framechange", (event) => {
    if (event.detail.id !== "watch") stopTraining();
    else if (!training.played) playTraining();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stopTraining();
  });
  connectCompare();
  evidence();
  const [graph] = await Promise.all([createGraphExplorer({ names, colours, influenceMean: summary.influence_distance }), drawIntroduction(), render(), drawTraining()]);
  await drawInputs(graph);
  window.flowgnnReady = true;
  document.addEventListener("slidescale", () => {
    drawIntroduction().catch(fail);
    render().catch(fail);
    drawTraining().catch(fail);
    evidence();
    drawInputs(graph).catch(fail);
  });
}
main().catch(fail);
