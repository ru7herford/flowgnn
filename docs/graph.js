// One round, How far and Influence: the recorded rounds of message passing around one point with its reach
// after each round, the points whose information reaches a point in 12 rounds, and each network's influence distribution.
// Every value drawn comes from docs/data/graph.json, the recorded trace of the trained M3 network, and
// docs/data/reach.json (a test airfoil's mesh and each network's influence distribution at one wake point).
const $ = (id) => document.getElementById(id);
const number = (value) => value.toFixed(2).replace("-", "−");
const STATE_STOPS = ["#2166ac", "#92c5de", "#ffffff", "#f4a582", "#b2182b"];
// Reach is coloured by how many rounds it took, light (near) to dark (far): viridis steps, so even the
// first band stands out on white and each step reads as one further along.
const BANDS = [
  { upTo: 4, colour: "#7ad151" },
  { upTo: 8, colour: "#22a884" },
  { upTo: 12, colour: "#2a788e" },
  { upTo: 16, colour: "#414487" },
  { upTo: 32, colour: "#440154" },
];
const REACH_VIEW = [-2.2, -2.0, 8.8, 2.0];
// Influence relative to the largest point, from 1e-3 to 1, light to dark, on a log scale.
const INFLUENCE_STOPS = ["#d3ecee", "#063845"];
// Playback time of each step of a round: send, add up (six messages arrive one after another), update. The first
// three rounds play every step; after that each round shows only the reach, ROUND_MS each.
const PHASE_MS = [1600, 3400, 1800];
const SLOW_ROUNDS = 3;
const ROUND_MS = 1000;
const STEP_TEXT = [
  "Send: each neighbour sends a message to the ringed point.",
  "Add up: the ringed point adds the six messages.",
  "Update: an MLP (a small neural network) computes the new value from the old value and the sum. Each round has its own MLPs.",
];
const PULSE_S = 0.9; // travel time of one message along its link
const PULSE_GAP_S = 0.35; // messages arrive one after another, so the running total can be read

/** Breadth-first graph distance (in links) from one node to every other; -1 where unreachable. */
function distances(neighbours, start) {
  const hops = new Int32Array(neighbours.length).fill(-1);
  const queue = [start];
  hops[start] = 0;
  for (let head = 0; head < queue.length; head++) {
    const node = queue[head];
    for (const next of neighbours[node])
      if (hops[next] < 0) {
        hops[next] = hops[node] + 1;
        queue.push(next);
      }
  }
  return hops;
}

async function load(path, json = false) {
  const response = await fetch(`data/${path}`);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return json ? response.json() : response.arrayBuffer();
}

/** Neighbour lists from an exported edge list, where each link is stored once. */
function neighboursOf(mesh) {
  const neighbours = Array.from({ length: mesh.pos.length / 2 }, () => []);
  for (let i = 0; i < mesh.edges.length; i += 2) {
    neighbours[mesh.edges[i]].push(mesh.edges[i + 1]);
    neighbours[mesh.edges[i + 1]].push(mesh.edges[i]);
  }
  return neighbours;
}

/** Split a recorded trace into per-round states and messages, in the order the exporter wrote them. */
function decodeTrace(buffer, info, targetCount) {
  let offset = 0;
  const states = info.layers * targetCount * info.hidden;
  const messages = info.layers * info.sources.length * info.hidden;
  const take = (count) => {
    const values = new Float32Array(buffer, offset, count);
    offset += 4 * count;
    return values;
  };
  const trace = { before: take(states), incoming: take(states), after: take(states), messages: take(messages) };
  if (offset !== buffer.byteLength) throw new Error("Unexpected message-trace layout");
  return trace;
}

/** Diverging colour for a signed value: blue negative, white zero, red positive. A square-root scale
 *  keeps small values visible while one fixed extent keeps colours comparable across rounds. */
function divergingColour(value, extent) {
  const t = Math.sign(value) * Math.sqrt(Math.min(1, Math.abs(value) / extent));
  const q = ((t + 1) / 2) * (STATE_STOPS.length - 1);
  const i = Math.min(STATE_STOPS.length - 2, Math.floor(q)),
    f = q - i;
  const [a, b] = STATE_STOPS.slice(i, i + 2).map((h) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16)));
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(",")})`;
}

const bandOf = (d) => BANDS.find((b) => d <= b.upTo);

/** Mix two hex colours: t = 0 gives the first, 1 the second. */
function mix([from, to], t) {
  const [a, b] = [from, to].map((h) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16)));
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * Math.max(0, Math.min(1, t)))).join(",")})`;
}

/** Load the graph data and connect the One round, How far and Influence slides, with the network names, colours
 *  and the mean influence distances over every wake point and seed (summary.json).
 *  Returns the traced graph for the inputs page. */
export async function createGraphExplorer({ names, colours, influenceMean }) {
  const [graph, mesh] = await Promise.all([load("graph.json", true), load("reach.json", true)]);
  const neighbours = neighboursOf(graph);
  const info = graph.traces.M3;
  const trace = decodeTrace(await load(info.file), info, graph.targets.length);
  const width = info.hidden,
    targets = graph.targets.length,
    centre = graph.centre,
    centreSlot = graph.targets.indexOf(centre);

  // ---- One round: one recorded round of message passing -------------------------------------------

  // Of the 64 numbers each point carries, show the one whose value at the centre changes most over the
  // twelve rounds. The arithmetic shown is exact for it.
  const component = (() => {
    let best = 0,
      bestSpread = -1;
    for (let k = 0; k < width; k++) {
      const values = Array.from({ length: info.layers }, (_, r) => trace.after[(r * targets + centreSlot) * width + k]);
      const spread = Math.max(...values) - Math.min(...values);
      if (spread > bestSpread) [bestSpread, best] = [spread, k];
    }
    return best;
  })();
  const state = (kind, round, slot) => trace[kind][(round * targets + slot) * width + component];
  const messageValue = (round, i) => trace.messages[(round * info.sources.length + i) * width + component];
  const incoming = info.destinations.map((node, i) => (node === centre ? i : -1)).filter((i) => i >= 0);
  const stateExtent = Math.max(
    ...["before", "after"].flatMap((kind) =>
      Array.from({ length: info.layers * targets }, (_, i) => Math.abs(trace[kind][i * width + component])),
    ),
    1e-9,
  );
  const messageExtent = Math.max(
    ...Array.from({ length: info.layers }, (_, r) => incoming.map((i) => Math.abs(messageValue(r, i)))).flat(),
    1e-9,
  );

  // Schematic layout: the centre in the middle and its neighbours on a circle, in the same order around it
  // as in the mesh, so every number has room. The links drawn are the real ones.
  const [cx, cy, R] = [430, 300, 225];
  const place = new Map([[centre, [cx, cy]]]);
  for (const i of incoming) {
    const s = info.sources[i];
    const angle = Math.atan2(graph.pos[2 * s + 1] - graph.pos[2 * centre + 1], graph.pos[2 * s] - graph.pos[2 * centre]);
    place.set(s, [cx + R * Math.cos(angle), cy - R * Math.sin(angle)]);
  }
  const ring = incoming.map((i) => info.sources[i]);

  const message = { round: 0, phase: 0, timer: null, played: false, counting: [] };

  /** Draw the round's current step and start its animation: messages grow along their links (send), arrive
   *  one by one while the centre's total counts up (add up), or every point changes colour (update). */
  function drawMessages() {
    const { round, phase } = message;
    message.counting.forEach(clearTimeout);
    message.counting = [];
    const colourOf = (kind, slot) => divergingColour(state(kind, round, slot), stateExtent);
    let svg = "";
    // Links between neighbours (faint), then a short stub outward from each: their other links.
    for (const a of ring)
      for (const b of neighbours[a])
        if (a < b && ring.includes(b)) svg += `<line x1="${place.get(a)[0]}" y1="${place.get(a)[1]}" x2="${place.get(b)[0]}" y2="${place.get(b)[1]}" class="link"/>`;
    for (const s of ring) {
      const [x, y] = place.get(s);
      svg += `<line x1="${x}" y1="${y}" x2="${x + ((x - cx) / R) * 55}" y2="${y + ((y - cy) / R) * 55}" class="stub"/>`;
    }
    incoming.forEach((i, k) => {
      const [x, y] = place.get(info.sources[i]),
        v = messageValue(round, i),
        length = Math.hypot(cx - x, cy - y);
      const grow = phase === 0 ? `<animate attributeName="stroke-dashoffset" from="${length}" to="0" dur="0.7s" begin="indefinite" fill="freeze"/>` : "";
      svg += `<line x1="${x}" y1="${y}" x2="${cx}" y2="${cy}" stroke="${divergingColour(v, messageExtent)}" stroke-width="${4 + (8 * Math.abs(v)) / messageExtent}" stroke-dasharray="${length}" class="carrier">${grow}</line>`;
      if (phase === 1) {
        const motion = `dur="${PULSE_S}s" begin="indefinite" fill="freeze" calcMode="spline" keyTimes="0;1" keySplines="0.2 0 0 1"`;
        svg += `<circle r="9" cx="${x}" cy="${y}" class="pulse" data-delay="${k * PULSE_GAP_S}"><animate attributeName="cx" from="${x}" to="${cx}" ${motion}/><animate attributeName="cy" from="${y}" to="${cy}" ${motion}/></circle>`;
      }
    });
    // In the update step every point changes from its old colour to its new one.
    const fill = (slot) =>
      phase === 2
        ? `fill="${colourOf("after", slot)}"><animate attributeName="fill" from="${colourOf("before", slot)}" to="${colourOf("after", slot)}" dur="0.9s" begin="indefinite" fill="freeze"/></circle>`
        : `fill="${colourOf("before", slot)}"></circle>`;
    for (const s of ring) svg += `<circle cx="${place.get(s)[0]}" cy="${place.get(s)[1]}" r="34" class="point" ${fill(graph.targets.indexOf(s))}`;
    svg += `<circle cx="${cx}" cy="${cy}" r="58" class="centre" ${fill(centreSlot)}`;
    if (phase < 2)
      incoming.forEach((i, k) => {
        const [x, y] = place.get(info.sources[i]),
          tx = x + (cx - x) * 0.5,
          ty = y + (cy - y) * 0.5;
        svg += `<g class="tag" data-k="${k}"><rect x="${tx - 54}" y="${ty - 24}" width="108" height="48" rx="24"/><text x="${tx}" y="${ty + 10}">${number(messageValue(round, i))}</text></g>`;
      });
    const centreText = phase === 0 ? number(state("before", round, centreSlot)) : phase === 1 ? `Σ ${number(0)}` : number(state("after", round, centreSlot));
    svg += `<text x="${cx}" y="${cy + 12}" class="centre-text" id="centre-text">${centreText}</text>`;
    const drawing = $("message-mesh");
    drawing.innerHTML = svg;
    delete drawing.dataset.sum; // set again once every message of this step has been added
    drawing.querySelectorAll(".carrier animate, .point animate, .centre animate").forEach((a) => a.beginElement());
    // Add up: each message flies in; when it lands, its tag fades and the centre's total grows by it.
    let total = 0;
    drawing.querySelectorAll(".pulse").forEach((p, k) => {
      const delay = Number(p.dataset.delay) * 1000;
      message.counting.push(setTimeout(() => p.querySelectorAll("animate").forEach((a) => a.beginElement()), delay));
      message.counting.push(
        setTimeout(() => {
          total += messageValue(round, incoming[k]);
          $("centre-text").textContent = `Σ ${number(total)}`;
          drawing.querySelector(`.tag[data-k="${k}"]`).classList.add("counted");
          if (k === incoming.length - 1) drawing.dataset.sum = total;
        }, delay + PULSE_S * 1000),
      );
    });
    Object.assign(drawing.dataset, { round, phase, component });

    for (const step of $("message-loop").querySelectorAll(".step"))
      step.classList.toggle("current", Number(step.dataset.phase) === phase);
    $("message-step").textContent = STEP_TEXT[phase];
    $("message-round").textContent = `Round ${round + 1} of ${info.layers}`;
    drawRoundMesh(phase === 2 ? round + 1 : round); // the reach grows when a round's update is done
    $("message-prev").disabled = round === 0 && phase === 0;
    $("message-next").disabled = round === info.layers - 1 && phase === 2;
  }

  function pauseMessages() {
    clearTimeout(message.timer);
    message.timer = null;
    $("message-play").textContent = "Play";
  }

  /** Move one step; three steps make one round. */
  function stepMessages(direction) {
    const next = Math.max(0, Math.min(info.layers * 3 - 1, message.round * 3 + message.phase + direction));
    message.round = Math.floor(next / 3);
    message.phase = next % 3;
    drawMessages();
  }

  /** Play the current step's animation at once, then move on: every step of the first rounds, then one whole
   *  round at a time, until the last round ends. */
  function playMessages() {
    if (message.timer) return pauseMessages();
    if ($("message-next").disabled) [message.round, message.phase] = [0, 0];
    message.played = true;
    $("message-play").textContent = "Pause";
    drawMessages();
    const next = () => {
      const fast = message.round >= SLOW_ROUNDS - 1 && message.phase === 2;
      message.timer = setTimeout(() => {
        if ($("message-next").disabled) return pauseMessages();
        if (fast) [message.round, message.phase] = [message.round + 1, 2];
        else stepMessages(1);
        if (fast) drawMessages();
        next();
      }, fast ? ROUND_MS : PHASE_MS[message.phase]);
    };
    next();
  }

  // ---- How far and Influence: the airfoil's mesh, one wake point, and each network's influence ---------------

  const meshNeighbours = neighboursOf(mesh);
  const startHops = distances(meshNeighbours, mesh.start);
  // The How far slide opens on the networks' own 12 rounds; the key shows other counts, a click picks another point.
  const reach = { node: mesh.start, round: 12, hops: startHops };
  const influence = mesh.influence;
  // On arrival the influence slide steps through the four networks once, M0 to M3, then stays on M3.
  const influencePlay = { network: "M0", timer: null, played: false };

  /** Clear a canvas to a mesh in a view [x0, y0, x1, y1]; returns a function that draws one point, and one that
   *  draws the obstacle and a ring around the chosen point. */
  function meshCanvas(canvas, m, view) {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const [x0, y0, x1, y1] = view;
    canvas.style.aspectRatio = (x1 - x0) / (y1 - y0);
    canvas.width = Math.round(canvas.clientWidth * dpr);
    canvas.height = Math.round((canvas.width * (y1 - y0)) / (x1 - x0));
    const ctx = canvas.getContext("2d");
    const Xp = (x) => ((x - x0) / (x1 - x0)) * canvas.width,
      Yp = (y) => ((y1 - y) / (y1 - y0)) * canvas.height;
    const visible = (n) => m.pos[2 * n] >= x0 && m.pos[2 * n] <= x1 && m.pos[2 * n + 1] >= y0 && m.pos[2 * n + 1] <= y1;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#dbe6ee";
    ctx.lineWidth = 0.8 * dpr;
    ctx.beginPath();
    for (let i = 0; i < m.edges.length; i += 2) {
      const a = m.edges[i],
        b = m.edges[i + 1];
      if (!visible(a) && !visible(b)) continue;
      ctx.moveTo(Xp(m.pos[2 * a]), Yp(m.pos[2 * a + 1]));
      ctx.lineTo(Xp(m.pos[2 * b]), Yp(m.pos[2 * b + 1]));
    }
    ctx.stroke();
    const dot = (n, colour, radius, outline = null) => {
      if (!visible(n)) return;
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.arc(Xp(m.pos[2 * n]), Yp(m.pos[2 * n + 1]), radius * dpr, 0, 2 * Math.PI);
      ctx.fill();
      if (!outline) return;
      ctx.strokeStyle = outline;
      ctx.lineWidth = 2 * dpr;
      ctx.stroke();
    };
    /** The obstacle (outlined in `edge` when given) and a ring around the chosen point, drawn last. */
    const finish = (node, edge = null) => {
      ctx.beginPath();
      m.outline.forEach(([x, y], i) => (i ? ctx.lineTo(Xp(x), Yp(y)) : ctx.moveTo(Xp(x), Yp(y))));
      ctx.closePath();
      ctx.fillStyle = "#152b48";
      ctx.fill();
      if (edge) {
        ctx.strokeStyle = edge;
        ctx.lineWidth = 6 * dpr;
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(Xp(m.pos[2 * node]), Yp(m.pos[2 * node + 1]), 10 * dpr, 0, 2 * Math.PI);
      ctx.strokeStyle = "#152b48";
      ctx.lineWidth = 4 * dpr;
      ctx.stroke();
    };
    return { dot, finish };
  }

  /** Receptive field: every point within the current number of rounds of the chosen point, coloured by round. */
  function drawReach() {
    const canvas = $("reach-mesh"),
      { dot, finish } = meshCanvas(canvas, mesh, REACH_VIEW);
    let count = 0;
    for (let n = 0; n < meshNeighbours.length; n++) {
      const d = reach.hops[n];
      if (d < 0 || d > reach.round) continue;
      count++;
      dot(n, bandOf(d).colour, 4);
    }
    const away = mesh.obstacle_hops[reach.node],
      heard = away <= reach.round;
    finish(reach.node, heard ? bandOf(away).colour : null);
    Object.assign(canvas.dataset, { node: reach.node, round: reach.round, count });
    for (const chip of $("reach-key").querySelectorAll("[data-round]"))
      chip.setAttribute("aria-pressed", String(Number(chip.dataset.round) === reach.round));
    $("reach-readout").innerHTML =
      `<b>After ${reach.round} rounds</b>: information from ${count.toLocaleString("en-GB")} ${count === 1 ? "point" : "points"} reaches this point.<br>` +
      `The airfoil is ${away} links away${heard ? `: its information arrives at round ${away}` : ""}.`;
  }

  /** Influence: the influence distribution of one network's predicted velocity at the wake point. Grey: every
   *  point inside the 12-link receptive field. Beside it, each network's mean link distance as a bar. */
  function drawInfluence() {
    const network = influencePlay.network;
    const canvas = $("influence-mesh"),
      { dot, finish } = meshCanvas(canvas, mesh, REACH_VIEW);
    for (let n = 0; n < meshNeighbours.length; n++) if (startHops[n] >= 0 && startHops[n] <= 12) dot(n, "#c9d2da", 2.5);
    const { nodes, share, mean_links } = influence[network];
    const largest = Math.max(...share);
    const order = nodes.map((_, i) => i).sort((i, j) => share[i] - share[j]); // largest drawn last
    for (const i of order) {
      const relative = share[i] / largest;
      if (relative >= 1e-3) dot(nodes[i], mix(INFLUENCE_STOPS, 1 + Math.log10(relative) / 3), 4.5);
    }
    finish(mesh.start);
    canvas.dataset.network = network;
    for (const chip of $("influence-view").children) chip.setAttribute("aria-pressed", String(chip.dataset.network === network));
    const near = mean_links < 1.5 ? ": the point and its direct neighbours" : "";
    $("influence-readout").innerHTML =
      `<b>${network}: ${names[network]}</b>. At the ringed point, the mean distance of the influence is ` +
      `${mean_links.toFixed(1)} links${near}.<br>Grey points: inside the 12 links, but below 0.1% of the largest influence.`;
    const top = Math.max(...Object.values(influence).map((v) => v.mean_links));
    $("influence-bars").innerHTML = Object.entries(influence)
      .map(([s, v]) => `<div class="bar${s === network ? " current" : ""}" data-model="${s}"><span><b>${s}:</b> ${names[s]}</span>` +
        `<i style="width:${(100 * v.mean_links) / top}%"></i><b>${v.mean_links.toFixed(1)}</b></div>`)
      .join("");
  }

  function pauseInfluence() {
    clearTimeout(influencePlay.timer);
    influencePlay.timer = null;
  }

  /** Show M0, M1, M2 and M3 in turn, 1.8 s each, so the influence visibly shrinks towards the ringed point. */
  function playInfluence() {
    pauseInfluence();
    influencePlay.played = true;
    const order = Object.keys(influence);
    let k = 0;
    const show = () => {
      influencePlay.network = order[k];
      drawInfluence();
      if (++k < order.length) influencePlay.timer = setTimeout(show, 1800);
    };
    show();
  }

  function selectReach(node) {
    reach.node = node;
    reach.hops = distances(meshNeighbours, node);
    drawReach();
  }

  // ---- One round: the reach of the ringed point on the test triangle's mesh, after each round -------------

  // Influence of every point on the ringed point's state after each round, for the points within 12 links
  // (graph.rounds, computed from the saved M3 network). The view frames those points.
  const roundShare = new Map(graph.rounds.nodes.map((n, i) => [n, i]));
  const roundHops = distances(neighbours, centre);
  const ROUND_VIEW = (() => {
    const xs = [...graph.rounds.nodes.map((n) => graph.pos[2 * n]), ...graph.outline.map((p) => p[0])],
      ys = [...graph.rounds.nodes.map((n) => graph.pos[2 * n + 1]), ...graph.outline.map((p) => p[1])];
    const [mx, my] = [(Math.min(...xs) + Math.max(...xs)) / 2, (Math.min(...ys) + Math.max(...ys)) / 2];
    const half = 0.55 * Math.max(Math.max(...xs) - Math.min(...xs), (Math.max(...ys) - Math.min(...ys)) * 1.4);
    return [mx - half, my - half / 1.4, mx + half, my + half / 1.4]; // aspect 1.4 : 1, as the canvas
  })();

  /** The reach after `done` rounds: each point within `done` links coloured by its influence on the ringed
   *  point's state (log scale over four decades; below that, grey), the newest ring outlined. */
  function drawRoundMesh(done) {
    const canvas = $("round-mesh"),
      { dot, finish } = meshCanvas(canvas, graph, ROUND_VIEW);
    const share = done ? graph.rounds.share[done - 1] : null;
    const largest = share ? Math.max(...share) : 1;
    let count = 0;
    for (const [n, i] of roundShare) {
      if (!share || !(share[i] > 0)) continue;
      count++;
      const relative = share[i] / largest;
      dot(n, relative >= 1e-4 ? mix(INFLUENCE_STOPS, 1 + Math.log10(relative) / 4) : "#c9d2da", 5, roundHops[n] === done ? "#152b48" : null);
    }
    finish(centre);
    Object.assign(canvas.dataset, { rounds: done, count });
    $("round-readout").innerHTML = done === 0
      ? "Before round 1, the ringed point has only its own inputs."
      : `<b>After round ${done}</b>, the ringed point's state depends on ${count} points, up to ${done} ` +
        `${done === 1 ? "link" : "links"} away.` +
        (done >= graph.obstacle_links ? ` The obstacle, ${graph.obstacle_links} links away, is inside.` : "") +
        (done === info.layers ? "<br>12 links: everything this point will ever know." : "");
  }

  // ---- Controls ------------------------------------------------------------------------------------

  // The key doubles as the stop buttons: one swatch per band, the network's own 12 rounds marked.
  $("reach-key").innerHTML =
    `<div class="band-row">${BANDS.map((b, i) => `<button class="band" data-round="${b.upTo}" style="--c:${b.colour}" aria-label="Show ${b.upTo} rounds"><i></i>${i === 0 ? "1" : BANDS[i - 1].upTo + 1}–${b.upTo}</button>`).join("")}</div>` +
    `<div class="band-notes"><span style="flex:3">Rounds 1–12: the receptive field of these networks</span><span style="flex:2">Rounds 13–32: need a deeper network</span></div>`;
  $("reach-key").addEventListener("click", (event) => {
    const chip = event.target.closest("[data-round]");
    if (!chip) return;
    reach.round = Number(chip.dataset.round);
    drawReach();
  });
  $("reach-finding").textContent =
    "Each round sends one message along each link. After 12 rounds, a point has information from points up to " +
    `12 links away (its receptive field). This point in the wake, the slower flow behind the airfoil, is ` +
    `${mesh.obstacle_hops[mesh.start]} links from it.`;
  // Influence: one chip per network; the map and readout switch to that network's influence distribution.
  $("influence-view").innerHTML = Object.keys(influence)
    .map((s) => `<button class="chip" data-network="${s}"><i style="background:${colours[s]}"></i>${s}: ${names[s]}</button>`)
    .join("");
  $("influence-view").addEventListener("click", (event) => {
    const chip = event.target.closest("[data-network]");
    if (!chip) return;
    pauseInfluence();
    influencePlay.network = chip.dataset.network;
    drawInfluence();
  });
  // The same measure averaged over more wake points and every trained network (the README's numbers).
  $("influence-mean").textContent = "Mean over 3 wake points on 2 test shapes and 3 trained networks: " +
    Object.entries(influenceMean).map(([s, [, m]]) => `${s} ${m.toFixed(1)}`).join(", ") + " links.";
  $("influence-key").innerHTML =
    `<span>low</span><i class="colourbar" style="background:linear-gradient(90deg,${INFLUENCE_STOPS.join(",")})"></i><span>high</span>` +
    "<small>Influence, scaled to the largest value of each map.</small>";
  $("round-key").innerHTML =
    `<span>low</span><i class="colourbar" style="background:linear-gradient(90deg,${INFLUENCE_STOPS.join(",")})"></i><span>high</span>` +
    "<small>Influence on the ringed point (log scale); outlined: the newest ring.</small>";
  $("reach-mesh").addEventListener("pointerdown", (event) => {
    const box = event.target.getBoundingClientRect();
    const [x0, y0, x1, y1] = REACH_VIEW;
    const x = x0 + ((event.clientX - box.left) / box.width) * (x1 - x0),
      y = y1 - ((event.clientY - box.top) / box.height) * (y1 - y0);
    let nearest = 0,
      best = Infinity;
    for (let n = 0; n < meshNeighbours.length; n++) {
      const d = (mesh.pos[2 * n] - x) ** 2 + (mesh.pos[2 * n + 1] - y) ** 2;
      if (d < best) [best, nearest] = [d, n];
    }
    selectReach(nearest);
  });
  $("message-prev").addEventListener("click", () => {
    pauseMessages();
    stepMessages(-1);
  });
  $("message-next").addEventListener("click", () => {
    pauseMessages();
    stepMessages(1);
  });
  $("message-play").addEventListener("click", playMessages);

  // Each page's replay starts the first time the page is reached and pauses whenever it is left.
  document.addEventListener("framechange", (event) => {
    const id = event.detail.id;
    if (id !== "message") pauseMessages();
    if (id !== "influence") pauseInfluence();
    if (id === "message" && !message.played) playMessages();
    if (id === "influence" && !influencePlay.played) playInfluence();
  });
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) return;
    pauseMessages();
    pauseInfluence();
  });
  document.addEventListener("slidescale", () => {
    drawMessages();
    drawReach();
    drawInfluence();
  });
  drawMessages();
  drawReach();
  drawInfluence();
  return graph;
}
