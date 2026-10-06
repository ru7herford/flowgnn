// Draw recorded scalar fields with shared colour bounds, inside a chosen window of the channel.
// A window is [x0, y0, x1, y1] in the flow's own units (the obstacle is about 1 long).
export const VIEW = [-2.7, -1.65, 9.4, 1.65];
const signedStops = ["#2166ac", "#92c5de", "#FFFFFF", "#f4a582", "#b2182b"];
const errorStops = ["#FFFFFF", "#fbe6c4", "#e7ac61", "#B5520B", "#71320a"];

/** Interpolate a bounded colour; clipping never changes the recorded numeric value. */
function colour(stops, fraction) {
  const q = Math.max(0, Math.min(1, fraction)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(q)),
    f = q - i;
  const rgb = stops
    .slice(i, i + 2)
    .map((h) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16)));
  return `rgb(${rgb[0].map((v, k) => Math.round(v + (rgb[1][k] - v) * f)).join(",")})`;
}

/** Return the colour function for a field or its absolute error. */
export function fieldColour(record, field, error = false) {
  const free = [1, 0, 0][field];
  const extent = Math.max(
    ...record.ranges.reference[field].map((v) => Math.abs(v - free)),
    1e-6,
  );
  const bounds = error
    ? [0, record.ranges.error[field]]
    : [free - extent, free + extent];
  const map = (value) =>
    colour(
      error ? errorStops : signedStops,
      (value - bounds[0]) / (bounds[1] - bounds[0]),
    );
  map.bounds = bounds;
  return map;
}

/** Colour bars for a field ("field"), its error ("error") or both, with the bounds every map shares. */
export function scales(record, field, show = "both") {
  const free = [1, 0, 0][field],
    extent = Math.max(
      ...record.ranges.reference[field].map((v) => Math.abs(v - free)),
    );
  const bar = (stops, a, b, name) =>
    `<span class="scale"><span>${name}</span> <span>${a}</span><i class="colourbar" style="background:linear-gradient(90deg,${stops.join(",")})"></i><span>${b}</span></span>`;
  const fieldBar = bar(
    signedStops,
    (free - extent).toFixed(2).replace("-", "−"),
    (free + extent).toFixed(2).replace("-", "−"),
    ["u (U∞ = 1)", "v", "p"][field],
  );
  const q = ["u", "v", "p"][field];
  const errorBar = bar(errorStops, "0", `≥ ${record.ranges.error[field].toFixed(2)}`, `|${q}<sub>GNN</sub> − ${q}<sub>SU2</sub>|`);
  return { both: fieldBar + errorBar, field: fieldBar, error: errorBar }[show];
}

const rasters = new WeakMap();

/** Cache pixel-to-triangle weights; only values change during model and epoch selection. */
export function rasterGeometry(data, width, height, view = VIEW) {
  let sizes = rasters.get(data);
  if (!sizes) {
    sizes = new Map();
    rasters.set(data, sizes);
  }
  const key = `${width}:${height}:${view.join()}`;
  if (sizes.has(key)) return sizes.get(key);
  const count = width * height;
  const raster = {
    a: new Int32Array(count).fill(-1),
    b: new Int32Array(count),
    c: new Int32Array(count),
    wb: new Float32Array(count),
    wc: new Float32Array(count),
  };
  const X = (i) => ((data.pos[2 * i] - view[0]) / (view[2] - view[0])) * width;
  const Y = (i) =>
    ((view[3] - data.pos[2 * i + 1]) / (view[3] - view[1])) * height;
  for (let t = 0; t < data.tri.length; t += 3) {
    const [a, b, c] = data.tri.subarray(t, t + 3);
    const ax = X(a),
      ay = Y(a),
      bx = X(b) - ax,
      by = Y(b) - ay,
      cx = X(c) - ax,
      cy = Y(c) - ay;
    const determinant = bx * cy - by * cx;
    if (Math.abs(determinant) < 1e-12) continue;
    const left = Math.max(0, Math.floor(Math.min(ax, ax + bx, ax + cx)));
    const right = Math.min(
      width - 1,
      Math.ceil(Math.max(ax, ax + bx, ax + cx)),
    );
    const top = Math.max(0, Math.floor(Math.min(ay, ay + by, ay + cy)));
    const bottom = Math.min(
      height - 1,
      Math.ceil(Math.max(ay, ay + by, ay + cy)),
    );
    for (let y = top; y <= bottom; y++)
      for (let x = left; x <= right; x++) {
        const px = x + 0.5 - ax,
          py = y + 0.5 - ay;
        const wb = (px * cy - py * cx) / determinant,
          wc = (bx * py - by * px) / determinant;
        if (wb < -1e-7 || wc < -1e-7 || wb + wc > 1 + 1e-7) continue;
        const i = y * width + x;
        raster.a[i] = a;
        raster.b[i] = b;
        raster.c[i] = c;
        raster.wb[i] = wb;
        raster.wc[i] = wc;
      }
  }
  sizes.set(key, raster);
  return raster;
}

/** Fill a canvas with recorded nodal values, linearly blended inside each mesh triangle. */
export function drawField(canvas, record, data, values, mapColour, view = VIEW) {
  const width = Math.max(100, canvas.clientWidth),
    dpr = Math.min(devicePixelRatio || 1, 2);
  const [x0, y0, x1, y1] = view;
  const aspect = (x1 - x0) / (y1 - y0);
  canvas.style.aspectRatio = aspect;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(canvas.width / aspect);
  const ctx = canvas.getContext("2d");
  const X = (x) => ((x - x0) / (x1 - x0)) * canvas.width,
    Y = (y) => ((y1 - y) / (y1 - y0)) * canvas.height;
  ctx.fillStyle = "#FFFFFF";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const raster = rasterGeometry(data, canvas.width, canvas.height, view);
  const image = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const [low, high] = mapColour.bounds;
  const palette = Array.from({ length: 1024 }, (_, i) =>
    mapColour(low + ((high - low) * i) / 1023)
      .match(/\d+/g)
      .map(Number),
  );
  for (let pixel = 0; pixel < raster.a.length; pixel++) {
    const a = raster.a[pixel];
    if (a < 0) continue;
    const b = raster.b[pixel],
      c = raster.c[pixel],
      wb = raster.wb[pixel],
      wc = raster.wc[pixel];
    const value = (1 - wb - wc) * values[a] + wb * values[b] + wc * values[c];
    const at = Math.round(
      Math.max(0, Math.min(1, (value - low) / (high - low))) * 1023,
    );
    image.data.set(palette[at], 4 * pixel);
  }
  ctx.putImageData(image, 0, 0);
  ctx.beginPath();
  record.outline.forEach(([x, y], i) =>
    i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)),
  );
  ctx.closePath();
  ctx.fillStyle = "#152B48";
  ctx.fill();
  ctx.strokeStyle = "#FFFFFF";
  ctx.lineWidth = dpr;
  ctx.stroke();
}
