"""Exercise the exported page in Chromium and compare what it shows with the exported data.

Run `python scripts/check_page.py`. Screenshots go to the ignored build/ directory.
Numerical export checks live in tests/test_page_data.py; these checks exercise the controls
and recompute every displayed number independently, in Python, from docs/data/.
"""

import json
from collections import deque

import numpy as np
from playwright.sync_api import sync_playwright

from page_browser import ROOT, open_page, set_slider

DATA = ROOT / "docs/data"
SECTIONS = ["start", "task", "message", "reach", "inputs", "watch", "found", "where", "influence", "see", "sources"]
NAMES = {"M0": "mesh only", "M1": "mesh + distance", "M2": "mesh + shape", "M3": "mesh + distance + shape"}


def expected_reach(mesh, start, rounds):
    """Points within `rounds` links of `start`, by breadth-first search over the full mesh."""
    adjacency = [[] for _ in range(len(mesh["pos"]) // 2)]
    for a, b in zip(mesh["edges"][::2], mesh["edges"][1::2]):
        adjacency[a].append(b)
        adjacency[b].append(a)
    distance = {start: 0}
    pending = deque([start])
    while pending:
        node = pending.popleft()
        if distance[node] == rounds:
            continue
        for neighbour in adjacency[node]:
            if neighbour not in distance:
                distance[neighbour] = distance[node] + 1
                pending.append(neighbour)
    return len(distance)


def trace(graph):
    """The recorded M3 trace as arrays: states [round, point, component] and messages [round, link, component]."""
    info = graph["traces"]["M3"]
    rounds, width, points, links = info["layers"], info["hidden"], len(graph["targets"]), len(info["sources"])
    values = np.fromfile(DATA / info["file"], "<f4")
    parts = np.split(values, np.cumsum([rounds * points * width] * 3))
    shape = (rounds, points, width)
    return {"before": parts[0].reshape(shape), "incoming": parts[1].reshape(shape),
            "after": parts[2].reshape(shape), "messages": parts[3].reshape(rounds, links, width)}


def stop(page, button):
    """Pause a replay that started on arrival, so each check controls the state itself."""
    if page.locator(f"#{button}").inner_text() == "Pause":
        page.locator(f"#{button}").click()


def show(page, section):
    page.locator(f"#{section}").evaluate("e => e.scrollIntoView({behavior:'instant', block:'start'})")
    page.wait_for_timeout(250)


def check_messages(page, graph):
    """Play starts at once; each step's numbers on the drawing equal the recorded trace, the running total ends
    at the recorded sum, and every step changes the drawing."""
    show(page, "message")
    stop(page, "message-play")
    t, info = trace(graph), graph["traces"]["M3"]
    centre = graph["targets"].index(graph["centre"])
    incoming = [i for i, d in enumerate(info["destinations"]) if d == graph["centre"]]
    drawing = page.locator("#message-mesh")
    component = int(drawing.get_attribute("data-component"))
    spread = t["after"][:, centre, :].max(0) - t["after"][:, centre, :].min(0)
    assert component == int(np.argmax(spread)), "shown component is not the one that changes most"
    while page.locator("#message-prev").is_enabled():
        page.locator("#message-prev").click()
    # The first step follows within 2 s of pressing Play (one send step), not after a fixed delay.
    page.locator("#message-play").click()
    page.wait_for_function("document.querySelector('#message-mesh').dataset.phase === '1'", timeout=2500)
    stop(page, "message-play")
    page.locator("#message-prev").click()
    value = lambda text: float(text.replace("Σ", "").replace("−", "-"))
    previous = None
    for _ in range(3 * info["layers"]):
        r, phase = int(drawing.get_attribute("data-round")), int(drawing.get_attribute("data-phase"))
        sent = t["messages"][r, incoming, component]
        if phase < 2:
            shown = [value(v) for v in page.locator("#message-mesh .tag text").all_text_contents()]
            np.testing.assert_allclose(shown, sent, atol=0.0051, rtol=0)
        else:
            assert abs(value(page.locator("#centre-text").text_content()) - t["after"][r, centre, component]) < 0.0051
        if phase == 1 and r in (0, info["layers"] - 1):  # watch the total build up to the recorded sum
            page.wait_for_function("document.querySelector('#message-mesh').dataset.sum !== undefined")
            assert abs(value(page.locator("#centre-text").text_content()) - t["incoming"][r, centre, component]) < 0.0051
        assert abs(sent.sum() - t["incoming"][r, centre, component]) < 1e-4
        # The reach view: after each completed round, exactly the points within that many links.
        done = r + 1 if phase == 2 else r
        mesh_view = page.locator("#round-mesh")
        assert int(mesh_view.get_attribute("data-rounds")) == done
        assert int(mesh_view.get_attribute("data-count")) == (expected_reach(graph, graph["centre"], done) if done else 0)
        inside = "is inside" in page.locator("#round-readout").inner_text()
        assert inside == (done >= graph["obstacle_links"]), (done, inside)
        html = drawing.inner_html()
        assert html != previous, "a step left the drawing unchanged"
        previous = html
        if not page.locator("#message-next").is_enabled():
            break
        page.locator("#message-next").click()


def check_reach(page, mesh):
    """The slide opens on 12 rounds; every stop shows the breadth-first count on the airfoil's mesh; clicking the
    mesh picks a new start point."""
    show(page, "reach")
    canvas = page.locator("#reach-mesh")
    assert canvas.get_attribute("data-round") == "12"
    for rounds in [4, 8, 12, 16, 32]:
        page.locator(f'#reach-key [data-round="{rounds}"]').click()
        node = int(canvas.get_attribute("data-node"))
        assert int(canvas.get_attribute("data-count")) == expected_reach(mesh, node, rounds)
        away = mesh["obstacle_hops"][node]
        readout = page.locator("#reach-readout").inner_text()
        assert f"After {rounds} rounds" in readout and f"The airfoil is {away} links away" in readout, readout
    old = canvas.get_attribute("data-node")
    canvas.click(position={"x": 30, "y": 30})
    assert canvas.get_attribute("data-node") != old


def check_influence(page, mesh, summary):
    """On arrival the slide steps from M0 to M3; each network's map, readout and bar show the exported mean
    distance of its influence distribution, and the note gives the mean over every wake point and seed."""
    show(page, "influence")
    page.wait_for_function("document.querySelector('#influence-mesh').dataset.network === 'M3'", timeout=10000)
    note = page.locator("#influence-mean").inner_text()
    for model, (_, mean, _) in summary["influence_distance"].items():
        assert f"{model} {mean:.1f}" in note, note
    for model in NAMES:
        assert abs(sum(mesh["influence"][model]["share"]) - 1) < 1e-2  # a distribution over the receptive field
        page.locator(f'#influence-view [data-network="{model}"]').click()
        assert page.locator("#influence-mesh").get_attribute("data-network") == model
        fits(page, "influence")
        readout = page.locator("#influence-readout").inner_text()
        assert f"mean distance of the influence is {mesh['influence'][model]['mean_links']:.1f} links" in readout, readout
        bar = page.locator(f'#influence-bars [data-model="{model}"]')
        assert "current" in bar.get_attribute("class") and bar.locator("> b").inner_text() == f"{mesh['influence'][model]['mean_links']:.1f}"


def check_inputs(page, summary):
    """Toggles select the four networks; the score strip equals the three-seed means for the shape."""
    show(page, "see")
    maps = page.locator("#maps")
    for record in summary["cases"]:
        page.locator(f'#shape-picker [data-case="{record["id"]}"]').click()
        page.wait_for_function("id => document.querySelector('#maps').dataset.case === id", arg=record["id"])
        for field in (0, 2):
            page.locator(f'#field-choice [data-field="{field}"]').click()
            page.wait_for_function("f => document.querySelector('#maps').dataset.field === String(f)", arg=field)
            metric = "pressure" if field == 2 else "velocity"
            definition = page.locator("#score-definition").inner_text()
            assert ("‖pGNN − pSU2‖ / ‖pSU2‖" in definition) == (field == 2), definition
            for model in NAMES:
                expected = 100 * np.mean([r["all"][metric] for r in record["errors"][model]])
                shown = page.locator(f'#score-bars [data-model="{model}"] > b').inner_text()
                assert shown == f"{expected:.1f}%", (record["id"], model, shown, expected)
            fits(page, "see")
    page.locator('#field-choice [data-field="0"]').click()
    for distance, shape, model in [(1, 0, "M1"), (1, 1, "M3"), (0, 1, "M2"), (0, 0, "M0")]:
        for button, wanted in (("add-distance", distance), ("add-shape", shape)):
            if (page.locator(f"#{button}").get_attribute("aria-pressed") == "true") != bool(wanted):
                page.locator(f"#{button}").click()
        page.wait_for_function("m => document.querySelector('#maps').dataset.model === m", arg=model)
        assert page.locator("#prediction-title").inner_text() == f"GNN · {model}: {NAMES[model]}"
        assert "current" in page.locator(f'#score-bars [data-model="{model}"]').get_attribute("class")
    assert maps.get_attribute("data-model") == "M0"


def check_training(page, summary):
    """Recorded snapshots agree with their epoch, both error labels and the chart cursor."""
    show(page, "watch")
    stop(page, "training-play")
    progress = summary["progress"]
    last = len(progress["epochs"]) - 1
    for index in [*range(0, last, 10), last]:
        set_slider(page, "#training-scrub", index)
        epoch = progress["epochs"][index]
        page.wait_for_function("e => document.querySelector('#training-chart').dataset.cursor === String(e)", arg=epoch)
        assert page.locator("#training-epoch").inner_text() == f"Epoch {epoch} of {summary['epochs']}"
        for model in ("M0", "M3"):
            error = 100 * progress["runs"][model]["velocity"][index]
            assert page.locator(f"#training-error-{model}").inner_text() == f"εu = {error:.1f}%"
        fits(page, "watch")
    # The epoch after which each error never again reaches 50%.
    settled = lambda m: progress["epochs"][max(i for i, v in enumerate(progress["runs"][m]["velocity"]) if v >= 0.5) + 1]
    labels = page.locator("#training-chart").text_content()
    for model in ("M3", "M0"):
        assert f"{model} below 50% from epoch {settled(model)}" in labels, labels
    assert page.locator("#training-finding").inner_text() == \
        f"M3 stays below εu = 50% from epoch {settled('M3')}, M0 only from epoch {settled('M0')}."
    page.locator('#training-view [data-view="error"]').click()
    assert page.locator('#training-view [data-view="error"]').get_attribute("aria-pressed") == "true"


def check_results(page, summary):
    """Each bar's height is the mean over three trained networks of their mean velocity error on that shape type."""
    show(page, "found")
    for model in NAMES:
        for shape in ("triangle", "naca"):
            cases = [c for c in summary["cases"] if c["split"] == "test" and c["shape"] == shape]
            seeds = range(len(cases[0]["errors"][model]))
            per_seed = [np.mean([c["errors"][model][k]["all"]["velocity"] for c in cases]) for k in seeds]
            bar = page.locator(f'#results-chart rect[data-model="{model}"][data-shape="{shape}"]')
            assert abs(float(bar.get_attribute("data-value")) - np.mean(per_seed)) < 1e-9


def check_inputs_rows(page, graph):
    """The input rows show the exported example (itself checked against the dataset in tests/)."""
    show(page, "inputs")
    example = graph["example_inputs"]
    fmt = lambda v: f"{v:.2f}".replace("-", "−")
    mesh = page.locator("#row-mesh dd").all_inner_texts()
    assert mesh == [fmt(example["x"]), fmt(example["y"]), example["type"]], mesh
    both = page.locator("#row-both dd").all_inner_texts()
    expected = mesh + [fmt(example["distance"]), f"{fmt(example['direction'][0])}, {fmt(example['direction'][1])}"]
    expected += [fmt(v) for v in example["shape"]]
    assert both == expected, both
    assert page.locator("#row-both .input.added dd").count() == 8


def check_compare(page, summary):
    """The opening line states the mean error of M0 and M2 on the test triangles, the tag M3's mean error on the
    shown triangle; the divider moves with the keyboard and reveals more of the network's picture."""
    show(page, "start")
    triangles = [c for c in summary["cases"] if c["split"] == "test" and c["shape"] == "triangle"]
    group = lambda m: 100 * np.mean([np.mean([c["errors"][m][k]["all"]["velocity"] for c in triangles]) for k in range(3)])
    lead = page.locator("#hero-lead").inner_text()
    assert f"from {group('M0'):.1f}% to {group('M2'):.1f}%" in lead, lead
    record = next(c for c in triangles if c["id"] == "triangle_05")
    error = 100 * np.mean([r["all"]["velocity"] for r in record["errors"]["M3"]])
    assert page.locator("#hero-tag").inner_text().endswith(f"· εu = {error:.0f}%")
    divider = page.locator("#divider")
    divider.focus()
    page.keyboard.press("ArrowLeft")
    assert divider.get_attribute("aria-valuenow") == "45"
    assert "45%" in page.locator("#hero-prediction").evaluate("e => e.style.clipPath")


def check_navigation(page):
    """Numbered links and unfocused arrow keys select the same pages."""
    page.locator('.rail a[href="#task"]').click()
    page.wait_for_function("document.querySelector('.rail a[aria-current]').hash === '#task'")
    page.evaluate("document.activeElement.blur()")
    page.keyboard.press("ArrowRight")
    page.wait_for_function("document.querySelector('.rail a[aria-current]').hash === '#message'")


def check_visibility(page):
    """Hiding the tab pauses every replay (the handler is exercised; Chromium keeps tabs visible)."""
    for section, button in [("message", "message-play"), ("watch", "training-play")]:
        show(page, section)
        if page.locator(f"#{button}").inner_text() == "Play":
            page.locator(f"#{button}").click()
        page.evaluate("""() => {
            Object.defineProperty(document, 'hidden', {configurable: true, get: () => true});
            document.dispatchEvent(new Event('visibilitychange'));
        }""")
        assert page.locator(f"#{button}").inner_text() == "Play"
        page.evaluate("delete document.hidden")


def check_interpolation(page):
    """A linear physical field must survive interpolation at every covered pixel centre."""
    values = page.evaluate("""async () => {
      const {rasterGeometry,VIEW}=await import('./fields.js');
      const data={pos:new Float32Array([-2,-1,4,-1,-2,2]),tri:new Uint32Array([0,1,2])};
      const width=50,height=30,r=rasterGeometry(data,width,height);
      const nodal=[4,16,-5]; // 2x - 3y + 5 at the three nodes.
      const samples=[];
      for(let i=0;i<width*height;i++){
        if(r.a[i]<0)continue;
        const x=VIEW[0]+((i%width)+.5)/width*(VIEW[2]-VIEW[0]);
        const y=VIEW[3]-(Math.floor(i/width)+.5)/height*(VIEW[3]-VIEW[1]);
        samples.push([x,y,(1-r.wb[i]-r.wc[i])*nodal[r.a[i]]+r.wb[i]*nodal[r.b[i]]+r.wc[i]*nodal[r.c[i]]]);
      }
      return samples;
    }""")
    assert len(values) > 100
    for x, y, value in values:
        assert abs(value - (2 * x - 3 * y + 5)) < 2e-6


def fits(page, section):
    """No visible part of the slide's content reaches past the slide, which would be cut off."""
    outside = page.locator(f"#{section} .slide").evaluate("""s => {
        const q = s.getBoundingClientRect();
        return [...s.querySelectorAll('*')].filter(e => { const r = e.getBoundingClientRect();
          return r.width > 0 && r.height > 0 && (r.top < q.top - 1 || r.left < q.left - 1 ||
            r.bottom > q.bottom + 1 || r.right > q.right + 1); })
          .map(e => e.tagName + (e.id ? '#' + e.id : '')); }""")
    assert not outside, f"{section}: content outside its slide: {outside[:5]}"


def check_layout(page, width, height, output):
    """Every page is one window tall and its slide fits inside it; saves a screenshot of each page to build/page-check/."""
    for section in SECTIONS:
        show(page, section)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert abs(page.locator(f"#{section}").bounding_box()["height"] - height) < 2
        slide = page.locator(f"#{section} .slide").bounding_box()
        assert slide["y"] >= -1 and slide["y"] + slide["height"] <= height + 1
        fits(page, section)
        page.screenshot(path=output / f"{width}-{section}.png")


def main():
    output = ROOT / "build" / "page-check"
    output.mkdir(parents=True, exist_ok=True)
    graph = json.loads((DATA / "graph.json").read_text())
    mesh = json.loads((DATA / "reach.json").read_text())
    summary = json.loads((DATA / "summary.json").read_text())
    with sync_playwright() as engine:
        browser = engine.chromium.launch(timeout=15000)
        try:
            # Two common screen sizes, and a browser window shorter than 16:9 (toolbars take height).
            for width, height in [(1366, 768), (1920, 1080), (1920, 950)]:
                page = open_page(browser, width, height)
                check_interpolation(page)
                check_navigation(page)
                check_compare(page, summary)
                check_inputs_rows(page, graph)
                check_messages(page, graph)
                check_reach(page, mesh)
                check_influence(page, mesh, summary)
                check_inputs(page, summary)
                check_training(page, summary)
                check_results(page, summary)
                check_visibility(page)
                check_layout(page, width, height, output)
                assert not page.errors, page.errors
                page.close()
                print(f"ok  controls, numbers and layout at {width} x {height}")
            page = open_page(browser, missing="docs/data/summary.json")
            page.locator("#load-error").wait_for(state="visible")
            assert "HTTP 404" in page.locator("#load-error").inner_text()
            assert page.locator("button:enabled").count() == 0
            page.close()
            print("ok  visible missing-data error")
        finally:
            browser.close()


if __name__ == "__main__":
    main()
