"use strict";
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const wait = ms => new Promise(r => setTimeout(r, ms));

  /* ------------------------------------------------------------ top plate */
  const plate = $("#plate");
  const onScroll = () => plate.classList.toggle("is-solid", scrollY > 40);
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* ------------------------------------------------------------ viewfinder */
  const vf = $("#vf");
  const stage = $("#stage");
  const shots = $$("img", stage);
  const locksEl = $("#locks");
  const af = $("#af");
  const iris = $("#iris");
  const flashEl = $("#flash");
  const roll = $("#roll");
  const countEl = $("#count");
  const modeEl = $("#mode");
  const modeName = $("#modeName");
  const modelEl = $("#model");
  const pauseBtn = $("#pause");

  const SHOTS = [
    { mode: "FACE ×2", model: "RetinaFace · ArcFace", pos: [0.5, 0.3], subjects: [
      { box: [0.482, 0.185, 0.066, 0.105], tag: "Face · Person 12", score: "0.88" },
      { box: [0.3, 0.505, 0.058, 0.1], tag: "Unassigned", score: "name me", dashed: true, up: true }] },
    { mode: "CLIP", model: "CLIP ViT-B/32 · zero-shot", pos: [0.55, 0.62], subjects: [
      { box: [0.43, 0.42, 0.34, 0.45], tag: "“leopard”", score: "0.31" }] },
    { mode: "FACE", model: "RetinaFace · ArcFace", pos: [0.5, 0.35], subjects: [
      { box: [0.4, 0.24, 0.29, 0.46], tag: "Face · Person 07", score: "0.93" }] },
    { mode: "GPS", model: "Offline reverse-geocode", pos: [0.5, 0.25], subjects: [
      { box: [0.44, 0.07, 0.165, 0.49], tag: "48.8584 N · 2.2945 E", score: "Paris, FR" }] },
  ];

  const cols = () => (innerWidth <= 860 ? 7 : 11);
  const rows = () => (innerWidth <= 860 ? 5 : 7);
  function buildAF() {
    af.innerHTML = "";
    for (let i = 0; i < 11 * 7; i++) af.appendChild(document.createElement("span"));
  }
  buildAF();

  shots.forEach((img, i) => { img.style.objectPosition = `${SHOTS[i].pos[0] * 100}% ${SHOTS[i].pos[1] * 100}%`; });

  function subjectRect(img, shot, box) {
    const sr = stage.getBoundingClientRect();
    const vr = vf.getBoundingClientRect();
    const iw = img.naturalWidth || 4, ih = img.naturalHeight || 3;
    const s = Math.max(sr.width / iw, sr.height / ih);
    const dw = iw * s, dh = ih * s;
    const dx = (sr.width - dw) * shot.pos[0];
    const dy = (sr.height - dh) * shot.pos[1];
    return {
      x: sr.left - vr.left + dx + box[0] * dw,
      y: sr.top - vr.top + dy + box[1] * dh,
      w: box[2] * dw,
      h: box[3] * dh,
    };
  }

  function makeLock(sub) {
    const el = document.createElement("div");
    el.className = "lock" + (sub.dashed ? " is-dashed" : "") + (sub.up ? " tag-up" : "");
    el.innerHTML = `<b></b><span class="tag">${sub.tag}<small>${sub.score}</small></span>`;
    locksEl.appendChild(el);
    return el;
  }
  const place = (el, r) => Object.assign(el.style, { left: r.x + "px", top: r.y + "px", width: r.w + "px", height: r.h + "px" });

  // iris: hexagonal aperture with blade edges
  const ictx = iris.getContext("2d");
  function sizeIris() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    iris.width = vf.clientWidth * dpr; iris.height = vf.clientHeight * dpr;
    ictx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  function drawIris(k) { // k: 0 open, 1 closed
    const w = vf.clientWidth, h = vf.clientHeight;
    ictx.clearRect(0, 0, w, h);
    if (k <= 0) return;
    const cx = w / 2, cy = h / 2, R = Math.hypot(w, h) / 2 + 20;
    const r = R * (1 - k);
    const rot = k * 0.9;
    ictx.fillStyle = "#050505";
    ictx.beginPath();
    ictx.rect(0, 0, w, h);
    const pts = [];
    for (let i = 0; i < 6; i++) {
      const a = rot + (i * Math.PI) / 3;
      pts.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]);
    }
    ictx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 5; i >= 0; i--) ictx.lineTo(pts[i][0], pts[i][1]);
    ictx.closePath();
    ictx.fill("evenodd");
    ictx.strokeStyle = "rgba(244,244,242,0.16)";
    ictx.lineWidth = 1;
    for (let i = 0; i < 6; i++) {
      const [x1, y1] = pts[i], [x2, y2] = pts[(i + 1) % 6];
      const dx = x2 - x1, dy = y2 - y1;
      ictx.beginPath(); ictx.moveTo(x1, y1); ictx.lineTo(x1 + dx * 40, y1 + dy * 40); ictx.stroke();
    }
  }
  function animIris(from, to, ms) {
    return new Promise(res => {
      const t0 = performance.now();
      const step = now => {
        const p = Math.min(1, (now - t0) / ms);
        const e = to > from ? p * p : 1 - Math.pow(1 - p, 3);
        drawIris(from + (to - from) * e);
        p < 1 ? requestAnimationFrame(step) : res();
      };
      requestAnimationFrame(step);
    });
  }

  let playing = !reduce, inView = true, current = 0, count = 0, runId = 0;
  const gate = async id => { while ((!playing || !inView || document.hidden) && id === runId) await wait(200); return id === runId; };

  function setMode(shot, locked) {
    modeName.textContent = shot.mode;
    modelEl.textContent = shot.model;
    modeEl.classList.toggle("is-locked", !!locked);
  }

  function hunt(ms) {
    const spans = [...af.children].filter(s => s.offsetParent !== null);
    let alive = true;
    const tick = () => {
      if (!alive) return;
      spans.forEach(s => s.classList.remove("hot"));
      for (let i = 0; i < 3; i++) spans[(Math.random() * spans.length) | 0]?.classList.add("hot");
      setTimeout(tick, 90);
    };
    tick();
    return () => { alive = false; spans.forEach(s => s.classList.remove("hot")); };
  }

  function addToRoll(img) {
    const t = document.createElement("span");
    t.style.backgroundImage = `url("${img.currentSrc || img.src}")`;
    t.className = "new";
    roll.appendChild(t);
    while (roll.children.length > 6) roll.firstChild.remove();
    setTimeout(() => t.classList.remove("new"), 900);
    count++;
    countEl.textContent = String(count).padStart(2, "0");
  }

  async function shoot(id) {
    const shot = SHOTS[current], img = shots[current];
    if (!img.complete) await new Promise(r => { img.onload = img.onerror = r; });
    setMode(shot, false);
    af.classList.remove("is-quiet");
    locksEl.innerHTML = "";
    const locks = shot.subjects.map(makeLock);
    const vr = vf.getBoundingClientRect();
    locks.forEach((el, i) => {
      const r = subjectRect(img, shot, shot.subjects[i].box);
      const j = i ? -1 : 1;
      place(el, { x: r.x + j * vr.width * 0.08 - r.w * 0.2, y: r.y + vr.height * 0.06, w: r.w * 1.5, h: r.h * 1.4 });
    });
    const stop = hunt();
    if (!(await gate(id))) return stop();
    await wait(650);
    locks.forEach((el, i) => {
      const r = subjectRect(img, shot, shot.subjects[i].box);
      place(el, { x: r.x - r.w * 0.06, y: r.y - r.h * 0.04, w: r.w * 1.12, h: r.h * 1.08 });
    });
    await wait(260);
    locks.forEach((el, i) => place(el, subjectRect(img, shot, shot.subjects[i].box)));
    await wait(420);
    stop();
    af.classList.add("is-quiet");
    for (const el of locks) { el.classList.add("is-locked"); await wait(160); }
    setMode(shot, true);
    await wait(1700);
    if (!(await gate(id))) return;
    flashEl.classList.remove("go"); void flashEl.offsetWidth; flashEl.classList.add("go");
    addToRoll(img);
    await wait(520);
    await animIris(0, 1, 230);
    locksEl.innerHTML = "";
    img.classList.remove("is-on");
    current = (current + 1) % SHOTS.length;
    shots[current].classList.add("is-on");
    await wait(90);
    await animIris(1, 0, 360);
  }

  async function loop() {
    const id = ++runId;
    sizeIris();
    while (id === runId) {
      if (!(await gate(id))) return;
      await shoot(id);
    }
  }

  function showStill() {
    const shot = SHOTS[current], img = shots[current];
    locksEl.innerHTML = "";
    af.classList.add("is-quiet");
    const locks = shot.subjects.map(makeLock);
    const go = () => locks.forEach((el, i) => { place(el, subjectRect(img, shot, shot.subjects[i].box)); el.classList.add("is-locked"); });
    img.complete ? go() : (img.onload = go);
    setMode(shot, true);
  }

  function setPlaying(p) {
    playing = p;
    pauseBtn.innerHTML = `<svg aria-hidden="true"><use href="#i-${p ? "pause" : "play"}"/></svg>`;
    pauseBtn.setAttribute("aria-label", p ? "Pause the viewfinder animation" : "Play the viewfinder animation");
  }
  pauseBtn.addEventListener("click", () => {
    setPlaying(!playing);
    if (playing && runId === 0) loop();
  });

  new IntersectionObserver(([e]) => { inView = e.isIntersecting; }, { threshold: 0.15 }).observe(vf);
  let rs;
  addEventListener("resize", () => {
    clearTimeout(rs);
    rs = setTimeout(() => { sizeIris(); if (!playing || runId === 0) showStill(); }, 150);
  });

  setPlaying(playing);
  if (reduce) { showStill(); } else { sizeIris(); loop(); }

  /* ------------------------------------------------------------ mode dial */
  const tabs = $$("#modeList [role=tab]");
  const dialFace = $("#dialFace");
  const LABELS = ["PPL", "WHO", "TXT", "GPS"];
  const ANG = [-45, 45, 135, 225];
  LABELS.forEach((l, i) => {
    const a = (ANG[i] - 90) * Math.PI / 180;
    const s = document.createElement("span");
    s.className = "mk" + (i === 0 ? " on" : "");
    s.textContent = l;
    s.style.left = 50 + Math.cos(a) * 34 + "%";
    s.style.top = 50 + Math.sin(a) * 34 + "%";
    s.style.transform = `translate(-50%,-50%) rotate(${ANG[i]}deg)`;
    dialFace.appendChild(s);
  });
  dialFace.style.transform = `rotate(${-ANG[0]}deg)`;

  const panelInit = {};
  function selectTab(i, focus) {
    tabs.forEach((t, j) => {
      const on = i === j;
      t.setAttribute("aria-selected", on);
      t.tabIndex = on ? 0 : -1;
      const p = document.getElementById(t.getAttribute("aria-controls"));
      p.hidden = !on;
      p.classList.toggle("is-on", on);
    });
    $$(".mk", dialFace).forEach((m, j) => m.classList.toggle("on", j === i));
    dialFace.style.transform = `rotate(${-ANG[i]}deg)`;
    if (focus) tabs[i].focus();
    const key = tabs[i].id;
    if (panelInit[key]) panelInit[key]();
  }
  tabs.forEach((t, i) => {
    t.addEventListener("click", () => selectTab(i));
    t.addEventListener("keydown", e => {
      const n = tabs.length;
      if (["ArrowRight", "ArrowDown"].includes(e.key)) { e.preventDefault(); selectTab((i + 1) % n, true); }
      if (["ArrowLeft", "ArrowUp"].includes(e.key)) { e.preventDefault(); selectTab((i - 1 + n) % n, true); }
      if (e.key === "Home") { e.preventDefault(); selectTab(0, true); }
      if (e.key === "End") { e.preventDefault(); selectTab(n - 1, true); }
    });
  });

  /* people */
  const FACES = [
    ["women-05", "P01"], ["men-17", "P02"], ["women-10", "P03"], ["men-29", "P04"], ["women-43", "P05"], ["men-70", "P06"],
    ["women-05-c1", "P01"], ["men-03", "P07"], ["women-31", "P08"], ["men-17-c2", "P02"], ["men-45", "P09"], ["women-19", "P10"],
    ["men-89", "P11"], ["women-77", "P12"], ["men-29-c3", "P04"], ["women-13", "P13"], ["men-71", "P14"], ["women-04", "P15"],
  ];
  const facesEl = $("#faces");
  facesEl.innerHTML = FACES.map(([f, p]) =>
    `<figure><img src="assets/faces/${f}.jpg" alt="" loading="lazy"><div class="lock"><b></b></div><figcaption>${p}</figcaption></figure>`).join("");
  $("#peopleRow").innerHTML = [["women-05", "Person 01", "5 photos"], ["men-17", "Person 02", "5 photos"], ["men-29", "Person 04", "5 photos"], ["women-77", "Person 12", "5 photos"]]
    .map(([f, n, c]) => `<span><img src="assets/faces/${f}.jpg" alt="">${n} <b>${c}</b></span>`).join("");
  let peopleDone = false;
  panelInit["t-people"] = async () => {
    if (peopleDone) return;
    peopleDone = true;
    const figs = $$("figure", facesEl);
    if (reduce) { figs.forEach(f => { f.classList.add("is-on"); f.querySelector(".lock").classList.add("is-locked"); }); return; }
    for (const f of figs) {
      f.classList.add("is-on");
      setTimeout(() => f.querySelector(".lock").classList.add("is-locked"), 180);
      await wait(90);
    }
  };

  /* who is this */
  const MATCHES = [["women-05-c1", 0.96], ["women-05-c2", 0.94], ["women-05-c3", 0.91], ["women-05-c4", 0.89], null, ["women-80", 0.41], ["women-77", 0.37]];
  const matchesEl = $("#matches");
  matchesEl.innerHTML = MATCHES.map(m => m
    ? `<li class="${m[1] < 0.5 ? "below" : ""}"><img src="assets/faces/${m[0]}.jpg" alt="" loading="lazy"><div><span class="lbl">${m[1] < 0.5 ? "Different person" : "Same person"}</span><div class="bar"><i data-w="${m[1] * 100}"></i></div></div><span class="sc">${m[1].toFixed(2)}</span></li>`
    : `<li class="thresh" style="display:block;border:0;border-top:1px dashed var(--flash);padding:6px 0 0">Threshold 0.50</li>`).join("");
  panelInit["t-who"] = () => requestAnimationFrame(() => $$(".bar i", matchesEl).forEach(i => { i.style.transform = `scaleX(${i.dataset.w / 100})`; }));

  /* free text */
  const PICS = {
    leopard: "scene-0102", lighthouse: "scene-0101", castle: "scene-0142", concert: "scene-0141",
    eiffel: "scene-0147", beach: "scene-0130", waves: "scene-0149", wheel: "scene-0104", dune: "scene-0106", clock: "scene-0133", car: "scene-0139", peaks: "scene-0118",
  };
  const QUERIES = {
    "leopard": ["leopard", "dune", "peaks", "beach", "car", "castle", "waves", "clock"],
    "lighthouse at night": ["lighthouse", "eiffel", "wheel", "waves", "beach", "castle", "concert", "peaks"],
    "castle in the forest": ["castle", "peaks", "lighthouse", "eiffel", "beach", "dune", "leopard", "wheel"],
    "live music": ["concert", "wheel", "clock", "eiffel", "car", "lighthouse", "beach", "waves"],
    "old rusty car": ["car", "clock", "concert", "dune", "leopard", "wheel", "castle", "peaks"],
    "stormy sea": ["waves", "beach", "lighthouse", "peaks", "dune", "castle", "eiffel", "leopard"],
  };
  const SCORES = [0.33, 0.24, 0.22, 0.21, 0.2, 0.19, 0.18, 0.17];
  const chips = $("#chips"), results = $("#results"), typed = $("#typed");
  chips.innerHTML = Object.keys(QUERIES).map(q => `<button type="button" aria-pressed="false">${q}</button>`).join("");
  results.innerHTML = Object.entries(PICS).slice(0, 8).map(([k, f]) =>
    `<figure data-k="${k}"><img src="assets/photos/${f}-sm.jpg" alt="${k}" loading="lazy"><figcaption></figcaption></figure>`).join("");
  let typeRun = 0;
  function runQuery(q) {
    $$("button", chips).forEach(b => b.setAttribute("aria-pressed", b.textContent === q));
    const order = QUERIES[q];
    const figs = Object.fromEntries($$("figure", results).map(f => [f.dataset.k, f]));
    // make sure every key in the order exists
    order.forEach(k => {
      if (!figs[k]) {
        const f = document.createElement("figure");
        f.dataset.k = k;
        f.innerHTML = `<img src="assets/photos/${PICS[k]}-sm.jpg" alt="${k}" loading="lazy"><figcaption></figcaption>`;
        results.appendChild(f); figs[k] = f;
      }
    });
    const first = new Map(Object.values(figs).map(f => [f, f.getBoundingClientRect()]));
    Object.values(figs).forEach(f => { if (!order.includes(f.dataset.k)) f.remove(); });
    order.forEach((k, i) => {
      const f = figs[k];
      results.appendChild(f);
      f.classList.toggle("top", i === 0);
      f.classList.toggle("low", i >= 4);
      f.querySelector("figcaption").textContent = SCORES[i].toFixed(2);
    });
    if (!reduce) order.forEach(k => {
      const f = figs[k], a = first.get(f), b = f.getBoundingClientRect();
      if (!a) return;
      f.animate([{ transform: `translate(${a.left - b.left}px, ${a.top - b.top}px)` }, { transform: "none" }], { duration: 600, easing: "cubic-bezier(0.16,1,0.3,1)" });
    });
    const id = ++typeRun;
    if (reduce) { typed.textContent = q; return; }
    typed.textContent = "";
    [...q].forEach((ch, i) => setTimeout(() => { if (id === typeRun) typed.textContent += ch; }, 35 * i));
  }
  chips.addEventListener("click", e => { if (e.target.matches("button")) runQuery(e.target.textContent); });
  let textDone = false;
  panelInit["t-text"] = () => { if (!textDone) { textDone = true; runQuery("leopard"); } };

  /* places */
  const GEO = [
    { f: "scene-0147", name: "Eiffel Tower", city: "Paris, France", lat: 48.8584, lon: 2.2945, dx: -5, dy: -5, anchor: "end" },
    { f: "scene-0142", name: "Neuschwanstein", city: "Schwangau, Germany", lat: 47.5576, lon: 10.7498, dx: 5, dy: 9, anchor: "start" },
    { f: "scene-0124", name: "Financial District", city: "San Francisco, USA", lat: 37.7946, lon: -122.4029, dx: 4, dy: 7, anchor: "start" },
    { f: "scene-0136", name: "Guggenheim", city: "New York, USA", lat: 40.783, lon: -73.959, dx: 4, dy: -4, anchor: "start" },
    { f: "bulacan", name: "Portrait", city: "Baliuag, Philippines", lat: 14.954, lon: 120.897, dx: -4, dy: -5, anchor: "end", sm: true },
  ];
  const fmtLat = v => `${Math.abs(v).toFixed(4)} ${v >= 0 ? "N" : "S"}`;
  const fmtLon = v => `${Math.abs(v).toFixed(4)} ${v >= 0 ? "E" : "W"}`;
  const chart = $("#chart");
  let g = "";
  for (let lon = -150; lon <= 150; lon += 30) g += `<line class="grat" x1="${lon + 180}" y1="0" x2="${lon + 180}" y2="180"/>`;
  for (let lat = -60; lat <= 60; lat += 30) g += `<line class="${lat === 0 ? "eq" : "grat"}" x1="0" y1="${90 - lat}" x2="360" y2="${90 - lat}"/>`;
  [-120, -60, 0, 60, 120].forEach(l => { g += `<text class="axis" x="${l + 181.5}" y="112">${Math.abs(l)}°${l < 0 ? "W" : l > 0 ? "E" : ""}</text>`; });
  [60, 30, 0].forEach(l => { g += `<text class="axis" x="32" y="${90 - l - 1.5}">${Math.abs(l)}°${l > 0 ? "N" : ""}</text>`; });
  GEO.forEach(p => {
    const x = p.lon + 180, y = 90 - p.lat;
    g += `<g class="pin"><rect x="${x - 3.5}" y="${y - 3.5}" width="7" height="7"/><circle cx="${x}" cy="${y}" r="1.1"/><text x="${x + p.dx}" y="${y + p.dy}" text-anchor="${p.anchor}">${p.city.split(",")[0].toUpperCase()}</text></g>`;
  });
  chart.innerHTML = g;
  $("#geo").innerHTML = GEO.map(p =>
    `<li><img src="assets/photos/${p.f}-sm.jpg" alt="${p.name}" loading="lazy"><div><b>${p.city}</b><span>${fmtLat(p.lat)} · ${fmtLon(p.lon)}</span></div></li>`).join("");

  // init the first visible panel when the section enters
  new IntersectionObserver((es, o) => {
    if (es[0].isIntersecting) { const i = tabs.findIndex(t => t.getAttribute("aria-selected") === "true"); panelInit[tabs[i].id]?.(); o.disconnect(); }
  }, { threshold: 0.25 }).observe($("#modes"));

  /* ------------------------------------------------------------ film */
  $("#poster").addEventListener("click", () => {
    const lang = (navigator.language || "en").toLowerCase().startsWith("fr") ? "fr" : "en";
    const f = document.createElement("iframe");
    f.src = `video/?embed=1&autoplay=1&lang=${lang}`;
    f.title = "Proton Faces film";
    f.allow = "fullscreen";
    f.allowFullscreen = true;
    $("#screen").replaceChildren(f);
    f.focus();
  });

  /* ------------------------------------------------------------ copy */
  $$(".copy").forEach(b => b.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(b.dataset.copy);
      b.textContent = "Copied"; b.classList.add("done");
    } catch { b.textContent = "Select and copy"; }
    setTimeout(() => { b.textContent = "Copy"; b.classList.remove("done"); }, 1800);
  }));

})();
