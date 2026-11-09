(() => {
  const SVGNS = "http://www.w3.org/2000/svg";
  const finePointer = window.matchMedia("(hover: hover) and (pointer: fine)").matches;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const lerp = (a, b, t) => a + (b - a) * t;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  let uid = 0;

  function el(tag, attrs = {}, parent) {
    const node = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }

  function duotone(defs, id, dark, light) {
    const f = el("filter", { id, "color-interpolation-filters": "sRGB" }, defs);
    el("feColorMatrix", { type: "saturate", values: "0" }, f);
    const t = el("feComponentTransfer", {}, f);
    ["R", "G", "B"].forEach((c, i) => el(`feFunc${c}`, { type: "table", tableValues: `${dark[i]} ${light[i]}` }, t));
  }

  class Lens {
    constructor(stage, { width, height, slice, base, baseFilter, reveal, revealFilter, scanlines, label, labelKind }) {
      this.stage = stage;
      this.w = width;
      this.h = height;
      const id = `lens${++uid}`;
      const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, preserveAspectRatio: slice ? "xMidYMid slice" : "none" });
      const defs = el("defs", {}, svg);
      const m = Math.min(width, height);
      this.r = m * 0.17;

      const goo = el("filter", { id: `${id}-goo`, filterUnits: "userSpaceOnUse", x: 0, y: 0, width, height }, defs);
      el("feGaussianBlur", { in: "SourceGraphic", stdDeviation: this.r * 0.32 }, goo);
      el("feColorMatrix", { values: "1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 24 -9" }, goo);
      duotone(defs, `${id}-mono`, [0.0, 0.0, 0.0], [0.34, 0.34, 0.33]);
      duotone(defs, `${id}-ai`, [0.06, 0.01, 0.0], [1.0, 0.46, 0.22]);
      duotone(defs, `${id}-real`, [0.0, 0.03, 0.06], [0.62, 0.95, 1.0]);
      const pattern = el("pattern", { id: `${id}-scan`, width: 6, height: 6, patternUnits: "userSpaceOnUse" }, defs);
      el("rect", { width: 6, height: 2, fill: "rgba(0,0,0,0.45)" }, pattern);

      const mask = el("mask", { id: `${id}-mask`, maskUnits: "userSpaceOnUse", x: 0, y: 0, width, height }, defs);
      const blobs = el("g", { filter: `url(#${id}-goo)` }, mask);
      const sizes = [1, 0.82, 0.68, 0.56, 0.46, 0.38];
      this.circles = sizes.map((s) => ({ el: el("circle", { r: 0, fill: "#fff" }, blobs), s, x: width / 2, y: height / 2 }));

      this.baseImg = el("image", { href: base, width, height, preserveAspectRatio: slice ? "xMidYMid slice" : "none" }, svg);
      if (baseFilter) this.baseImg.setAttribute("filter", `url(#${id}-${baseFilter})`);
      const revealGroup = el("g", { mask: `url(#${id}-mask)` }, svg);
      this.revealImg = el("image", { href: reveal, width, height, preserveAspectRatio: slice ? "xMidYMid slice" : "none" }, revealGroup);
      this.filterId = id;
      if (revealFilter) this.revealImg.setAttribute("filter", `url(#${id}-${revealFilter})`);
      if (scanlines) el("rect", { width, height, fill: `url(#${id}-scan)` }, revealGroup);
      this.ring = el("circle", { r: 0, class: `lens-ring ${labelKind}` }, svg);

      stage.querySelectorAll("img.base, .no-image").forEach((n) => n.remove());
      stage.prepend(svg);
      stage.classList.add("lens-ready");
      this.svg = svg;
      this.label = document.createElement("div");
      this.label.className = `lens-label ${labelKind}`;
      this.label.textContent = label;
      stage.appendChild(this.label);

      this.target = { x: width / 2, y: height / 2 };
      this.size = 0;
      this.active = false;
      this.lastX = null;
      this.dir = 0;
      this.onDirection = null;
      this.running = false;

      stage.addEventListener("pointerenter", (e) => this.move(e, true));
      stage.addEventListener("pointermove", (e) => this.move(e));
      stage.addEventListener("pointerleave", () => {
        this.active = false;
        stage.classList.remove("active");
        this.lastX = null;
        this.loop();
      });
    }

    toLocal(e) {
      const pt = this.svg.createSVGPoint();
      pt.x = e.clientX;
      pt.y = e.clientY;
      return pt.matrixTransform(this.svg.getScreenCTM().inverse());
    }

    move(e, entering) {
      const p = this.toLocal(e);
      this.target = { x: p.x, y: p.y };
      if (entering) this.circles.forEach((c) => { c.x = p.x; c.y = p.y; });
      if (this.lastX !== null) {
        const dx = e.clientX - this.lastX;
        if (Math.abs(dx) > 0.5) this.dir = lerp(this.dir, Math.sign(dx), Math.min(Math.abs(dx) / 25, 0.35));
        if (this.onDirection) this.onDirection(this.dir);
      }
      this.lastX = e.clientX;
      const rect = this.stage.getBoundingClientRect();
      const scale = this.svg.getScreenCTM().a;
      this.label.style.left = `${e.clientX - rect.left}px`;
      this.label.style.top = `${e.clientY - rect.top - this.r * scale - 22}px`;
      this.active = true;
      this.stage.classList.add("active");
      this.loop();
    }

    loop() {
      if (this.running) return;
      this.running = true;
      const tick = () => {
        this.size = lerp(this.size, this.active ? 1 : 0, 0.16);
        const lead = this.circles[0];
        lead.x = lerp(lead.x, this.target.x, 0.32);
        lead.y = lerp(lead.y, this.target.y, 0.32);
        for (let i = 1; i < this.circles.length; i++) {
          const c = this.circles[i];
          const prev = this.circles[i - 1];
          c.x = lerp(c.x, prev.x, 0.36);
          c.y = lerp(c.y, prev.y, 0.36);
        }
        this.circles.forEach((c) => {
          c.el.setAttribute("cx", c.x);
          c.el.setAttribute("cy", c.y);
          c.el.setAttribute("r", this.r * c.s * this.size);
        });
        this.ring.setAttribute("cx", lead.x);
        this.ring.setAttribute("cy", lead.y);
        this.ring.setAttribute("r", this.r * 0.98 * this.size);
        const still = !this.active && this.size < 0.01;
        const settled = this.circles.every((c) => Math.abs(c.x - this.target.x) + Math.abs(c.y - this.target.y) < 0.5);
        if (still || (this.active && settled && Math.abs(this.size - 1) < 0.01)) {
          this.running = false;
          return;
        }
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    }

    setReveal(href, filter, kind, text) {
      this.revealImg.setAttribute("href", href);
      if (filter) this.revealImg.setAttribute("filter", `url(#${this.filterId}-${filter})`);
      else this.revealImg.removeAttribute("filter");
      this.label.className = `lens-label ${kind}`;
      this.label.textContent = text;
      this.ring.setAttribute("class", `lens-ring ${kind}`);
    }
  }

  function initResult(stage) {
    const id = stage.dataset.id;
    let src = stage.querySelector("img.base")?.src;
    if (!src) {
      try { src = sessionStorage.getItem(`vg-preview-${id}`); } catch (e) { src = null; }
      if (src) {
        const img = document.createElement("img");
        img.className = "base";
        img.alt = "Uploaded image";
        img.src = src;
        stage.querySelector(".no-image")?.remove();
        stage.prepend(img);
      }
    }
    if (!src || !finePointer) return;
    const probe = new Image();
    probe.onload = () => {
      const kind = stage.dataset.kind;
      const maxH = 560;
      stage.style.aspectRatio = `${probe.naturalWidth} / ${probe.naturalHeight}`;
      stage.style.maxWidth = `${(maxH * probe.naturalWidth) / probe.naturalHeight}px`;
      stage.style.margin = "0 auto";
      new Lens(stage, {
        width: probe.naturalWidth, height: probe.naturalHeight, slice: false,
        base: src, reveal: src, revealFilter: kind, scanlines: true,
        label: stage.dataset.label, labelKind: kind,
      });
      const hint = document.getElementById("lens-hint");
      if (hint) hint.hidden = false;
    };
    probe.src = src;
  }

  function countUp() {
    document.querySelectorAll(".count-up").forEach((node) => {
      const target = parseFloat(node.dataset.target);
      if (reduceMotion) return;
      const start = performance.now() + 500;
      const step = (now) => {
        const t = Math.min(Math.max((now - start) / 1100, 0), 1);
        const eased = 1 - Math.pow(1 - t, 3);
        node.textContent = (target * eased).toFixed(1);
        if (t < 1) requestAnimationFrame(step);
      };
      node.textContent = "0.0";
      requestAnimationFrame(step);
    });
    requestAnimationFrame(() => {
      document.querySelectorAll(".bar .fill").forEach((f) => { f.style.width = `${f.dataset.width}%`; });
    });
  }

  class Scan {
    constructor(img, canvas, status) {
      this.img = img;
      this.canvas = canvas;
      this.status = status;
      this.ctx = canvas.getContext("2d");
      this.stopped = false;
      this.collapsing = false;
    }

    setup() {
      const dpr = window.devicePixelRatio || 1;
      this.W = this.img.clientWidth;
      this.H = this.img.clientHeight;
      this.canvas.width = this.W * dpr;
      this.canvas.height = this.H * dpr;
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.canvas.style.opacity = 1;

      const edge = document.createElement("canvas");
      edge.width = Math.round(this.W);
      edge.height = Math.round(this.H);
      const ectx = edge.getContext("2d");
      ectx.drawImage(this.img, 0, 0, edge.width, edge.height);
      const src = ectx.getImageData(0, 0, edge.width, edge.height);
      const out = ectx.createImageData(edge.width, edge.height);
      const w = edge.width;
      const gray = new Float32Array(w * edge.height);
      for (let i = 0; i < gray.length; i++) gray[i] = 0.3 * src.data[i * 4] + 0.59 * src.data[i * 4 + 1] + 0.11 * src.data[i * 4 + 2];
      for (let y = 1; y < edge.height - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
          const i = y * w + x;
          const gx = -gray[i - w - 1] - 2 * gray[i - 1] - gray[i + w - 1] + gray[i - w + 1] + 2 * gray[i + 1] + gray[i + w + 1];
          const gy = -gray[i - w - 1] - 2 * gray[i - w] - gray[i - w + 1] + gray[i + w - 1] + 2 * gray[i + w] + gray[i + w + 1];
          const mag = Math.min(255, Math.hypot(gx, gy) * 1.3);
          out.data[i * 4] = 119;
          out.data[i * 4 + 1] = 231;
          out.data[i * 4 + 2] = 254;
          out.data[i * 4 + 3] = mag;
        }
      }
      const tmp = document.createElement("canvas");
      tmp.width = edge.width;
      tmp.height = edge.height;
      tmp.getContext("2d").putImageData(out, 0, 0);
      ectx.fillStyle = "#07080a";
      ectx.fillRect(0, 0, edge.width, edge.height);
      ectx.drawImage(tmp, 0, 0);
      this.edge = edge;
    }

    start() {
      this.setup();
      const t0 = performance.now();
      this.firstPass = sleep(1600);
      const tick = (now) => {
        if (this.stopped || this.collapsing) return;
        const elapsed = now - t0;
        const fadeIn = Math.min(elapsed / 1000, 1);
        const pulse = elapsed > 1000 ? 0.82 + 0.18 * Math.sin((elapsed - 1000) / 260) : 1;
        this.ctx.clearRect(0, 0, this.W, this.H);
        this.ctx.globalAlpha = fadeIn * pulse;
        this.ctx.drawImage(this.edge, 0, 0, this.W, this.H);
        this.ctx.globalAlpha = 1;
        this.status.textContent = "CONVOLUTION · EDGE FILTERS";
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    }

    stop() {
      this.stopped = true;
      this.ctx.clearRect(0, 0, this.W, this.H);
    }

    async collapse() {
      this.collapsing = true;
      this.ctx.clearRect(0, 0, this.W, this.H);
      this.ctx.drawImage(this.edge, 0, 0, this.W, this.H);
      this.status.textContent = "ANALYSIS COMPLETE";
      await sleep(300);
      this.canvas.closest(".preview-box").classList.add("leaving");
      await sleep(380);
    }
  }

  function seeded(seed) {
    let s = seed;
    return () => (s = (s * 16807) % 2147483647) / 2147483647;
  }

  function clipPoly(poly, a, b, c) {
    const out = [];
    for (let i = 0; i < poly.length; i++) {
      const p = poly[i];
      const q = poly[(i + 1) % poly.length];
      const pin = a * p[0] + b * p[1] <= c;
      const qin = a * q[0] + b * q[1] <= c;
      if (pin) out.push(p);
      if (pin !== qin) {
        const t = (c - a * p[0] - b * p[1]) / (a * (q[0] - p[0]) + b * (q[1] - p[1]));
        out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
      }
    }
    return out;
  }

  function shatterPattern(w, h, n, seed) {
    const rand = seeded(seed);
    const impact = [w * (0.35 + rand() * 0.3), h * (0.35 + rand() * 0.3)];
    const pts = [];
    for (let i = 0; i < n; i++) {
      if (i < n * 0.5) {
        const ang = rand() * Math.PI * 2;
        const r = Math.pow(rand(), 1.4) * Math.max(w, h) * 0.38;
        pts.push([Math.min(w - 1, Math.max(1, impact[0] + Math.cos(ang) * r)), Math.min(h - 1, Math.max(1, impact[1] + Math.sin(ang) * r))]);
      } else {
        pts.push([rand() * w, rand() * h]);
      }
    }
    const shards = pts.map((p, i) => {
      let poly = [[0, 0], [w, 0], [w, h], [0, h]];
      pts.forEach((q, j) => {
        if (i === j) return;
        poly = clipPoly(poly, q[0] - p[0], q[1] - p[1], (q[0] * q[0] + q[1] * q[1] - p[0] * p[0] - p[1] * p[1]) / 2);
      });
      const cx = poly.reduce((s, v) => s + v[0], 0) / poly.length;
      const cy = poly.reduce((s, v) => s + v[1], 0) / poly.length;
      const dist = Math.hypot(cx - impact[0], cy - impact[1]);
      return { poly, cx, cy, dist, spin: (rand() - 0.5) * 2, lift: 0.6 + rand() * 0.8 };
    }).filter((s) => s.poly.length > 2);
    const maxDist = Math.max(...shards.map((s) => s.dist)) || 1;
    shards.forEach((s) => { s.order = s.dist / maxDist; });

    const seen = new Set();
    const cracks = [];
    shards.forEach((s) => s.poly.forEach((p, k) => {
      const q = s.poly[(k + 1) % s.poly.length];
      const onBorder = (v) => v[0] < 0.5 || v[1] < 0.5 || v[0] > w - 0.5 || v[1] > h - 0.5;
      if (onBorder(p) && onBorder(q)) return;
      const key = [p, q].map((v) => `${Math.round(v[0])},${Math.round(v[1])}`).sort().join("|");
      if (seen.has(key)) return;
      seen.add(key);
      const [near, far] = Math.hypot(p[0] - impact[0], p[1] - impact[1]) < Math.hypot(q[0] - impact[0], q[1] - impact[1]) ? [p, q] : [q, p];
      cracks.push({ near, far, d: Math.hypot(near[0] - impact[0], near[1] - impact[1]), len: Math.hypot(far[0] - near[0], far[1] - near[1]) });
    }));
    const maxCrack = Math.max(...cracks.map((c) => c.d + c.len)) || 1;
    return { shards, cracks, impact, maxCrack };
  }

  const clamp01 = (v) => Math.min(1, Math.max(0, v));
  const easeOut = (t) => 1 - Math.pow(1 - t, 3);
  const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

  function initSorter(stage) {
    const section = document.getElementById("sort-section");
    const frame = document.getElementById("sort-frame");
    const canvas = document.getElementById("sort-canvas");
    const ctx = canvas.getContext("2d");
    const bins = { ai: document.getElementById("bin-ai"), real: document.getElementById("bin-real") };
    const counts = { ai: document.getElementById("count-ai"), real: document.getElementById("count-real") };
    const dots = [...document.querySelectorAll("#sort-dots span")];
    const IW = 960;
    const IH = 720;
    const deck = stage.dataset.images.split(",").map((entry, i) => {
      const [src, kind] = entry.split("|");
      const img = new Image();
      img.src = src;
      return { img, kind, pattern: shatterPattern(IW, IH, 30, 11 + i * 97) };
    });
    const colors = { ai: [241, 84, 25], real: [119, 231, 254] };
    let geom = null;
    let queued = false;

    function measure() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const c = canvas.getBoundingClientRect();
      canvas.width = c.width * dpr;
      canvas.height = c.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const f = frame.getBoundingClientRect();
      const center = (el) => {
        const r = el.querySelector(".bin-icon").getBoundingClientRect();
        return [r.left + r.width / 2 - c.left, r.top + r.height / 2 - c.top];
      };
      geom = { x: f.left - c.left, y: f.top - c.top, w: f.width, h: f.height, s: f.width / IW, bin: { ai: center(bins.ai), real: center(bins.real) } };
    }

    function roundRectPath(x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
    }

    function drawWhole(item, alpha, scale) {
      if (!item.img.complete) return;
      const { x, y, w, h } = geom;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.translate(x + w / 2, y + h / 2);
      ctx.scale(scale, scale);
      roundRectPath(-w / 2, -h / 2, w, h, 16);
      ctx.clip();
      ctx.drawImage(item.img, -w / 2, -h / 2, w, h);
      ctx.restore();
    }

    function drawCracks(item, p) {
      const { x, y, s } = geom;
      const reach = p * item.pattern.maxCrack;
      const rgb = colors[item.kind];
      ctx.save();
      ctx.lineCap = "round";
      for (const pass of [{ w: 7, a: 0.16 }, { w: 2.6, a: 0.5 }, { w: 1, a: 0.95 }]) {
        ctx.lineWidth = pass.w;
        ctx.strokeStyle = pass.w === 1 ? `rgba(255,255,255,${pass.a})` : `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${pass.a})`;
        ctx.beginPath();
        for (const c of item.pattern.cracks) {
          const k = clamp01((reach - c.d) / c.len);
          if (k <= 0) continue;
          ctx.moveTo(x + c.near[0] * s, y + c.near[1] * s);
          ctx.lineTo(x + (c.near[0] + (c.far[0] - c.near[0]) * k) * s, y + (c.near[1] + (c.far[1] - c.near[1]) * k) * s);
        }
        ctx.stroke();
      }
      const [ix, iy] = item.pattern.impact;
      const g = ctx.createRadialGradient(x + ix * s, y + iy * s, 0, x + ix * s, y + iy * s, 60 * p + 1);
      g.addColorStop(0, `rgba(255,255,255,${0.5 * p})`);
      g.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = g;
      ctx.fillRect(x + ix * s - 70, y + iy * s - 70, 140, 140);
      ctx.restore();
    }

    function drawShards(item, q) {
      if (!item.img.complete) return;
      const { x, y, w, h, s } = geom;
      const [bx, by] = geom.bin[item.kind];
      const rgb = colors[item.kind];
      const [ix, iy] = item.pattern.impact;
      for (const sh of item.pattern.shards) {
        const u = clamp01((q - sh.order * 0.3) / 0.7);
        if (u >= 1) continue;
        const sx = x + sh.cx * s;
        const sy = y + sh.cy * s;
        let dx = sh.cx - ix;
        let dy = sh.cy - iy;
        const dl = Math.hypot(dx, dy) || 1;
        dx /= dl;
        dy /= dl;
        const pop = easeOut(clamp01(u / 0.3));
        const px = sx + dx * 34 * pop * sh.lift;
        const py = sy + dy * 34 * pop * sh.lift - 18 * pop;
        const fly = easeInOut(clamp01((u - 0.3) / 0.7));
        const cxp = px + (bx - px) * 0.25;
        const cyp = Math.min(py, by) - 90 * sh.lift;
        const mt = 1 - fly;
        const X = mt * mt * px + 2 * mt * fly * cxp + fly * fly * bx;
        const Y = mt * mt * py + 2 * mt * fly * cyp + fly * fly * by;
        const sc = 1 - 0.88 * fly;
        const rot = sh.spin * (0.25 * pop + 2.6 * fly);
        const alpha = 1 - clamp01((fly - 0.82) / 0.18);
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.translate(X, Y);
        ctx.rotate(rot);
        ctx.scale(sc, sc);
        ctx.beginPath();
        sh.poly.forEach((v, k) => (k ? ctx.lineTo : ctx.moveTo).call(ctx, (v[0] - sh.cx) * s, (v[1] - sh.cy) * s));
        ctx.closePath();
        ctx.save();
        ctx.clip();
        ctx.drawImage(item.img, -sh.cx * s, -sh.cy * s, w, h);
        ctx.restore();
        ctx.lineWidth = 1.4 / Math.max(sc, 0.2);
        ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${0.55 + 0.45 * fly})`;
        ctx.shadowColor = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
        ctx.shadowBlur = 10;
        ctx.stroke();
        ctx.restore();
      }
    }

    function drawTag(text, kind, alpha) {
      const { x, y } = geom;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.font = "700 12px 'JetBrains Mono', monospace";
      const w = ctx.measureText(text).width + 22;
      roundRectPath(x + 14, y + 14, w, 26, 13);
      if (kind) {
        const rgb = colors[kind];
        ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
      } else {
        ctx.fillStyle = "rgba(12,12,10,0.7)";
      }
      ctx.fill();
      ctx.fillStyle = kind === "real" ? "#0c0c0a" : "#f2f2ee";
      ctx.fillText(text, x + 25, y + 31);
      ctx.restore();
    }

    function render() {
      queued = false;
      if (!geom) measure();
      const N = deck.length;
      const r = section.getBoundingClientRect();
      const total = section.offsetHeight - window.innerHeight;
      const P = clamp01(-r.top / Math.max(total, 1)) * N;
      const i = Math.min(Math.floor(P), N - 1);
      const t = P - i;
      const crack = clamp01(t / 0.45);
      const shatter = clamp01((t - 0.52) / 0.4);
      const appear = clamp01((t - 0.78) / 0.2);
      const item = deck[i];

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (shatter <= 0) {
        drawWhole(item, 1, 1);
        if (crack > 0) drawCracks(item, crack);
        drawTag(crack > 0.98 ? "ANALYSING…" : `SPECIMEN 0${i + 1}`, null, 1);
      } else {
        if (i + 1 < N) drawWhole(deck[i + 1], appear, 0.94 + 0.06 * appear);
        drawShards(item, shatter);
        if (shatter < 0.6) drawTag(item.kind === "ai" ? "→ AI BIN" : "→ REAL BIN", item.kind, 1 - clamp01((shatter - 0.4) / 0.2));
        if (i === N - 1 && appear > 0) {
          const { x, y, w, h } = geom;
          ctx.save();
          ctx.globalAlpha = appear;
          ctx.fillStyle = "#f2f2ee";
          ctx.textAlign = "center";
          ctx.font = "600 26px 'Space Grotesk', sans-serif";
          ctx.fillText(`${N} / ${N} sorted.`, x + w / 2, y + h / 2 - 6);
          ctx.fillStyle = "#77e7fe";
          ctx.font = "500 13px 'JetBrains Mono', monospace";
          ctx.fillText("NOW TRY YOUR OWN ↓", x + w / 2, y + h / 2 + 24);
          ctx.restore();
        }
      }

      const tally = { ai: 0, real: 0 };
      deck.forEach((d, j) => { if (j < i || (j === i && shatter > 0.92)) tally[d.kind] += 1; });
      counts.ai.textContent = tally.ai;
      counts.real.textContent = tally.real;
      bins.ai.classList.toggle("receiving", item.kind === "ai" && shatter > 0.35 && shatter < 1);
      bins.real.classList.toggle("receiving", item.kind === "real" && shatter > 0.35 && shatter < 1);
      dots.forEach((dot, j) => {
        dot.className = j < i || (j === i && shatter > 0.92) ? `done ${deck[j].kind}` : j === i ? "current" : "";
      });
    }

    const queue = () => {
      if (!queued) {
        queued = true;
        requestAnimationFrame(render);
      }
    };
    window.addEventListener("scroll", queue, { passive: true });
    window.addEventListener("resize", () => { geom = null; queue(); });
    deck.forEach((d) => { d.img.onload = queue; });
    if (document.fonts) document.fonts.ready.then(queue);
    queue();
  }

  function initFlow(canvas) {
    const ctx = canvas.getContext("2d");
    const rand = seeded(42);
    const strands = Array.from({ length: 34 }, (_, i) => ({
      y0: 0.12 + rand() * 0.8,
      amp: 50 + rand() * 120,
      f1: 0.0015 + rand() * 0.002,
      f2: 0.003 + rand() * 0.003,
      sp: 0.00012 + rand() * 0.00025,
      ph: rand() * Math.PI * 2,
      warm: i % 7 === 0,
      w: 0.6 + rand() * 1.1,
      a: 0.05 + rand() * 0.09,
    }));
    const mouse = { x: -9999, y: -9999, tx: -9999, ty: -9999 };
    let W = 0;
    let H = 0;
    let last = 0;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      W = window.innerWidth;
      H = window.innerHeight;
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const draw = (now) => {
      ctx.clearRect(0, 0, W, H);
      ctx.globalCompositeOperation = "lighter";
      mouse.x = lerp(mouse.x, mouse.tx, 0.08);
      mouse.y = lerp(mouse.y, mouse.ty, 0.08);
      for (const st of strands) {
        ctx.beginPath();
        for (let x = -40; x <= W + 40; x += 20) {
          let y = H * st.y0 + st.amp * Math.sin(x * st.f1 + now * st.sp + st.ph) +
            st.amp * 0.45 * Math.sin(x * st.f2 - now * st.sp * 0.7 + st.ph * 2);
          const dx = x - mouse.x;
          const dy = y - mouse.y;
          const d = Math.hypot(dx, dy);
          if (d < 220) y += (dy >= 0 ? 1 : -1) * (220 - d) * 0.45;
          if (x === -40) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.lineWidth = st.w;
        ctx.strokeStyle = st.warm ? `rgba(241,84,25,${st.a * 0.8})` : `rgba(119,231,254,${st.a})`;
        ctx.stroke();
      }
      ctx.globalCompositeOperation = "source-over";
    };

    const loop = (now) => {
      if (!document.hidden && now - last > 33) {
        last = now;
        draw(now);
      }
      requestAnimationFrame(loop);
    };

    resize();
    window.addEventListener("resize", resize);
    window.addEventListener("pointermove", (e) => { mouse.tx = e.clientX; mouse.ty = e.clientY; });
    if (reduceMotion) draw(0);
    else requestAnimationFrame(loop);
  }

  function savePreview(img, id) {
    try {
      const scale = Math.min(1, 1100 / Math.max(img.naturalWidth, img.naturalHeight));
      const c = document.createElement("canvas");
      c.width = Math.round(img.naturalWidth * scale);
      c.height = Math.round(img.naturalHeight * scale);
      c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
      sessionStorage.setItem(`vg-preview-${id}`, c.toDataURL("image/jpeg", 0.86));
    } catch (e) {
      /* preview is optional */
    }
  }

  function initUpload(form) {
    const input = document.getElementById("file-input");
    const zone = document.getElementById("dropzone");
    const box = document.getElementById("preview-box");
    const img = document.getElementById("preview-img");
    const canvas = document.getElementById("scan-canvas");
    const status = document.getElementById("scan-status");
    const nameEl = document.getElementById("file-name");
    const btn = document.getElementById("analyse-btn");
    const errBox = document.getElementById("inline-error");
    let objectUrl = null;

    const showError = (title, message) => {
      errBox.querySelector("strong").textContent = title;
      errBox.querySelector("span").textContent = message;
      errBox.classList.add("show");
    };

    const showFile = (file) => {
      errBox.classList.remove("show");
      if (!file) return;
      nameEl.textContent = file.name;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = URL.createObjectURL(file);
      img.onload = () => box.classList.add("has-image");
      img.onerror = () => box.classList.remove("has-image");
      img.src = objectUrl;
    };

    zone.addEventListener("pointermove", (e) => {
      const r = zone.getBoundingClientRect();
      zone.style.setProperty("--mx", `${e.clientX - r.left}px`);
      zone.style.setProperty("--my", `${e.clientY - r.top}px`);
    });
    ["dragenter", "dragover"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.add("dragging"); }));
    ["dragleave", "drop"].forEach((t) => zone.addEventListener(t, () => zone.classList.remove("dragging")));
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      if (!e.dataTransfer.files.length) return;
      input.files = e.dataTransfer.files;
      showFile(input.files[0]);
    });
    input.addEventListener("change", () => showFile(input.files[0]));

    form.addEventListener("submit", async (e) => {
      if (!window.fetch || !input.files.length) return;
      e.preventDefault();
      errBox.classList.remove("show");
      btn.disabled = true;
      btn.textContent = "Scanning…";
      const canAnimate = box.classList.contains("has-image") && !reduceMotion;
      const scan = canAnimate ? new Scan(img, canvas, status) : null;
      if (scan) {
        box.classList.add("scanning");
        scan.start();
      }
      const reset = () => {
        if (scan) scan.stop();
        box.classList.remove("scanning");
        btn.disabled = false;
        btn.textContent = "Analyse image";
      };
      try {
        const response = await fetch(form.action, {
          method: "POST",
          body: new FormData(form),
          headers: { "X-Requested-With": "fetch" },
        });
        let body = {};
        try { body = await response.json(); } catch (err) { body = {}; }
        if (scan) await scan.firstPass;
        if (!response.ok) {
          reset();
          showError(body.title || `Error ${response.status}`, body.message || "The image could not be analysed.");
          return;
        }
        if (scan) await scan.collapse();
        if (box.classList.contains("has-image")) savePreview(img, body.id);
        window.location.href = body.url;
      } catch (err) {
        reset();
        showError("Connection problem", "The server could not be reached. Please try again.");
      }
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    const flow = document.getElementById("bg-flow");
    if (flow) initFlow(flow);
    const sorter = document.getElementById("sort-stage");
    if (sorter) initSorter(sorter);
    document.querySelectorAll('[data-lens="result"]').forEach(initResult);
    const form = document.getElementById("upload-form");
    if (form) initUpload(form);
    countUp();
    document.querySelectorAll("tr[data-href]").forEach((row) => {
      row.addEventListener("click", (e) => {
        if (e.target.closest("a")) return;
        window.location.href = row.dataset.href;
      });
    });
  });
})();
