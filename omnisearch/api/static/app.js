"use strict";

const $ = (id) => document.getElementById(id);
const state = { coll: null, info: null, image: null, hits: [], branches: [], seq: 0 };
const SERIES = ["--s1", "--s2", "--s3", "--s4"];
const TIMING_COLORS = ["--s1", "--s2", "--s3", "--s4", "--faint"];
const ICON_SEARCH = `<svg viewBox="0 0 24 24"><circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.5 15.5l5 5"/></svg>`;
const ICON_X = `<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg>`;

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Escape `text` and wrap words starting with any of `terms` in <mark>.
function highlight(text, terms) {
  text = String(text ?? "");
  const words = (terms || []).filter(Boolean).map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (!words.length) return esc(text);
  const re = new RegExp(`\\b(?:${words.join("|")})\\w*`, "gi");
  let out = "", last = 0;
  for (const m of text.matchAll(re)) {
    out += esc(text.slice(last, m.index)) + `<mark>${esc(m[0])}</mark>`;
    last = m.index + m[0].length;
  }
  return out + esc(text.slice(last));
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || res.statusText);
  return body;
}

function status(html, isError = false) {
  $("status").innerHTML = isError ? `<span class="error">${esc(html)}</span>` : html;
}

function collBase() {
  return `/api/collections/${encodeURIComponent(state.coll)}`;
}

/* ---------- collections & filters ---------- */

async function loadCollections() {
  const list = await api("/api/collections");
  const sel = $("collection");
  sel.innerHTML = list.map((c) => `<option value="${esc(c.name)}">${esc(c.name)} · ${c.docs} docs</option>`).join("");
  if (!list.length) {
    $("results").innerHTML = emptyState("No collections yet", "Build the demo collection with <code>make demo</code>, then reload this page.");
    return;
  }
  await selectCollection(list[0].name);
}

async function selectCollection(name) {
  state.coll = name;
  state.info = await api(collBase());
  const st = state.info.stats;
  const pills = [];
  for (const [k, v] of Object.entries(st.vectors)) pills.push(`<span class="pill" title="vector field">${esc(k)} · ${esc(v.index)}</span>`);
  if (st.cost_model.calibrated) pills.push(`<span class="pill ok">cost model calibrated</span>`);
  $("collstats").innerHTML = pills.join("");
  renderFilters();
  search();
}

function isColor(v) {
  return typeof CSS !== "undefined" && CSS.supports("color", v);
}

function renderFilters() {
  let html = "";
  for (const [field, values] of Object.entries(state.info.facets)) {
    if (!values.length) continue;
    const colors = values.every(([v]) => isColor(v));
    html += `<div class="facet"><div class="facet-title">${esc(field)}</div><div class="chips" data-facet="${esc(field)}">` +
      values.map(([v, n]) => `<button type="button" class="chip" data-v="${esc(v)}" aria-pressed="false">` +
        `${colors ? `<span class="swatch" style="background:${esc(v)}"></span>` : ""}${esc(v)}<span class="n">${n}</span></button>`).join("") +
      `</div></div>`;
  }
  for (const [field, [lo, hi]] of Object.entries(state.info.ranges)) {
    if (lo === null) continue;
    html += `<div class="facet"><div class="facet-title">${esc(field)} <span class="small">${esc(lo)} – ${esc(hi)}</span></div>
      <div class="range" data-range="${esc(field)}">
        <input type="number" step="any" placeholder="${esc(lo)}" data-k="min" aria-label="${esc(field)} minimum">–
        <input type="number" step="any" placeholder="${esc(hi)}" data-k="max" aria-label="${esc(field)} maximum"></div></div>`;
  }
  const box = $("filters");
  box.innerHTML = html;
  box.querySelectorAll(".chip").forEach((chip) => chip.addEventListener("click", () => {
    chip.classList.toggle("on");
    chip.setAttribute("aria-pressed", chip.classList.contains("on"));
    filtersChanged();
  }));
  box.querySelectorAll(".range input").forEach((i) => i.addEventListener("change", filtersChanged));
  updateFilterCount();
}

function collectFilters() {
  const out = {};
  document.querySelectorAll("[data-facet]").forEach((g) => {
    const on = [...g.querySelectorAll(".chip.on")].map((c) => c.dataset.v);
    if (on.length) out[g.dataset.facet] = on;
  });
  document.querySelectorAll("[data-range]").forEach((g) => {
    const r = {};
    g.querySelectorAll("input").forEach((i) => { if (i.value !== "") r[i.dataset.k] = Number(i.value); });
    if (Object.keys(r).length) out[g.dataset.range] = r;
  });
  return out;
}

function updateFilterCount() {
  const f = collectFilters();
  const n = Object.values(f).reduce((a, v) => a + (Array.isArray(v) ? v.length : 1), 0);
  for (const id of ["fcount", "fcount2"]) { $(id).textContent = n; $(id).hidden = !n; }
  $("clearfilters").hidden = !n;
}

let filterTimer = null;
function filtersChanged() {
  updateFilterCount();
  clearTimeout(filterTimer);
  filterTimer = setTimeout(search, 120);
}

function clearFilters() {
  document.querySelectorAll("#filters .chip.on").forEach((c) => { c.classList.remove("on"); c.setAttribute("aria-pressed", "false"); });
  document.querySelectorAll("#filters .range input").forEach((i) => { i.value = ""; });
  filtersChanged();
}

/* ---------- results ---------- */

function mediaKinds() {
  return Object.fromEntries((state.info.schema.fields || []).filter((f) => f.field_type === "media").map((f) => [f.name, f.kind]));
}

function mediaFor(hit, large = false) {
  const base = `${collBase()}/media/${encodeURIComponent(hit.id)}`;
  for (const [field, kind] of Object.entries(mediaKinds())) {
    if (!hit.record[field]) continue;
    if (kind === "image") return `<img loading="lazy" src="${base}/${field}" alt="${esc(hit.record.title || hit.id)}">`;
    if (kind === "video") {
      const t = hit.moment != null ? `#t=${hit.moment}` : "";
      return `<video ${large ? "controls autoplay muted" : "muted"} preload="metadata" src="${base}/${field}${t}"></video>`;
    }
    if (kind === "audio") return `<audio controls src="${base}/${field}"></audio>`;
  }
  return `<span class="muted">no media</span>`;
}

function fmtTime(s) {
  const m = Math.floor(s / 60), r = Math.floor(s % 60);
  return `${m}:${String(r).padStart(2, "0")}`;
}

function attrBadges(r) {
  const out = [];
  if (r.price != null) out.push(`<span class="badge price">$${Number(r.price).toFixed(2)}</span>`);
  for (const k of ["brand", "category", "year"]) if (r[k] != null) out.push(`<span class="badge">${esc(r[k])}</span>`);
  for (const c of [].concat(r.color ?? [])) {
    out.push(`<span class="badge">${isColor(c) ? `<span class="swatch" style="background:${esc(c)}"></span>` : ""}${esc(c)}</span>`);
  }
  return out.join("");
}

function emptyState(title, body, extra = "") {
  return `<div class="empty">${ICON_SEARCH}<h3>${title}</h3><p>${body}</p>${extra}</div>`;
}

function renderSkeleton() {
  const n = Math.min(12, Number($("limit").value) || 12);
  $("results").innerHTML = Array.from({ length: n }, () => `<div class="card skeleton" aria-hidden="true"><div class="media"></div>
    <div class="body"><div class="line" style="width:75%"></div><div class="line" style="width:45%"></div><div class="line" style="width:90%;margin-top:10px"></div></div></div>`).join("");
}

function renderLegend() {
  $("legend").innerHTML = state.branches.map((n, i) => `<span><i style="background:var(${SERIES[i % 4]})"></i>${esc(n)}</span>`).join("");
}

function renderResults(data) {
  const hits = data.hits;
  state.hits = hits;
  state.branches = (data.plan?.branches || []).map((b) => b.name);
  renderLegend();
  $("rcount").textContent = hits.length;
  $("rcount").hidden = false;
  if (!hits.length) {
    const hasFilters = Object.keys(collectFilters()).length > 0;
    $("results").innerHTML = emptyState("No results",
      hasFilters ? "Nothing matches this query with the current filters." : "Try different words or search with an image.",
      hasFilters ? `<button type="button" class="link" data-action="clear">Clear filters</button>` : "");
    return;
  }
  const names = state.branches;
  const maxBy = Object.fromEntries(names.map((n) => [n, Math.max(1e-9, ...hits.map((h) => h.sources[n]?.score ?? 0))]));
  $("results").innerHTML = hits.map((h, idx) => {
    const r = h.record;
    const bars = names.map((n, i) => {
      const s = h.sources[n];
      const v = s?.score ?? 0;
      const pct = Math.max(0, Math.min(100, (100 * v) / maxBy[n]));
      const rank = s?.rank ? `#${s.rank}` : "–";
      return `<div class="bar" title="${esc(n)}: score ${v.toFixed(4)}, rank ${rank}">
        <span class="track"><span class="fill" style="width:${pct}%;background:var(${SERIES[i % 4]})"></span></span><span>${rank}</span></div>`;
    }).join("");
    const moment = h.moment != null ? `<span class="badge moment">▶ ${fmtTime(h.moment)}</span>` : "";
    const desc = r.description ? `<div class="desc">${highlight(r.description, h.matched_terms)}</div>` : "";
    return `<article class="card" tabindex="0" data-i="${idx}" style="animation-delay:${Math.min(idx, 12) * 25}ms">
      <div class="media">${mediaFor(h)}<span class="rank">${idx + 1}</span></div>
      <div class="body">
        <div class="title">${highlight(r.title || h.id, h.matched_terms)}</div>
        ${desc}
        <div class="attrs">${attrBadges(r)}${moment}</div>
        ${bars ? `<div class="bars">${bars}</div>` : ""}
        <div class="foot"><span>${esc(h.id)}</span>${names.length ? `<span>fused ${h.score.toFixed(4)}</span>` : ""}</div>
      </div></article>`;
  }).join("");
}

/* ---------- detail dialog ---------- */

function openDetail(i) {
  const h = state.hits[i];
  if (!h) return;
  const r = h.record;
  const kinds = mediaKinds();
  const skip = new Set(["title", "description", ...Object.keys(kinds)]);
  const ocr = Object.entries(r).filter(([k, v]) => /\.ocr$/.test(k) && v);
  const fields = Object.entries(r).filter(([k, v]) => !skip.has(k) && !k.includes(".ocr") && v != null && v !== "");
  const scores = state.branches.map((n, j) => {
    const s = h.sources[n];
    return `<tr><td><i style="background:var(${SERIES[j % 4]})"></i>${esc(n)}</td><td>${s ? s.score.toFixed(4) : "–"}</td>
      <td>${s?.rank ? "#" + s.rank : "–"}</td><td>${s?.weight != null ? s.weight.toFixed(2) : "–"}</td></tr>`;
  }).join("");
  $("detail-body").innerHTML = `
    <button type="button" class="icon-btn d-close" data-action="close" aria-label="Close">${ICON_X}</button>
    <div class="d-grid">
      <div class="d-media">${mediaFor(h, true)}</div>
      <div class="d-info">
        <h2>${highlight(r.title || h.id, h.matched_terms)}</h2>
        <div class="attrs">${attrBadges(r)}${h.moment != null ? `<span class="badge moment">best moment ${fmtTime(h.moment)}</span>` : ""}</div>
        ${r.description ? `<div class="d-sec"><p>${highlight(r.description, h.matched_terms)}</p></div>` : ""}
        ${scores ? `<div class="d-sec"><h4>Why it ranked #${i + 1}</h4>
          <table class="scores"><thead><tr><th>branch</th><th>score</th><th>rank</th><th>weight</th></tr></thead><tbody>${scores}</tbody></table>
          <p class="muted" style="font-size:12.5px;margin-top:6px">fused score ${h.score.toFixed(4)}${h.matched_terms.length ? ` · matched ${h.matched_terms.map(esc).join(", ")}` : ""}</p></div>` : ""}
        ${ocr.map(([k, v]) => `<div class="d-sec"><h4>${esc(k)} (text found in media)</h4><div class="ocr">${highlight(v, h.matched_terms)}</div></div>`).join("")}
        <div class="d-sec"><h4>Fields</h4><dl class="kv"><dt>id</dt><dd>${esc(h.id)}</dd>
          ${fields.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(Array.isArray(v) ? v.join(", ") : typeof v === "object" ? JSON.stringify(v) : v)}</dd>`).join("")}</dl></div>
      </div>
    </div>`;
  $("detail").showModal();
}

/* ---------- query plan ---------- */

function renderPlan(data) {
  const t = Object.entries(data.timings || {}).filter(([k]) => k !== "total_ms");
  const total = data.timings?.total_ms ?? t.reduce((a, [, v]) => a + v, 0);
  const sum = t.reduce((a, [, v]) => a + v, 0) || 1;
  const timing = `<div class="plan-card"><h3>Timing <span class="muted">${total.toFixed(2)} ms total</span></h3>
    <div class="stack">${t.map(([, v], i) => `<span style="width:${(100 * v) / sum}%;background:var(${TIMING_COLORS[i % 5]})"></span>`).join("")}</div>
    <div class="stack-legend">${t.map(([k, v], i) => `<span><i style="background:var(${TIMING_COLORS[i % 5]})"></i>${esc(k.replace("_ms", ""))} <b>${v.toFixed(2)} ms</b></span>`).join("")}</div></div>`;

  const p = data.plan;
  let body = "";
  if (p?.fuse) {
    const f = p.fuse;
    const weights = Object.entries(f.weights || {}).map(([k, v]) => `${esc(k)}=${Number(v).toFixed(2)}`).join(", ");
    body += `<div class="plan-card"><h3>Fusion · ${esc(f.method?.toUpperCase())} <span class="muted">est ${fmtMs(f.est_cost_ms)} · actual ${fmtMs(f.actual?.ms)}</span></h3>
      <dl class="kv"><dt>algorithm</dt><dd>${esc(f.algorithm)}</dd><dt>depth</dt><dd>${esc(f.depth)}</dd>
      ${weights ? `<dt>weights</dt><dd>${weights}</dd>` : ""}<dt>candidates</dt><dd>${esc(f.actual?.candidates ?? "–")}</dd><dt>limit</dt><dd>${esc(p.limit)}</dd></dl>
      ${rejected(f.rejected)}</div>`;
  }
  if (p?.branches?.length) {
    body += `<div class="plan-card"><h3>Branches <span class="muted">${p.branches.length} retrieval path${p.branches.length > 1 ? "s" : ""}</span></h3><div class="branches">` +
      p.branches.map((b, i) => {
        const actual = Object.entries(b.actual || {}).filter(([k]) => k !== "ms").map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
        return `<div class="branch" style="--c:var(${SERIES[i % 4]})">
          <div class="name"><span>${esc(b.name)}</span><span class="muted">${esc(b.kind)}</span></div>
          <div class="op">${esc(b.operator)}</div>
          ${b.clause ? `<div class="clause">${esc(b.clause)}</div>` : ""}
          <dl class="kv"><dt>strategy</dt><dd>${esc(b.strategy)}</dd><dt>k</dt><dd>${esc(b.k)}</dd>
            <dt>estimated</dt><dd>${fmtMs(b.est_cost_ms)}</dd><dt>actual</dt><dd>${fmtMs(b.actual?.ms)}</dd>${actual}</dl>
          ${rejected(b.rejected)}</div>`;
      }).join("") + `</div></div>`;
  }
  if (p?.filter) body += `<div class="plan-card"><h3>Filter</h3><pre class="code">${esc(JSON.stringify(p.filter, null, 2))}</pre></div>`;
  const raw = data.explain ? `<details class="raw" ${body ? "" : "open"}><summary>Raw EXPLAIN output</summary><pre class="code">${esc(data.explain)}</pre></details>` : "";
  $("plan").innerHTML = timing + body + raw;
}

function fmtMs(v) {
  return v == null ? "–" : `${Number(v).toFixed(3)} ms`;
}

function rejected(list) {
  if (!list?.length) return "";
  return `<div class="rejected">rejected: ${list.map((r) => `<s>${esc(r.strategy || r.algorithm)}</s> ${fmtMs(r.est_cost_ms)}${r.note ? ` (${esc(r.note)})` : ""}`).join(" · ")}</div>`;
}

/* ---------- running queries ---------- */

// The URL hash mirrors the query and tab (#q=red+mug&tab=plan), so searches can be shared or bookmarked.
function syncHash() {
  const p = new URLSearchParams();
  if ($("q").value.trim()) p.set("q", $("q").value.trim());
  const tab = document.querySelector(".tabs button.active")?.dataset.tab;
  if (tab && tab !== "results") p.set("tab", tab);
  const h = p.toString();
  history.replaceState(null, "", h ? `#${h}` : location.pathname + location.search);
}

function show(data, label) {
  renderResults(data);
  renderPlan(data);
  syncHash();
  if (state.pending) {
    const { tab, open } = state.pending;
    state.pending = null;
    if (tab) selectTab(tab);
    if (open != null && open !== "") openDetail(Number(open));
  }
  if (data.omniql) $("omniql").value = data.omniql;
  const browsing = !$("q").value.trim() && !state.image && label !== "OmniQL";
  const total = state.info?.stats?.docs;
  status(browsing
    ? `Browsing <b>${data.hits.length}</b> of ${total ?? "?"} documents · ${data.timings.total_ms.toFixed(1)} ms`
    : `<b>${data.hits.length}</b> results · ${esc(label)} · <b>${data.timings.total_ms.toFixed(1)} ms</b>`);
}

async function search(ev) {
  ev?.preventDefault();
  if (!state.coll) return;
  const seq = ++state.seq;
  const fd = new FormData();
  fd.append("text", $("q").value);
  fd.append("filters", JSON.stringify(collectFilters()));
  fd.append("fusion", $("fusion").value);
  fd.append("limit", $("limit").value || "12");
  fd.append("strategy", $("strategy").value);
  fd.append("topk", $("topk").value);
  if (state.image) fd.append("image", state.image);
  status("Searching…");
  renderSkeleton();
  try {
    const data = await api(`${collBase()}/search`, { method: "POST", body: fd });
    if (seq !== state.seq) return;
    show(data, data.plan.branches.length ? `${data.plan.fuse.method.toUpperCase()} fusion` : "filter only");
  } catch (e) {
    if (seq !== state.seq) return;
    status(e.message, true);
    $("results").innerHTML = "";
  }
}

async function runQL() {
  const seq = ++state.seq;
  status("Running…");
  try {
    const data = await api(`${collBase()}/query`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: $("omniql").value }),
    });
    if (seq !== state.seq) return;
    show(data, "OmniQL");
    if (!data.hits.length && /^\s*EXPLAIN/i.test($("omniql").value)) selectTab("plan");
  } catch (e) {
    if (seq === state.seq) status(e.message, true);
  }
}

/* ---------- image query ---------- */

function setImage(file) {
  if (file && !file.type.startsWith("image/")) { status("Only image files can be used as a query.", true); return; }
  state.image = file;
  $("imgchip").hidden = !file;
  if (file) $("preview").src = URL.createObjectURL(file);
  search();
}

function selectTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === name);
    b.setAttribute("aria-selected", b.dataset.tab === name);
  });
  document.querySelectorAll(".tab").forEach((t) => { t.hidden = t.id !== `tab-${name}`; });
  syncHash();
}

function effectiveTheme() {
  const t = document.documentElement.dataset.theme;
  if (t) return t;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function init() {
  $("searchform").addEventListener("submit", search);
  $("runql").addEventListener("click", runQL);
  $("omniql").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); runQL(); } });
  $("collection").addEventListener("change", (e) => selectCollection(e.target.value).catch((err) => status(err.message, true)));
  $("clearfilters").addEventListener("click", clearFilters);
  $("togglefilters").addEventListener("click", () => $("panel").classList.toggle("open"));
  for (const id of ["fusion", "strategy", "topk", "limit"]) $(id).addEventListener("change", () => search());
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => selectTab(b.dataset.tab)));
  document.querySelectorAll(".sugg").forEach((b) => b.addEventListener("click", () => { $("q").value = b.textContent; selectTab("results"); search(); }));

  $("theme").addEventListener("click", () => {
    const next = effectiveTheme() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("omnisearch-theme", next); } catch (e) { /* storage unavailable */ }
  });

  // Result cards open the detail dialog; the empty state can clear filters.
  $("results").addEventListener("click", (e) => {
    if (e.target.closest("[data-action=clear]")) return clearFilters();
    const card = e.target.closest(".card[data-i]");
    if (card) openDetail(Number(card.dataset.i));
  });
  $("results").addEventListener("keydown", (e) => {
    const card = e.target.closest(".card[data-i]");
    if (card && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); openDetail(Number(card.dataset.i)); }
  });
  const dlg = $("detail");
  dlg.addEventListener("click", (e) => { if (e.target === dlg || e.target.closest("[data-action=close]")) dlg.close(); });
  dlg.addEventListener("close", () => { $("detail-body").innerHTML = ""; });

  // Image query: attach button, or drop an image anywhere on the page.
  $("attach").addEventListener("click", () => $("image").click());
  $("image").addEventListener("change", (e) => setImage(e.target.files[0] || null));
  $("clearimg").addEventListener("click", () => { $("image").value = ""; setImage(null); });
  let dragDepth = 0;
  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
  window.addEventListener("dragenter", (e) => { if (hasFiles(e)) { dragDepth++; $("dropzone").hidden = false; } });
  window.addEventListener("dragleave", (e) => { if (hasFiles(e) && --dragDepth <= 0) { dragDepth = 0; $("dropzone").hidden = true; } });
  window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault(); dragDepth = 0; $("dropzone").hidden = true;
    if (e.dataTransfer.files[0]) setImage(e.dataTransfer.files[0]);
  });

  // "/" focuses the search box from anywhere outside a text field.
  document.addEventListener("keydown", (e) => {
    if (e.key !== "/" || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target.closest("input, textarea, select, [contenteditable]") || dlg.open) return;
    e.preventDefault();
    $("q").focus();
    $("q").select();
  });

  const h = new URLSearchParams(location.hash.slice(1));
  if (h.get("q")) $("q").value = h.get("q");
  state.pending = { tab: h.get("tab"), open: h.get("open") };

  loadCollections().catch((e) => status(e.message, true));
}

init();
