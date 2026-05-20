// CVE Spectral Search Engine — frontend app.js
// Fixes applied:
//   1. refreshHealth() was missing its closing brace (broke entire app silently)
//   2. tab-drift button was never wired to switchView
//   3. switchView("drift") did not call loadDriftView()
//   4. /api/drift/lambdas returns .lambdas not .eigenvalues — added field fallback
//   5. drift panel was embedded inside audit; promoted to its own top-level view

const $ = (sel) => document.querySelector(sel);

const DEFAULT_TAU = 0.75;
const DEFAULT_LAM = 0.7;

const state = {
  lastResults: [],
  searchQuery: "",
  recentSearches: [],
};

// ─── UTILITIES ───────────────────────────────────────────────────────────────

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`);
  return res.json();
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

function setText(sel, text) {
  const el = $(sel);
  if (el) el.textContent = text;
}

function highlightQuery(text, query) {
  if (!query) return escapeHtml(text);
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const regex = new RegExp(`(${escaped})`, "gi");
  return escapeHtml(text).replace(regex, '<mark class="prompt-highlight">$1</mark>');
}

// ─── HEALTH ──────────────────────────────────────────────────────────────────
// FIX 1: was missing closing brace — the entire function was never closed,
// causing a syntax error that silently prevented the whole script from running.
async function refreshHealth() {
  const el = $("#health");
  try {
    const h = await api("/api/health");
    el.textContent = `zarr=${h.zarr_available} · arro=${h.arrowspace_backend} · roots=${h.data_roots.join(",") || "—"}`;
    el.className = "health ok";
  } catch (e) {
    el.textContent = `health: ${e.message}`;
    el.className = "health err";
  }
} // <-- this closing brace was missing in the original

// ─── TAB / VIEW SWITCHER ─────────────────────────────────────────────────────
// FIX 2+3: tab-drift was not wired, and switchView never called loadDriftView
function switchView(name) {
  document.querySelectorAll(".view").forEach(v => v.classList.add("hidden"));
  document.querySelectorAll(".tab-button").forEach(b => b.classList.remove("active"));
  const view = document.getElementById(`${name}-view`);
  const tab  = document.getElementById(`tab-${name}`);
  if (view) view.classList.remove("hidden");
  if (tab)  tab.classList.add("active");
  if (name === "audit") loadAuditPanel();
  if (name === "drift") loadDriftView(); // FIX 3
}

// ─── SEARCH ──────────────────────────────────────────────────────────────────

async function runSearch() {
  const query = ($("#filter")?.value ?? "").trim();
  state.searchQuery = query;

  if (!query) {
    $("#grid").innerHTML = `
      <div class="welcome-screen">
        <h2>CVE Spectral Search</h2>
        <p>Try queries like "buffer overflow in network daemon" or "remote code execution via SQL injection".</p>
      </div>`;
    return;
  }

  try {
    setText("#health", "Searching...");
    const alpha    = Number($("#alpha-slider")?.value    ?? 0.6);
    const salience = Number($("#salience-slider")?.value ?? 0.3);
    $("#grid").innerHTML = `<div class="loading-screen"><div class="loader"></div><p>Searching CVE spectral space…</p></div>`;
    const startedAt = performance.now();
    const result = await api("/api/prompts/nl_search", {
      method: "POST",
      body: JSON.stringify({
        query,
        k: Number($("#topk-select")?.value || 19),
        tau: DEFAULT_TAU, alpha, lam: DEFAULT_LAM, salience,
      }),
    });
    const latencyMs = Math.round(performance.now() - startedAt);
    renderPromptResults(result.results || [], { latencyMs, resultCount: result.result_count || 0, alpha, salience });
    await renderSearchVisualizations(result.results || []);
    const healthEl = $("#health");
    if (healthEl) { healthEl.textContent = "CVE Ready"; healthEl.className = "health ok"; }
    setText("#search-mode-label", `\u03b1 ${alpha.toFixed(2)} \u00b7 sal ${salience.toFixed(2)}`);
    setText("#search-hint", `${result.result_count || 0} CVE results`);
    if (query && !state.recentSearches.includes(query)) {
      state.recentSearches.unshift(query);
      state.recentSearches = state.recentSearches.slice(0, 8);
      renderRecentSearches();
    }
  } catch (e) {
    const healthEl = $("#health");
    if (healthEl) { healthEl.textContent = "Search Error"; healthEl.classList.add("err"); }
    $("#grid").innerHTML = `<div class="error-screen"><h2>Search failed</h2><p>${e.message}</p></div>`;
    wirePromptCards();
  }
}

function renderRecentSearches() {
  const el = $("#recent-searches");
  if (!el) return;
  if (!state.recentSearches.length) {
    el.innerHTML = `<span class="signal-empty">No recent searches</span>`;
    return;
  }
  el.innerHTML = state.recentSearches.map(q =>
    `<button class="recent-search-chip" type="button">${escapeHtml(q)}</button>`
  ).join("");
  el.querySelectorAll(".recent-search-chip").forEach(btn => {
    btn.addEventListener("click", () => {
      const filter = $("#filter");
      if (filter) { filter.value = btn.textContent; runSearch(); }
    });
  });
}

function renderPromptResults(results, analytics = {}) {
  state.lastResults = results;
  if (!results.length) {
    $("#grid").innerHTML = `<div class="welcome-screen"><h2>No results</h2><p>No CVEs matched your query.</p></div>`;
    return;
  }
  $("#grid").innerHTML = `
    <div class="prompt-results">
      <div class="search-analytics">
        <div><span>Latency</span><strong>${analytics.latencyMs ?? "—"} ms</strong></div>
        <div><span>Results</span><strong>${analytics.resultCount ?? results.length}</strong></div>
        <div><span>Alpha</span><strong>${analytics.alpha?.toFixed?.(2) ?? "—"}</strong></div>
        <div><span>Salience</span><strong>${analytics.salience?.toFixed?.(2) ?? "—"}</strong></div>
      </div>
      ${results.map((item, index) => renderPromptCard(item, index)).join("")}
    </div>`;
  wirePromptCards();
}

function renderPromptCard(item, index) {
  const content = item.content || item.body || "No content";
  return `
    <div class="prompt-result-card" data-index="${index}">
      <div class="prompt-result-header">
        <strong>${escapeHtml(item.title || item.id || "Untitled CVE")}</strong>
        <div class="prompt-score-wrap">
          <span class="prompt-score">Score: ${(item.score ?? 0).toFixed(4)}</span>
          <div class="prompt-score-bar"><div class="prompt-score-fill" style="width:${Math.min(100, (item.score ?? 0) * 100)}%"></div></div>
        </div>
      </div>
      <p class="prompt-content">${highlightQuery(content, state.searchQuery)}</p>
      <div class="prompt-card-actions">
        <button class="prompt-toggle" type="button">Expand</button>
        <button class="prompt-copy-btn" type="button" data-copy="${escapeHtml(content)}">Copy</button>
      </div>
      <div class="prompt-result-meta">
        <span>ID: ${item.id ?? "—"}</span>
        <span>Salience: ${(item.salience ?? 0).toFixed(3)}</span>
        <span>Upvotes: ${item.upvotes ?? 0}</span>
        <span>Views: ${item.views ?? 0}</span>
      </div>
    </div>`;
}

function wirePromptCards() {
  document.querySelectorAll(".prompt-result-card").forEach((card) => {
    card.classList.add("collapsed");
    const btn = card.querySelector(".prompt-toggle");
    const idx = Number(card.dataset.index);
    card.addEventListener("dblclick", () => openPromptModal(state.lastResults[idx]));
    if (!btn) return;
    btn.addEventListener("click", () => {
      card.classList.toggle("collapsed");
      btn.textContent = card.classList.contains("collapsed") ? "Expand" : "Collapse";
    });
  });
  document.querySelectorAll(".prompt-copy-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      navigator.clipboard?.writeText(btn.dataset.copy || "").catch(() => {});
      btn.textContent = "Copied!";
      setTimeout(() => { btn.textContent = "Copy"; }, 1500);
    });
  });
}

function openPromptModal(item) {
  const modal = $("#prompt-modal");
  if (!modal) return;
  const body = $("#prompt-modal-body");
  const content = item?.content || item?.body || "No content";
  body.innerHTML = `
    <h2 class="prompt-modal-title">${escapeHtml(item?.title || item?.id || "CVE")}</h2>
    <div class="prompt-modal-text">${highlightQuery(content, state.searchQuery)}</div>
    <div class="prompt-modal-meta">
      <div class="prompt-modal-chip">Score ${(item?.score ?? 0).toFixed(4)}</div>
      <div class="prompt-modal-chip">Salience ${(item?.salience ?? 0).toFixed(3)}</div>
      <div class="prompt-modal-chip">Upvotes ${item?.upvotes ?? 0}</div>
      <div class="prompt-modal-chip">Views ${item?.views ?? 0}</div>
      <div class="prompt-modal-chip">${item?.id ?? "—"}</div>
    </div>`;
  modal.classList.remove("hidden");
}

// ─── SEARCH VISUALISATIONS ───────────────────────────────────────────────────

async function renderSearchVisualizations(results) {
  if (!window.Plotly) return;
  renderQueryManifold({ pca_2d: [], degrees: [] }, new Set(results.map(r => r.id)), "#query-manifold");
  renderQueryLambdaChart(null, "#query-lambda-chart");
}

function renderQueryManifold(audit, matchedSet, targetSel) {
  const el = document.querySelector(targetSel);
  if (!el || !window.Plotly) return;
  const points  = Array.isArray(audit?.pca_2d)  ? audit.pca_2d  : [];
  const degrees = Array.isArray(audit?.degrees) ? audit.degrees : [];
  if (!points.length) {
    el.innerHTML = `<div class="manifold-empty-msg">Manifold data unavailable</div>`;
    return;
  }
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  const matched = points.map((_, i) => matchedSet.has(String(i)));
  Plotly.newPlot(el, [{
    x: xs, y: ys, mode: "markers", type: "scatter",
    marker: {
      color: matched.map((m, i) => m ? "#ef4444" : (degrees[i] ?? 0)),
      colorscale: "Viridis", size: 5, opacity: 0.7,
    },
    hoverinfo: "none",
  }], {
    paper_bgcolor: "transparent", plot_bgcolor: "rgba(15,23,42,0.75)",
    margin: { t: 10, r: 10, b: 30, l: 40 },
    xaxis: { color: "#9aa6bd", gridcolor: "rgba(255,255,255,0.06)", zeroline: false },
    yaxis: { color: "#9aa6bd", gridcolor: "rgba(255,255,255,0.06)", zeroline: false },
    font: { color: "#9aa6bd", size: 11 },
  }, { responsive: true, displayModeBar: false });
}

function renderQueryLambdaChart(source, targetSel) {
  const el = document.querySelector(targetSel);
  if (!el || !window.Plotly) return;
  const lambdas = Array.isArray(source?.lambdas)
    ? source.lambdas
    : Array.isArray(source?.eigenvalues) ? source.eigenvalues : [];
  if (!lambdas.length) {
    el.innerHTML = `<div class="manifold-empty-msg">Eigenvalue data unavailable</div>`;
    return;
  }
  const vals = lambdas.map(Number).filter(v => !isNaN(v));
  Plotly.newPlot(el, [{
    x: vals, type: "histogram", histnorm: "probability density", nbinsx: 50,
    marker: { color: "rgba(94,162,255,0.65)" }, name: "\u03bb density",
  }], {
    paper_bgcolor: "transparent", plot_bgcolor: "rgba(15,23,42,0.75)",
    margin: { t: 10, r: 10, b: 30, l: 40 },
    xaxis: { title: "\u03bb", color: "#9aa6bd", gridcolor: "rgba(255,255,255,0.06)", zeroline: false },
    yaxis: { title: "density", color: "#9aa6bd", gridcolor: "rgba(255,255,255,0.06)", zeroline: false },
    font: { color: "#9aa6bd", size: 11 },
    bargap: 0.05,
  }, { responsive: true, displayModeBar: false });
}

// ─── AUDIT ───────────────────────────────────────────────────────────────────

async function ensureLeafReady() {
  const health = await api("/api/prompts/health");
  if (health.status === "ready") return health;
  setText("#health", "Warming CVE engine\u2026");
  await api("/api/prompts/warm");
  for (let i = 0; i < 30; i++) {
    await new Promise(r => setTimeout(r, 2000));
    const polled = await api("/api/prompts/health");
    if (polled.status === "ready") return polled;
  }
  throw new Error("CVE engine warmup timeout");
}

async function loadAuditPanel() {
  try {
    setText("#health", "Loading audit\u2026");
    await ensureLeafReady();
    const [health, graph, lambdas, audit] = await Promise.all([
      api("/api/prompts/health"),
      api("/api/prompts/graph_laplacian"),
      api("/api/prompts/lambdas"),
      api("/api/prompts/audit"),
    ]);
    setText("#health", "CVE Ready");
    $("#health")?.classList.add("ok");
    renderAuditHealth(health);
    renderAuditGraph(graph);
    renderAuditLambdas(audit, lambdas);
    renderAuditStats(audit);
    renderAuditManifold(audit);
    renderAuditSpectral(audit, lambdas);
    renderAuditPCA(audit);
  } catch (e) {
    console.error(e);
    setText("#health", "CVE Audit Error");
    $("#health")?.classList.add("err");
    const ac = $("#audit-content");
    if (ac) ac.innerHTML = `<div class="error-screen"><h2>CVE Audit Error</h2><p>${e.message}</p></div>`;
  }
}

function renderAuditHealth(health) {
  const container = $("#audit-health");
  if (!container) return;
  container.innerHTML = `
    <div class="signal-card"><span class="signal-label">Status</span><strong>${health.status ?? "unknown"}</strong></div>
    <div class="signal-card"><span class="signal-label">Prompt Engine</span><strong>${health.prompt_engine_ready}</strong></div>
    <div class="signal-card"><span class="signal-label">Embedder</span><strong>${health.embedder_ready}</strong></div>
    <div class="signal-card"><span class="signal-label">Model</span><strong>${health.embedder_model ?? "—"}</strong></div>`;
}

function renderAuditGraph(graph) {
  const container = $("#audit-graph");
  if (!container) return;
  container.innerHTML = `
    <div class="signal-card"><span class="signal-label">Corpus Size</span><strong>${(graph.n_documents ?? "—").toLocaleString?.() ?? graph.n_documents}</strong></div>
    <div class="signal-card"><span class="signal-label">Embedding Dim</span><strong>${graph.embedding_dim ?? "—"}</strong></div>
    <div class="signal-card"><span class="signal-label">GL Nodes</span><strong>${graph.gl_nodes ?? "—"}</strong></div>
    <div class="signal-card"><span class="signal-label">GL Shape</span><strong>${Array.isArray(graph.gl_shape) ? graph.gl_shape.join(" \u00d7 ") : "—"}</strong></div>`;
}

function renderAuditStats(audit) {
  const container = $("#audit-stats");
  if (!container) return;
  const bp = audit?.build_params;
  container.innerHTML = `
    <div class="signal-card"><span class="signal-label">eps</span><strong>${bp?.eps ?? "—"}</strong></div>
    <div class="signal-card"><span class="signal-label">k</span><strong>${bp?.k ?? "—"}</strong></div>
    <div class="signal-card"><span class="signal-label">topk</span><strong>${bp?.topk ?? "—"}</strong></div>
    <div class="signal-card"><span class="signal-label">p</span><strong>${bp?.p ?? "—"}</strong></div>`;
  const bpEl = $("#build-params-display");
  if (bpEl && bp) {
    const fmt = v => (v === null || v === undefined ? "None" : String(v));
    bpEl.innerHTML = `<code>eps=${fmt(bp.eps)} \u00b7 k=${fmt(bp.k)} \u00b7 topk=${fmt(bp.topk)} \u00b7 p=${fmt(bp.p)} \u00b7 sigma=${fmt(bp.sigma)}</code>`;
  }
}

function renderAuditLambdas(audit, lambdas) {
  const container = $("#audit-lambdas");
  if (!container) return;
  const lambdaArr = Array.isArray(audit?.eigenvalues) ? audit.eigenvalues
    : Array.isArray(lambdas?.lambdas) ? lambdas.lambdas
    : Array.isArray(lambdas?.eigenvalues) ? lambdas.eigenvalues : [];
  const vals = lambdaArr.map(Number).filter(v => !isNaN(v));
  const sortedVals = [...vals].sort((a, b) => a - b);
  const median = _quantile(sortedVals, 0.5);
  container.innerHTML = `
    <div class="signal-card"><span class="signal-label">\u03bb count</span><strong>${vals.length}</strong></div>
    <div class="signal-card"><span class="signal-label">\u03bb max</span><strong>${vals.length ? Math.max(...vals).toFixed(4) : "—"}</strong></div>
    <div class="signal-card"><span class="signal-label">\u03bb median</span><strong>${vals.length ? median.toFixed(4) : "—"}</strong></div>
    <div class="signal-card"><span class="signal-label">Spectral gap</span><strong>${vals.length > 1 ? (sortedVals[1] - sortedVals[0]).toFixed(4) : "—"}</strong></div>`;
}

function renderAuditManifold(audit) {
  const el = document.getElementById("audit-query-manifold");
  if (!el || !window.Plotly) {
    if (el) el.innerHTML = `<div class="manifold-empty-msg">Plotly not available</div>`;
    return;
  }
  const lm = audit?.laplacian_manifold;
  if (lm && Array.isArray(lm.z_grid) && lm.z_grid.length) {
    renderServerLaplacianManifold(audit, lm, el);
    return;
  }
  const points  = Array.isArray(audit?.pca_2d)  ? audit.pca_2d  : [];
  if (!points.length) {
    el.innerHTML = `<div class="manifold-empty-msg">Graph Laplacian Manifold unavailable: missing pca_2d.</div>`;
    return;
  }
  renderQueryManifold(audit, new Set(), "#audit-query-manifold");
}

function renderServerLaplacianManifold(audit, lm, el) {
  const zMin = Number(lm.degree_p05), zMax = Number(lm.degree_p95);
  const colorscale = [[0,"#1e3a8a"],[0.25,"#2c5dff"],[0.5,"#f8fafc"],[0.75,"#fb7185"],[1,"#ef4444"]];
  Plotly.newPlot(el, [{
    x: lm.x_grid, y: lm.y_grid, z: lm.z_grid, type: "surface",
    colorscale, cmin: zMin, cmax: zMax, opacity: 0.94, showscale: true,
    lighting: { ambient: 0.65, diffuse: 0.85, specular: 0.18 }, name: "Laplacian surface",
  }, {
    x: lm.hub_x, y: lm.hub_y,
    z: lm.hub_x?.map(() => zMax + (zMax - zMin) * 0.06) ?? [],
    type: "scatter3d", mode: "markers", name: "High-degree hubs",
    marker: { size: 4.5, color: "#fbbf24", symbol: "diamond" },
  }], {
    paper_bgcolor: "rgba(0,0,0,0)", font: { color: "#cbd5e1" },
    scene: {
      xaxis: { title: lm.x_label || "PC1", gridcolor: "rgba(255,255,255,0.10)", color: "#cbd5e1" },
      yaxis: { title: lm.y_label || "PC2", gridcolor: "rgba(255,255,255,0.10)", color: "#cbd5e1" },
      zaxis: { title: "Curvature (L\u1d35\u1d35)", gridcolor: "rgba(255,255,255,0.10)", color: "#cbd5e1" },
      bgcolor: "rgba(15,23,42,0.92)", aspectmode: "manual",
      aspectratio: { x: 1.35, y: 1.1, z: 0.85 },
      camera: { eye: { x: 1.55, y: 1.55, z: 0.95 } },
    },
    margin: { l: 0, r: 0, t: 20, b: 0 },
  }, { responsive: true, displayModeBar: false });
}

function renderAuditSpectral(audit, lambdas) {
  const source = (audit && Array.isArray(audit.eigenvalues) && audit.eigenvalues.length)
    ? { lambdas: audit.eigenvalues }
    : lambdas;
  renderQueryLambdaChart(source, "#audit-spectral-fingerprint");
}

function renderAuditPCA(audit) {
  const canvas = document.getElementById("audit-pca");
  const tooltip = document.getElementById("audit-pca-tooltip");
  if (!canvas) return;
  const points = Array.isArray(audit?.pca_2d) ? audit.pca_2d : [];
  if (!points.length) {
    if (tooltip) tooltip.textContent = "PCA data unavailable";
    return;
  }
  const ctx = canvas.getContext("2d");
  const W = canvas.offsetWidth || 400, H = canvas.offsetHeight || 360;
  canvas.width = W; canvas.height = H;
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const pad = 20;
  const toCanvasX = x => pad + (x - minX) / ((maxX - minX) || 1) * (W - 2 * pad);
  const toCanvasY = y => H - pad - (y - minY) / ((maxY - minY) || 1) * (H - 2 * pad);
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "rgba(15,23,42,0.7)";
  ctx.fillRect(0, 0, W, H);
  for (const p of points) {
    ctx.beginPath();
    ctx.arc(toCanvasX(p[0]), toCanvasY(p[1]), 2.5, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(94,162,255,0.55)";
    ctx.fill();
  }
  if (tooltip) tooltip.textContent = `${points.length.toLocaleString()} points rendered`;
}

// ─── SPECTRAL DRIFT VIEW ─────────────────────────────────────────────────────
// FIX 4: /api/drift/lambdas returns .lambdas (not .eigenvalues) — use with fallback

async function loadDriftView() {
  const statusEl = $("#drift-view-status");
  const scoreEl  = $("#drift-view-score");

  ["drift-view-kpi-a","drift-view-kpi-b","drift-view-kpi-delta","drift-view-kpi-gap"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = `<div class="signal-card skeleton-card"><span class="signal-label">Loading\u2026</span></div>`;
  });
  if (statusEl) statusEl.textContent = "Fetching /api/drift/lambdas \u2026";

  try {
    const data = await api("/api/drift/lambdas");
    _renderDriftView(data);
    if (statusEl) statusEl.textContent = "Drift data loaded successfully.";
  } catch (e) {
    if (statusEl) statusEl.textContent = `Endpoint unavailable: ${e.message} \u00b7 Showing synthetic demo data.`;
    if (scoreEl)  scoreEl.textContent   = "W\u2081: N/A";
    _renderDriftView({
      period_a: { label: "cve_99_14 (1999\u20132014) [demo]", lambdas: _syntheticLambdas(200, 0.05, 0.8) },
      period_b: { label: "cve_99_25 (1999\u20132025) [demo]", lambdas: _syntheticLambdas(200, 0.10, 1.2) },
      drift_score: null,
    });
  }
}

function _syntheticLambdas(n, mean, spread) {
  return Array.from({ length: n }, () => Math.abs(mean + (Math.random() - 0.5) * spread));
}

function _renderDriftView(data) {
  // FIX 4: primary field is .lambdas; .eigenvalues is the fallback
  const lambdasA   = data?.period_a?.lambdas    ?? data?.period_a?.eigenvalues    ?? [];
  const lambdasB   = data?.period_b?.lambdas    ?? data?.period_b?.eigenvalues    ?? [];
  const labelA     = data?.period_a?.label      ?? "Period A \u2014 cve_99_14 (1999\u20132014)";
  const labelB     = data?.period_b?.label      ?? "Period B \u2014 cve_99_25 (1999\u20132025)";
  const driftScore = typeof data?.drift_score === "number" ? data.drift_score : null;

  _renderDriftKPI("drift-view-kpi-a", [
    ["Period", labelA],
    ["\u03bb count", lambdasA.length.toLocaleString()],
    ["\u03bb max",   lambdasA.length ? Math.max(...lambdasA).toFixed(4) : "—"],
    ["\u03bb mean",  lambdasA.length ? (lambdasA.reduce((a,b) => a+b, 0) / lambdasA.length).toFixed(4) : "—"],
  ]);
  _renderDriftKPI("drift-view-kpi-b", [
    ["Period", labelB],
    ["\u03bb count", lambdasB.length.toLocaleString()],
    ["\u03bb max",   lambdasB.length ? Math.max(...lambdasB).toFixed(4) : "—"],
    ["\u03bb mean",  lambdasB.length ? (lambdasB.reduce((a,b) => a+b, 0) / lambdasB.length).toFixed(4) : "—"],
  ]);

  const ks = _computeKS(lambdasA, lambdasB);
  const w1 = driftScore !== null ? driftScore : _computeW1(lambdasA, lambdasB);
  const level      = w1 < 0.05 ? "Low" : w1 < 0.15 ? "Medium" : "High";
  const levelColor = w1 < 0.05 ? "var(--good)" : w1 < 0.15 ? "#f59e0b" : "var(--bad)";

  _renderDriftKPI("drift-view-kpi-delta", [
    ["W\u2081 Wasserstein", w1 !== null ? w1.toFixed(4) : "—"],
    ["KS statistic",    ks.toFixed(4)],
    ["Drift level",     level],
    ["Score source",    driftScore !== null ? "server" : "client-computed"],
  ]);

  const sortA = [...lambdasA].sort((a,b) => a-b);
  const sortB = [...lambdasB].sort((a,b) => a-b);
  _renderDriftKPI("drift-view-kpi-gap", [
    ["Spectral gap A", sortA.length > 1 ? (sortA[1] - sortA[0]).toFixed(4) : "—"],
    ["Spectral gap B", sortB.length > 1 ? (sortB[1] - sortB[0]).toFixed(4) : "—"],
    ["\u03bb\u2082 (A)", sortA.length > 1 ? sortA[1].toFixed(4) : "—"],
    ["\u03bb\u2082 (B)", sortB.length > 1 ? sortB[1].toFixed(4) : "—"],
  ]);

  const scoreEl = $("#drift-view-score");
  if (scoreEl) {
    scoreEl.textContent = `W\u2081 = ${w1 !== null ? w1.toFixed(4) : "N/A"}`;
    scoreEl.style.color = levelColor;
  }

  if (!window.Plotly) return;
  _plotDriftOverlay(lambdasA, lambdasB, labelA, labelB, "drift-view-lambda-overlay");
  _plotDriftECDF(lambdasA, lambdasB, labelA, labelB, "drift-view-ecdf");
}

function _renderDriftKPI(id, rows) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = rows.map(([label, val]) =>
    `<div class="signal-card"><span class="signal-label">${escapeHtml(label)}</span><strong>${escapeHtml(String(val))}</strong></div>`
  ).join("");
}

function _computeKS(a, b) {
  if (!a.length || !b.length) return 0;
  const sortA = [...a].sort((x,y) => x-y);
  const sortB = [...b].sort((x,y) => x-y);
  const allX  = [...sortA, ...sortB].sort((x,y) => x-y);
  let maxDiff = 0;
  for (const x of allX) {
    const cdfA = sortA.filter(v => v <= x).length / sortA.length;
    const cdfB = sortB.filter(v => v <= x).length / sortB.length;
    maxDiff = Math.max(maxDiff, Math.abs(cdfA - cdfB));
  }
  return maxDiff;
}

function _computeW1(a, b) {
  if (!a.length || !b.length) return 0;
  const sortA = [...a].sort((x,y) => x-y);
  const sortB = [...b].sort((x,y) => x-y);
  const n = Math.max(sortA.length, sortB.length);
  const interp = (arr, t) => {
    const pos = t * (arr.length - 1);
    const lo = Math.floor(pos), hi = Math.min(lo + 1, arr.length - 1);
    return arr[lo] + (pos - lo) * (arr[hi] - arr[lo]);
  };
  let sum = 0;
  for (let i = 0; i < n; i++) sum += Math.abs(interp(sortA, i/(n-1)) - interp(sortB, i/(n-1)));
  return sum / n;
}

function _plotDriftOverlay(lambdasA, lambdasB, labelA, labelB, targetId) {
  const el = document.getElementById(targetId);
  if (!el) return;
  Plotly.react(el, [
    {
      x: lambdasA, type: "histogram", name: labelA, opacity: 0.68,
      histnorm: "probability density", nbinsx: 60,
      marker: { color: "rgba(1,105,111,0.78)" },
    },
    {
      x: lambdasB, type: "histogram", name: labelB, opacity: 0.60,
      histnorm: "probability density", nbinsx: 60,
      marker: { color: "rgba(218,113,1,0.72)" },
    },
  ], {
    barmode: "overlay", bargap: 0.04,
    paper_bgcolor: "transparent", plot_bgcolor: "rgba(15,23,42,0.75)",
    font: { family: "Inter, sans-serif", size: 12, color: "#cbd5e1" },
    margin: { t: 36, r: 24, b: 52, l: 60 },
    xaxis: { title: "Eigenvalue (\u03bb)", gridcolor: "rgba(255,255,255,0.08)", zeroline: false, color: "#cbd5e1" },
    yaxis: { title: "Density",            gridcolor: "rgba(255,255,255,0.08)", zeroline: false, color: "#cbd5e1" },
    legend: { orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "right", x: 1, font: { color: "#cbd5e1" } },
    annotations: [{
      xref: "paper", yref: "paper", x: 0, y: 1.1,
      text: `<b>Spectral Drift \u2014 Eigenvalue Overlay</b> \u00b7 A: ${lambdasA.length.toLocaleString()} \u03bb \u00b7 B: ${lambdasB.length.toLocaleString()} \u03bb`,
      showarrow: false, align: "left", xanchor: "left", font: { color: "#e2e8f0", size: 12 },
    }],
  }, { responsive: true, displayModeBar: false });
}

function _plotDriftECDF(lambdasA, lambdasB, labelA, labelB, targetId) {
  const el = document.getElementById(targetId);
  if (!el) return;
  const ecdf = (arr) => {
    const sorted = [...arr].sort((a,b) => a-b);
    return { x: sorted, y: sorted.map((_, i) => (i + 1) / sorted.length) };
  };
  const ecA = ecdf(lambdasA), ecB = ecdf(lambdasB);
  Plotly.react(el, [
    { x: ecA.x, y: ecA.y, type: "scatter", mode: "lines", name: labelA, line: { color: "rgba(1,105,111,0.9)", width: 2 } },
    { x: ecB.x, y: ecB.y, type: "scatter", mode: "lines", name: labelB, line: { color: "rgba(218,113,1,0.9)", width: 2 } },
  ], {
    paper_bgcolor: "transparent", plot_bgcolor: "rgba(15,23,42,0.75)",
    font: { family: "Inter, sans-serif", size: 12, color: "#cbd5e1" },
    margin: { t: 36, r: 24, b: 52, l: 60 },
    xaxis: { title: "\u03bb", gridcolor: "rgba(255,255,255,0.08)", zeroline: false, color: "#cbd5e1" },
    yaxis: { title: "CDF",   gridcolor: "rgba(255,255,255,0.08)", zeroline: false, color: "#cbd5e1", range: [0, 1] },
    legend: { orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "right", x: 1, font: { color: "#cbd5e1" } },
    annotations: [{
      xref: "paper", yref: "paper", x: 0, y: 1.1,
      text: "<b>Cumulative Spectral Mass (ECDF)</b> \u00b7 vertical gap = KS statistic",
      showarrow: false, align: "left", xanchor: "left", font: { color: "#e2e8f0", size: 12 },
    }],
  }, { responsive: true, displayModeBar: false });
}

// ─── HELPERS ──────────────────────────────────────────────────────────────────

function _quantile(sorted, q) {
  if (!sorted.length) return 0;
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (base + 1 < sorted.length) return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  return sorted[base];
}

// ─── INIT ────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  // Tab wiring — FIX 2: tab-drift was never wired
  document.getElementById("tab-search")?.addEventListener("click", () => switchView("search"));
  document.getElementById("tab-audit") ?.addEventListener("click", () => switchView("audit"));
  document.getElementById("tab-drift") ?.addEventListener("click", () => switchView("drift"));

  // Search input
  const filterEl = document.getElementById("filter");
  if (filterEl) {
    let searchTimer = null;
    filterEl.addEventListener("input", () => {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(runSearch, 350);
    });
    filterEl.addEventListener("keydown", e => {
      if (e.key === "Enter") { clearTimeout(searchTimer); runSearch(); }
    });
  }

  // Sliders
  document.getElementById("alpha-slider")?.addEventListener("input", e => {
    setText("#alpha-value", Number(e.target.value).toFixed(2));
  });
  document.getElementById("salience-slider")?.addEventListener("input", e => {
    setText("#salience-value", Number(e.target.value).toFixed(2));
  });
  document.getElementById("topk-select")?.addEventListener("change", e => {
    setText("#topk-value", e.target.value);
  });

  // Buttons
  document.getElementById("refresh-audit-btn")?.addEventListener("click", loadAuditPanel);
  document.getElementById("refresh-drift-btn")?.addEventListener("click", loadDriftView);

  // Modal close
  document.getElementById("prompt-modal")?.addEventListener("click", e => {
    if (e.target === e.currentTarget || e.target.id === "prompt-modal-close") {
      e.currentTarget.classList.add("hidden");
    }
  });

  // Initial health ping
  refreshHealth();
  setInterval(refreshHealth, 30_000);
});
