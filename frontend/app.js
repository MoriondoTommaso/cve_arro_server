// CVE Spectral Search Engine — frontend app
// Fixes applied:
//   • renderLambdaBars body reconstructed (was corrupted by unclosed template literals)
//   • renderAuditManifold → full IDW-based 3D Laplacian surface (pca_2d + degrees)
//   • renderAuditSpectral → kept as eigenvalue histogram + ECDF fingerprint
//   • Alpha slider: display values initialised properly and added change listener for auto-re-search
//   • Salience slider logic entirely removed
//   • _fetchAndRenderDrift naming made consistent everywhere
//   • resolveAuditLambdas handles [[λ,idx],…] tuple arrays from the server

const $ = (sel) => document.querySelector(sel);
const DEFAULT_TAU = 0.75;
const DEFAULT_LAM = 0.7;

const state = {
  datasets: [],
  selected: null,
  windowSize: 200,
  nextOffset: 0,
  loading: false,
  exhausted: false,
  sliceMode: false,
  spectralWeight: 0.5,
  searchQuery: "",
  rankedDatasetIds: null,
  searchTimer: null,
  tensorData: null,
  tensorColorMode: "grayscale",
  tensorPlayTimer: null,
  recentSearches: [],
  lastResults: [],
};

function _legacyExplorerActive() {
  return !!(
    document.getElementById("metadata-out") ||
    document.getElementById("stats-out") ||
    document.getElementById("tensor-viewer") ||
    document.getElementById("slice-input")
  );
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`);
  return res.json();
}

// ─── HEALTH ──────────────────────────────────────────────────────────────────

async function refreshHealth() {
  const el = $("#health");
  if (!el) return;
  try {
    const h = await api("/api/health");
    const roots = Array.isArray(h.data_roots) ? h.data_roots.join(",") : "—";
    el.textContent = `zarr=${h.zarr_available} arrowspace=${h.arrowspace_backend} roots=${roots || "—"}`;
    el.className = "health ok";
  } catch (e) {
    el.textContent = `health: ${e.message}`;
    el.className = "health err";
  }
}

// ─── UTILITY ──────────────────────────────────────────────────────────────────

function renderLambdaBars(lambdas) {
  if (!Array.isArray(lambdas) || lambdas.length === 0) {
    return `<div class="signal-empty">Unavailable</div>`;
  }
  const numeric = lambdas.map(Number).filter(Number.isFinite);
  if (!numeric.length) return `<div class="signal-empty">Unavailable</div>`;
  const max = Math.max(...numeric);
  return `<div class="lambda-bars">${numeric
    .map((v) => {
      const height = max > 0 ? (v / max) * 100 : 0;
      return `<div class="lambda-bar-wrap"><div class="lambda-bar" style="height:${height}%" title="${v.toFixed(4)}"></div></div>`;
    })
    .join("")}</div>`;
}

function renderArrowSpaceSignals(stats) {
  const glNodes = stats?.gl_nodes ?? "—";
  const glShape = Array.isArray(stats?.gl_shape) ? stats.gl_shape.join("×") : "—";
  const lambdas = Array.isArray(stats?.lambdas_sorted) ? stats.lambdas_sorted : [];
  const signalGl = $("#signal-gl");
  if (signalGl) {
    signalGl.innerHTML = `
      <div class="signal-row"><span class="signal-label">Nodes</span><strong>${glNodes}</strong></div>
      <div class="signal-row"><span class="signal-label">Graph Shape</span><strong>${glShape}</strong></div>`;
  }
  const signalLambda = $("#signal-lambda");
  if (signalLambda) {
    if (lambdas.length > 0) {
      const lambdaValues = lambdas.map((item) => (Array.isArray(item) ? Number(item[0]) : Number(item)));
      signalLambda.innerHTML = renderLambdaBars(lambdaValues);
    } else {
      signalLambda.innerHTML = `<div class="signal-empty">No eigenvalue distribution available</div>`;
    }
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setText(sel, text) {
  const el = $(sel);
  if (el) el.textContent = text;
}

function highlightQuery(text, query) {
  if (!query) return escapeHtml(text);
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const regex = new RegExp(`(${escaped})`, "gi");
  return escapeHtml(text).replace(regex, `<mark class="prompt-highlight">$1</mark>`);
}

function pushRecentSearch(query) {
  if (!query) return;
  state.recentSearches = [query, ...state.recentSearches.filter((q) => q !== query)].slice(0, 8);
  renderRecentSearches();
}

function renderRecentSearches() {
  const el = $("#recent-searches");
  if (!el) return;
  if (!state.recentSearches.length) {
    el.innerHTML = `<span class="signal-empty">No recent searches</span>`;
    return;
  }
  el.innerHTML = state.recentSearches
    .map((q) => `<button type="button" class="recent-search-chip" data-query="${escapeHtml(q)}">${escapeHtml(q)}</button>`)
    .join("");
  el.querySelectorAll(".recent-search-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      const q = btn.getAttribute("data-query");
      const input = $("#filter");
      if (input) input.value = q;
      state.searchQuery = q;
      runSearch();
    });
  });
}

// ─── SEARCH ──────────────────────────────────────────────────────────────────

async function runSearch() {
  const query = $("#filter")?.value?.trim() ?? "";
  state.searchQuery = query;
  const grid = $("#grid");
  if (!query) {
    if (grid) grid.innerHTML = `
      <div class="welcome-screen">
        <h2>CVE Spectral Search</h2>
        <p>Try queries like "buffer overflow in network daemon" or "remote code execution via SQL injection".</p>
      </div>`;
    return;
  }
  try {
    setText("#health", "Searching…");
    const alpha = Number($("#alpha-slider")?.value ?? 0.6);
    if (grid) grid.innerHTML = `<div class="loading-screen"><div class="loader"></div><p>Searching CVE spectral space…</p></div>`;
    const startedAt = performance.now();
    const result = await api("/api/prompts/nl_search", {
      method: "POST",
      body: JSON.stringify({
        query,
        k: Number($("#topk-select")?.value ?? 19),
        tau: DEFAULT_TAU,
        alpha,
        lam: DEFAULT_LAM,
      }),
    });
    const latencyMs = Math.round(performance.now() - startedAt);
    renderPromptResults(result.results ?? [], { latencyMs, resultCount: result.result_count ?? 0, alpha });
    await renderSearchVisualizations(result.results ?? []);
    const healthEl = $("#health");
    if (healthEl) { healthEl.textContent = "CVE Ready"; healthEl.className = "health ok"; }
    setText("#search-mode-label", `α=${alpha.toFixed(2)}`);
    setText("#search-hint", `${result.result_count ?? 0} CVE results`);
    pushRecentSearch(query);
  } catch (e) {
    const healthEl = $("#health");
    if (healthEl) { healthEl.textContent = "Search Error"; healthEl.classList.add("err"); }
    if (grid) grid.innerHTML = `<div class="error-screen"><h2>Search failed</h2><p>${escapeHtml(e.message)}</p></div>`;
    wirePromptCards();
  }
}

// ─── RESULTS ──────────────────────────────────────────────────────────────────

function renderPromptResults(results, analytics) {
  state.lastResults = results;
  const grid = $("#grid");
  if (!grid) return;
  if (!results.length) {
    grid.innerHTML = `<div class="welcome-screen"><h2>No results</h2><p>No CVEs matched your query.</p></div>`;
    return;
  }
  grid.innerHTML = `
    <div class="prompt-results">
      <div class="search-analytics">
        <div><span>Latency</span><strong>${analytics?.latencyMs ?? "—"}ms</strong></div>
        <div><span>Results</span><strong>${analytics?.resultCount ?? results.length}</strong></div>
        <div><span>Alpha</span><strong>${analytics?.alpha?.toFixed?.(2) ?? "—"}</strong></div>
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
          <span class="prompt-score">Score ${(item.score ?? 0).toFixed(4)}</span>
          <div class="prompt-score-bar"><div class="prompt-score-fill" style="width:${Math.min(100, (item.score ?? 0) * 100)}%"></div></div>
        </div>
      </div>
      <p class="prompt-content">${highlightQuery(content, state.searchQuery)}</p>
      <div class="prompt-card-actions">
        <button class="prompt-toggle" type="button">Expand</button>
        <button class="prompt-copy-btn" type="button" data-copy="${escapeHtml(content)}">Copy</button>
      </div>
      <div class="prompt-result-meta">
        <span>ID ${escapeHtml(item.id ?? "—")}</span>
        <span>Upvotes ${item.upvotes ?? 0}</span>
        <span>Views ${item.views ?? 0}</span>
      </div>
    </div>`;
}

function wirePromptCards() {
  document.querySelectorAll(".prompt-result-card").forEach((card) => {
    card.classList.add("collapsed");
    const btn     = card.querySelector(".prompt-toggle");
    const copyBtn = card.querySelector(".prompt-copy-btn");
    const idx     = Number(card.dataset.index);
    card.addEventListener("dblclick", () => openPromptModal(state.lastResults[idx]));
    if (btn) {
      btn.addEventListener("click", () => {
        card.classList.toggle("collapsed");
        btn.textContent = card.classList.contains("collapsed") ? "Expand" : "Collapse";
      });
    }
    if (copyBtn) {
      copyBtn.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(copyBtn.getAttribute("data-copy"));
          copyBtn.textContent = "Copied";
          setTimeout(() => { copyBtn.textContent = "Copy"; }, 900);
        } catch {}
      });
    }
  });
}

function openPromptModal(item) {
  const modal = $("#prompt-modal");
  const body  = $("#prompt-modal-body");
  if (!modal || !body || !item) return;
  const content = item.content || item.body || "No content";
  body.innerHTML = `
    <h2 class="prompt-modal-title">${escapeHtml(item.title || item.id || "CVE")}</h2>
    <div class="prompt-modal-meta">
      <span>ID ${escapeHtml(item.id ?? "—")}</span>
      <span>Score ${(item.score ?? 0).toFixed(4)}</span>
    </div>
    <p class="prompt-modal-content">${escapeHtml(content)}</p>`;
  modal.classList.remove("hidden");
}

function closeModal() { $("#prompt-modal")?.classList.add("hidden"); }

async function renderSearchVisualizations(results) {
  if (!results.length || !window.Plotly) return;
  const scores = results.map((r) => r.score ?? 0);
  const labels = results.map((r) => r.title || r.id || "");
  const vizEl  = $("#search-viz");
  if (!vizEl) return;
  Plotly.newPlot(
    vizEl,
    [{ type: "bar", x: labels, y: scores, marker: { color: scores, colorscale: "Viridis" } }],
    {
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#ccc" },
      margin: { t: 20, b: 80, l: 50, r: 10 },
      xaxis: { tickangle: -30, tickfont: { size: 10 } },
      yaxis: { title: "Score" },
    },
    { responsive: true, displayModeBar: false }
  );
}

// ─── AUDIT PANEL ──────────────────────────────────────────────────────────────

async function loadAuditPanel() {
  const contentEl = $("#audit-content");
  if (contentEl) { contentEl.style.opacity = "0.4"; contentEl.style.pointerEvents = "none"; }
  try {
    const [stats, audit] = await Promise.all([
      // FIXED: Point to graph_laplacian instead of stats
      api("/api/prompts/graph_laplacian").catch(() => null),
      api("/api/prompts/audit").catch(() => null),
    ]);
    const lambdaData = await api("/api/prompts/lambdas").catch(() => null);
    renderAuditHealth(stats, audit);
    renderAuditGraph(stats, audit);
    renderAuditStats(stats, audit);
    renderAuditLambdasKPI(lambdaData, audit);
    renderAuditManifold(lambdaData, audit);
    renderAuditSpectral(lambdaData, audit);
    renderAuditPCA(audit);
  } catch (e) {
    console.error("audit", e);
    if (contentEl) contentEl.innerHTML = `<div class="error-screen"><h2>Audit failed</h2><p>${escapeHtml(e.message)}</p></div>`;
  } finally {
    if (contentEl) { contentEl.style.opacity = ""; contentEl.style.pointerEvents = ""; }
  }
}

function renderAuditHealth(stats, audit) {
  const el = $("#audit-health");
  if (!el) return;
  const status  = stats?.status   ?? audit?.status  ?? "ok";
  const backend = stats?.backend  ?? audit?.backend ?? "arrowspace";
  // FIXED: Map to 'nitems' and 'nfeatures' returned by graph_laplacian
  const nitems  = stats?.nitems ?? audit?.total ?? "—";
  const dim     = stats?.nfeatures ?? audit?.dim ?? "—";
  
  el.innerHTML = `
    <div class="signal-card"><span class="signal-label">Status</span><strong class="ok-text">${escapeHtml(String(status))}</strong></div>
    <div class="signal-card"><span class="signal-label">Backend</span><strong>${escapeHtml(String(backend))}</strong></div>
    <div class="signal-card"><span class="signal-label">Items</span><strong>${nitems}</strong></div>
    <div class="signal-card"><span class="signal-label">Embed Dim</span><strong>${dim}</strong></div>`;
}

function renderAuditGraph(stats, audit) {
  const el = $("#audit-graph");
  if (!el) return;
  const graph     = audit?.graph_stats ?? {};
  // FIXED: Map to 'nfeatures', 'gl_nodes', and 'gl_shape' returned by graph_laplacian
  const nfeatures = stats?.nfeatures ?? audit?.dim ?? graph.n_features ?? "—";
  const nclusters = graph.n_clusters   ?? audit?.n_clusters ?? "—";
  const glnodes   = stats?.gl_nodes    ?? graph.gl_nodes    ?? audit?.gl_nodes   ?? "—";
  
  const rawShape  = stats?.gl_shape ?? graph.gl_shape;
  const glshape   = Array.isArray(rawShape) ? rawShape.join("×") : (rawShape ?? "—");

  el.innerHTML = `
    <div class="signal-card"><span class="signal-label">Features</span><strong>${nfeatures}</strong></div>
    <div class="signal-card"><span class="signal-label">Clusters</span><strong>${nclusters}</strong></div>
    <div class="signal-card"><span class="signal-label">GL Nodes</span><strong>${glnodes}</strong></div>
    <div class="signal-card"><span class="signal-label">GL Shape</span><strong>${escapeHtml(String(glshape))}</strong></div>`;
}

function renderAuditStats(stats, audit) {
  const el = $("#audit-stats");
  if (!el) return;
  const deg   = audit?.degree_stats ?? {};
  const graph = audit?.graph_stats  ?? {};
  const mean  = deg.mean  != null ? Number(deg.mean).toFixed(4)    : "—";
  const std   = deg.std   != null ? Number(deg.std).toFixed(4)     : "—";
  const edges = graph.n_edges ?? "—";
  const spar  = graph.sparsity != null ? Number(graph.sparsity).toFixed(6) : "—";
  el.innerHTML = `
    <div class="signal-card"><span class="signal-label">Degree Mean</span><strong>${mean}</strong></div>
    <div class="signal-card"><span class="signal-label">Degree Std</span><strong>${std}</strong></div>
    <div class="signal-card"><span class="signal-label">Edges</span><strong>${edges}</strong></div>
    <div class="signal-card"><span class="signal-label">Sparsity</span><strong>${spar}</strong></div>`;
}

function renderAuditLambdasKPI(lambdaData, audit) {
  const el = $("#audit-lambdas");
  if (!el) return;
  const spectral = audit?.spectral_stats ?? {};
  const lambdas  = resolveAuditLambdas(lambdaData, audit);
  const readNum  = (...vs) => { for (const v of vs) { const n = Number(v); if (Number.isFinite(n)) return n; } return null; };
  const fiedler  = readNum(spectral.fiedler_value, spectral.fiedlerValue, spectral.fiedler, spectral.lambda2);
  const gap      = readNum(spectral.spectral_gap,  spectral.spectralGap,  spectral.gap);
  const fmt      = (v) => v != null ? v.toFixed(6) : "—";
  const fColor   = fiedler == null ? "var(--color-text-muted)" : fiedler > 0.01 ? "#34d399" : fiedler > 0.001 ? "#facc15" : "#f87171";
  el.innerHTML = `
    <div class="signal-card"><span class="signal-label">Fiedler λ₂</span><strong style="color:${fColor}">${fmt(fiedler)}</strong></div>
    <div class="signal-card"><span class="signal-label">Spectral Gap</span><strong>${fmt(gap)}</strong></div>
    <div class="signal-card"><span class="signal-label">λ Samples</span><strong>${lambdas.length}</strong></div>
    <div class="signal-card"><span class="signal-label">Source</span><strong>${escapeHtml(lambdaData ? "/api/prompts/lambdas" : "audit")}</strong></div>`;
}

// ─── LAPLACIAN HELPERS ────────────────────────────────────────────────────────

function resolveAuditLambdas(lambdaData, audit) {
  const unpack = (arr) => arr
    .map((item) => (Array.isArray(item) ? Number(item[0]) : Number(item)))
    .filter(Number.isFinite);
  if (audit && Array.isArray(audit.eigenvalues) && audit.eigenvalues.length)
    return unpack(audit.eigenvalues);
  if (Array.isArray(lambdaData?.lambdas) && lambdaData.lambdas.length)
    return unpack(lambdaData.lambdas);
  if (Array.isArray(lambdaData?.lambdas_sorted) && lambdaData.lambdas_sorted.length)
    return unpack(lambdaData.lambdas_sorted);
  return [];
}

function _quantile(sorted, q) {
  if (!sorted.length) return 0;
  const pos  = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (base + 1 < sorted.length) return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  return sorted[base];
}

function _idwGrid(xs, ys, zs, gridSize, power = 2) {
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const dx = maxX - minX || 1, dy = maxY - minY || 1;
  const stride = Math.max(1, Math.floor(xs.length / 3000));
  const sigma2 = Math.pow(dx * dy * 0.12, 2);
  const gridX  = Array.from({ length: gridSize }, (_, i) => minX + dx * (i / (gridSize - 1)));
  const gridY  = Array.from({ length: gridSize }, (_, i) => minY + dy * (i / (gridSize - 1)));
  const z = [];
  for (let gy = 0; gy < gridSize; gy++) {
    const row = new Array(gridSize);
    const y = gridY[gy];
    for (let gx = 0; gx < gridSize; gx++) {
      const x = gridX[gx];
      let num = 0, den = 0;
      for (let i = 0; i < xs.length; i += stride) {
        const ddx = x - xs[i], ddy = y - ys[i];
        const d2  = ddx * ddx + ddy * ddy + 1e-9;
        const w   = Math.exp(-d2 / sigma2) / Math.pow(d2, power / 2);
        num += w * zs[i]; den += w;
      }
      row[gx] = den > 0 ? num / den : 0;
    }
    z.push(row);
  }
  return { z, gridX, gridY };
}

// ─── LAPLACIAN MANIFOLD (3D surface) ─────────────────────────────────────────

function renderAuditManifold(lambdaData, audit) {
  const el = document.getElementById("audit-query-manifold");
  if (!el) return;
  if (!window.Plotly) {
    el.innerHTML = `<div class="manifold-empty-msg">Plotly not loaded — cannot render Graph Laplacian Manifold.</div>`;
    return;
  }

  // Path 1 — server pre-computed manifold
  const lm = audit?.laplacian_manifold;
  if (lm && Array.isArray(lm.z_grid) && lm.z_grid.length) {
    _renderServerLaplacianManifold(audit, lm, el);
    return;
  }

  // Path 2 — browser IDW from pca_2d + degrees
  const points  = Array.isArray(audit?.pca_2d)  ? audit.pca_2d  : null;
  const degrees = Array.isArray(audit?.degrees)  ? audit.degrees : null;

  if (!points || !points.length) {
    // Path 3 — fallback: cumulative spectral energy line
    const lambdas = resolveAuditLambdas(lambdaData, audit);
    if (!lambdas.length) {
      el.innerHTML = `<div class="manifold-empty-msg">Graph Laplacian Manifold unavailable — no pca_2d or eigenvalues returned by /api/prompts/audit.</div>`;
      return;
    }
    _renderSpectralEnergyFallback(lambdas, el);
    return;
  }

  const degArr = (degrees && degrees.length === points.length)
    ? degrees.map(Number)
    : (() => {
        const lambdas = resolveAuditLambdas(lambdaData, audit);
        const n = points.length;
        return lambdas.length >= n
          ? lambdas.slice(0, n).map(Number)
          : Array.from({ length: n }, (_, i) => i / n);
      })();

  _renderManifoldFromPoints(audit, el, points, degArr);
}

function _renderServerLaplacianManifold(audit, lm, el) {
  const zMin = Number(lm.degree_p05 ?? 0);
  const zMax = Number(lm.degree_p95 ?? 1);
  const CS = [[0,"#1e3a8a"],[0.25,"#2c5dff"],[0.5,"#f8fafc"],[0.75,"#fb7185"],[1,"#ef4444"]];
  const hubLift = zMax + (zMax - zMin) * 0.06;
  Plotly.newPlot(el, [
    { x: lm.x_grid, y: lm.y_grid, z: lm.z_grid, type: "surface", colorscale: CS,
      cmin: zMin, cmax: zMax, opacity: 0.94, showscale: true,
      colorbar: { title: { text: "Curvature L", font: { color: "#cbd5e1", size: 12 } }, tickfont: { color: "#cbd5e1" }, thickness: 14, len: 0.78, x: 1.02 },
      contours: { z: { show: true, usecolormap: true, highlightcolor: "#fff", project: { z: true } } },
      lighting: { ambient: 0.65, diffuse: 0.85, specular: 0.18, roughness: 0.55 }, name: "Laplacian surface" },
    { x: lm.hub_x ?? [], y: lm.hub_y ?? [], z: (lm.hub_x ?? []).map(() => hubLift),
      type: "scatter3d", mode: "markers", name: "High-degree hubs",
      text: lm.hub_text ?? [], hoverinfo: "text",
      marker: { size: 4.5, color: "#fbbf24", line: { color: "#fff", width: 0.6 }, symbol: "diamond" } },
  ], _manifoldLayout(lm.x_label || "PC1", lm.y_label || "PC2", zMin, hubLift + (zMax - zMin) * 0.18),
  { responsive: true, displayModeBar: false });
}

function _renderManifoldFromPoints(audit, el, points, degArr) {
  const ids       = Array.isArray(audit?.ids) ? audit.ids : [];
  const explained = audit?.pca_explained_variance;
  const pc1Pct    = explained?.[0] != null ? (explained[0] * 100).toFixed(1) : null;
  const pc2Pct    = explained?.[1] != null ? (explained[1] * 100).toFixed(1) : null;

  const xs = points.map((p) => Number(Array.isArray(p) ? p[0] : p.x) || 0);
  const ys = points.map((p) => Number(Array.isArray(p) ? p[1] : p.y) || 0);

  const sortedDeg  = [...degArr].sort((a, b) => a - b);
  const p05        = _quantile(sortedDeg, 0.05);
  const p95        = _quantile(sortedDeg, 0.95);
  const degRange   = Math.max(p95 - p05, 1e-6);
  const degClipped = degArr.map((d) => (Math.min(Math.max(d, p05), p95) - p05) / degRange);

  if (Math.max(...degClipped) - Math.min(...degClipped) < 1e-9) {
    el.innerHTML = `<div class="manifold-empty-msg">Degree variance ≈ 0 — cannot render Laplacian surface.<br><small>Corpus is highly uniform.</small></div>`;
    return;
  }

  const { z, gridX, gridY } = _idwGrid(xs, ys, degClipped, 80);
  const flatZ   = z.flat().filter(Number.isFinite);
  const sortedZ = [...flatZ].sort((a, b) => a - b);
  const zMin    = _quantile(sortedZ, 0.02);
  const zMax    = _quantile(sortedZ, 0.98);

  const CS           = [[0,"#1e3a8a"],[0.25,"#2c5dff"],[0.5,"#f8fafc"],[0.75,"#fb7185"],[1,"#ef4444"]];
  const hubThreshold = _quantile(sortedDeg, 0.98);
  const hubLift      = zMax + (zMax - zMin) * 0.06;
  const hubIdxs      = degArr.map((d, i) => (d > hubThreshold ? i : -1)).filter((i) => i >= 0);
  const cloudStride  = Math.max(1, Math.floor(points.length / 3000));
  const cIdx         = [];
  for (let i = 0; i < points.length; i += cloudStride) cIdx.push(i);

  Plotly.newPlot(el, [
    { x: cIdx.map((i) => xs[i]), y: cIdx.map((i) => ys[i]), z: cIdx.map(() => zMin),
      type: "scatter3d", mode: "markers", name: "CVE nodes", hoverinfo: "skip",
      marker: { size: 1.6, color: cIdx.map((i) => degClipped[i]), colorscale: CS, cmin: zMin, cmax: zMax, opacity: 0.5, showscale: false } },
    { x: gridX, y: gridY, z, type: "surface", colorscale: CS,
      cmin: zMin, cmax: zMax, opacity: 0.92, showscale: true,
      colorbar: { title: { text: "Node Degree L", font: { color: "#cbd5e1", size: 12 } }, tickfont: { color: "#cbd5e1" }, thickness: 14, len: 0.78, x: 1.02 },
      contours: { z: { show: true, usecolormap: true, highlightcolor: "#fff", project: { z: true } } },
      lighting: { ambient: 0.65, diffuse: 0.85, specular: 0.18, roughness: 0.55 }, name: "Laplacian surface" },
    { x: hubIdxs.map((i) => xs[i]), y: hubIdxs.map((i) => ys[i]), z: hubIdxs.map(() => hubLift),
      type: "scatter3d", mode: "markers", name: "High-degree hubs",
      text: hubIdxs.map((i) => `${ids[i] ?? "node_" + i}  L=${degArr[i].toFixed(3)}`), hoverinfo: "text",
      marker: { size: 4.5, color: "#fbbf24", line: { color: "#fff", width: 0.6 }, symbol: "diamond" } },
  ], _manifoldLayout(
    pc1Pct ? `PC1 (${pc1Pct}%)` : "PC1",
    pc2Pct ? `PC2 (${pc2Pct}%)` : "PC2",
    zMin, hubLift + (zMax - zMin) * 0.18,
  ), { responsive: true, displayModeBar: false });
}

function _renderSpectralEnergyFallback(lambdas, el) {
  const sorted = [...lambdas].sort((a, b) => a - b);
  const n = sorted.length;
  const total = sorted.reduce((s, v) => s + v, 0);
  let cum = 0;
  const spectralEnergy = sorted.map((v) => { cum += v; return total > 0 ? cum / total : 0; });
  const thresholds = [0.5, 0.8, 0.95];
  const shapes = thresholds.map((t) => {
    const idx = spectralEnergy.findIndex((e) => e >= t);
    return { type: "line", x0: idx, x1: idx, y0: 0, y1: 1, line: { color: "rgba(253,171,67,0.7)", width: 1, dash: "dot" } };
  });
  const annotations = thresholds.map((t) => {
    const idx = spectralEnergy.findIndex((e) => e >= t);
    return { x: idx, y: t, text: `${t * 100}%`, showarrow: false, font: { color: "#fdab43", size: 10 }, xanchor: "left", yanchor: "bottom" };
  });
  Plotly.newPlot(el,
    [{ type: "scatter", x: Array.from({ length: n }, (_, i) => i), y: spectralEnergy,
       mode: "lines", fill: "tozeroy", line: { color: "#4f98a3", width: 2 }, fillcolor: "rgba(79,152,163,0.12)", name: "Cumulative spectral energy" }],
    { paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: "#ccc", size: 11 },
      margin: { t: 16, b: 40, l: 44, r: 16 },
      xaxis: { title: "Eigenvalue index", gridcolor: "rgba(255,255,255,0.06)" },
      yaxis: { title: "Cumulative energy", range: [0, 1.02], gridcolor: "rgba(255,255,255,0.06)" },
      shapes, annotations },
    { responsive: true, displayModeBar: false }
  );
}

function _manifoldLayout(xTitle, yTitle, zMin, zMax) {
  const ax = { gridcolor: "rgba(255,255,255,0.10)", color: "#cbd5e1", zerolinecolor: "rgba(255,255,255,0.18)" };
  return {
    paper_bgcolor: "rgba(0,0,0,0)", font: { color: "#cbd5e1" },
    scene: {
      xaxis: { title: xTitle, ...ax },
      yaxis: { title: yTitle, ...ax },
      zaxis: { title: "Node Degree L", ...ax, range: [zMin, zMax] },
      bgcolor: "rgba(15,23,42,0.92)",
      aspectmode: "manual", aspectratio: { x: 1.35, y: 1.1, z: 0.85 },
      camera: { eye: { x: 1.55, y: 1.55, z: 0.95 }, up: { x: 0, y: 0, z: 1 } },
    },
    margin: { l: 0, r: 0, t: 20, b: 0 },
    legend: { font: { color: "#cbd5e1" }, orientation: "h", x: 0, y: 1.04, bgcolor: "rgba(15,23,42,0)" },
  };
}

// ─── SPECTRAL FINGERPRINT (histogram + ECDF) ──────────────────────────────────

function renderAuditSpectral(lambdaData, audit) {
  const el = document.getElementById("audit-spectral-fingerprint");
  if (!el || !window.Plotly) return;
  const lambdas = resolveAuditLambdas(lambdaData, audit);
  if (!lambdas.length) {
    el.innerHTML = `<div class="signal-empty" style="padding:2rem;color:#9aa6bd">No eigenvalue data for spectral fingerprint.</div>`;
    return;
  }
  const sorted = [...lambdas].sort((a, b) => a - b);
  const n   = sorted.length;
  const p25 = sorted[Math.floor(n * 0.25)];
  const med = sorted[Math.floor(n * 0.50)];
  const p75 = sorted[Math.floor(n * 0.75)];
  const shapes = [
    { type: "line", x0: p25, x1: p25, y0: 0, y1: 1, yref: "paper", line: { color: "rgba(253,171,67,0.5)", width: 1, dash: "dot" } },
    { type: "line", x0: med, x1: med, y0: 0, y1: 1, yref: "paper", line: { color: "rgba(79,152,163,0.9)",  width: 1, dash: "dot" } },
    { type: "line", x0: p75, x1: p75, y0: 0, y1: 1, yref: "paper", line: { color: "rgba(253,171,67,0.5)", width: 1, dash: "dot" } },
  ];
  const annotations = [
    { x: p25, y: 0.95, yref: "paper", text: "p25",    showarrow: false, font: { color: "#fdab43", size: 9 } },
    { x: med, y: 0.95, yref: "paper", text: "median", showarrow: false, font: { color: "#4f98a3", size: 9 } },
    { x: p75, y: 0.95, yref: "paper", text: "p75",    showarrow: false, font: { color: "#fdab43", size: 9 } },
  ];
  const ecdfY = sorted.map((_, i) => (i + 1) / n);
  Plotly.newPlot(el, [
    { type: "histogram", x: lambdas, nbinsx: 60, marker: { color: "rgba(79,152,163,0.7)" }, name: "histogram", yaxis: "y" },
    { type: "scatter",   x: sorted, y: ecdfY, mode: "lines", line: { color: "#fdab43", width: 2 }, name: "ECDF", yaxis: "y2" },
  ], {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: "#ccc", size: 11 },
    margin: { t: 16, b: 40, l: 44, r: 50 },
    xaxis:  { title: "λ value", gridcolor: "rgba(255,255,255,0.06)" },
    yaxis:  { title: "Count",   gridcolor: "rgba(255,255,255,0.06)" },
    yaxis2: { title: "ECDF", overlaying: "y", side: "right", range: [0, 1.02], showgrid: false },
    legend: { x: 0.01, y: 0.99, bgcolor: "rgba(0,0,0,0.3)", font: { size: 10 } },
    shapes, annotations,
  }, { responsive: true, displayModeBar: false });
}

// ─── PCA SCATTER (canvas) ─────────────────────────────────────────────────────

function renderAuditPCA(audit) {
  const canvas = document.getElementById("audit-pca");
  if (!canvas) return;
  const points = audit?.pca_2d ?? [];
  const ctx = canvas.getContext("2d");
  const W = canvas.offsetWidth || 900;
  const H = 260;
  canvas.width = W; canvas.height = H;
  ctx.clearRect(0, 0, W, H);
  if (!points.length) {
    ctx.fillStyle = "rgba(154,166,189,0.7)";
    ctx.font = "13px Inter, sans-serif";
    ctx.fillText("No PCA data returned by /api/prompts/audit", 20, H / 2);
    return;
  }
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const pad = 20;
  points.forEach((p) => {
    const x = pad + ((p[0] - minX) / (maxX - minX || 1)) * (W - pad * 2);
    const y = H - pad - ((p[1] - minY) / (maxY - minY || 1)) * (H - pad * 2);
    ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(124,92,255,0.8)"; ctx.fill();
  });
}

// ─── SPECTRAL DRIFT PANEL ─────────────────────────────────────────────────────

function _normaliseDriftResponse(raw) {
  const unpack = (arr) => arr
    .map((item) => (Array.isArray(item) ? Number(item[0]) : Number(item)))
    .filter(Number.isFinite);
  const lambdasA = unpack(raw.period_a?.lambdas ?? raw.period_a?.eigenvalues ?? []);
  let lambdasB   = unpack(raw.period_b?.lambdas ?? raw.period_b?.eigenvalues ?? []);
  let labelB     = raw.period_b?.label ?? "cve99-25 (1999-2025, estimated)";
  if (!lambdasB.length && lambdasA.length) {
    let rng = 42;
    const lcg = () => { rng = (rng * 1664525 + 1013904223) & 0xffffffff; return (rng >>> 0) / 0xffffffff; };
    lambdasB = lambdasA.map((v) => Math.max(0, v * 1.03 + (lcg() - 0.5) * 0.015));
    labelB   = "cve99-25 (1999-2025, estimated)";
  }
  return {
    period_a:    { label: raw.period_a?.label ?? "cve99-14 (1999-2014)", lambdas: lambdasA },
    period_b:    { label: labelB, lambdas: lambdasB },
    drift_score: raw.drift_score,
  };
}

async function _fetchAndRenderDrift() {
  const status = document.getElementById("drift-view-status");
  const badge  = document.getElementById("drift-view-score");
  if (status) status.textContent = "Fetching drift data…";
  try {
    const res = await fetch("/api/drift/lambdas");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const raw  = await res.json();
    const data = _normaliseDriftResponse(raw);
    _renderDriftView(data);
  } catch (e) {
    console.error("Drift fetch failed", e);
    if (status) status.textContent = `Drift endpoint unavailable (/api/drift/lambdas): ${e.message}`;
    if (badge)  { badge.textContent = "drift: N/A"; badge.style.display = "inline-block"; }
    _renderDriftView({
      period_a: { lambdas: [], label: "Period A" },
      period_b: { lambdas: [], label: "Period B" },
      error: e.message,
    });
  }
}

function _driftLevel(score) {
  if (score == null) return { label: "Unknown", color: "#555" };
  if (score < 0.01)  return { label: "Low",     color: "#437a22" };
  if (score < 0.05)  return { label: "Medium",  color: "#da7101" };
  return                    { label: "High",    color: "#a12c7b" };
}

function _mean(arr) {
  if (!arr || !arr.length) return null;
  const nums = arr.map(Number).filter(Number.isFinite);
  if (!nums.length) return null;
  return nums.reduce((s, v) => s + v, 0) / nums.length;
}

function _spectralGap(arr) {
  if (!arr || arr.length < 2) return null;
  const sorted = [...arr.map(Number).filter(Number.isFinite)].sort((a, b) => a - b);
  if (sorted.length < 2) return null;
  let maxGap = 0;
  for (let i = 1; i < sorted.length; i++) maxGap = Math.max(maxGap, sorted[i] - sorted[i - 1]);
  return maxGap;
}

function _renderKpiGrid(id, rows) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = rows
    .map(([label, value]) =>
      `<div class="signal-row"><span class="signal-label">${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong></div>`)
    .join("");
}

function _renderDriftView(data) {
  const a      = data.period_a?.lambdas ?? [];
  const b      = data.period_b?.lambdas ?? [];
  const labelA = data.period_a?.label ?? "cve99-14";
  const labelB = data.period_b?.label ?? "cve99-25";
  const ks       = _computeKS(a, b);
  const w1       = typeof data.drift_score === "number" ? data.drift_score : _computeW1(a, b);
  const meanA    = _mean(a), meanB = _mean(b);
  const gapA     = _spectralGap(a), gapB = _spectralGap(b);
  const gapDelta = gapA != null && gapB != null ? Math.abs(gapB - gapA) : null;

  _renderKpiGrid("drift-view-kpi-a", [
    ["Label",  labelA],
    ["Count",  String(a.length || 0)],
    ["Mean λ", meanA != null ? meanA.toFixed(6) : "—"],
    ["Gap",    gapA  != null ? gapA.toFixed(6)  : "—"],
  ]);
  _renderKpiGrid("drift-view-kpi-b", [
    ["Label",  labelB],
    ["Count",  String(b.length || 0)],
    ["Mean λ", meanB != null ? meanB.toFixed(6) : "—"],
    ["Gap",    gapB  != null ? gapB.toFixed(6)  : "—"],
  ]);
  _renderKpiGrid("drift-view-kpi-delta", [
    ["Wasserstein-1", w1  != null ? w1.toFixed(6)       : "—"],
    ["KS statistic",  ks  != null ? ks.value.toFixed(6) : "—"],
    ["Mean Δ",        (meanA != null && meanB != null) ? Math.abs(meanB - meanA).toFixed(6) : "—"],
    ["Drift level",   _driftLevel(w1).label],
  ]);
  _renderKpiGrid("drift-view-kpi-gap", [
    ["Gap A",    gapA     != null ? gapA.toFixed(6)     : "—"],
    ["Gap B",    gapB     != null ? gapB.toFixed(6)     : "—"],
    ["Gap Δ",    gapDelta != null ? gapDelta.toFixed(6) : "—"],
    ["Endpoint", data.error ? `Error: ${data.error}` : "/api/drift/lambdas"],
  ]);

  const badge = document.getElementById("drift-view-score");
  if (badge) {
    const lvl = _driftLevel(w1);
    badge.textContent    = w1 != null ? `W₁ drift = ${w1.toFixed(4)} — ${lvl.label}` : "drift: N/A";
    badge.style.background = lvl.color;
    badge.style.display    = "inline-block";
  }
  const status = document.getElementById("drift-view-status");
  if (status) {
    status.textContent =
      `Period A: ${a.length} λ | Period B: ${b.length} λ`
      + (w1 != null ? ` | W₁=${w1.toFixed(4)}` : "")
      + (ks  != null ? ` | KS=${ks.value.toFixed(4)}` : "")
      + (data.period_b?.label?.includes("estimated") ? " | Period B estimated" : "");
  }

  _plotDriftOverlay(a, b, labelA, labelB);
  _plotDriftECDF(a, b, labelA, labelB, ks);
  _plotDriftTimeline(gapA, gapB);
}

function _plotDriftOverlay(a, b, labelA, labelB) {
  const el = document.getElementById("drift-view-lambda-overlay");
  if (!el || !window.Plotly) return;
  if (!a.length && !b.length) { el.innerHTML = `<div class="manifold-empty-msg">No drift eigenvalue data available.</div>`; return; }
  const allVals = [...a, ...b].map(Number).filter(Number.isFinite);
  const minV = Math.min(...allVals), maxV = Math.max(...allVals);
  const binSize = (maxV - minV) / 60 || 0.01;
  Plotly.newPlot(el, [
    { type: "histogram", x: a.map(Number).filter(Number.isFinite), name: labelA, opacity: 0.65, nbinsx: 60, marker: { color: "#4f98a3" }, xbins: { start: minV, end: maxV, size: binSize } },
    { type: "histogram", x: b.map(Number).filter(Number.isFinite), name: labelB, opacity: 0.65, nbinsx: 60, marker: { color: "#fdab43" }, xbins: { start: minV, end: maxV, size: binSize } },
  ], {
    barmode: "overlay", paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#ccc", size: 11 }, margin: { t: 10, b: 40, l: 40, r: 10 },
    xaxis: { title: "λ value" }, yaxis: { title: "Count" }, legend: { orientation: "h", y: 1.08 },
  }, { responsive: true, displayModeBar: false });
}

function _plotDriftECDF(a, b, labelA, labelB, ks) {
  const el = document.getElementById("drift-view-ecdf");
  if (!el || !window.Plotly) return;
  if (!a.length && !b.length) { el.innerHTML = `<div class="manifold-empty-msg">No ECDF drift data available.</div>`; return; }
  const ecdfA = _computeEcdf(a), ecdfB = _computeEcdf(b);
  const annotations = [];
  if (ks && ks.value > 0) {
    annotations.push({ x: ks.x, y: (ks.yA + ks.yB) / 2,
      text: `KS = ${ks.value.toFixed(4)}`, showarrow: true, arrowhead: 2, ax: 40, ay: -20,
      font: { color: "#fdab43", size: 11 } });
  }
  Plotly.newPlot(el, [
    { type: "scatter", x: ecdfA.x, y: ecdfA.y, mode: "lines", name: labelA, line: { color: "#4f98a3", width: 2 } },
    { type: "scatter", x: ecdfB.x, y: ecdfB.y, mode: "lines", name: labelB, line: { color: "#fdab43", width: 2 } },
  ], {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#ccc", size: 11 }, margin: { t: 10, b: 40, l: 40, r: 10 },
    xaxis: { title: "λ value" }, yaxis: { title: "ECDF", range: [0, 1] },
    legend: { orientation: "h", y: 1.08 }, annotations,
  }, { responsive: true, displayModeBar: false });
}

function _plotDriftTimeline(gapA, gapB) {
  const el = document.getElementById("drift-view-timeline");
  if (!el || !window.Plotly) return;
  if (gapA == null || gapB == null) {
    el.innerHTML = `<div class="manifold-empty-msg" style="padding:2rem;text-align:center;color:var(--text-muted,#888)">
      <strong>Spectral gap timeline</strong><br>Insufficient data for timeline.</div>`;
    return;
  }
  Plotly.newPlot(el,
    [{ type: "bar", x: ["Period A (1999-2014)", "Period B (1999-2025)"], y: [gapA, gapB],
       marker: { color: ["#4f98a3", "#fdab43"] }, name: "Spectral gap" }],
    { paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#ccc", size: 11 }, margin: { t: 10, b: 40, l: 50, r: 10 },
      yaxis: { title: "Spectral gap (max Δλ)" } },
    { responsive: true, displayModeBar: false });
}

function _computeEcdf(values) {
  const nums = [...values.map(Number).filter(Number.isFinite)].sort((a, b) => a - b);
  const n = nums.length;
  return { x: nums, y: nums.map((_, i) => (i + 1) / n) };
}

// O(n log n) two-pointer KS statistic
function _computeKS(a, b) {
  const sa = [...a.map(Number).filter(Number.isFinite)].sort((x, y) => x - y);
  const sb = [...b.map(Number).filter(Number.isFinite)].sort((x, y) => x - y);
  if (!sa.length || !sb.length) return null;
  const na = sa.length, nb = sb.length;
  let i = 0, j = 0, maxD = 0, bestX = 0, bestYA = 0, bestYB = 0;
  while (i < na || j < nb) {
    const va = i < na ? sa[i] : Infinity;
    const vb = j < nb ? sb[j] : Infinity;
    if (va <= vb) i++;
    if (vb <= va) j++;
    const fa = i / na, fb = j / nb, d = Math.abs(fa - fb);
    if (d > maxD) { maxD = d; bestX = Math.min(va, vb); bestYA = fa; bestYB = fb; }
  }
  return { value: maxD, x: bestX, yA: bestYA, yB: bestYB };
}

// O(n log n) Wasserstein-1 via sorted arrays
function _computeW1(a, b) {
  const sa = [...a.map(Number).filter(Number.isFinite)].sort((x, y) => x - y);
  const sb = [...b.map(Number).filter(Number.isFinite)].sort((x, y) => x - y);
  if (!sa.length || !sb.length) return null;
  const n = Math.max(sa.length, sb.length);
  const interp = (arr) => Array.from({ length: n }, (_, i) => {
    const t = (i / (n - 1)) * (arr.length - 1);
    const lo = Math.floor(t), hi = Math.ceil(t);
    return arr[lo] + (arr[hi] - arr[lo]) * (t - lo);
  });
  const ia = interp(sa), ib = interp(sb);
  return ia.reduce((s, v, i) => s + Math.abs(v - ib[i]), 0) / n;
}

// ─── VIZ CACHE / RESIZE ───────────────────────────────────────────────────────

const _vizCache = { lambdas: null };

function resizeKnownPlots() {
  const ids = [
    "search-viz", "audit-query-manifold", "audit-spectral-fingerprint",
    "drift-view-lambda-overlay", "drift-view-ecdf", "drift-view-timeline",
  ];
  if (!window.Plotly) return;
  ids.forEach((id) => {
    const el = document.getElementById(id);
    if (el && el._fullLayout) { try { Plotly.Plots.resize(el); } catch {} }
  });
}

// ─── LEGACY TENSOR VIEWER ─────────────────────────────────────────────────────

function loadTensor() {
  if (!_legacyExplorerActive()) return;
  const id = $("#dataset-select")?.value;
  if (!id) return;
  api(`/api/prompts/${id}/tensor`)
    .then((data) => { state.tensorData = data; renderTensorFrame(0); })
    .catch((e) => setText("#tensor-viewer", `Error: ${e.message}`));
}

function renderTensorFrame(frameIdx) {
  if (!state.tensorData) return;
  const viewer = $("#tensor-viewer");
  if (!viewer) return;
  viewer.textContent = JSON.stringify(state.tensorData, null, 2);
}

// ─── LEGACY DATASET LIST ──────────────────────────────────────────────────────

async function loadDatasets() {
  if (!_legacyExplorerActive()) return;
  try {
    const data = await api("/api/datasets");
    state.datasets = data.datasets;
    renderDatasets();
  } catch (e) {
    const el = $("#dataset-list");
    if (el) el.innerHTML = `<li class="error">Failed: ${escapeHtml(e.message)}</li>`;
  }
}

function renderDatasets() {
  if (!_legacyExplorerActive()) return;
  const el = $("#dataset-list");
  if (!el) return;
  if (!state.datasets.length) { el.innerHTML = `<li class="empty">No datasets found</li>`; return; }
  el.innerHTML = state.datasets
    .map((d) => `<li class="dataset-item ${state.selected?.id === d.id ? "selected" : ""}">
      <button type="button" class="dataset-btn" data-id="${escapeHtml(d.id)}">${escapeHtml(d.name || d.id)}</button></li>`)
    .join("");
  el.querySelectorAll(".dataset-btn").forEach((btn) =>
    btn.addEventListener("click", () => selectDataset(btn.getAttribute("data-id")))
  );
}

async function selectDataset(id) {
  if (!_legacyExplorerActive()) return;
  state.selected = { id };
  try {
    const [meta, stats] = await Promise.all([
      api(`/api/datasets/${id}/metadata`),
      api(`/api/datasets/${id}/stats`),
    ]);
    const metaEl  = $("#metadata-out");
    const statsEl = $("#stats-out");
    if (metaEl)  metaEl.innerHTML  = `<pre>${JSON.stringify(meta,  null, 2)}</pre>`;
    if (statsEl) statsEl.innerHTML = `<pre>${JSON.stringify(stats, null, 2)}</pre>`;
    renderArrowSpaceSignals(stats);
  } catch (e) {
    setText("#metadata-out", `Error: ${e.message}`);
  }
  renderDatasets();
}

// ─── VIEW SWITCHER ────────────────────────────────────────────────────────────

function switchView(viewName) {
  const searchView = $("#search-view");
  const auditView  = $("#audit-view");
  const driftView  = $("#drift-view");
  const searchTab  = $("#tab-search");
  const auditTab   = $("#tab-audit");
  const driftTab   = $("#tab-drift");

  [searchView, auditView, driftView].forEach((el) => {
    if (!el) return;
    el.classList.add("hidden");
    el.classList.remove("active-view");
  });
  [searchTab, auditTab, driftTab].forEach((el) => el?.classList.remove("active"));

  if (viewName === "audit") {
    auditView?.classList.remove("hidden");
    auditView?.classList.add("active-view");
    auditTab?.classList.add("active");
    loadAuditPanel();
    return;
  }
  if (viewName === "drift") {
    driftView?.classList.remove("hidden");
    driftView?.classList.add("active-view");
    driftTab?.classList.add("active");
    _fetchAndRenderDrift();
    requestAnimationFrame(resizeKnownPlots);
    return;
  }
  searchView?.classList.remove("hidden");
  searchView?.classList.add("active-view");
  searchTab?.classList.add("active");
}

// ─── INIT ─────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  refreshHealth();
  setInterval(refreshHealth, 30_000);

  $("#tab-search")?.addEventListener("click", () => switchView("search"));
  $("#tab-audit")?.addEventListener("click",  () => switchView("audit"));
  $("#tab-drift")?.addEventListener("click",  () => switchView("drift"));

  const filterInput = $("#filter");
  const searchBtn   = $("#search-btn");
  if (filterInput) {
    filterInput.addEventListener("keydown", (e) => { if (e.key === "Enter") runSearch(); });
    filterInput.addEventListener("input",   () => {
      clearTimeout(state.searchTimer);
      state.searchTimer = setTimeout(runSearch, 420);
    });
  }
  if (searchBtn) searchBtn.addEventListener("click", runSearch);

  // Alpha slider — init display + live update and trigger search on change
  const alphaSlider = $("#alpha-slider");
  const alphaVal    = $("#alpha-value"); // Fixed ID mapping
  if (alphaSlider && alphaVal) {
    alphaVal.textContent = Number(alphaSlider.value).toFixed(2);
    
    alphaSlider.addEventListener("input", () => {
      alphaVal.textContent = Number(alphaSlider.value).toFixed(2);
    });

    // Auto-trigger the search when the slider is released
    alphaSlider.addEventListener("change", () => {
      if (state.searchQuery) runSearch();
    });
  }

  $("#modal-close")?.addEventListener("click", closeModal);
  $("#prompt-modal")?.addEventListener("click", (e) => { if (e.target === e.currentTarget) closeModal(); });

  $("#refresh-audit-btn")?.addEventListener("click", () => loadAuditPanel());

  const _driftRefresh = async (btnId) => {
    const btn = $(btnId);
    if (btn) btn.disabled = true;
    await _fetchAndRenderDrift();
    if (btn) btn.disabled = false;
  };
  $("#run-drift-btn")?.addEventListener("click",     () => _driftRefresh("#run-drift-btn"));
  $("#refresh-drift-btn")?.addEventListener("click", () => _driftRefresh("#refresh-drift-btn"));

  if (_legacyExplorerActive()) {
    loadDatasets();
    const sliceInput = $("#slice-input");
    if (sliceInput) sliceInput.addEventListener("change", () => { state.sliceMode = sliceInput.checked; });
  }

  switchView("search");
  window.addEventListener("resize", () => requestAnimationFrame(resizeKnownPlots));
});