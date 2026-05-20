// CVE Spectral Search Engine — frontend app
// All LEAF / Prompt Kaban strings replaced with CVE domain equivalents.
// Spectral drift panel added.
// Calls /api/drift/lambdas expecting:
//   { period_a: { eigenvalues: number[], label: string },
//     period_b: { eigenvalues: number[], label: string },
//     drift_score: number }   (drift_score is optional)

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

// FIX 1: properly closed function
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
  const glNodes = stats.gl_nodes ?? "—";
  const glShape = Array.isArray(stats.gl_shape) ? stats.gl_shape.join(" × ") : "—";
  const lambdas = Array.isArray(stats.lambdas_sorted) ? stats.lambdas_sorted : [];
  const signalGl = $("#signal-gl");
  if (signalGl) {
    signalGl.innerHTML = `
      <div class="signal-row"><span>Nodes</span><strong>${glNodes}</strong></div>
      <div class="signal-row"><span>Graph Shape</span><strong>${glShape}</strong></div>`;
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
  return escapeHtml(text).replace(regex, '<mark class="prompt-highlight">$1</mark>');
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
      const q = btn.getAttribute("data-query") || "";
      const input = $("#filter");
      if (input) input.value = q;
      state.searchQuery = q;
      runSearch();
    });
  });
}

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
    const alpha = Number($("#alpha-slider")?.value ?? 0.6);
    const salience = Number($("#salience-slider")?.value ?? 0.3);
    $("#grid").innerHTML = `<div class="loading-screen"><div class="loader"></div><p>Searching CVE spectral space...</p></div>`;
    const startedAt = performance.now();
    const result = await api("/api/prompts/nl_search", {
      method: "POST",
      body: JSON.stringify({
        query,
        k: Number($("#topk-select")?.value || 19),
        tau: DEFAULT_TAU,
        alpha,
        lam: DEFAULT_LAM,
        salience,
      }),
    });
    const latencyMs = Math.round(performance.now() - startedAt);
    renderPromptResults(result.results || [], {
      latencyMs,
      resultCount: result.result_count || 0,
      alpha,
      salience,
    });
    await renderSearchVisualizations(result.results || []);
    const healthEl = $("#health");
    if (healthEl) {
      healthEl.textContent = "CVE Ready";
      healthEl.className = "health ok";
    }
    setText("#search-mode-label", `α ${alpha.toFixed(2)} · sal ${salience.toFixed(2)}`);
    setText("#search-hint", `${result.result_count || 0} CVE results`);
    pushRecentSearch(query);
  } catch (e) {
    const healthEl = $("#health");
    if (healthEl) {
      healthEl.textContent = "Search Error";
      healthEl.classList.add("err");
    }
    $("#grid").innerHTML = `<div class="error-screen"><h2>Search failed</h2><p>${escapeHtml(e.message)}</p></div>`;
    wirePromptCards();
  }
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
          <div class="prompt-score-bar"><div class="prompt-score-fill" style="width:${Math.min(
            100,
            (item.score ?? 0) * 100
          )}%"></div></div>
        </div>
      </div>
      <p class="prompt-content">${highlightQuery(content, state.searchQuery)}</p>
      <div class="prompt-card-actions">
        <button class="prompt-toggle" type="button">Expand</button>
        <button class="prompt-copy-btn" type="button" data-copy="${escapeHtml(content)}">Copy</button>
      </div>
      <div class="prompt-result-meta">
        <span>ID: ${escapeHtml(item.id ?? "—")}</span>
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
    const copyBtn = card.querySelector(".prompt-copy-btn");
    const idx = Number(card.dataset.index);
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
          await navigator.clipboard.writeText(copyBtn.getAttribute("data-copy") || "");
          copyBtn.textContent = "Copied";
          setTimeout(() => (copyBtn.textContent = "Copy"), 900);
        } catch (_) {}
      });
    }
  });
}

function openPromptModal(item) {
  const modal = $("#prompt-modal");
  const body = $("#prompt-modal-body");
  if (!modal || !body || !item) return;
  const content = item.content || item.body || "No content";
  body.innerHTML = `
    <h2 class="prompt-modal-title">${escapeHtml(item.title || item.id || "CVE")}</h2>
    <div class="prompt-modal-text">${highlightQuery(content, state.searchQuery)}</div>
    <div class="prompt-modal-meta">
      <div class="prompt-modal-chip">Score ${(item.score ?? 0).toFixed(4)}</div>
      <div class="prompt-modal-chip">Salience ${(item.salience ?? 0).toFixed(3)}</div>
      <div class="prompt-modal-chip">Upvotes ${item.upvotes ?? 0}</div>
      <div class="prompt-modal-chip">Views ${item.views ?? 0}</div>
      <div class="prompt-modal-chip">${escapeHtml(item.id ?? "—")}</div>
    </div>`;
  modal.classList.remove("hidden");
}

async function ensureLeafReady() {
  try {
    const health = await api("/api/prompts/health");
    if (health.status === "ready") return health;
    $("#health").textContent = "Warming CVE engine...";
    await api("/api/prompts/warm");
    for (let i = 0; i < 30; i++) {
      await new Promise((r) => setTimeout(r, 2000));
      const polled = await api("/api/prompts/health");
      if (polled.status === "ready") return polled;
    }
    throw new Error("CVE engine warmup timeout");
  } catch (e) {
    console.error("CVE readiness error:", e);
    throw e;
  }
}

async function loadAuditPanel() {
  try {
    $("#health").textContent = "Loading audit...";
    await ensureLeafReady();
    const [health, graph, lambdas, audit] = await Promise.all([
      api("/api/prompts/health"),
      api("/api/prompts/graph_laplacian"),
      api("/api/prompts/lambdas"),
      api("/api/prompts/audit"),
    ]);
    $("#health").textContent = "CVE Ready";
    $("#health").className = "health ok";
    renderAuditHealth(health);
    renderAuditGraph(graph);
    renderAuditLambdas(audit, lambdas);
    renderAuditStats(audit);
    renderAuditManifold(audit);
    renderAuditSpectral(audit, lambdas);
    renderAuditPCA(audit);
    requestAnimationFrame(resizeKnownPlots);
    setTimeout(_fetchAndRenderDrift, 400);
  } catch (e) {
    console.error(e);
    $("#health").textContent = "CVE Audit Error";
    $("#health").className = "health err";
    $("#audit-content").innerHTML = `
      <div class="error-screen">
        <h2>CVE Audit Error</h2><p>${escapeHtml(e.message)}</p>
      </div>`;
  }
}

function renderAuditHealth(health) {
  const container = $("#audit-health");
  if (!container) return;
  container.innerHTML = `
    <div class="signal-card"><span class="signal-label">Status</span><strong>${escapeHtml(health.status ?? "unknown")}</strong></div>
    <div class="signal-card"><span class="signal-label">Prompt Engine</span><strong>${health.prompt_engine_ready}</strong></div>
    <div class="signal-card"><span class="signal-label">Embedder</span><strong>${health.embedder_ready}</strong></div>
    <div class="signal-card"><span class="signal-label">Model</span><strong>${escapeHtml(health.embedder_model ?? "—")}</strong></div>`;
}

function renderAuditGraph(graph) {
  const container = $("#audit-graph");
  if (!container) return;
  container.innerHTML = `
    <div class="signal-card"><span class="signal-label">Items</span><strong>${graph.nitems ?? "—"}</strong></div>
    <div class="signal-card"><span class="signal-label">Features</span><strong>${graph.nfeatures ?? "—"}</strong></div>
    <div class="signal-card"><span class="signal-label">Clusters</span><strong>${graph.nclusters ?? "—"}</strong></div>
    <div class="signal-card"><span class="signal-label">GL Nodes</span><strong>${graph.gl_nodes ?? "—"}</strong></div>
    <div class="signal-card"><span class="signal-label">GL Shape</span><strong>${Array.isArray(graph.gl_shape) ? graph.gl_shape.join(" × ") : "—"}</strong></div>`;
}

function renderAuditLambdas(audit, lambdaData) {
  const container = $("#audit-lambdas");
  if (!container) return;
  const spectral = (audit && audit.spectral_stats) || {};
  const readNumber = (...candidates) => {
    for (const v of candidates) {
      if (v === null || v === undefined) continue;
      const n = Number(v);
      if (Number.isFinite(n)) return n;
    }
    return null;
  };
  const fiedler = readNumber(spectral.fiedler_value, spectral.fiedlerValue, spectral.fiedler, spectral.lambda2);
  const gap = readNumber(spectral.spectral_gap, spectral.spectralGap, spectral.gap);
  const fmt = (v) => (v === null ? "—" : v.toFixed(6));
  const fiedlerColor =
    fiedler === null ? "#9aa6bd" : fiedler > 0.01 ? "#34d399" : fiedler > 0.001 ? "#facc15" : "#f87171";
  const lambdaArr =
    Array.isArray(audit?.eigenvalues) && audit.eigenvalues.length
      ? audit.eigenvalues
      : Array.isArray(lambdaData?.lambdas)
      ? lambdaData.lambdas
      : null;
  const lambdaSamples = lambdaArr ? lambdaArr.length : null;
  let source = "—";
  if (typeof spectral.source === "string" && spectral.source.trim()) {
    source = spectral.source.trim();
  } else if (audit?.build_params && typeof audit.build_params === "object") {
    const bp = audit.build_params;
    const parts = [];
    if (bp.k != null) parts.push(`k=${bp.k}`);
    if (bp.eps != null) parts.push(`eps=${Number(bp.eps).toFixed(3)}`);
    if (bp.p != null) parts.push(`p=${bp.p}`);
    source = parts.length ? `graph L (${parts.join(", ")})` : "audit endpoint";
  } else {
    source = "audit endpoint";
  }
  container.innerHTML = `
    <div class="signal-card"><span class="signal-label">Fiedler Value</span><strong style="color:${fiedlerColor}">${fmt(fiedler)}</strong></div>
    <div class="signal-card"><span class="signal-label">Spectral Gap</span><strong>${fmt(gap)}</strong></div>
    <div class="signal-card"><span class="signal-label">λ Samples</span><strong>${lambdaSamples == null ? "—" : lambdaSamples}</strong></div>
    <div class="signal-card"><span class="signal-label">Source</span><strong>${escapeHtml(source)}</strong></div>`;
}

function renderAuditStats(audit) {
  const container = $("#audit-stats");
  if (!container) return;
  const stats = audit?.degree_stats || {};
  const graph = audit?.graph_stats || {};
  const sparsity = Number(graph.sparsity ?? 0);
  container.innerHTML = `
    <div class="signal-card"><span class="signal-label">Degree Mean</span><strong>${(stats.mean ?? 0).toFixed(4)}</strong></div>
    <div class="signal-card"><span class="signal-label">Degree Std</span><strong>${(stats.std ?? 0).toFixed(4)}</strong></div>
    <div class="signal-card"><span class="signal-label">Degree Min</span><strong>${(stats.min ?? 0).toFixed(4)}</strong></div>
    <div class="signal-card"><span class="signal-label">Degree Max</span><strong>${(stats.max ?? 0).toFixed(4)}</strong></div>
    <div class="signal-card"><span class="signal-label">Edges</span><strong>${graph.n_edges ?? "—"}</strong></div>
    <div class="signal-card"><span class="signal-label">Sparsity</span><strong>${sparsity ? sparsity.toFixed(6) : "—"}</strong></div>`;
}

function renderAuditPCA(audit) {
  const canvas = $("#audit-pca");
  if (!canvas) return;
  const points = audit.pca_2d || [];
  const ids = audit.ids || [];
  const ctx = canvas.getContext("2d");
  const width = 900;
  const height = 260;
  canvas.width = width;
  canvas.height = height;
  ctx.clearRect(0, 0, width, height);
  if (!points.length) {
    ctx.fillStyle = "#9aa6bd";
    ctx.font = "14px Inter";
    ctx.fillText("No PCA data available", 24, 40);
    return;
  }
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const pad = 24;
  const projected = points.map((p, i) => ({
    x: pad + ((p[0] - minX) / (maxX - minX || 1)) * (width - pad * 2),
    y: height - pad - ((p[1] - minY) / (maxY - minY || 1)) * (height - pad * 2),
    id: ids[i] || `point_${i}`,
  }));
  projected.forEach((p) => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(124,92,255,0.85)";
    ctx.fill();
  });
  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let hovered = null;
    for (const p of projected) {
      if (Math.sqrt((p.x - mx) ** 2 + (p.y - my) ** 2) < 8) {
        hovered = p;
        break;
      }
    }
    const tooltip = $("#audit-pca-tooltip");
    if (tooltip) tooltip.textContent = hovered ? hovered.id : "Hover points";
  };
  canvas.onclick = (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let clicked = null;
    for (const p of projected) {
      if (Math.sqrt((p.x - mx) ** 2 + (p.y - my) ** 2) < 10) {
        clicked = p;
        break;
      }
    }
    if (!clicked) return;
    switchView("search");
    const filter = $("#filter");
    if (filter) filter.value = clicked.id;
    setText("#search-hint", `Searching CVE ${clicked.id}...`);
    runSearch();
  };
}

function renderAuditManifold(audit) {
  const el = document.getElementById("audit-query-manifold");
  const showError = (msg) => {
    if (el) el.innerHTML = `<div class="manifold-empty-msg">${escapeHtml(msg)}</div>`;
  };
  if (!el) return;
  if (!window.Plotly) {
    showError("Plotly failed to load.");
    return;
  }
  const bpEl = document.getElementById("build-params-display");
  if (bpEl && audit?.build_params) {
    const bp = audit.build_params;
    const fmt = (v) => (v === null || v === undefined ? "None" : String(v));
    bpEl.innerHTML = `<code>eps=${fmt(bp.eps)} · k=${fmt(bp.k)} · topk=${fmt(bp.topk)} · p=${fmt(bp.p)} · sigma=${fmt(
      bp.sigma
    )}</code>`;
  }
  const lm = audit?.laplacian_manifold;
  if (lm && Array.isArray(lm.z_grid) && lm.z_grid.length) {
    renderServerLaplacianManifold(audit, lm, el);
    return;
  }
  const points = Array.isArray(audit?.pca_2d) ? audit.pca_2d : null;
  const degrees = Array.isArray(audit?.degrees) ? audit.degrees : null;
  if (!points || !points.length) {
    showError("Graph Laplacian Manifold unavailable: missing pca_2d.");
    return;
  }
  if (!degrees || !degrees.length) {
    showError("Graph Laplacian Manifold unavailable: missing degrees.");
    return;
  }
  renderQueryManifold(audit, new Set(), "#audit-query-manifold");
}

function renderServerLaplacianManifold(audit, lm, el) {
  const zMin = Number(lm.degree_p05);
  const zMax = Number(lm.degree_p95);
  const colorscale = [
    [0.0, "#1e3a8a"],
    [0.25, "#2c5dff"],
    [0.5, "#f8fafc"],
    [0.75, "#fb7185"],
    [1.0, "#ef4444"],
  ];
  const surface = {
    x: lm.x_grid,
    y: lm.y_grid,
    z: lm.z_grid,
    type: "surface",
    colorscale,
    cmin: zMin,
    cmax: zMax,
    opacity: 0.94,
    showscale: true,
    colorbar: {
      title: { text: "Curvature (Lᵢᵢ)", font: { color: "#cbd5e1", size: 12 } },
      tickfont: { color: "#cbd5e1" },
      thickness: 14,
      len: 0.78,
      x: 1.02,
    },
    contours: { z: { show: true, usecolormap: true, highlightcolor: "#ffffff", project: { z: true } } },
    lighting: { ambient: 0.65, diffuse: 0.85, specular: 0.18, roughness: 0.55 },
    name: "Laplacian surface",
  };
  const hubLift = zMax + (zMax - zMin) * 0.06;
  const hubsTrace = {
    x: lm.hub_x,
    y: lm.hub_y,
    z: lm.hub_x.map(() => hubLift),
    type: "scatter3d",
    mode: "markers",
    name: "High-degree hubs (top 15%)",
    text: lm.hub_text,
    hoverinfo: "text",
    marker: { size: 4.5, color: "#fbbf24", line: { color: "#ffffff", width: 0.6 }, symbol: "diamond" },
  };
  Plotly.newPlot(
    el,
    [surface, hubsTrace],
    {
      paper_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#cbd5e1" },
      scene: {
        xaxis: {
          title: lm.x_label || "PC1",
          gridcolor: "rgba(255,255,255,0.10)",
          zerolinecolor: "rgba(255,255,255,0.18)",
          color: "#cbd5e1",
        },
        yaxis: {
          title: lm.y_label || "PC2",
          gridcolor: "rgba(255,255,255,0.10)",
          zerolinecolor: "rgba(255,255,255,0.18)",
          color: "#cbd5e1",
        },
        zaxis: {
          title: "Curvature (Lᵢᵢ)",
          gridcolor: "rgba(255,255,255,0.10)",
          zerolinecolor: "rgba(255,255,255,0.18)",
          color: "#cbd5e1",
          range: [zMin, hubLift + (zMax - zMin) * 0.18],
        },
        bgcolor: "rgba(15,23,42,0.92)",
        aspectmode: "manual",
        aspectratio: { x: 1.35, y: 1.1, z: 0.85 },
        camera: { eye: { x: 1.55, y: 1.55, z: 0.95 }, up: { x: 0, y: 0, z: 1 } },
      },
      margin: { l: 0, r: 0, t: 20, b: 0 },
      legend: { font: { color: "#cbd5e1" }, orientation: "h", x: 0, y: 1.04, bgcolor: "rgba(15,23,42,0)" },
    },
    { responsive: true, displayModeBar: false }
  );
}

function renderAuditSpectral(audit, lambdas) {
  const source =
    audit && Array.isArray(audit.eigenvalues) && audit.eigenvalues.length
      ? { lambdas: audit.eigenvalues, n: audit.eigenvalues.length }
      : lambdas;
  renderQueryLambdaChart(source, "#audit-spectral-fingerprint");
}

// ─── SPECTRAL DRIFT ───────────────────────────────────────────────────────────

function _computeEcdf(values) {
  const sorted = [...values].map(Number).filter(Number.isFinite).sort((a, b) => a - b);
  return {
    x: sorted,
    y: sorted.map((_, i) => (i + 1) / sorted.length),
  };
}

function _computeKS(a, b) {
  const A = [...a].map(Number).filter(Number.isFinite).sort((x, y) => x - y);
  const B = [...b].map(Number).filter(Number.isFinite).sort((x, y) => x - y);
  if (!A.length || !B.length) return null;
  const all = [...A, ...B].sort((x, y) => x - y);
  let maxDiff = 0;
  let bestX = all[0];
  let i = 0;
  let j = 0;
  for (const x of all) {
    while (i < A.length && A[i] <= x) i++;
    while (j < B.length && B[j] <= x) j++;
    const cdfA = i / A.length;
    const cdfB = j / B.length;
    const diff = Math.abs(cdfA - cdfB);
    if (diff > maxDiff) {
      maxDiff = diff;
      bestX = x;
    }
  }
  return { value: maxDiff, x: bestX };
}

function _mean(arr) {
  const v = arr.map(Number).filter(Number.isFinite);
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}

function _spectralGap(arr) {
  const s = [...arr].map(Number).filter(Number.isFinite).sort((a, b) => a - b);
  if (s.length < 2) return null;
  return s[1] - s[0];
}

function _computeW1(a, b) {
  const A = [...a].map(Number).filter(Number.isFinite).sort((x, y) => x - y);
  const B = [...b].map(Number).filter(Number.isFinite).sort((x, y) => x - y);
  if (!A.length || !B.length) return null;
  const n = Math.min(A.length, B.length);
  if (!n) return null;
  let total = 0;
  for (let i = 0; i < n; i++) {
    const ai = A[Math.floor((i * A.length) / n)];
    const bi = B[Math.floor((i * B.length) / n)];
    total += Math.abs(ai - bi);
  }
  return total / n;
}

function _driftLevel(score) {
  if (score == null) return { label: "N/A", bg: "rgba(148,163,184,0.15)", color: "#cbd5e1" };
  if (score < 0.05) return { label: "Low", bg: "rgba(67,122,34,0.15)", color: "#86efac" };
  if (score < 0.15) return { label: "Medium", bg: "rgba(209,153,0,0.18)", color: "#fcd34d" };
  return { label: "High", bg: "rgba(161,44,123,0.18)", color: "#f9a8d4" };
}

async function _fetchAndRenderDrift() {
  const status = document.getElementById("drift-status");
  const badge = document.getElementById("drift-score-badge");
  if (status) status.textContent = "Fetching drift data…";
  try {
    const res = await fetch("/api/drift/lambdas");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    _renderDriftPanel(data);
    _renderDriftView(data);
  } catch (e) {
    console.error("Drift fetch failed:", e);
    if (status) status.textContent = "Drift endpoint unavailable (/api/drift/lambdas): " + e.message;
    if (badge) {
      badge.textContent = "drift: N/A";
      badge.style.display = "inline-block";
    }
    _renderDriftView({
      period_a: { eigenvalues: [], label: "Period A" },
      period_b: { eigenvalues: [], label: "Period B" },
      error: e.message,
    });
  }
}

function _renderDriftPanel(data) {
  const el = document.getElementById("drift-chart");
  const status = document.getElementById("drift-status");
  const badge = document.getElementById("drift-score-badge");
  if (!el || !window.Plotly) return;

  const lambdasA = (data.period_a && data.period_a.eigenvalues) || [];
  const lambdasB = (data.period_b && data.period_b.eigenvalues) || [];
  const labelA = (data.period_a && data.period_a.label) || "Period A — cve_99_14 (1999–2014)";
  const labelB = (data.period_b && data.period_b.label) || "Period B — cve_99_25 (1999–2025)";
  const driftScore = typeof data.drift_score === "number" ? data.drift_score : _computeW1(lambdasA, lambdasB);
  const ks = _computeKS(lambdasA, lambdasB);

  if (!lambdasA.length && !lambdasB.length) {
    if (status) status.textContent = "No eigenvalue data returned by /api/drift/lambdas.";
    return;
  }

  const traceA = {
    x: lambdasA,
    type: "histogram",
    name: labelA,
    opacity: 0.65,
    histnorm: "probability density",
    nbinsx: 60,
    marker: { color: "rgba(1,105,111,0.78)" },
  };
  const traceB = {
    x: lambdasB,
    type: "histogram",
    name: labelB,
    opacity: 0.6,
    histnorm: "probability density",
    nbinsx: 60,
    marker: { color: "rgba(218,113,1,0.72)" },
  };

  Plotly.react(
    el,
    [traceA, traceB],
    {
      barmode: "overlay",
      paper_bgcolor: "transparent",
      plot_bgcolor: "rgba(15,23,42,0.75)",
      font: { family: "Inter, sans-serif", size: 12, color: "#cbd5e1" },
      margin: { t: 24, r: 24, b: 52, l: 60 },
      xaxis: { title: "Eigenvalue (λ)", gridcolor: "rgba(255,255,255,0.08)", zeroline: false, color: "#cbd5e1" },
      yaxis: { title: "Density", gridcolor: "rgba(255,255,255,0.08)", zeroline: false, color: "#cbd5e1" },
      legend: { orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "right", x: 1, font: { color: "#cbd5e1" } },
      bargap: 0.04,
      annotations: [
        {
          xref: "paper",
          yref: "paper",
          x: 0,
          y: 1.08,
          text: `<b>Spectral Drift</b> · ${lambdasA.length.toLocaleString()} λ (A) · ${lambdasB.length.toLocaleString()} λ (B)${
            ks ? " · KS = " + ks.value.toFixed(4) : ""
          }`,
          showarrow: false,
          align: "left",
          xanchor: "left",
          yanchor: "bottom",
          font: { color: "#e2e8f0", size: 12 },
        },
      ],
    },
    { responsive: true, displayModeBar: false }
  );

  if (badge) {
    const lvl = _driftLevel(driftScore);
    badge.textContent = driftScore == null ? "W₁ drift: N/A" : `W₁ drift: ${driftScore.toFixed(4)} · ${lvl.label}`;
    badge.style.display = "inline-block";
    badge.style.background = lvl.bg;
    badge.style.color = lvl.color;
  }

  if (status) {
    status.textContent =
      `${lambdasA.length.toLocaleString()} eigenvalues (A) · ${lambdasB.length.toLocaleString()} eigenvalues (B)` +
      (driftScore != null ? ` · W₁ = ${driftScore.toFixed(4)}` : "") +
      (ks ? ` · KS = ${ks.value.toFixed(4)}` : "");
  }
}

function _renderKpiGrid(targetId, entries) {
  const el = document.getElementById(targetId);
  if (!el) return;
  el.innerHTML = entries
    .map(
      ([label, value]) => `
      <div class="signal-card">
        <span class="signal-label">${escapeHtml(label)}</span>
        <strong>${escapeHtml(value)}</strong>
      </div>`
    )
    .join("");
}

function _renderDriftView(data) {
  const a = (data.period_a && data.period_a.eigenvalues) || [];
  const b = (data.period_b && data.period_b.eigenvalues) || [];
  const labelA = (data.period_a && data.period_a.label) || "cve_99_14";
  const labelB = (data.period_b && data.period_b.label) || "cve_99_25";
  const ks = _computeKS(a, b);
  const w1 = typeof data.drift_score === "number" ? data.drift_score : _computeW1(a, b);
  const meanA = _mean(a);
  const meanB = _mean(b);
  const gapA = _spectralGap(a);
  const gapB = _spectralGap(b);
  const gapDelta =
    gapA != null && gapB != null ? Math.abs(gapB - gapA) : null;

  _renderKpiGrid("drift-kpi-a", [
    ["Label", labelA],
    ["λ count", String(a.length || 0)],
    ["Mean λ", meanA == null ? "—" : meanA.toFixed(6)],
    ["Gap", gapA == null ? "—" : gapA.toFixed(6)],
  ]);

  _renderKpiGrid("drift-kpi-b", [
    ["Label", labelB],
    ["λ count", String(b.length || 0)],
    ["Mean λ", meanB == null ? "—" : meanB.toFixed(6)],
    ["Gap", gapB == null ? "—" : gapB.toFixed(6)],
  ]);

  _renderKpiGrid("drift-kpi-delta", [
    ["Wasserstein-1", w1 == null ? "—" : w1.toFixed(6)],
    ["KS statistic", ks == null ? "—" : ks.value.toFixed(6)],
    ["Mean Δ", meanA == null || meanB == null ? "—" : Math.abs(meanB - meanA).toFixed(6)],
    ["Drift level", _driftLevel(w1).label],
  ]);

  _renderKpiGrid("drift-kpi-gap", [
    ["Gap A", gapA == null ? "—" : gapA.toFixed(6)],
    ["Gap B", gapB == null ? "—" : gapB.toFixed(6)],
    ["|Gap Δ|", gapDelta == null ? "—" : gapDelta.toFixed(6)],
    ["Endpoint", data.error ? `Error: ${data.error}` : "/api/drift/lambdas"],
  ]);

  _plotDriftOverlay(a, b, labelA, labelB);
  _plotDriftECDF(a, b, labelA, labelB, ks);
  _plotDriftPCAPlaceholder(labelA, labelB);
  _plotDriftTimelinePlaceholder(gapA, gapB);
}

function _plotDriftOverlay(a, b, labelA, labelB) {
  const el = document.getElementById("drift-lambda-overlay");
  if (!el || !window.Plotly) return;
  if (!a.length && !b.length) {
    el.innerHTML = `<div class="manifold-empty-msg">No drift eigenvalue data available.</div>`;
    return;
  }
  Plotly.react(
    el,
    [
      {
        x: a,
        type: "histogram",
        name: labelA,
        opacity: 0.62,
        histnorm: "probability density",
        nbinsx: 60,
        marker: { color: "rgba(94,162,255,0.72)" },
      },
      {
        x: b,
        type: "histogram",
        name: labelB,
        opacity: 0.58,
        histnorm: "probability density",
        nbinsx: 60,
        marker: { color: "rgba(218,113,1,0.72)" },
      },
    ],
    {
      barmode: "overlay",
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(15,23,42,0.85)",
      font: { color: "#cbd5e1" },
      margin: { l: 60, r: 24, t: 20, b: 52 },
      xaxis: { title: "Eigenvalue λ", gridcolor: "rgba(255,255,255,0.08)" },
      yaxis: { title: "Density", gridcolor: "rgba(255,255,255,0.08)" },
      legend: { orientation: "h", x: 0, y: 1.08 },
    },
    { responsive: true, displayModeBar: false }
  );
}

function _plotDriftECDF(a, b, labelA, labelB, ks) {
  const el = document.getElementById("drift-ecdf");
  if (!el || !window.Plotly) return;
  if (!a.length && !b.length) {
    el.innerHTML = `<div class="manifold-empty-msg">No ECDF drift data available.</div>`;
    return;
  }
  const ecdfA = _computeEcdf(a);
  const ecdfB = _computeEcdf(b);
  const traces = [];
  if (ecdfA.x.length) {
    traces.push({
      x: ecdfA.x,
      y: ecdfA.y,
      type: "scatter",
      mode: "lines",
      name: labelA,
      line: { color: "#5ea2ff", width: 2.5 },
    });
  }
  if (ecdfB.x.length) {
    traces.push({
      x: ecdfB.x,
      y: ecdfB.y,
      type: "scatter",
      mode: "lines",
      name: labelB,
      line: { color: "#da7101", width: 2.5 },
    });
  }
  const annotations = [];
  const shapes = [];
  if (ks) {
    annotations.push({
      x: ks.x,
      y: 1.02,
      xref: "x",
      yref: "paper",
      text: `KS = ${ks.value.toFixed(4)}`,
      showarrow: false,
      font: { color: "#f8fafc", size: 11 },
    });
    shapes.push({
      type: "line",
      x0: ks.x,
      x1: ks.x,
      y0: 0,
      y1: 1,
      xref: "x",
      yref: "y",
      line: { color: "rgba(248,250,252,0.35)", dash: "dash", width: 1.5 },
    });
  }
  Plotly.react(
    el,
    traces,
    {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(15,23,42,0.85)",
      font: { color: "#cbd5e1" },
      margin: { l: 60, r: 24, t: 20, b: 52 },
      xaxis: { title: "Eigenvalue λ", gridcolor: "rgba(255,255,255,0.08)" },
      yaxis: { title: "ECDF", range: [0, 1], gridcolor: "rgba(255,255,255,0.08)" },
      legend: { orientation: "h", x: 0, y: 1.08 },
      annotations,
      shapes,
    },
    { responsive: true, displayModeBar: false }
  );
}

function _plotDriftPCAPlaceholder(labelA, labelB) {
  const el = document.getElementById("drift-pca");
  if (!el || !window.Plotly) return;
  Plotly.react(
    el,
    [
      {
        x: [0],
        y: [0],
        mode: "markers+text",
        type: "scatter",
        text: [`${labelA} vs ${labelB}<br>PCA comparison placeholder`],
        textposition: "top center",
        marker: { size: 10, color: "#7c5cff" },
        hoverinfo: "skip",
      },
    ],
    {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(15,23,42,0.85)",
      font: { color: "#cbd5e1" },
      margin: { l: 40, r: 20, t: 20, b: 40 },
      xaxis: { visible: false },
      yaxis: { visible: false },
      annotations: [
        {
          x: 0.5,
          y: 0.5,
          xref: "paper",
          yref: "paper",
          text: "Backend does not currently expose period-specific PCA coordinates.<br>This panel is wired and ready once the API returns them.",
          showarrow: false,
          align: "center",
          font: { size: 13, color: "#cbd5e1" },
        },
      ],
    },
    { responsive: true, displayModeBar: false }
  );
}

function _plotDriftTimelinePlaceholder(gapA, gapB) {
  const el = document.getElementById("drift-timeline");
  if (!el || !window.Plotly) return;
  const x = ["1999–2014", "1999–2025"];
  const y = [gapA, gapB].map((v) => (Number.isFinite(v) ? v : null));
  Plotly.react(
    el,
    [
      {
        x,
        y,
        type: "scatter",
        mode: "lines+markers",
        line: { color: "#5ea2ff", width: 3 },
        marker: { size: 9, color: "#7c5cff" },
        name: "Spectral gap",
      },
    ],
    {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(15,23,42,0.85)",
      font: { color: "#cbd5e1" },
      margin: { l: 60, r: 24, t: 20, b: 52 },
      xaxis: { title: "Period", gridcolor: "rgba(255,255,255,0.08)" },
      yaxis: { title: "Spectral gap", gridcolor: "rgba(255,255,255,0.08)" },
    },
    { responsive: true, displayModeBar: false }
  );
}

// ─────────────────────────────────────────────────────────────────────────────

function _quantile(sorted, q) {
  if (!sorted.length) return 0;
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (base + 1 < sorted.length) return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  return sorted[base];
}

function _idwGrid(xs, ys, zs, gridSize, power = 2) {
  const minX = Math.min(...xs),
    maxX = Math.max(...xs);
  const minY = Math.min(...ys),
    maxY = Math.max(...ys);
  const dx = maxX - minX || 1,
    dy = maxY - minY || 1;
  const stride = Math.max(1, Math.floor(xs.length / 4000));
  const sigma2 = Math.pow((dx + dy) * 0.12, 2);
  const gridX = new Array(gridSize),
    gridY = new Array(gridSize);
  for (let i = 0; i < gridSize; i++) {
    gridX[i] = minX + (dx * i) / (gridSize - 1);
    gridY[i] = minY + (dy * i) / (gridSize - 1);
  }
  const z = [];
  for (let gy = 0; gy < gridSize; gy++) {
    const row = new Array(gridSize);
    const y = gridY[gy];
    for (let gx = 0; gx < gridSize; gx++) {
      const x = gridX[gx];
      let num = 0,
        den = 0;
      for (let i = 0; i < xs.length; i += stride) {
        const ddx = x - xs[i],
          ddy = y - ys[i],
          d2 = ddx * ddx + ddy * ddy + 1e-9;
        const w = Math.exp(-d2 / sigma2) / Math.pow(d2, power / 2);
        num += w * zs[i];
        den += w;
      }
      row[gx] = den > 0 ? num / den : 0;
    }
    z.push(row);
  }
  return { z, gridX, gridY };
}

function renderQueryManifold(audit, resultIds, target = "#query-manifold") {
  const el = typeof target === "string" ? $(target) : target;
  if (!el || !audit?.pca_2d || !window.Plotly) return;
  const points = audit.pca_2d,
    ids = audit.ids || [];
  const explained = audit.pca_explained_variance || [];
  const pc1Pct = explained[0] != null ? (explained[0] * 100).toFixed(1) : "—";
  const pc2Pct = explained[1] != null ? (explained[1] * 100).toFixed(1) : "—";
  let degrees = audit.degrees;
  if (!Array.isArray(degrees) || degrees.length !== points.length) degrees = new Array(points.length).fill(1);
  const degArr = degrees.map(Number);
  const sortedDeg = [...degArr].sort((a, b) => a - b);
  const p05 = _quantile(sortedDeg, 0.05),
    p95 = _quantile(sortedDeg, 0.95);
  const degRange = Math.max(p95 - p05, 1e-6);
  const degClipped = degArr.map((d) => (Math.min(Math.max(d, p05), p95) - p05) / degRange);
  const xs = points.map((p) => Number(p[0])),
    ys = points.map((p) => Number(p[1]));
  const highlightedIndices = new Set(
    [...resultIds]
      .map((id) => {
        const m = String(id).match(/\d+/);
        return m ? Number(m[0]) : NaN;
      })
      .filter(Number.isFinite)
  );
  const isAuditTarget = typeof target === "string" && target.includes("audit-query-manifold");
  const gridSize = isAuditTarget ? 110 : 80;
  const { z, gridX, gridY } = _idwGrid(xs, ys, degClipped, gridSize, 2);
  const flatZ = z.flat().filter(Number.isFinite);
  const sortedZ = [...flatZ].sort((a, b) => a - b);
  const zMin = _quantile(sortedZ, 0.02),
    zMax = _quantile(sortedZ, 0.98);
  if (zMax - zMin < 1e-9) {
    if (isAuditTarget) el.innerHTML = '<div class="manifold-empty-msg">Degree variance ≈ 0, cannot render surface.</div>';
    return;
  }
  const colorscale = [
    [0.0, "#1e3a8a"],
    [0.25, "#2c5dff"],
    [0.5, "#f8fafc"],
    [0.75, "#fb7185"],
    [1.0, "#ef4444"],
  ];
  const surface = {
    x: gridX,
    y: gridY,
    z,
    type: "surface",
    colorscale,
    cmin: zMin,
    cmax: zMax,
    opacity: 0.94,
    showscale: true,
    colorbar: {
      title: { text: "Node Degree (Lᵢᵢ)", font: { color: "#cbd5e1", size: 12 } },
      tickfont: { color: "#cbd5e1" },
      thickness: 14,
      len: 0.78,
      x: 1.02,
    },
    contours: { z: { show: true, usecolormap: true, highlightcolor: "#ffffff", project: { z: true } } },
    lighting: { ambient: 0.65, diffuse: 0.85, specular: 0.18, roughness: 0.55 },
    name: "Laplacian surface",
  };
  const cloudStride = Math.max(1, Math.floor(points.length / 4000));
  const cloudIdx = [];
  for (let i = 0; i < points.length; i += cloudStride) cloudIdx.push(i);
  const cloudTrace = {
    x: cloudIdx.map((i) => xs[i]),
    y: cloudIdx.map((i) => ys[i]),
    z: cloudIdx.map(() => zMin),
    type: "scatter3d",
    mode: "markers",
    name: "Corpus nodes",
    hoverinfo: "skip",
    marker: { size: 1.6, color: cloudIdx.map((i) => degClipped[i]), colorscale, cmin: zMin, cmax: zMax, opacity: 0.55, showscale: false },
  };
  const hubThreshold = _quantile(sortedDeg, 0.98);
  const hubIdxs = [];
  for (let i = 0; i < degArr.length; i++) if (degArr[i] >= hubThreshold) hubIdxs.push(i);
  const hubLift = zMax + (zMax - zMin) * 0.06;
  const hubsTrace = {
    x: hubIdxs.map((i) => xs[i]),
    y: hubIdxs.map((i) => ys[i]),
    z: hubIdxs.map(() => hubLift),
    type: "scatter3d",
    mode: "markers",
    name: "High-degree hubs",
    text: hubIdxs.map((i) => `${ids[i] ?? `#${i}`} (Lᵢᵢ=${degArr[i].toFixed(3)})`),
    hoverinfo: "text",
    marker: { size: 4.5, color: "#fbbf24", line: { color: "#ffffff", width: 0.6 }, symbol: "diamond" },
  };
  const matched = [...highlightedIndices].filter((i) => i >= 0 && i < points.length);
  const matchedTrace = {
    x: matched.map((i) => xs[i]),
    y: matched.map((i) => ys[i]),
    z: matched.map(() => hubLift + (zMax - zMin) * 0.08),
    type: "scatter3d",
    mode: "markers",
    name: "Matched CVEs",
    text: matched.map((i) => ids[i] ?? `#${i}`),
    hoverinfo: "text",
    marker: { size: 5.5, color: "#ef4444", line: { color: "#ffffff", width: 1.1 }, symbol: "circle" },
  };
  const data = [cloudTrace, surface, hubsTrace];
  if (matched.length) data.push(matchedTrace);
  Plotly.newPlot(
    el,
    data,
    {
      paper_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#cbd5e1" },
      scene: {
        xaxis: { title: `PC1 (${pc1Pct}%)`, gridcolor: "rgba(255,255,255,0.10)", zerolinecolor: "rgba(255,255,255,0.18)", color: "#cbd5e1" },
        yaxis: { title: `PC2 (${pc2Pct}%)`, gridcolor: "rgba(255,255,255,0.10)", zerolinecolor: "rgba(255,255,255,0.18)", color: "#cbd5e1" },
        zaxis: {
          title: "Node Degree (Lᵢᵢ)",
          gridcolor: "rgba(255,255,255,0.10)",
          zerolinecolor: "rgba(255,255,255,0.18)",
          color: "#cbd5e1",
          range: [zMin, hubLift + (zMax - zMin) * 0.18],
        },
        bgcolor: "rgba(15,23,42,0.92)",
        aspectmode: "manual",
        aspectratio: { x: 1.35, y: 1.1, z: 0.85 },
        camera: { eye: { x: 1.55, y: 1.55, z: 0.95 }, up: { x: 0, y: 0, z: 1 } },
      },
      margin: { l: 0, r: 0, t: 20, b: 0 },
      legend: { font: { color: "#cbd5e1" }, orientation: "h", x: 0, y: 1.04, bgcolor: "rgba(15,23,42,0)" },
    },
    { responsive: true, displayModeBar: false }
  );
}

function renderQueryLambdaChart(data, target = "#query-lambda-chart") {
  const el = typeof target === "string" ? $(target) : target;
  if (!el || !data?.lambdas || !window.Plotly) return;
  const lambdas = data.lambdas.map(Number).filter(Number.isFinite);
  if (!lambdas.length) return;
  const sorted = [...lambdas].sort((a, b) => a - b);
  const n = sorted.length,
    lambdaMax = sorted[n - 1] || 1;
  const p25 = _quantile(sorted, 0.25),
    p50 = _quantile(sorted, 0.5),
    p75 = _quantile(sorted, 0.75),
    p60 = _quantile(sorted, 0.6);
  const NBINS = 200,
    xMaxHist = Math.max(p60, 1e-9),
    binWidth = xMaxHist / NBINS;
  const bulkX = sorted.filter((v) => v <= p60);
  const tailCount = n - bulkX.length,
    tailPct = (tailCount / n) * 100;
  const histTrace = {
    x: bulkX,
    type: "histogram",
    name: "λ histogram",
    xbins: { start: 0, end: xMaxHist, size: binWidth },
    marker: { color: "rgba(94,162,255,0.78)", line: { color: "rgba(94,162,255,1.0)", width: 0.4 } },
    opacity: 0.92,
    xaxis: "x",
    yaxis: "y",
    showlegend: false,
  };
  const ecdfTrace = {
    x: sorted,
    y: sorted.map((_, i) => (i + 1) / n),
    type: "scatter",
    mode: "lines",
    name: "ECDF",
    line: { width: 2.5, color: "#7c5cff" },
    xaxis: "x2",
    yaxis: "y2",
    showlegend: false,
  };
  const quartileMarkers = [
    { x: p25, color: "#34d399", label: "p25" },
    { x: p50, color: "#fbbf24", label: "median" },
    { x: p75, color: "#f87171", label: "p75" },
  ];
  const ecdfShapes = quartileMarkers.map((q) => ({
    type: "line",
    xref: "x2",
    yref: "y2",
    x0: q.x,
    x1: q.x,
    y0: 0,
    y1: 1,
    line: { color: q.color, width: 2, dash: "dash" },
  }));
  Plotly.newPlot(
    el,
    [histTrace, ecdfTrace],
    {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(15,23,42,0.85)",
      font: { color: "#cbd5e1" },
      margin: { l: 64, r: 32, t: 40, b: 56 },
      xaxis: { title: "λ eigenvalue (bulk)", gridcolor: "rgba(255,255,255,0.08)", range: [0, xMaxHist], domain: [0, 1], anchor: "y" },
      yaxis: { title: "count", gridcolor: "rgba(255,255,255,0.08)", domain: [0.56, 0.96], anchor: "x" },
      xaxis2: { title: "λ eigenvalue (full range)", gridcolor: "rgba(255,255,255,0.08)", range: [0, lambdaMax], domain: [0, 1], anchor: "y2" },
      yaxis2: { title: "ECDF", gridcolor: "rgba(255,255,255,0.08)", range: [0, 1], domain: [0, 0.42], anchor: "x2" },
      shapes: ecdfShapes,
      annotations: [
        {
          xref: "paper",
          yref: "paper",
          x: 0,
          y: 1.0,
          text: `<b>λ Histogram</b> · ${n.toLocaleString()} samples · clipped at p60=${p60.toFixed(4)}`,
          showarrow: false,
          align: "left",
          xanchor: "left",
          yanchor: "bottom",
          font: { color: "#e2e8f0", size: 12 },
        },
        {
          xref: "paper",
          yref: "paper",
          x: 0,
          y: 0.46,
          text: "<b>ECDF</b> · full λ range",
          showarrow: false,
          align: "left",
          xanchor: "left",
          yanchor: "bottom",
          font: { color: "#e2e8f0", size: 12 },
        },
        {
          xref: "paper",
          yref: "paper",
          x: 0.99,
          y: 0.99,
          text: `Tail (λ &gt; p60): ${tailCount} samples (${tailPct.toFixed(1)}%) up to λ=${lambdaMax.toFixed(3)}`,
          showarrow: false,
          align: "right",
          xanchor: "right",
          yanchor: "top",
          bgcolor: "rgba(15,23,42,0.85)",
          bordercolor: "rgba(124,92,255,0.6)",
          borderwidth: 1,
          font: { color: "#fbbf24", size: 11 },
        },
        ...quartileMarkers.map((q) => ({
          xref: "x2",
          yref: "paper",
          x: q.x,
          y: 0.42,
          text: `${q.label}=${q.x.toFixed(4)}`,
          showarrow: false,
          yanchor: "top",
          xanchor: "left",
          font: { color: q.color, size: 11 },
        })),
      ],
      showlegend: false,
    },
    { responsive: true, displayModeBar: false }
  );
}

const _vizCache = { audit: null, lambdas: null };
async function _getCachedAudit() {
  if (!_vizCache.audit) _vizCache.audit = await api("/api/prompts/audit");
  return _vizCache.audit;
}
async function _getCachedLambdas() {
  if (!_vizCache.lambdas) _vizCache.lambdas = await api("/api/prompts/lambdas");
  return _vizCache.lambdas;
}
function invalidateVizCache() {
  _vizCache.audit = null;
  _vizCache.lambdas = null;
}

async function renderSearchVisualizations(results) {
  const resultIds = new Set(results.map((item) => item.id));
  try {
    const [audit, lambdas] = await Promise.all([_getCachedAudit(), _getCachedLambdas()]);
    renderQueryManifold(audit, resultIds);
    renderQueryLambdaChart(lambdas);
  } catch (e) {
    console.warn("Search visualizations unavailable:", e);
  }
}

// FIX 2: support search / audit / drift views
function switchView(viewName) {
  const searchView = $("#search-view");
  const auditView = $("#audit-view");
  const driftView = $("#drift-view");
  const searchTab = $("#tab-search");
  const auditTab = $("#tab-audit");
  const driftTab = $("#tab-drift");

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
  requestAnimationFrame(resizeKnownPlots);
}

function resizeKnownPlots() {
  if (!window.Plotly) return;
  for (const id of [
    "query-manifold",
    "query-lambda-chart",
    "audit-query-manifold",
    "audit-spectral-fingerprint",
    "drift-chart",
    "drift-lambda-overlay",
    "drift-ecdf",
    "drift-pca",
    "drift-timeline",
  ]) {
    const node = document.getElementById(id);
    if (node && node.data) {
      try {
        window.Plotly.Plots.resize(node);
      } catch (_) {}
    }
  }
}

let _leafResizeTimer = null;
window.addEventListener("resize", () => {
  if (!window.Plotly) return;
  clearTimeout(_leafResizeTimer);
  _leafResizeTimer = setTimeout(resizeKnownPlots, 120);
});

function wireControls() {
  $("#alpha-slider")?.addEventListener("input", (e) => {
    setText("#alpha-value", Number(e.target.value).toFixed(2));
    clearTimeout(state.searchTimer);
    state.searchTimer = setTimeout(runSearch, 300);
  });

  $("#salience-slider")?.addEventListener("input", (e) => {
    setText("#salience-value", Number(e.target.value).toFixed(2));
    clearTimeout(state.searchTimer);
    state.searchTimer = setTimeout(runSearch, 300);
  });

  $("#tab-search")?.addEventListener("click", () => switchView("search"));
  $("#tab-audit")?.addEventListener("click", () => switchView("audit"));
  $("#tab-drift")?.addEventListener("click", () => switchView("drift"));

  $("#filter")?.addEventListener("input", () => {
    clearTimeout(state.searchTimer);
    state.searchTimer = setTimeout(runSearch, 350);
  });

  $("#filter")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") runSearch();
  });

  $("#topk-select")?.addEventListener("change", (e) => {
    setText("#topk-value", e.target.value);
    clearTimeout(state.searchTimer);
    state.searchTimer = setTimeout(runSearch, 300);
  });

  $("#refresh-audit-btn")?.addEventListener("click", () => {
    invalidateVizCache();
    loadAuditPanel();
  });

  $("#run-drift-btn")?.addEventListener("click", async () => {
    const btn = $("#run-drift-btn");
    if (!btn) return;
    btn.disabled = true;
    btn.textContent = "Loading…";
    await _fetchAndRenderDrift();
    btn.disabled = false;
    btn.textContent = "Run Drift Analysis";
  });

  $("#refresh-drift-btn")?.addEventListener("click", async () => {
    const btn = $("#refresh-drift-btn");
    if (!btn) return;
    btn.disabled = true;
    btn.textContent = "Refreshing…";
    await _fetchAndRenderDrift();
    btn.disabled = false;
    btn.textContent = "Refresh Drift";
  });

  $("#prompt-modal-close")?.addEventListener("click", () => $("#prompt-modal")?.classList.add("hidden"));
  $(".prompt-modal-backdrop")?.addEventListener("click", () => $("#prompt-modal")?.classList.add("hidden"));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") $("#prompt-modal")?.classList.add("hidden");
  });
}

(async function main() {
  wireControls();
  renderRecentSearches();
  try {
    const health = await api("/api/prompts/health");
    const el = $("#health");
    if (el) {
      el.textContent = health.status === "ready" ? "CVE Ready" : "CVE Warming...";
      el.className = health.status === "ready" ? "health ok" : "health";
    }
  } catch (e) {
    const el = $("#health");
    if (el) {
      el.textContent = "Backend offline";
      el.className = "health err";
    }
  }
})();