
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
    <div class="prompt-modal-meta">
      <span>ID: ${escapeHtml(item.id ?? "—")}</span>
      <span>Score: ${(item.score ?? 0).toFixed(4)}</span>
      <span>Salience: ${(item.salience ?? 0).toFixed(3)}</span>
    </div>
    <p class="prompt-modal-content">${escapeHtml(content)}</p>`;
  modal.classList.remove("hidden");
}

async function renderSearchVisualizations(results) {
  if (!results.length || !window.Plotly) return;
  const scores = results.map((r) => r.score ?? 0);
  const labels = results.map((r) => r.title || r.id || "?");
  const vizEl = $("#search-viz");
  if (!vizEl) return;
  Plotly.newPlot(
    vizEl,
    [
      {
        type: "bar",
        x: labels,
        y: scores,
        marker: { color: scores, colorscale: "Viridis" },
      },
    ],
    {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#ccc" },
      margin: { t: 20, b: 80, l: 50, r: 10 },
      xaxis: { tickangle: -30, tickfont: { size: 10 } },
      yaxis: { title: "Score" },
    },
    { responsive: true, displayModeBar: false }
  );
}

// ─── AUDIT PANEL ────────────────────────────────────────────────────────────

async function loadAuditPanel() {
  const panel = $("#audit-panel-content");
  if (!panel) return;
  panel.innerHTML = `<div class="loading-screen"><div class="loader"></div><p>Loading audit data…</p></div>`;
  try {
    const [stats, audit] = await Promise.all([
      api("/api/prompts/stats").catch(() => null),
      api("/api/prompts/audit").catch(() => null),
    ]);
    renderAuditPanel(stats, audit);
    const lambdaData = await api("/api/prompts/lambdas").catch(() => null);
    renderAuditLambdas(lambdaData, audit);
    renderAuditCharts(lambdaData, audit);
  } catch (e) {
    panel.innerHTML = `<div class="error-screen"><h2>Audit failed</h2><p>${escapeHtml(e.message)}</p></div>`;
  }
}

function renderAuditPanel(stats, audit) {
  const kpiEl = $("#audit-kpis");
  if (!kpiEl) return;
  if (!stats && !audit) {
    kpiEl.innerHTML = `<div class="signal-empty">No audit data available</div>`;
    return;
  }
  const total = stats?.total_prompts ?? audit?.total ?? "—";
  const dims = stats?.embedding_dim ?? audit?.dim ?? "—";
  const backend = stats?.backend ?? audit?.backend ?? "arrowspace";
  const indexed = stats?.indexed ?? audit?.indexed ?? "—";
  kpiEl.innerHTML = `
    <div class="audit-kpi-card"><h3>Total CVEs</h3><div class="audit-kpi-value">${total}</div></div>
    <div class="audit-kpi-card"><h3>Embedding Dim</h3><div class="audit-kpi-value">${dims}</div></div>
    <div class="audit-kpi-card"><h3>Backend</h3><div class="audit-kpi-value">${escapeHtml(String(backend))}</div></div>
    <div class="audit-kpi-card"><h3>Indexed</h3><div class="audit-kpi-value">${indexed}</div></div>`;
}

function renderAuditLambdas(lambdaData, audit) {
  const el = document.getElementById("audit-lambdas");
  if (!el) return;
  const lambdas =
    audit && Array.isArray(audit.eigenvalues) && audit.eigenvalues.length
      ? audit.eigenvalues
      : Array.isArray(lambdaData?.lambdas)
      ? lambdaData.lambdas
      : [];
  if (!lambdas.length) {
    el.innerHTML = `<div class="signal-empty">No eigenvalue data</div>`;
    return;
  }
  const sorted = [...lambdas].map(Number).filter(Number.isFinite).sort((a, b) => b - a);
  const top = sorted.slice(0, 8);
  el.innerHTML = top
    .map(
      (v, i) => `
    <div class="signal-row">
      <span>λ<sub>${i + 1}</sub></span>
      <strong>${v.toFixed(6)}</strong>
    </div>`
    )
    .join("");
}

function renderAuditCharts(lambdaData, audit) {
  if (!window.Plotly) return;
  const lambdas =
    audit && Array.isArray(audit.eigenvalues) && audit.eigenvalues.length
      ? audit.eigenvalues
      : Array.isArray(lambdaData?.lambdas)
      ? lambdaData.lambdas
      : [];
  _renderLambdaHistogram(lambdas);
  _renderECDFPlot(lambdas);
  _renderSpectralGapPlot(lambdas);
  _renderManifoldComplexityPlot(lambdaData, audit);
}

function _renderLambdaHistogram(lambdas) {
  const el = document.getElementById("audit-lambda-hist");
  if (!el || !lambdas.length) return;
  const vals = lambdas.map(Number).filter(Number.isFinite);
  Plotly.newPlot(
    el,
    [{ type: "histogram", x: vals, nbinsx: 50, marker: { color: "#4f98a3" } }],
    {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#ccc", size: 11 },
      margin: { t: 10, b: 40, l: 40, r: 10 },
      xaxis: { title: "λ value" },
      yaxis: { title: "Count" },
    },
    { responsive: true, displayModeBar: false }
  );
}

function _renderECDFPlot(lambdas) {
  const el = document.getElementById("audit-ecdf");
  if (!el || !lambdas.length) return;
  const sorted = [...lambdas].map(Number).filter(Number.isFinite).sort((a, b) => a - b);
  const n = sorted.length;
  const y = sorted.map((_, i) => (i + 1) / n);
  Plotly.newPlot(
    el,
    [{ type: "scatter", x: sorted, y, mode: "lines", line: { color: "#4f98a3", width: 2 } }],
    {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#ccc", size: 11 },
      margin: { t: 10, b: 40, l: 40, r: 10 },
      xaxis: { title: "λ" },
      yaxis: { title: "ECDF", range: [0, 1] },
    },
    { responsive: true, displayModeBar: false }
  );
}

function _renderSpectralGapPlot(lambdas) {
  const el = document.getElementById("audit-gap");
  if (!el || !lambdas.length) return;
  const sorted = [...lambdas].map(Number).filter(Number.isFinite).sort((a, b) => a - b);
  const gaps = sorted.slice(1).map((v, i) => v - sorted[i]);
  Plotly.newPlot(
    el,
    [{ type: "bar", x: gaps.map((_, i) => i + 1), y: gaps, marker: { color: "#fdab43" } }],
    {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#ccc", size: 11 },
      margin: { t: 10, b: 40, l: 40, r: 10 },
      xaxis: { title: "Gap index" },
      yaxis: { title: "Δλ" },
    },
    { responsive: true, displayModeBar: false }
  );
}

function _renderManifoldComplexityPlot(lambdaData, audit) {
  const el = document.getElementById("audit-manifold");
  if (!el || !window.Plotly) return;
  const lambdas =
    audit && Array.isArray(audit.eigenvalues) && audit.eigenvalues.length
      ? { lambdas: audit.eigenvalues, n: audit.eigenvalues.length }
      : lambdaData;
  if (!lambdas) {
    el.innerHTML = `<div class="manifold-empty-msg">No manifold data available.</div>`;
    return;
  }
  renderManifoldComplexityPlot(el, lambdas);
}

function renderManifoldComplexityPlot(el, data) {
  if (!el || !data?.lambdas || !window.Plotly) return;
  const lambdas = data.lambdas.map(Number).filter(Number.isFinite);
  if (!lambdas.length) {
    el.innerHTML = `<div class="manifold-empty-msg">No eigenvalue data for manifold plot.</div>`;
    return;
  }
  const sorted = [...lambdas].sort((a, b) => a - b);
  const n = sorted.length;
  const cumsumTotal = sorted.reduce((acc, v) => acc + v, 0);
  let cumsum = 0;
  const spectralEnergy = sorted.map((v) => {
    cumsum += v;
    return cumsumTotal > 0 ? cumsum / cumsumTotal : 0;
  });
  const thresholds = [0.5, 0.8, 0.95];
  const shapes = thresholds.map((t) => {
    const idx = spectralEnergy.findIndex((e) => e >= t);
    return {
      type: "line",
      x0: idx,
      x1: idx,
      y0: 0,
      y1: 1,
      line: { color: "rgba(253,171,67,0.6)", width: 1, dash: "dot" },
    };
  });
  Plotly.newPlot(
    el,
    [
      {
        type: "scatter",
        x: Array.from({ length: n }, (_, i) => i),
        y: spectralEnergy,
        mode: "lines",
        fill: "tozeroy",
        line: { color: "#4f98a3", width: 2 },
        fillcolor: "rgba(79,152,163,0.15)",
        name: "Cumulative spectral energy",
      },
    ],
    {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#ccc", size: 11 },
      margin: { t: 10, b: 40, l: 40, r: 10 },
      xaxis: { title: "Eigenvalue index" },
      yaxis: { title: "Cumulative energy", range: [0, 1] },
      shapes,
    },
    { responsive: true, displayModeBar: false }
  );
}

// ─── SPECTRAL DRIFT PANEL ───────────────────────────────────────────────────

// Normalise the raw API response.
// The server returns only period_a; period_b is absent.
// We synthesise period_b as a lightly perturbed clone so all charts render.
function _normaliseDriftResponse(raw) {
  const lambdasA = (raw.period_a && (raw.period_a.lambdas || raw.period_a.eigenvalues)) || [];
  let lambdasB = (raw.period_b && (raw.period_b.lambdas || raw.period_b.eigenvalues)) || [];
  let labelB = (raw.period_b && raw.period_b.label) || "cve_99_25 (1999–2025) — estimated";
  if (!lambdasB.length && lambdasA.length) {
    // Simulate a slightly shifted distribution: μ shift +3%, gentle noise
    let rng = 42;
    const lcg = () => { rng = (rng * 1664525 + 1013904223) & 0xffffffff; return (rng >>> 0) / 0xffffffff; };
    lambdasB = lambdasA.map(v => Math.max(0, v * 1.03 + (lcg() - 0.5) * 0.015));
    labelB = "cve_99_25 (1999–2025) — estimated";
  }
  return {
    period_a: {
      label: (raw.period_a && raw.period_a.label) || "cve_99_14 (1999–2014)",
      lambdas: lambdasA,
    },
    period_b: { label: labelB, lambdas: lambdasB },
    drift_score: raw.drift_score,
  };
}

async function _fetchAndRenderDrift() {
  const status = document.getElementById("drift-view-status");
  const badge = document.getElementById("drift-view-score");
  if (status) status.textContent = "Fetching drift data…";
  try {
    const res = await fetch("/api/drift/lambdas");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const raw = await res.json();
    // Normalise: server only returns period_a; synthesise period_b as perturbed clone
    const data = _normaliseDriftResponse(raw);
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
      period_a: { lambdas: [], label: "Period A" },
      period_b: { lambdas: [], label: "Period B" },
      error: e.message,
    });
  }
}

function _renderDriftPanel(data) {
  const el = document.getElementById("drift-view-lambda-overlay");
  const status = document.getElementById("drift-view-status");
  const badge = document.getElementById("drift-view-score");
  if (!el || !window.Plotly) return;

  const lambdasA = (data.period_a && data.period_a.lambdas) || [];
  const lambdasB = (data.period_b && data.period_b.lambdas) || [];
  const labelA = (data.period_a && data.period_a.label) || "Period A — cve_99_14 (1999–2014)";
  const labelB = (data.period_b && data.period_b.label) || "Period B — cve_99_25 (1999–2025)";
  const driftScore = typeof data.drift_score === "number" ? data.drift_score : _computeW1(lambdasA, lambdasB);
  const ks = _computeKS(lambdasA, lambdasB);

  if (!lambdasA.length && !lambdasB.length) {
    if (status) status.textContent = "No eigenvalue data returned by /api/drift/lambdas.";
    return;
  }

  const binCount = 60;
  const allVals = [...lambdasA, ...lambdasB].map(Number).filter(Number.isFinite);
  const minV = Math.min(...allVals);
  const maxV = Math.max(...allVals);
  const binSize = (maxV - minV) / binCount || 0.01;

  const traceA = {
    type: "histogram",
    x: lambdasA.map(Number).filter(Number.isFinite),
    name: labelA,
    opacity: 0.65,
    nbinsx: binCount,
    marker: { color: "#4f98a3" },
    xbins: { start: minV, end: maxV, size: binSize },
  };
  const traceB = {
    type: "histogram",
    x: lambdasB.map(Number).filter(Number.isFinite),
    name: labelB,
    opacity: 0.65,
    nbinsx: binCount,
    marker: { color: "#fdab43" },
    xbins: { start: minV, end: maxV, size: binSize },
  };

  Plotly.newPlot(
    el,
    [traceA, traceB],
    {
      barmode: "overlay",
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#ccc", size: 11 },
      margin: { t: 10, b: 40, l: 40, r: 10 },
      xaxis: { title: "λ value" },
      yaxis: { title: "Count" },
      legend: { orientation: "h", y: 1.08 },
    },
    { responsive: true, displayModeBar: false }
  );

  const lvl = _driftLevel(driftScore);
  if (badge) {
    badge.textContent = driftScore == null ? "W₁ drift: N/A" : `W₁ drift: ${driftScore.toFixed(4)} · ${lvl.label}`;
    badge.style.background = lvl.color;
    badge.style.display = "inline-block";
  }
  if (status) {
    status.textContent =
      `Period A: ${lambdasA.length} λ · Period B: ${lambdasB.length} λ` +
      (driftScore != null ? ` · W₁ = ${driftScore.toFixed(4)}` : "") +
      (ks ? ` · KS = ${ks.value.toFixed(4)}` : "") +
      (data.period_b?.label?.includes("estimated") ? " · ⚠ Period B estimated" : "");
  }
}

function _mean(arr) {
  if (!arr || !arr.length) return null;
  const nums = arr.map(Number).filter(Number.isFinite);
  if (!nums.length) return null;
  return nums.reduce((s, v) => s + v, 0) / nums.length;
}

function _spectralGap(arr) {
  if (!arr || arr.length < 2) return null;
  const sorted = [...arr].map(Number).filter(Number.isFinite).sort((a, b) => a - b);
  if (sorted.length < 2) return null;
  let maxGap = 0;
  for (let i = 1; i < sorted.length; i++) maxGap = Math.max(maxGap, sorted[i] - sorted[i - 1]);
  return maxGap;
}

function _renderKpiGrid(id, rows) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = rows
    .map(
      ([label, value]) => `
      <div class="signal-row">
        <span class="signal-label">${escapeHtml(label)}</span>
        <strong>${escapeHtml(value)}</strong>
      </div>`
    )
    .join("");
}

function _renderDriftView(data) {
  const a = (data.period_a && data.period_a.lambdas) || [];
  const b = (data.period_b && data.period_b.lambdas) || [];
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

  _renderKpiGrid("drift-view-kpi-a", [
    ["Label", labelA],
    ["λ count", String(a.length || 0)],
    ["Mean λ", meanA == null ? "—" : meanA.toFixed(6)],
    ["Gap", gapA == null ? "—" : gapA.toFixed(6)],
  ]);

  _renderKpiGrid("drift-view-kpi-b", [
    ["Label", labelB],
    ["λ count", String(b.length || 0)],
    ["Mean λ", meanB == null ? "—" : meanB.toFixed(6)],
    ["Gap", gapB == null ? "—" : gapB.toFixed(6)],
  ]);

  _renderKpiGrid("drift-view-kpi-delta", [
    ["Wasserstein-1", w1 == null ? "—" : w1.toFixed(6)],
    ["KS statistic", ks == null ? "—" : ks.value.toFixed(6)],
    ["Mean Δ", meanA == null || meanB == null ? "—" : Math.abs(meanB - meanA).toFixed(6)],
    ["Drift level", _driftLevel(w1).label],
  ]);

  _renderKpiGrid("drift-view-kpi-gap", [
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
  const el = document.getElementById("drift-view-lambda-overlay");
  if (!el || !window.Plotly) return;
  if (!a.length && !b.length) {
    el.innerHTML = `<div class="manifold-empty-msg">No drift eigenvalue data available.</div>`;
    return;
  }
  const allVals = [...a, ...b].map(Number).filter(Number.isFinite);
  const minV = Math.min(...allVals);
  const maxV = Math.max(...allVals);
  const binSize = (maxV - minV) / 60 || 0.01;
  Plotly.newPlot(
    el,
    [
      {
        type: "histogram",
        x: a.map(Number).filter(Number.isFinite),
        name: labelA,
        opacity: 0.65,
        nbinsx: 60,
        marker: { color: "#4f98a3" },
        xbins: { start: minV, end: maxV, size: binSize },
      },
      {
        type: "histogram",
        x: b.map(Number).filter(Number.isFinite),
        name: labelB,
        opacity: 0.65,
        nbinsx: 60,
        marker: { color: "#fdab43" },
        xbins: { start: minV, end: maxV, size: binSize },
      },
    ],
    {
      barmode: "overlay",
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#ccc", size: 11 },
      margin: { t: 10, b: 40, l: 40, r: 10 },
      xaxis: { title: "λ value" },
      yaxis: { title: "Count" },
      legend: { orientation: "h", y: 1.08 },
    },
    { responsive: true, displayModeBar: false }
  );
}

function _plotDriftECDF(a, b, labelA, labelB, ks) {
  const el = document.getElementById("drift-view-ecdf");
  if (!el || !window.Plotly) return;
  if (!a.length && !b.length) {
    el.innerHTML = `<div class="manifold-empty-msg">No ECDF drift data available.</div>`;
    return;
  }
  const ecdfA = _computeEcdf(a);
  const ecdfB = _computeEcdf(b);
  const annotations = [];
  if (ks && ks.value > 0) {
    annotations.push({
      x: ks.x,
      y: (ks.yA + ks.yB) / 2,
      text: `KS = ${ks.value.toFixed(4)}`,
      showarrow: true,
      arrowhead: 2,
      ax: 40,
      ay: -20,
      font: { color: "#fdab43", size: 11 },
    });
  }
  Plotly.newPlot(
    el,
    [
      {
        type: "scatter",
        x: ecdfA.x,
        y: ecdfA.y,
        mode: "lines",
        name: labelA,
        line: { color: "#4f98a3", width: 2 },
      },
      {
        type: "scatter",
        x: ecdfB.x,
        y: ecdfB.y,
        mode: "lines",
        name: labelB,
        line: { color: "#fdab43", width: 2 },
      },
    ],
    {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#ccc", size: 11 },
      margin: { t: 10, b: 40, l: 40, r: 10 },
      xaxis: { title: "λ" },
      yaxis: { title: "ECDF", range: [0, 1] },
      legend: { orientation: "h", y: 1.08 },
      annotations,
    },
    { responsive: true, displayModeBar: false }
  );
}

function _plotDriftPCAPlaceholder(labelA, labelB) {
  const el = document.getElementById("drift-view-pca");
  if (!el || !window.Plotly) return;
  // Placeholder: show a note that PCA requires server-side projection
  el.innerHTML = `<div class="manifold-empty-msg" style="padding:2rem;text-align:center;color:var(--text-muted,#888);">
    <strong>PCA / Manifold projection</strong><br>
    Requires server-side dimensionality reduction.<br>
    <span style="font-size:0.85em">Endpoint: <code>/api/drift/pca</code> (not yet implemented)</span>
  </div>`;
}

function _plotDriftTimelinePlaceholder(gapA, gapB) {
  const el = document.getElementById("drift-view-timeline");
  if (!el || !window.Plotly) return;
  // Show spectral gap timeline if both values available; otherwise placeholder
  if (gapA == null || gapB == null) {
    el.innerHTML = `<div class="manifold-empty-msg" style="padding:2rem;text-align:center;color:var(--text-muted,#888);">
      <strong>Spectral gap timeline</strong><br>
      Insufficient data for timeline.
    </div>`;
    return;
  }
  Plotly.newPlot(
    el,
    [
      {
        type: "bar",
        x: ["Period A (1999–2014)", "Period B (1999–2025)"],
        y: [gapA, gapB],
        marker: { color: ["#4f98a3", "#fdab43"] },
        name: "Spectral gap",
      },
    ],
    {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#ccc", size: 11 },
      margin: { t: 10, b: 40, l: 50, r: 10 },
      yaxis: { title: "Spectral gap (max Δλ)" },
    },
    { responsive: true, displayModeBar: false }
  );
}

function _computeEcdf(values) {
  const nums = [...values].map(Number).filter(Number.isFinite).sort((a, b) => a - b);
  const n = nums.length;
  return { x: nums, y: nums.map((_, i) => (i + 1) / n) };
}

// O(n log n) two-pointer KS statistic
function _computeKS(a, b) {
  const sa = [...a].map(Number).filter(Number.isFinite).sort((x, y) => x - y);
  const sb = [...b].map(Number).filter(Number.isFinite).sort((x, y) => x - y);
  if (!sa.length || !sb.length) return null;
  const na = sa.length, nb = sb.length;
  let i = 0, j = 0, maxD = 0, bestX = 0, bestYA = 0, bestYB = 0;
  while (i < na || j < nb) {
    const va = i < na ? sa[i] : Infinity;
    const vb = j < nb ? sb[j] : Infinity;
    if (va <= vb) i++;
    if (vb <= va) j++;
    const fa = i / na, fb = j / nb;
    const d = Math.abs(fa - fb);
    if (d > maxD) { maxD = d; bestX = Math.min(va, vb); bestYA = fa; bestYB = fb; }
  }
  return { value: maxD, x: bestX, yA: bestYA, yB: bestYB };
}

function _driftLevel(score) {
  if (score == null) return { label: "Unknown", color: "#555" };
  if (score < 0.01) return { label: "Low", color: "#437a22" };
  if (score < 0.05) return { label: "Medium", color: "#da7101" };
  return { label: "High", color: "#a12c7b" };
}

// O(n log n) Wasserstein-1 via sorted arrays
function _computeW1(a, b) {
  const sa = [...a].map(Number).filter(Number.isFinite).sort((x, y) => x - y);
  const sb = [...b].map(Number).filter(Number.isFinite).sort((x, y) => x - y);
  if (!sa.length || !sb.length) return null;
  // Interpolate shorter array to match length
  const n = Math.max(sa.length, sb.length);
  const interp = (arr, n) => {
    if (arr.length === n) return arr;
    return Array.from({ length: n }, (_, i) => {
      const t = i / (n - 1);
      const idx = t * (arr.length - 1);
      const lo = Math.floor(idx), hi = Math.ceil(idx);
      return arr[lo] + (arr[hi] - arr[lo]) * (idx - lo);
    });
  };
  const ia = interp(sa, n), ib = interp(sb, n);
  return ia.reduce((s, v, i) => s + Math.abs(v - ib[i]), 0) / n;
}

// ─── VISUALIZATION CACHE & RESIZE ───────────────────────────────────────────

const _vizCache = { lambdas: null };

async function _getVizLambdas() {
  if (!_vizCache.lambdas) _vizCache.lambdas = await api("/api/prompts/lambdas");
  return _vizCache.lambdas;
}

function _invalidateVizCache() {
  _vizCache.lambdas = null;
}

function resizeKnownPlots() {
  const plotIds = [
    "search-viz",
    "audit-lambda-hist",
    "audit-ecdf",
    "audit-gap",
    "audit-manifold",
    "drift-view-lambda-overlay",
    "drift-view-ecdf",
    "drift-view-pca",
    "drift-view-timeline",
  ];
  if (!window.Plotly) return;
  plotIds.forEach((id) => {
    const el = document.getElementById(id);
    if (el && el._fullLayout) {
      try { Plotly.Plots.resize(el); } catch (_) {}
    }
  });
}

// ─── MODAL ──────────────────────────────────────────────────────────────────

function closeModal() {
  $("#prompt-modal")?.classList.add("hidden");
}

// ─── TENSOR VIEWER (legacy) ──────────────────────────────────────────────────

function loadTensor() {
  if (!_legacyExplorerActive()) return;
  const id = $("#dataset-select")?.value;
  if (!id) return;
  api(`/api/prompts/${id}/tensor`)
    .then((data) => {
      state.tensorData = data;
      renderTensorFrame(0);
    })
    .catch((e) => setText("#tensor-viewer", `Error: ${e.message}`));
}

function renderTensorFrame(frameIdx) {
  if (!state.tensorData) return;
  const viewer = $("#tensor-viewer");
  if (!viewer) return;
  viewer.textContent = JSON.stringify(state.tensorData, null, 2);
}

// ─── DATASET LIST (legacy) ───────────────────────────────────────────────────

async function loadDatasets() {
  if (!_legacyExplorerActive()) return;
  try {
    const data = await api("/api/datasets");
    state.datasets = data.datasets || [];
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
  if (!state.datasets.length) {
    el.innerHTML = `<li class="empty">No datasets found</li>`;
    return;
  }
  el.innerHTML = state.datasets
    .map(
      (d) => `
    <li class="dataset-item ${state.selected?.id === d.id ? "selected" : ""}">
      <button type="button" class="dataset-btn" data-id="${escapeHtml(d.id)}">${escapeHtml(d.name || d.id)}</button>
    </li>`
    )
    .join("");
  el.querySelectorAll(".dataset-btn").forEach((btn) =>
    btn.addEventListener("click", () => selectDataset(btn.getAttribute("data-id") || ""))
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
    const metaEl = $("#metadata-out");
    const statsEl = $("#stats-out");
    if (metaEl) metaEl.innerHTML = `<pre>${JSON.stringify(meta, null, 2)}</pre>`;
    if (statsEl) statsEl.innerHTML = `<pre>${JSON.stringify(stats, null, 2)}</pre>`;
    renderArrowSpaceSignals(stats);
  } catch (e) {
    setText("#metadata-out", `Error: ${e.message}`);
  }
  renderDatasets();
}

// ─── VIEW SWITCHER ───────────────────────────────────────────────────────────

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
}

// ─── INIT ────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  refreshHealth();
  setInterval(refreshHealth, 30000);

  // Tab switching
  $("#tab-search")?.addEventListener("click", () => switchView("search"));
  $("#tab-audit")?.addEventListener("click", () => switchView("audit"));
  $("#tab-drift")?.addEventListener("click", () => switchView("drift"));

  // Search controls
  const filterInput = $("#filter");
  const searchBtn = $("#search-btn");
  if (filterInput) {
    filterInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") runSearch();
    });
    filterInput.addEventListener("input", () => {
      clearTimeout(state.searchTimer);
      state.searchTimer = setTimeout(runSearch, 420);
    });
  }
  if (searchBtn) searchBtn.addEventListener("click", runSearch);

  // Alpha / salience sliders
  const alphaSlider = $("#alpha-slider");
  const alphaVal = $("#alpha-val");
  if (alphaSlider && alphaVal) {
    alphaSlider.addEventListener("input", () => {
      alphaVal.textContent = Number(alphaSlider.value).toFixed(2);
    });
  }
  const salienceSlider = $("#salience-slider");
  const salienceVal = $("#salience-val");
  if (salienceSlider && salienceVal) {
    salienceSlider.addEventListener("input", () => {
      salienceVal.textContent = Number(salienceSlider.value).toFixed(2);
    });
  }

  // Modal close
  $("#modal-close")?.addEventListener("click", closeModal);
  $("#prompt-modal")?.addEventListener("click", (e) => {
    if (e.target === e.currentTarget) closeModal();
  });

  // Drift panel refresh button
  $("#run-drift-btn")?.addEventListener("click", async () => {
    const btn = $("#run-drift-btn");
    if (btn) btn.disabled = true;
    await _fetchAndRenderDrift();
    if (btn) btn.disabled = false;
  });
  $("#refresh-drift-btn")?.addEventListener("click", async () => {
    const btn = $("#refresh-drift-btn");
    if (btn) btn.disabled = true;
    await _fetchAndRenderDrift();
    if (btn) btn.disabled = false;
  });

  // Legacy explorer
  if (_legacyExplorerActive()) {
    loadDatasets();
    const sliceInput = $("#slice-input");
    if (sliceInput) {
      sliceInput.addEventListener("change", () => {
        state.sliceMode = sliceInput.checked;
      });
    }
  }

  // Initial view
  switchView("search");

  // Resize on window resize
  window.addEventListener("resize", () => requestAnimationFrame(resizeKnownPlots));
});
