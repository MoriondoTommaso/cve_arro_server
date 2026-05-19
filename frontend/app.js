// CVE ArrowSpace frontend — adapted from LEAF Prompt Kaban
// No build step. Hits /api/* directly.

const $ = (sel) => document.querySelector(sel);

const DEFAULT_TAU = 0.75;
const DEFAULT_LAM = 0.7;

// Dataset IDs for the two CVE periods
const CVE_DATASET_A = "cve_99_14";
const CVE_DATASET_B = "cve_99_25";

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
  return !!(document.getElementById("metadata-out")
    || document.getElementById("stats-out")
    || document.getElementById("tensor-viewer")
    || document.getElementById("slice-input"));
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`);
  }
  return res.json();
}

async function refreshHealth() {
  const el = $("#health");
  try {
    const h = await api("/api/health");
    el.textContent = `zarr=${h.zarr_available} arrowspace=${h.arrowspace_backend} roots=${h.data_roots.join(",") || "\u2014"}`;
    el.className = "health ok";
  } catch (e) {
    el.textContent = `health: ${e.message}`;
    el.className = "health err";
  }
}

function setText(sel, text) {
  const el = $(sel);
  if (el) el.textContent = text;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function highlightQuery(text, query) {
  if (!query) return escapeHtml(text);
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const regex = new RegExp(`(${escaped})`, "gi");
  return escapeHtml(text).replace(regex, '<mark class="prompt-highlight">$1</mark>');
}

// ─── SEARCH ──────────────────────────────────────────────────────────────────

function renderRecentSearches() {
  const el = $("#recent-searches");
  if (!el) return;
  if (!state.recentSearches.length) {
    el.innerHTML = '<span class="signal-empty">No recent searches</span>';
    return;
  }
  el.innerHTML = state.recentSearches
    .map((q) => `<button class="recent-search-chip" type="button">${escapeHtml(q)}</button>`)
    .join("");
  el.querySelectorAll(".recent-search-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      const input = $("#filter");
      if (input) { input.value = btn.textContent; runSearch(); }
    });
  });
}

async function runSearch() {
  const query = ($("#filter")?.value ?? "").trim();
  state.searchQuery = query;

  if (query && !state.recentSearches.includes(query)) {
    state.recentSearches.unshift(query);
    state.recentSearches = state.recentSearches.slice(0, 8);
    renderRecentSearches();
  }

  if (!query) {
    $("#grid").innerHTML = `
      <div class="welcome-screen">
        <h2>CVE Semantic Search</h2>
        <p>Try queries like &ldquo;buffer overflow&rdquo; or &ldquo;remote code execution&rdquo;.</p>
      </div>`;
    return;
  }

  try {
    setText("#health", "Searching...");
    const alpha = Number($("#alpha-slider")?.value ?? 0.6);
    const salience = Number($("#salience-slider")?.value ?? 0.3);

    $("#grid").innerHTML = `
      <div class="loading-screen">
        <div class="loader"></div>
        <p>Searching CVE corpus...</p>
      </div>`;

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

    renderPromptResults(result.results || [], { latencyMs, resultCount: result.result_count || 0, alpha, salience });
    await renderSearchVisualizations(result.results || []);

    const healthEl = $("#health");
    if (healthEl) { healthEl.textContent = "CVE Ready"; healthEl.className = "health ok"; }
    setText("#search-mode-label", `\u03b1 ${alpha.toFixed(2)} (spectral\u2194cosine) \u00b7 sal ${salience.toFixed(2)}`);
    setText("#search-hint", `${result.result_count || 0} semantic results`);
  } catch (e) {
    console.error(e);
    const healthEl = $("#health");
    if (healthEl) { healthEl.textContent = "Search Error"; healthEl.classList.add("err"); }
    $("#grid").innerHTML = `<div class="error-screen"><h2>Search failed</h2><p>${e.message}</p></div>`;
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
        <div><span>Latency</span><strong>${analytics.latencyMs ?? "\u2014"} ms</strong></div>
        <div><span>Results</span><strong>${analytics.resultCount ?? results.length}</strong></div>
        <div><span>Alpha (spectral\u2194cosine)</span><strong>${analytics.alpha?.toFixed?.(2) ?? "\u2014"}</strong></div>
        <div><span>Salience</span><strong>${analytics.salience?.toFixed?.(2) ?? "\u2014"}</strong></div>
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
        <strong>${item.title || item.id || "Untitled CVE"}</strong>
        <div class="prompt-score-wrap">
          <span class="prompt-score">Score: ${(item.score ?? 0).toFixed(4)}</span>
          <div class="prompt-score-bar">
            <div class="prompt-score-fill" style="width:${Math.min(100,(item.score??0)*100)}%"></div>
          </div>
        </div>
      </div>
      <p class="prompt-content">${highlightQuery(content, state.searchQuery)}</p>
      <div class="prompt-card-actions">
        <button class="prompt-toggle" type="button">Expand</button>
        <button class="prompt-copy-btn" type="button" data-copy="${escapeHtml(content)}">Copy</button>
      </div>
      <div class="prompt-result-meta">
        <span>ID: ${item.id ?? "\u2014"}</span>
        <span>Salience: ${(item.salience ?? 0).toFixed(3)}</span>
        <span>Score: ${(item.score ?? 0).toFixed(4)}</span>
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
  document.querySelectorAll(".prompt-copy-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      navigator.clipboard?.writeText(btn.dataset.copy || "");
      btn.textContent = "Copied!";
      setTimeout(() => { btn.textContent = "Copy"; }, 1500);
    });
  });
}

function openPromptModal(item) {
  const modal = $("#prompt-modal");
  const body = $("#prompt-modal-body");
  const content = item.content || item.body || "No content";
  body.innerHTML = `
    <h2 class="prompt-modal-title">${escapeHtml(item.title || item.id || "CVE")}</h2>
    <div class="prompt-modal-text">${highlightQuery(content, state.searchQuery)}</div>
    <div class="prompt-modal-meta">
      <div class="prompt-modal-chip">Score ${(item.score ?? 0).toFixed(4)}</div>
      <div class="prompt-modal-chip">Salience ${(item.salience ?? 0).toFixed(3)}</div>
      <div class="prompt-modal-chip">${item.id ?? "\u2014"}</div>
    </div>`;
  modal.classList.remove("hidden");
}

// ─── SEARCH VISUALIZATIONS ───────────────────────────────────────────────────

async function renderSearchVisualizations(results) {
  try {
    const health = await api("/api/prompts/health");
    const lambdas = health.lambdas_sorted || [];
    renderQueryLambdaChart(lambdas, results);
    if (health.pca_coords) renderQueryManifold(health.pca_coords, results, health.laplacian_diag);
  } catch (e) {
    console.warn("Search viz unavailable:", e);
  }
}

function renderQueryLambdaChart(lambdas, results) {
  const el = $("#query-lambda-chart");
  if (!el || !window.Plotly) return;
  if (!lambdas.length) { el.innerHTML = '<div class="signal-empty">No eigenvalue data</div>'; return; }
  const vals = lambdas.map((v) => (Array.isArray(v) ? Number(v[0]) : Number(v)));
  Plotly.newPlot(el, [{
    x: vals, type: "histogram", nbinsx: 60,
    marker: { color: "rgba(94,162,255,0.7)" }, name: "\u03bb",
  }], {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#ccc", size: 11 },
    margin: { l: 40, r: 10, t: 10, b: 36 },
    xaxis: { title: "\u03bb eigenvalue", gridcolor: "rgba(255,255,255,0.07)" },
    yaxis: { title: "count", gridcolor: "rgba(255,255,255,0.07)" },
    showlegend: false,
  }, { responsive: true, displayModeBar: false });
}

function renderQueryManifold(pcaCoords, results, lapDiag) {
  const el = $("#query-manifold");
  if (!el || !window.Plotly) return;
  const xs = pcaCoords.map((p) => p[0]);
  const ys = pcaCoords.map((p) => p[1]);
  const colors = lapDiag ? lapDiag : xs.map(() => 0.5);
  Plotly.newPlot(el, [{
    x: xs, y: ys, mode: "markers",
    marker: { color: colors, colorscale: "Viridis", size: 4, opacity: 0.7 },
    type: "scatter", name: "corpus",
  }], {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#ccc", size: 11 },
    margin: { l: 30, r: 10, t: 10, b: 30 },
    xaxis: { gridcolor: "rgba(255,255,255,0.07)", title: "PC1" },
    yaxis: { gridcolor: "rgba(255,255,255,0.07)", title: "PC2" },
    showlegend: false,
  }, { responsive: true, displayModeBar: false });
}

// ─── AUDIT ───────────────────────────────────────────────────────────────────

function kpiHtml(label, value) {
  return `<div class="kpi-cell"><span class="signal-label">${label}</span><strong>${value}</strong></div>`;
}

async function runAudit() {
  const healthEl = $("#audit-health");
  const graphEl = $("#audit-graph");
  const statsEl = $("#audit-stats");
  const lambdasEl = $("#audit-lambdas");

  if (healthEl) healthEl.innerHTML = '<span class="signal-empty">Loading\u2026</span>';

  try {
    const health = await api("/api/prompts/health");

    if (healthEl) {
      healthEl.innerHTML =
        kpiHtml("STATUS", health.status ?? "\u2014") +
        kpiHtml("PROMPT ENGINE", String(health.prompt_engine ?? health.arrowspace_backend ?? false)) +
        kpiHtml("EMBEDDER", String(health.embedder ?? true)) +
        kpiHtml("MODEL", health.model ?? health.embedder_model ?? "sentence-transformers/all-MiniLM-L6-v2");
    }

    const items = health.nitems ?? health.corpus_size ?? "\u2014";
    const features = health.nfeatures ?? "\u2014";
    const clusters = health.nclusters ?? "\u2014";
    const glNodes = health.gl_nodes ?? items;
    const glShape = Array.isArray(health.gl_shape) ? health.gl_shape.join(" \u00d7 ") : `${glNodes} \u00d7 ${glNodes}`;

    if (graphEl) graphEl.innerHTML =
      kpiHtml("ITEMS", items) + kpiHtml("FEATURES", features) +
      kpiHtml("CLUSTERS", clusters) + kpiHtml("GL NODES", glNodes) +
      kpiHtml("GL SHAPE", glShape);

    const degMean = health.degree_mean ?? "\u2014";
    const degStd = health.degree_std ?? "\u2014";
    const degMin = health.degree_min ?? "\u2014";
    const degMax = health.degree_max ?? "\u2014";
    const edges = health.edges ?? "\u2014";
    const sparsity = health.sparsity ?? "\u2014";

    if (statsEl) statsEl.innerHTML =
      kpiHtml("DEGREE MEAN", typeof degMean === "number" ? degMean.toFixed(4) : degMean) +
      kpiHtml("DEGREE STD", typeof degStd === "number" ? degStd.toFixed(4) : degStd) +
      kpiHtml("DEGREE MIN", typeof degMin === "number" ? degMin.toFixed(4) : degMin) +
      kpiHtml("DEGREE MAX", typeof degMax === "number" ? degMax.toFixed(4) : degMax) +
      kpiHtml("EDGES", edges) + kpiHtml("SPARSITY", typeof sparsity === "number" ? sparsity.toFixed(6) : sparsity);

    const fiedler = health.fiedler_value ?? "\u2014";
    const specGap = health.spectral_gap ?? "\u2014";
    const nSamples = health.n_samples ?? health.nitems ?? "\u2014";
    const source = health.source ?? `graph_L (k=${health.k ?? 38}, eps=${health.eps ?? 1}, \u03c3=${health.sigma ?? "None"}, p=${health.p ?? 2})`;

    if (lambdasEl) lambdasEl.innerHTML =
      kpiHtml("FIEDLER VALUE", typeof fiedler === "number" ? fiedler.toFixed(6) : fiedler) +
      kpiHtml("SPECTRAL GAP", typeof specGap === "number" ? specGap.toFixed(6) : specGap) +
      kpiHtml("\u03bb SAMPLES", nSamples) + kpiHtml("SOURCE", source);

    // Manifold 3D
    if (health.pca_coords && health.laplacian_diag) renderAuditManifold(health);
    // Spectral fingerprint
    if (health.lambdas_sorted?.length) renderAuditSpectralFingerprint(health.lambdas_sorted);
    // PCA 2D
    if (health.pca_coords) renderAuditPca(health.pca_coords, health.laplacian_diag);

  } catch (e) {
    if (healthEl) healthEl.innerHTML = `<span class="signal-empty">Audit error: ${e.message}</span>`;
    console.error("Audit error:", e);
  }
}

function renderAuditManifold(health) {
  const el = $("#audit-query-manifold");
  if (!el || !window.Plotly) return;
  const coords = health.pca_coords;
  const diag = health.laplacian_diag;
  const xs = coords.map((p) => p[0]);
  const ys = coords.map((p) => p[1]);
  const zs = diag || xs.map(() => 0);
  Plotly.newPlot(el, [{
    type: "mesh3d",
    x: xs, y: ys, z: zs,
    intensity: zs, colorscale: "RdBu",
    opacity: 0.85, name: "Laplacian surface",
  }], {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#ccc", size: 11 },
    margin: { l: 0, r: 0, t: 20, b: 0 },
    scene: {
      xaxis: { title: `PC1 (${health.pca_var?.[0] ?? "?"}%)`, gridcolor: "rgba(255,255,255,0.1)" },
      yaxis: { title: `PC2 (${health.pca_var?.[1] ?? "?"}%)`, gridcolor: "rgba(255,255,255,0.1)" },
      zaxis: { title: "Curvature (L\u1D35\u1D35)", gridcolor: "rgba(255,255,255,0.1)" },
      bgcolor: "rgba(0,0,0,0)",
    },
  }, { responsive: true });
}

function renderAuditSpectralFingerprint(lambdas) {
  const el = $("#audit-spectral-fingerprint");
  if (!el || !window.Plotly) return;
  const vals = lambdas.map((v) => (Array.isArray(v) ? Number(v[0]) : Number(v))).filter(Number.isFinite);
  const sorted = [...vals].sort((a, b) => a - b);
  const n = sorted.length;
  const p60idx = Math.floor(n * 0.6);
  const clipped = sorted.slice(0, p60idx + 1);
  const ecdfX = sorted;
  const ecdfY = sorted.map((_, i) => (i + 1) / n);
  Plotly.newPlot(el, [
    { x: clipped, type: "histogram", nbinsx: 80, marker: { color: "rgba(94,162,255,0.7)" }, name: "\u03bb histogram", xaxis: "x", yaxis: "y" },
    { x: ecdfX, y: ecdfY, mode: "lines", line: { color: "#a78bfa", width: 2 }, name: "ECDF", xaxis: "x2", yaxis: "y2" },
  ], {
    grid: { rows: 2, columns: 1, pattern: "independent" },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#ccc", size: 11 },
    margin: { l: 50, r: 20, t: 20, b: 50 },
    xaxis: { title: "\u03bb eigenvalue (bulk)", gridcolor: "rgba(255,255,255,0.07)" },
    yaxis: { title: "count", gridcolor: "rgba(255,255,255,0.07)" },
    xaxis2: { title: "\u03bb eigenvalue (full range)", gridcolor: "rgba(255,255,255,0.07)" },
    yaxis2: { title: "ECDF", gridcolor: "rgba(255,255,255,0.07)" },
    showlegend: true,
    legend: { x: 0.8, y: 1, font: { color: "#ccc" } },
  }, { responsive: true, displayModeBar: false });
}

function renderAuditPca(coords, diag) {
  const canvas = $("#audit-pca");
  if (!canvas) return;
  canvas.width = canvas.offsetWidth || 420;
  canvas.height = 300;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const xs = coords.map((p) => p[0]);
  const ys = coords.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  coords.forEach((p) => {
    const x = ((p[0] - minX) / (maxX - minX || 1)) * (canvas.width - 40) + 20;
    const y = canvas.height - (((p[1] - minY) / (maxY - minY || 1)) * (canvas.height - 40) + 20);
    ctx.beginPath();
    ctx.arc(x, y, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(124,92,255,0.85)";
    ctx.fill();
  });
}

// ─── SPECTRAL DRIFT ───────────────────────────────────────────────────────────

async function runDriftAnalysis() {
  const kpiA = $("#drift-kpi-a");
  const kpiB = $("#drift-kpi-b");
  const kpiDelta = $("#drift-kpi-delta");
  const kpiGap = $("#drift-kpi-gap");

  [kpiA, kpiB, kpiDelta, kpiGap].forEach((el) => {
    if (el) el.innerHTML = '<span class="signal-empty">Loading\u2026</span>';
  });
  ["#drift-lambda-overlay","#drift-ecdf","#drift-pca","#drift-timeline"].forEach((sel) => {
    const el = $(sel);
    if (el) el.innerHTML = '<span class="signal-empty">Loading\u2026</span>';
  });

  let healthA = null, healthB = null;

  try {
    // Fetch lambdas from each dataset's stats/lambdas endpoint
    // Try dedicated per-dataset health or stats endpoints first,
    // falling back to the global /api/prompts/health for the active dataset.
    const [resA, resB] = await Promise.allSettled([
      fetchDatasetSpectral(CVE_DATASET_A),
      fetchDatasetSpectral(CVE_DATASET_B),
    ]);

    healthA = resA.status === "fulfilled" ? resA.value : null;
    healthB = resB.status === "fulfilled" ? resB.value : null;

    // If both failed, fall back to the global health endpoint for active dataset
    if (!healthA && !healthB) {
      const globalHealth = await api("/api/prompts/health");
      healthA = globalHealth;
      healthB = globalHealth;
    }

    renderDriftKpis(healthA, healthB, kpiA, kpiB, kpiDelta, kpiGap);

    const lambdasA = extractLambdas(healthA);
    const lambdasB = extractLambdas(healthB);

    if (lambdasA.length || lambdasB.length) {
      renderDriftLambdaOverlay(lambdasA, lambdasB);
      renderDriftEcdf(lambdasA, lambdasB);
    } else {
      ["#drift-lambda-overlay","#drift-ecdf"].forEach((sel) => {
        const el = $(sel);
        if (el) el.innerHTML = '<div class="signal-empty">No eigenvalue data available from the server. Ensure both datasets are indexed.</div>';
      });
    }

    const pcaA = healthA?.pca_coords ?? null;
    const pcaB = healthB?.pca_coords ?? null;
    if (pcaA || pcaB) renderDriftPca(pcaA, pcaB, healthA?.laplacian_diag, healthB?.laplacian_diag);
    else $("#drift-pca").innerHTML = '<div class="signal-empty">No PCA coordinates available.</div>';

    renderDriftTimeline(healthA, healthB);

  } catch (e) {
    console.error("Drift analysis error:", e);
    [kpiA, kpiB, kpiDelta, kpiGap].forEach((el) => {
      if (el) el.innerHTML = `<span class="signal-empty">Error: ${e.message}</span>`;
    });
  }
}

async function fetchDatasetSpectral(datasetId) {
  // Try multiple API paths in order
  const paths = [
    `/api/datasets/${encodeURIComponent(datasetId)}/stats`,
    `/api/datasets/${encodeURIComponent(datasetId)}/lambdas`,
    `/api/datasets/${encodeURIComponent(datasetId)}/health`,
    `/api/prompts/health?dataset=${encodeURIComponent(datasetId)}`,
  ];
  for (const path of paths) {
    try {
      const data = await api(path);
      if (data && (data.lambdas_sorted || data.fiedler_value !== undefined || data.nitems)) {
        return { ...data, _dataset: datasetId };
      }
    } catch (_) {
      // try next
    }
  }
  throw new Error(`No spectral data found for ${datasetId}`);
}

function extractLambdas(health) {
  if (!health) return [];
  const raw = health.lambdas_sorted || [];
  return raw.map((v) => (Array.isArray(v) ? Number(v[0]) : Number(v))).filter(Number.isFinite);
}

function wassersteinApprox(a, b) {
  // 1-Wasserstein via sorted difference (equal-weight empirical)
  if (!a.length || !b.length) return null;
  const sa = [...a].sort((x, y) => x - y);
  const sb = [...b].sort((x, y) => x - y);
  const n = Math.min(sa.length, sb.length);
  let sum = 0;
  for (let i = 0; i < n; i++) sum += Math.abs(sa[i] - sb[i]);
  return sum / n;
}

function ksStatistic(a, b) {
  if (!a.length || !b.length) return null;
  const sa = [...a].sort((x, y) => x - y);
  const sb = [...b].sort((x, y) => x - y);
  const allX = [...new Set([...sa, ...sb])].sort((x, y) => x - y);
  let maxDiff = 0;
  const na = sa.length, nb = sb.length;
  allX.forEach((x) => {
    const fa = sa.filter((v) => v <= x).length / na;
    const fb = sb.filter((v) => v <= x).length / nb;
    maxDiff = Math.max(maxDiff, Math.abs(fa - fb));
  });
  return maxDiff;
}

function renderDriftKpis(hA, hB, kpiA, kpiB, kpiDelta, kpiGap) {
  const fmt = (v, d = 4) => (typeof v === "number" ? v.toFixed(d) : (v ?? "\u2014"));

  if (kpiA) kpiA.innerHTML =
    kpiHtml("ITEMS", hA?.nitems ?? hA?.corpus_size ?? "\u2014") +
    kpiHtml("FEATURES", hA?.nfeatures ?? "\u2014") +
    kpiHtml("FIEDLER", fmt(hA?.fiedler_value)) +
    kpiHtml("SPECTRAL GAP", fmt(hA?.spectral_gap)) +
    kpiHtml("\u03bb SAMPLES", hA?.n_samples ?? hA?.nitems ?? "\u2014");

  if (kpiB) kpiB.innerHTML =
    kpiHtml("ITEMS", hB?.nitems ?? hB?.corpus_size ?? "\u2014") +
    kpiHtml("FEATURES", hB?.nfeatures ?? "\u2014") +
    kpiHtml("FIEDLER", fmt(hB?.fiedler_value)) +
    kpiHtml("SPECTRAL GAP", fmt(hB?.spectral_gap)) +
    kpiHtml("\u03bb SAMPLES", hB?.n_samples ?? hB?.nitems ?? "\u2014");

  const lambdasA = extractLambdas(hA);
  const lambdasB = extractLambdas(hB);
  const w1 = wassersteinApprox(lambdasA, lambdasB);
  const ks = ksStatistic(lambdasA, lambdasB);
  const hasBoth = lambdasA.length > 0 && lambdasB.length > 0;

  if (kpiDelta) kpiDelta.innerHTML =
    kpiHtml("W\u2081 DISTANCE", hasBoth ? (w1 !== null ? w1.toFixed(6) : "\u2014") : "\u2014 (no \u03bb data)") +
    kpiHtml("KS STATISTIC", hasBoth ? (ks !== null ? ks.toFixed(6) : "\u2014") : "\u2014 (no \u03bb data)") +
    kpiHtml("\u03bb COUNT A", lambdasA.length || "\u2014") +
    kpiHtml("\u03bb COUNT B", lambdasB.length || "\u2014");

  const fA = hA?.fiedler_value ?? null;
  const fB = hB?.fiedler_value ?? null;
  const gA = hA?.spectral_gap ?? null;
  const gB = hB?.spectral_gap ?? null;

  if (kpiGap) kpiGap.innerHTML =
    kpiHtml("\u03bb\u2082 PERIOD A", fmt(fA)) +
    kpiHtml("\u03bb\u2082 PERIOD B", fmt(fB)) +
    kpiHtml("\u03bb\u2082 DELTA", (fA !== null && fB !== null) ? fmt(fB - fA) : "\u2014") +
    kpiHtml("GAP DELTA", (gA !== null && gB !== null) ? fmt(gB - gA) : "\u2014");
}

function renderDriftLambdaOverlay(lambdasA, lambdasB) {
  const el = $("#drift-lambda-overlay");
  if (!el || !window.Plotly) return;

  const tracesA = lambdasA.length ? [{
    x: lambdasA, type: "histogram", nbinsx: 80, opacity: 0.6,
    marker: { color: "rgba(94,162,255,0.75)" }, name: "cve_99_14 (1999\u20132014)",
  }] : [];
  const tracesB = lambdasB.length ? [{
    x: lambdasB, type: "histogram", nbinsx: 80, opacity: 0.6,
    marker: { color: "rgba(255,165,0,0.75)" }, name: "cve_99_25 (1999\u20132025)",
  }] : [];

  Plotly.newPlot(el, [...tracesA, ...tracesB], {
    barmode: "overlay",
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#ccc", size: 11 },
    margin: { l: 50, r: 20, t: 20, b: 50 },
    xaxis: { title: "\u03bb eigenvalue", gridcolor: "rgba(255,255,255,0.07)" },
    yaxis: { title: "count", gridcolor: "rgba(255,255,255,0.07)" },
    legend: { x: 0.7, y: 1, font: { color: "#ccc" } },
  }, { responsive: true, displayModeBar: false });
}

function renderDriftEcdf(lambdasA, lambdasB) {
  const el = $("#drift-ecdf");
  if (!el || !window.Plotly) return;

  function ecdfTrace(vals, name, color) {
    if (!vals.length) return null;
    const sorted = [...vals].sort((a, b) => a - b);
    const n = sorted.length;
    return {
      x: sorted,
      y: sorted.map((_, i) => (i + 1) / n),
      mode: "lines",
      line: { color, width: 2 },
      name,
    };
  }

  const traces = [
    ecdfTrace(lambdasA, "cve_99_14 (1999\u20132014)", "rgba(94,162,255,0.9)"),
    ecdfTrace(lambdasB, "cve_99_25 (1999\u20132025)", "rgba(255,165,0,0.9)"),
  ].filter(Boolean);

  Plotly.newPlot(el, traces, {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#ccc", size: 11 },
    margin: { l: 50, r: 20, t: 20, b: 50 },
    xaxis: { title: "\u03bb eigenvalue", gridcolor: "rgba(255,255,255,0.07)" },
    yaxis: { title: "ECDF", gridcolor: "rgba(255,255,255,0.07)", range: [0, 1] },
    legend: { x: 0.6, y: 0.3, font: { color: "#ccc" } },
  }, { responsive: true, displayModeBar: false });
}

function renderDriftPca(pcaA, pcaB, diagA, diagB) {
  const el = $("#drift-pca");
  if (!el || !window.Plotly) return;

  const traces = [];
  if (pcaA) {
    traces.push({
      x: pcaA.map((p) => p[0]), y: pcaA.map((p) => p[1]),
      mode: "markers",
      marker: { color: diagA || pcaA.map(() => 0.5), colorscale: "Blues", size: 4, opacity: 0.7 },
      type: "scatter", name: "cve_99_14", xaxis: "x", yaxis: "y",
    });
  }
  if (pcaB) {
    traces.push({
      x: pcaB.map((p) => p[0]), y: pcaB.map((p) => p[1]),
      mode: "markers",
      marker: { color: diagB || pcaB.map(() => 0.5), colorscale: "Oranges", size: 4, opacity: 0.7 },
      type: "scatter", name: "cve_99_25", xaxis: "x2", yaxis: "y2",
    });
  }

  Plotly.newPlot(el, traces, {
    grid: { rows: 1, columns: 2, pattern: "independent" },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#ccc", size: 11 },
    margin: { l: 40, r: 20, t: 30, b: 50 },
    xaxis: { title: "PC1 (1999\u20132014)", gridcolor: "rgba(255,255,255,0.07)" },
    yaxis: { title: "PC2", gridcolor: "rgba(255,255,255,0.07)" },
    xaxis2: { title: "PC1 (1999\u20132025)", gridcolor: "rgba(255,255,255,0.07)" },
    yaxis2: { title: "PC2", gridcolor: "rgba(255,255,255,0.07)" },
    legend: { x: 0.85, y: 1, font: { color: "#ccc" } },
  }, { responsive: true, displayModeBar: false });
}

function renderDriftTimeline(hA, hB) {
  const el = $("#drift-timeline");
  if (!el || !window.Plotly) return;

  // Build a 2-point timeline from the two period endpoints.
  // When full yearly data is available (health.yearly_fiedler etc.),
  // this will automatically pick up more points.
  const labels = [];
  const fiedlerVals = [];
  const specGapVals = [];

  const yearly = hA?.yearly_fiedler || hB?.yearly_fiedler || null;
  if (yearly && Array.isArray(yearly)) {
    yearly.forEach((pt) => {
      labels.push(String(pt.year));
      fiedlerVals.push(pt.fiedler ?? null);
      specGapVals.push(pt.spectral_gap ?? null);
    });
  } else {
    // Fallback: two anchor points from the two period stats
    if (hA?.fiedler_value !== undefined) { labels.push("1999\u20132014"); fiedlerVals.push(hA.fiedler_value); specGapVals.push(hA.spectral_gap ?? null); }
    if (hB?.fiedler_value !== undefined) { labels.push("1999\u20132025"); fiedlerVals.push(hB.fiedler_value); specGapVals.push(hB.spectral_gap ?? null); }
  }

  if (!labels.length) {
    el.innerHTML = '<div class="signal-empty">No timeline data (fiedler_value not returned by server)</div>';
    return;
  }

  Plotly.newPlot(el, [
    { x: labels, y: fiedlerVals, mode: "lines+markers", line: { color: "rgba(94,162,255,0.9)", width: 2 }, marker: { size: 8 }, name: "Fiedler \u03bb\u2082" },
    { x: labels, y: specGapVals, mode: "lines+markers", line: { color: "rgba(255,165,0,0.9)", width: 2, dash: "dot" }, marker: { size: 8 }, name: "Spectral Gap" },
  ], {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#ccc", size: 11 },
    margin: { l: 50, r: 20, t: 20, b: 60 },
    xaxis: { title: "Period", gridcolor: "rgba(255,255,255,0.07)" },
    yaxis: { title: "Value", gridcolor: "rgba(255,255,255,0.07)" },
    legend: { x: 0.7, y: 1, font: { color: "#ccc" } },
  }, { responsive: true, displayModeBar: false });
}

// ─── ENSURE READY ─────────────────────────────────────────────────────────────

async function ensureReady() {
  try {
    const health = await api("/api/prompts/health");
    if (health.status === "ready") return health;
    const el = $("#health");
    if (el) el.textContent = "Warming up...";
    await api("/api/prompts/warm");
    for (let i = 0; i < 30; i++) {
      await new Promise((r) => setTimeout(r, 2000));
      const polled = await api("/api/prompts/health");
      if (polled.status === "ready") return polled;
    }
    throw new Error("Warmup timeout");
  } catch (e) {
    console.error("Readiness error:", e);
    return null;
  }
}

// ─── TABS ────────────────────────────────────────────────────────────────────

function switchTab(name) {
  document.querySelectorAll(".view").forEach((v) => v.classList.add("hidden"));
  document.querySelectorAll(".tab-button").forEach((b) => b.classList.remove("active"));
  const view = document.getElementById(`${name}-view`);
  if (view) view.classList.remove("hidden");
  const tab = document.getElementById(`tab-${name}`);
  if (tab) tab.classList.add("active");

  if (name === "audit") runAudit();
  if (name === "drift") runDriftAnalysis();
}

// ─── INIT ─────────────────────────────────────────────────────────────────────

function init() {
  // Tab wiring
  $("#tab-search")?.addEventListener("click", () => switchTab("search"));
  $("#tab-audit")?.addEventListener("click", () => switchTab("audit"));
  $("#tab-drift")?.addEventListener("click", () => switchTab("drift"));

  // Search
  const filterInput = $("#filter");
  if (filterInput) {
    filterInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") runSearch();
    });
    filterInput.addEventListener("input", () => {
      clearTimeout(state.searchTimer);
      state.searchTimer = setTimeout(runSearch, 600);
    });
  }

  // Sliders
  $("#alpha-slider")?.addEventListener("input", (e) => setText("#alpha-value", Number(e.target.value).toFixed(2)));
  $("#salience-slider")?.addEventListener("input", (e) => setText("#salience-value", Number(e.target.value).toFixed(2)));
  $("#topk-select")?.addEventListener("change", (e) => setText("#topk-value", e.target.value));

  // Audit refresh
  $("#refresh-audit-btn")?.addEventListener("click", runAudit);

  // Drift refresh
  $("#refresh-drift-btn")?.addEventListener("click", runDriftAnalysis);

  // Modal close
  $("#prompt-modal-close")?.addEventListener("click", () => $("#prompt-modal")?.classList.add("hidden"));
  $("#prompt-modal")?.querySelector(".prompt-modal-backdrop")?.addEventListener("click", () => $("#prompt-modal")?.classList.add("hidden"));

  // Boot
  refreshHealth();
  ensureReady().then((health) => {
    const el = $("#health");
    if (health?.status === "ready") {
      if (el) { el.textContent = "CVE Ready"; el.className = "health ok"; }
    } else if (health === null) {
      if (el) { el.textContent = "Server offline"; el.className = "health err"; }
    }
  });

  renderRecentSearches();
}

document.addEventListener("DOMContentLoaded", init);
