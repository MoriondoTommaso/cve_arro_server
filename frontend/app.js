// Minimal vanilla-JS frontend for the ArroSpace server.
// No build step. Hits /api/* directly.

const $ = (sel) => document.querySelector(sel);

// Internal defaults for parameters no longer exposed in the UI.
// Backend schemas use tau=0.75 and lam=0.7 as defaults.
const DEFAULT_TAU = 0.75;
const DEFAULT_LAM = 0.7;

const state = {
  datasets: [],
  selected: null,        // dataset summary
  windowSize: 200,       // rows per scroll page
  nextOffset: 0,
  loading: false,
  exhausted: false,
  sliceMode: false,      // true when explicit slice spec is in use
  spectralWeight: 0.5,
  searchQuery: "",
  rankedDatasetIds: null,
  searchTimer: null,
  tensorData: null,
  tensorColorMode: "grayscale",
  tensorPlayTimer: null,
  recentSearches: JSON.parse(
    localStorage.getItem("leafRecentSearches") || "[]"
  ),  
};

// Guard for legacy dataset-explorer code paths that reference DOM ids
// (`#metadata-out`, `#stats-out`, `#grid` table view, `#tensor-viewer`, etc.)
// that no longer exist in the LEAF UI.  These functions are dead in the
// current build but are kept in-source for now; they short-circuit if their
// target nodes are missing so accidental calls cannot throw.
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

async function runPromptNLSearch(query) {
  return api("/api/prompts/nl_search", {
    method: "POST",
    body: JSON.stringify({
      query,
      k: Number($("#topk-select")?.value || 50),
      tau: DEFAULT_TAU,
      lam: DEFAULT_LAM,
    }),
  });
}

async function refreshHealth() {
  const el = $("#health");
  try {
    const h = await api("/api/health");
    el.textContent = `zarr=${h.zarr_available} arrowspace=${h.arrowspace_backend} roots=${h.data_roots.join(",") || "—"}`;
    el.className = "health ok";
  } catch (e) {
    el.textContent = `health: ${e.message}`;
    el.className = "health err";
  }
}



function renderMetadata(metadata) {
  const id = metadata.id ?? "—";
  const root = metadata.root ?? "—";
  const path = metadata.path ?? "—";
  const kind = metadata.kind ?? "—";
  const shape = metadata.shape
    ? `[${metadata.shape.join(", ")}]`
    : "—";
  const dtype = metadata.dtype ?? "—";
  const chunks = metadata.chunks
    ? `[${metadata.chunks.join(", ")}]`
    : "—";

  $("#metadata-out").innerHTML = `
    <div class="signal-card">
      <span class="signal-label">Dataset ID</span>
      <strong>${id}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">Root</span>
      <strong>${root}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">Path</span>
      <strong>${path}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">Kind</span>
      <strong>${kind}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">Shape</span>
      <strong>${shape}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">Dtype</span>
      <strong>${dtype}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">Chunks</span>
      <strong>${chunks}</strong>
    </div>
  `;
}

async function loadMetadata(d) {
  if (!_legacyExplorerActive()) return;
  try {
    const m = await api(`/api/datasets/${encodeURIComponent(d.id)}/metadata`);

    renderMetadata(m);
  } catch (e) {
    $("#metadata-out").innerHTML = `
      <div class="signal-card">
        <span class="signal-label">Status</span>
        <strong>Error</strong>
      </div>
    `;
  }
}

function renderManifold(manifold) {
  const hasEmbedding =
    Array.isArray(manifold.embedding) &&
    Array.isArray(manifold.embedding[0]);

  if (!hasEmbedding) {
    $("#manifold-out").innerHTML = `
      <div class="signal-card">
        <span class="signal-label">Status</span>
        <strong>No embedding projection available</strong>
      </div>

      <div class="signal-card">
        <span class="signal-label">Source</span>
        <strong>${manifold.source ?? manifold.backend ?? "local"}</strong>
      </div>
    `;
    return;
  }

  const method = manifold.method ?? manifold.kind ?? "Unknown";
  const embeddingDim = manifold.embedding_dim ?? manifold.intrinsic_dim ?? "—";
  const embeddingPoints = manifold.embedding.length;
  const embeddingAxes = manifold.embedding[0].length;
  const source = manifold.source ?? manifold.backend ?? "local";

  $("#manifold-out").innerHTML = `
    <div class="signal-card">
      <span class="signal-label">Method</span>
      <strong>${method}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">Embedding Dim</span>
      <strong>${embeddingDim}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">Embedding Points</span>
      <strong>${embeddingPoints}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">Embedding Axes</span>
      <strong>${embeddingAxes}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">Source</span>
      <strong>${source}</strong>
    </div>

    <div class="manifold-plot-card">
      <div class="manifold-plot-title">
        <span>Embedding Projection</span>
        <strong>${embeddingPoints} points · ${embeddingAxes}D</strong>
      </div>

      <div class="manifold-canvas-stack">
        <canvas id="manifold-plot"></canvas>
        <canvas id="manifold-highlight"></canvas>
      </div>

      <div id="manifold-tooltip" class="manifold-tooltip">
        Hover points
      </div>
    </div>
  `;

  renderManifoldPlot(manifold);
}

function renderManifoldPlot(manifold) {
  const canvas = document.getElementById("manifold-plot");

  if (!canvas) return;

  const embedding = manifold.embedding;

  if (
    !Array.isArray(embedding) ||
    !Array.isArray(embedding[0])
  ) {
    canvas.replaceWith(createEmptyManifoldMessage());
    return;
  }

  const ctx = canvas.getContext("2d");
  const highlightCanvas = document.getElementById("manifold-highlight");
  const highlightCtx = highlightCanvas?.getContext("2d");

  if (highlightCanvas) {
    highlightCanvas.width = 420;
    highlightCanvas.height = 260;
  }

  canvas.width = 420;
  canvas.height = 260;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 1;

  ctx.strokeRect(
    20,
    20,
    canvas.width - 40,
    canvas.height - 40
  );

  ctx.strokeStyle = "rgba(255,255,255,0.05)";

  ctx.beginPath();
  ctx.moveTo(20, canvas.height / 2);
  ctx.lineTo(canvas.width - 20, canvas.height / 2);
  ctx.moveTo(canvas.width / 2, 20);
  ctx.lineTo(canvas.width / 2, canvas.height - 20);
  ctx.stroke();

  const xs = embedding.map((p) => Number(p[0]) || 0);
  const ys = embedding.map((p) => Number(p[1]) || 0);

  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);

  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  embedding.forEach((p) => {
    const x =
      ((p[0] - minX) / (maxX - minX || 1)) *
        (canvas.width - 40) +
      20;

    const y =
      ((p[1] - minY) / (maxY - minY || 1)) *
        (canvas.height - 40) +
      20;

    ctx.beginPath();

    ctx.arc(
      x,
      canvas.height - y,
      2.5,
      0,
      Math.PI * 2
    );

    const gradient = ctx.createRadialGradient(x, canvas.height - y, 0, x, canvas.height - y, 5);
    gradient.addColorStop(0, "rgba(124, 92, 255, 0.95)");
    gradient.addColorStop(1, "rgba(94, 162, 255, 0.15)");

    ctx.fillStyle = gradient;
    ctx.fill();
  });
  const tooltip = document.getElementById("manifold-tooltip");

  canvas.addEventListener("mousemove", (event) => {
    if (!tooltip) return;

    const rect = canvas.getBoundingClientRect();

    const mouseX =
      ((event.clientX - rect.left) / rect.width) * canvas.width;

    const mouseY =
      ((event.clientY - rect.top) / rect.height) * canvas.height;

    let closest = null;
    let bestDistance = Infinity;

    embedding.forEach((p, index) => {
      const px =
        ((p[0] - minX) / (maxX - minX || 1)) *
          (canvas.width - 40) +
        20;

      const py =
        canvas.height -
        (((p[1] - minY) / (maxY - minY || 1)) *
          (canvas.height - 40) +
         20);

      const distance = Math.hypot(mouseX - px, mouseY - py);

      if (distance < bestDistance) {
        bestDistance = distance;
        closest = { index, x: p[0], y: p[1] };
      }
    });
    
    if (highlightCtx && closest) {
      highlightCtx.clearRect(
        0,
        0,
        highlightCanvas.width,
        highlightCanvas.height
      );

      const hx =
        ((closest.x - minX) / (maxX - minX || 1)) *
          (highlightCanvas.width - 40) +
        20;

      const hy =
        highlightCanvas.height -
        (((closest.y - minY) / (maxY - minY || 1)) *
          (highlightCanvas.height - 40) +
          20);

      highlightCtx.beginPath();
      highlightCtx.arc(hx, hy, 7, 0, Math.PI * 2);
      highlightCtx.strokeStyle = "rgba(255,255,255,0.85)";
      highlightCtx.lineWidth = 2;
      highlightCtx.stroke();
    }

    if (!closest || bestDistance > 18) {
      tooltip.textContent = "Hover points";

    if (highlightCtx) {
      highlightCtx.clearRect(
        0,
        0,
        highlightCanvas.width,
        highlightCanvas.height
      );
    }

    return;
}

    tooltip.textContent =
      `#${closest.index} · x=${Number(closest.x).toFixed(3)} · y=${Number(closest.y).toFixed(3)}`;
  });
}

function createEmptyManifoldMessage() {
  const div = document.createElement("div");
  div.className = "signal-empty";
  div.textContent =
    "No embedding coordinates available for this dataset";
  return div;
}


async function loadManifold(d) {
  if (!_legacyExplorerActive()) return;
  try {
    const m = await api(`/api/datasets/${encodeURIComponent(d.id)}/manifold`);
    renderManifold(m);
  } catch (e) {
    console.warn("Manifold unavailable:", e);

    renderManifold({
      source: "local",
      embedding: null,
    });
  }
}

function renderLambdaBars(lambdas) {
  if (!Array.isArray(lambdas) || lambdas.length === 0) {
    return `<div class="signal-empty">Unavailable</div>`;
  }

  const max = Math.max(...lambdas.map(Number));

  return `
    <div class="lambda-bars">
      ${lambdas
        .map((v) => {
          const value = Number(v);
          const height = max > 0 ? (value / max) * 100 : 0;

          return `
            <div class="lambda-bar-wrap">
              <div
                class="lambda-bar"
                style="height: ${height}%"
                title="${value.toFixed(4)}"
              ></div>
            </div>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderArrowSpaceSignals(stats) {
  console.log("ArrowSpace stats received:", stats);

  const glNodes = stats.gl_nodes ?? "—";

  const glShape = Array.isArray(stats.gl_shape)
    ? stats.gl_shape.join(" × ")
    : "—";

  const lambdas = Array.isArray(stats.lambdas_sorted)
    ? stats.lambdas_sorted
    : [];

  $("#signal-gl").innerHTML = `
    <div class="signal-row">
      <span>Nodes</span>
      <strong>${glNodes}</strong>
    </div>

    <div class="signal-row">
      <span>Graph Shape</span>
      <strong>${glShape}</strong>
    </div>
  `;

  if (lambdas.length > 0) {
    const lambdaValues = lambdas.map((item) =>
      Array.isArray(item) ? Number(item[0]) : Number(item)
    );

    $("#signal-lambda").innerHTML = renderLambdaBars(lambdaValues);
  } else {
    $("#signal-lambda").innerHTML = `
      <div class="signal-empty">
        No eigenvalue distribution available
      </div>
    `;
  }
}


function renderStats(stats) {
  $("#stats-out").innerHTML = `
    <div class="signal-card">
      <span class="signal-label">Items</span>
      <strong>${stats.nitems ?? "—"}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">Features</span>
      <strong>${stats.nfeatures ?? "—"}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">Clusters</span>
      <strong>${stats.nclusters ?? "—"}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">GL Nodes</span>
      <strong>${stats.gl_nodes ?? "—"}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">GL Shape</span>
      <strong>${
        Array.isArray(stats.gl_shape)
          ? stats.gl_shape.join(" × ")
          : "—"
      }</strong>
    </div>
  `;
}


async function loadStats(d) {
  if (!_legacyExplorerActive()) return;
  let stats = {
    nitems: Array.isArray(d.shape) ? d.shape[0] : "—",
    nfeatures: Array.isArray(d.shape) ? d.shape[1] : "—",
    nclusters: "—",
    gl_nodes: Array.isArray(d.shape) ? d.shape[0] : "—",
    gl_shape: Array.isArray(d.shape) ? [d.shape[0], d.shape[0]] : null,
    lambdas_sorted: [],
  };

  try {
    const backendStats = await api(
      `/api/datasets/${encodeURIComponent(d.id)}/stats`
    );

    stats = {
      ...stats,
      ...backendStats,
    };
  } catch (e) {
    console.warn("Stats endpoint unavailable, trying index fallback:", e);

    try {
      const indexStats = await api(
        `/api/datasets/${encodeURIComponent(d.id)}/index`,
        {
          method: "POST",
          body: JSON.stringify({}),
        }
      );

      stats = {
        ...stats,
        ...indexStats,
        gl_nodes: indexStats.nitems ?? stats.gl_nodes,
        gl_shape: indexStats.nitems
          ? [indexStats.nitems, indexStats.nitems]
          : stats.gl_shape,
      };
    } catch (indexError) {
      console.warn("Index fallback unavailable:", indexError);
    }
  }

  try {
    const lambdas = await api(
      `/api/datasets/${encodeURIComponent(d.id)}/lambdas`
    );

    stats = {
      ...stats,
      ...lambdas,
    };
  } catch (e) {
    console.warn("Lambdas unavailable:", e);
  }

  renderStats(stats);
  renderArrowSpaceSignals(stats);
}

function rowsCountFromPage(page) {
  const rows =
    page?.rows ??
    page?.data?.rows ??
    page?.values ??
    page?.data?.values ??
    [];

  return Array.isArray(rows) ? rows.length : "preview";
}

async function loadNextPage() {
  if (!_legacyExplorerActive()) return;
  if (!state.selected || state.loading || state.exhausted || state.sliceMode) return;
  state.loading = true;
  $("#data-status").textContent = `loading rows ${state.nextOffset}…`;
  try {
    const url = `/api/datasets/${encodeURIComponent(state.selected.id)}/data?offset=${state.nextOffset}&limit=${state.windowSize}`;
    const page = await api(url);
    appendRows(page);
    if (page.next_offset == null) {
      state.exhausted = true;
      $("#data-status").textContent =
        `loaded ${page.total ?? rowsCountFromPage(page)} rows (end)`;
    } else {
      state.nextOffset = page.next_offset;
      $("#data-status").textContent = `loaded ${state.nextOffset} of ${page.total}`;
    }
  } catch (e) {
    $("#data-status").textContent = "";
    $("#grid").innerHTML = `
      <div class="error-screen">
        <h2>Unable to load data</h2>
        <p>${e.message}</p>
      </div>
    `;
  } finally {
    state.loading = false;
  }
}

async function applySlice() {
  if (!_legacyExplorerActive()) return;
  if (!state.selected) return;

  const spec = $("#slice-input").value.trim();

  if (!spec) {
    state.sliceMode = false;
    state.nextOffset = 0;
    state.exhausted = false;
    $("#grid").innerHTML = "";
    return loadNextPage();
  }

  state.sliceMode = true;
  $("#data-status").textContent = `slice ${spec} loading…`;

  try {
    const url =
      `/api/datasets/${encodeURIComponent(state.selected.id)}/slice?spec=${encodeURIComponent(spec)}`;

    const r = await api(url);

    $("#grid").innerHTML = "";

    const raw =
      r.data?.rows ??
      r.data?.values ??
      r.rows ??
      r.values ??
      r.data ??
      [];

    const rows =
      typeof raw === "function"
        ? []
        : raw;

    appendRows({
      rows: Array.isArray(rows) ? rows : [rows],
    });

    const outShape = r.out_shape ?? r.shape ?? r.data?.shape ?? [];

    $("#data-status").textContent =
      `slice ${spec} → shape [${
        Array.isArray(outShape) ? outShape.join(",") : "preview"
      }]`;
  } catch (e) {
    $("#data-status").textContent = "";

    $("#grid").innerHTML = `
      <div class="error-screen">
        <h2>Unable to load data</h2>
        <p>${e.message}</p>
      </div>
    `;
  }
}

function extractTensor(payload) {
  const data =
    payload?.data?.values ??
    payload?.data?.rows ??
    payload?.values ??
    payload?.rows ??
    payload?.data ??
    payload;

  if (!Array.isArray(data)) return null;

  if (
    Array.isArray(data[0]) &&
    Array.isArray(data[0][0])
  ) {
    return data;
  }

  return null;
}

function appendRows(payload) {
  const grid = $("#grid");
  const tensor = extractTensor(payload);

    if (tensor) {
      renderTensorViewer(tensor);
      return;
    }

  let rows = payload?.rows ?? payload?.data?.rows;

  if (!rows) {
    const values =
      payload?.values ??
      payload?.data?.values ??
      payload?.data ??
      [];

    if (Array.isArray(values[0]) && Array.isArray(values[0][0])) {
      rows = values[0];
    } else if (Array.isArray(values[0])) {
      rows = values;
    } else {
      rows = [values];
    }
  }

  if (!rows || rows.length === 0) {
    grid.innerHTML = `
      <div class="error-screen">
        <h2>No preview available</h2>
        <p>This dataset returned no displayable rows.</p>
      </div>
    `;
    return;
  }

  grid.innerHTML = "";

  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const tr = document.createElement("tr");

  const th0 = document.createElement("th");
  th0.className = "row-idx";
  th0.textContent = "#";
  tr.appendChild(th0);

  const firstRow = Array.isArray(rows[0]) ? rows[0] : [rows[0]];

  const maxPreviewCols = 20;
  for (let c = 0; c < Math.min(firstRow.length, maxPreviewCols); c++) {
    const th = document.createElement("th");
    th.textContent = c;
    tr.appendChild(th);
  }

  thead.appendChild(tr);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");

  rows.forEach((row, i) => {
    const tr = document.createElement("tr");

    const idx = document.createElement("td");
    idx.className = "row-idx";
    idx.textContent = i;
    tr.appendChild(idx);

    const cells = Array.isArray(row) ? row : [row];

    cells.slice(0, maxPreviewCols).forEach((v) => {
      const td = document.createElement("td");

      if (Array.isArray(v)) {
        td.appendChild(renderTensorPreview(v));
      } else {
        td.textContent = formatCell(v);
      }

      tr.appendChild(td);
    });

    tbody.appendChild(tr);
  });

  table.appendChild(tbody);
  grid.appendChild(table);
}

function formatCell(v) {
  if (v == null) return "";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return String(v);
    if (Number.isInteger(v)) return String(v);
    return v.toPrecision(6);
  }
  if (typeof v === "object" && "re" in v && "im" in v) {
    return `${v.re.toPrecision(4)}${v.im >= 0 ? "+" : ""}${v.im.toPrecision(4)}i`;
  }
  return String(v);
}

function isTensorDataset(dataset) {
  return (
    dataset &&
    dataset.kind === "array" &&
    Array.isArray(dataset.shape) &&
    dataset.shape.length === 3
  );
}

async function loadTensorPreview(dataset) {
  if (!_legacyExplorerActive()) return;
  if (!isTensorDataset(dataset)) return;

  $("#data-status").textContent = "loading tensor preview…";

  renderTensorViewer({
    sliceCount: dataset.shape[0],
    height: dataset.shape[1],
    width: dataset.shape[2],
  });

  await updateTensorSlice(0);

  $("#data-status").textContent = `tensor preview loaded`;
}

async function updateTensorSlice(sliceIndex) {
  if (!_legacyExplorerActive()) return;
  if (!state.selected || !isTensorDataset(state.selected)) return;

  try {
    const spec = encodeURIComponent(`${sliceIndex},:,:`);

    const url =
      `/api/datasets/${encodeURIComponent(state.selected.id)}/slice?spec=${spec}`;

    const response = await api(url);

    const matrix = normalizeMatrix(
      response?.data?.values ??
      response?.data?.rows ??
      response?.values ??
      response?.rows ??
      response?.data ??
      response
    );

    if (!matrix) {
      console.warn("Invalid tensor matrix:", response);
      $("#data-status").textContent = "tensor slice unavailable";
      return;
    }

    renderTensorSlice(matrix);

    $("#tensor-slice-value").textContent = String(sliceIndex);
    $("#data-status").textContent = `tensor slice ${sliceIndex}`;
  } catch (e) {
    console.warn("Tensor slice unavailable:", e);
    $("#data-status").textContent = "tensor slice error";
  }

}

function renderTensorHistogram(values) {
  const canvas = $("#tensor-histogram");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");

  canvas.width = 260;
  canvas.height = 90;

  const finiteValues = values.map(Number).filter(Number.isFinite);
  if (finiteValues.length === 0) return;

  const min = Math.min(...finiteValues);
  const max = Math.max(...finiteValues);
  const range = max - min || 1;

  const bins = 24;
  const counts = Array(bins).fill(0);

  finiteValues.forEach((value) => {
    const idx = Math.min(
      bins - 1,
      Math.floor(((value - min) / range) * bins)
    );

    counts[idx]++;
  });

  const maxCount = Math.max(...counts) || 1;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const barWidth = canvas.width / bins;

  counts.forEach((count, i) => {
    const barHeight =
      (count / maxCount) * (canvas.height - 12);

    ctx.fillStyle = "rgba(94,162,255,0.75)";

    ctx.fillRect(
      i * barWidth,
      canvas.height - barHeight,
      Math.max(1, barWidth - 2),
      barHeight
    );
  });
}


function normalizeMatrix(data) {
  if (!Array.isArray(data)) return null;

  if (
    Array.isArray(data[0]) &&
    Array.isArray(data[0][0])
  ) {
    return data[0];
  }

  if (Array.isArray(data[0])) {
    return data;
  }

  if (
    state.selected &&
    Array.isArray(state.selected.shape) &&
    data.every((v) => Number.isFinite(Number(v)))
  ) {
    const height = state.selected.shape[1];
    const width = state.selected.shape[2];

    if (data.length === height * width) {
      const matrix = [];

      for (let r = 0; r < height; r++) {
        matrix.push(data.slice(r * width, (r + 1) * width));
      }

      return matrix;
    }
  }

  return null;
}

function renderTensorSlice(matrix) {
  const wrap = $("#tensor-canvas-wrap");

  if (!wrap) return;

  let canvas = $("#tensor-canvas");

  if (!canvas) {
    canvas = document.createElement("canvas");
    canvas.id = "tensor-canvas";
    canvas.width = 256;
    canvas.height = 256;

    wrap.innerHTML = "";
    wrap.appendChild(canvas);
  }

  const ctx = canvas.getContext("2d");

  const h = matrix.length;
  const w = matrix[0].length;

  const tmp = document.createElement("canvas");
  tmp.width = w;
  tmp.height = h;

  const tmpCtx = tmp.getContext("2d");

  const img = tmpCtx.createImageData(w, h);

  const flatValues =
    matrix.flat().map(Number).filter(Number.isFinite);

  const min = Math.min(...flatValues);
  const max = Math.max(...flatValues);

  const range = max - min || 1;

  const mean =
    flatValues.reduce((a, b) => a + b, 0) /
    flatValues.length;

  $("#tensor-min").textContent = min.toFixed(2);
  $("#tensor-max").textContent = max.toFixed(2);
  $("#tensor-mean").textContent = mean.toFixed(2);

  renderTensorHistogram(flatValues);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 4;

      const normalized =
        ((Number(matrix[y][x]) - min) / range) * 255;

      const v = Math.max(
        0,
        Math.min(255, normalized)
      );

      if (state.tensorColorMode === "heatmap") {
        img.data[idx] = v;
        img.data[idx + 1] =
          Math.max(0, 180 - v / 2);
        img.data[idx + 2] =
          255 - v;
      } else {
        img.data[idx] = v;
        img.data[idx + 1] = v;
        img.data[idx + 2] = v;
      }

      img.data[idx + 3] = 255;
    }
  }

  tmpCtx.putImageData(img, 0, 0);

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.imageSmoothingEnabled = false;

  ctx.drawImage(
    tmp,
    0,
    0,
    canvas.width,
    canvas.height
  );
}


function hideTensorViewer() {
  state.tensorData = null;

  const viewer = $("#tensor-viewer");
  if (viewer) viewer.classList.add("hidden");
}

function renderTensorPreview(tensor) {
  const canvas = document.createElement("canvas");
  const size = 96;

  canvas.width = size;
  canvas.height = size;
  canvas.className = "tensor-preview";

  const ctx = canvas.getContext("2d");
  const matrix = normalizeMatrix(tensor);

  if (!matrix) return canvas;

  const h = matrix.length;
  const w = matrix[0]?.length || 1;

  const flatValues = matrix.flat().map(Number).filter(Number.isFinite);

  const min = Math.min(...flatValues);
  const max = Math.max(...flatValues);
  const range = max - min || 1;

  const img = ctx.createImageData(w, h);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 4;
      const normalized = ((Number(matrix[y][x]) - min) / range) * 255;
      const v = Math.max(0, Math.min(255, normalized));

      img.data[idx + 0] = v;
      img.data[idx + 1] = v;
      img.data[idx + 2] = v;
      img.data[idx + 3] = 255;
    }
  }

  const tmp = document.createElement("canvas");
  tmp.width = w;
  tmp.height = h;

  tmp.getContext("2d").putImageData(img, 0, 0);

  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tmp, 0, 0, size, size);

  return canvas;
}

function attachInfiniteScroll() {
  if (!_legacyExplorerActive()) return;
  $("#grid").addEventListener("scroll", (e) => {
    const el = e.currentTarget;
    if (el.scrollTop + el.clientHeight >= el.scrollHeight - 50) {
      loadNextPage();
    }
  });
}



async function buildDatasetIndex(datasetId) {
  return api(`/api/datasets/${encodeURIComponent(datasetId)}/index`, {
    method: "POST",
    body: JSON.stringify({}),
  });
}


function setText(sel, text) {
  const el = $(sel);
  if (el) el.textContent = text;
}

async function runSearch() {
  const query = ($("#filter")?.value ?? "").trim();

  state.searchQuery = query;

  if (
    query &&
    !state.recentSearches.includes(query)
  ) {
    state.recentSearches.unshift(query);

    state.recentSearches =
      state.recentSearches.slice(0, 8);

    renderRecentSearches();
    localStorage.setItem(
      "leafRecentSearches",
      JSON.stringify(state.recentSearches)
    );
  }

  if (!query) {
    $("#grid").innerHTML = `
      <div class="welcome-screen">
        <h2>LEAF Semantic Search</h2>
        <p>
          Try queries like “fitness app translation”
          or “software localization”.
        </p>
      </div>
    `;
    return;
  }

  try {
    setText("#health", "Searching...");

    const tau = DEFAULT_TAU;
    const alpha = Number($("#alpha-slider")?.value ?? 0.6);
    const lam = DEFAULT_LAM;
    const salience = Number($("#salience-slider")?.value ?? 0.3);

    $("#grid").innerHTML = `
      <div class="loading-screen">
        <div class="loader"></div>
        <p>Searching LEAF prompt space...</p>
      </div>
    `;

    const startedAt = performance.now();
    const result = await api("/api/prompts/nl_search", {
      method: "POST",

      body: JSON.stringify({
        query,
        k: Number($("#topk-select")?.value || 19),
        tau,
        alpha,
        lam,
        salience,
      }),
    });

    const latencyMs = Math.round(performance.now() - startedAt);

    renderPromptResults(result.results || [], {
      latencyMs,
      resultCount: result.result_count || 0,
      tau,
      alpha,
      lam,
      salience,
    });

    await renderSearchVisualizations(result.results || []);

    const healthEl = $("#health");
    if (healthEl) {
      healthEl.textContent = "LEAF Ready";
      healthEl.className = "health ok";
    }

    setText(
      "#search-mode-label",
      `α ${alpha.toFixed(2)} (spectral↔cosine) · sal ${salience.toFixed(2)}`
    );

    setText("#search-hint", `${result.result_count || 0} semantic results`);

  } catch (e) {
    console.error(e);

    const healthEl = $("#health");
    if (healthEl) {
      healthEl.textContent = "Search Error";
      healthEl.classList.add("err");
    }

    $("#grid").innerHTML = `
      <div class="error-screen">
        <h2>Search failed</h2>
        <p>${e.message}</p>
      </div>
    `;
    wirePromptCards();
  }
}

function renderPromptResults(results, analytics = {}) {
  state.lastResults = results;
  if (!results.length) {
    $("#grid").innerHTML = `
      <div class="welcome-screen">
        <h2>No results</h2>
        <p>No prompts matched your query.</p>
      </div>
    `;
    return;
  }

  $("#grid").innerHTML = `
    <div class="prompt-results">
      <div class="search-analytics">
        <div>
          <span>Latency</span>
          <strong>${analytics.latencyMs ?? "—"} ms</strong>
        </div>

        <div>
          <span>Results</span>
          <strong>${analytics.resultCount ?? results.length}</strong>
        </div>

        <div>
          <span>Alpha (spectral↔cosine)</span>
          <strong>${analytics.alpha?.toFixed?.(2) ?? "—"}</strong>
        </div>

        <div>
          <span>Salience</span>
          <strong>${analytics.salience?.toFixed?.(2) ?? "—"}</strong>
        </div>
      </div>
        ${results
          .map((item, index) =>
          renderPromptCard(item, index)
        )
        .join("")}
      </div>
    `;
    wirePromptCards();
}

function highlightQuery(text, query) {
  if (!query) return escapeHtml(text);

  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

  const regex = new RegExp(`(${escaped})`, "gi");

  return escapeHtml(text).replace(
    regex,
    '<mark class="prompt-highlight">$1</mark>'
  );
}


function renderPromptCard(item, index) {
  const content =
    item.content ||
    item.body ||
    "No content";

  return `
    <div class="prompt-result-card" data-index="${index}">
      <div class="prompt-result-header">
        <strong>
          ${item.title || item.id || "Untitled Prompt"}
        </strong>

        <div class="prompt-score-wrap">
          <span class="prompt-score">
            Score: ${(item.score ?? 0).toFixed(4)}
          </span>

          <div class="prompt-score-bar">
            <div
              class="prompt-score-fill"
              style="width:${Math.min(
                100,
                (item.score ?? 0) * 100
              )}%"
            ></div>
          </div>
        </div>
      </div>

      <p class="prompt-content">
        ${highlightQuery(content, state.searchQuery)}
      </p>

      <div class="prompt-card-actions">
        <button class="prompt-toggle" type="button">
        Expand
      </button>

      <button
        class="prompt-copy-btn"
        type="button"
        data-copy="${escapeHtml(content)}"
      >
        Copy
      </button>
    </div>

      <div class="prompt-result-meta">
        <span>ID: ${item.id ?? "—"}</span>

        <span>
          Salience:
          ${(item.salience ?? 0).toFixed(3)}
        </span>

        <span>
          Upvotes:
          ${item.upvotes ?? 0}
        </span>

        <span>
          Views:
          ${item.views ?? 0}
        </span>
      </div>
    </div>
  `;
}

function wirePromptCards() {
  document.querySelectorAll(".prompt-result-card").forEach((card) => {
    card.classList.add("collapsed");

    const btn = card.querySelector(".prompt-toggle");
    const idx = Number(card.dataset.index);

    card.addEventListener("dblclick", () => {
      openPromptModal(state.lastResults[idx]);
    });

    card.addEventListener("click", (e) => {
      if (e.target.classList.contains("prompt-toggle")) {
        return;
      }

    });

    if (!btn) return;

    btn.addEventListener("click", () => {
      card.classList.toggle("collapsed");

      btn.textContent = card.classList.contains("collapsed")
        ? "Expand"
        : "Collapse";
    });
  });
}

function openPromptModal(item) {
  const modal = $("#prompt-modal");
  const body = $("#prompt-modal-body");

  const content =
    item.content ||
    item.body ||
    "No content";

  body.innerHTML = `
    <h2 class="prompt-modal-title">
      ${escapeHtml(item.title || item.id || "Prompt")}
    </h2>

    <div class="prompt-modal-text">
      ${highlightQuery(content, state.searchQuery)}
    </div>

    <div class="prompt-modal-meta">
      <div class="prompt-modal-chip">
        Score ${(item.score ?? 0).toFixed(4)}
      </div>

      <div class="prompt-modal-chip">
        Salience ${(item.salience ?? 0).toFixed(3)}
      </div>

      <div class="prompt-modal-chip">
        Upvotes ${item.upvotes ?? 0}
      </div>

      <div class="prompt-modal-chip">
        Views ${item.views ?? 0}
      </div>

      <div class="prompt-modal-chip">
        ${item.id ?? "—"}
      </div>
    </div>
  `;

  modal.classList.remove("hidden");
}


function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}


async function ensureLeafReady() {
  try {
    const health = await api("/api/prompts/health");

    if (health.status === "ready") {
      return health;
    }

    $("#health").textContent = "Warming LEAF...";
    
    await api("/api/prompts/warm");

    for (let i = 0; i < 30; i++) {
      await new Promise((r) => setTimeout(r, 2000));

      const polled = await api("/api/prompts/health");

      if (polled.status === "ready") {
        return polled;
      }
    }

    throw new Error("LEAF warmup timeout");
  } catch (e) {
    console.error("LEAF readiness error:", e);
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

    $("#health").textContent = "LEAF Ready";
    $("#health").classList.add("ok");

    renderAuditHealth(health);
    renderAuditGraph(graph);
    renderAuditLambdas(audit, lambdas);
    renderAuditStats(audit);

    renderAuditManifold(audit);
    renderAuditSpectral(audit, lambdas);
    renderAuditPCA(audit);

    // The audit view was hidden when loadAuditPanel started. Although it is
    // visible by the time Plotly.newPlot fires, some browsers cache a zero
    // size from the first layout pass. Force a resize on the next animation
    // frame so the manifold + fingerprint cards fill their containers.
    requestAnimationFrame(() => {
      try {
        const ids = ["audit-query-manifold", "audit-spectral-fingerprint"];
        for (const id of ids) {
          const node = document.getElementById(id);
          if (node && window.Plotly && node.data) {
            window.Plotly.Plots.resize(node);
          }
        }
      } catch (resizeErr) {
        console.warn("Plotly resize failed:", resizeErr);
      }
    });

  } catch (e) {
    console.error(e);

    $("#health").textContent = "Audit Error";
    $("#health").classList.add("err");

    $("#audit-content").innerHTML = `
      <div class="error-screen">
        <h2>LEAF Audit Error</h2>
        <p>${e.message}</p>
      </div>
    `;
  }
}

function renderAuditHealth(health) {
  const container = $("#audit-health");

  if (!container) return;

  container.innerHTML = `
    <div class="signal-card">
      <span class="signal-label">Status</span>
      <strong>${health.status ?? "unknown"}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">Prompt Engine</span>
      <strong>${health.prompt_engine_ready}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">Embedder</span>
      <strong>${health.embedder_ready}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">Model</span>
      <strong>${health.embedder_model ?? "—"}</strong>
    </div>
  `;
}

function renderAuditGraph(graph) {
  const container = $("#audit-graph");

  if (!container) return;

  container.innerHTML = `
    <div class="signal-card">
      <span class="signal-label">Items</span>
      <strong>${graph.nitems ?? "—"}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">Features</span>
      <strong>${graph.nfeatures ?? "—"}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">Clusters</span>
      <strong>${graph.nclusters ?? "—"}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">GL Nodes</span>
      <strong>${graph.gl_nodes ?? "—"}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">GL Shape</span>
      <strong>
        ${Array.isArray(graph.gl_shape)
          ? graph.gl_shape.join(" × ")
          : "—"}
      </strong>
    </div>
  `;
}

function renderAuditLambdas(audit, lambdaData) {
  const container = $("#audit-lambdas");

  if (!container) return;

  // Spectral Diagnostics reads strictly from /api/prompts/audit (spectral_stats).
  // Fiedler / spectral gap come from the normalised Laplacian eigensolve on
  // the backend. The lambda sample count is taken from whichever array is
  // actually used to render the Spectral Fingerprint plot so the numbers
  // stay in sync.
  const spectral = (audit && audit.spectral_stats) || {};

  const readNumber = (...candidates) => {
    for (const v of candidates) {
      if (v === null || v === undefined) continue;
      const n = Number(v);
      if (Number.isFinite(n)) return n;
    }
    return null;
  };

  const fiedler = readNumber(
    spectral.fiedler_value,
    spectral.fiedlerValue,
    spectral.fiedler,
    spectral.lambda2,
  );
  const gap = readNumber(
    spectral.spectral_gap,
    spectral.spectralGap,
    spectral.gap,
  );

  const fmt = (v) => (v === null ? "—" : v.toFixed(6));
  const fiedlerColor =
    fiedler === null
      ? "#9aa6bd"
      : fiedler > 0.01
        ? "#34d399"
        : fiedler > 0.001
          ? "#facc15"
          : "#f87171";

  // Match the source the fingerprint plot uses: audit.eigenvalues when
  // present, else /api/prompts/lambdas. Falsy fallback to "—" only when no
  // array exists at all.
  const lambdaArr = Array.isArray(audit?.eigenvalues) && audit.eigenvalues.length
    ? audit.eigenvalues
    : Array.isArray(lambdaData?.lambdas)
      ? lambdaData.lambdas
      : null;
  const lambdaSamples = lambdaArr ? lambdaArr.length : null;

  // Prefer an explicit source label from the endpoint; otherwise summarise
  // build params or fall back to a neutral label. Never imply a normalised
  // Laplacian if the endpoint did not say so.
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
  } else if (audit?.graph_stats?.n_nodes) {
    source = `audit endpoint · n=${audit.graph_stats.n_nodes}`;
  } else {
    source = "audit endpoint";
  }

  container.innerHTML = `
    <div class="signal-card">
      <span class="signal-label">Fiedler Value</span>
      <strong style="color:${fiedlerColor}">
        ${fmt(fiedler)}
      </strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">Spectral Gap</span>
      <strong>${fmt(gap)}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">λ Samples</span>
      <strong>${lambdaSamples == null ? "—" : lambdaSamples}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">Source</span>
      <strong>${source}</strong>
    </div>
  `;
}

function renderAuditStats(audit) {
  const container = $("#audit-stats");

  if (!container) return;

  const stats = audit?.degree_stats || {};
  const graph = audit?.graph_stats || {};
  const sparsity = Number(graph.sparsity ?? 0);

  container.innerHTML = `
    <div class="signal-card">
      <span class="signal-label">Degree Mean</span>
      <strong>${(stats.mean ?? 0).toFixed(4)}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">Degree Std</span>
      <strong>${(stats.std ?? 0).toFixed(4)}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">Degree Min</span>
      <strong>${(stats.min ?? 0).toFixed(4)}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">Degree Max</span>
      <strong>${(stats.max ?? 0).toFixed(4)}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">Edges</span>
      <strong>${graph.n_edges ?? "—"}</strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">Sparsity</span>
      <strong>${sparsity ? sparsity.toFixed(6) : "—"}</strong>
    </div>
  `;
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

  const projected = points.map((p, i) => {
    const x =
      pad +
      ((p[0] - minX) / (maxX - minX || 1)) *
        (width - pad * 2);

    const y =
      height -
      pad -
      ((p[1] - minY) / (maxY - minY || 1)) *
        (height - pad * 2);

    return {
      x,
      y,
      id: ids[i] || `point_${i}`,
    };
  });

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
      const dx = p.x - mx;
      const dy = p.y - my;

      if (Math.sqrt(dx * dx + dy * dy) < 8) {
        hovered = p;
        break;
      }
    }

    const tooltip = $("#audit-pca-tooltip");

    if (hovered) {
      tooltip.textContent = hovered.id;
    } else {
      tooltip.textContent = "Hover points";
    }
  };
  canvas.onclick = (e) => {
    const rect = canvas.getBoundingClientRect();

    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    let clicked = null;

    for (const p of projected) {
      const dx = p.x - mx;
      const dy = p.y - my;

      if (Math.sqrt(dx * dx + dy * dy) < 10) {
        clicked = p;
        break;
      }
    }

    if (!clicked) return;

    switchView("search");

    const filter = $("#filter");
    if (filter) filter.value = clicked.id;

    setText("#search-hint", `Searching selected PCA prompt ${clicked.id}...`);

    runSearch();
  };
}

function renderAuditManifold3D(audit) {
  const el = document.getElementById("audit-manifold-3d");
  if (!el || !audit?.pca_2d || !window.Plotly) return;

  const points = audit.pca_2d;
  const degrees = audit.degrees || points.map(() => 1);

  const xs = points.map((p) => Number(p[0]));
  const ys = points.map((p) => Number(p[1]));
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;

  const gridSize = 40;
  const z = [];
  const sigma2 = Math.pow((rangeX + rangeY) * 0.08, 2);

  for (let gy = 0; gy < gridSize; gy++) {
    const row = [];
    for (let gx = 0; gx < gridSize; gx++) {
      const gxVal = minX + (rangeX * gx) / (gridSize - 1);
      const gyVal = minY + (rangeY * gy) / (gridSize - 1);
      let val = 0, wSum = 0;
      for (let i = 0; i < points.length; i += 10) {
        const dx = gxVal - xs[i], dy = gyVal - ys[i];
        const w = Math.exp(-(dx * dx + dy * dy) / sigma2);
        val += w * Number(degrees[i] || 1);
        wSum += w;
      }
      row.push(wSum > 1e-10 ? val / wSum : 0);
    }
    z.push(row);
  }

  Plotly.newPlot(el, [{
    z,
    type: "surface",
    colorscale: [[0,"#0f1f3d"],[0.4,"#1e3a8a"],[0.7,"#2563eb"],[1,"#ef4444"]],
    opacity: 0.92,
    colorbar: {
      title: "Node Degree (Lᵢᵢ)",
      titlefont: { color: "#cbd5e1", size: 11 },
      tickfont: { color: "#cbd5e1" }
    }
  }], {
    paper_bgcolor: "rgba(0,0,0,0)",
    scene: {
      xaxis: { title: "PCA 1", color: "#cbd5e1", gridcolor: "rgba(255,255,255,0.1)" },
      yaxis: { title: "PCA 2", color: "#cbd5e1", gridcolor: "rgba(255,255,255,0.1)" },
      zaxis: { title: "Degree", color: "#cbd5e1", gridcolor: "rgba(255,255,255,0.1)" },
      bgcolor: "rgba(15,23,42,0.65)",
      camera: { eye: { x: 1.5, y: 1.6, z: 0.75 } }
    },
    margin: { l: 0, r: 0, t: 10, b: 0 }
  }, { responsive: true, displayModeBar: false });
}

function renderAuditManifold(audit) {
  const el = document.getElementById("audit-query-manifold");
  const showError = (msg) => {
    if (!el) return;
    el.innerHTML = `<div class="manifold-empty-msg">${msg}</div>`;
  };

  if (!el) return;
  if (!window.Plotly) {
    showError("Plotly failed to load — cannot render Graph Laplacian Manifold.");
    return;
  }

  // Reflect server-side build params (eps/k/topk/p/sigma) regardless of which
  // rendering path runs below.
  const bpEl = document.getElementById("build-params-display");
  if (bpEl && audit?.build_params) {
    const bp = audit.build_params;
    const fmt = (v) => (v === null || v === undefined ? "None" : String(v));
    bpEl.innerHTML = `<code>eps=${fmt(bp.eps)} · k=${fmt(bp.k)} · topk=${fmt(bp.topk)} · p=${fmt(bp.p)} · sigma=${fmt(bp.sigma)}</code>`;
  }

  // Prefer the server-precomputed Laplacian manifold (z_grid + hubs). When
  // present, render it directly — exact parity with the reference Python
  // Plotly script. This bypasses the browser-side IDW path that was hitting
  // "degree variance ≈ 0" when degrees/pca_2d lengths disagreed.
  const lm = audit?.laplacian_manifold;
  if (lm && Array.isArray(lm.z_grid) && lm.z_grid.length) {
    renderServerLaplacianManifold(audit, lm, el);
    return;
  }

  // Fallback: old browser-side IDW rendering (kept for resilience).
  const points = Array.isArray(audit?.pca_2d) ? audit.pca_2d : null;
  const degrees = Array.isArray(audit?.degrees) ? audit.degrees : null;
  if (!points || !points.length) {
    showError("Graph Laplacian Manifold unavailable: missing pca_2d from /api/prompts/audit.");
    return;
  }
  if (!degrees || !degrees.length) {
    showError("Graph Laplacian Manifold unavailable: missing degrees from /api/prompts/audit.");
    return;
  }
  if (degrees.length !== points.length) {
    console.warn(
      `[audit] degrees/pca_2d length mismatch: ${degrees.length} vs ${points.length}; falling back to client IDW.`,
    );
  }
  renderQueryManifold(audit, new Set(), "#audit-query-manifold");

  try {
    const titleEl = el.parentElement?.querySelector("h3");
    if (titleEl) {
      const n = points.length;
      const ex = audit?.pca_explained_variance || [];
      const pc1 = ex[0] != null ? (ex[0] * 100).toFixed(1) : null;
      const pc2 = ex[1] != null ? (ex[1] * 100).toFixed(1) : null;
      const subtitle = pc1 && pc2
        ? ` — Node connectivity as proxy for local manifold curvature (PC1 ${pc1}%, PC2 ${pc2}%)`
        : "";
      titleEl.textContent = `Graph Laplacian Manifold (${n} nodes)${subtitle}`;
    }
  } catch (_) {}
}

// Render the server-precomputed Laplacian manifold (Plotly Surface + hub
// Scatter3d). Mirrors the reference Python script visual: dark scene, cubic
// griddata surface, projected contours, top-15% hubs as markers.
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
    contours: {
      z: {
        show: true,
        usecolormap: true,
        highlightcolor: "#ffffff",
        project: { z: true },
      },
    },
    lighting: {
      ambient: 0.65,
      diffuse: 0.85,
      specular: 0.18,
      roughness: 0.55,
    },
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
    marker: {
      size: 4.5,
      color: "#fbbf24",
      line: { color: "#ffffff", width: 0.6 },
      symbol: "diamond",
    },
  };

  const data = [surface, hubsTrace];

  Plotly.newPlot(
    el,
    data,
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
        camera: {
          eye: { x: 1.55, y: 1.55, z: 0.95 },
          up: { x: 0, y: 0, z: 1 },
        },
      },
      margin: { l: 0, r: 0, t: 20, b: 0 },
      legend: {
        font: { color: "#cbd5e1" },
        orientation: "h",
        x: 0,
        y: 1.04,
        bgcolor: "rgba(15,23,42,0)",
      },
    },
    {
      responsive: true,
      displayModeBar: false,
    }
  );

  try {
    const titleEl = el.parentElement?.querySelector("h3");
    if (titleEl) {
      const title = lm.title || `Graph Laplacian Manifold (${lm.n_nodes} nodes)`;
      const sub = lm.subtitle ? ` — ${lm.subtitle}` : "";
      titleEl.textContent = `${title}${sub}`;
    }
  } catch (_) {}
}

function renderAuditSpectral(audit, lambdas) {
  // Prefer eigenvalues stored on the audit payload if present, otherwise
  // fall back to the dedicated /api/prompts/lambdas endpoint. Either way the
  // audit panel is the single rendering pipeline (no per-search recompute).
  const source = (audit && Array.isArray(audit.eigenvalues) && audit.eigenvalues.length)
    ? { lambdas: audit.eigenvalues, n: audit.eigenvalues.length }
    : lambdas;
  renderQueryLambdaChart(source, "#audit-spectral-fingerprint");
}


function renderTensorViewer(tensor) {
  if (!_legacyExplorerActive()) return;
  state.tensorData = tensor;

  const grid = $("#grid");

  grid.innerHTML = `
    <div id="tensor-viewer" class="tensor-viewer">
      <div class="tensor-toolbar">
        <div class="tensor-info">
          <span>Slice Shape</span>
          <strong>${tensor.height} × ${tensor.width}</strong>
        </div>

        <div class="tensor-info">
          <span>Tensor Volume</span>
          <strong>${tensor.sliceCount} slices</strong>
        </div>

        <div class="tensor-mode-toggle">
          <button id="tensor-grayscale-btn" class="active">
            Grayscale
          </button>

          <button id="tensor-heatmap-btn">
            Heatmap
          </button>
        </div>

        <button id="tensor-play-btn" class="tensor-play-btn">
          Play
        </button>

        <div class="tensor-slider-wrap">
          <input
            id="tensor-slice-slider"
            type="range"
            min="0"
            max="${tensor.sliceCount - 1}"
            value="0"
          />

          <span id="tensor-slice-value">0</span>
        </div>
      </div>

      <div class="tensor-preview-layout">
        <div id="tensor-canvas-wrap" class="tensor-canvas-wrap"></div>

        <div class="tensor-slice-stats">
          <div class="tensor-stat-card">
            <span>Min</span>
            <strong id="tensor-min">—</strong>
          </div>

          <div class="tensor-stat-card">
            <span>Max</span>
            <strong id="tensor-max">—</strong>
          </div>

          <div class="tensor-stat-card">
            <span>Mean</span>
            <strong id="tensor-mean">—</strong>
          </div>

          <div class="tensor-stat-card tensor-histogram-card">
            <span>Distribution</span>
            <canvas id="tensor-histogram"></canvas>
          </div>
        </div>
      </div>
    </div>
  `;

  $("#tensor-slice-slider").addEventListener("input", (e) => {
    const sliceIndex = Number(e.target.value);

    $("#tensor-slice-value").textContent = String(sliceIndex);

    updateTensorSlice(sliceIndex);
  });

  $("#tensor-grayscale-btn").addEventListener("click", () => {
    state.tensorColorMode = "grayscale";

    $("#tensor-grayscale-btn").classList.add("active");
    $("#tensor-heatmap-btn").classList.remove("active");

    updateTensorSlice(Number($("#tensor-slice-slider").value));
  });

  $("#tensor-heatmap-btn").addEventListener("click", () => {
    state.tensorColorMode = "heatmap";

    $("#tensor-heatmap-btn").classList.add("active");
    $("#tensor-grayscale-btn").classList.remove("active");

    updateTensorSlice(Number($("#tensor-slice-slider").value));
  });

  const playBtn = $("#tensor-play-btn");

playBtn.addEventListener("click", () => {
  const slider = $("#tensor-slice-slider");

  if (!slider || !playBtn) return;

  if (state.tensorPlayTimer) {
    clearInterval(state.tensorPlayTimer);
    state.tensorPlayTimer = null;
    playBtn.textContent = "Play";
    return;
  }

  if (Number(slider.value) >= Number(slider.max)) {
    slider.value = "0";
    $("#tensor-slice-value").textContent = "0";
    updateTensorSlice(0);
  }

  playBtn.textContent = "Pause";

  state.tensorPlayTimer = setInterval(() => {
    const current = Number(slider.value);
    const max = Number(slider.max);
    const next = current + 1;

    if (next > max) {
      clearInterval(state.tensorPlayTimer);
      state.tensorPlayTimer = null;
      playBtn.textContent = "Play";
      return;
    }

    slider.value = String(next);
    $("#tensor-slice-value").textContent = String(next);
    updateTensorSlice(next);
  }, 500);
});

  updateTensorSlice(0);
}

function switchView(viewName) {
  const searchView = $("#search-view");
  const auditView = $("#audit-view");
  const searchTab = $("#tab-search");
  const auditTab = $("#tab-audit");

  if (viewName === "audit") {
    searchView.classList.add("hidden");
    searchView.classList.remove("active-view");
    auditView.classList.remove("hidden");
    auditView.classList.add("active-view");

    searchTab.classList.remove("active");
    auditTab.classList.add("active");

    loadAuditPanel();
    return;
  }

  auditView.classList.add("hidden");
  auditView.classList.remove("active-view");
  searchView.classList.remove("hidden");
  searchView.classList.add("active-view");

  auditTab.classList.remove("active");
  searchTab.classList.add("active");

  // Force a Plotly resize for the search-view charts after they become
  // visible again, otherwise some browsers keep a zero-sized layout cached
  // from the hidden state.
  requestAnimationFrame(() => {
    try {
      const ids = ["query-manifold", "query-lambda-chart"];
      for (const id of ids) {
        const node = document.getElementById(id);
        if (node && window.Plotly && node.data) {
          window.Plotly.Plots.resize(node);
        }
      }
    } catch (resizeErr) {
      console.warn("Plotly resize failed:", resizeErr);
    }
  });
}

// Resize Plotly charts whenever the window resizes, so all stable-height
// containers stay properly fitted on responsive layout changes.
let _leafResizeTimer = null;
window.addEventListener("resize", () => {
  if (!window.Plotly) return;
  clearTimeout(_leafResizeTimer);
  _leafResizeTimer = setTimeout(() => {
    const ids = [
      "query-manifold",
      "query-lambda-chart",
      "audit-query-manifold",
      "audit-spectral-fingerprint",
    ];
    for (const id of ids) {
      const node = document.getElementById(id);
      if (node && node.data) {
        try {
          window.Plotly.Plots.resize(node);
        } catch (_) { /* noop */ }
      }
    }
  }, 120);
});


function wireControls() {
  const alphaSlider = $("#alpha-slider");
  if (alphaSlider) {
    alphaSlider.addEventListener("input", (e) => {
      setText("#alpha-value", Number(e.target.value).toFixed(2));

      clearTimeout(state.searchTimer);
      state.searchTimer = setTimeout(runSearch, 300);
    });
  }

  const salienceSlider = $("#salience-slider");
  if (salienceSlider) {
    salienceSlider.addEventListener("input", (e) => {
      setText("#salience-value", Number(e.target.value).toFixed(2));

      clearTimeout(state.searchTimer);
      state.searchTimer = setTimeout(runSearch, 300);
    });
  }

  const searchTab = $("#tab-search");
  if (searchTab) {
    searchTab.addEventListener("click", () => switchView("search"));
  }

  const auditTab = $("#tab-audit");
  if (auditTab) {
    auditTab.addEventListener("click", () => switchView("audit"));
  }

  const filter = $("#filter");
  if (filter) {
    filter.addEventListener("input", () => {
      clearTimeout(state.searchTimer);
      state.searchTimer = setTimeout(runSearch, 350);
    });

    filter.addEventListener("keydown", (e) => {
      if (e.key === "Enter") runSearch();
    });
  }

  const topkSelect = $("#topk-select");
  if (topkSelect) {
    topkSelect.addEventListener("change", (e) => {
      setText("#topk-value", e.target.value);

      clearTimeout(state.searchTimer);
      state.searchTimer = setTimeout(runSearch, 300);
    });
  }

  $("#refresh-audit-btn")?.addEventListener("click", () => {
    invalidateVizCache();
    loadAuditPanel();
  });

  $("#prompt-modal-close")?.addEventListener("click", () => {
    $("#prompt-modal").classList.add("hidden");
  });

  $(".prompt-modal-backdrop")?.addEventListener("click", () => {
    $("#prompt-modal").classList.add("hidden");
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      $("#prompt-modal")?.classList.add("hidden");
    }
  });
}

renderRecentSearches();


function purgeStaleSliderCards() {
  // Defensive purge: if a cached or proxied copy of the old HTML is loaded,
  // remove the obsolete Tau / Result Diversity / MMR slider cards so the
  // rendered UI matches the current spec (alpha + salience visible only).
  // Never remove a card that still hosts a live control we depend on.
  const PROTECTED_IDS = [
    "alpha-slider",
    "salience-slider",
    "topk-select",
    "alpha-value",
    "salience-value",
    "topk-value",
  ];
  const isProtected = (card) =>
    PROTECTED_IDS.some((pid) => card.querySelector("#" + pid));

  const STALE_IDS = ["spectral-slider", "lam-slider", "tau-slider"];
  for (const id of STALE_IDS) {
    const el = document.getElementById(id);
    if (el) {
      const card = el.closest(".spectral-control") || el.parentElement;
      if (card && card.remove && !isProtected(card)) card.remove();
    }
  }
  const STALE_LABEL_RE =
    /(Tau spectral sharpness|Result Diversity|λ\s*MMR|MMR\s*λ|spectral sharpness)/i;
  document.querySelectorAll(".spectral-control").forEach((card) => {
    if (isProtected(card)) return;
    const label = card.querySelector("label, strong, h3, .spectral-label-row");
    if (label && STALE_LABEL_RE.test(label.textContent || "")) {
      card.remove();
    }
  });
}

(async function main() {
  purgeStaleSliderCards();
  wireControls();

  try {
    const health = await api("/api/prompts/health");

    const el = $("#health");
    if (el) {
      el.textContent =
        health.status === "ready" ? "LEAF Ready" : "LEAF Warming...";
      el.className =
        health.status === "ready" ? "health ok" : "health";
    }
  } catch (e) {
    console.error("Health check failed:", e);

    const el = $("#health");
    if (el) {
      el.textContent = "Backend offline";
      el.className = "health err";
    }
  }
})();

function renderRecentSearches() {
  const el = $("#recent-searches");

  if (!el) return;

  if (!state.recentSearches.length) {
    el.innerHTML = `
      <span class="signal-empty">
        No recent searches
      </span>
    `;
    return;
  }

  el.innerHTML = state.recentSearches
    .map(
      (query) => `
        <button
          class="recent-search-chip"
          type="button"
          data-query="${escapeHtml(query)}"
        >
          ${escapeHtml(query)}
        </button>
      `
    )
    .join("");

  el.querySelectorAll(".recent-search-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      const filter = $("#filter");
      if (filter) filter.value = btn.dataset.query;
      runSearch();
    });
  });
}

// Client-side cache for /api/prompts/audit and /api/prompts/lambdas.
// PCA on the 20k × 768 corpus is the slow part; the result does not change
// across queries, so a single in-tab cache avoids re-fetching it on every search.
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
  const resultIds = new Set(
    results.map((item) => item.id)
  );

  try {
    const [audit, lambdas] = await Promise.all([
      _getCachedAudit(),
      _getCachedLambdas(),
    ]);

    renderQueryManifold(audit, resultIds);
    renderQueryLambdaChart(lambdas);

  } catch (e) {
    console.warn("Search visualizations unavailable:", e);
  }
}

// Lightweight quantile (linear interpolation between samples).
function _quantile(sorted, q) {
  if (!sorted.length) return 0;
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (base + 1 < sorted.length) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }
  return sorted[base];
}

// Inverse-distance-weighted interpolation onto a regular grid. Cheap enough
// in JS for ~20k points × 60×60 grid using stride sampling, but we further
// limit the per-cell loop with a stride so the demo stays responsive.
function _idwGrid(xs, ys, zs, gridSize, power = 2) {
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const dx = (maxX - minX) || 1;
  const dy = (maxY - minY) || 1;
  const stride = Math.max(1, Math.floor(xs.length / 4000));
  // smoothing length proportional to grid cell size
  const sigma2 = Math.pow((dx + dy) * 0.12, 2);

  const gridX = new Array(gridSize);
  const gridY = new Array(gridSize);
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
      let num = 0;
      let den = 0;
      for (let i = 0; i < xs.length; i += stride) {
        const ddx = x - xs[i];
        const ddy = y - ys[i];
        const d2 = ddx * ddx + ddy * ddy + 1e-9;
        // Gaussian-like falloff (smoother than pure IDW at small radii)
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

  const points = audit.pca_2d;
  const ids = audit.ids || [];
  const explained = audit.pca_explained_variance || [];
  const pc1Pct = explained[0] != null ? (explained[0] * 100).toFixed(1) : "—";
  const pc2Pct = explained[1] != null ? (explained[1] * 100).toFixed(1) : "—";

  let degrees = audit.degrees;
  if (!Array.isArray(degrees) || degrees.length !== points.length) {
    degrees = new Array(points.length).fill(1);
  }

  const degArr = degrees.map(Number);


  const sortedDeg = [...degArr].sort((a, b) => a - b);
  const p05 = _quantile(sortedDeg, 0.05);
  const p95 = _quantile(sortedDeg, 0.95);
  const degRange = Math.max(p95 - p05, 1e-6);
  const degClipped = degArr.map(d =>
    (Math.min(Math.max(d, p05), p95) - p05) / degRange
  );

  const xs = points.map((p) => Number(p[0]));
  const ys = points.map((p) => Number(p[1]));

  const highlightedIndices = new Set(
    [...resultIds]
      .map((id) => {
        const m = String(id).match(/\d+/);
        return m ? Number(m[0]) : NaN;
      })
      .filter(Number.isFinite)
  );

  // Detect audit panel target so we can spend more budget on grid resolution
  // and a denser scatter overlay there. Search-side view stays light.
  const isAuditTarget = typeof target === "string"
    && target.includes("audit-query-manifold");
  const gridSize = isAuditTarget ? 110 : 80;
  const { z, gridX, gridY } = _idwGrid(xs, ys, degClipped, gridSize, 2);

  // z-range clipped to surface percentiles so the colorbar stays meaningful.
  const flatZ = z.flat().filter(Number.isFinite);
  const sortedZ = [...flatZ].sort((a, b) => a - b);
  const zMin = _quantile(sortedZ, 0.02);
  const zMax = _quantile(sortedZ, 0.98);
  const zRange = zMax - zMin;
  if (zRange < 1e-9) {
    // All degrees are effectively identical — surface would be flat and
    // colormap-degenerate. Show a clear message instead of leaving the card
    // blank, which is what users were seeing in the audit panel.
    if (typeof target === "string" && target.includes("audit-query-manifold")) {
      el.innerHTML =
        '<div class="manifold-empty-msg">Graph Laplacian Manifold: degree variance ≈ 0, cannot render surface.</div>';
    }
    return;
  }

  const surface = {
    x: gridX,
    y: gridY,
    z,
    type: "surface",
    colorscale: [
      [0.0, "#1e3a8a"],
      [0.25, "#2c5dff"],
      [0.5, "#f8fafc"],
      [0.75, "#fb7185"],
      [1.0, "#ef4444"],
    ],
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
    contours: {
      z: {
        show: true,
        usecolormap: true,
        highlightcolor: "#ffffff",
        project: { z: true },
      },
    },
    lighting: {
      ambient: 0.65,
      diffuse: 0.85,
      specular: 0.18,
      roughness: 0.55,
    },
    name: "Laplacian surface",
  };

  // Background point cloud — light, downsampled, sits below the surface to
  // anchor the manifold structure when the surface is rotated.
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
    marker: {
      size: 1.6,
      color: cloudIdx.map((i) => degClipped[i]),
      colorscale: surface.colorscale,
      cmin: zMin,
      cmax: zMax,
      opacity: 0.55,
      showscale: false,
    },
  };

  // High-degree hubs (top 2%) as 3D markers above the surface peak.
  const hubThreshold = _quantile(sortedDeg, 0.98);
  const hubIdxs = [];
  for (let i = 0; i < degArr.length; i++) {
    if (degArr[i] >= hubThreshold) hubIdxs.push(i);
  }
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
    marker: {
      size: 4.5,
      color: "#fbbf24",
      line: { color: "#ffffff", width: 0.6 },
      symbol: "diamond",
    },
  };

  const matched = [...highlightedIndices].filter(
    (i) => i >= 0 && i < points.length
  );
  const matchedTrace = {
    x: matched.map((i) => xs[i]),
    y: matched.map((i) => ys[i]),
    z: matched.map(() => hubLift + (zMax - zMin) * 0.08),
    type: "scatter3d",
    mode: "markers",
    name: "Matched prompts",
    text: matched.map((i) => ids[i] ?? `#${i}`),
    hoverinfo: "text",
    marker: {
      size: 5.5,
      color: "#ef4444",
      line: { color: "#ffffff", width: 1.1 },
      symbol: "circle",
    },
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
        xaxis: {
          title: `PC1 (${pc1Pct}%)`,
          gridcolor: "rgba(255,255,255,0.10)",
          zerolinecolor: "rgba(255,255,255,0.18)",
          color: "#cbd5e1",
        },
        yaxis: {
          title: `PC2 (${pc2Pct}%)`,
          gridcolor: "rgba(255,255,255,0.10)",
          zerolinecolor: "rgba(255,255,255,0.18)",
          color: "#cbd5e1",
        },
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
        camera: {
          eye: { x: 1.55, y: 1.55, z: 0.95 },
          up: { x: 0, y: 0, z: 1 },
        },
      },
      margin: { l: 0, r: 0, t: 20, b: 0 },
      legend: {
        font: { color: "#cbd5e1" },
        orientation: "h",
        x: 0,
        y: 1.04,
        bgcolor: "rgba(15,23,42,0)",
      },
    },
    {
      responsive: true,
      displayModeBar: false,
    }
  );
}

function renderQueryLambdaChart(data, target = "#query-lambda-chart") {
  const el = typeof target === "string" ? $(target) : target;
  if (!el || !data?.lambdas || !window.Plotly) return;

  const lambdas = data.lambdas
    .map(Number)
    .filter(Number.isFinite);

  if (!lambdas.length) return;

  const sorted = [...lambdas].sort((a, b) => a - b);
  const n = sorted.length;
  const lambdaMax = sorted[n - 1] || 1;

  const p25 = _quantile(sorted, 0.25);
  const p50 = _quantile(sorted, 0.50);
  const p75 = _quantile(sorted, 0.75);
  const p60 = _quantile(sorted, 0.60);

  // Histogram clipped at p60.
  const NBINS = 200;
  const xMaxHist = Math.max(p60, 1e-9);
  const binWidth = xMaxHist / NBINS;
  const bulkX = sorted.filter((v) => v <= p60);
  const tailCount = n - bulkX.length;
  const tailPct = (tailCount / n) * 100;

  const histTrace = {
    x: bulkX,
    type: "histogram",
    name: "λ histogram",
    xbins: { start: 0, end: xMaxHist, size: binWidth },
    marker: {
      color: "rgba(94,162,255,0.78)",
      line: { color: "rgba(94,162,255,1.0)", width: 0.4 },
    },
    opacity: 0.92,
    xaxis: "x",
    yaxis: "y",
    showlegend: false,
  };

  // ECDF over [0, lambdaMax].
  const ecdfY = sorted.map((_, i) => (i + 1) / n);
  const ecdfTrace = {
    x: sorted,
    y: ecdfY,
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

  // Histogram subplot title (drawn as paper-anchored annotation to avoid
  // colliding with the global plot title).
  const histTitle = {
    xref: "paper",
    yref: "paper",
    x: 0.0,
    y: 1.0,
    text: `<b>λ Histogram</b> · ${n.toLocaleString()} samples · clipped at p60=${p60.toFixed(4)}`,
    showarrow: false,
    align: "left",
    xanchor: "left",
    yanchor: "bottom",
    font: { color: "#e2e8f0", size: 12 },
  };

  // ECDF subplot title.
  const ecdfTitle = {
    xref: "paper",
    yref: "paper",
    x: 0.0,
    y: 0.46,
    text: "<b>ECDF</b> · full λ range",
    showarrow: false,
    align: "left",
    xanchor: "left",
    yanchor: "bottom",
    font: { color: "#e2e8f0", size: 12 },
  };

  const tailAnnotation = {
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
  };

  const quartileAnnotations = quartileMarkers.map((q) => ({
    xref: "x2",
    yref: "paper",
    x: q.x,
    y: 0.42,
    text: `${q.label}=${q.x.toFixed(4)}`,
    showarrow: false,
    yanchor: "top",
    xanchor: "left",
    font: { color: q.color, size: 11 },
  }));

  Plotly.newPlot(
    el,
    [histTrace, ecdfTrace],
    {
      // Plot title is omitted — the panel's <h3> already labels this chart.
      // The histogram and ECDF carry their own paper-anchored subtitles below.
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(15,23,42,0.85)",
      font: { color: "#cbd5e1" },
      margin: { l: 64, r: 32, t: 40, b: 56 },
      xaxis: {
        title: "λ eigenvalue (bulk)",
        gridcolor: "rgba(255,255,255,0.08)",
        range: [0, xMaxHist],
        domain: [0, 1],
        anchor: "y",
      },
      yaxis: {
        title: "count",
        gridcolor: "rgba(255,255,255,0.08)",
        domain: [0.56, 0.96],
        anchor: "x",
      },
      xaxis2: {
        title: "λ eigenvalue (full range)",
        gridcolor: "rgba(255,255,255,0.08)",
        range: [0, lambdaMax],
        domain: [0, 1],
        anchor: "y2",
      },
      yaxis2: {
        title: "ECDF",
        gridcolor: "rgba(255,255,255,0.08)",
        range: [0, 1],
        domain: [0, 0.42],
        anchor: "x2",
      },
      shapes: ecdfShapes,
      annotations: [
        histTitle,
        ecdfTitle,
        tailAnnotation,
        ...quartileAnnotations,
      ],
      showlegend: false,
    },
    {
      responsive: true,
      displayModeBar: false,
    }
  );
}