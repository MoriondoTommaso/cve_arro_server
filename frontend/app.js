// Minimal vanilla-JS frontend for the ArroSpace server.
// No build step. Hits /api/* directly.

const $ = (sel) => document.querySelector(sel);

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
};

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
    el.textContent = `zarr=${h.zarr_available} arrowspace=${h.arrowspace_backend} roots=${h.data_roots.join(",") || "—"}`;
    el.className = "health ok";
  } catch (e) {
    el.textContent = `health: ${e.message}`;
    el.className = "health err";
  }
}

async function refreshDatasets() {
  const data = await api("/api/datasets");
  state.datasets = data.datasets;
  $("#dataset-count").textContent = state.datasets.length;
  renderDatasetList();
}

function renderDatasetList() {
  const ul = $("#dataset-list");
  ul.innerHTML = "";
  const f = $("#filter").value.toLowerCase();
  let visibleCount = 0;
  const datasetsToRender = state.rankedDatasetIds
    ? [...state.datasets].sort((a, b) => {
        const aRank = state.rankedDatasetIds.indexOf(a.id);
        const bRank = state.rankedDatasetIds.indexOf(b.id);

        return (
          (aRank === -1 ? Number.MAX_SAFE_INTEGER : aRank) -
          (bRank === -1 ? Number.MAX_SAFE_INTEGER : bRank)
        );
      })
    : state.datasets;

  for (const d of datasetsToRender) {
    if (f && !d.id.toLowerCase().includes(f)) continue;
    visibleCount++;
    const li = document.createElement("li");
    li.dataset.id = d.id;
    if (state.selected && state.selected.id === d.id) li.classList.add("active");
    const idLine = document.createElement("div");
    idLine.textContent = d.id;
    idLine.className = "ds-title";
    const meta = document.createElement("div");
    meta.className = "ds-shape";
    meta.textContent = `[${d.shape.join(",")}] · ${d.dtype || "—"}`;

    const badge = document.createElement("span");
    badge.className = `ds-badge ${d.kind}`;
    badge.textContent = d.kind;

    const scoreBadge = document.createElement("span");
    scoreBadge.className = "ds-score";

    if (state.rankedDatasetIds) {
      const rank = state.rankedDatasetIds.indexOf(d.id);

      scoreBadge.textContent =
        rank >= 0 ? `rank #${rank + 1}` : "";
    }

    li.appendChild(idLine);
    li.appendChild(meta);
    li.appendChild(badge);
    if (scoreBadge.textContent) {
      li.appendChild(scoreBadge);
    }
    li.addEventListener("click", () => selectDataset(d));
    ul.appendChild(li);
  }
  if (visibleCount === 0) {
    ul.innerHTML = `
      <li class="empty-list">
        No datasets found
      </li>
  `;
}

$("#filter-status").textContent =
  f
    ? `${visibleCount} result${visibleCount === 1 ? "" : "s"} for "${f}"`
    : "All datasets";
}

function resetMetrics() {
  $("#metric-kind").textContent = "—";
  $("#metric-shape").textContent = "—";
  $("#metric-dtype").textContent = "—";
}

async function selectDataset(d) {
  state.selected = d;
  state.nextOffset = 0;
  state.exhausted = false;
  state.sliceMode = false;
  if (state.tensorPlayTimer) {
    clearInterval(state.tensorPlayTimer);
    state.tensorPlayTimer = null;
  }
  $("#dataset-title").textContent = d.id;
  $("#copy-id-btn").disabled = false;
  $("#copy-id-btn").disabled = false;
  $("#metric-kind").textContent = d.kind;
  $("#metric-shape").textContent =
    `[${d.shape.join(", ")}]`;
  $("#metric-dtype").textContent =
    d.dtype || "—";
  $("#slice-input").value = "";
  $("#grid").innerHTML = `
  <div class="loading-screen">
    <div class="loader"></div>
    <p>Loading dataset...</p>
  </div>
`;
  $("#data-status").textContent = "loading…";
  renderDatasetList();
  try {
    await Promise.all([loadMetadata(d), loadManifold(d), loadStats(d)]);
  } catch (e) {
    console.warn(e);
  }

  if (isTensorDataset(d)) {
    await loadTensorPreview(d);
    return;
  }

  if (d.kind === "array") {
    await loadNextPage();
  } else {
    $("#data-status").textContent = `(${d.kind} — no data view)`;
  }}

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
  try {
    const m = await api(`/api/datasets/${encodeURI(d.id)}/metadata`);

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
  if (!state.selected || state.loading || state.exhausted || state.sliceMode) return;
  state.loading = true;
  $("#data-status").textContent = `loading rows ${state.nextOffset}…`;
  try {
    const url = `/api/datasets/${encodeURI(state.selected.id)}/data?offset=${state.nextOffset}&limit=${state.windowSize}`;
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
  if (!state.selected || !isTensorDataset(state.selected)) return;

  try {
    const spec = encodeURIComponent(`${sliceIndex},:,:`);

    const url =
      `/api/datasets/${encodeURI(state.selected.id)}/slice?spec=${spec}`;

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
  $("#grid").addEventListener("scroll", (e) => {
    const el = e.currentTarget;
    if (el.scrollTop + el.clientHeight >= el.scrollHeight - 50) {
      loadNextPage();
    }
  });
}

function buildSearchVector(query) {
  const vector = Array(8).fill(0);

  for (let i = 0; i < query.length; i++) {
    vector[i % vector.length] += query.charCodeAt(i) / 255;
  }

  return vector;
}

async function runHybridDatasetSearch(datasetId, query) {
  const body = {
    vector: buildSearchVector(query),
    alpha: state.spectralWeight,
    k: 10,
  };

  return api(`/api/datasets/${encodeURIComponent(datasetId)}/search/hybrid`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

async function buildDatasetIndex(datasetId) {
  return api(`/api/datasets/${encodeURIComponent(datasetId)}/index`, {
    method: "POST",
    body: JSON.stringify({}),
  });
}

async function runSearch() {
  const query = $("#filter").value.trim().toLowerCase();

  state.searchQuery = query;

  if (!query) {
    state.rankedDatasetIds = null;

    $("#search-mode-label").textContent = "Local filter";
    $("#search-hint").textContent =
      "Type a query to filter datasets. Spectral weighting is ready for ArrowSpace search.";

    renderDatasetList();
    return;
  }

  const scored = state.datasets
    .map((d) => {
      const id = d.id.toLowerCase();
      const dtype = String(d.dtype || "").toLowerCase();
      const kind = String(d.kind || "").toLowerCase();
      const shape = Array.isArray(d.shape)
        ? d.shape.join("x")
        : "";

      let score = 0;

      if (id.includes(query)) score += 5;
      if (kind.includes(query)) score += 2;
      if (dtype.includes(query)) score += 2;
      if (shape.includes(query)) score += 1;

      if (d.shape?.length === 2) {
        score += 3 * state.spectralWeight;
      }

      if (d.shape?.length === 3) {
        score += 0.5;
      }

      return {
        id: d.id,
        score,
      };
    })
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score);

    try {
      const backendScores = [];

      for (const d of state.datasets) {
        if (d.shape?.length !== 2) {
          continue;
        }
        let result;

        try {
          result = await runHybridDatasetSearch(d.id, query);
        } catch (e) {
          if (String(e.message).includes("No index built")) {
            await buildDatasetIndex(d.id);
            result = await runHybridDatasetSearch(d.id, query);
          } else {
            throw e;
          }
       }
        
        console.log(
          "Hybrid search result for",
          d.id,
          result
        );

    backendScores.push({
      id: d.id,
      score:
        result.score ??
        result.best_score ??
        result.results?.[0]?.score ??
        0,
    });
  }

  const validBackendScores = backendScores
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score);

  if (validBackendScores.length > 0) {
    state.rankedDatasetIds = validBackendScores.map((item) => item.id);

    $("#search-mode-label").textContent =
      `Hybrid ${state.spectralWeight.toFixed(2)}`;

    $("#search-hint").textContent =
      `${validBackendScores.length} hybrid 2-D result${
        validBackendScores.length === 1 ? "" : "s"
      } for "${query}". Tensor datasets are shown through local fallback ranking.`;

    renderDatasetList();
    return;
  }
} catch (e) {
  console.warn("Hybrid search unavailable, using local ranking:", e);
}

  state.rankedDatasetIds = scored.map((item) => item.id);

  $("#search-mode-label").textContent =
    `Local ${state.spectralWeight.toFixed(2)}`;

  $("#search-hint").textContent =
    `${scored.length} local ranked result${scored.length === 1 ? "" : "s"} for "${query}". Hybrid search is available only for 2-D datasets.`;

  renderDatasetList();
}

function renderTensorViewer(tensor) {
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

function wireControls() {
  const filter = $("#filter");
  if (filter) {
    filter.addEventListener("input", () => {
      clearTimeout(state.searchTimer);

      state.searchTimer = setTimeout(() => {
        runSearch();
      }, 180);
    });
  }

  const refreshBtn = $("#refresh-btn");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", async () => {
      refreshBtn.textContent = "⟳";

      resetMetrics();

      await refreshHealth();
      await refreshDatasets();

      refreshBtn.textContent = "↻";
    });
  }

  const applySliceBtn = $("#apply-slice");
  if (applySliceBtn) {
    applySliceBtn.addEventListener("click", applySlice);
  }

  const resetSliceBtn = $("#reset-slice");
  if (resetSliceBtn) {
    resetSliceBtn.addEventListener("click", () => {
      $("#slice-input").value = "";
      applySlice();
    });
  }

  const sliceInput = $("#slice-input");
  if (sliceInput) {
    sliceInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") applySlice();
    });
  }

  const copyBtn = $("#copy-id-btn");
  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      if (!state.selected) return;

      await navigator.clipboard.writeText(state.selected.id);

      copyBtn.textContent = "Copied!";

      setTimeout(() => {
        copyBtn.textContent = "Copy Dataset ID";
      }, 1200);
    });
  }

  const spectralSlider = $("#spectral-slider");
  if (spectralSlider) {
    spectralSlider.addEventListener("input", (e) => {
      state.spectralWeight = Number(e.target.value);

      $("#spectral-value").textContent =
        state.spectralWeight.toFixed(2);

      runSearch();
    });
  }
}

(async function main() {
  wireControls();
  attachInfiniteScroll();
  await refreshHealth();
  try {
    await refreshDatasets();
  } catch (e) {
    $("#dataset-list").innerHTML = `<li>error: ${e.message}</li>`;
  }
})();