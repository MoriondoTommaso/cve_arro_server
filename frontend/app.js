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
  recentSearches: JSON.parse(
    localStorage.getItem("leafRecentSearches") || "[]"
  ),  
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

async function runPromptNLSearch(query) {
  return api("/api/prompts/nl_search", {
    method: "POST",
    body: JSON.stringify({
      query,
      k: 10,
      tau: 0.8,
      lam: 0.7,
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


async function runSearch() {
  const query = $("#filter").value.trim();

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
    $("#health").textContent = "Searching...";

    const tau = Number($("#spectral-slider").value);
    const alpha = Number($("#alpha-slider").value);
    const lam = Number($("#lam-slider").value);
    
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
        k: Number($("#topk-select")?.value || 10),
        tau,
        alpha,
        lam,
      }),
    });

    const latencyMs = Math.round(performance.now() - startedAt);

    renderPromptResults(result.results || [], {
      latencyMs,
      resultCount: result.result_count || 0,
      tau,
      alpha,
      lam,
    });

    await renderSearchVisualizations(result.results || []);

    $("#health").textContent = "LEAF Ready";
    $("#health").className = "health ok";

    $("#search-mode-label").textContent =
      `τ ${tau.toFixed(2)} · α ${alpha.toFixed(2)} · λ ${lam.toFixed(2)}`;

    $("#search-hint").textContent =
      `${result.result_count || 0} semantic results`;

  } catch (e) {
    console.error(e);

    $("#health").textContent = "Search Error";
    $("#health").classList.add("err");

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
          <span>Tau</span>
          <strong>${analytics.tau?.toFixed?.(2) ?? "—"}</strong>
        </div>

        <div>
          <span>Alpha</span>
          <strong>${analytics.alpha?.toFixed?.(2) ?? "—"}</strong>
        </div>

        <div>
          <span>Lambda</span>
          <strong>${analytics.lam?.toFixed?.(2) ?? "—"}</strong>
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
          <span class="prompt-score-value">
            ${(item.score ?? 0).toFixed(4)}
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
    renderAuditLambdas(lambdas);
    renderAuditStats(audit)
    renderAuditPCA(audit);

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

function renderAuditLambdas(data) {
  const container = $("#audit-lambdas");

  if (!container) return;

  const lambdas = data.lambdas || [];

  if (!lambdas.length) {
    container.innerHTML = `
      <div class="signal-empty">
        No eigenvalues available.
      </div>
    `;
    return;
  }

  const maxVal = Math.max(...lambdas, 1);

  const bars = lambdas
    .slice(0, 32)
    .map((v) => {
      const h = Math.max(4, (v / maxVal) * 100);

      return `
        <div class="lambda-bar-wrap">
          <div
            class="lambda-bar"
            style="height:${h}%"
            title="${v.toFixed(5)}"
          ></div>
        </div>
      `;
    })
    .join("");

  container.innerHTML = `
    <div class="signal-card">
      <span class="signal-label">Fiedler Value</span>
      <strong>
        ${(lambdas[1] ?? 0).toFixed(6)}
      </strong>
    </div>

    <div class="signal-card">
      <span class="signal-label">Eigenvalue Count</span>
      <strong>${data.n ?? lambdas.length}</strong>
    </div>

    <div class="lambda-bars">
      ${bars}
    </div>
  `;
}

function renderAuditStats(audit) {
  const container = $("#audit-stats");

  if (!container) return;

  const stats = audit.degree_stats || {};

  const fiedler = audit.fiedler_value ?? 0;

  let fiedlerColor = "#f87171";

  if (fiedler > 0.01) {
    fiedlerColor = "#34d399";
  } else if (fiedler > 0.001) {
    fiedlerColor = "#facc15";
  }

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
      <span class="signal-label">Fiedler Value</span>
      <strong style="color:${fiedlerColor}">
        ${fiedler.toFixed(6)}
      </strong>
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
    filter.value = clicked.id;

    $("#search-hint").textContent =
    `Searching selected PCA prompt ${clicked.id}...`;

    runSearch();
  };
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

function switchView(viewName) {
  const searchView = $("#search-view");
  const auditView = $("#audit-view");
  const searchTab = $("#tab-search");
  const auditTab = $("#tab-audit");

  if (viewName === "audit") {
    searchView.classList.add("hidden");
    auditView.classList.remove("hidden");

    searchTab.classList.remove("active");
    auditTab.classList.add("active");

    loadAuditPanel();
    return;
  }

  auditView.classList.add("hidden");
  searchView.classList.remove("hidden");

  auditTab.classList.remove("active");
  searchTab.classList.add("active");
}


function wireControls() {
  const alphaSlider = $("#alpha-slider");
  if (alphaSlider) {
    alphaSlider.addEventListener("input", (e) => {
      $("#alpha-value").textContent = Number(e.target.value).toFixed(2);

      clearTimeout(state.searchTimer);
      state.searchTimer = setTimeout(runSearch, 300);
    });
  }

  const lamSlider = $("#lam-slider");
  if (lamSlider) {
    lamSlider.addEventListener("input", (e) => {
      $("#lam-value").textContent = Number(e.target.value).toFixed(2);

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

  const spectralSlider = $("#spectral-slider");
  if (spectralSlider) {
    spectralSlider.addEventListener("input", (e) => {
      state.spectralWeight = Number(e.target.value);

      $("#spectral-value").textContent =
        state.spectralWeight.toFixed(2);

      clearTimeout(state.searchTimer);
      state.searchTimer = setTimeout(runSearch, 300);
    });
  }

  const topkSelect = $("#topk-select");
  if (topkSelect) {
    topkSelect.addEventListener("change", (e) => {
      $("#topk-value").textContent = e.target.value;

      clearTimeout(state.searchTimer);
      state.searchTimer = setTimeout(runSearch, 300);
    });
  }

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


(async function main() {
  wireControls();

  try {
    const health = await api("/api/prompts/health");

    $("#health").textContent =
      health.status === "ready" ? "LEAF Ready" : "LEAF Warming...";

    $("#health").className =
      health.status === "ready" ? "health ok" : "health";
  } catch (e) {
    console.error("Health check failed:", e);

    $("#health").textContent = "Backend offline";
    $("#health").className = "health err";
  }
})();

function renderRecentSearches() {
  const el = $("#recent-searches");

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
          data-query="${escapeHtml(query)}"
        >
          ${escapeHtml(query)}
        </button>
      `
    )
    .join("");

  el.querySelectorAll(".recent-search-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      $("#filter").value = btn.dataset.query;
      runSearch();
    });
  });
}

async function renderSearchVisualizations(results) {
  const resultIds = new Set(
    results.map((item) => item.id)
  );

  try {
    const [audit, lambdas] = await Promise.all([
      api("/api/prompts/audit"),
      api("/api/prompts/lambdas"),
    ]);

    renderQueryManifold(audit, resultIds);
    renderQueryLambdaChart(lambdas);

  } catch (e) {
    console.warn("Search visualizations unavailable:", e);
  }
}

function renderQueryManifold(audit, resultIds) {
  const el = $("#query-manifold");
  if (!el || !audit?.pca_2d || !window.Plotly) return;

  const points = audit.pca_2d;

  const highlightedIndices = new Set(
    [...resultIds]
      .map((id) => Number(String(id).replace(/\D/g, "")))
      .filter(Number.isFinite)
  );

  const xs = points.map((p) => Number(p[0]));
  const ys = points.map((p) => Number(p[1]));

  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const gridSize = 45;
  const z = [];

  for (let gy = 0; gy < gridSize; gy++) {
    const row = [];

    for (let gx = 0; gx < gridSize; gx++) {
      const x = minX + ((maxX - minX) * gx) / (gridSize - 1);
      const y = minY + ((maxY - minY) * gy) / (gridSize - 1);

      let density = 0;

      for (let i = 0; i < points.length; i += 20) {
        const dx = x - points[i][0];
        const dy = y - points[i][1];

        density += Math.exp(-(dx * dx + dy * dy) / 3.5);
      }

      row.push(density);
    }

    z.push(row);
  }

  const surface = {
    z,
    type: "surface",
    colorscale: "Blues",
    opacity: 0.88,
    showscale: false,
    name: "Laplacian Surface",
  };

  const highlighted = points
    .map((p, i) => ({ p, i }))
    .filter(({ i }) => highlightedIndices.has(i));

  const scatter = {
    x: highlighted.map(({ p }) => p[0]),
    y: highlighted.map(({ p }) => p[1]),
    z: highlighted.map(() => Math.max(...z.flat()) * 1.15),
    type: "scatter3d",
    mode: "markers",
    name: "Matched Prompts",
    marker: {
      size: 5,
      color: "#ff4d4d",
      line: {
        color: "#ffffff",
        width: 1,
      },
    },
  };

  Plotly.newPlot(
    el,
    [surface, scatter],
    {
      title: {
        text: "Graph Laplacian Manifold Interpretation",
        font: { color: "#e2e8f0", size: 14 },
      },
      paper_bgcolor: "rgba(0,0,0,0)",
      scene: {
        xaxis: {
          title: "PCA 1",
          gridcolor: "rgba(255,255,255,0.12)",
          color: "#cbd5e1",
        },
        yaxis: {
          title: "PCA 2",
          gridcolor: "rgba(255,255,255,0.12)",
          color: "#cbd5e1",
        },
        zaxis: {
          title: "Node Degree / Density",
          gridcolor: "rgba(255,255,255,0.12)",
          color: "#cbd5e1",
        },
        bgcolor: "rgba(15,23,42,0.65)",
        camera: {
          eye: { x: 1.5, y: 1.6, z: 0.75 },
        },
      },
      margin: { l: 0, r: 0, t: 45, b: 0 },
      legend: {
        font: { color: "#cbd5e1" },
      },
    },
    {
      responsive: true,
      displayModeBar: false,
    }
  );
}

function renderQueryLambdaChart(data) {
  const el = $("#query-lambda-chart");
  if (!el || !data?.lambdas || !window.Plotly) return;

  const lambdas = data.lambdas
    .map(Number)
    .filter(Number.isFinite);

  const sorted = [...lambdas].sort((a, b) => a - b);
  const ecdfY = sorted.map((_, i) => (i + 1) / sorted.length);

  const histTrace = {
    x: lambdas,
    type: "histogram",
    nbinsx: 60,
    name: "Lambda Distribution",
    marker: {
      color: "rgba(124,92,255,0.75)",
    },
    opacity: 0.75,
  };

  const ecdfTrace = {
    x: sorted,
    y: ecdfY,
    type: "scatter",
    mode: "lines",
    name: "Cumulative",
    yaxis: "y2",
    line: {
      width: 3,
      color: "#5ea2ff",
    },
  };

  Plotly.newPlot(
    el,
    [histTrace, ecdfTrace],
    {
      title: {
        text: `Spectral Fingerprint: ${lambdas.length} Samples`,
        font: { color: "#e2e8f0", size: 14 },
      },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(15,23,42,0.65)",
      font: { color: "#cbd5e1" },
      margin: { l: 45, r: 45, t: 45, b: 45 },
      xaxis: {
        title: "λ eigenvalue",
        gridcolor: "rgba(255,255,255,0.08)",
      },
      yaxis: {
        title: "Frequency",
        gridcolor: "rgba(255,255,255,0.08)",
      },
      yaxis2: {
        title: "ECDF",
        overlaying: "y",
        side: "right",
        range: [0, 1],
      },
      legend: {
        orientation: "h",
        x: 0.02,
        y: 1.12,
      },
    },
    {
      responsive: true,
      displayModeBar: false,
    }
  );
}