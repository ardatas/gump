// TerraWatch — Deforestation Viewer
// Client-side GeoJSON visualiser für osapiens Makeathon Submissions.

const YEAR_MIN = 2020;
const YEAR_MAX = 2025;

const YEAR_COLORS = {
  2020: "#fde68a",
  2021: "#fbbf24",
  2022: "#f97316",
  2023: "#ef4444",
  2024: "#b91c1c",
  2025: "#7f1d1d",
  unknown: "#64748b",
};

// ============ MAP ============
const map = L.map("map", {
  zoomControl: true,
  worldCopyJump: true,
  attributionControl: true,
  zoomSnap: 0.1,
  zoomDelta: 0.5,
  wheelPxPerZoomLevel: 80,
  wheelDebounceTime: 20,
}).setView([0, 0], 3);

map.zoomControl.remove();
L.control.zoom({ position: "topright" }).addTo(map);

const basemaps = {
  satellite: L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    {
      attribution: "Imagery © Esri, Maxar, Earthstar Geographics",
      maxZoom: 19,
    }
  ),
  terrain: L.tileLayer(
    "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
    {
      attribution: "© OpenTopoMap (CC-BY-SA)",
      maxZoom: 17,
    }
  ),
  osm: L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "© OpenStreetMap",
    maxZoom: 19,
  }),
};

basemaps.satellite.addTo(map);
let currentBasemap = "satellite";

document.querySelectorAll(".seg-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    basemaps[currentBasemap].remove();
    currentBasemap = btn.dataset.basemap;
    basemaps[currentBasemap].addTo(map);
  });
});

// Legend
const legend = L.control({ position: "bottomright" });
legend.onAdd = function () {
  const div = L.DomUtil.create("div", "legend");
  let html = '<div class="title">Jahr der Deforestation</div>';
  for (let y = YEAR_MIN; y <= YEAR_MAX; y++) {
    html += `<div class="legend-row"><span class="swatch" style="background:${YEAR_COLORS[y]}"></span>${y}</div>`;
  }
  html += `<div class="legend-row"><span class="swatch" style="background:${YEAR_COLORS.unknown}"></span>unbekannt</div>`;
  div.innerHTML = html;
  return div;
};
legend.addTo(map);

// ============ STATE ============
const state = {
  files: [],       // {id, name, layer, features, visible}
  nextId: 1,
  yearFilter: null,       // max year to show
  confFilter: 0,          // min confidence
  areaFilter: 0,          // min area ha
};

// ============ HELPERS ============
function extractYear(feature) {
  const p = feature.properties || {};
  const ts = p.time_step ?? p.year ?? p.alert_year;
  if (ts == null) return null;
  if (typeof ts === "number") {
    if (ts > 1900 && ts < 2100) return Math.floor(ts);
    if (ts < 100) return 2000 + Math.floor(ts);
  }
  if (typeof ts === "string") {
    const m = ts.match(/\b(20\d{2})\b/);
    if (m) return parseInt(m[1], 10);
  }
  return null;
}

function extractConfidence(feature) {
  const p = feature.properties || {};
  const c = p.confidence ?? p.conf ?? p.probability;
  return typeof c === "number" ? c : null;
}

function centroid(feature) {
  try {
    const coords = feature.geometry.coordinates;
    const flat = [];
    const walk = (a) => {
      if (typeof a[0] === "number") flat.push(a);
      else a.forEach(walk);
    };
    walk(coords);
    let sx = 0, sy = 0;
    flat.forEach(([x, y]) => { sx += x; sy += y; });
    return [sx / flat.length, sy / flat.length];
  } catch {
    return [null, null];
  }
}

function areaHa(feature) {
  if (feature.properties && typeof feature.properties.area_ha === "number") {
    return feature.properties.area_ha;
  }
  try {
    const [, lat0] = centroid(feature);
    if (lat0 == null) return null;
    const latRad = (lat0 * Math.PI) / 180;
    const mPerDegLat = 111320;
    const mPerDegLon = 111320 * Math.cos(latRad);
    const geom = feature.geometry;
    const rings = geom.type === "Polygon" ? [geom.coordinates] : geom.coordinates;
    let totalM2 = 0;
    rings.forEach((poly) => {
      poly.forEach((ring, ringIdx) => {
        let a = 0;
        for (let i = 0, n = ring.length; i < n - 1; i++) {
          const [x1, y1] = ring[i];
          const [x2, y2] = ring[i + 1];
          a += (x1 * mPerDegLon) * (y2 * mPerDegLat) - (x2 * mPerDegLon) * (y1 * mPerDegLat);
        }
        totalM2 += Math.abs(a / 2) * (ringIdx === 0 ? 1 : -1);
      });
    });
    return totalM2 / 10000;
  } catch {
    return null;
  }
}

function styleFor(feature) {
  const year = extractYear(feature);
  const color = YEAR_COLORS[year] || YEAR_COLORS.unknown;
  return {
    color: color,
    weight: 1.5,
    fillColor: color,
    fillOpacity: 0.55,
    opacity: 0.9,
  };
}

function styleHover(feature) {
  const year = extractYear(feature);
  const color = YEAR_COLORS[year] || YEAR_COLORS.unknown;
  return { color: "#fff", weight: 2.5, fillColor: color, fillOpacity: 0.8, opacity: 1 };
}

function popupHtml(feature, featureId, fileName) {
  const [lon, lat] = centroid(feature);
  const year = extractYear(feature);
  const ha = areaHa(feature);
  const conf = extractConfidence(feature);
  const color = YEAR_COLORS[year] || YEAR_COLORS.unknown;
  const coordStr = lon != null ? `${lat.toFixed(5)}, ${lon.toFixed(5)}` : "—";

  return `
    <div class="popup">
      <div class="popup-header">
        <span class="popup-badge" style="background:${color}"></span>
        <span class="popup-title">Deforestation Event</span>
      </div>
      <dl class="popup-grid">
        <dt>ID</dt><dd>#${featureId}</dd>
        <dt>Tile</dt><dd>${fileName.replace(/\.(geo)?json$/i, "")}</dd>
        <dt>Lat, Lon</dt><dd>${coordStr}</dd>
        <dt>Fläche</dt><dd>${ha != null ? ha.toFixed(2) + " ha" : "—"}</dd>
        <dt>Jahr</dt><dd>${year ?? "—"}</dd>
        ${conf != null ? `<dt>Confidence</dt><dd>${conf.toFixed(2)}</dd>` : ""}
      </dl>
      <div class="popup-actions">
        <button class="popup-action" onclick="window.copyCoords('${lat?.toFixed(6)}, ${lon?.toFixed(6)}')">📋 Koordinaten</button>
        <button class="popup-action" onclick="window.openGmaps(${lat}, ${lon})">🗺 Google Maps</button>
      </div>
    </div>
  `;
}

window.copyCoords = function (text) {
  navigator.clipboard.writeText(text).then(() => toast("✓ Koordinaten kopiert"));
};
window.openGmaps = function (lat, lon) {
  window.open(`https://www.google.com/maps?q=${lat},${lon}&t=k&z=16`, "_blank");
};

function toast(msg, ms = 2400) {
  const t = document.getElementById("info-toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { t.hidden = true; }, ms);
}

// ============ FILE INGEST ============
let globalFeatureCounter = 0;

function ingestGeoJSON(name, data) {
  if (!data || data.type !== "FeatureCollection" || !Array.isArray(data.features)) {
    toast(`⚠ ${name}: keine gültige FeatureCollection`);
    return;
  }
  if (data.features.length === 0) {
    toast(`⚠ ${name}: keine Polygone`);
    return;
  }

  // Assign stable IDs
  data.features.forEach((f) => {
    f._uid = ++globalFeatureCounter;
  });

  const id = state.nextId++;
  const layer = L.geoJSON(data, {
    style: styleFor,
    onEachFeature: (f, lyr) => {
      lyr.bindPopup(popupHtml(f, f._uid, name), { maxWidth: 320 });
      lyr.on("mouseover", () => lyr.setStyle(styleHover(f)));
      lyr.on("mouseout", () => lyr.setStyle(styleFor(f)));
    },
  }).addTo(map);

  state.files.push({
    id, name,
    layer,
    features: data.features,
    visible: true,
  });

  refreshAll();
  fitToAll();
  toast(`✓ ${name} geladen · ${data.features.length} Polygone`);
}

function fitToAll() {
  const group = L.featureGroup(
    state.files.filter((f) => f.visible).map((f) => f.layer)
  );
  if (group.getLayers().length > 0) {
    map.fitBounds(group.getBounds(), { padding: [40, 40], maxZoom: 13 });
  }
}

async function handleFiles(fileList) {
  const files = Array.from(fileList).filter((f) =>
    /\.(geojson|json)$/i.test(f.name)
  );
  if (files.length === 0) {
    toast("⚠ Keine GeoJSON-Dateien dabei");
    return;
  }
  for (const file of files) {
    try {
      const text = await file.text();
      const json = JSON.parse(text);
      ingestGeoJSON(file.name, json);
    } catch (err) {
      toast(`⚠ ${file.name}: ${err.message}`);
    }
  }
}

// ============ FILTER ENGINE ============
function featurePassesFilters(feature) {
  const year = extractYear(feature);
  const conf = extractConfidence(feature);
  const ha = areaHa(feature);
  if (state.yearFilter != null && year != null && year > state.yearFilter) return false;
  if (state.confFilter > 0 && conf != null && conf < state.confFilter) return false;
  if (state.areaFilter > 0 && ha != null && ha < state.areaFilter) return false;
  return true;
}

function applyFilters() {
  state.files.forEach((f) => {
    f.layer.eachLayer((lyr) => {
      const pass = featurePassesFilters(lyr.feature);
      const el = lyr.getElement && lyr.getElement();
      if (el) {
        el.style.display = pass ? "" : "none";
        el.style.pointerEvents = pass ? "" : "none";
      }
    });
  });
  refreshHeadlineStats();
  refreshYearChart();
}

// ============ UI REFRESH ============
function refreshAll() {
  refreshHeadlineStats();
  refreshFileList();
  refreshYearChart();
  refreshSearchPanel();
  refreshPanelsVisibility();
}

function visibleFeatures() {
  const out = [];
  state.files.forEach((f) => {
    if (!f.visible) return;
    f.features.forEach((ft) => {
      if (featurePassesFilters(ft)) out.push({ file: f, feature: ft });
    });
  });
  return out;
}

function refreshHeadlineStats() {
  const vis = visibleFeatures();
  const totalArea = vis.reduce((s, { feature }) => s + (areaHa(feature) || 0), 0);
  const years = vis.map(({ feature }) => extractYear(feature)).filter((y) => y != null);
  const yrMin = years.length ? Math.min(...years) : null;
  const yrMax = years.length ? Math.max(...years) : null;

  document.getElementById("hstat-polys").textContent = vis.length.toLocaleString("de-DE");
  document.getElementById("hstat-area").textContent = totalArea >= 1000
    ? (totalArea / 1000).toFixed(1) + "k"
    : totalArea.toFixed(1);
  document.getElementById("hstat-tiles").textContent = state.files.filter((f) => f.visible).length;
  document.getElementById("hstat-years").textContent = yrMin
    ? (yrMin === yrMax ? `${yrMin}` : `${yrMin}–${yrMax}`)
    : "—";
}

function refreshFileList() {
  const ul = document.getElementById("file-list");
  ul.innerHTML = "";
  state.files.forEach((f) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <input type="checkbox" ${f.visible ? "checked" : ""} data-id="${f.id}" />
      <span class="fname" title="${f.name}">${f.name}</span>
      <span class="fcount">${f.features.length}</span>
      <button class="ficon" data-action="zoom" data-id="${f.id}" title="Zu Layer zoomen">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
      </button>
      <button class="ficon" data-action="remove" data-id="${f.id}" title="Entfernen">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    `;
    ul.appendChild(li);
  });

  ul.querySelectorAll("input[type='checkbox']").forEach((cb) => {
    cb.addEventListener("change", (e) => {
      const id = parseInt(e.target.dataset.id, 10);
      const f = state.files.find((x) => x.id === id);
      f.visible = e.target.checked;
      if (f.visible) f.layer.addTo(map);
      else f.layer.remove();
      refreshAll();
    });
  });

  ul.querySelectorAll(".ficon").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const id = parseInt(btn.dataset.id, 10);
      const action = btn.dataset.action;
      const f = state.files.find((x) => x.id === id);
      if (!f) return;
      if (action === "zoom" && f.visible) {
        map.fitBounds(f.layer.getBounds(), { padding: [30, 30] });
      } else if (action === "remove") {
        f.layer.remove();
        state.files = state.files.filter((x) => x.id !== id);
        refreshAll();
        toast(`${f.name} entfernt`);
      }
    });
  });

  document.getElementById("export-csv").disabled = state.files.length === 0;
  document.getElementById("export-geojson").disabled = state.files.length === 0;
}

function refreshYearChart() {
  const chart = document.getElementById("year-chart");
  chart.innerHTML = "";
  const counts = {};
  for (let y = YEAR_MIN; y <= YEAR_MAX; y++) counts[y] = 0;
  visibleFeatures().forEach(({ feature }) => {
    const y = extractYear(feature);
    if (y != null && counts[y] != null) counts[y]++;
  });
  const max = Math.max(1, ...Object.values(counts));
  for (let y = YEAR_MIN; y <= YEAR_MAX; y++) {
    const h = (counts[y] / max) * 100;
    const bar = document.createElement("div");
    bar.className = "bar";
    bar.style.height = `${h}%`;
    bar.style.setProperty("--bar-color", YEAR_COLORS[y]);
    bar.innerHTML = `
      <span class="bar-count">${counts[y] || ""}</span>
      <span class="bar-label">'${String(y).slice(2)}</span>
    `;
    bar.addEventListener("click", () => {
      document.getElementById("year-slider").value = y;
      state.yearFilter = y;
      document.getElementById("year-display").textContent = `≤ ${y}`;
      applyFilters();
    });
    chart.appendChild(bar);
  }
}

function refreshPanelsVisibility() {
  const hasData = state.files.length > 0;
  document.getElementById("filter-panel").hidden = !hasData;
  document.getElementById("chart-panel").hidden = !hasData;
  document.getElementById("files-panel").hidden = !hasData;
  document.getElementById("search-panel").hidden = !hasData;
}

// ============ SEARCH ============
function refreshSearchPanel() {
  const query = document.getElementById("search-input").value.trim().toLowerCase();
  const ul = document.getElementById("search-results");
  ul.innerHTML = "";

  const all = [];
  state.files.forEach((f) => {
    if (!f.visible) return;
    f.features.forEach((ft) => {
      if (!featurePassesFilters(ft)) return;
      const [lon, lat] = centroid(ft);
      all.push({ file: f, feature: ft, lon, lat });
    });
  });

  let filtered = all;
  if (query.length > 0) {
    filtered = all.filter(({ feature, lon, lat }) => {
      const id = `#${feature._uid}`;
      const coord = `${lat?.toFixed(3)} ${lon?.toFixed(3)}`;
      return id.toLowerCase().includes(query) || coord.includes(query);
    });
  }

  filtered.slice(0, 40).forEach(({ file, feature, lon, lat }) => {
    const year = extractYear(feature);
    const ha = areaHa(feature);
    const li = document.createElement("li");
    li.innerHTML = `
      <div>
        <div class="sr-id">#${feature._uid}</div>
        <div class="sr-coords">${lat?.toFixed(4)}, ${lon?.toFixed(4)}</div>
      </div>
      <div class="sr-meta">${year ?? "?"} · ${ha?.toFixed(1) ?? "?"} ha</div>
    `;
    li.addEventListener("click", () => {
      map.flyTo([lat, lon], 15, { duration: 0.6 });
      file.layer.eachLayer((lyr) => {
        if (lyr.feature === feature) {
          setTimeout(() => lyr.openPopup(), 600);
        }
      });
    });
    ul.appendChild(li);
  });

  if (filtered.length === 0 && query.length > 0) {
    ul.innerHTML = `<li style="cursor:default;border-color:transparent;color:var(--text-dim)">Keine Treffer</li>`;
  } else if (all.length > 40 && query.length === 0) {
    const info = document.createElement("li");
    info.style.cursor = "default";
    info.style.borderColor = "transparent";
    info.style.color = "var(--text-dim)";
    info.textContent = `Zeige 40 von ${all.length}. Mit Suche einschränken.`;
    ul.appendChild(info);
  }
}

document.getElementById("search-input").addEventListener("input", refreshSearchPanel);

// ============ FILTER CONTROLS ============
const yearSlider = document.getElementById("year-slider");
const yearDisplay = document.getElementById("year-display");
const confSlider = document.getElementById("conf-slider");
const confDisplay = document.getElementById("conf-display");
const areaSlider = document.getElementById("area-slider");
const areaDisplay = document.getElementById("area-display");

yearSlider.addEventListener("input", (e) => {
  const v = parseInt(e.target.value, 10);
  state.yearFilter = v;
  yearDisplay.textContent = `≤ ${v}`;
  applyFilters();
});

confSlider.addEventListener("input", (e) => {
  const v = parseFloat(e.target.value);
  state.confFilter = v;
  confDisplay.textContent = v.toFixed(2);
  applyFilters();
});

areaSlider.addEventListener("input", (e) => {
  const v = parseFloat(e.target.value);
  state.areaFilter = v;
  areaDisplay.textContent = `${v.toFixed(1)} ha`;
  applyFilters();
});

document.getElementById("filter-reset").addEventListener("click", () => {
  state.yearFilter = null; state.confFilter = 0; state.areaFilter = 0;
  yearSlider.value = YEAR_MAX; yearDisplay.textContent = "alle";
  confSlider.value = 0; confDisplay.textContent = "0.00";
  areaSlider.value = 0; areaDisplay.textContent = "0 ha";
  applyFilters();
});

// ============ EXPORT ============
function exportCSV() {
  const rows = [["file", "polygon_id", "centroid_lon", "centroid_lat", "area_ha", "year", "confidence"]];
  visibleFeatures().forEach(({ file, feature }) => {
    const [lon, lat] = centroid(feature);
    rows.push([
      file.name,
      feature._uid,
      lon != null ? lon.toFixed(6) : "",
      lat != null ? lat.toFixed(6) : "",
      (areaHa(feature) ?? "").toString(),
      extractYear(feature) ?? "",
      extractConfidence(feature) ?? "",
    ]);
  });
  const csv = rows.map((r) => r.map((x) => `"${String(x).replace(/"/g, '""')}"`).join(",")).join("\n");
  download(csv, "deforestation_coordinates.csv", "text/csv");
  toast(`✓ ${rows.length - 1} Polygone als CSV exportiert`);
}

function exportMergedGeoJSON() {
  const features = [];
  visibleFeatures().forEach(({ file, feature }) => {
    const copy = JSON.parse(JSON.stringify(feature));
    copy.properties = copy.properties || {};
    copy.properties._source_file = file.name;
    delete copy._uid;
    features.push(copy);
  });
  const fc = { type: "FeatureCollection", features };
  download(JSON.stringify(fc, null, 2), "deforestation_merged.geojson", "application/geo+json");
  toast(`✓ ${features.length} Polygone gemerged`);
}

function download(content, name, mime) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

document.getElementById("export-csv").addEventListener("click", exportCSV);
document.getElementById("export-geojson").addEventListener("click", exportMergedGeoJSON);

// ============ DRAG & DROP ============
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");

["dragenter", "dragover"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault(); e.stopPropagation();
    dropZone.classList.add("drag-active");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault(); e.stopPropagation();
    dropZone.classList.remove("drag-active");
  })
);
dropZone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files?.length) handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener("change", (e) => {
  if (e.target.files?.length) handleFiles(e.target.files);
  e.target.value = "";
});

// Also allow drop on map
const mapEl = document.getElementById("map");
mapEl.addEventListener("dragover", (e) => e.preventDefault());
mapEl.addEventListener("drop", (e) => {
  e.preventDefault();
  if (e.dataTransfer.files?.length) handleFiles(e.dataTransfer.files);
});

// ============ MODAL ============
const modal = document.getElementById("modal-about");
document.getElementById("btn-about").addEventListener("click", () => { modal.classList.add("is-open"); });
document.querySelector(".modal-content").addEventListener("click", (e) => e.stopPropagation());
modal.addEventListener("click", () => { modal.classList.remove("is-open"); });

// ============ KEYBOARD SHORTCUTS ============
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") {
    if (e.key === "Escape") e.target.blur();
    return;
  }
  if (e.key === "/") {
    e.preventDefault();
    document.getElementById("search-input").focus();
  } else if (e.key.toLowerCase() === "r") {
    document.getElementById("filter-reset").click();
  } else if (e.key.toLowerCase() === "f") {
    fitToAll();
  } else if (e.key === "Escape") {
    modal.classList.remove("is-open");
  } else if (e.key.toLowerCase() === "i") {
    modal.classList.toggle("is-open");
  }
});

// ============ AUTO-LOAD from ./data/ ============
(async function autoLoad() {
  try {
    const idx = await fetch("./data/index.json");
    if (!idx.ok) return;
    const files = await idx.json();
    for (const fname of files) {
      const r = await fetch(`./data/${fname}`);
      if (r.ok) ingestGeoJSON(fname, await r.json());
    }
  } catch {
    // Silent — kein Manifest → kein Auto-Load
  }
})();

refreshAll();
