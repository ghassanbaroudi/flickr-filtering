/**
 * Embedded in context_comparison_report.html after const clustersData / overlayData / csvColumns.
 * Depends on: Leaflet, clustersData, overlayData, csvColumns (array of column names).
 */

function rowOf(p) {
  return p.row || p;
}

function navKeyForPoint(p, idx) {
  const r = rowOf(p);
  const id = r.id != null && String(r.id).trim() !== "" ? String(r.id) : "idx_" + idx;
  return id;
}

function latLonFromRow(p) {
  const r = rowOf(p);
  return [parseFloat(r.latitude), parseFloat(r.longitude)];
}

function makeClusterPopupHtml(p, clusterId, imageId) {
  const r = rowOf(p);
  const safeId = String(imageId).replace(/"/g, "&quot;");
  let html = "";
  if (r.title) html += "<strong>" + String(r.title).replace(/</g, "&lt;") + "</strong><br/>";
  if (r.id) html += "ID: " + String(r.id).replace(/</g, "&lt;") + "<br/>";
  if (r.source_dataset)
    html += '<span style="color:#666;">Source: ' + String(r.source_dataset) + "</span><br/>";
  if (r.context)
    html += '<span style="color:#666;">Context: ' + String(r.context) + " (clustered)</span><br/>";
  if (r.vision_reason) {
    const vr = String(r.vision_reason).replace(/</g, "&lt;").replace(/"/g, "&quot;");
    html += '<span style="color:#555;font-size:0.85rem;">Vision: ' + vr + "</span><br/>";
  }
  if (r.image_url) {
    const safe = String(r.image_url).replace(/"/g, "&quot;");
    html +=
      '<div style="margin-top:4px;"><img src="' +
      safe +
      '" alt="" style="max-width:260px;max-height:180px;object-fit:contain;"/></div>';
  }
  html +=
    '<div style="margin-top:6px;display:flex;gap:6px;">' +
    '<button type="button" class="nav-btn" data-nav="prev" data-cluster-id="' +
    clusterId +
    '" data-image-id="' +
    safeId +
    '">Prev</button>' +
    '<button type="button" class="nav-btn" data-nav="next" data-cluster-id="' +
    clusterId +
    '" data-image-id="' +
    safeId +
    '">Next</button>' +
    '<span style="color:#666;font-size:0.8rem;">Arrow keys</span></div>' +
    '<div style="margin-top:6px;">' +
    '<button type="button" class="err-btn" data-cluster-id="' +
    clusterId +
    '" data-image-id="' +
    safeId +
    '" data-mode="error">Mark as erroneous</button></div>';
  return html;
}

function makeOverlayPopupHtml(p) {
  const r = rowOf(p);
  let html = "";
  if (r.title) html += "<strong>" + String(r.title).replace(/</g, "&lt;") + "</strong><br/>";
  if (r.id) html += "ID: " + String(r.id) + "<br/>";
  if (r.source_dataset)
    html += '<span style="color:#666;">Source: ' + String(r.source_dataset) + "</span><br/>";
  if (r.context)
    html += '<span style="color:#666;">Context: ' + String(r.context) + " (overlay)</span><br/>";
  if (r.image_url) {
    const safe = String(r.image_url).replace(/"/g, "&quot;");
    html +=
      '<div style="margin-top:4px;"><img src="' +
      safe +
      '" alt="" style="max-width:240px;max-height:160px;object-fit:contain;"/></div>';
  }
  html +=
    '<div style="margin-top:8px;display:flex;gap:6px;align-items:center;">' +
    '<button type="button" class="ov-nav-btn" data-nav="prev">Prev</button>' +
    '<button type="button" class="ov-nav-btn" data-nav="next">Next</button>' +
    '<span style="color:#666;font-size:0.8rem;">Arrow keys</span></div>';
  return html;
}

function createImageMarker(p, centerLat, centerLon, markerLat, markerLon, clusterId, imageId) {
  const pinIcon = L.divIcon({
    className: "pin-marker",
    html:
      '<div style="background: #1f77ff; width: 10px; height: 10px; border-radius: 50% 50% 50% 50% / 60% 60% 40% 40%; border: 1.5px solid #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.25);"></div>',
    iconSize: [14, 14],
    iconAnchor: [7, 14],
  });

  const marker = L.marker([markerLat, markerLon], { icon: pinIcon });
  const html = makeClusterPopupHtml(p, clusterId, imageId);
  marker.bindPopup(html);

  let stem = null;
  if (centerLat !== markerLat || centerLon !== markerLon) {
    stem = L.polyline(
      [
        [centerLat, centerLon],
        [markerLat, markerLon],
      ],
      { color: "#555555", weight: 1, opacity: 0.7 }
    );
  }

  return { point: p, marker, stem };
}

function updateErroneousCount(entry) {
  const errCount = Object.values(entry.pointsById).filter(function (r) {
    return entry.errGroup.hasLayer(r.marker);
  }).length;
  entry.errItem.querySelector(".cluster-meta").textContent =
    errCount === 1 ? "1 image" : errCount + " images";
}

function moveToErroneous(clusterId, imageId) {
  const entry = markersByCluster[clusterId];
  if (!entry) return;
  const rec = entry.pointsById[imageId];
  if (!rec || !rec.marker) return;
  if (entry.errGroup.hasLayer(rec.marker)) return;

  entry.mainGroup.removeLayer(rec.marker);
  if (rec.stem) entry.mainGroup.removeLayer(rec.stem);
  entry.errGroup.addLayer(rec.marker);
  if (rec.stem) entry.errGroup.addLayer(rec.stem);

  updateErroneousCount(entry);
  // Do not call setActiveCluster here — it re-runs fitBounds and jumps the map zoom.
}

function moveToMain(clusterId, imageId) {
  const entry = markersByCluster[clusterId];
  if (!entry) return;
  const rec = entry.pointsById[imageId];
  if (!rec || !rec.marker) return;
  if (!entry.errGroup.hasLayer(rec.marker)) return;

  entry.errGroup.removeLayer(rec.marker);
  if (rec.stem) entry.errGroup.removeLayer(rec.stem);
  entry.mainGroup.addLayer(rec.marker);
  if (rec.stem) entry.mainGroup.addLayer(rec.stem);

  updateErroneousCount(entry);
}

function getVisibleImageIds(clusterId) {
  const entry = markersByCluster[clusterId];
  if (!entry) return [];
  const layer = activeSubset === "err" ? entry.errGroup : entry.mainGroup;
  return Object.keys(entry.pointsById).filter(function (id) {
    const rec = entry.pointsById[id];
    return rec && rec.marker && layer.hasLayer(rec.marker);
  });
}

function navigateImage(clusterId, imageId, direction) {
  const entry = markersByCluster[clusterId];
  if (!entry) return;

  const ids = getVisibleImageIds(clusterId);
  if (!ids.length) return;

  const idx = ids.indexOf(String(imageId));
  if (idx === -1) return;

  const nextIdx = (idx + direction + ids.length) % ids.length;
  const targetId = ids[nextIdx];
  const targetRec = entry.pointsById[targetId];
  if (!targetRec || !targetRec.marker) return;

  targetRec.marker.openPopup();
  const latLng = targetRec.marker.getLatLng();
  if (latLng) map.panTo(latLng, { animate: true });
}

function navigateOverlay(direction) {
  if (!overlayData.length) return;
  const cur = currentPopupContext && currentPopupContext.imageId;
  if (!cur) return;
  let j = overlayData.findIndex(function (p, i) {
    return navKeyForPoint(p, i) === cur;
  });
  if (j === -1)
    j = overlayData.findIndex(function (p) {
      return String(rowOf(p).id || "") === String(cur);
    });
  if (j === -1) return;
  const nextIdx = (j + direction + overlayData.length) % overlayData.length;
  const next = overlayData[nextIdx];
  const nextKey = navKeyForPoint(next, nextIdx);
  const marker = overlayMarkerById[nextKey];
  if (marker) {
    marker.openPopup();
    const ll = marker.getLatLng();
    if (ll) map.panTo(ll, { animate: true });
  }
}

function csvEscapeCell(s) {
  const str = String(s ?? "");
  if (str.indexOf(",") !== -1 || str.indexOf('"') !== -1 || str.indexOf("\n") !== -1) {
    return '"' + str.replace(/"/g, '""') + '"';
  }
  return str;
}

function exportErroneousCsv() {
  const rows = [];
  rows.push(["cluster_id", "image_id"].concat(csvColumns));

  Object.keys(markersByCluster).forEach(function (cidStr) {
    const cid = Number(cidStr);
    const entry = markersByCluster[cidStr];
    if (!entry) return;
    Object.keys(entry.pointsById).forEach(function (imageId) {
      const rec = entry.pointsById[imageId];
      if (!rec || !entry.errGroup.hasLayer(rec.marker)) return;
      const row = rowOf(rec.point);
      const line = [cid, imageId].concat(csvColumns.map(function (c) {
        return row[c] != null ? row[c] : "";
      }));
      rows.push(line);
    });
  });

  if (rows.length === 1) {
    alert("No erroneous images selected yet.");
    return;
  }

  const csvText = rows
    .map(function (cols) {
      return cols.map(csvEscapeCell).join(",");
    })
    .join("\n");

  downloadBlob(csvText, "context_comparison_erroneous_" + timestampSuffix() + ".csv");
}

function exportSanitizedCsv() {
  const rows = [];
  rows.push(csvColumns);

  Object.keys(markersByCluster).forEach(function (cidStr) {
    const entry = markersByCluster[cidStr];
    if (!entry) return;
    Object.keys(entry.pointsById).forEach(function (imageId) {
      const rec = entry.pointsById[imageId];
      if (!rec || entry.errGroup.hasLayer(rec.marker)) return;
      const row = rowOf(rec.point);
      rows.push(csvColumns.map(function (c) {
        return row[c] != null ? row[c] : "";
      }));
    });
  });

  if (rows.length === 1) {
    alert("No remaining (non-erroneous) images.");
    return;
  }

  const csvText = rows
    .map(function (cols) {
      return cols.map(csvEscapeCell).join(",");
    })
    .join("\n");

  downloadBlob(csvText, "context_comparison_sanitized_" + timestampSuffix() + ".csv");
}

function timestampSuffix() {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

function downloadBlob(text, filename) {
  const blob = new Blob([text], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function parseCsvLine(line) {
  const out = [];
  let cur = "";
  let inQ = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inQ) {
      if (c === '"') {
        if (line[i + 1] === '"') {
          cur += '"';
          i++;
        } else inQ = false;
      } else cur += c;
    } else {
      if (c === '"') inQ = true;
      else if (c === ",") {
        out.push(cur);
        cur = "";
      } else cur += c;
    }
  }
  out.push(cur);
  return out;
}

function importErroneousFromText(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) {
    alert("CSV is empty.");
    return;
  }
  const header = parseCsvLine(lines[0]).map(function (h) {
    return h.trim();
  });
  const ic = header.indexOf("image_id");
  const cc = header.indexOf("cluster_id");
  if (ic === -1) {
    alert("Missing image_id column.");
    return;
  }

  let n = 0;
  for (let li = 1; li < lines.length; li++) {
    if (!lines[li].trim()) continue;
    const cols = parseCsvLine(lines[li]);
    const imageId = cols[ic] != null ? String(cols[ic]).trim() : "";
    if (!imageId) continue;
    let clusterId =
      cc >= 0 && cols[cc] != null && String(cols[cc]).trim() !== ""
        ? Number(cols[cc])
        : NaN;
    if (isNaN(clusterId)) {
      let found = false;
      Object.keys(markersByCluster).forEach(function (cidStr) {
        if (found) return;
        if (markersByCluster[cidStr].pointsById[imageId]) {
          moveToErroneous(Number(cidStr), imageId);
          found = true;
        }
      });
      if (found) n++;
      continue;
    }
    moveToErroneous(clusterId, imageId);
    n++;
  }
  alert("Imported " + n + " erroneous flag(s).");
}

const markersByCluster = {};
let activeClusterId = null;
let activeSubset = "main";
let currentPopupContext = null;

const overlayLayer = L.featureGroup();
const overlayMarkerById = {};

const map = L.map("map", { maxZoom: 22 });
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 22,
  maxNativeZoom: 19,
  attribution:
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
}).addTo(map);

function initOverlay() {
  if (!overlayData.length) return;
  overlayData.forEach(function (p, i) {
    const key = navKeyForPoint(p, i);
    const ll = latLonFromRow(p);
    const m = L.circleMarker(ll, {
      radius: 2.5,
      color: "#1f77ff",
      weight: 1,
      fillColor: "#66a3ff",
      fillOpacity: 0.8,
    });
    m._cmpKind = "overlay";
    m._cmpImageId = key;
    m.bindPopup(makeOverlayPopupHtml(p));
    m.addTo(overlayLayer);
    overlayMarkerById[key] = m;
  });
  overlayLayer.addTo(map);
}

function initClusters() {
  const listEl = document.getElementById("cluster-list");

  if (!clustersData.length) {
    listEl.innerHTML =
      "<p>No clusters found (try relaxing keywords or DBSCAN parameters).</p>";
    map.setView([0, 0], 2);
    return;
  }

  clustersData.forEach(function (cluster) {
    const mainItem = document.createElement("div");
    mainItem.className = "cluster-item";
    mainItem.dataset.clusterId = cluster.cluster_id;
    mainItem.dataset.subset = "main";
    mainItem.innerHTML =
      "<div><strong>Cluster #" +
      cluster.cluster_id +
      "</strong></div>" +
      '<div class="cluster-meta">' +
      (cluster.count === 1 ? "1 image" : cluster.count + " images") +
      "</div>";
    mainItem.addEventListener("click", function () {
      setActiveCluster(cluster.cluster_id, "main");
    });

    const errItem = document.createElement("div");
    errItem.className = "cluster-item";
    errItem.dataset.clusterId = cluster.cluster_id;
    errItem.dataset.subset = "err";
    errItem.innerHTML =
      "<div><strong>Erroneous in #" +
      cluster.cluster_id +
      "</strong></div>" +
      '<div class="cluster-meta">0 images</div>';
    errItem.addEventListener("click", function () {
      setActiveCluster(cluster.cluster_id, "err");
    });

    listEl.appendChild(mainItem);
    listEl.appendChild(errItem);

    const mainGroup = L.featureGroup();
    const errGroup = L.featureGroup();

    const buckets = {};
    const keyPrecision = 6;

    cluster.points.forEach(function (p) {
      const ll = latLonFromRow(p);
      const key =
        ll[0].toFixed(keyPrecision) + "," + ll[1].toFixed(keyPrecision);
      if (!buckets[key]) {
        buckets[key] = { centerLat: ll[0], centerLon: ll[1], items: [] };
      }
      buckets[key].items.push(p);
    });

    const pointsById = {};
    let autoIdCounter = 0;

    Object.keys(buckets).forEach(function (bk) {
      const bucket = buckets[bk];
      const centerLat = bucket.centerLat;
      const centerLon = bucket.centerLon;
      const items = bucket.items;

      if (items.length === 1) {
        const p = items[0];
        const imageId = String(rowOf(p).id || "auto_" + autoIdCounter++);
        const feature = createImageMarker(
          p,
          centerLat,
          centerLon,
          centerLat,
          centerLon,
          cluster.cluster_id,
          imageId
        );
        feature.marker.addTo(mainGroup);
        if (feature.stem) feature.stem.addTo(mainGroup);
        pointsById[imageId] = feature;
        return;
      }

      const count = items.length;
      const angleStep = (2 * Math.PI) / count;
      const baseLatRad = (centerLat * Math.PI) / 180.0;
      const radiusDegLat = 0.00015;
      const radiusDegLon = radiusDegLat / Math.max(Math.cos(baseLatRad), 0.1);

      items.forEach(function (p, index) {
        const angle = index * angleStep;
        const markerLat = centerLat + radiusDegLat * Math.sin(angle);
        const markerLon = centerLon + radiusDegLon * Math.cos(angle);
        const imageId = String(rowOf(p).id || "auto_" + autoIdCounter++);
        const feature = createImageMarker(
          p,
          centerLat,
          centerLon,
          markerLat,
          markerLon,
          cluster.cluster_id,
          imageId
        );
        feature.marker.addTo(mainGroup);
        if (feature.stem) feature.stem.addTo(mainGroup);
        pointsById[imageId] = feature;
      });
    });

    markersByCluster[cluster.cluster_id] = {
      mainGroup: mainGroup,
      errGroup: errGroup,
      pointsById: pointsById,
      mainItem: mainItem,
      errItem: errItem,
    };
  });

  setActiveCluster(clustersData[0].cluster_id, "main", true);
}

function setActiveCluster(clusterId, subset, initial) {
  if (
    activeClusterId === clusterId &&
    activeSubset === subset &&
    !initial
  ) {
    return;
  }

  activeClusterId = clusterId;
  activeSubset = subset;

  document.querySelectorAll(".cluster-item").forEach(function (el) {
    if (
      String(el.dataset.clusterId) === String(clusterId) &&
      el.dataset.subset === subset
    ) {
      el.classList.add("active");
    } else {
      el.classList.remove("active");
    }
  });

  Object.keys(markersByCluster).forEach(function (cid) {
    const entry = markersByCluster[cid];
    map.removeLayer(entry.mainGroup);
    map.removeLayer(entry.errGroup);
  });

  const entry = markersByCluster[clusterId];
  if (!entry) {
    map.setView([0, 0], 2);
    return;
  }

  const layer = subset === "err" ? entry.errGroup : entry.mainGroup;
  if (layer) {
    layer.addTo(map);
    const toggle = document.getElementById("toggle-overlay");
    const showOverlay = toggle ? toggle.checked : false;
    const boundsParts = [layer];
    if (showOverlay && overlayData.length) boundsParts.push(overlayLayer);
    const bounds = L.featureGroup(boundsParts).getBounds();
    if (bounds && bounds.isValid && bounds.isValid()) {
      map.fitBounds(bounds.pad(0.15));
    } else {
      const b = layer.getBounds();
      if (b && b.isValid && b.isValid()) map.fitBounds(b.pad(0.2));
      else map.setView([0, 0], 2);
    }
  }
}

map.on("popupopen", function (e) {
  const layer = e.popup && e.popup._source;
  if (!layer) return;

  if (layer._cmpKind === "overlay") {
    currentPopupContext = { kind: "overlay", imageId: layer._cmpImageId };
    const root = e.popup.getElement();
    if (!root) return;
    root.querySelectorAll(".ov-nav-btn").forEach(function (btn) {
      btn.addEventListener(
        "click",
        function () {
          const nav = btn.getAttribute("data-nav");
          navigateOverlay(nav === "prev" ? -1 : 1);
        },
        { once: true }
      );
    });
    return;
  }

  const entry = markersByCluster[activeClusterId];
  if (!entry) return;
  let clusterId = activeClusterId;
  let imageId = null;
  Object.keys(entry.pointsById).forEach(function (iid) {
    const rec = entry.pointsById[iid];
    if (rec && rec.marker === layer) imageId = iid;
  });
  if (imageId === null) return;

  currentPopupContext = { kind: "cluster", clusterId: clusterId, imageId: String(imageId) };

  const root = e.popup.getElement();
  if (!root) return;

  const rec = entry.pointsById[imageId];
  const isErroneous = Boolean(rec && entry.errGroup.hasLayer(rec.marker));

  root.querySelectorAll(".nav-btn").forEach(function (navBtn) {
    navBtn.addEventListener(
      "click",
      function () {
        const navDir = navBtn.getAttribute("data-nav") === "prev" ? -1 : 1;
        navigateImage(clusterId, imageId, navDir);
      },
      { once: true }
    );
  });

  const btn = root.querySelector(".err-btn");
  if (btn) {
    btn.dataset.clusterId = String(clusterId);
    btn.dataset.imageId = String(imageId);
    if (isErroneous) {
      btn.dataset.mode = "restore";
      btn.textContent = "Mark as non-erroneous";
    } else {
      btn.dataset.mode = "error";
      btn.textContent = "Mark as erroneous";
    }
    btn.addEventListener(
      "click",
      function (ev) {
        if (ev.preventDefault) ev.preventDefault();
        if (ev.stopPropagation) ev.stopPropagation();
        const mode = btn.dataset.mode || "error";

        // Capture next target BEFORE the move, while imageId is still in the visible list.
        const ids = getVisibleImageIds(clusterId);
        const idx = ids.indexOf(String(imageId));
        let nextId = null;
        if (idx !== -1 && ids.length > 1) {
          nextId = ids[(idx + 1) % ids.length];
          if (nextId === imageId) nextId = null;
        }

        if (mode === "error") moveToErroneous(clusterId, imageId);
        else if (mode === "restore") moveToMain(clusterId, imageId);

        // Open the next marker's popup directly (bypasses the stale-id problem).
        if (nextId) {
          const entry = markersByCluster[clusterId];
          const rec = entry && entry.pointsById[nextId];
          if (rec && rec.marker) {
            rec.marker.openPopup();
            const ll = rec.marker.getLatLng();
            if (ll) map.panTo(ll, { animate: true });
          }
        }
      },
      { once: true }
    );
  }
});

map.on("popupclose", function () {
  currentPopupContext = null;
});

document.addEventListener("keydown", function (ev) {
  if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;
  if (!currentPopupContext) return;
  ev.preventDefault();
  const dir = ev.key === "ArrowLeft" ? -1 : 1;
  if (currentPopupContext.kind === "overlay") {
    navigateOverlay(dir);
    return;
  }
  navigateImage(currentPopupContext.clusterId, currentPopupContext.imageId, dir);
});

const toggleEl = document.getElementById("toggle-overlay");
if (toggleEl) {
  toggleEl.addEventListener("change", function (ev) {
    if (ev.target.checked) map.addLayer(overlayLayer);
    else map.removeLayer(overlayLayer);
    if (activeClusterId !== null) setActiveCluster(activeClusterId, activeSubset, true);
  });
}

(function () {
  var b1 = document.getElementById("export-err-btn");
  var b2 = document.getElementById("export-sanitized-btn");
  var b3 = document.getElementById("import-err-btn");
  var f = document.getElementById("import-err-file");
  if (b1) b1.addEventListener("click", exportErroneousCsv);
  if (b2) b2.addEventListener("click", exportSanitizedCsv);
  if (b3 && f) {
    b3.addEventListener("click", function () {
      f.click();
    });
    f.addEventListener("change", function (ev) {
      const file = ev.target.files && ev.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = function () {
        importErroneousFromText(String(reader.result || ""));
      };
      reader.readAsText(file);
      ev.target.value = "";
    });
  }
})();

initOverlay();
initClusters();
