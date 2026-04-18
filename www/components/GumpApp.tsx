'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import type { Map as LeafletMap, GeoJSON as LeafletGeoJSON, Rectangle as LeafletRectangle, ImageOverlay as LeafletImageOverlay } from 'leaflet'

const YEAR_COLORS: Record<number | string, string> = {
  2020: '#fde68a', 2021: '#fbbf24', 2022: '#f97316',
  2023: '#ef4444', 2024: '#b91c1c', 2025: '#7f1d1d', unknown: '#64748b',
}
const YEAR_MIN = 2020
const YEAR_MAX = 2025

interface Feature {
  type: 'Feature'
  geometry: { type: string; coordinates: unknown[] }
  properties: Record<string, unknown>
  _uid?: number
}
interface FileEntry { id: number; name: string; layer: LeafletGeoJSON; features: Feature[]; visible: boolean }

interface RegionMetric { f1: number | null; precision: number | null; recall: number | null; pr_auc?: number | null; n_pos?: number }
interface Metrics {
  status?: string
  model?: string
  overall: RegionMetric
  per_region: Record<string, RegionMetric>
  threshold: number | null
  n_tiles_train: number
  n_tiles_val: number
  val_split_strategy?: string
}

// Backend API Interfaces
interface RunSummary {
  run_id: string
  status: string
  model_type: string | null
  created_at: string | null
  artifact_tile_count: number
  metrics: { val_f1: number | null; val_iou: number | null; pr_auc: number | null }
}

interface RunDetail {
  run_id: string
  status: string
  config: any
  metrics: any
  history: any
  tiles: string[]
  artifacts_ready: boolean
}

interface TileBounds { min_lon: number; min_lat: number; max_lon: number; max_lat: number }
interface TileSummary {
  tile_id: string
  status: string
  threshold: number | null
  artifacts: { geojson: boolean; overlay_png: boolean; prob_tif: boolean }
  paths: { geojson?: string; overlay_png?: string; prob_tif?: string }
  bounds: TileBounds | null
  months_s2: [number, number][] | null
}

type ColorMode = 'year' | 'confidence'
type Composite = 'rgb' | 'falsecolor' | 'swir' | 'redge' | 'ndvi' | 'sar'
const COMPOSITES: { id: Composite; label: string; desc: string }[] = [
  { id: 'rgb',        label: 'True Color',   desc: 'B04·B03·B02' },
  { id: 'falsecolor', label: 'False Color',  desc: 'B08·B04·B03' },
  { id: 'swir',       label: 'SWIR',         desc: 'B12·B08·B04' },
  { id: 'redge',      label: 'Red Edge',     desc: 'B06·B05·B04' },
  { id: 'ndvi',       label: 'NDVI',         desc: '(B08−B04)/(B08+B04)' },
  { id: 'sar',        label: 'SAR',          desc: 'S1 VV asc.' },
]

// Ensure API URL points to the backend automatically
const API_BASE = (typeof window !== 'undefined' && window.location.hostname) 
  ? `http://${window.location.hostname}:8000/api` 
  : 'http://localhost:8000/api';

function confColor(c: number): string {
  if (c < 0.5) return '#ef4444'
  if (c < 0.7) return '#f97316'
  if (c < 0.85) return '#fbbf24'
  return '#10b981'
}

let globalUid = 0

function extractYear(f: Feature): number | null {
  const p = f.properties ?? {}
  const ts = p.time_step ?? p.year ?? p.alert_year
  if (ts == null) return null
  if (typeof ts === 'number') { if (ts > 1900 && ts < 2100) return Math.floor(ts); if (ts < 100) return 2000 + Math.floor(ts) }
  if (typeof ts === 'string') { const m = ts.match(/\b(20\d{2})\b/); if (m) return parseInt(m[1], 10) }
  return null
}
function extractConf(f: Feature): number | null {
  const c = f.properties?.confidence ?? f.properties?.conf
  return typeof c === 'number' ? c : null
}
function centroid(f: Feature): [number, number] | [null, null] {
  try {
    const flat: number[][] = []
    const walk = (a: unknown[]): void => { if (typeof a[0] === 'number') flat.push(a as number[]); else (a as unknown[][]).forEach(walk) }
    walk(f.geometry.coordinates as unknown[])
    const sx = flat.reduce((s, p) => s + p[0], 0) / flat.length
    const sy = flat.reduce((s, p) => s + p[1], 0) / flat.length
    return [sx, sy]
  } catch { return [null, null] }
}
function areaHa(f: Feature): number | null {
  if (typeof f.properties?.area_ha === 'number') return f.properties.area_ha as number
  try {
    const [, lat0] = centroid(f); if (lat0 == null) return null
    const mLat = 111320, mLon = 111320 * Math.cos(lat0 * Math.PI / 180)
    const rings = f.geometry.type === 'Polygon' ? [f.geometry.coordinates as number[][][]] : f.geometry.coordinates as number[][][][]
    let total = 0
    rings.forEach(poly => poly.forEach((ring, ri) => {
      let a = 0
      for (let i = 0; i < ring.length - 1; i++) {
        a += (ring[i][0] * mLon) * (ring[i+1][1] * mLat) - (ring[i+1][0] * mLon) * (ring[i][1] * mLat)
      }
      total += Math.abs(a / 2) * (ri === 0 ? 1 : -1)
    }))
    return total / 10000
  } catch { return null }
}

export default function GumpApp() {
  const mapRef = useRef<HTMLDivElement>(null)
  const leafletMap = useRef<LeafletMap | null>(null)
  const filesRef = useRef<FileEntry[]>([])
  const tileBoundsLayersRef = useRef<{ tile_id: string; rect: LeafletRectangle }[]>([])
  const satelliteOverlayRef = useRef<LeafletImageOverlay | null>(null)
  const nextId = useRef(1)
  const colorModeRef = useRef<ColorMode>('year')

  const [files, setFiles] = useState<FileEntry[]>([])
  const [stats, setStats] = useState({ polys: 0, area: 0, tiles: 0, yearMin: null as number | null, yearMax: null as number | null })
  const [yearFilter, setYearFilter] = useState<number>(YEAR_MAX)
  const [yearActive, setYearActive] = useState(false)
  const [confFilter, setConfFilter] = useState(0)
  const [areaFilter, setAreaFilter] = useState(0)
  const [toast, setToast] = useState<string | null>(null)
  const [modalOpen, setModalOpen] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [basemap, setBasemap] = useState('satellite')
  const [searchQuery, setSearchQuery] = useState('')
  const [metrics, setMetrics] = useState<Metrics | null>(null) // Deprecated mostly
  const [colorMode, setColorMode] = useState<ColorMode>('year')
  const toastTimer = useRef<ReturnType<typeof setTimeout>>()

  // Backend State
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [activeRunId, setActiveRunId] = useState<string | null>(null)
  const [runDetail, setRunDetail] = useState<RunDetail | null>(null)
  const [availableTiles, setAvailableTiles] = useState<TileSummary[]>([])
  const [activeTileIdx, setActiveTileIdx] = useState<number | null>(null)
  const [composite, setComposite] = useState<Composite>('rgb')
  const [selectedMonth, setSelectedMonth] = useState<number | null>(null) // null = auto
  const [satelliteLoading, setSatelliteLoading] = useState(false)

  const showToast = useCallback((msg: string) => {
    setToast(msg); clearTimeout(toastTimer.current)
    toastTimer.current = setTimeout(() => setToast(null), 2400)
  }, [])

  const passes = useCallback((f: Feature) => {
    const y = extractYear(f), c = extractConf(f), ha = areaHa(f)
    if (yearActive && y != null && y > yearFilter) return false
    if (confFilter > 0 && c != null && c < confFilter) return false
    if (areaFilter > 0 && ha != null && ha < areaFilter) return false
    return true
  }, [yearFilter, yearActive, confFilter, areaFilter])

  // init map
  useEffect(() => {
    if (!mapRef.current || leafletMap.current) return
    let cancelled = false
    import('leaflet').then(L => {
      if (cancelled || !mapRef.current || leafletMap.current) return
      const map = L.map(mapRef.current!, {
        zoomControl: false, worldCopyJump: true,
        zoomSnap: 0.1, zoomDelta: 0.5, wheelPxPerZoomLevel: 80, wheelDebounceTime: 20,
      }).setView([0, 0], 3)
      L.control.zoom({ position: 'topright' }).addTo(map)
      leafletMap.current = map

      // Custom pane for satellite imagery — sits above basemap (200) but below vectors (400)
      const satPane = map.createPane('satellite')
      satPane.style.zIndex = '250'
      satPane.style.pointerEvents = 'none'

      const tiles: Record<string, L.TileLayer> = {
        satellite: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', { attribution: 'Imagery © Esri', maxZoom: 19 }),
        terrain: L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', { attribution: '© OpenTopoMap', maxZoom: 17 }),
        osm: L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '© OpenStreetMap', maxZoom: 19 }),
      }
      tiles.satellite.addTo(map)
      ;(map as unknown as Record<string, unknown>)._basemapLayers = tiles

      const legend = (L.control as unknown as (opts: object) => L.Control)({ position: 'bottomright' })
      legend.onAdd = () => {
        const div = L.DomUtil.create('div', 'legend')
        let html = '<div class="title">Jahr der Deforestation</div>'
        for (let y = YEAR_MIN; y <= YEAR_MAX; y++) html += `<div class="legend-row"><span class="swatch" style="background:${YEAR_COLORS[y]}"></span>${y}</div>`
        html += `<div class="legend-row"><span class="swatch" style="background:${YEAR_COLORS.unknown}"></span>unbekannt</div>`
        div.innerHTML = html; return div
      }
      legend.addTo(map)

      // Fetch available runs from API
      fetch(`${API_BASE}/runs`)
        .then(r => r.ok ? r.json() : null)
        .then((data: { runs: RunSummary[] } | null) => {
          if (data && data.runs.length > 0) {
            setRuns(data.runs)
            // Auto-select latest finished run with artifacts
            const bestRun = data.runs.find(r => r.status === 'finished' && r.artifact_tile_count > 0) || data.runs[0]
            if (bestRun) setActiveRunId(bestRun.run_id)
          }
        }).catch(err => console.error("Failed to load runs", err))
    })
    return () => { cancelled = true; leafletMap.current?.remove(); leafletMap.current = null }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Load run details & tiles when activeRunId changes
  useEffect(() => {
    if (!activeRunId) return
    
    // Load metrics/config
    fetch(`${API_BASE}/runs/${activeRunId}`)
      .then(r => r.ok ? r.json() : null)
      .then((detail: RunDetail | null) => {
        if (detail) setRunDetail(detail)
      })

    // Load available tiles (GeoJSONs)
    fetch(`${API_BASE}/runs/${activeRunId}/tiles`)
      .then(r => r.ok ? r.json() : null)
      .then(async (data: { tiles: TileSummary[] } | null) => {
        if (!data || !data.tiles) return
        setAvailableTiles(data.tiles)
        
        // Clear existing map layers and tile bounds
        filesRef.current.forEach(f => f.layer.remove())
        filesRef.current = []
        setFiles([])
        tileBoundsLayersRef.current.forEach(({ rect }) => rect.remove())
        tileBoundsLayersRef.current = []
        satelliteOverlayRef.current?.remove()
        satelliteOverlayRef.current = null
        setActiveTileIdx(null)

        // Load GeoJSON for each tile
        if (!leafletMap.current) return
        const L = (await import('leaflet')).default

        for (const t of data.tiles) {
          // Draw tile boundary rectangle
          if (t.bounds) {
            const { min_lat, min_lon, max_lat, max_lon } = t.bounds
            const rect = L.rectangle([[min_lat, min_lon], [max_lat, max_lon]], {
              color: '#94a3b8', weight: 1.5, fill: false, dashArray: '6 4', opacity: 0.7,
            }).addTo(leafletMap.current!)
            rect.bindTooltip(t.tile_id, { permanent: false, direction: 'center', className: 'tile-label' })
            tileBoundsLayersRef.current.push({ tile_id: t.tile_id, rect })
          }

          if (t.paths.geojson) {
            const url = `http://${window.location.hostname}:8000${t.paths.geojson}`
            try {
              const res = await fetch(url)
              if (res.ok) {
                const geojson = await res.json()
                ingest(t.tile_id, geojson, leafletMap.current, L)
              }
            } catch (e) {
              console.error(`Failed to load tile ${t.tile_id}`, e)
            }
          }
        }

        if (data.tiles.length > 0) setActiveTileIdx(0)
      })
  }, [activeRunId])

  // Reset month selection when tile changes
  useEffect(() => { setSelectedMonth(null) }, [activeTileIdx])

  // Satellite preview overlay
  useEffect(() => {
    if (activeTileIdx === null || !leafletMap.current) return
    const tile = availableTiles[activeTileIdx]
    if (!tile?.bounds) return

    const { min_lat, min_lon, max_lat, max_lon } = tile.bounds
    const bounds: [[number, number], [number, number]] = [[min_lat, min_lon], [max_lat, max_lon]]
    const monthParam = selectedMonth !== null ? selectedMonth : 'auto'
    const url = `http://${window.location.hostname}:8000/api/tiles/${tile.tile_id}/preview.png?composite=${composite}&month=${monthParam}`

    setSatelliteLoading(true)
    import('leaflet').then(({ default: L }) => {
      // Remove previous overlay
      satelliteOverlayRef.current?.remove()
      const overlay = L.imageOverlay(url, bounds, { opacity: 0.85, pane: 'satellite' })
      overlay.addTo(leafletMap.current!)
      overlay.on('load', () => setSatelliteLoading(false))
      overlay.on('error', () => setSatelliteLoading(false))
      satelliteOverlayRef.current = overlay
    })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTileIdx, composite, selectedMonth])

  // basemap switch
  useEffect(() => {
    if (!leafletMap.current) return
    import('leaflet').then(() => {
      const map = leafletMap.current!
      const layers = (map as unknown as Record<string, Record<string, L.TileLayer>>)._basemapLayers
      if (!layers) return
      Object.values(layers).forEach(l => map.removeLayer(l as L.Layer))
      layers[basemap].addTo(map)
    })
  }, [basemap])

  // filter changes
  useEffect(() => {
    filesRef.current.forEach(f => {
      f.layer.eachLayer(lyr => {
        const el = (lyr as unknown as { getElement?: () => HTMLElement | null }).getElement?.()
        if (el) { const show = passes((lyr as unknown as { feature: Feature }).feature); el.style.display = show ? '' : 'none'; el.style.pointerEvents = show ? '' : 'none' }
      })
    })
    refreshStats()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [yearFilter, yearActive, confFilter, areaFilter])

  // color mode changes → restyle existing layers
  useEffect(() => {
    colorModeRef.current = colorMode
    filesRef.current.forEach(f => {
      f.layer.eachLayer(lyr => {
        const feat = (lyr as unknown as { feature: Feature }).feature
        const conf = extractConf(feat)
        const c = (colorMode === 'confidence' && conf != null)
          ? confColor(conf)
          : YEAR_COLORS[extractYear(feat) ?? 'unknown']
        ;(lyr as unknown as { setStyle: (s: object) => void }).setStyle({ color: c, fillColor: c })
      })
    })
  }, [colorMode])

  function refreshStats() {
    const vis = filesRef.current.flatMap(f => f.visible ? f.features.filter(passes) : [])
    const area = vis.reduce((s, f) => s + (areaHa(f) ?? 0), 0)
    const years = vis.map(extractYear).filter((y): y is number => y != null)
    setStats({ polys: vis.length, area, tiles: filesRef.current.filter(f => f.visible).length, yearMin: years.length ? Math.min(...years) : null, yearMax: years.length ? Math.max(...years) : null })
  }

  function ingest(name: string, data: { type: string; features: Feature[] }, map: LeafletMap, L: typeof import('leaflet')) {
    if (data?.type !== 'FeatureCollection' || !data.features?.length) { showToast(`⚠ ${name}: leer oder ungültig`); return }
    data.features.forEach(f => { f._uid = ++globalUid })

    const styleFor = (f: Feature) => {
      const mode = colorModeRef.current
      const conf = extractConf(f)
      const c = (mode === 'confidence' && conf != null)
        ? confColor(conf)
        : YEAR_COLORS[extractYear(f) ?? 'unknown']
      return { color: c, weight: 1.5, fillColor: c, fillOpacity: 0.55, opacity: 0.9 }
    }
    const layer = L.geoJSON(data as unknown as Parameters<typeof L.geoJSON>[0], {
      style: (f) => styleFor(f as unknown as Feature),
      onEachFeature: (f, lyr) => {
        const feat = f as unknown as Feature
        const [lon, lat] = centroid(feat)
        const y = extractYear(feat), ha = areaHa(feat), c = extractConf(feat)
        const color = YEAR_COLORS[y ?? 'unknown']
        lyr.bindPopup(`<div class="popup">
          <div class="popup-header"><span class="popup-badge" style="background:${color}"></span><span class="popup-title">Deforestation Event</span></div>
          <dl class="popup-grid">
            <dt>ID</dt><dd>#${feat._uid}</dd>
            <dt>Lat, Lon</dt><dd>${lat != null ? `${lat.toFixed(5)}, ${lon!.toFixed(5)}` : '—'}</dd>
            <dt>Fläche</dt><dd>${ha != null ? ha.toFixed(2) + ' ha' : '—'}</dd>
            <dt>Jahr</dt><dd>${y ?? '—'}</dd>
            ${c != null ? `<dt>Confidence</dt><dd>${c.toFixed(2)}</dd>` : ''}
          </dl>
          <div class="popup-actions">
            <button class="popup-action" onclick="navigator.clipboard.writeText('${lat?.toFixed(6)}, ${lon?.toFixed(6)}')">📋 Kopieren</button>
            <button class="popup-action" onclick="window.open('https://www.google.com/maps?q=${lat},${lon}&t=k&z=16','_blank')">🗺 Maps</button>
          </div>
        </div>`, { maxWidth: 320 })
        lyr.on('mouseover', () => (lyr as unknown as { setStyle: (s: object) => void }).setStyle({ weight: 3, fillOpacity: 0.8 }))
        lyr.on('mouseout', () => (lyr as unknown as { setStyle: (s: object) => void }).setStyle(styleFor(feat)))
      },
    }).addTo(map)

    const entry: FileEntry = { id: nextId.current++, name, layer, features: data.features, visible: true }
    filesRef.current = [...filesRef.current, entry]
    setFiles([...filesRef.current])
    map.fitBounds(layer.getBounds(), { padding: [40, 40], maxZoom: 13 })
    showToast(`✓ ${name} · ${data.features.length} Polygone`)
    refreshStats()
  }

  async function handleFiles(fileList: FileList) {
    if (!leafletMap.current) return
    const L = (await import('leaflet')).default
    const arr = Array.from(fileList).filter(f => /\.(geojson|json)$/i.test(f.name))
    if (!arr.length) { showToast('⚠ Keine GeoJSON-Dateien'); return }
    for (const file of arr) {
      try { ingest(file.name, JSON.parse(await file.text()), leafletMap.current, L) }
      catch (e) { showToast(`⚠ ${file.name}: ${(e as Error).message}`) }
    }
  }

  function jumpToTile(idx: number) {
    const tiles = availableTiles
    if (!tiles.length || !leafletMap.current) return
    const i = (idx + tiles.length) % tiles.length
    setActiveTileIdx(i)
    const t = tiles[i]
    // Highlight: reset all rects, emphasise the active one
    tileBoundsLayersRef.current.forEach(({ tile_id, rect }) => {
      if (tile_id === t.tile_id) {
        rect.setStyle({ color: '#38bdf8', weight: 2.5, opacity: 1 })
      } else {
        rect.setStyle({ color: '#94a3b8', weight: 1.5, opacity: 0.7 })
      }
    })
    if (t.bounds) {
      const { min_lat, min_lon, max_lat, max_lon } = t.bounds
      leafletMap.current.fitBounds([[min_lat, min_lon], [max_lat, max_lon]], { padding: [60, 60], maxZoom: 13 })
    }
  }

  function toggleVisible(id: number) {
    const f = filesRef.current.find(x => x.id === id); if (!f) return
    f.visible = !f.visible
    if (f.visible) leafletMap.current && f.layer.addTo(leafletMap.current); else f.layer.remove()
    filesRef.current = [...filesRef.current]
    setFiles([...filesRef.current]); refreshStats()
  }
  function removeFile(id: number) {
    const f = filesRef.current.find(x => x.id === id); if (!f) return
    f.layer.remove(); filesRef.current = filesRef.current.filter(x => x.id !== id)
    setFiles([...filesRef.current]); refreshStats(); showToast(`${f.name} entfernt`)
  }
  function zoomToFile(id: number) {
    const f = filesRef.current.find(x => x.id === id)
    if (f?.visible) leafletMap.current?.fitBounds(f.layer.getBounds(), { padding: [30, 30] })
  }
  function fitAll() {
    const visible = filesRef.current.filter(f => f.visible)
    if (!visible.length || !leafletMap.current) return
    import('leaflet').then(L => {
      const group = L.featureGroup(visible.map(f => f.layer))
      leafletMap.current!.fitBounds(group.getBounds(), { padding: [40, 40], maxZoom: 13 })
    })
  }

  function exportCSV() {
    const rows = [['file','id','lon','lat','area_ha','year','confidence']]
    filesRef.current.forEach(f => f.features.filter(passes).forEach(ft => {
      const [lon, lat] = centroid(ft)
      rows.push([f.name, String(ft._uid), lon?.toFixed(6) ?? '', lat?.toFixed(6) ?? '', (areaHa(ft) ?? '').toString(), (extractYear(ft) ?? '').toString(), (extractConf(ft) ?? '').toString()])
    }))
    const csv = rows.map(r => r.map(x => `"${x.replace(/"/g, '""')}"`).join(',')).join('\n')
    const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' })); a.download = 'deforestation.csv'; a.click()
    showToast(`✓ ${rows.length - 1} Polygone exportiert`)
  }
  function exportGeoJSON() {
    const features = filesRef.current.flatMap(f => f.features.filter(passes).map(ft => ({ ...ft, properties: { ...ft.properties, _source: f.name } })))
    const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([JSON.stringify({ type: 'FeatureCollection', features }, null, 2)], { type: 'application/geo+json' })); a.download = 'deforestation.geojson'; a.click()
    showToast(`✓ Merged GeoJSON exportiert`)
  }

  // bar chart data
  const barCounts: Record<number, number> = {}
  for (let y = YEAR_MIN; y <= YEAR_MAX; y++) barCounts[y] = 0
  filesRef.current.flatMap(f => f.visible ? f.features.filter(passes) : []).forEach(f => { const y = extractYear(f); if (y != null && barCounts[y] != null) barCounts[y]++ })
  const barMax = Math.max(1, ...Object.values(barCounts))

  // search
  const allVisible = filesRef.current.flatMap(f => f.visible ? f.features.filter(passes).map(ft => ({ file: f, ft })) : [])
  const searchResults = searchQuery.trim()
    ? allVisible.filter(({ ft }) => `#${ft._uid}`.includes(searchQuery) || centroid(ft).map(x => x?.toFixed(3)).join(' ').includes(searchQuery)).slice(0, 40)
    : allVisible.slice(0, 40)

  function flyTo(ft: Feature) {
    const [lon, lat] = centroid(ft); if (lat == null) return
    leafletMap.current?.flyTo([lat, lon!], 15, { duration: 0.6 })
    const entry = filesRef.current.find(f => f.features.includes(ft))
    if (entry) setTimeout(() => entry.layer.eachLayer(l => { if ((l as unknown as { feature: Feature }).feature === ft) (l as unknown as { openPopup: () => void }).openPopup() }), 650)
  }

  const hasData = files.length > 0
  const hasYears = files.some(f => f.features.some(ft => extractYear(ft) != null))

  return (
    <div className="layout">
      {/* TOPBAR */}
      <header className="topbar">
        <div className="brand">
          <div className="logo">
            <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2L12 22M5 9L12 2L19 9M7 15L12 10L17 15"/>
            </svg>
          </div>
          <div><h1>Gump</h1><p className="tagline">Deforestation Detection · osapiens Makeathon 2026</p></div>
        </div>
        <div className="headline-stats">
          <div className="hstat"><span className="hstat-value">{stats.polys.toLocaleString('de-DE')}</span><span className="hstat-label">Polygone</span></div>
          <div className="hstat"><span className="hstat-value">{stats.area >= 1000 ? (stats.area/1000).toFixed(1)+'k' : stats.area.toFixed(1)}</span><span className="hstat-label">Hektar</span></div>
          <div className="hstat"><span className="hstat-value">{stats.tiles || '—'}</span><span className="hstat-label">Tiles</span></div>
          <div className="hstat"><span className="hstat-value">{stats.yearMin ? (stats.yearMin === stats.yearMax ? `${stats.yearMin}` : `${stats.yearMin}–${stats.yearMax}`) : '—'}</span><span className="hstat-label">Zeitraum</span></div>
        </div>
        <div className="topbar-actions">
          {/* Run Selector */}
          <div style={{ marginRight: 15 }}>
            <select 
              className="input" 
              style={{ width: 220, padding: '6px 10px' }}
              value={activeRunId || ''} 
              onChange={e => setActiveRunId(e.target.value)}
            >
              <option value="" disabled>Select Model Run...</option>
              {runs.map(r => (
                <option key={r.run_id} value={r.run_id}>
                  {r.run_id} {r.status === 'finished' ? '✓' : ''}
                </option>
              ))}
            </select>
          </div>
          <button className="btn-ghost" onClick={() => setModalOpen(true)}>
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
            Info
          </button>
        </div>
      </header>

      {/* SIDEBAR */}
      <aside className="sidebar">
        {availableTiles.length > 0 && (
          <section className="panel">
            <h2>Tile Navigator</h2>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
              <button className="btn" style={{ padding: '4px 10px', minWidth: 32 }}
                onClick={() => jumpToTile((activeTileIdx ?? 0) - 1)}>‹</button>
              <span style={{ flex: 1, textAlign: 'center', fontFamily: 'monospace', fontSize: 13 }}>
                {activeTileIdx !== null ? availableTiles[activeTileIdx]?.tile_id : '—'}
              </span>
              <button className="btn" style={{ padding: '4px 10px', minWidth: 32 }}
                onClick={() => jumpToTile((activeTileIdx ?? 0) + 1)}>›</button>
            </div>
            <ul className="file-list" style={{ maxHeight: 140, overflowY: 'auto' }}>
              {availableTiles.map((t, i) => (
                <li key={t.tile_id}
                  style={{ cursor: 'pointer', background: i === activeTileIdx ? 'var(--surface-2, #1e293b)' : undefined }}
                  onClick={() => jumpToTile(i)}>
                  <span className="fname" style={{ fontFamily: 'monospace', fontSize: 12 }}>{t.tile_id}</span>
                  <span className="fcount">{t.artifacts.geojson ? '●' : '○'}</span>
                </li>
              ))}
            </ul>
            {(() => {
              const tile = availableTiles[activeTileIdx ?? 0]
              const months = tile?.months_s2
              if (!months?.length) return null
              const idx = selectedMonth ?? months.length - 1
              const [yr, mo] = months[Math.min(idx, months.length - 1)]
              const label = `${['Jan','Feb','Mär','Apr','Mai','Jun','Jul','Aug','Sep','Okt','Nov','Dez'][mo-1]} ${yr}`
              return (
                <div style={{ marginTop: 10 }}>
                  <div className="filter-label-row">
                    <label>Zeitpunkt</label>
                    <strong style={{ fontFamily: 'monospace', fontSize: 12 }}>{selectedMonth !== null ? label : `Auto · ${label}`}</strong>
                  </div>
                  <input type="range" min={0} max={months.length - 1} step={1}
                    value={selectedMonth ?? months.length - 1}
                    onChange={e => setSelectedMonth(+e.target.value)}
                    style={{ width: '100%', marginTop: 4 }}
                  />
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-dim,#94a3b8)', marginTop: 2 }}>
                    <span>{months[0][0]}</span>
                    {[...new Set(months.map(m => m[0]))].slice(1).map(y => <span key={y}>{y}</span>)}
                  </div>
                  {selectedMonth !== null && (
                    <button className="btn-link" style={{ marginTop: 4, fontSize: 11 }}
                      onClick={() => setSelectedMonth(null)}>↺ Auto (bestes Bild)</button>
                  )}
                </div>
              )
            })()}

            <div style={{ marginTop: 10 }}>
              <div className="filter-label-row">
                <label>Spektrum</label>
                {satelliteLoading && <span className="muted small">lädt…</span>}
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 4, marginTop: 4 }}>
                {COMPOSITES.map(c => (
                  <button
                    key={c.id}
                    title={c.desc}
                    onClick={() => setComposite(c.id)}
                    style={{
                      padding: '4px 6px', fontSize: 11, borderRadius: 4, cursor: 'pointer', border: '1px solid',
                      borderColor: composite === c.id ? '#38bdf8' : 'var(--border, #334155)',
                      background: composite === c.id ? 'rgba(56,189,248,0.15)' : 'transparent',
                      color: composite === c.id ? '#38bdf8' : 'var(--text-dim, #94a3b8)',
                      fontWeight: composite === c.id ? 600 : 400,
                    }}
                  >{c.label}</button>
                ))}
              </div>
            </div>
            <p className="muted small" style={{ marginTop: 6 }}>
              {activeTileIdx !== null ? `${activeTileIdx + 1} / ${availableTiles.length}` : `${availableTiles.length} tiles`}
            </p>
          </section>
        )}

        <section className="panel metrics-panel">
          <h2>Modell-Metriken</h2>
          {runDetail ? (
            <>
              <div className="metric-big">
                <div className="metric-big-value">{runDetail.metrics?.val_f1 != null ? (runDetail.metrics.val_f1 * 100).toFixed(1) : '—'}</div>
                <div className="metric-big-label">F1-Score (Val)</div>
              </div>
              <div className="metric-row">
                <div className="metric-cell"><span className="metric-val">{runDetail.metrics?.val_iou != null ? (runDetail.metrics.val_iou * 100).toFixed(1) : '—'}</span><span className="metric-lbl">Val IoU</span></div>
                <div className="metric-cell"><span className="metric-val">{runDetail.metrics?.best_threshold != null ? runDetail.metrics.best_threshold.toFixed(2) : '—'}</span><span className="metric-lbl">Threshold</span></div>
                <div className="metric-cell"><span className="metric-val">{runDetail.config?.model ?? 'Baseline'}</span><span className="metric-lbl">Architektur</span></div>
              </div>
              <p className="muted small" style={{ marginTop: 10 }}>
                Run: {runDetail.run_id} <br/>
                Status: {runDetail.status} · Artifacts: {runDetail.tiles.length} Tiles
              </p>
            </>
          ) : (
            <div className="metrics-placeholder">
              <div className="metrics-placeholder-pulse" />
              <p><strong>Lade Modelldaten…</strong></p>
              <p className="muted small">Bitte einen Run im Menü oben auswählen.</p>
            </div>
          )}
        </section>

        <section className="panel">
          <h2>Daten laden</h2>
          <div
            className={`drop-zone${isDragging ? ' drag-active' : ''}`}
            onDragEnter={e => { e.preventDefault(); setIsDragging(true) }}
            onDragOver={e => { e.preventDefault(); setIsDragging(true) }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={e => { e.preventDefault(); setIsDragging(false); if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files) }}
          >
            <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <p><strong>Drag &amp; Drop</strong></p>
            <p className="muted small">Lokale GeoJSON-Dateien hier ablegen (optional)</p>
            <label className="btn" style={{ cursor: 'pointer', marginTop: 8 }}>
              Oder Dateien wählen
              <input type="file" accept=".geojson,.json" multiple hidden onChange={e => { if (e.target.files) handleFiles(e.target.files); e.target.value = '' }} />
            </label>
          </div>
        </section>

        {hasData && (
          <section className="panel">
            <h2>Filter</h2>
            <div className="filter-row">
              <div className="filter-label-row"><label>Jahr ≤</label><strong>{yearActive ? `≤ ${yearFilter}` : 'alle'}</strong></div>
              <input type="range" min={YEAR_MIN} max={YEAR_MAX} step={1} value={yearFilter} onChange={e => { setYearFilter(+e.target.value); setYearActive(true) }} />
            </div>
            <div className="filter-row">
              <div className="filter-label-row"><label>Confidence ≥</label><strong>{confFilter.toFixed(2)}</strong></div>
              <input type="range" min={0} max={1} step={0.05} value={confFilter} onChange={e => setConfFilter(+e.target.value)} />
            </div>
            <div className="filter-row">
              <div className="filter-label-row"><label>Fläche ≥</label><strong>{areaFilter.toFixed(1)} ha</strong></div>
              <input type="range" min={0} max={50} step={0.5} value={areaFilter} onChange={e => setAreaFilter(+e.target.value)} />
            </div>
            <button className="btn-link" onClick={() => { setYearActive(false); setYearFilter(YEAR_MAX); setConfFilter(0); setAreaFilter(0) }}>↺ Filter zurücksetzen</button>
          </section>
        )}

        {hasData && hasYears && (
          <section className="panel">
            <h2>Zeitliche Verteilung</h2>
            <div className="year-chart">
              {Array.from({ length: YEAR_MAX - YEAR_MIN + 1 }, (_, i) => YEAR_MIN + i).map(y => (
                <div key={y} className="bar" style={{ height: `${(barCounts[y] / barMax) * 100}%`, background: `linear-gradient(180deg, ${YEAR_COLORS[y]} 0%, transparent 120%)` }}
                  onClick={() => { setYearFilter(y); setYearActive(true) }}>
                  <span className="bar-count">{barCounts[y] || ''}</span>
                  <span className="bar-label">&apos;{String(y).slice(2)}</span>
                </div>
              ))}
            </div>
          </section>
        )}

        {hasData && (
          <section className="panel">
            <h2>Layer</h2>
            <ul className="file-list">
              {files.map(f => (
                <li key={f.id}>
                  <input type="checkbox" checked={f.visible} onChange={() => toggleVisible(f.id)} />
                  <span className="fname" title={f.name}>{f.name}</span>
                  <span className="fcount">{f.features.length}</span>
                  <button className="ficon" title="Zoomen" onClick={() => zoomToFile(f.id)}>
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                  </button>
                  <button className="ficon" title="Entfernen" onClick={() => removeFile(f.id)}>
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )}

        {hasData && (
          <section className="panel">
            <h2>Polygon-Suche</h2>
            <input className="input" type="text" placeholder="Suche #ID oder Koordinaten …" value={searchQuery} onChange={e => setSearchQuery(e.target.value)} />
            <ul className="search-results">
              {searchResults.map(({ file, ft }) => {
                const [lon, lat] = centroid(ft)
                return (
                  <li key={ft._uid} onClick={() => flyTo(ft)}>
                    <div><div className="sr-id">#{ft._uid}</div><div className="sr-coords">{lat?.toFixed(4)}, {lon?.toFixed(4)}</div></div>
                    <div className="sr-meta">{extractYear(ft) ?? '?'} · {areaHa(ft)?.toFixed(1) ?? '?'} ha</div>
                  </li>
                )
              })}
              {allVisible.length > 40 && !searchQuery && <li style={{ cursor: 'default', borderColor: 'transparent', color: 'var(--text-dim)' }}>Zeige 40 von {allVisible.length} — Suche zum Filtern</li>}
            </ul>
          </section>
        )}

        <section className="panel">
          <h2>Export</h2>
          <div className="btn-row">
            <button className="btn" disabled={!hasData} onClick={exportCSV}>CSV</button>
            <button className="btn-secondary" disabled={!hasData} onClick={exportGeoJSON}>GeoJSON</button>
          </div>
          <p className="muted small" style={{ marginTop: 8 }}>Koordinaten EPSG:4326 (lon/lat)</p>
        </section>

        <section className="panel">
          <h2>Ansicht</h2>
          <div className="filter-label-row"><label>Basiskarte</label></div>
          <div className="segmented">
            {(['satellite', 'terrain', 'osm'] as const).map(b => (
              <button key={b} className={`seg-btn${basemap === b ? ' active' : ''}`} onClick={() => setBasemap(b)}>
                {b === 'satellite' ? 'Satellit' : b === 'terrain' ? 'Terrain' : 'OSM'}
              </button>
            ))}
          </div>
          <div className="filter-label-row" style={{ marginTop: 12 }}><label>Einfärben nach</label></div>
          <div className="segmented">
            {(['year', 'confidence'] as const).map(m => (
              <button key={m} className={`seg-btn${colorMode === m ? ' active' : ''}`} onClick={() => setColorMode(m)}>
                {m === 'year' ? 'Jahr' : 'Konfidenz'}
              </button>
            ))}
          </div>
          {colorMode === 'confidence' && (
            <div className="conf-legend">
              <div className="conf-legend-bar" />
              <div className="conf-legend-labels"><span>0.0</span><span>0.5</span><span>1.0</span></div>
            </div>
          )}
          <button className="btn-link" style={{ marginTop: 12 }} onClick={fitAll}>⌖ Auf alle zoomen</button>
        </section>
      </aside>

      {/* MAP */}
      <div
        className="map-container"
        ref={mapRef}
        onDragOver={e => e.preventDefault()}
        onDrop={e => { e.preventDefault(); if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files) }}
      />

      {/* TOAST */}
      {toast && <div className="toast">{toast}</div>}

      {/* MODAL */}
      <div className={`modal${modalOpen ? ' is-open' : ''}`} onClick={() => setModalOpen(false)}>
        <div className="modal-content" onClick={e => e.stopPropagation()}>
          <button className="modal-close" onClick={() => setModalOpen(false)}>×</button>
          <h2>Gump — Deforestation Detection</h2>
          <p className="modal-lead">Pixel-genaue Entwaldungs-Erkennung aus multimodalen Satelliten-Zeitreihen für die EU Deforestation Regulation (EUDR).</p>

          <div className="story-grid">
            <div className="story-item">
              <div className="story-num">1</div>
              <div>
                <h4>Problem</h4>
                <p>EUDR verpflichtet Unternehmen zum Nachweis entwaldungsfreier Lieferketten. Eine Packung Kaffee hat hunderte Ursprünge — manuelle Prüfung skaliert nicht.</p>
              </div>
            </div>
            <div className="story-item">
              <div className="story-num">2</div>
              <div>
                <h4>Daten</h4>
                <p>Drei Modalitäten: <strong>Sentinel-1</strong> Radar (wetterunabhängig), <strong>Sentinel-2</strong> Optik (12 Spektralbänder), <strong>AlphaEarth</strong> 64-dim Foundation-Embeddings. Labels aus RADD, GLAD-L, GLAD-S2 — alle verrauscht.</p>
              </div>
            </div>
            <div className="story-item">
              <div className="story-num">3</div>
              <div>
                <h4>Ansatz</h4>
                <p>Konsens-Label (≥2 von 3 Quellen stimmen) + per-Pixel-Konfidenz-Gewichtung. Forest-2020-Maske als harter Post-Filter. Region-basierter Train/Val-Split — kein Leakage.</p>
              </div>
            </div>
            <div className="story-item">
              <div className="story-num">4</div>
              <div>
                <h4>Output</h4>
                <p>Polygone ≥ 0.5 ha in EPSG:4326 mit Jahr und Konfidenz. Klick auf ein Polygon zeigt Details. Filter nach Jahr/Konfidenz/Fläche, Export als CSV oder merged GeoJSON.</p>
              </div>
            </div>
          </div>

          <button className="btn" style={{ marginTop: 20, width: '100%' }} onClick={() => setModalOpen(false)}>
            Start Exploring →
          </button>
          <p className="muted small" style={{ marginTop: 12, textAlign: 'center' }}>osapiens Makeathon 2026 · Team Gump</p>
        </div>
      </div>
    </div>
  )
}
