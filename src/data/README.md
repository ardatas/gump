# `src/data/` — Preprocessing-Pipeline (PLAN.md §3.2)

Dieses Modul löst genau **eine** Aufgabe: die heterogenen Satelliten-Rohdaten (Sentinel-2, Sentinel-1, AlphaEarth-Embeddings) in einen **mmap-baren, pro-Tile gecacheten Tensor-Stack** überführen, damit der spätere Dataloader Rasterio-frei und I/O-leicht arbeiten kann.

Ziel dieser README: erklären **warum** wir jede Entscheidung so getroffen haben — nicht nur **was** im Code steht.

---

## 1. Warum Preprocessing überhaupt?

Der Hot-Loop beim Training (siehe PLAN §5) zieht pro Batch tausende Pixel-Samples. Würde der Dataloader bei jedem Zugriff `rasterio.open()` auf ein 10–250 MB TIFF ausführen und on-the-fly reprojizieren, wären wir **I/O- und CPU-bound**, nicht GPU-bound. Die MI300X (192 GB VRAM) würde leer laufen.

**Kernidee:** Einmalig alle Kosten bezahlen (Decoding, Reprojection, Dtype-Cast) und das Ergebnis so ablegen, dass der Dataloader nur noch `np.load(..., mmap_mode="r")` macht und über Slice-Operationen Patches zieht. Mmap-Zugriffe gehen direkt gegen den Kernel-Page-Cache — in der Praxis sub-millisecond.

---

## 2. Warum dieses Cache-Format (`.npy` pro Array + `meta.json`)?

Drei Optionen wurden gegeneinander abgewogen:

| Option | Mmap-bar? | Dependency | Bemerkung |
|---|---|---|---|
| **Verzeichnis mit `.npy` + `meta.json`** (gewählt) | ✅ | keine | Simpel, direkt, stdlib-only. |
| `.npz` pro Tile | ❌ | keine | Ist ein ZIP-Archiv. `np.load(..., mmap_mode=...)` erlaubt nur Single-`.npy`. |
| Zarr-Store | ✅ | `zarr` | Chunked & parallel-write-safe, aber Overkill für 21 Tiles. |

Das Format pro Tile (`/workspace/cache/{tile}/`):

```
meta.json      # Grid-Metadaten, Monats-/Jahres-Index, Split
s2.npy         # (T, 12, H, W)  float16
s1_asc.npy     # (T, 1, H, W)   float16, dB, clip [-30, 0]
s1_desc.npy    # (T, 1, H, W)   float16
aef.npy        # (Y, 64, H, W)  float16
```

**Warum `float16`?** Platz-Halbierung gegenüber `float32` (60 GB → 30 GB relevant) bei vernachlässigbarem Informationsverlust für Satellitendaten. Bei S2-Reflectance bis ~15000 und S1 dB in [-30, 0] reicht die fp16-Mantisse (11 Bit ≈ 1 Teil in 2048) locker. AEF liegt meist in [-0.5, 0.5] — ebenfalls unkritisch.

**Warum `meta.json` separat statt `.npz`-Zip-Metadaten?** `meta.json` ist human-lesbar, diff-bar und muss nicht zum Lesen der Arrays geöffnet werden. Person B (Label-Harmonisierung) kann ihn ohne Numpy laden, um das Ziel-Grid zu kennen.

---

## 3. Warum ein gemeinsames Ziel-Grid — und warum das S2-Grid?

Die drei Quellen liegen nativ in **unterschiedlichen Koordinatensystemen und Auflösungen**:

| Quelle | CRS | Auflösung | Bänder |
|---|---|---|---|
| Sentinel-2 L2A | UTM (pro MGRS-Zone) | 10 m | 12 spektral |
| Sentinel-1 RTC | UTM (i.d.R. gleich wie S2) | 10 m | 1 (VV) |
| AlphaEarth | **EPSG:4326 (lat/lon)** | ~10 m äquatornah | 64 |

Das Modell will pro Pixel einen gemeinsamen Feature-Vektor. Also muss **alles auf dasselbe Raster**. Drei Gründe, warum das S2-UTM-Grid die Wahl ist:

1. **Metrische Korrektheit:** UTM ist ein kartesisches, metrisches System. Polygonflächen (z.B. für die `min_area_ha`-Filterung in der Submission) sind direkt in Quadratmetern berechenbar — in EPSG:4326 wäre das verzerrt.
2. **S2 ist unser Informationsreichstes Signal:** 12 Bänder × monatlich × 10 m. Das Grid um seine native Auflösung zu wählen, vermeidet Resampling-Blur auf dem wichtigsten Input.
3. **S1 liegt sowieso schon auf diesem Grid** (oder sehr nah dran) — minimiert Warping.

**Konsequenz:** AEF muss **immer** gewarpt werden (EPSG:4326 → UTM). S1 wird ebenfalls durch `read_on_grid` geschickt, weil gelegentlich winzige Transform-Abweichungen vorkommen und wir eine Pixel-genaue Alignment-Garantie wollen.

---

## 4. Warum eine "Referenz-Frame"-Logik?

In `prepare_cache.py::prepare_tile`:

```python
def _area(p: Path) -> int:
    with rasterio.open(p) as s:
        return s.height * s.width

ref_path = max((p for _, p in s2_list), key=_area)
transform, crs, H, W = open_reference_grid(ref_path)
```

**Warum nicht einfach das erste (chronologisch älteste) S2-File?**

Ursprüngliches Design war genau das. Der erste volle Run förderte zwei kaputte Tiles zutage:

- `18NYH_9_9`: S2 2020-01 ist nur **6×4 Pixel** groß (korrupte/truncated Datei). Spätere Monate sind 1004×1004 normal. Hätten wir 6×4 als Ziel-Grid genommen, wären alle 72 Frames × 12 Bänder auf 6×4 heruntergesampelt worden → Tile effektiv wertlos.
- `18NWM_9_4`: **alle** Frames sind 1004×2 (Tile liegt am UTM-Zonen-Rand, nur ein schmaler Streifen schneidet die Zone). Hier hilft die max-Area-Heuristik nichts — intrinsisches Problem. Der Tile hat real nur wenige Pixel echter Daten. Wird später im Dataloader-Sampling automatisch vernachlässigt, weil er gewichtet nach Pixelzahl kaum Batches beiträgt.

**Lehre:** Nie blind auf "erste Datei" verlassen. Schon eine triviale Robustheits-Heuristik (max-area) rettet einen Tile vollständig.

---

## 5. Warum S1 linear → dB?

Sentinel-1 RTC liefert linear-skalierte Power-Werte (im Bereich ~0.001–1+). Der Dynamikbereich ist **multiplikativ**: Unterschied zwischen Wald und Brachland ist Faktor 10–100, nicht additiv 10–100. Ein neuronales Netz lernt auf linearen Werten schlecht, weil Gradienten durch die extreme Verteilung verzerrt werden.

dB-Transformation `10·log10(x)` macht den Kanal **additiv skaliert**:
- Linearer Faktor 10× → +10 dB
- Linearer Faktor 100× → +20 dB

Danach clip auf `[-30, 0]`, weil Werte außerhalb des Bereichs fast ausschließlich Sensor-Rauschen oder Layover-Artefakte sind (Forstliche Backscatter-Range ist ~[-20, -5] dB).

```python
db = 10.0 * np.log10(np.clip(lin, epsilon, None))   # epsilon gegen log(0)
db = np.clip(db, lo, hi)                            # [-30, 0]
```

**Warum Clip vor dem Cast auf fp16?** Außerhalb des gültigen Bereichs liegende Werte (z.B. NaN oder ±inf durch `log(0)`) sollen nicht in den Cache gelangen. Nach dem Clip ist garantiert, dass jede Zelle in `[-30, 0]` liegt oder NaN ist (aus dem Reprojection-Rand).

---

## 6. Warum multiprocessing, nicht threading?

Rasterio nutzt GDAL, und **GDAL ist nicht thread-safe**. Parallelisierung über Threads im selben Prozess → race conditions auf Dataset-Handles → Abstürze oder stille Datenkorruption.

`multiprocessing.Pool` umgeht das, indem jeder Worker einen eigenen Prozess mit eigenem GDAL-State hat. Kosten: kein geteilter Speicher (kein Problem hier, weil jeder Worker einen eigenen Tile bearbeitet) + Pickling-Overhead beim Job-Dispatch (vernachlässigbar, weil Argumente winzig sind).

**Worker-Anzahl 20** matched die 20 vCPUs im Container. Peak-RAM pro Worker ~2–3 GB (ein S2-Stack in float32 in der Luft), 20×2 GB = 40 GB — passt in 240 GB RAM.

---

## 7. Warum Idempotenz (`skip if meta.json exists`)?

Ein 48h-Hackathon ist interrupt-lastig: Docker-Restart, OOM-Kill, Ctrl-C, "ich starte kurz nochmal neu". Ohne Idempotenz würde jeder Restart die Pipeline bei 0 beginnen und 50 GB neu schreiben.

Die Heuristik ist bewusst einfach: **wenn `meta.json` existiert → überspringen**. `meta.json` wird **als letzte Aktion** geschrieben (nach allen `.npy`-Dateien). Existiert es, ist der Tile garantiert vollständig. Tritt der Prozess mitten im `.npy`-Schreiben ab, fehlt `meta.json` → beim Neustart wird der Tile neu gebaut.

Für erzwungene Neuläufe: `--force`.

---

## 8. Warum eine separate `stats.py`?

Normalisierungs-Stats (Mean/Std pro Band) sind **keine** Eigenschaft eines einzelnen Tiles, sondern der **gesamten Trainingsverteilung**. Sie werden vom Dataloader gebraucht, um beim Training z-normalisierte Inputs zu liefern.

**Warum zwei Pässe, nicht Welford?** Welford-Running-Stats sind numerisch stabiler, aber schwieriger NaN-safe zu halten und verteilt zu aggregieren. Hier sind wir nicht numerisch gefährdet (float64-Akku, ~10¹⁰ Pixel × Werte in sinnvoller Größenordnung → summen bleiben weit unter float64-Overflow). Sum + SumSq pro Band, am Ende einmal dividieren. Simpler Code, kein Präzisionsproblem.

**Warum NaN-safe?** AEF-Warp produziert NaNs an Tile-Rändern (Bereiche außerhalb der Quelle). Die dürfen Mean/Std nicht verzerren.

**Warum nur Train-Split?** Test-Daten sollen unser Normalisierungs-Statistiken nicht sehen — klassischer ML-Hygiene-Punkt. Sonst hätten wir subtiles Data Leakage.

---

## 9. Tradeoffs, die bewusst offen bleiben

### 9.1 Fehlende Monats-Frames
Einige Tiles haben nur 32/72 S1-ascending-Frames (z.B. 18N-Tiles, Equator-Gebiete — andere Orbit-Coverage). Wir **padden nicht** im Cache; wir speichern das, was da ist, und listen die tatsächlichen Monate in `meta.json`. Konsequenz: der Dataloader muss mit variabler Zeitreihen-Länge pro Tile umgehen (oder auf Jahres-Aggregate umsteigen, was für die MLP-Baseline ohnehin der Plan ist — PLAN §4.1).

**Alternative wäre:** NaN-Frames an den Rändern einfügen, um immer `T=72` zu haben. Das vereinfacht den Dataloader, kostet aber Platz und versteckt fehlende Daten hinter NaN. Bewusst nicht gemacht — fehlende Daten sollen explizit im Index stehen.

### 9.2 Kein Forest-Mask, keine Labels
`forest_2020.npy`, `label_consensus.npy`, `label_weight.npy`, `alert_year.npy` fehlen hier absichtlich (PLAN §3.3 + §3.4, Person B). Person B schreibt sie in dasselbe `{tile}/`-Verzeichnis und nutzt `meta.json` als Single Source of Truth fürs Ziel-Grid. Wichtig bei Umsetzung §3.3/§3.4: **Nearest-Neighbor**-Resampling für Labels (nicht Bilinear) — Klassen-IDs dürfen nicht interpoliert werden.

### 9.3 Tile `18NWM_9_4` (1004×2)
Nicht repariert. Das Raster ist real 2 Pixel breit. Jede "Reparatur" wäre Halluzination. Wird beim Patch-Sampling automatisch ignoriert, weil es kaum ganze 128×128-Patches gibt.

### 9.4 Kein Per-Band-Scaling beim fp16-Cast für S2
S2-Reflectance liegt im Bereich 0–15000 (Scale-Factor 10000). In float16 (max ≈ 65504) passt das ohne Normalisierung — aber Bänder-individuelle Skalen bleiben unterschiedlich. Die Normalisierung macht der Dataloader später mit `norm_stats.json`, nicht das Preprocessing. **Warum?** Weil wir Rohwerte im Cache wollen — wenn wir später die Normalisierung ändern (z.B. log-Reflectance, robust-Scaler), müssen wir den Cache nicht neu bauen.

---

## 10. Debugging-Rezepte

**Cache für einen einzelnen Tile neu bauen:**
```bash
python -m src.data.prepare_cache --split train --tiles 18NWG_6_6 --workers 1 --force
```

**Einen Tile-Stack anschauen:**
```python
import numpy as np, json
tile = "18NWG_6_6"
meta = json.load(open(f"/workspace/cache/{tile}/meta.json"))
s2 = np.load(f"/workspace/cache/{tile}/s2.npy", mmap_mode="r")
print(s2.shape, s2.dtype, "crs=", meta["crs"])
# Mittleres NDVI 2022-06: B08 (NIR) = Band 7 (0-indexiert), B04 (Red) = Band 3
import matplotlib.pyplot as plt
frame_idx = meta["months_s2"].index([2022, 6])
nir, red = s2[frame_idx, 7].astype("f4"), s2[frame_idx, 3].astype("f4")
ndvi = (nir - red) / (nir + red + 1e-6)
plt.imshow(ndvi, vmin=-0.2, vmax=0.9); plt.colorbar(); plt.show()
```

**AEF-CRS-Alignment prüfen:**
S2-Pixel (i, j) und AEF-Pixel (i, j) sollen geografisch den selben Fleck abbilden. Eine 2-Panel-Plot-Ansicht bei grober Auflösung sollte dieselben Strukturen zeigen.

**S1-Plausibilität:**
```python
s1 = np.load("/workspace/cache/18NWG_6_6/s1_desc.npy", mmap_mode="r")
print(np.nanmin(s1), np.nanmax(s1))   # muss in [-30, 0] liegen
```

---

## 11. Module-Zusammenfassung

| Datei | Public API | Verantwortung |
|---|---|---|
| `io.py` | `list_tiles`, `s2_paths`, `s1_paths`, `aef_paths` | Filesystem-Layer: finden & sortieren. Kein Rasterio. |
| `reproject.py` | `open_reference_grid`, `read_on_grid` | Rasterio-Wrapper um `warp.reproject`. Einzige Stelle, an der wir GDAL-Mechanik berühren. |
| `prepare_cache.py` | CLI (`python -m src.data.prepare_cache`) | Orchestrator pro Tile + multiprocessing. Schreibt `.npy`s + `meta.json`. |
| `stats.py` | CLI (`python -m src.data.stats`) | Aggregiert `norm_stats.json` aus dem Cache. NaN-safe, zwei Pässe. |
| `../../configs/preprocess.yaml` | — | Default-Werte: Pfade, Worker-Zahl, S1-Clip. |

---

## 12. Wichtigste Lern-Takeaways

1. **Trenne Filesystem-Layer (`io.py`) von IO-Layer (`reproject.py`) von Orchestrierung (`prepare_cache.py`).** Jede Datei eine Verantwortung. Einfacher testbar, einfacher zu parallelisieren.
2. **Idempotenz ist günstig zu implementieren und teuer zu missen.** Ein einziger Flag (`meta.json exists`) spart stundenlanges Neurechnen in einem Hackathon.
3. **fp16 im Cache, fp32 im Rechner.** Speicher halbieren, Präzision beim Verbrauch zurückgewinnen durch expliziten Cast im Dataloader.
4. **Ziel-Grid einmal festlegen und alle anderen draufwarpen** — nicht umgekehrt. Pixel-genaue Alignment-Garantie über die gesamte Pipeline.
5. **Normalisierungs-Stats sind Trainings-Artefakt, keine Cache-Artefakt.** Nicht bei den Rohdaten mischen; sie können sich ändern, ohne dass man 50 GB neu schreibt.
6. **Robustheit durch Inspektion, nicht durch Annahme.** Die max-Area-Heuristik entstand erst, nachdem wir einen echten korrupten Frame gesehen haben. Vorher hätte man sie nicht erfunden.
7. **Multiprocessing > Threading bei C-Libraries mit globalem State** (GDAL, OpenCV, ffmpeg, …).
8. **Metadaten in separater Datei** (`meta.json`) statt in die Arrays gezwungen. Lesbar, diff-bar, niedrige Kopplung.
