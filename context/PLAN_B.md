# PLAN — Person B (Labels / Ground Truth)

> **Kontext:** Spezialisierung von `PLAN.md` §2 / §3.3 / §3.4 für Person B. Person A hat den Raster-Cache gebaut (`src/data/`, siehe `src/data/README.md`). Person B setzt darauf auf und produziert die Label-Artefakte, die C fürs Training und D fürs Post-Processing braucht.

---

## 0. Ziel in einem Satz

Aus drei verrauschten Label-Quellen (RADD, GLAD-L, GLAD-S2) und einer 2020-Forest-Mask **ein harmonisiertes Raster pro Tile** auf dem S2-UTM-Grid erzeugen, das C als Supervision (Konsens + Per-Pixel-Gewicht) und D als Inferenz-Postfilter nutzen kann.

---

## 1. Was von Person A bereits steht (nicht neu bauen!)

- Cache-Layout pro Tile in `/workspace/cache/{tile}/`:
  - `meta.json` — **Single Source of Truth fürs Ziel-Grid** (`crs`, `transform`, `height`, `width`). Immer von hier lesen, niemals eigenes Grid definieren.
  - `s2.npy` `(T, 12, H, W) f16` — für Forest-Mask 2020 (NDVI aus Jahresstack).
  - `s1_asc.npy`, `s1_desc.npy`, `aef.npy` — brauchen wir für B nicht direkt.
- `src/data/io.py` → kein Label-Loader. B schreibt einen eigenen.
- `src/data/reproject.py::read_on_grid(..., resampling=...)` → **wiederverwenden** für Label-Warps (mit `Resampling.nearest`).

**Wichtig:** Person A hat bewusst *keine* Labels im Cache abgelegt und erwartet, dass B in dasselbe `{tile}/`-Verzeichnis schreibt (`src/data/README.md` §9.2).

---

## 2. Outputs (was wir am Ende liefern)

Pro Tile im gleichen `{tile}/` Cache-Verzeichnis, **auf exakt dem Grid aus `meta.json`**:

| Datei | Shape | Dtype | Bedeutung |
|---|---|---|---|
| `forest_2020.npy` | `(H, W)` | `bool` | 1 = 2020 Wald (Rest des Pipelines ignoriert Nicht-Wald). |
| `label_consensus.npy` | `(H, W)` | `uint8` | 0 / 1, harte Target-Klasse (≥2 Quellen sagen "deforested ≥2020"). |
| `label_weight.npy` | `(H, W)` | `float32` | Per-Pixel-Loss-Gewicht ∈ [0, 1]. |
| `alert_year.npy` | `(H, W)` | `uint16` | Frühestes konsistentes Alert-Jahr (0 = kein Alert). Für Bonus "When". |
| `labels_meta.json` | — | — | Provenance: welche Label-Dateien eingegangen sind, Parameter, Zeitstempel. |

Nur für `split=train`. Test-Tiles bekommen nichts (keine Labels verfügbar), aber `forest_2020.npy` **schon** — D braucht es als Post-Filter auch für Test.

---

## 3. Label-Quellen dekodieren (Kernlogik)

Diese Funktionen landen in `src/labels/decoders.py`. Jede Quelle liefert einheitlich `(deforested_after_2020: bool[H,W], alert_year: uint16[H,W])`.

### 3.1 RADD — `radd/radd_{tile}_labels.tif`
Ein Band, Integer-kodiert:
- **Leading-Digit** (tausender Stelle): `2` = low confidence, `3` = high confidence, alles andere = kein Alert.
- **Rest** (`value % 10000`): Tage seit **2014-12-31**.

```python
conf = value // 10000            # 0 / 2 / 3
days = value % 10000
date = datetime(2014, 12, 31) + timedelta(days=int(days))
is_defo = (conf >= 2) & (date >= 2020-01-01)
alert_year = date.year if is_defo else 0
```

### 3.2 GLAD-L — `gladl/gladl_{tile}_alert{YY}.tif` + `alertDate{YY}.tif`
Pro Jahr `YY` (z. B. 20, 21, 22, 23) **zwei** Bänder, die gepaart kommen:
- `alert{YY}` ∈ {0, 2, 3} → 0=kein, 2=probable, 3=confirmed.
- `alertDate{YY}` = Tag-des-Jahres in Jahr 2000+YY.

```python
for YY in [20, 21, 22, 23]:
    a = read(alert_YY)           # (H, W)
    d = read(alertDate_YY)       # DOY in 2000+YY
    mask_defo |= (a >= 2)        # YY ≥ 20 ⇒ automatisch nach 2020
    year_candidate[mask_defo & first_hit] = 2000+YY
```

**Achtung:** alle YY-Files auf Ziel-Grid warpen und **OR-aggregieren**. Frühestes YY gewinnt für `alert_year`.

### 3.3 GLAD-S2 — `glads2/glads2_{tile}_alert.tif` + `alertDate.tif`
Ein Band pro:
- `alert` ∈ {0, 1, 2, 3, 4}. **`1` ist "nur in letzter Observation"** — konservativ als unsicher behandeln und **nicht** als positiv zählen (PLAN §3.4).
- `alertDate` = Tage seit **2019-01-01**.

```python
date = datetime(2019, 1, 1) + timedelta(days=int(days))
is_defo = (alert >= 2) & (date >= 2020-01-01)
alert_year = date.year if is_defo else 0
```

### 3.4 Fehlende Quelle für einen Tile
Nicht jeder Tile hat alle drei Label-Files. Wenn eine Quelle fehlt → Rückgabe `(zeros, zeros)`, aber `n_sources_available` um 1 reduzieren (siehe §4 — Konsens wird dann relativ gebildet).

---

## 4. Harmonisierung (`src/labels/harmonize.py`)

Pro Tile, nach Dekodierung und Warp auf Ziel-Grid:

```python
votes = sum(is_defo_radd, is_defo_gladl, is_defo_glads2)      # 0..3
n_avail = sum(source_available)                                # 1..3

consensus = (votes >= 2).astype(uint8)                         # harte Positive

# Per-Pixel-Gewicht
weight = np.where(votes == n_avail,              1.0,          # alle einig
          np.where(votes == n_avail - 1,          0.8,          # fast alle
          np.where(votes == 1,                    0.3,          # nur eine Quelle = unsicher
          1.0)))                                                # 0 votes = sicher-negativ (im Wald)

# Pixel außerhalb Forest 2020 → irrelevant, im Loss gemasked
weight[~forest_2020] = 0.0

# alert_year: frühestes Jahr über alle positiv-stimmenden Quellen
alert_year = min_nonzero(radd_year, gladl_year, glads2_year)
```

**Wichtige Details:**
- `votes == n_avail - 1` für n_avail=3 → 2 Stimmen, gleich `consensus=1` und Gewicht 0.8. Für n_avail=2 → 1 Stimme, aber das ist dann 0.3 nicht 0.8. Per `if-elif` sauber fallunterscheiden.
- Bei `n_avail=1` (nur eine Quelle vorhanden) hart auf `weight=0.3` deckeln, sogar wenn die eine Quelle positiv → wir können dann nichts konsentieren.
- `weight=1.0` für negative Pixel im Forest (sichere Negative) ist der Hebel gegen Class-Imbalance: das Netz lernt Forest-ohne-Alert als zuverlässige Klasse 0.
- **Ablation-Schalter (CLI-Flag):** `--w-single {0.0, 0.3, 0.5}` — C/D wollen später testen, ob Single-Source-Pixel helfen oder schaden. Konfiguration in `configs/labels.yaml`.

---

## 5. Forest-Mask 2020 (`src/labels/forest_mask.py`)

PLAN §3.3 — **Start mit der NDVI-Variante**, Hansen nur wenn Zeit.

```python
s2 = np.load(f"{cache}/s2.npy", mmap_mode="r")         # (T, 12, H, W) f16
months = meta["months_s2"]                              # [[YYYY, M], ...]

# Monate aus 2020
idx_2020 = [i for i,(y,m) in enumerate(months) if y == 2020]
if not idx_2020:                                        # Fallback
    idx_2020 = [i for i,(y,m) in enumerate(months) if y in (2019, 2020, 2021)]

# NDVI = (B08 - B04) / (B08 + B04).  B04=Index 3, B08=Index 7 (0-basiert).
nir = s2[idx_2020, 7].astype(np.float32)
red = s2[idx_2020, 3].astype(np.float32)
ndvi_median = np.nanmedian((nir - red) / (nir + red + 1e-6), axis=0)

# NDWI gegen Wasser (B03=2, B08=7): NDWI = (B03 - B08)/(B03 + B08)
grn = s2[idx_2020, 2].astype(np.float32)
ndwi_median = np.nanmedian((grn - nir) / (grn + nir + 1e-6), axis=0)

forest_2020 = (ndvi_median > 0.6) & (ndwi_median < 0.0)
```

- `ndvi > 0.6` → dichter Bewuchs. Schwellwert ist Tropen-typisch; ggf. 0.55 ausprobieren.
- `ndwi < 0.0` → kein Wasser.
- **Keine Wolken-Maskierung** in V1. Wir nehmen den Jahres-Median, und der filtert Wolken schon effektiv weg (Wolke = Outlier → Median robust).
- **Sanity-Plot:** B zeigt 3–5 Tiles Forest-Mask neben dem S2-True-Color-Median in einer Notebook-Zelle. Wenn offensichtlich Flusslauf als Wald drin → Threshold nachziehen.

**Ausgabe:** `forest_2020.npy (H, W) bool`.

---

## 6. Modul-Struktur

```
src/labels/
  __init__.py
  io.py            # Finden radd/gladl/glads2-Files pro Tile (analog src/data/io.py)
  decoders.py      # RADD / GLAD-L / GLAD-S2 → (is_defo, alert_year) im Quell-CRS
  warp.py          # Dünne Wrapper: read & warp auf meta.json-Grid, Resampling.nearest
  forest_mask.py   # NDVI+NDWI → bool mask
  harmonize.py     # Votes → consensus + weight + alert_year
  build_labels.py  # CLI-Orchestrator, analog prepare_cache.py, multiprocessing
  README.md        # "Warum"-Doku wie src/data/README.md
configs/
  labels.yaml      # Schwellwerte (NDVI, NDWI, w_single), Quell-Pfade
notebooks/
  eda_labels.ipynb # Sanity-Plots: pro Tile Overlay der 3 Label-Quellen auf S2
```

---

## 7. CLI & Idempotenz (Person A's Muster kopieren)

Ein 1:1-Mirror von `src/data/prepare_cache.py::main`:

```
python -m src.labels.build_labels \
    --config configs/labels.yaml \
    --split train \
    [--tiles 18NWG_6_6 ...] \
    [--workers 20] \
    [--force]
```

- **Idempotent:** wenn `{tile}/labels_meta.json` existiert und `--force` nicht gesetzt → skippen. `labels_meta.json` wird als *allerletzte* Aktion geschrieben (nach allen `.npy`). Gleiches Muster wie A, aus gutem Grund (siehe `src/data/README.md` §7).
- **Multiprocessing:** `mp.Pool(workers)` — GDAL ist nicht thread-safe, gleiche Begründung wie A (README §6).
- **Fehlertoleranz:** eine kaputte Label-Quelle eines Tiles darf den Gesamtjob **nicht** crashen. Worker fängt `Exception`, loggt `FAIL {tile}: ...` und läuft weiter (wie A). Am Ende eine Zusammenfassung, wie viele Tiles fehlen.

---

## 8. Sanity-Checks (vor Übergabe an C)

**Ohne die bricht alles Folgende stillschweigend — bitte wirklich machen.**

1. **Shape-Match:** für jedes Tile `forest_2020.shape == (meta.H, meta.W)`. Trivialer Assert im Build.
2. **Werte-Verteilung:** `label_consensus.mean()` über alle Train-Tiles loggen. Erwartung: 0.1 – 3 %. Wenn 0 → Bug in Dekodierung. Wenn 20 % → Grid-Alignment-Bug.
3. **Gewichts-Histogramm:** `np.bincount` über diskrete Gewichte (erwartet nur {0, 0.3, 0.8, 1.0}). Wenn kontinuierliche Werte drin sind → versehentliches Bilinear-Resampling, fix the `Resampling.nearest`.
4. **Forest-Mask-Plausibilität:** Mittelwert pro Tile in `[0.3, 0.95]`. Wenn unter 0.1 → NDVI-Schwelle zu hart. Wenn über 0.99 → NDWI-Clause fehlt.
5. **Alert-Year-Range:** alle Nicht-Null-Werte in `{2020, 2021, 2022, 2023, 2024}`. Alles darunter = Decoder-Bug.
6. **Visueller Spot-Check:** Für 3 Tiles Overlay (Notebook):
   - Panel 1: S2-True-Color 2020
   - Panel 2: `forest_2020`
   - Panel 3: `consensus=1` in Rot auf Forest-Mask
   - Panel 4: `weight` als Heatmap
   Wenn Panel 3 zufällig aussieht → Grid-Alignment kaputt oder RADD-Dekodierung falsch.

Checks laufen als `python -m src.labels.validate [--tile ...]`, geben Exit-Code ≠ 0 bei Verletzung.

---

## 9. Schnittstellen

### An C (Training)
C liest per Tile:
```python
forest = np.load(f"{cache}/{tile}/forest_2020.npy", mmap_mode="r")      # bool
y      = np.load(f"{cache}/{tile}/label_consensus.npy", mmap_mode="r")  # uint8
w      = np.load(f"{cache}/{tile}/label_weight.npy", mmap_mode="r")     # f32
```
**Garantie von B:** gleiche `(H, W)` wie `s2.npy`, gleiche CRS (aus `meta.json`), `w == 0` exakt dort, wo `forest == 0`.

### An D (Inference)
D liest `forest_2020.npy` für Test-Tiles und maskiert damit Predictions (PLAN §5.7). Also: auch für `split=test` muss B die Forest-Mask bauen — die Label-Dateien gibt's zwar nicht, aber NDVI-Features aus dem Test-Cache schon.

### An E (Jury)
`alert_year.npy` befüllt später im Polygon-GeoJSON das `time_step`-Feld (Bonus). E braucht keine eigene Schnittstelle, nimmt die fertige Datei.

---

## 10. Zeitplan (in PLAN §6 eingehängt)

| PLAN-Phase | Person B |
|---|---|
| **T0 – T4** | Label-Format auf 1 Tile im Notebook verifizieren (nicht auf Doku verlassen — tatsächliche Werte-Verteilung ansehen). `src/labels/decoders.py` mit Parsern, isoliert testbar. Erste Skizze `forest_mask.py` auf 1 Tile. |
| **T4 – T8** | `build_labels.py` läuft für **alle Train-Tiles**. Alle Artefakte im Cache. Sanity-Checks laufen grün. Übergabe an C für Baseline-Training. Forest-Mask auch für Test-Tiles. |
| **T8 – T20** | Disagreement-Ablation: `configs/labels.yaml` zweite Variante mit `w_single=0.0` bauen → D trainiert Ablation. Forest-Mask V2 prüfen (Hansen, falls Zeit). |
| **T20 – T36** | `alert_year` für Bonus verfeinern. Eventuell per-Quelle-Confidence als separate Kanäle für C's V2. |
| **T36 – T44** | QA: Visuals für Jury (E kopiert die Plots ins Deck). |

**Gate @ T8:** C hat Konsens-Labels. Ohne die kann er nicht mal die Baseline trainieren — Priorität 1.

---

## 11. Risiken & Gegenmaßnahmen (B-spezifisch)

| Risiko | Auswirkung | Mitigation |
|---|---|---|
| Label-TIFFs liegen in anderem CRS als S2-Grid | Falsch-Alignment, Modell lernt Quatsch | `Resampling.nearest` + Grid aus `meta.json` + Sanity #6 (visueller Spot-Check) früh. |
| Datum-Decoding falsch (off-by-one, falscher Epoch) | Alle Alerts vor/nach 2020 misklassifiziert | Gegen 1 manuell geprüften Pixel im Notebook validieren. Beispiel aus `challenge.ipynb` wenn vorhanden. |
| `Resampling.bilinear` versehentlich auf Labels | Klassen-IDs interpoliert → Müll | Unit-Test: nach Warp `np.unique(labels) ⊆ {0, 2, 3}` o. ä. |
| Forest-Mask schließt legitimen Wald aus (zu hohe NDVI-Schwelle) | Recall-Deckel bei C | Threshold in `configs/labels.yaml`, leicht nachzuziehen. Ablation: 0.55 vs. 0.6. |
| Ein GLAD-L-Jahr fehlt (nur 3 von 4 YY-Files) | `alert_year` falsch | In `io.py::gladl_paths` tolerant über vorhandene Jahre iterieren, nicht hardkodieren. |
| Tile `18NWM_9_4` (1004×2) — Rand-Tile | Edge-Case bei NDVI-Median (wenig Pixel) | Nicht Sonderlocke. Läuft durch, wenige Pixel, kein Crash. Workflow-Robustheit wie A. |
| Multiprocessing-Speicher | OOM bei 20 Workern × großes Tile | Worker zählt Peak-RAM nur Warp eines Rasters à ~1 GB fp32 — unkritisch. Bei OOM: `workers=10`. |

---

## 12. Was B NICHT tut

- **Kein** eigenes Cache-Grid definieren. Immer aus `meta.json` lesen.
- **Kein** Bilinear-Resampling auf Label-Rastern.
- **Keine** Modellierungs-Entscheidungen (Focal-Loss, Label-Smoothing, Pseudo-Labels) — das ist C's Hoheit. B liefert Rohdaten-Labels plus Gewicht, mehr nicht.
- **Kein** Umschreiben von Person A's Cache oder `meta.json`. Nur *hinzuschreiben* in `{tile}/`.
- **Keine** test-split Labels erfinden. Nur `forest_2020.npy` für Test (für D's Post-Filter).

---

## 13. Definition of Done (B)

1. Für alle Train-Tiles im Cache liegen `forest_2020.npy`, `label_consensus.npy`, `label_weight.npy`, `alert_year.npy`, `labels_meta.json`.
2. Für alle Test-Tiles liegt `forest_2020.npy`.
3. `python -m src.labels.validate` läuft mit Exit 0.
4. `notebooks/eda_labels.ipynb` enthält Spot-Check-Plots für ≥3 Tiles.
5. `src/labels/README.md` erklärt die *Warum*-Entscheidungen (Analog zu `src/data/README.md`), insbesondere: Konsens-Schwelle, Gewichts-Staffel, NDVI-Threshold, Umgang mit fehlenden Quellen.
6. C konnte gegen diese Artefakte eine Baseline trainieren — ohne Rückfragen an B.

---

**Kurzformel:** Parser → Warp-auf-A's-Grid (nearest!) → Votes → Consensus + gestaffeltes Gewicht → Forest-Mask killt irrelevante Pixel → vier `.npy` im selben `{tile}/`-Ordner. Alles andere ist Optimierung.
