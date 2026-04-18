# 48h Plan — osapiens Deforestation Challenge

> **Ziel:** Pixel-genaues Deforestation-Detection-System, das über multimodale Satelliten-Zeitreihen (Sentinel-1, Sentinel-2, AlphaEarth Embeddings) trainiert wird, robust gegenüber verrauschten/konfligierenden Weak Labels (RADD, GLAD-L, GLAD-S2) ist und auf unbekannte Regionen generalisiert. Submission = binärer GeoTIFF pro Test-Tile → GeoJSON via `submission_utils.raster_to_geojson`.
>
> **Team:** 5 Personen, alle Vibe-Coder.
> **Hardware:** 1× AMD MI300X (192 GB VRAM), 20 vCPU, 240 GB RAM, 720 GB Boot + 5 TB Scratch NVMe. PyTorch 2.6.0 / ROCm 7.0.
> **Deadline:** 48 h ab Kick-off.

---

## 0. Leitprinzipien

1. **Baseline zuerst, Optimierung danach.** Innerhalb der ersten 8 h muss ein End-to-End-Pfad (Datenload → Modell → Prediction-TIFF → GeoJSON → Leaderboard-Submission) laufen — egal wie schlecht. Alles danach ist Verbesserung.
2. **Pixel-Klassifikation > Segmentation für Baseline.** Weil Labels verrauscht und sparsam sind, starten wir mit einem pro-Pixel-Zeitreihen-Klassifikator (MLP/1D-CNN über den Zeitachsen-Feature-Stack). Semantische Segmentation (U-Net über Patches) kommt als V2.
3. **Labels sind Signale, nicht Wahrheit.** Wir behandeln RADD/GLAD-L/GLAD-S2 als drei Rauschquellen. Konsens-Label (z. B. ≥2 von 3 stimmen zu) ist unser harter positiver Target. "Nur 1 Quelle" ist "unsicher" und wird entweder ignoriert oder mit reduziertem Gewicht gelernt.
4. **Generalisierung ist ein Kriterium der Jury.** Wir teilen Training nicht zufällig, sondern **nach Tile / geografischer Region** (Held-out Regions als Val-Set) — sonst überschätzen wir Performance massiv.
5. **Eine GPU, ein Trainer.** Nur eine Person drückt "Trainingsjob starten". Andere iterieren auf Features / Dataloader / Eval offline auf CPU.
6. **Jede 6 h ein Sync** (15 min, hart). Status, Blocker, nächste 6 h.

---

## 1. Challenge-Zusammenfassung (Kurz)

- **Eingabe pro Tile** (MGRS `{grid}_{x}_{y}`, z. B. `18NWG_6_6`):
  - **Sentinel-2 L2A:** monatliche `.tif`, 12 Bänder @ 10 m, UTM-projiziert (`{tile}__s2_l2a_{yyyy}_{m}.tif`).
  - **Sentinel-1 RTC:** monatliche `.tif`, 1 Band VV (linear), ascending + descending, UTM (`{tile}__s1_rtc_{yyyy}_{m}_{asc|desc}.tif`).
  - **AlphaEarth Foundations:** jährliche `.tiff`, 64 Embedding-Dimensionen, **EPSG:4326 — reprojizieren!**
- **Labels (nur train):**
  - `radd/radd_{tile}_labels.tif` — Integer kodiert Konfidenz + Tage seit 2014-12-31 (Leading-Digit 2=low, 3=high).
  - `gladl/gladl_{tile}_alert{YY}.tif` + `alertDate{YY}.tif` — 0/2/3 (no/probable/confirmed), Tag-des-Jahres.
  - `glads2/glads2_{tile}_alert.tif` + `alertDate.tif` — 0–4 Konfidenz, Tage seit 2019-01-01.
- **Definition Deforestation:** permanenter Forest→Non-Forest-Übergang **nach 2020** (Area muss 2020 Wald gewesen sein).
- **Submission:** Pro Test-Tile ein binäres Raster (1=deforested, 0=nicht). `raster_to_geojson` → Polygone ≥0.5 ha → EPSG:4326 GeoJSON FeatureCollection → Leaderboard-Upload.
- **Bewertung:** Quantitative Leaderboard-Score + Jury (Design, Reasoning, Label-Noise-Handling, Generalisierung, Interpretierbarkeit).
- **Bonus:** When (Monat/Jahr) der Deforestation, Uncertainty-Schätzung, Visualisierungstool.

---

## 2. Team-Rollen (5 Personen)

| Rolle | Owner | Verantwortung |
|---|---|---|
| **A — Data / Dataloader** | 1 | S3-Download, Reprojection, Cache auf Scratch-NVMe, Patch-Generator, Train/Val-Split nach Region, PyTorch `Dataset` + `DataLoader`. |
| **B — Labels / Ground Truth** | 1 | Alle 3 Label-Quellen dekodieren, harmonisieren auf gemeinsames Raster, Konsens-Mask + Per-Pixel-Konfidenz-Gewicht + `alert_year` bauen. Forest-Mask 2020. |
| **C — Model / Training** | 1 | Architektur, Loss (mit Label-Gewichten), Training-Loop, Checkpointing, Mixed-Precision, Eval. Einzige Person am GPU-Job. |
| **D — Inference / Submission** | 1 | Test-Tile-Inference, Schiebe-Fenster, Nachbearbeitung (Morphologie, Forest-Mask, min-Area), TIFF → GeoJSON, Leaderboard-Upload. Führt auch Ablationen auf Val. |
| **E — Jury / Präsentation / QA** | 1 | Metriken visualisieren (Confusion-Map pro Region, Beispiel-Predictions), Notebook/Slides, Interpretability (saliency, welche Bänder/Monate treiben Prediction), Bonus: When-Deforestation + Uncertainty, Doku & Storyline für Jury. Parallel Sanity-Checks auf Predictions. |

**GPU-Policy:** C hat exklusiven Zugriff während Trainingsruns. D/E inference-en in Lücken (nach Checkpoint-Save). A/B entwickeln CPU-lokal, syncen Code via Git.

---

## 3. Daten-Strategie

### 3.1 Download & Layout
- Sofort bei Kick-off: `make download_data_from_s3`. Ziel-Ordner auf **Scratch-NVMe** (`/scratch/data/makeathon-challenge/`), nicht Boot-Disk.
- Größe vorher prüfen (`aws s3 ls s3://osapiens-terra-challenge/makeathon-challenge --no-sign-request --recursive --human-readable --summarize`).
- Parallel-Download: `boto3` mit Thread-Pool, oder `aws s3 sync` im Container — falls verfügbar — deutlich schneller als serielles `boto3`.

### 3.2 Preprocessing (einmalig, dann cachen)
Alles Pro-Tile in **eine kompakte Form** bringen, die der Dataloader direkt mmap-en kann:

- **Räumliches Ziel-Grid:** UTM des jeweiligen S2-Tiles, 10 m. S1 bereits kompatibel. **AEF + GLAD-S2 + GLAD-L + RADD** alle per `rasterio.warp.reproject` (nearest für Labels, bilinear für kontinuierliche Daten) auf dieses Grid warpen.
- **Cache-Format:** pro Tile eine `.npz` oder Zarr-Store mit:
  - `s2` : `(T_months, 12, H, W)` float16 (oder uint16 + Skalierung).
  - `s1_asc`, `s1_desc` : `(T_months, 1, H, W)` float16 (in dB umgerechnet, clippen auf [-30, 0]).
  - `aef` : `(Y_years, 64, H, W)` float16.
  - `forest_2020` : `(H, W)` bool — Forest-Mask für 2020 (siehe 3.3).
  - `label_consensus` : `(H, W)` uint8 — 0 / 1 / "weich".
  - `label_weight` : `(H, W)` float32 — Per-Pixel-Konfidenz.
  - `alert_year` : `(H, W)` uint16 — frühestes konsistentes Alert-Jahr (für Bonus).
- Warum: der Dataloader zieht später nur noch Patches aus mmap-en Arrays, kein Rasterio-Overhead im Hot-Loop. Auf MI300X sonst I/O-bound.
- Normalisierungs-Stats (Mean/Std pro Band) einmalig auf Train berechnen und als `.json` speichern.

### 3.3 Forest-Mask 2020 — Hansen GFC (implementiert)

**Status:** ✅ `src/data/forest_mask.py` + `src/patch_forest_masks.py` fertig. Alle Zarr-Caches gepatcht.

**Definition:**
```
forest_2020 = (Hansen treecover2000 ≥ 30 %) AND (lossyear == 0)
```
- `treecover2000`: Baumbedeckung im Jahr 2000 (0–100 %, 30 m → reprojiziert auf 10 m UTM-Grid)
- `lossyear == 0`: kein aufgezeichneter Waldverlust zwischen 2001 und 2020 (Hansen GFC v1.8 geht bis Jahr 20 = 2020)

**Warum nicht NDVI-Threshold:**

| | NDVI-Threshold (alt) | Hansen GFC (neu) |
|---|---|---|
| Datenquelle | Sentinel-2 — **dieselben Bänder die als Features ins Modell gehen** | Unabhängige externe Referenz |
| Kreiszirkularität | Modell sieht Forest-Mask-Signal doppelt (NDVI als Feature + als Mask-Basis) | Keine — orthogonale Quelle |
| Falsch-Positiv-Risiko | Savanne, Crops, Buschland mit NDVI > 0.6 landet in der Maske | Nur struktureller Wald (≥ 30 % Kronenschluss) |
| Temporal-Logik | 2020-Median kann bereits leicht degradierte Pixel als Wald sehen | `lossyear == 0` = exakt kein Verlust bis Ende 2020 |
| Industriestandard | Projektinterner Proxy | Offizieller GFW-Standard, den Jury kennt |

**Konkreter Mehrwert im System:**

1. **Negatives werden zuverlässiger.** Alle Negatives (label=0) werden aus `forest_2020 == True`-Pixeln gezogen. Mit NDVI-Mask landen Savanne/Crops als Negatives im Training — das Modell lernt "nicht-Wald → kein Alarm", nicht "Wald → kein Alarm".

2. **Kein Signal-Leakage.** NDVI und NBR-Drop sind unsere stärksten Delta-Features. Wenn die Forest-Mask aus demselben NDVI berechnet wird, maskiert sie genau die Pixel raus, auf denen das Modell seinen stärksten Signal-Drop sehen würde — self-defeating.

3. **Harter Post-Processing-Filter.** `prediction &= forest_2020_mask` in `postprocess.py` schneidet False Positives auf Stadträndern, Agrarland, Wasserflächen ab — unabhängig vom Modell-Score.

4. **persistence_score / Label-Refinement nutzt die Maske.** `compute_persistence_targets()` in `feature_bank.py` berechnet `refined_positive &= forest_mask` — d.h. die Label-Qualitäts-Schicht baut direkt auf der Mask-Qualität auf.

**Laufzeit:** Download via `/vsicurl/` (kein Full-Download), ~5–10 s pro Tile. Patch-Script überschreibt nur `forest_2020` in bestehenden Zarr-Stores — kein Re-Warping von S2/S1/AEF nötig.

### 3.4 Label-Harmonisierung (B's Kern-Aufgabe)
Alle drei Quellen werden zu zwei Rastern pro Tile reduziert:

- **`deforested_after_2020`** (binär, uint8):
  - RADD: Leading-Digit ∈ {2,3} UND `(days_since_2014-12-31) → Datum ≥ 2020-01-01` → 1.
  - GLAD-L: `alertYY ∈ {2,3}` UND `YY ≥ 20` → 1.
  - GLAD-S2: `alert ∈ {2,3,4}` (die `1` = "nur in letzter Observation" konservativ als unsicher behandeln) UND `alertDate → Datum ≥ 2020-01-01` → 1.
- **`consensus`** und **`weight`**:
  - `n_votes` = Summe (je Quelle 0/1 wie oben).
  - `consensus` = 1 wenn `n_votes ≥ 2`, 0 sonst.
  - `weight` = Mapping: `n_votes=3` → 1.0; `n_votes=2` → 0.8; `n_votes=1` → 0.3 (oder 0 — Ablation); `n_votes=0` (inkl. hard-neg Wald 2020 ohne Alert) → 1.0 als sicher-negativ.
  - Pixel außerhalb Forest-2020 bekommen `weight=0` (im Loss gemasked), weil sie per Definition irrelevant sind.
- **`alert_year`** für Bonus: frühestes Jahr über alle Quellen, die ein Alert liefern.

Diese Harmonisierung ist **das wichtigste einzelne Artefakt**. Wenn sie schlecht ist, ist alles andere Müll.

### 3.5 Splits (keine leaky Splits!)
- `metadata/train_tiles.geojson` enthält alle Train-Tiles.
- Gruppieren nach **MGRS-Grid** (Präfix vor `_x_y`, z. B. `18NWG`) oder gröber nach kontinentalem Cluster aus Tile-Centroid-Lat/Lon (k-means, k=5).
- **4 Cluster → Train, 1 Cluster → Val**, rotierend falls Zeit. Innerhalb Train/Val KEINE Tiles mischen.
- Dieses Split ist das einzige, das unsere Leaderboard-Generalisierung ehrlich approximiert.

---

## 4. Modell-Strategie

### 4.1 Baseline — Per-Pixel-MLP (Ziel: Stunde 8)
Pro Pixel ein Feature-Vektor über die Zeit → einfacher MLP-Klassifikator.

**Feature-Konstruktion pro Pixel (`x ∈ ℝ^D`):**
- S2: pro Band pro Jahr {min, max, mean, std, slope-linear} über die 12 Monate → 12 Bänder × ~3 Jahre × 5 Stats (~180).
- Abgeleitete Indizes: NDVI, NBR, EVI, NDWI — Mean + Annual-Delta (2020→2023).
- S1 VV (dB): pro Jahr {mean, std, min} → 3×3=9.
- AEF: direkt 64 Dims × Anzahl Jahre (z. B. 4 Jahre → 256) — oder Differenzen (`2023 − 2020`) × 64 → 64.
- **Schlüssel-Feature:** Delta-Features **zwischen 2020 und letztem verfügbaren Jahr** (NDVI-Drop, NBR-Drop, AEF-Δ). Deforestation = sprunghafte Änderung, genau das wollen wir detektieren.

**Modell:** MLP `[D → 512 → 256 → 128 → 1]`, GELU, Dropout 0.2, BatchNorm, Sigmoid-Output.

**Loss:** Weighted BCE (per-pixel `weight` aus 3.4). Optional Focal Loss (`γ=2`), weil Klassen stark imbalanced.

**Warum Baseline so:** Trainiert in Minuten auf MI300X, lässt sich über ganze Tiles sofort inference-en, ist Debugging-freundlich und liefert ein ehrliches Unterboden-Score.

### 4.2 V2 — 1D-Temporal-CNN / kleines Transformer
Dieselben Features **nicht aggregiert**, sondern als Zeitreihe:
- Input: `(T_months, C)` mit `C` = S2-Bänder + S1 + NDVI + NBR.
- 4 × Conv1D (kernel=3) + Global-Pool → MLP-Head.
- ODER: 2-Layer Transformer Encoder (`d_model=128`, 4 heads) mit Positional-Encoding auf Monatsindex, + `[CLS]`-Pool.
- AEF-Jahres-Embeddings als separater Token / konkateniert an CLS.
- Training: selber Loss, aber Mini-Batches von Pixel-Samples (z. B. 16384 Pixel) aus Patches gezogen.

### 4.3 V3 (falls Zeit) — Patch-basierte U-Net-Segmentation
- Input: stackte Zeit-Features als Channels (`T*C`) auf 128×128-Patches.
- U-Net mit ResNet-18-Backbone (torchvision/SMP-ähnlich, ROCm-kompatibel).
- Loss: Weighted Dice + Weighted BCE (Pixelgewicht aus 3.4).
- Vorteil: räumlicher Kontext → besser gegen Einzel-Pixel-Noise.
- **Risk:** Labels sind an den Rändern diffus, U-Net kann overfitten. Nur angehen, wenn V2 solide.

### 4.4 V4 Bonus — Ensemble + Kalibrierung
- Mittelwert der Sigmoid-Outputs von Baseline + V2 (+ ggf. V3).
- Temperature-Scaling auf Val für Uncertainty.
- Threshold auf Val so wählen, dass Leaderboard-Metrik optimiert ist — **nicht** 0.5 blind verwenden.

---

## 5. Training-Pipeline im Detail (C's Kernverantwortung)

### 5.1 Setup
- **Device:** `torch.device("cuda")` — ROCm exponiert MI300X als `cuda`. Sanity: `torch.cuda.get_device_name(0)` muss MI300X zeigen.
- **AMP:** `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)`. MI300X liebt bf16, besser als fp16 (kein Loss-Scaler nötig).
- **Batchgrößen:** 192 GB VRAM → wir sind effektiv nie VRAM-limitiert. Optimieren auf I/O-Durchsatz, nicht Batchgröße.
- **DataLoader:** `num_workers=16`, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=4`. Daten liegen auf Scratch-NVMe (5 TB).
- **Determinismus:** Seed setzen, aber `torch.backends.cudnn.benchmark = True`.
- **Reproducibility:** Jeder Run bekommt `runs/{timestamp}_{commit}_{tag}/` mit Config-YAML, Logs, Checkpoints.

### 5.2 Dataset-Klassen

```
PixelDataset       # zieht N zufällige Pixel pro __getitem__, gibt (features, label, weight)
                   # → für Baseline & V2
PatchDataset       # zieht 128x128-Patches, gibt (C_stack, mask, weight)
                   # → für V3 U-Net
```

- **Sampling-Strategie (wichtig):** Deforestation-Pixel sind selten (~<1 %). Weighted Random Sampling mit 50 / 50 pos/neg. Negative kommen aus Forest-2020-Pixeln OHNE jeglichen Alert.
- **Augmentation (V2/V3):** Random-Flip, Random-90°-Rotate, zufälliger Monats-Shift (±1 Monat) für Robustheit. Keine Farb-Augmentation — Spektralwerte sind das Signal.

### 5.3 Training-Loop (Pseudo)
```
for epoch in range(E):
    for batch in loader:
        x, y, w = to_device(batch)
        with autocast(bfloat16):
            logits = model(x)
            loss_per = bce_with_logits(logits, y, reduction="none")
            loss = (loss_per * w).sum() / w.sum().clamp_min(1)
        loss.backward()
        optim.step(); optim.zero_grad()
    eval_on_val_region()     # Precision, Recall, F1, PR-AUC, IoU
    save_checkpoint_if_best()
```

- Optimizer: **AdamW**, lr=3e-4, weight_decay=1e-4, cosine schedule, 1 Epoch warmup.
- Early stopping nach 5 Epochs ohne Val-F1-Verbesserung.
- Log: loss, per-class metrics, **pro Region** (getrennt) — wenn eine Region deutlich schlechter läuft, wissen wir wo wir hinschauen müssen.

### 5.4 Evaluation — was wir tracken
Pro Val-Region UND global:
- **F1** (primär — balanciert Precision/Recall, robust gegen Imbalance).
- **PR-AUC** (threshold-frei).
- **IoU** der binarisierten Prediction gegen Konsens-Label.
- **Pixel-Accuracy pro Klasse** (nicht blind Gesamtaccuracy — ist bei Imbalance irreführend).
- **Polygon-F1 nach** `raster_to_geojson` (Übereinstimmung auf Polygon-Ebene mit IoU≥0.5) — das ist näher an Leaderboard-Metrik (wir kennen die exakte nicht, nehmen aber an, sie vergleicht Polygone).

### 5.5 Loss-Ablationen (in dieser Reihenfolge, falls Zeit)
1. BCE ungewichtet (Sanity).
2. BCE mit `weight` (3.4).
3. BCE + Focal (γ=2).
4. Dice + BCE (nur V3).
5. Label-Smoothing für Konsens-1 (0.05) — rechnet "auch Konsens kann falsch sein" ein.

### 5.6 Rauschbehandlung — explizite Tricks
- **Disagreement-Mask:** Pixel, wo genau 1 Quelle positiv und 2 negativ sind → `weight` halbieren. Das ist der zentrale "weak label"-Hebel.
- **Co-Teaching (Stretch):** zwei parallele Modelle, wähle pro Batch nur die Samples mit kleinstem Loss im jeweils anderen Modell zum Update. Robust gegen Label-Noise. Nur versuchen, wenn V2 steht und noch ≥12 h übrig.
- **Self-Training (Stretch):** nach erster guter Epoche Predictions auf Unlabeled-Regions zurücklesen und Pseudo-Labels mit hoher Konfidenz (>0.9) dem Trainingsset zuführen.

### 5.7 Inference über Test-Tiles (D's Aufgabe)
```
for tile in test_tiles:
    x = build_features(tile)              # gleicher Stack wie Train
    logits = model(x)                     # Sliding-Window oder pro Pixel
    prob = sigmoid(logits)
    pred = (prob > thr) & forest_2020_mask(tile)
    pred = morph_cleanup(pred)            # opening (3x3) um Einzel-Noise-Pixel zu killen
    write_geotiff(pred, tile_transform, tile_crs)
    raster_to_geojson(path, output=submission/{tile}.geojson, min_area_ha=0.5)
merge_all_geojsons() → leaderboard
```
- Threshold `thr` wird auf Val-Tiles per-F1 optimiert.
- `morph_cleanup`: Erosion(1)→Dilation(1) oder scikit-image `remove_small_objects(min_size=50 px ≈ 0.5 ha)`.

---

## 6. 48-Stunden-Zeitplan

### T0–T4 (0–4 h) — Setup & Daten liegen lokal
- **A:** S3-Download startet parallel zu allem (lang). Verify `metadata/train_tiles.geojson` + `test_tiles.geojson` lesbar.
- **B:** Label-Encodings prüfen auf 1 Beispiel-Tile (Notebook-Cells nachvollziehen). Schreibt `build_labels.py` mit den 3 Parsern.
- **C:** Env hochziehen, `requirements.txt` installieren + `torch==2.6.0+rocm7`. Smoke-Test: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`. Schreibt Skeleton `train.py`.
- **D:** `submission_utils.raster_to_geojson` auf dem im Notebook gezeigten RADD-Beispiel ausführen — **dry-run Leaderboard-Submission** mit diesem Dummy, um Upload-Format zu verifizieren. Wenn Upload funktioniert: Pipeline-Ende bereits bewiesen.
- **E:** Liest Task & Notebook, macht Risiken-Liste, richtet `runs/`-Ordner, Slides-Deck-Gerüst.

**Gate @ T4:** Daten sind heruntergeladen, ein Dummy-Submission wurde akzeptiert.

### T4–T8 (4–8 h) — Baseline steht
- **A:** Preprocessing-Script `prepare_cache.py` für 1 Tile → funktioniert → dann in parallel pool für alle Train-Tiles (`multiprocessing.Pool(20)`).
- **B:** `label_consensus` + `weight` + `forest_2020` für alle Tiles.
- **C:** `PixelDataset` + Baseline-MLP. Trainieren auf 1 Region → Val auf 1 held-out Region. Erstes Scoring.
- **D:** Inference-Script läuft über **alle Test-Tiles**, erzeugt GeoTIFFs, `raster_to_geojson` → Leaderboard.
- **E:** Erste Val-Metriken visualisieren (Confusion-Map, Prediction-Overlay-PNG pro Val-Tile).

**Gate @ T8:** Baseline-Submission ist auf dem Leaderboard. Wir haben eine Zahl.

### T8–T20 (8–20 h — inkl. Schlaf-Rotation!) — V2 + Ablationen
- **C:** V2 (1D-CNN oder Transformer über Zeitreihe). 2–3 Trainings-Runs, jede mit Checkpoint.
- **A:** `PatchDataset` als Vorbereitung für V3.
- **B:** Label-Disagreement-Maske als separate Ablation bereitstellen.
- **D:** Threshold-Sweep auf Val. Morphologie-Nachbearbeitung. Neue Submission nach jedem besseren Checkpoint.
- **E:** Interpretability — welche Features treiben Prediction? (Permutation-Importance auf Val). Slide-Entwurf Kapitel "Approach".

**Schlaf:** mindestens 5 h gestaffelt. Niemand macht 48 h durch — Fehler werden teuer.

**Gate @ T20:** V2 schlägt Baseline auf Val-F1 um ≥5 Punkte. Sonst: zurück zu Baseline optimieren, V2 verwerfen.

### T20–T36 (20–36 h) — V3 U-Net oder Vertiefung
- **Entscheidung @ T20:** V3 angehen oder V2 ausreizen?
  - V3 nur, wenn V2 stabil ist und Val-Performance klar limitiert durch fehlenden räumlichen Kontext (viele Einzelpixel-FPs).
  - Sonst: V2 mit Co-Teaching / Focal / Pseudo-Labels ausbauen.
- **C:** Trainiert gewählten Pfad. 2 große Runs max.
- **D:** Ensembling Baseline + V2 (+ V3). Uncertainty-Map (Std der Ensemble-Sigmoids) — Bonus.
- **E:** Bonus-Task "When": Zweite Output-Head für `alert_year` (Regression oder Klassifikation Jahr 2021/2022/2023). Beim Polygon-Export `time_step` im GeoJSON füllen (Feld existiert bereits).
- **A/B:** QA — Test-Tile-Predictions visuell prüfen, Outlier-Tiles identifizieren.

**Gate @ T36:** Beste Submission steht. Jeder weitere Run muss Val-F1 nachweislich verbessern.

### T36–T44 (36–44 h) — Einfrieren & Präsentation
- **Kein neues Training ab T40.** Wir frieren das Modell ein.
- **D:** Finale Inference-Runs, finale Submission auf Leaderboard.
- **E:** Slides fertig. Storyline:
  1. Problem & Impact (EUDR, Coffee, Generalisierung).
  2. Daten & Label-Noise — was haben wir gesehen.
  3. Unser Ansatz — Konsens-Labels + Temporale Modellierung + Forest-Mask.
  4. Ergebnisse (Val-Metriken pro Region, Leaderboard-Score).
  5. Interpretability (Feature-Importance, Uncertainty-Maps).
  6. Bonus: When-Deforestation + Viz-Tool.
  7. Generalisierung: warum wir glauben, dass das überträgt.
  8. Limits & nächste Schritte.
- **C:** Präsentations-Backup: Notebook (`results.ipynb`) mit allen Plots reproduzierbar.

### T44–T48 (44–48 h) — Puffer
- Bugs, Hot-Fixes, letzte Visualisierung, Generalprobe Pitch.
- **Nicht trainieren.** Regression-Risiko > erwarteter Gain.

---

## 7. Konkrete Datei-Struktur im Repo (anlegen in Stunde 0)

```
src/
  data/
    prepare_cache.py        # S3-Raws → per-Tile .npz Cache
    labels.py               # Parser RADD/GLAD-L/GLAD-S2 → consensus/weight
    forest_mask.py          # NDVI-basierte 2020-Forest-Mask
    dataset.py              # PixelDataset, PatchDataset
    splits.py               # Region-basierte Train/Val-Splits
  models/
    mlp_baseline.py
    temporal_cnn.py
    unet.py                 # V3, optional
  train.py
  eval.py
  inference.py              # Test-Tile → binäres TIFF
  submit.py                 # TIFF → GeoJSON via submission_utils
configs/
  baseline.yaml
  v2_transformer.yaml
  v3_unet.yaml
runs/                       # Trainingsruns (gitignore)
submission/                 # Finale GeoJSONs
notebooks/
  eda_labels.ipynb          # B's Arbeitsbuch
  results.ipynb             # E's Präsentations-Notebook
```

---

## 8. Risiken & Mitigation

| Risiko | Auswirkung | Mitigation |
|---|---|---|
| Daten-Download dauert länger als 4 h | Verzögert alles | Parallel-Download, zuerst nur 1–2 Test-Tiles ziehen, damit Code-Entwicklung parallel läuft |
| AEF-Reprojection langsam / hässlich (EPSG:4326→UTM) | Blockiert Training | Mit `resampling=Resampling.bilinear` und `nodata=nan` testen; Cache auf Scratch persistieren |
| Labels widersprechen sich massiv | Modell lernt Noise | Konsens-Mask + Disagreement-Maske als Weight, nicht harte 0/1 |
| Val-F1 bleibt zufällig | Ansatz falsch | Sanity-Checks: Forest-Mask überlagert Prediction, S2-Bänder visuell plausibel |
| ROCm-Kompatibilitätsproblem mit einer Library | Trainingsausfall | Vor V3 testen, ob `segmentation_models_pytorch` auf ROCm importiert; sonst simple U-Net von Hand |
| Submission-Format falsch | Leaderboard rejected | Dummy-Submission in den ersten 4 h durchspielen |
| Ein Teammitglied blockiert (Schlaf/krank) | Engpass | Klar dokumentierte Schnittstellen (Dataset-Shapes, Label-Format). Rollen können 1 Shift gegenseitig übernehmen |
| Overfitting auf Val-Region | Leaderboard crash | Val-Split rotieren, Region-K-Fold als schnelle Prüfung |

---

## 9. Was wir NICHT tun

- **Keine** zufälligen Pixel-Splits innerhalb Tiles. Leaky und schönt Val.
- **Kein** naives MSE oder Crossentropy ohne Gewichtung. Klassen sind viel zu imbalanced.
- **Kein** DistributedDataParallel / Multi-GPU-Code — wir haben 1 GPU.
- **Kein** großes Foundation-Model von Hugging Face fine-tunen in Stunde 1. AEF-Embeddings sind bereits die "Foundation"-Zutat.
- **Kein** Leaderboard-only Fokus. Jury-Score ist ~50 % Darstellung.
- **Kein** Code-Bloat "für die Zukunft". Alles für 48 h, schlank.

---

## 10. Definition of Done

Am Ende der 48 h liefern wir:

1. **Leaderboard-Submission** — beste Version, alle Test-Tiles abgedeckt, GeoJSON korrekt in EPSG:4326, Polygone ≥0.5 ha.
2. **Reproduzierbarer Code** im Repo — `python src/inference.py --checkpoint runs/best/best.pt --out submission/` funktioniert.
3. **`results.ipynb`** — Val-Metriken pro Region, Beispiel-Predictions über Tiles, Interpretability-Plots, Uncertainty-Map.
4. **Präsentation (≤10 Slides)** für Jury.
5. **Bonus:** When-Prediction im GeoJSON `time_step`-Feld + Confidence-Map.

---

## 11. Quick-Reference — Submission-Format (aus `submission_utils.py`)

- Input: ein GeoTIFF pro Test-Tile, single-band, uint8, `1`=deforested, `0`=nicht, **gleiches CRS wie Tile**.
- `raster_to_geojson(tif, out_geojson, min_area_ha=0.5)`:
  - Vektorisiert Pixel==1 zu Polygonen.
  - Reprojektiert auf EPSG:4326.
  - Filtert Polygone < `min_area_ha` (metrisch korrekt über UTM-Zwischenprojektion).
  - Schreibt FeatureCollection mit `time_step`-Feld (initial `None` — für Bonus mit Jahr füllen).
- Wenn Output leer (kein Pixel==1 oder alle Polygone zu klein): `ValueError`. **Vor Submission prüfen.**

---

## 12. Kick-off-Checklist (erste 30 Minuten)

- [ ] Git-Branch pro Rolle (`dev/data`, `dev/labels`, `dev/model`, `dev/inference`, `dev/viz`).
- [ ] `main` geschützt, Merges nur per PR mit ≥1 Review (leicht, schnell).
- [ ] `make download_data_from_s3` gestartet auf Scratch.
- [ ] Alle lesen `Task_Description.md` + Notebook-Sections 4+5 (Datenverständnis!).
- [ ] 6-h-Sync-Slot im Kalender (T6, T12, T18, …).
- [ ] Dieses Dokument als `PLAN.md` im Repo — genau was du gerade liest.

---

**Let's ship. Baseline first, hero later.**
