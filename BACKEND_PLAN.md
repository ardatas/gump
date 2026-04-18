# BACKEND_PLAN.md

## Ziel

Dieses Dokument beschreibt die Backend-Architektur und die konkrete Umsetzungsreihenfolge für die GUMP-Website und die bestehende AI-Train/Test-Pipeline auf **einem einzigen Server**.

Die Spezifikation ist so formuliert, dass ein Coding-LLM sie direkt in Code umsetzen kann.

Der Backend-Plan basiert auf folgenden Randbedingungen:

- **Ein Server** für Training, Inferenz, API und Website.
- **Code liegt in** `/shared-docker`.
- **Laufzeitdaten liegen in** `/workspace`.
- **Kein zusätzlicher `/workspace/gump`-Ordner**.
- **Frontend liegt in** `/shared-docker/www`.
- **Pipeline-Code liegt in** `/shared-docker/src`.
- Es soll **keine direkte Kopplung** zwischen Frontend und Trainingscode geben.
- Lange Operationen laufen **asynchron** über Jobs.
- GPU-Zugriff wird **zentral kontrolliert**.
- Die Website soll mittelfristig sowohl **fertige Run-Ergebnisse visualisieren** als auch **gezielte Tile-Inferenz anstoßen** können.

---

## 1. Architekturprinzipien

### 1.1 Grundsatz

Ein Server ist zulässig. Nicht zulässig ist ein ungeordneter Mischbetrieb.

Deshalb wird die Anwendung logisch in vier Schichten getrennt:

1. **Frontend**
   - Next.js App in `/shared-docker/www`
   - liefert UI und spricht nur mit der API

2. **Backend API**
   - FastAPI-App in `/shared-docker/src/api`
   - liefert Metadaten, Run-Listen, Tile-Infos, Job-Status
   - erstellt Jobs, führt aber keine schweren Modellberechnungen im Request-Thread aus

3. **Worker**
   - separater Python-Prozess in `/shared-docker/src/worker`
   - führt Jobs aus
   - lädt Modelle
   - ruft Pipeline-Funktionen auf
   - schreibt Artefakte in `/workspace/artifacts`

4. **State / Storage Layer**
   - Dateisystem unter `/workspace/*`
   - optional SQLite in `/workspace/state/app.db`
   - Jobs, Artefakte, Logs, Cache, Locks

### 1.2 Zentrale Regeln

- Das Frontend darf **niemals direkt** auf Trainingscode oder Modellgewichte zugreifen.
- Die API darf **niemals direkt** lange GPU-Jobs synchron im HTTP-Request ausführen.
- Der Worker ist der **einzige Prozess**, der Modellgewichte lädt und Inferenz/Training ausführt.
- Große Artefakte werden **als Dateien** gespeichert und vom Backend nur referenziert.
- Der Datenaustausch zwischen Komponenten erfolgt über:
  - JSON-Metadaten
  - standardisierte Dateipfade
  - Jobs
- `/workspace/runs` ist die **Quelle der Wahrheit für Runs und Checkpoints**.
- `/workspace/artifacts` ist die **Quelle der Wahrheit für webfähige Outputs**.

---

## 2. Zielstruktur im Dateisystem

### 2.1 Code-Verzeichnis

Der bestehende Code bleibt in:

```text
/shared-docker/
  src/
  www/
  tests/
  configs/
  context/
  presentation/
```

### 2.2 Laufzeitverzeichnisse

Es wird folgende Struktur unter `/workspace` verwendet:

```text
/workspace/
  runs/
  artifacts/
  jobs/
    queued/
    running/
    done/
    failed/
  logs/
    training/
  state/
  cache/
```

### 2.3 Bedeutung der Verzeichnisse

#### `/workspace/runs`
Enthält Trainingsruns, Checkpoints und Trainingsmetriken.

Beispiel:

```text
/workspace/runs/
  20260418-120805-baseline/
    config.json
    history.json
    metrics.json
    best.pt
    last.pt
    manifest.json
```

#### `/workspace/artifacts`
Enthält webfähige Artefakte pro Run und Tile.

Beispiel:

```text
/workspace/artifacts/
  20260418-120805-baseline/
    18NWG_6_6/
      summary.json
      pred.geojson
      overlay.png
      prob.tif
      uncertainty.tif
```

#### `/workspace/jobs`
Dateibasierte Job-Queue.

```text
/workspace/jobs/
  queued/
  running/
  done/
  failed/
```

Jeder Job ist eine JSON-Datei.

#### `/workspace/logs`
Logs für API, Worker und Training.

Beispiel:

```text
/workspace/logs/
  api.log
  worker.log
  training/
    20260418-120805-baseline.log
```

#### `/workspace/state`
Persistenter Runtime-State.

Beispiel:

```text
/workspace/state/
  app.db
  gpu.lock
  registry_cache.json
```

#### `/workspace/cache`
Vorbereitete Features, Zwischenstände, tile-basierte Caches.

---

## 3. Backend-Zielbild

### 3.1 Request-Flow

#### Lesender Flow
1. Frontend ruft API auf.
2. API liest Metadaten aus `/workspace/runs` und `/workspace/artifacts`.
3. API gibt normalisierte JSON-Daten zurück.
4. Frontend rendert Run-Liste, Metriken, Tiles und Artefakte.

#### Schreibender Flow
1. Frontend erstellt über API einen Job.
2. API schreibt Job-Datei nach `/workspace/jobs/queued`.
3. Worker pollt die Queue.
4. Worker verschiebt den Job nach `running`.
5. Worker führt Pipeline-Logik aus.
6. Worker schreibt Artefakte nach `/workspace/artifacts/...`.
7. Worker schreibt Abschlussstatus nach `/workspace/jobs/done` oder `/workspace/jobs/failed`.
8. Frontend pollt den Jobstatus über API.

### 3.2 GPU-Regel

- Es darf immer nur **ein GPU-Job gleichzeitig** laufen.
- GPU-intensive Jobs sind mindestens:
  - Training
  - Modell-Inferenz
  - Uncertainty-Berechnung, falls GPU-basiert
- Der Worker nutzt einen zentralen Lock unter `/workspace/state/gpu.lock`.
- Wenn der Lock existiert, wird kein weiterer GPU-Job gestartet.
- Neue Jobs bleiben in der Queue oder werden mit Status `queued` belassen.

---

## 4. Implementierungsumfang

Das Backend soll in zwei Ausbaustufen entwickelt werden.

### Phase A — Read-Only Viewer
Ziel: Website kann bestehende Runs und Artefakte anzeigen.

Pflichtumfang:
- API für Runs
- API für Tile-Artefakte
- Registry/Scanner für bestehende Run-Ordner
- Normalisierte JSON-Antworten
- Nginx oder API kann statische Artefakte aus `/workspace/artifacts` ausliefern

### Phase B — Interaktive Jobs
Ziel: Website kann kontrolliert Inferenzjobs erstellen.

Pflichtumfang:
- Job-Erstellung
- Job-Status
- Worker
- GPU-Lock
- Tile-Inferenz-Job
- Artefakt-Export

### Phase C — Optional
- Threshold-Recompute auf Basis vorhandener Probability-Raster
- Overlay-Render on demand
- Run-Refresh/Rescan
- Submission-Export-Job

---

## 5. Verwendete Technologien

### 5.1 Backend
- Python 3.10+
- FastAPI
- Uvicorn
- Pydantic
- Standardbibliothek für file-based queue
- optional SQLite (`sqlite3` aus stdlib genügt)

### 5.2 Frontend
- Next.js in `/shared-docker/www`
- Frontend spricht ausschließlich die API

### 5.3 Queue
Für die erste Implementierung **keine externe Queue**.
Es wird eine dateibasierte Queue verwendet.

Begründung:
- ein Server
- schnell umsetzbar
- keine zusätzliche Infrastruktur nötig
- ausreichend für Hackathon- und Demo-Betrieb

---

## 6. Neue Backend-Ordner im Repo

Es sind folgende neuen Python-Module anzulegen:

```text
/shared-docker/src/
  api/
    __init__.py
    main.py
    config.py
    schemas.py
    dependencies.py
    routes/
      __init__.py
      health.py
      runs.py
      tiles.py
      jobs.py
  worker/
    __init__.py
    main.py
    executor.py
    gpu_lock.py
    file_queue.py
  registry/
    __init__.py
    scan_runs.py
    manifests.py
    paths.py
  services/
    __init__.py
    run_service.py
    tile_service.py
    artifact_service.py
    job_service.py
```

Optional zusätzlich:

```text
/shared-docker/src/common/
  __init__.py
  paths.py
  logging.py
  utils.py
```

---

## 7. Pfad- und Namenskonventionen

Alle Pfade müssen zentral definiert sein. Keine hartkodierten String-Literale quer im Code.

### 7.1 Zentrale Pfade

In `src/common/paths.py` oder `src/registry/paths.py` definieren:

- `WORKSPACE_ROOT = Path("/workspace")`
- `RUNS_DIR = WORKSPACE_ROOT / "runs"`
- `ARTIFACTS_DIR = WORKSPACE_ROOT / "artifacts"`
- `JOBS_DIR = WORKSPACE_ROOT / "jobs"`
- `JOBS_QUEUED_DIR = JOBS_DIR / "queued"`
- `JOBS_RUNNING_DIR = JOBS_DIR / "running"`
- `JOBS_DONE_DIR = JOBS_DIR / "done"`
- `JOBS_FAILED_DIR = JOBS_DIR / "failed"`
- `LOGS_DIR = WORKSPACE_ROOT / "logs"`
- `STATE_DIR = WORKSPACE_ROOT / "state"`
- `CACHE_DIR = WORKSPACE_ROOT / "cache"`
- `GPU_LOCK_PATH = STATE_DIR / "gpu.lock"`
- `SQLITE_PATH = STATE_DIR / "app.db"`

### 7.2 Run-ID-Konvention

Run-IDs sind die Namen der Run-Ordner, z. B.:

- `20260418-120805-baseline`
- `20260418-131627-kfold`

Diese IDs werden in API und Frontend unverändert verwendet.

### 7.3 Tile-ID-Konvention

Tile-IDs z. B.:

- `18NWG_6_6`
- `19LCH_4_7`

Artefaktpfade immer nach Schema:

```text
/workspace/artifacts/{run_id}/{tile_id}/
```

---

## 8. Run-Contract

Jeder Run in `/workspace/runs/{run_id}` soll standardisiert lesbar sein.

### 8.1 Mindestdateien

Pflicht:
- `config.json`

Optional aber empfohlen:
- `metrics.json`
- `history.json`
- `best.pt`
- `last.pt`
- `manifest.json`

### 8.2 Run-Status

Der Backend-Scanner soll folgende Status ableiten:

- `created`
- `running`
- `finished`
- `failed`
- `partial`

Ableitungslogik:

- `running`:
  - wenn explizit ein laufender Job oder Prozess zugeordnet ist
- `finished`:
  - wenn `config.json` vorhanden ist und mindestens einer aus `metrics.json` oder `best.pt` existiert
- `partial`:
  - nur `config.json`, aber keine Abschlussartefakte
- `failed`:
  - optional per Marker-Datei oder Job-Status
- `created`:
  - Run-Ordner existiert, aber fast leer

### 8.3 manifest.json

Wenn nicht vorhanden, kann es vom Backend on-the-fly erzeugt werden.

Zielstruktur:

```json
{
  "run_id": "20260418-120805-baseline",
  "status": "finished",
  "model_type": "baseline",
  "created_at": "2026-04-18T12:08:05Z",
  "checkpoint": "best.pt",
  "metrics": {
    "val_f1": 0.61,
    "val_iou": 0.44,
    "pr_auc": 0.67
  },
  "artifacts_ready": true,
  "tile_count": 12,
  "available_tiles": ["18NWG_6_6", "19LCH_4_7"]
}
```

Die API darf dieses Manifest auch dynamisch aus `config.json`, `metrics.json` und vorhandenen Artefakten erzeugen.

---

## 9. Tile-Artifact-Contract

Für jeden Tile-Ordner unter `/workspace/artifacts/{run_id}/{tile_id}/` gelten standardisierte Dateien.

### 9.1 Empfohlene Dateien

- `summary.json`
- `pred.geojson`
- `overlay.png`
- `prob.tif`
- `uncertainty.tif`
- `raw_pred.tif`

Nicht alle Dateien sind Pflicht. `summary.json` ist Pflicht, sobald Tile-Artefakte für die Web-App bereitgestellt werden.

### 9.2 summary.json

Schema:

```json
{
  "run_id": "20260418-120805-baseline",
  "tile_id": "18NWG_6_6",
  "status": "ready",
  "threshold": 0.42,
  "artifacts": {
    "geojson": true,
    "overlay_png": true,
    "prob_tif": true,
    "uncertainty_tif": false
  },
  "paths": {
    "geojson": "/artifacts/20260418-120805-baseline/18NWG_6_6/pred.geojson",
    "overlay_png": "/artifacts/20260418-120805-baseline/18NWG_6_6/overlay.png",
    "prob_tif": "/artifacts/20260418-120805-baseline/18NWG_6_6/prob.tif"
  },
  "metrics": {
    "pixel_f1": null,
    "pixel_iou": null
  }
}
```

### 9.3 Web-Regel

Das Frontend nutzt ausschließlich `summary.json` und API-Rückgaben. Es soll keine Dateisystemkonventionen erraten.

---

## 10. Job-Contract

Jobs werden als JSON-Dateien gespeichert.

### 10.1 Job-Typen

Mindestens implementieren:

- `infer_tile`
- `refresh_registry`

Optional:
- `render_overlay`
- `export_submission`
- `recompute_threshold`

### 10.2 Job-Datei

Beispiel:

```json
{
  "job_id": "job_20260418_150000_0001",
  "type": "infer_tile",
  "status": "queued",
  "created_at": "2026-04-18T15:00:00Z",
  "updated_at": "2026-04-18T15:00:00Z",
  "run_id": "20260418-120805-baseline",
  "tile_id": "18NWG_6_6",
  "requires_gpu": true,
  "params": {
    "threshold": 0.42,
    "export_geojson": true,
    "export_overlay": true,
    "morph_cleanup": true
  },
  "result": null,
  "error": null
}
```

### 10.3 Job-Status

Zulässige Stati:
- `queued`
- `running`
- `done`
- `failed`

Die Status ergeben sich aus dem Verzeichnis, in dem die Job-Datei liegt.

### 10.4 Queue-Mechanik

- API erzeugt JSON in `/workspace/jobs/queued`
- Worker scannt dieses Verzeichnis regelmäßig
- Worker verschiebt Job atomar nach `/workspace/jobs/running`
- Nach Abschluss wird Job nach `/workspace/jobs/done` oder `/workspace/jobs/failed` verschoben

### 10.5 Atomare Dateioperationen

Unbedingt `os.replace` oder äquivalente atomare Moves verwenden.
Keine halbgeschriebenen JSON-Dateien lesen.

Empfohlene Schreibweise:
1. in temporäre Datei schreiben
2. `fsync`
3. atomar umbenennen

---

## 11. API-Design

Die API wird als FastAPI-App implementiert.

### 11.1 Basisendpunkte

#### `GET /api/health`
Antwort:
```json
{ "status": "ok" }
```

Zusätzlich optional:
- API-Version
- Pfade validiert
- Worker sichtbar

#### `GET /api/runs`
Liefert Liste aller Runs.

Antwort:
```json
{
  "runs": [
    {
      "run_id": "20260418-120805-baseline",
      "status": "finished",
      "model_type": "baseline",
      "created_at": "2026-04-18T12:08:05Z",
      "metrics": {
        "val_f1": 0.61,
        "val_iou": 0.44
      },
      "artifact_tile_count": 2
    }
  ]
}
```

#### `GET /api/runs/{run_id}`
Liefert Details zu einem Run.

Antwort:
```json
{
  "run_id": "20260418-120805-baseline",
  "status": "finished",
  "config": {...},
  "metrics": {...},
  "history": {...},
  "tiles": ["18NWG_6_6", "19LCH_4_7"],
  "artifacts_ready": true
}
```

#### `GET /api/runs/{run_id}/tiles`
Liefert alle Tile-Summaries zu einem Run.

Antwort:
```json
{
  "run_id": "20260418-120805-baseline",
  "tiles": [
    {
      "tile_id": "18NWG_6_6",
      "status": "ready",
      "has_geojson": true,
      "has_overlay": true,
      "has_probability": true
    }
  ]
}
```

#### `GET /api/runs/{run_id}/tiles/{tile_id}`
Liefert Details zu genau einem Tile.

Antwort:
```json
{
  "run_id": "20260418-120805-baseline",
  "tile_id": "18NWG_6_6",
  "summary": {...}
}
```

#### `POST /api/jobs/infer-tile`
Erstellt Inferenzjob.

Request:
```json
{
  "run_id": "20260418-120805-baseline",
  "tile_id": "18NWG_6_6",
  "threshold": 0.42,
  "export_geojson": true,
  "export_overlay": true,
  "morph_cleanup": true
}
```

Antwort:
```json
{
  "job_id": "job_20260418_150000_0001",
  "status": "queued"
}
```

#### `GET /api/jobs/{job_id}`
Liefert Jobstatus.

Antwort:
```json
{
  "job_id": "job_20260418_150000_0001",
  "type": "infer_tile",
  "status": "running",
  "run_id": "20260418-120805-baseline",
  "tile_id": "18NWG_6_6",
  "error": null,
  "result": null
}
```

#### `GET /api/jobs`
Optional: Liste aller oder letzter Jobs.

---

## 12. API-Schemas

Alle Request-/Response-Schemas sollen in `src/api/schemas.py` als Pydantic-Modelle liegen.

Benötigte Modelle:
- `HealthResponse`
- `RunSummary`
- `RunListResponse`
- `RunDetailResponse`
- `TileSummary`
- `TileListResponse`
- `TileDetailResponse`
- `CreateInferTileJobRequest`
- `CreateJobResponse`
- `JobStatusResponse`

Anforderungen:
- stabile Feldnamen
- keine internen Python-Objekte nach außen leaken
- Pfade als Web-Pfade, nicht als interne Absolute Paths ausgeben

---

## 13. Registry / Run-Scanner

### 13.1 Aufgabe

Der Registry-Layer liest `/workspace/runs` und `/workspace/artifacts` und bildet daraus normalisierte Datenstrukturen für die API.

### 13.2 Funktionen

#### `scan_all_runs()`
Liest alle Unterordner in `/workspace/runs`.

Je Run:
- `config.json` lesen, falls vorhanden
- `metrics.json` lesen, falls vorhanden
- `history.json` lesen, falls vorhanden
- Checkpoint-Dateien erkennen
- zugehörige Tile-Artefakte unter `/workspace/artifacts/{run_id}` scannen
- Run-Status ableiten
- RunSummary erzeugen

#### `get_run_detail(run_id)`
Liefert erweitertes Run-Objekt mit Config, Metrics und Tile-Liste.

#### `get_tile_summary(run_id, tile_id)`
Liest `summary.json` oder baut Summary aus Dateivorhandensein.

### 13.3 Toleranz gegen unvollständige Runs

Der Scanner muss robust sein gegen:
- fehlende `metrics.json`
- unvollständige `history.json`
- leere `fold_0_18N`-Unterordner
- alte Runs ohne `manifest.json`
- defekte JSON-Dateien

Verhalten:
- niemals API komplett crashen lassen
- fehlerhafte Runs markieren
- Fehler loggen
- soweit möglich degradierte Antwort liefern

---

## 14. Worker-Design

### 14.1 Worker-Hauptschleife

`src/worker/main.py` implementiert einen Polling-Loop:

1. queued jobs lesen
2. nächstbesten Job wählen
3. nach `running` verschieben
4. falls GPU-Job: Lock anfordern
5. Job ausführen
6. Ergebnis schreiben
7. nach `done` oder `failed` verschieben
8. Lock freigeben
9. schlafen, dann wiederholen

Polling-Intervall:
- 1 bis 3 Sekunden

### 14.2 Job-Ausführung

`src/worker/executor.py` enthält Dispatcher:

```python
if job.type == "infer_tile":
    execute_infer_tile(job)
elif job.type == "refresh_registry":
    execute_refresh_registry(job)
else:
    raise ValueError(...)
```

### 14.3 infer_tile-Job

Schritte:

1. Run validieren
   - Run-Ordner existiert
   - Checkpoint vorhanden
2. Tile validieren
3. Checkpoint-Pfad bestimmen
4. Pipeline-Inferenzfunktion aufrufen
5. Vorhersage/GeoJSON/Overlay schreiben
6. `summary.json` schreiben
7. Job `done`

### 14.4 Wiederverwendung bestehender Pipeline

Bestehende Module sollen wiederverwendet werden.
Der Worker soll **nicht** eigene Modelllogik neu implementieren.

Bevorzugte Wiederverwendung:
- `src/infer.py`
- `src/postprocess.py`
- `src/raster_utils.py`
- `src/metrics.py`
- `src/features.py`
- `src/feature_bank.py`

Anpassung nötig, falls diese Module bisher nur CLI-fähig und nicht library-fähig sind.

Ziel:
- Pipeline-Code in aufrufbare Funktionen extrahieren
- CLI bleibt dünne Hülle um dieselben Funktionen

Beispiel:

```python
def run_tile_inference(run_id: str, tile_id: str, ... ) -> TileInferenceResult:
    ...
```

---

## 15. GPU-Lock

### 15.1 Motivation

Da nur ein Server und eine GPU vorhanden sind, dürfen GPU-Jobs sich nicht unkoordiniert überlagern.

### 15.2 Implementierung

Dateibasierter Lock:

Pfad:
```text
/workspace/state/gpu.lock
```

Inhalt der Lock-Datei:
```json
{
  "job_id": "job_20260418_150000_0001",
  "type": "infer_tile",
  "created_at": "2026-04-18T15:01:00Z",
  "pid": 12345
}
```

### 15.3 Verhalten

- Vor Start eines GPU-Jobs: Lock prüfen
- Wenn nicht vorhanden: Lock setzen
- Wenn vorhanden und gültig: Job nicht starten
- Nach Abschluss oder Fehler: Lock löschen

### 15.4 Robustheit

Wenn Worker abstürzt, kann stale lock entstehen.
Deshalb optional:
- PID prüfen
- Lock-Timestamp prüfen
- TTL oder Recovery-Mechanismus

Einfacher erster Schritt:
- Lock-Datei + PID
- beim Worker-Start prüfen, ob PID noch lebt

---

## 16. Logging

### 16.1 Ziele

- API-Fehler nachvollziehbar
- Job-Lebenszyklus nachvollziehbar
- Worker-Fehler nachvollziehbar
- keine unkontrollierte Log-Flut

### 16.2 Dateien

- `/workspace/logs/api.log`
- `/workspace/logs/worker.log`
- `/workspace/logs/training/{run_id}.log`

### 16.3 Inhalt

Loggen:
- API Requests auf Info-Level nur knapp
- Job-Erstellung
- Job-Start / Job-Ende / Dauer
- Lock-Erwerb / Lock-Freigabe
- Exceptions mit Stacktrace

Nicht loggen:
- riesige Arrays
- komplette Modellzustände
- Binärpfade in Schleifen ohne Mehrwert

---

## 17. Fehlerbehandlung

### 17.1 API

- 404 bei unbekanntem Run oder Tile
- 400 bei invalidem Request
- 409 bei Job-Konflikten, falls sinnvoll
- 500 nur bei echten internen Fehlern

### 17.2 Worker

Wenn ein Job fehlschlägt:
- Job nach `/workspace/jobs/failed`
- `error`-Feld im Job befüllen
- soweit möglich Stacktrace oder Fehlermeldung in Log schreiben
- GPU-Lock sicher freigeben

### 17.3 Teilweise fehlende Artefakte

Die API soll niemals abstürzen, wenn z. B. nur `overlay.png` fehlt.
Stattdessen klare boolesche Felder liefern:
- `has_geojson`
- `has_overlay`
- `has_probability`

---

## 18. Security / Betriebsgrenzen

### 18.1 Nur interne API-Operationen

Das Backend darf keine beliebigen Shell-Kommandos aus dem Web entgegennehmen.
Keine freien Dateipfade aus User-Input direkt verwenden.

### 18.2 Validierung

- `run_id` nur aus vorhandenen Run-IDs erlauben
- `tile_id` nur erlaubte Zeichen und bekannte Tiles
- Threshold nur in zulässigem Bereich, z. B. `0.0 <= thr <= 1.0`

### 18.3 Keine direkte Modellwahl aus beliebigem Pfad

Checkpoint wird immer aus dem Run-Kontext aufgelöst. Keine freien absoluten Pfade im API-Request.

---

## 19. Nginx / Serving-Konzept

Auch wenn Nginx nicht Teil des Python-Backends ist, muss das Backend darauf ausgerichtet werden.

### 19.1 Zielrouting

- `/` → Next.js
- `/api/*` → FastAPI
- `/artifacts/*` → statische Dateien aus `/workspace/artifacts/*`

### 19.2 Backend-Pfade vs. Web-Pfade

Das Backend darf intern absolute Pfade kennen, soll aber in API-Antworten nur Web-Pfade ausgeben.

Beispiel:
- intern: `/workspace/artifacts/20260418-120805-baseline/18NWG_6_6/pred.geojson`
- extern: `/artifacts/20260418-120805-baseline/18NWG_6_6/pred.geojson`

---

## 20. Anpassungen an bestehender Pipeline

### 20.1 Ziel

Bestehende Pipeline-Dateien existieren bereits, sind aber wahrscheinlich primär CLI-orientiert.
Sie müssen so refaktoriert werden, dass der Worker sie programmgesteuert aufrufen kann.

### 20.2 Konkrete Anforderungen

Folgende Module prüfen und ggf. refaktorieren:

- `src/train.py`
- `src/infer.py`
- `src/postprocess.py`
- `src/evaluate.py`
- `src/raster_utils.py`

### 20.3 Regel

Jedes CLI-Skript sollte intern auf Funktionen aufbauen.

Beispiel:

Statt:
```python
# infer.py
if __name__ == "__main__":
    ... alles hier ...
```

Soll:
```python
def infer_tile(...):
    ...

def main():
    ... parse args ...
    infer_tile(...)

if __name__ == "__main__":
    main()
```

Damit kann der Worker `infer_tile(...)` direkt importieren.

---

## 21. Minimaler Lieferumfang für die erste Backend-Version

### 21.1 Muss implementiert werden

1. Verzeichnisinitialisierung für `/workspace/*`
2. zentraler Pfadlayer
3. Run-Scanner
4. FastAPI-App
5. Endpunkte:
   - `GET /api/health`
   - `GET /api/runs`
   - `GET /api/runs/{run_id}`
   - `GET /api/runs/{run_id}/tiles`
   - `GET /api/runs/{run_id}/tiles/{tile_id}`
   - `POST /api/jobs/infer-tile`
   - `GET /api/jobs/{job_id}`
6. dateibasierte Queue
7. Worker-Hauptschleife
8. GPU-Lock
9. `infer_tile`-Job
10. `summary.json`-Erzeugung je Tile

### 21.2 Kann zunächst vereinfacht werden

- SQLite ist optional in V1
- `manifest.json` kann dynamisch erzeugt werden
- `refresh_registry` darf zunächst ein No-Op oder einfacher Rescan sein
- `history.json` darf unstrukturiert durchgereicht oder gekürzt werden

---

## 22. Empfohlene Umsetzungsreihenfolge

### Schritt 1 — Workspace vorbereiten
Implementiere Initialisierung:
- Verzeichnisse erzeugen, falls fehlend
- Logging initialisieren
- zentrale Pfade testen

### Schritt 2 — Registry bauen
Implementiere:
- `scan_all_runs()`
- `get_run_detail()`
- `get_tile_summary()`

Teste gegen die aktuellen vorhandenen Run-Ordner.

### Schritt 3 — API Read-Only
Implementiere:
- `GET /api/health`
- `GET /api/runs`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/tiles`
- `GET /api/runs/{run_id}/tiles/{tile_id}`

### Schritt 4 — Queue und Job-Schemas
Implementiere:
- Job-Datei-Schema
- Queue-Write
- Queue-Read
- atomare File-Moves

### Schritt 5 — Worker-Grundgerüst
Implementiere:
- Polling-Loop
- `running`/`done`/`failed`
- Logging
- GPU-Lock

### Schritt 6 — infer_tile integrieren
Refaktoriere bestehende Inferenzlogik in importierbare Funktion.
Worker soll daraus Tile-Artefakte erzeugen.

### Schritt 7 — Tile-Summary und Web-Pfade
Stelle sicher, dass nach einem Job `summary.json` und Artefakte im erwarteten Pfad liegen.

### Schritt 8 — Frontend anschließen
Frontend ersetzt statische `public/data`-Demo-Daten durch API-Calls.

---

## 23. Definition of Done

Das Backend gilt als fertig für V1, wenn alle folgenden Punkte erfüllt sind:

1. `GET /api/runs` listet die bestehenden Run-Ordner aus `/workspace/runs` korrekt auf.
2. `GET /api/runs/{run_id}` liefert Config/Metrics robust, auch bei unvollständigen Runs.
3. `POST /api/jobs/infer-tile` erzeugt einen Job.
4. Worker verarbeitet den Job automatisch.
5. Bei erfolgreicher Inferenz entstehen Tile-Artefakte unter `/workspace/artifacts/{run_id}/{tile_id}/`.
6. `GET /api/jobs/{job_id}` zeigt Statuswechsel korrekt.
7. GPU-Jobs laufen niemals parallel.
8. Frontend kann Run-Liste, Tile-Liste und mindestens ein GeoJSON/Overlay laden.
9. Das System bleibt stabil, wenn ein Run unvollständig oder defekt ist.

---

## 24. Out of Scope für V1

Diese Dinge gehören nicht in die erste Backend-Iteration:

- Multi-GPU oder verteiltes Training
- freie Shell-Ausführung über die API
- Nutzerverwaltung / komplexe Auth
- echtes Message-Broker-System
- Live-Streaming großer Raster durch Python, wenn statisches Serving reicht
- vollständiges Experiment-Tracking-System
- Datenbank-First-Architektur

---

## 25. Umsetzungshinweise für ein Coding-LLM

### 25.1 Erwartete Arbeitsweise

Das Coding-LLM soll:
- vorhandenen Code in `/shared-docker/src` analysieren
- fehlende Backend-Module neu anlegen
- bestehende CLI-Logik in importierbare Funktionen extrahieren
- keine unnötigen neuen Frameworks einführen
- alle Pfade auf `/workspace/*` standardisieren
- robuste Fehlerbehandlung implementieren

### 25.2 Nicht tun

- keine harte Kopplung an `www/public/data`
- keine direkte Inferenz im FastAPI-Request-Handler
- keine beliebigen absoluten Pfade aus Web-Input akzeptieren
- keine Modelllogik doppelt implementieren, wenn sie bereits in `src/infer.py` vorhanden ist

### 25.3 Bevorzugtes Ergebnis

Das Endergebnis soll so aussehen, dass folgende Prozesse separat gestartet werden können:

#### API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

#### Worker
```bash
python -m src.worker.main
```

#### Frontend
```bash
cd /shared-docker/www && npm run dev
```

---

## 26. Abschluss

Dieses Backend ist absichtlich pragmatisch entworfen:
- ein Server
- getrennte Prozesse
- dateibasierte Queue
- standardisierte Artefakte
- API als Orchestrator
- Worker als einziger Rechenpfad für schwere Jobs

Damit bleibt die Website stabil, obwohl Training und Inferenz auf derselben Maschine laufen.