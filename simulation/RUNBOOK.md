# RUNBOOK — Thesis-Simulation auf SCCKN

Anleitung, um nach einem Login auf `scc2` (oder nach einem Crash) die Simulation
wieder zum Laufen zu bringen. Setup ist einmalig erfolgt; dieses Runbook
beschreibt nur den wiederkehrenden Startvorgang.

---

## Voraussetzungen (einmalig, sollten bereits vorhanden sein)

- venv unter `/data/scc/jonas.schrade/envs/thesis`
- Ollama-User-Installation unter `/data/scc/jonas.schrade/ollama/install`
- Modelle `qwen2.5:14b` und `nomic-embed-text` unter
  `/data/scc/jonas.schrade/ollama/models`
- Setup-Skript `~/setup_thesis_env.sh`
- Code unter `~/projects/thesis/simulation`

Falls eines dieser Elemente fehlt: Setup-Schritte aus dem SETUP-Verlauf
nachholen, nicht aus diesem Runbook.

---

## Schritt 1 — GPU-Knoten anfordern

Auf `scc2` (Frontend):

```bash
qrsh -q gpu -l gpu=1,h_vmem=64G -l hostname=scc213
```

**Wichtig:** `h_vmem=64G` ist kein optionaler Komfort. Der Default ist 1 GB
Virtual Memory, und damit scheitert das `mmap` der CUDA-Bibliotheken — Ollama
fällt dann stillschweigend auf CPU zurück.

Bei Erfolg ändert sich der Prompt zu `scc213`. Falls `no suitable queues`
kommt, ist der Knoten ausgelastet — entweder warten oder ein anderes Memory-
Limit probieren (32G, 16G).

---

## Schritt 2 — Environment laden

```bash
source ~/setup_thesis_env.sh
```

Erwarteter Output (vier Zeilen):

```
Thesis environment ready.
  Python:               /data/scc/jonas.schrade/envs/thesis/bin/python
  Ollama:               /data/scc/jonas.schrade/ollama/install/bin/ollama
  OLLAMA_MODELS:        /data/scc/jonas.schrade/ollama/models
  CUDA_VISIBLE_DEVICES: 7
```

Das Skript setzt: Anaconda-Modul, venv-Aktivierung, PATH/LD_LIBRARY_PATH für
Ollama, Modell-Verzeichnis, GPU-Auswahl (Default 7).

---

## Schritt 3 — Freie GPU prüfen und ggf. wechseln

GPU-Belegung anschauen:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
```

Frei = `memory.used` deutlich unter 1 GB. Wenn GPU 7 belegt ist, eine andere
freie GPU wählen:

```bash
export CUDA_VISIBLE_DEVICES=<index>
```

Andere User auf demselben Knoten arbeiten parallel auf weiteren GPUs — Ollama
muss strikt auf eine freie GPU eingeschränkt werden, sonst Kollision.

---

## Schritt 4 — Ollama-Daemon starten

```bash
ollama serve > ~/ollama.log 2>&1 &
echo "Server-PID: $!"
sleep 5
ollama list
```

Erwartung: Die Modelle `qwen2.5:14b` und `nomic-embed-text` erscheinen in der
Liste. Falls `could not connect to ollama app`, ist der Daemon abgestürzt —
`cat ~/ollama.log` zeigt die Ursache.

Kurzcheck, dass die GPU wirklich erkannt wurde:

```bash
grep "inference compute" ~/ollama.log
```

Es muss `library=CUDA` und `description="NVIDIA L40S"` erscheinen.
Wenn dort `library=cpu` steht, läuft Ollama im CPU-Modus — dann zurück zu
Schritt 1 und prüfen, ob `h_vmem=64G` wirklich gesetzt war.

---

## Schritt 5 — Simulation starten

Outputs landen relativ zum CWD im Ordner `logs/run_<timestamp>/`. Daher vor
dem Start ins Output-Verzeichnis wechseln:

```bash
mkdir -p /data/scc/jonas.schrade/thesis-runs
cd /data/scc/jonas.schrade/thesis-runs
```

Dann je nach Modus eines der beiden Skripte starten:

**Pairwise-Smoke-Test (schnell, 2 Agents):**

```bash
python ~/projects/thesis/simulation/main_pairwise.py
```

**Network-Simulation (langsamer, NUM_AGENTS aus `config.py`):**

```bash
python ~/projects/thesis/simulation/main_network.py
```

Parameter (Anzahl Agents, Runden, Modell, Topic) werden in
`~/projects/thesis/simulation/config.py` gesetzt.

---

## Schritt 6 — Längere Runs im Hintergrund

Für Network-Simulationen mit größeren Parametern empfiehlt sich ein
robuster Background-Start, der eine SSH-Disconnection übersteht:

```bash
cd /data/scc/jonas.schrade/thesis-runs
nohup python ~/projects/thesis/simulation/main_network.py > run_$(date +%Y%m%d_%H%M%S).out 2>&1 &
echo "Sim-PID: $!"
```

Fortschritt beobachten:

```bash
tail -f run_*.out         # neueste stdout/stderr
ls logs/                  # neue run_<timestamp>/-Ordner
```

---

## Schritt 7 — Ergebnisse prüfen

```bash
ls /data/scc/jonas.schrade/thesis-runs/logs/run_<timestamp>/
```

Erwartete Files:

- `events.jsonl` — eine Zeile pro Interaktion (primäre Analyse-Daten)
- `personas.json` — die für den Run gezogenen Personas
- `network_rounds/` — Snapshots pro Runde

---

## Schritt 8 — Sauberes Herunterfahren

Ollama-Daemon stoppen und qrsh-Session verlassen:

```bash
kill %1 2>/dev/null
exit
```

Damit gibt SGE den Knoten und die GPU für andere User frei. Daten unter
`/data/scc/` und unter `~/` bleiben erhalten.

---

## Fehlersuche — Häufige Stolpersteine

| Symptom | Ursache | Gegenmittel |
|---|---|---|
| `no suitable queues` | h_vmem fehlt / zu hoch / scc213 voll | `qrsh` mit `h_vmem=64G`, ggf. anderer Host |
| Ollama startet, `library=cpu` | h_vmem zu niedrig → mmap scheitert | qrsh neu mit `h_vmem=64G` |
| `could not connect to ollama app` | Daemon abgestürzt | `cat ~/ollama.log` analysieren |
| `failed to map segment` | h_vmem-Limit | h_vmem erhöhen |
| Python ImportError | venv nicht aktiv | `source ~/setup_thesis_env.sh` |
| Ollama findet Modell nicht | OLLAMA_MODELS nicht gesetzt | Setup-Skript erneut sourcen |

---

## Hinweise für später

- Aktuelle `~/projects/thesis/.gitignore` hat ein zu breites `*.json` — irgendwann
  durch gezieltere Patterns ersetzen (z. B. nur `data/*.json`, `logs/`).
- Für persistente ChromaDB-Memory (`MEMORY_PERSIST=True`) sollte
  `MEMORY_DIR` aus `config.py` auf einen Pfad unter `/data/scc/` zeigen.
- Bei wiederkehrenden Network-Runs mit identischer Konfiguration lohnt es sich,
  diesen Workflow in ein qsub-Batch-Skript zu überführen — dann läuft der Job
  ohne offene Shell-Session.
- GPU-Auswahl im Setup-Skript ist hartkodiert auf 7. Wenn das auf Dauer nervt,
  ließe sich das skripten („freieste GPU automatisch wählen").
