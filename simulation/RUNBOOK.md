
raw
Runbook · MD
# RUNBOOK — Thesis-Simulation auf SCCKN
 
Anleitung, um nach einem Login auf `scc2` (oder nach einem Crash) die Simulation
wieder zum Laufen zu bringen. Setup ist einmalig erfolgt; dieses Runbook
beschreibt nur den wiederkehrenden Startvorgang.
 
> **Stand:** Überarbeitet nach Debugging-Session vom 04.06.2026. Frühere
> Annahmen (insb. „h_vmem verursacht CPU-Fallback") waren falsch und wurden
> korrigiert — siehe Fehlersuche.
 
---
 
## Voraussetzungen (einmalig, sollten bereits vorhanden sein)
 
- venv unter `/data/scc/jonas.schrade/envs/thesis`
- Ollama-User-Installation unter `/data/scc/jonas.schrade/ollama/install`
  (Version **0.24.0** — nicht die System-Installation `/software/bin/ollama`, die 0.1.26 ist!)
- Modelle unter `/data/scc/jonas.schrade/ollama/models`:
  - `qwen2.5:14b` — validiert, schnell, primäres Modell
  - `qwen2.5:32b` — läuft sauber, langsamer
  - `nomic-embed-text` — Embeddings
- Setup-Skript `~/setup_thesis_env.sh`
- Code unter `~/projects/thesis/simulation`
Falls eines dieser Elemente fehlt: Setup-Schritte aus dem SETUP-Verlauf
nachholen, nicht aus diesem Runbook.
 
---
 
## Schritt 1 — GPU-Knoten anfordern
 
Auf `scc2` (Frontend). **Nicht** den Hostnamen pinnen (`-l hostname=...`) — das
schlägt sofort fehl, wenn der Knoten belegt/offline ist. Stattdessen über das
GPU-Typ-Feature `tesla_l40` einen L40-Knoten anfordern:
 
```bash
qrsh -q gpu -l h_vmem=128G,tesla_l40=1
```
 
**Memory-Wahl (`h_vmem`):**
 
- `h_vmem=64G` reicht für `qwen2.5:14b`.
- `h_vmem=128G` für `qwen2.5:32b` (und größere Modelle) — bei 64G scheitert
  beim Laden großer Modelle die **Host-Allokation** (`ggml_aligned_malloc:
  insufficient memory`), nicht die GPU. Im Zweifel 128G.
**Warum kein L40S, sondern L40?** Die verfügbaren Knoten melden sich als
`NVIDIA L40` (compute 8.9, 45 GiB) — funktional dieselbe Generation, stabile
Architektur. Das `tesla_l40`-Feature trifft beide.
 
Bei Erfolg ändert sich der Prompt zum Knotennamen (z. B. `scc192`). Falls
`could not be scheduled` kommt: kurz warten und erneut probieren (der Scheduler
weist irgendeinen freien L40-Knoten zu).
 
---
 
## Schritt 2 — Environment laden
 
```bash
source ~/setup_thesis_env.sh
```
 
Erwarteter Output:
 
```
Thesis environment ready.
  Python:               /data/scc/jonas.schrade/envs/thesis/bin/python
  Ollama:               /data/scc/jonas.schrade/ollama/install/bin/ollama
  OLLAMA_MODELS:        /data/scc/jonas.schrade/ollama/models
  CUDA_VISIBLE_DEVICES: 7
```
 
**Sofort die Ollama-Binary prüfen** (häufigste, teuerste Fehlerquelle):
 
```bash
which ollama && ollama --version
```
 
Muss `/data/scc/jonas.schrade/ollama/install/bin/ollama` und **client version
0.24.0** zeigen. Falls `/software/bin/ollama` oder `0.1.26` erscheint, wurde das
Environment nicht korrekt geladen — erneut sourcen. Die System-Binary 0.1.26
unterstützt moderne GPUs nicht und stürzt beim Modell-Laden stillschweigend ab.
 
---
 
## Schritt 3 — GPU wählen (WICHTIG: nicht 7!)
 
Das Setup-Skript setzt `CUDA_VISIBLE_DEVICES=7` als Default. **Die L40-Knoten
haben aber nur GPUs 0–3.** GPU 7 existiert dort nicht → Ollama findet keine GPU.
Daher immer explizit auf eine vorhandene, freie GPU setzen:
 
```bash
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
export CUDA_VISIBLE_DEVICES=0   # oder 1–3, je nachdem was frei ist
```
 
Frei = `memory.used` deutlich unter 1 GB. Andere User arbeiten parallel auf
anderen GPUs — Ollama muss strikt auf eine freie GPU eingeschränkt werden.
 
---
 
## Schritt 4 — Port wählen und Ollama-Daemon starten
 
**Achtung scc192 (und ggf. andere Knoten):** Dort läuft ein **root-eigener
`ollama serve` auf dem Default-Port 11434**. Der eigene Server muss deshalb auf
einen anderen Port. Da der Code `OLLAMA_HOST` liest, folgt die Simulation
automatisch:
 
```bash
export OLLAMA_HOST=127.0.0.1:11500
```
 
Sicherstellen, dass kein eigener Ollama mehr läuft (**nur eigene Prozesse
killen**, niemals den root-Prozess anfassen):
 
```bash
pkill -u jonas.schrade -f ollama; sleep 2; pgrep -u jonas.schrade -af ollama
```
 
(Die zweite Ausgabe sollte leer sein. `pkill -f ollama` ohne `-u` schlägt auf
root-Prozessen mit „Operation not permitted" fehl — das ist korrekt, der
root-Server ist nicht deiner.)
 
Dann starten:
 
```bash
ollama serve > ~/ollama.log 2>&1 &
echo "Server-PID: $!"
sleep 5
ollama list
```
 
Erwartung: Die Modelle erscheinen in der Liste. Falls `could not connect`, ist
der Daemon abgestürzt oder der Port belegt — `tail -n 30 ~/ollama.log` zeigt die
Ursache (z. B. `address already in use` → falscher Port).
 
**GPU-Erkennung prüfen:**
 
```bash
grep "inference compute" ~/ollama.log
```
 
Erwartung: `library=CUDA compute=8.9 ... description="NVIDIA L40" total="45.0 GiB"`.
Steht dort eine CPU-Fallback-Meldung bzw. `offloaded 0/NN layers to GPU`, liegt
es **nicht** an `h_vmem`, sondern an der Context-/Layout-Größe — siehe Fehlersuche.
 
---
 
## Schritt 5 — Modell in config.py setzen
 
Vor dem Start sicherstellen, dass `LLM_MODEL` auf ein **lauffähiges** Modell
zeigt (`~/projects/thesis/simulation/config.py`, Zeile ~20):
 
```python
LLM_MODEL = "qwen2.5:14b"   # oder "qwen2.5:32b"
```
 
`num_ctx=8192` muss im `OllamaLLM(...)`-Konstruktor stehen (in **beiden**
Entry-Points: `main_pairwise.py` und `main_network.py`). Ohne diesen Cap
allokieren große Modelle einen riesigen Default-Context und sprengen entweder
die GPU (OOM) oder fallen auf CPU zurück.
 
> **NICHT verwenden:** `qwen3.5:9b` / `qwen3.5:27b` — stürzen mit `exit status 2`
> (Runner-Panic) ab, sowohl auf Blackwell als auch auf L40. Architektur-/Build-
> Inkompatibilität mit Ollama 0.24.0, **nicht** per Konfiguration behebbar.
> Erst mit neuerem Ollama-Build erneut testen.
 
---
 
## Schritt 6 — Simulation starten
 
Outputs landen relativ zum CWD im Ordner `logs/run_<timestamp>/`. Daher vor
dem Start ins Output-Verzeichnis wechseln:
 
```bash
mkdir -p /data/scc/jonas.schrade/thesis-runs
cd /data/scc/jonas.schrade/thesis-runs
```
 
**Pairwise-Smoke-Test (schnell, 2 Agents):**
 
```bash
python ~/projects/thesis/simulation/main_pairwise.py
```
 
**Network-Simulation (langsamer, NUM_AGENTS aus `config.py`):**
 
```bash
python ~/projects/thesis/simulation/main_network.py
```
 
Parameter (Anzahl Agents, Runden, Modell, Topic) in `config.py`.
 
---
 
## Schritt 7 — Längere Runs im Hintergrund
 
Für Network-Simulationen mit größeren Parametern ein Background-Start, der eine
SSH-Disconnection übersteht:
 
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
 
> Hinweis: `qwen2.5:32b` ist deutlich langsamer pro Call als `14b`. Bei
> Network-Runs (viele sequentielle Calls) vorher die Laufzeit eines kleinen
> Runs hochrechnen, bevor ein großer Lauf gestartet wird.
 
---
 
## Schritt 8 — Ergebnisse prüfen
 
```bash
ls /data/scc/jonas.schrade/thesis-runs/logs/run_<timestamp>/
```
 
Erwartete Files:
 
- `events.jsonl` — eine Zeile pro Interaktion (primäre Analyse-Daten)
- `personas.json` — die für den Run gezogenen Personas
- `network_rounds/` — Snapshots pro Runde
---
 
## Schritt 9 — Sauberes Herunterfahren
 
Nur den **eigenen** Daemon stoppen, dann qrsh-Session verlassen:
 
```bash
pkill -u jonas.schrade -f ollama; sleep 2
exit
```
 
Damit gibt SGE den Knoten und die GPU für andere User frei. Daten unter
`/data/scc/` und unter `~/` bleiben erhalten.
 
**Nach dem Verlassen auf `scc2` prüfen, dass nichts hängengeblieben ist:**
 
```bash
qstat -u jonas.schrade
```
 
Listet alle eigenen Jobs clusterweit. Hängt noch eine alte Session, mit
`qdel <job-id>` beenden (Job-ID vorher prüfen).
 
---

## Fehlersuche — Häufige Stolpersteine
 
| Symptom | Ursache | Gegenmittel |
|---|---|---|
| `qrsh ... could not be scheduled` | Hostname gepinnt oder Knoten belegt | Pin entfernen, `tesla_l40=1` nutzen, ggf. erneut probieren |
| `ollama --version` zeigt `0.1.26` / `/software/bin` | Environment nicht gesourced → System-Ollama | `source ~/setup_thesis_env.sh`, erneut prüfen |
| `address already in use` (Port 11434) | root-Ollama belegt Default-Port (scc192) | `export OLLAMA_HOST=127.0.0.1:11500` vor `ollama serve` |
| `pkill: Operation not permitted` | Versuch, root-Prozess zu killen | `-u jonas.schrade` ergänzen, root-Prozess ignorieren |
| `offloaded 0/NN layers to GPU` / CPU-Fallback | Context-/Layout-Größe passt nicht — **NICHT** h_vmem | `num_ctx=8192` im Konstruktor prüfen; Server sauber neu starten, damit keine Altmodelle Speicher belegen |
| `ggml_aligned_malloc: insufficient memory` | `h_vmem` zu niedrig für großes Modell | `qrsh` neu mit `h_vmem=128G` |
| Embeddings hängen/500 | `OLLAMA_CONTEXT_LENGTH` global gesetzt (nomic kann nur 2048) | Variable **nicht** global exportieren; `echo $OLLAMA_CONTEXT_LENGTH` muss leer/0 sein |
| `llama runner ... exit status 2` | qwen3.5-Architektur auf Ollama 0.24.0 instabil | Auf `qwen2.5`-Modell wechseln |
| `CUDA_VISIBLE_DEVICES=7` → keine GPU | GPU 7 existiert auf L40-Knoten nicht (nur 0–3) | `export CUDA_VISIBLE_DEVICES=0` (oder 1–3) |
| `could not connect to ollama app` | Daemon abgestürzt oder nicht gestartet | `tail -n 30 ~/ollama.log`, Port/Binary prüfen |
| Python ImportError | venv nicht aktiv | `source ~/setup_thesis_env.sh` |
 
---