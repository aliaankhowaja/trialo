# TrialO — Trial Eligibility Screening Prototype

An educational screening application. A user supplies a patient record,
selects or enters a trial's eligibility criteria, runs the fine-tuned
TrialGPT-style criterion-annotation Transformer, inspects each criterion-level
result with exactly quoted supporting evidence, and exports a readable
eligibility screening report. A benchmark view compares the implemented greedy
decoding mechanisms (`generate`, KV-`cache`, `recompute`) live, and shows the
*recorded* GPU speculative-decoding results.

> **⚠ This is an educational screening prototype — NOT a clinical enrollment
> decision.** Outputs are model-generated annotations for teaching and
> demonstration only. It must never be used for actual patient enrollment,
> clinical care, or regulatory decisions.

## Architecture

```
docker-compose.yml            # starts both services
app/
  backend/                    # FastAPI (port 8000)
    trailo_app/               # app package
      engine.py               # model load, note normalization, criterion scoring
      benchmark.py            # decode-mechanism comparison runner
      report.py               # Markdown/HTML/PDF report generation
      routers/                # screening, benchmark, report, library endpoints
      data/library.json       # bundled trials + sample patients (w/ expert refs)
      Dockerfile
  frontend/                   # vanilla React + Vite, served by nginx (port 3000)
    src/components/           # PatientForm, TrialPicker, ResultsView,
                              # BenchmarkView, ReportExport, Disclaimer
    Dockerfile, nginx.conf
```

The backend imports the repository's own validated research code —
`trailo.eval.inference`, `trailo.eval.decoding` (custom KV-cache and recompute
greedy loops), `trailo.eval.parsing` (strict JSON validation) — mounted
read-only into the container, so the app serves exactly the machinery the
reference evaluation validated. The frontend is a plain React (no UI framework)
app that talks to the API through the nginx `/api` proxy or directly through
server-sent events for live per-criterion progress.

## Model presets

The served Transformer is configuration-driven (`TRIALO_MODEL_CONFIG`):

| Value | Base model | LoRA adapter | Precision | Hardware |
|---|---|---|---|---|
| `final_prod` (default) | `google/gemma-3-270m-it` | `checkpoints/final_prod_seed42/best` (full 4-epoch finetune) | bf16 (fp32 on CPU) | CPU or GPU |
| `smoke_cpu` | `google/gemma-3-270m-it` | `checkpoints/smoke_cpu_seed42/best` (3-step smoke run — pipeline proof only) | fp32 | CPU |
| `final_a100` | `google/gemma-3n-E4B-it` | `checkpoints/final_a100_seed42/best` | bf16 | ≥40 GB GPU |

Note: locally, the `smoke_cpu` adapter file is a pointer stub (weights only
materialize where DVC pulled them); `final_prod_seed42` contains real weights.


To serve a **newly fine-tuned adapter** without code changes, set:

```bash
export TRIALO_MODEL_CONFIG=final_a100        # or point at any other preset name
export TRIALO_BASE_MODEL_ID=<new base model>
export TRIALO_CHECKPOINT_DIR=src/trailo/checkpoints/<new_run_id>/best
docker compose up --build
```

(GPU requires rebuilding the backend image with the CUDA torch wheel — see the
comment in `app/backend/Dockerfile`.)

## Startup

Requirements: Docker (with Compose v2). First start downloads the base model
from the Hugging Face Hub (~1 GB for the 270m preset; the cache lives in the
`trailo_hf_cache` named volume, so it is a one-time cost). The first screening
request also triggers a lazy ~5-15 s model load; every later request is fast.

```bash
# from the repository root
docker compose up --build
```

- Frontend (the user interface): **http://localhost:3000**
- Backend API directly: http://localhost:8000 — interactive docs at
  http://localhost:8000/api/docs, health at http://localhost:8000/api/health

Useful variants:

```bash
docker compose up                    # build only on first run
docker compose logs -f backend       # watch model load / request logs
docker compose down                  # stop; caches survive
docker compose down -v               # also drop the HF model cache
```

Running locally without Docker (handy for development):

```bash
# backend (repo root, repo venv)
PYTHONPATH=src:app/backend TRIALO_REPO_ROOT=. uvicorn trailo_app.main:app --app-dir app/backend --port 8000
# frontend
cd app/frontend && npm install && npm run dev   # http://localhost:5173, proxies /api to 8000
```

The gated Gemma checkpoints need `HF_TOKEN` set when the base model is not
already cached (`HF_HOME` accepts an env value or the compose volume).

## Using the app

1. **Patient record** — paste a free-text clinical note. The backend normalizes
   it into the numbered `N. sentence` format the model was trained on (or load a
   bundled sample record, which also preselects its trial and carries expert
   reference annotations).
2. **Trial criteria** — pick one of 8 bundled trials (206 criteria from the
   TrialGPT-Criterion-Annotations test split), edit/delete/add criteria, or
   start a free-form criterion set typed as inclusion/exclusion.
3. **Screening results** — the model scores each criterion independently
   (strict greedy decoding, strict JSON parsing). Every criterion card shows the
   4-way verdict, the exact cited evidence sentences, the model explanation,
   parse validity, and the raw output; bundled samples additionally show the
   expert reference annotation for comparison.
4. **Export report** — human-readable eligibility screening report (run summary,
   aggregate verdict table, criterion-by-criterion findings with quoted
   evidence) as downloadable Markdown, printable HTML, or server-rendered PDF.
   Every export carries the educational-disclaimer prominently.

## Decode benchmark view

The *Decode benchmark* tab runs the current screening workload through the three
implemented greedy mechanisms and reports wall time, tokens/second, peak-memory
delta, and per-criterion **token-stream agreement** — the paths are engineered to
produce identical streams, so divergence is made visible rather than assumed.
Because `recompute` is O(sequence²) per token, the workload is capped (defaults:
4 criteria, 48 new tokens; both adjustable in the UI).

The speculative-decoding panel shows the recorded A100 benchmark (E4B target +
E2B draft): γ-sweep latency, acceptance rates, and the knife-edge-tie agreement
forensics — clearly labeled as a pre-recorded GPU run, since the live E4B/E2B
loop needs a large GPU.

## API overview

| Method & path | Purpose |
|---|---|
| `GET /api/health` | model status, device, loaded preset |
| `GET /api/library` | bundled trials + sample notes + expert references |
| `POST /api/normalize-note` | note → numbered-sentence preview |
| `POST /api/screen` | batch screening (all criteria, one response) |
| `POST /api/screen/stream` | SSE streaming screening (result per criterion) |
| `POST /api/benchmark/decode` | live generate/cache/recompute comparison |
| `GET /api/benchmark/speculative` | recorded A100 speculative results |
| `POST /api/report` | Markdown eligibility report |
| `POST /api/report/html` | printable HTML |
| `POST /api/report/pdf` | server-rendered PDF |

## Limitations (read before teaching with this)

- The default `final_prod` model is the same 270m backbone used in the smoke
  runs — a small model. Its labels are better than random but imperfect; always
  study the quoted evidence and the expert-reference comparison before drawing
  conclusions.
- The `smoke_cpu` preset is a 3-step-trained adapter from the CPU smoke run:
  pipeline-proof, not a classifier. Point TRIALO_MODEL_CONFIG at it only to
  exercise the workflow.
- Retrieval/RAG is explicitly out of scope of the research repo; the app
  annotates criteria against the pasted note only.
- Automated note normalization is heuristic. Sentence numbering directly gates
  evidence citation, so always sanity-check the numbered preview.
- The backend enforces no authentication and holds no data after the response;
  the frontend sends patient text only to the locally running backend.
