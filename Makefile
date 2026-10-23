# Works with GNU make on Linux/macOS and on Windows (mingw32-make, choco/scoop make).
# On Windows the recipes always run under cmd.exe, even when make is launched from Git Bash.
ifeq ($(OS),Windows_NT)
SHELL := cmd.exe
PYTHON ?= python
PY ?= .venv\Scripts\python.exe
else
PYTHON ?= python3
PY ?= .venv/bin/python
endif

.PHONY: install models test demo demo-clip serve bench bench-quick plots clean

install:            ## core engine, OCR, API, benchmarks, tests (no model downloads)
	$(PYTHON) -m venv .venv
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

models:             ## CLIP / BLIP / Whisper / CLAP (downloads weights on first use)
	$(PY) -m pip install -r requirements-models.txt

test:
	$(PY) -m pytest -q

demo:               ## synthetic multimodal demo collection (offline)
	$(PY) -m omnisearch.cli demo

demo-clip:          ## the same demo with CLIP + BLIP + Whisper (needs `make models`)
	$(PY) -m omnisearch.cli demo --profile clip

serve:              ## web UI on http://127.0.0.1:8000
	$(PY) -m omnisearch.cli serve

bench:              ## all systems benchmarks at default sizes, then figures
	$(PY) -m benchmarks.run_all

bench-quick:
	$(PY) -m benchmarks.run_all --quick

plots:
	$(PY) -m benchmarks.plots

clean:
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('collections', '.pytest_cache')]"
