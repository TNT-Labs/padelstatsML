# Percorso senza Docker: installazione, avvio e verifica, in locale o sul Pi.
#
#   make install     venv + dipendenze Python e Node + build della UI
#   make run         API in foreground con reload (sviluppo)
#   make worker      worker di analisi (in un altro terminale)
#   make test        tutti i test
#
# Tutto gira in un virtualenv (.venv). Non è una preferenza: Raspberry Pi OS
# Bookworm segue la PEP 668 e rifiuta `pip install` nell'interprete di
# sistema, quindi senza venv l'installazione fallisce con
# "externally-managed-environment".
#
# In produzione sul Pi non si usa `make run`: si installano i due servizi
# systemd in deploy/systemd (vedi INSTALL_RASPBERRY.md), che eseguono gli
# stessi comandi senza reload e con riavvio automatico.

SHELL := /bin/bash
VENV ?= .venv
PY := $(VENV)/bin/python
HOST ?= 0.0.0.0
PORT ?= 8000

.PHONY: help install venv deps web run worker test test-backend test-frontend check clean

help: ## Mostra questo elenco
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: deps web ## Installa tutto e compila la UI
	@echo
	@echo "Fatto. Prima di avviare:"
	@echo "  1. cp .env.example .env   e imposta DATA_DIR e API_BASE_URL"
	@echo "  2. esporta il modello:    make -s model-help"
	@echo "  3. make run   (e in un altro terminale: make worker)"

venv: $(VENV)/bin/python

$(VENV)/bin/python:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

deps: venv ## Dipendenze Python (versioni pinnate, le stesse della CI)
	$(PY) -m pip install -r backend/requirements.txt

# La UI compilata deve finire in backend/web: è lì che FastAPI la cerca
# (app/main.py, WEB_ROOT). Nell'immagine Docker ci arriva dallo stage di
# build; senza Docker la copia la fa questo target.
web: ## Compila il frontend dentro backend/web
	cd frontend && npm install --no-audit --no-fund && npm run build
	rm -rf backend/web
	cp -r frontend/dist backend/web
	@echo "UI compilata in backend/web"

run: ## Avvia l'API con reload (sviluppo)
	cd backend && ../$(PY) -m uvicorn app.main:app --host $(HOST) --port $(PORT) --reload

worker: ## Avvia il worker di analisi
	cd backend && ../$(PY) -m app.worker.runner

check: ## Verifica che l'API risponda e che il modello sia al suo posto
	@curl -fsS http://localhost:$(PORT)/api/health | $(PY) -m json.tool || \
		echo "L'API non risponde su http://localhost:$(PORT)"

model-help: ## Come esportare il modello ONNX del detector
	@echo "Il Pi esegue l'inferenza con onnxruntime e non installa PyTorch."
	@echo "Esporta il modello una volta sola (anche da un PC, il file è portabile):"
	@echo
	@echo "  python3 -m venv /tmp/export && source /tmp/export/bin/activate"
	@echo "  pip install -r backend/requirements.export.txt"
	@echo "  python backend/scripts/export_yolo_onnx.py --imgsz 480 --out weights/yolov8n.onnx"
	@echo "  deactivate && rm -rf /tmp/export"
	@echo
	@echo "Poi imposta DETECTOR_MODEL nel .env (percorso assoluto o relativo a backend/)."

test: test-backend test-frontend ## Esegue tutti i test

test-backend:
	cd backend && ../$(PY) -m pytest -q

test-frontend:
	cd frontend && npm test

clean: ## Rimuove UI compilata, venv e cache
	rm -rf backend/web frontend/dist $(VENV)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf backend/.pytest_cache
