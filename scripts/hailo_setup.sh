#!/usr/bin/env bash
# Prepare the Raspberry Pi 5 for the AI HAT+ (Hailo) and give it to the
# analysis worker. Run on the Pi, from the repository folder:
#
#   bash scripts/hailo_setup.sh
#
# 1. checks that the board is on the PCIe bus;
# 2. installs the Hailo driver and runtime on the host (hailo-all) if missing
#    — a reboot is then needed, and the script asks to be run again;
# 3. checks that the board answers;
# 4. adds docker-compose.hailo.yml to .env (COMPOSE_FILE), rebuilds and
#    restarts the containers;
# 5. shows what the worker found.
set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

say "1. Scheda sul bus PCIe"
if ! lspci 2>/dev/null | grep -qi hailo; then
  echo "Nessuna scheda Hailo trovata (lspci). Controlla il cavo piatto PCIe e che la scheda"
  echo "sia ben fissata, poi riavvia il Pi. L'SSD resta su USB: la porta PCIe è una sola."
  exit 1
fi
lspci | grep -i hailo

say "2. Driver e runtime Hailo sul Pi"
if ! command -v hailortcli >/dev/null 2>&1 || [ ! -e /dev/hailo0 ]; then
  sudo apt-get update
  sudo apt-get install -y hailo-all
  echo
  echo "Installato. Riavvia il Pi (sudo reboot) e lancia di nuovo questo script."
  exit 0
fi
echo "HailoRT sul Pi: $(hailortcli --version 2>/dev/null | head -1)"

say "3. La scheda risponde"
hailortcli fw-control identify

say "4. Accesso del worker alla scheda"
touch .env
if grep -q '^COMPOSE_FILE=' .env; then
  grep -q 'docker-compose.hailo.yml' .env \
    || sed -i 's|^COMPOSE_FILE=\(.*\)|COMPOSE_FILE=\1:docker-compose.hailo.yml|' .env
else
  printf '\n# Raspberry Pi AI HAT+: il worker usa la scheda (scripts/hailo_setup.sh).\n# Senza la scheda togli questa riga, o i container non partono.\nCOMPOSE_FILE=docker-compose.yml:docker-compose.hailo.yml\n' >> .env
fi
grep '^COMPOSE_FILE=' .env
docker compose up -d --build

say "5. Cosa ha trovato il worker"
sleep 15
docker exec padel-worker python - <<'PY'
from app.core.config import get_settings
from app.ml.accel import recorded_status

status = recorded_status(get_settings().data_path) or {}
print(("IN USO      " if status.get("usable") else "NON IN USO  ") + str(status.get("reason")))
PY
docker exec padel-worker sh -c \
  'command -v hailortcli >/dev/null && echo "HailoRT nel container: $(hailortcli --version | head -1)" || echo "HailoRT assente nel container"'
echo
echo "Le versioni di HailoRT sul Pi e nel container devono coincidere. Se l'acceleratore"
echo "risulta NON IN USO, copia l'output di questo script e mandalo."
