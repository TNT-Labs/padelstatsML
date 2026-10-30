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
# 4. finds the host's HailoRT (library, Python binding, networks), which the
#    worker uses: the same version as the driver, the only one the board
#    accepts. Writes it to .env with COMPOSE_FILE, rebuilds and restarts;
# 5. shows what the worker found.
set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31m%s\033[0m\n' "$*"; exit 1; }

say "1. Scheda sul bus PCIe"
if ! lspci 2>/dev/null | grep -qi hailo; then
  fail "Nessuna scheda Hailo trovata (lspci). Controlla il cavo piatto PCIe e il fermo, poi riavvia il Pi."
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
hailortcli fw-control identify | grep -E "Firmware Version|Board Name|Device Architecture"

say "4. Runtime del Pi per il worker"
lib_line=$(/sbin/ldconfig -p | awk '/libhailort\.so\./ {print $1, $NF; exit}')
[ -n "$lib_line" ] || fail "Libreria libhailort non trovata (ldconfig -p): reinstalla con sudo apt install --reinstall hailort"
lib_name=${lib_line%% *}
lib_path=$(readlink -f "${lib_line##* }")
py_version=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
py_pkg=$(python3 -c 'import hailo_platform, os; print(os.path.dirname(hailo_platform.__file__))' 2>/dev/null) \
  || fail "Modulo Python hailo_platform assente sul Pi: sudo apt install python3-hailort"
[ "$py_version" = "3.11" ] \
  || fail "Python del Pi $py_version: il worker usa Python 3.11 e non può caricare questo modulo. Manda questo messaggio."
[ -d /usr/share/hailo-models ] || sudo apt-get install -y hailo-models
echo "Libreria:      $lib_path ($lib_name)"
echo "Modulo Python: $py_pkg (Python $py_version)"
echo "Reti:          $(ls /usr/share/hailo-models | grep -E '^yolov8' | tr '\n' ' ')"

touch .env
set_env() {   # set_env NAME VALUE: replaces the line or appends it
  if grep -q "^$1=" .env; then
    sed -i "s|^$1=.*|$1=$2|" .env
  else
    printf '%s=%s\n' "$1" "$2" >> .env
  fi
}
if ! grep -q '^COMPOSE_FILE=.*docker-compose.hailo.yml' .env; then
  printf '\n# Raspberry Pi AI HAT+: il worker usa la scheda (scripts/hailo_setup.sh).\n# Senza la scheda togli la riga COMPOSE_FILE, o i container non partono.\n' >> .env
  if grep -q '^COMPOSE_FILE=' .env; then
    sed -i 's|^COMPOSE_FILE=\(.*\)|COMPOSE_FILE=\1:docker-compose.hailo.yml|' .env
  else
    set_env COMPOSE_FILE docker-compose.yml:docker-compose.hailo.yml
  fi
fi
set_env HAILO_LIB "$lib_path"
set_env HAILO_LIB_NAME "$lib_name"
set_env HAILO_PYTHON "$py_pkg"
grep -E '^(COMPOSE_FILE|HAILO_)' .env
docker compose up -d --build

say "5. Cosa ha trovato il worker"
sleep 15
docker exec padel-worker python -c 'import hailo_platform; print("HailoRT nel worker: ok")' \
  || fail "Il worker non carica HailoRT: manda l'errore qui sopra."
docker exec -i padel-worker python - <<'PY'
from app.core.config import get_settings
from app.ml.accel import recorded_status

status = recorded_status(get_settings().data_path) or {}
print(("IN USO      " if status.get("usable") else "NON IN USO  ") + str(status.get("reason")))
PY
echo
echo "Se l'acceleratore risulta NON IN USO, copia l'output di questo script e mandalo."
