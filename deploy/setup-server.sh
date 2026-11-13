#!/usr/bin/env bash
# Setup layanan AI Maggot (API prediksi + chatbot RAG) di VPS Ubuntu 24.04.
# Jalankan:  sudo bash setup-server.sh
# Aman dijalankan ulang: langkah yang sudah selesai akan dilewati.
#
# Kedua layanan HANYA mendengarkan di 127.0.0.1 (tidak terbuka ke internet).
# Yang memanggilnya adalah backend BE-BEM di server yang sama:
#   ML_BASE_URL=http://127.0.0.1:5000
#   CHATBOT_BASE_URL=http://127.0.0.1:5002
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/gilngns/AI-BEM.git}"
BRANCH="${BRANCH:-main}"
PDF_SOURCE="${PDF_SOURCE:-/root/2._Bahan_Ajar_Budidaya_Maggot_Black_Soldier_Fly_BSF.pdf}"

APP_USER=aibem
APP_HOME=/srv/ai-bem
APP_DIR=$APP_HOME/app
# Folder repo memakai spasi ("VERSI 2"); symlink tanpa spasi supaya systemd & venv aman.
API_DIR=$APP_HOME/api
BOT_DIR=$APP_HOME/chatbot
KNOWLEDGE_DIR=$APP_HOME/knowledge
ML_PORT=5000
CHATBOT_PORT=5002
PY_VERSION=3.10
TORCH_VERSION=2.14.1
DB_NAME=maggot_chatbot
DB_USER=aibem

step() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[!] %s\033[0m\n' "$*"; }
as_app() { sudo -u "$APP_USER" -H bash -c "cd '$APP_HOME' && $*"; }

[ "$(id -u)" -eq 0 ] || { echo "Jalankan dengan sudo: sudo bash $0"; exit 1; }

# ---------------------------------------------------------------- input
if [ ! -f "$APP_DIR/VERSI 2/chatbot/.env" ]; then
  while :; do
    read -rsp "API key Groq (OPENAI_API_KEY, diawali gsk_): " GROQ_KEY </dev/tty; echo
    [[ "$GROQ_KEY" =~ ^[A-Za-z0-9_-]{20,}$ ]] && break
    warn "Format API key tidak dikenali. Coba paste ulang."
  done
fi

# ---------------------------------------------------------------- paket
step "Memasang paket sistem (PostgreSQL + pgvector, library OpenCV)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y curl ca-certificates git openssl postgresql libgl1 libglib2.0-0t64
PG_MAJOR=$(ls /usr/lib/postgresql | sort -n | tail -1)
apt-get install -y "postgresql-$PG_MAJOR-pgvector"

step "Memasang uv (pengelola Python)"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 INSTALLER_NO_MODIFY_PATH=1 sh
fi
uv --version

# ---------------------------------------------------------------- kode
step "Membuat user aplikasi dan mengambil kode"
id "$APP_USER" >/dev/null 2>&1 || useradd --system --create-home --home-dir "$APP_HOME" --shell /bin/bash "$APP_USER"
if [ -d "$APP_DIR/.git" ]; then
  sudo -u "$APP_USER" -H git -C "$APP_DIR" pull --ff-only
else
  sudo -u "$APP_USER" -H git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi
ln -sfn "$APP_DIR/VERSI 2/api" "$API_DIR"
ln -sfn "$APP_DIR/VERSI 2/chatbot" "$BOT_DIR"
install -d -o "$APP_USER" -g "$APP_USER" "$KNOWLEDGE_DIR" "$APP_HOME/.cache"

# ---------------------------------------------------------------- python
step "Memasang Python $PY_VERSION dan dependency API prediksi"
as_app "uv python install $PY_VERSION"
[ -x "$API_DIR/.venv/bin/python" ] || as_app "uv venv --python $PY_VERSION '$API_DIR/.venv'"
as_app "uv pip install --python '$API_DIR/.venv/bin/python' -r '$API_DIR/requirements_api.lock.txt'"

step "Memasang dependency chatbot (PyTorch versi CPU, beberapa menit)"
[ -x "$BOT_DIR/.venv/bin/python" ] || as_app "uv venv --python $PY_VERSION '$BOT_DIR/.venv'"
as_app "uv pip install --python '$BOT_DIR/.venv/bin/python' --index-url https://download.pytorch.org/whl/cpu 'torch==$TORCH_VERSION'" \
  || { warn "torch $TORCH_VERSION (CPU) tidak tersedia, memakai versi CPU terbaru"; \
       as_app "uv pip install --python '$BOT_DIR/.venv/bin/python' --index-url https://download.pytorch.org/whl/cpu torch"; }
as_app "uv pip install --python '$BOT_DIR/.venv/bin/python' -r '$BOT_DIR/requirements_chatbot.lock.txt'"

# ---------------------------------------------------------------- database + env
BOT_ENV="$BOT_DIR/.env"
if [ -f "$BOT_ENV" ]; then
  step ".env chatbot sudah ada, database dan rahasia tidak diubah"
else
  step "Membuat database chatbot (pgvector) dan .env"
  DB_PASS=$(openssl rand -hex 24)
  sudo -u postgres psql -v ON_ERROR_STOP=1 -q <<SQL
DO \$\$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '$DB_USER') THEN CREATE ROLE $DB_USER LOGIN; END IF;
END \$\$;
ALTER ROLE $DB_USER WITH PASSWORD '$DB_PASS';
SQL
  sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" | grep -q 1 \
    || sudo -u postgres createdb -O "$DB_USER" "$DB_NAME"
  # Ekstensi butuh superuser; dibuat di sini supaya user aplikasi tidak perlu hak superuser.
  sudo -u postgres psql -d "$DB_NAME" -v ON_ERROR_STOP=1 -qc "CREATE EXTENSION IF NOT EXISTS vector;"

  install -o "$APP_USER" -g "$APP_USER" -m 600 /dev/null "$BOT_ENV"
  cat > "$BOT_ENV" <<ENV
# Dibuat oleh deploy/setup-server.sh pada $(date -u +%Y-%m-%dT%H:%MZ). Jangan commit file ini.
USE_AZURE_OPENAI=False
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_API_KEY=$GROQ_KEY
OPENAI_MODEL=openai/gpt-oss-120b

SENTENCE_TRANSFORMER_MODEL=paraphrase-multilingual-MiniLM-L12-v2
VECTOR_DIMENSION=384

SKIP_OCR=true
OCR_LANG=eng+ind

POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5432
POSTGRES_DB=$DB_NAME
POSTGRES_USER=$DB_USER
POSTGRES_PASSWORD=$DB_PASS

CHUNK_SIZE=800
CHUNK_OVERLAP=100
TOP_K_RESULTS=5
MIN_SIMILARITY=0.25
TEMPERATURE=0.3
MAX_TOKENS=2000

FLASK_HOST=127.0.0.1
FLASK_PORT=$CHATBOT_PORT
FLASK_DEBUG=False
UPLOAD_FOLDER=./storage/uploads
MAX_FILE_SIZE=50
ENV
fi

step "Menyiapkan tabel chatbot"
as_app "cd '$BOT_DIR' && .venv/bin/python setup_database.py"

# ---------------------------------------------------------------- service
step "Mendaftarkan service systemd (ai-predict :$ML_PORT, ai-chatbot :$CHATBOT_PORT)"
cat > /etc/systemd/system/ai-predict.service <<UNIT
[Unit]
Description=Maggot AI - API prediksi penetasan & panen
After=network.target

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$API_DIR
ExecStart=$API_DIR/.venv/bin/python -m gunicorn --workers 2 --bind 127.0.0.1:$ML_PORT --timeout 60 api_server:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

cat > /etc/systemd/system/ai-chatbot.service <<UNIT
[Unit]
Description=Maggot AI - chatbot RAG
After=network.target postgresql.service
Wants=postgresql.service

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$BOT_DIR
Environment=HF_HOME=$APP_HOME/.cache/huggingface
Environment=TIKTOKEN_CACHE_DIR=$APP_HOME/.cache/tiktoken
# 1 worker (model embedding cukup dimuat sekali), beberapa thread untuk request bersamaan.
# Timeout panjang karena request pertama memuat model dan memproses PDF.
ExecStart=$BOT_DIR/.venv/bin/python -m gunicorn --workers 1 --threads 4 --bind 127.0.0.1:$CHATBOT_PORT --timeout 900 chatbot_api:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable ai-predict ai-chatbot
systemctl restart ai-predict ai-chatbot

wait_for() { # url, detik
  for _ in $(seq 1 "$2"); do curl -fsS -o /dev/null "$1" 2>/dev/null && return 0; sleep 1; done; return 1
}

# ---------------------------------------------------------------- materi pdf
step "Memasukkan materi PDF ke chatbot"
[ -f "$PDF_SOURCE" ] && install -o "$APP_USER" -g "$APP_USER" -m 644 "$PDF_SOURCE" "$KNOWLEDGE_DIR/"
wait_for "http://127.0.0.1:$CHATBOT_PORT/health" 60 || warn "Chatbot belum merespons. Cek: sudo journalctl -u ai-chatbot -n 50"
bash "$APP_DIR/deploy/ingest-pdf.sh" || true

# ---------------------------------------------------------------- cek
step "Pemeriksaan"
wait_for "http://127.0.0.1:$ML_PORT/api/health" 30 || true
check() { # label, maks karakter, argumen curl...
  local label=$1 max=$2 out; shift 2
  if out=$(curl -fsS "$@" 2>&1); then printf '%s: %s\n' "$label" "${out:0:$max}"
  else printf '\033[1;31m%s: GAGAL\033[0m %s\n' "$label" "${out:0:200}"; fi
}
check "API prediksi " 200 "http://127.0.0.1:$ML_PORT/api/health"
check "Prediksi panen" 200 -X POST "http://127.0.0.1:$ML_PORT/api/predict/panen" -H 'Content-Type: application/json' \
  -d '{"jumlah_telur_gram":10,"makanan_gram":5000}'
check "Chatbot      " 200 "http://127.0.0.1:$CHATBOT_PORT/health"
check "Tanya chatbot" 400 --max-time 180 -X POST "http://127.0.0.1:$CHATBOT_PORT/api/query" -H 'Content-Type: application/json' \
  -d '{"question":"Apa itu maggot BSF?"}'

echo
echo "Selesai. Untuk BE-BEM di server ini, isi .env backend dengan:"
echo "  ML_BASE_URL=http://127.0.0.1:$ML_PORT"
echo "  CHATBOT_BASE_URL=http://127.0.0.1:$CHATBOT_PORT"
exit 0
