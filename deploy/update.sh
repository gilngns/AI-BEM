#!/usr/bin/env bash
# Menarik versi terbaru dari GitHub, memperbarui dependency, lalu restart layanan AI.
# Jalankan:  sudo bash /srv/ai-bem/app/deploy/update.sh
set -euo pipefail

APP_USER=aibem
APP_DIR=/srv/ai-bem/app
API_DIR=/srv/ai-bem/api
BOT_DIR=/srv/ai-bem/chatbot

[ "$(id -u)" -eq 0 ] || { echo "Jalankan dengan sudo: sudo bash $0"; exit 1; }

sudo -u "$APP_USER" -H bash -c "git -C '$APP_DIR' pull --ff-only \
  && uv pip install --python '$API_DIR/.venv/bin/python' -r '$API_DIR/requirements_api.lock.txt' \
  && uv pip install --python '$BOT_DIR/.venv/bin/python' -r '$BOT_DIR/requirements_chatbot.lock.txt'"

systemctl restart ai-predict ai-chatbot
sleep 5
curl -fsS http://127.0.0.1:5000/api/health && echo
curl -fsS http://127.0.0.1:5002/health && echo
echo "Versi aktif: $(sudo -u "$APP_USER" git -C "$APP_DIR" log -1 --format='%h %s')"
