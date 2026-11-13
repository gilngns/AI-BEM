#!/usr/bin/env bash
# Mengunggah PDF di /srv/ai-bem/knowledge ke chatbot, jika database chatbot masih kosong.
# Jalankan:  sudo bash /srv/ai-bem/app/deploy/ingest-pdf.sh
# Tambahkan --force untuk tetap mengunggah walau sudah ada dokumen.
set -euo pipefail

KNOWLEDGE_DIR=/srv/ai-bem/knowledge
URL=http://127.0.0.1:5002
PY=/srv/ai-bem/chatbot/.venv/bin/python

shopt -s nullglob
pdfs=("$KNOWLEDGE_DIR"/*.pdf)
if [ ${#pdfs[@]} -eq 0 ]; then
  printf '\033[1;33m[!] Belum ada PDF di %s.\033[0m\n' "$KNOWLEDGE_DIR"
  echo "    Upload PDF bahan ajar ke /root/ lewat SFTP Termius, lalu jalankan:"
  echo "    sudo install -m 644 /root/*.pdf $KNOWLEDGE_DIR/ && sudo bash $0"
  exit 1
fi

# Request pertama memuat model embedding (unduh ~470 MB saat pertama kali).
total=$(curl -fsS --max-time 600 "$URL/api/stats" | "$PY" -c 'import json,sys; print(int(json.load(sys.stdin).get("total_documents") or 0))')
if [ "$total" -gt 0 ] && [ "${1:-}" != "--force" ]; then
  echo "Chatbot sudah punya $total dokumen, tidak mengunggah ulang (pakai --force untuk memaksa)."
  exit 0
fi

for pdf in "${pdfs[@]}"; do
  echo "Mengunggah $(basename "$pdf") (bisa beberapa menit)..."
  out=$(curl -fsS --max-time 1200 -F "file=@$pdf" "$URL/api/upload") \
    || { echo "Gagal mengunggah $(basename "$pdf"). Cek: sudo journalctl -u ai-chatbot -n 80"; exit 1; }
  echo "${out:0:300}"
done
