# Deploy layanan AI ke VPS Ubuntu

Panduan ini memasang dua layanan dari `VERSI 2/` di VPS Ubuntu 24.04:

| Layanan | Service systemd | Alamat (hanya di dalam VPS) |
|---|---|---|
| API prediksi penetasan & panen | `ai-predict` | `http://127.0.0.1:5000` |
| Chatbot RAG | `ai-chatbot` | `http://127.0.0.1:5002` |

Keduanya **tidak terbuka ke internet**. Yang memanggil adalah backend BE-BEM di VPS yang sama, lewat:

```
ML_BASE_URL=http://127.0.0.1:5000
CHATBOT_BASE_URL=http://127.0.0.1:5002
```

Jadi tidak perlu domain, Nginx, atau HTTPS untuk layanan ini.

## Yang perlu disiapkan

- **API key Groq** (`OPENAI_API_KEY` di `VERSI 2/chatbot/.env` laptop, diawali `gsk_`)
- **PDF bahan ajar** `2._Bahan_Ajar_Budidaya_Maggot_Black_Soldier_Fly_BSF.pdf` dari root project di laptop. File ini sengaja tidak ada di GitHub karena materi milik pihak lain.

## 1. Upload PDF ke VPS

Di Termius buka **SFTP**, sambungkan ke VPS, lalu seret file PDF ke folder `/root/`.

Alternatif dari PowerShell laptop:

```powershell
scp "C:\Users\NITRO 5\Projects\MAGGOT_PREDICTION\2._Bahan_Ajar_Budidaya_Maggot_Black_Soldier_Fly_BSF.pdf" root@IP_VPS:/root/
```

## 2. Jalankan skrip setup

```bash
curl -fsSL https://raw.githubusercontent.com/gilngns/AI-BEM/main/deploy/setup-server.sh -o setup-ai.sh
sudo bash setup-ai.sh
```

Skrip hanya menanyakan API key Groq. Saat di-paste, key tidak tampil di layar; itu normal.

Lalu skrip otomatis:

1. Memasang PostgreSQL + pgvector dan library sistem untuk OpenCV
2. Memasang Python 3.10 terpisah lewat `uv`, sama dengan versi di laptop. Python bawaan Ubuntu 3.12 tidak dipakai karena library chatbot dikunci untuk 3.10.
3. Membuat dua venv dengan versi library terkunci:
   - `VERSI 2/api/requirements_api.lock.txt`: scikit-learn 1.7.2 + numpy 2.2. Model `.pkl` hanya bisa dimuat dengan kombinasi ini.
   - `VERSI 2/chatbot/requirements_chatbot.lock.txt`: sama persis dengan venv chatbot di laptop. PyTorch versi CPU dipasang terpisah, supaya tidak mengunduh paket GPU berukuran beberapa GB.
4. Membuat database `maggot_chatbot` (user `aibem`, password acak) beserta `.env` chatbot
5. Mendaftarkan service `ai-predict` dan `ai-chatbot` (gunicorn), yang otomatis jalan lagi setelah crash atau reboot
6. Memproses PDF dari `/root/` ke chatbot. Langkah ini pertama kali mengunduh model embedding sekitar 470 MB.
7. Menguji: health check, satu prediksi panen, dan satu pertanyaan ke chatbot

Total sekitar 15–25 menit, kebanyakan untuk mengunduh PyTorch dan model.

## 3. Cek hasil

Di bagian akhir output harus ada:

- `API prediksi : {"status":"healthy",...}`
- `Prediksi panen: {...}` berisi angka panen
- `Chatbot      : {"status":"healthy",...}`
- `Tanya chatbot: {...}` berisi jawaban berbahasa Indonesia

Jika PDF belum di-upload saat skrip berjalan, upload dulu ke `/root/`, lalu:

```bash
sudo install -m 644 /root/*.pdf /srv/ai-bem/knowledge/
sudo bash /srv/ai-bem/app/deploy/ingest-pdf.sh
```

## Update setelah ada perubahan kode

Push ke branch `main` repo `AI-BEM`, lalu di VPS:

```bash
sudo bash /srv/ai-bem/app/deploy/update.sh
```

## Perintah yang sering dipakai

| Keperluan | Perintah |
|---|---|
| Status | `sudo systemctl status ai-predict ai-chatbot` |
| Log prediksi | `sudo journalctl -u ai-predict -f` |
| Log chatbot | `sudo journalctl -u ai-chatbot -f` |
| Restart | `sudo systemctl restart ai-predict ai-chatbot` |
| Ubah API key / pengaturan chatbot | `sudo nano /srv/ai-bem/chatbot/.env` lalu restart `ai-chatbot` |
| Statistik dokumen chatbot | `curl -s http://127.0.0.1:5002/api/stats` |

## Pemecahan masalah

- **`ai-predict` gagal start dengan error numpy/pickle:** versi numpy/scikit-learn tidak sesuai lockfile. Jalankan `deploy/update.sh` supaya venv kembali ke versi terkunci.
- **Chatbot menjawab "layanan tidak tersedia":** cek API key Groq di `.env` dan kuotanya, lalu lihat `journalctl -u ai-chatbot`.
- **Chatbot menjawab tapi tidak sesuai materi:** dokumen belum masuk. Cek `api/stats`; jika `total_documents` 0, jalankan `ingest-pdf.sh`.
- **Request pertama chatbot lama:** wajar, karena model embedding sedang dimuat. Berikutnya cepat.
