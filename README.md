# 🐛 Maggot Prediction

Proyek machine learning untuk memprediksi pertumbuhan atau hasil panen maggot (larva Black Soldier Fly / BSF) menggunakan Python dan Jupyter Notebook.

## 📌 Deskripsi

Maggot BSF (*Hermetia illucens*) banyak digunakan sebagai pakan ternak dan solusi pengolahan sampah organik. Proyek ini membangun model prediksi berbasis data untuk membantu optimasi budidaya maggot.

## 🛠️ Teknologi

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

## 📁 Struktur Folder

```
└── VERSI 2/
    ├── api/        # REST API prediksi penetasan & panen (port 5000)
    ├── chatbot/    # REST API chatbot RAG berbahasa Indonesia (port 5001)
    ├── models/     # Model ML terlatih (.pkl)
    ├── scripts/    # Script training, demo, dan notebook
    ├── data/       # Dataset
    └── docs/       # Grafik evaluasi model & panduan integrasi mobile
```

## 🚀 Cara Menjalankan

Clone repo ini:
```bash
git clone https://github.com/gilngns/AI-BEM.git
```

### 1. API Prediksi (port 5000)
```bash
cd "VERSI 2/api"
pip install -r requirements_api.txt
python api_server.py
```

### 2. API Chatbot (port 5001)
Chatbot menjawab pertanyaan budidaya maggot **dalam Bahasa Indonesia** berdasarkan materi PDF
yang di-upload. Default-nya tidak butuh Azure: LLM memakai [Groq](https://console.groq.com)
(gratis, API kompatibel OpenAI) dan embedding berjalan lokal (Sentence Transformers).

Kebutuhan: Python 3.10/3.11, PostgreSQL dengan ekstensi pgvector, dan API key Groq.

```powershell
cd "VERSI 2/chatbot"
python -m venv venv
.\venv\Scripts\Activate.ps1          # Linux/macOS: source venv/bin/activate
pip install -r requirements_chatbot.txt
copy .env.example .env               # lalu isi OPENAI_*, POSTGRES_PASSWORD, dll.
python setup_database.py             # membuat database & tabel
python chatbot_api.py
```

Contoh `.env` minimal untuk mode gratis:
```env
USE_AZURE_OPENAI=False
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_API_KEY=gsk_xxx
OPENAI_MODEL=openai/gpt-oss-120b
SENTENCE_TRANSFORMER_MODEL=paraphrase-multilingual-MiniLM-L12-v2
VECTOR_DIMENSION=384
SKIP_OCR=true
```

> ⚠️ Jangan pernah meng-commit file `.env`, karena berisi API key dan password database.

### 3. Notebook (training model)
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook
```
Buka notebook di folder `VERSI 2/scripts/`.

## 📖 Dokumentasi API (Swagger)

Setelah server berjalan, buka dokumentasi interaktif di browser:

| API | Swagger UI | Spesifikasi OpenAPI |
|-----|------------|---------------------|
| Prediksi | http://localhost:5000/docs | [`VERSI 2/api/openapi.yaml`](VERSI%202/api/openapi.yaml) |
| Chatbot  | http://localhost:5001/docs | [`VERSI 2/chatbot/openapi.yaml`](VERSI%202/chatbot/openapi.yaml) |

Endpoint bisa dicoba langsung dari halaman Swagger lewat tombol **Try it out**.

## 📊 Dataset & Model

- **Input fitur:** parameter lingkungan budidaya (suhu, kelembaban, pakan, dll.)
- **Target:** prediksi berat/populasi maggot
- **Algoritma:** tersedia di dalam notebook (folder VERSI 2)

## 👤 Author

**Iqbal Reza** — Universitas Komputer Indonesia (UNIKOM)

## 📄 Lisensi

Proyek ini dibuat untuk keperluan penelitian dan akademik.