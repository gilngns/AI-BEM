"""
Chatbot API Server
REST API for PDF upload and RAG chatbot (jawaban dalam Bahasa Indonesia)
"""

import os
import sys
import time
import uuid
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import traceback

# Windows console/redirected output defaults to cp1252, which crashes on the
# emoji used in log messages (UnicodeEncodeError) -> force UTF-8
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Add this directory to path so `utils` / `models` resolve from any CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.pdf_processor import PDFProcessor
from utils.text_processor import TextChunker, EmbeddingGenerator
from utils.azure_embeddings import AzureOpenAIEmbedding
from models.vector_db import VectorDatabase
from models.rag_system import RAGSystem

# Load environment variables
load_dotenv()

# Configuration
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./storage/uploads")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE", 50))
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024  # MB to bytes
ALLOWED_EXTENSIONS = {'pdf'}

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Flask configuration for large uploads
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['PREFERRED_URL_SCHEME'] = 'http'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize components (lazy loading)
pdf_processor = None
text_chunker = None
embedder = None
vector_db = None
rag_system = None
_init_lock = threading.Lock()


def init_components():
    """Initialize all components (call on first request, thread-safe)"""
    global pdf_processor, text_chunker, embedder, vector_db, rag_system

    if rag_system is not None:
        return

    with _init_lock:
        if rag_system is not None:
            return

        print("🚀 Initializing chatbot components...")

        # PDF Processor with OCR
        tesseract_cmd = os.getenv("TESSERACT_CMD", None)
        ocr_lang = os.getenv("OCR_LANG", "eng+ind")  # English + Indonesian
        _pdf_processor = PDFProcessor(tesseract_cmd=tesseract_cmd, lang=ocr_lang)

        # Text Chunker
        chunk_size = int(os.getenv("CHUNK_SIZE", 500))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 50))
        _text_chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Embedder - Check if using Azure OpenAI
        use_azure = os.getenv("USE_AZURE_OPENAI", "False").lower() == "true"

        if use_azure:
            print("🔵 Using Azure OpenAI embeddings...")
            _embedder = AzureOpenAIEmbedding()
        else:
            print("🟢 Using Sentence Transformers embeddings...")
            st_model = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
            _embedder = EmbeddingGenerator(model_name=st_model)

        # Vector Database - dimension must match the embedder output
        vector_dimension = int(os.getenv("VECTOR_DIMENSION", _embedder.dimension))
        if vector_dimension != _embedder.dimension:
            raise ValueError(
                f"VECTOR_DIMENSION ({vector_dimension}) tidak sama dengan dimensi embedding "
                f"model ({_embedder.dimension}). Sesuaikan VECTOR_DIMENSION di file .env."
            )
        _vector_db = VectorDatabase(vector_dimension=vector_dimension)
        if not _vector_db.connect():
            raise ConnectionError("Gagal terhubung ke database")

        # RAG System - will auto-detect Azure chatbot settings
        temperature = float(os.getenv("TEMPERATURE", 0.7))
        max_tokens = int(os.getenv("MAX_TOKENS", 1000))

        _rag_system = RAGSystem(
            vector_db=_vector_db,
            embedder=_embedder,
            temperature=temperature,
            max_tokens=max_tokens,
            use_azure=use_azure  # Will use Azure chatbot if True
        )

        # Publish only after everything succeeded (no half-initialized state)
        pdf_processor = _pdf_processor
        text_chunker = _text_chunker
        embedder = _embedder
        vector_db = _vector_db
        rag_system = _rag_system

        print("✅ All components initialized!")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _remove_file(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError as e:
        print(f"⚠️  Could not remove {path}: {e}")


def _parse_query_options(data):
    """Validate and convert optional query parameters. Returns (options, error)."""
    try:
        top_k = int(data.get('top_k', int(os.getenv("TOP_K_RESULTS", 5))))
        min_similarity = data.get('min_similarity', None)
        if min_similarity is not None:
            min_similarity = float(min_similarity)
        document_id = data.get('document_id', None)
        if document_id is not None:
            document_id = int(document_id)
    except (TypeError, ValueError):
        return None, "Parameter top_k, min_similarity, atau document_id tidak valid"

    if not 1 <= top_k <= 50:
        return None, "top_k harus antara 1 dan 50"

    return {"top_k": top_k, "min_similarity": min_similarity, "document_id": document_id}, None



# ==================== DOKUMENTASI API (SWAGGER) ====================
OPENAPI_SPEC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'openapi.yaml')

SWAGGER_UI_HTML = """<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>%(title)s</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui.css">
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui-bundle.js"></script>
  <script>
    window.ui = SwaggerUIBundle({
      url: "/openapi.yaml",
      dom_id: "#swagger-ui",
      deepLinking: true,
      tryItOutEnabled: true
    });
  </script>
</body>
</html>"""


@app.route('/openapi.yaml', methods=['GET'])
def openapi_spec():
    """OpenAPI specification (YAML)"""
    import re
    with open(OPENAPI_SPEC_PATH, encoding='utf-8') as f:
        spec = f.read()
    # Point "Try it out" at the host the docs were opened from (localhost,
    # LAN IP for testing from a phone, ngrok URL, ...)
    base_url = request.host_url.rstrip('/')
    spec = re.sub(r'(servers:\s*\n\s*- url: )\S+', lambda m: m.group(1) + base_url, spec, count=1)
    return app.response_class(spec, mimetype='application/yaml')


@app.route('/docs', methods=['GET'])
@app.route('/docs/', methods=['GET'])
def swagger_ui():
    """Swagger UI - dokumentasi API interaktif"""
    return app.response_class(SWAGGER_UI_HTML % {'title': 'Maggot BSF Chatbot API - Dokumentasi'}, mimetype='text/html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Maggot BSF Chatbot API",
        "version": "1.0.0"
    })


@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    """
    Upload and process PDF file

    Returns:
        document_id and processing stats
    """
    start_time = time.time()
    filepath = None

    try:
        init_components()

        # Check if file is present
        if 'file' not in request.files:
            return jsonify({"error": "Tidak ada file yang dikirim"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "Tidak ada file yang dipilih"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Tipe file tidak valid. Hanya file PDF yang diperbolehkan"}), 400

        # secure_filename can strip everything (e.g. non-latin names) -> fallback.
        # Prefix with a unique id so uploads with the same name don't overwrite
        # each other (and a failed upload doesn't delete an earlier file).
        filename = secure_filename(file.filename)
        if not filename.lower().endswith('.pdf'):
            filename = "document.pdf"
        stored_name = f"{uuid.uuid4().hex[:8]}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, stored_name)

        print(f"📄 Processing uploaded file: {filename}")
        print(f"   Saving file...")

        try:
            file.save(filepath)
        except Exception as e:
            print(f"❌ Error saving file: {e}")
            return jsonify({"error": f"Gagal menyimpan file: {str(e)}"}), 500

        file_size = os.path.getsize(filepath)
        print(f"   File size: {file_size / 1024:.2f} KB")

        # Check if should skip OCR (for faster processing)
        skip_ocr = os.getenv("SKIP_OCR", "false").lower() == "true"

        # Process PDF with OCR
        if skip_ocr:
            print(f"   ⚡ Fast mode: Extracting text only (no OCR)...")
        else:
            print(f"   Extracting text with OCR (this may take a while)...")

        try:
            pdf_result = pdf_processor.process_pdf_full(filepath, skip_ocr=skip_ocr)
        except Exception as e:
            print(f"❌ Error processing PDF: {e}")
            _remove_file(filepath)
            return jsonify({"error": f"Gagal memproses PDF: {str(e)}"}), 500

        if not pdf_result["combined_text"].strip():
            _remove_file(filepath)
            return jsonify({
                "error": "Tidak ada teks yang dapat diekstrak dari PDF. "
                         "Jika PDF berupa hasil scan, pastikan OCR aktif (SKIP_OCR=false)."
            }), 400

        # Chunk text
        print(f"   Chunking text ({len(pdf_result['combined_text'])} chars)...")
        try:
            chunks = text_chunker.chunk_text(
                text=pdf_result["combined_text"],
                metadata={
                    "filename": filename,
                    "num_images": pdf_result["num_images"]
                }
            )
            print(f"   Created {len(chunks)} chunks")
        except Exception as e:
            print(f"❌ Error chunking text: {e}")
            _remove_file(filepath)
            return jsonify({"error": f"Gagal memecah teks: {str(e)}"}), 500

        if not chunks:
            _remove_file(filepath)
            return jsonify({"error": "Dokumen tidak menghasilkan potongan teks"}), 400

        # Generate embeddings
        print(f"   Generating embeddings for {len(chunks)} chunks...")
        try:
            chunks_with_embeddings = embedder.embed_chunks(chunks)
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            _remove_file(filepath)
            return jsonify({"error": f"Gagal membuat embedding: {str(e)}. Periksa koneksi Azure OpenAI"}), 500

        # Store in database
        print(f"   Storing in database...")
        doc_id = vector_db.insert_document(
            filename=filename,
            filepath=filepath,
            file_size=file_size,
            total_chunks=len(chunks_with_embeddings),
            metadata={
                "num_images": pdf_result["num_images"],
                "text_length": len(pdf_result["full_text"]),
                "ocr_text_length": len(pdf_result.get("ocr_text", ""))
            },
            full_text=pdf_result["combined_text"]
        )

        if doc_id is None:
            _remove_file(filepath)
            return jsonify({"error": "Gagal menyimpan dokumen ke database"}), 500

        if not vector_db.insert_chunks(doc_id, chunks_with_embeddings):
            # Don't leave a document record without chunks behind
            try:
                vector_db.delete_document(doc_id)
            except Exception:
                pass
            _remove_file(filepath)
            return jsonify({"error": "Gagal menyimpan potongan teks (chunks) ke database"}), 500

        processing_time = time.time() - start_time
        print(f"✅ Document processed successfully! ID: {doc_id}")
        print(f"   Processing time: {processing_time:.2f}s")

        return jsonify({
            "success": True,
            "document_id": doc_id,
            "filename": filename,
            "file_size": file_size,
            "total_chunks": len(chunks_with_embeddings),
            "processing_time": f"{processing_time:.2f}s",
            "metadata": {
                "num_images": pdf_result["num_images"],
                "text_length": len(pdf_result["full_text"]),
                "ocr_text_length": len(pdf_result.get("ocr_text", ""))
            }
        }), 201

    except Exception as e:
        print(f"❌ Error processing upload: {e}")
        traceback.print_exc()
        _remove_file(filepath)
        return jsonify({"error": str(e)}), 500


@app.route('/api/query', methods=['POST'])
def query_chatbot():
    """
    Query the chatbot

    Request body:
        {
            "question": "Pertanyaan Anda di sini",
            "top_k": 5,  # optional
            "document_id": 1,  # optional
            "min_similarity": 0.3  # optional
        }

    Returns:
        answer (Bahasa Indonesia) and sources
    """
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict) or 'question' not in data:
            return jsonify({"error": "Pertanyaan (question) tidak ditemukan"}), 400

        question = data['question']
        if not isinstance(question, str) or not question.strip():
            return jsonify({"error": "Pertanyaan (question) harus berupa teks dan tidak boleh kosong"}), 400

        options, error = _parse_query_options(data)
        if error:
            return jsonify({"error": error}), 400

        init_components()

        # Query RAG system
        response = rag_system.query(
            question=question.strip(),
            top_k=options["top_k"],
            document_id=options["document_id"],
            min_similarity=options["min_similarity"],
            return_sources=True
        )

        return jsonify(response), 200

    except Exception as e:
        print(f"❌ Error processing query: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Multi-turn chat conversation

    Request body:
        {
            "messages": [
                {"role": "user", "content": "Pertanyaan pertama"},
                {"role": "assistant", "content": "Jawaban pertama"},
                {"role": "user", "content": "Pertanyaan lanjutan"}
            ],
            "top_k": 5,  # optional
            "document_id": 1,  # optional
            "min_similarity": 0.3  # optional
        }
    """
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict) or 'messages' not in data:
            return jsonify({"error": "Pesan (messages) tidak ditemukan"}), 400

        messages = data['messages']
        if not isinstance(messages, list) or not messages:
            return jsonify({"error": "messages harus berupa list dan tidak boleh kosong"}), 400

        options, error = _parse_query_options(data)
        if error:
            return jsonify({"error": error}), 400

        init_components()

        # Chat with RAG system
        response = rag_system.chat(
            messages=messages,
            top_k=options["top_k"],
            document_id=options["document_id"],
            min_similarity=options["min_similarity"]
        )

        if response.get("error") and "question" not in response:
            return jsonify(response), 400

        return jsonify(response), 200

    except Exception as e:
        print(f"❌ Error processing chat: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/documents', methods=['GET'])
def list_documents():
    """List all uploaded documents"""
    try:
        init_components()

        limit = request.args.get('limit', 100, type=int)
        documents = vector_db.list_documents(limit=limit)

        return jsonify({
            "documents": documents,
            "count": len(documents)
        }), 200

    except Exception as e:
        print(f"❌ Error listing documents: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/documents/<int:doc_id>', methods=['GET'])
def get_document(doc_id):
    """Get document details by ID"""
    try:
        init_components()

        document = vector_db.get_document(doc_id)

        if document:
            return jsonify(document), 200
        else:
            return jsonify({"error": "Dokumen tidak ditemukan"}), 404

    except Exception as e:
        print(f"❌ Error getting document: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/documents/<int:doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a document and all its chunks"""
    try:
        init_components()

        document = vector_db.get_document(doc_id)
        if not document:
            return jsonify({"error": "Dokumen tidak ditemukan"}), 404

        if not vector_db.delete_document(doc_id):
            return jsonify({"error": "Dokumen tidak ditemukan"}), 404

        # Remove the stored PDF as well
        _remove_file(document.get("filepath"))

        return jsonify({"success": True, "message": f"Dokumen {doc_id} berhasil dihapus"}), 200

    except Exception as e:
        print(f"❌ Error deleting document: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics"""
    try:
        init_components()

        stats = vector_db.get_stats()

        return jsonify(stats), 200

    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({"error": f"File terlalu besar. Ukuran maksimal {MAX_FILE_SIZE_MB}MB"}), 413


if __name__ == '__main__':
    # Get configuration from environment
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", 5001))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"

    print("="*80)
    print("🤖 MAGGOT BSF CHATBOT API SERVER")
    print("="*80)
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🐛 Debug: {debug}")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📖 Dokumentasi API: http://localhost:{port}/docs")
    print("="*80)
    print("\n✨ Starting server...\n")

    app.run(host=host, port=port, debug=debug)
