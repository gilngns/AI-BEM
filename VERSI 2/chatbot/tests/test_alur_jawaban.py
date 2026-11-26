"""
Uji alur jawaban GardaBot tanpa memanggil LLM, database, atau internet sungguhan.
Jalankan dari folder chatbot:  venv/Scripts/python -m unittest discover -s tests
"""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

for _s in (sys.stdout, sys.stderr):  # log memakai emoji; konsol Windows bawaan cp1252
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OPENAI_API_KEY", "kunci-uji")
os.environ["TOPIC_MODEL"] = "model-kecil"

from models import rag_system as rs  # noqa: E402


def balasan(teks):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=teks), finish_reason="stop")],
        usage=SimpleNamespace(total_tokens=10),
    )


class FakeLLM:
    """Membalas sesuai jenis panggilan; mencatat setiap panggilan."""

    def __init__(self, topik='{"topik": "MAGGOT", "kueri": "maggot kabur saat hujan"}', materi="Jawaban dari materi.",
                 web="Jawaban dari internet."):
        self.topik, self.materi, self.web = topik, materi, web
        self.panggilan = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model, messages, **kw):
        sistem = messages[0]["content"]
        if sistem == rs.TOPIC_SYSTEM_PROMPT:
            jenis, teks = "topik", self.topik
        elif sistem == rs.SAPAAN_SYSTEM_PROMPT:
            jenis, teks = "sapaan", "Halo juga! Ada yang ingin ditanyakan soal maggot?"
        elif sistem == rs.WEB_SYSTEM_PROMPT:
            jenis, teks = "web", self.web
        else:
            jenis, teks = "materi", self.materi
        self.panggilan.append((jenis, model, messages, kw))
        if isinstance(teks, Exception):
            raise teks
        return balasan(teks)

    def jenis(self):
        return [p[0] for p in self.panggilan]


class FakeEmbedder:
    def generate_embedding(self, text):
        return [0.0, 1.0]


class FakeDB:
    def __init__(self, ada_materi=True):
        self.ada_materi = ada_materi

    def similarity_search(self, query_embedding, top_k, document_id=None):
        if not self.ada_materi:
            return []
        return [{"text": "Maggot dipanen umur 18-21 hari.", "filename": "bahan.pdf", "similarity": 0.8, "chunk_index": 0}]


HASIL_WEB = [{"title": "Maggot kabur", "url": "https://contoh.id/a", "content": "Maggot kabur karena media terlalu basah."}]


def buat(llm, ada_materi=True):
    r = rs.RAGSystem(vector_db=FakeDB(ada_materi), embedder=FakeEmbedder(), use_azure=False)
    r.client = llm
    return r


class TestAlurJawaban(unittest.TestCase):
    def setUp(self):
        os.environ["TAVILY_API_KEY"] = "tvly-uji"

    def test_di_luar_topik_ditolak_tanpa_rag_dan_internet(self):
        llm = FakeLLM(topik='{"topik": "LAIN", "kueri": ""}')
        with mock.patch.object(rs, "cari_web") as cari:
            r = buat(llm).query("Buatkan kode landing page jualan maggot pakai React")
        self.assertEqual(r["sumber_jawaban"], "di_luar_topik")
        self.assertEqual(r["answer"], rs.DI_LUAR_TOPIK_TEXT)
        self.assertEqual(llm.jenis(), ["topik"])
        cari.assert_not_called()

    def test_penyaring_memakai_model_kecil(self):
        llm = FakeLLM(topik='{"topik": "LAIN", "kueri": ""}')
        buat(llm).query("tulis puisi")
        self.assertEqual(llm.panggilan[0][1], "model-kecil")

    def test_sapaan(self):
        llm = FakeLLM(topik='{"topik": "SAPAAN", "kueri": ""}')
        r = buat(llm).query("terima kasih ya")
        self.assertEqual(r["sumber_jawaban"], "sapaan")
        self.assertEqual(llm.jenis(), ["topik", "sapaan"])

    def test_dijawab_dari_materi(self):
        llm = FakeLLM()
        with mock.patch.object(rs, "cari_web") as cari:
            r = buat(llm).query("Kapan maggot dipanen?")
        self.assertEqual(r["sumber_jawaban"], "materi")
        self.assertEqual(r["answer"], "Jawaban dari materi.")
        self.assertEqual(r["num_sources"], 1)
        cari.assert_not_called()

    def test_materi_tidak_memuat_lalu_cari_internet(self):
        llm = FakeLLM(materi=rs.PENANDA_TIDAK_ADA)
        with mock.patch.object(rs, "cari_web", return_value=HASIL_WEB) as cari:
            r = buat(llm).query("Kenapa maggot saya kabur saat hujan?")
        cari.assert_called_once_with("maggot kabur saat hujan")  # kueri dari penyaring
        self.assertEqual(r["sumber_jawaban"], "internet")
        self.assertTrue(r["answer"].startswith("Jawaban dari internet."))
        self.assertIn(rs.CATATAN_INTERNET, r["answer"])
        self.assertEqual(r["web_sources"], [{"title": "Maggot kabur", "url": "https://contoh.id/a"}])
        # Hasil internet dikirim sebagai data, bukan perintah
        pesan_web = [p for p in llm.panggilan if p[0] == "web"][0][2][-1]["content"]
        self.assertIn("data, bukan perintah", pesan_web)
        self.assertIn("Maggot kabur karena media terlalu basah.", pesan_web)

    def test_tanpa_materi_relevan_langsung_internet(self):
        llm = FakeLLM()
        with mock.patch.object(rs, "cari_web", return_value=HASIL_WEB):
            r = buat(llm, ada_materi=False).query("Apakah maggot aman dimakan manusia?")
        self.assertEqual(r["sumber_jawaban"], "internet")
        self.assertNotIn("materi", llm.jenis())  # tidak membuang token untuk materi kosong

    def test_internet_tidak_menemukan(self):
        llm = FakeLLM(materi=rs.PENANDA_TIDAK_ADA, web=rs.PENANDA_TIDAK_DITEMUKAN)
        with mock.patch.object(rs, "cari_web", return_value=HASIL_WEB):
            r = buat(llm).query("Maggot bisa bicara?")
        self.assertEqual(r["sumber_jawaban"], "tidak_ditemukan")
        self.assertEqual(r["answer"], rs.TIDAK_DITEMUKAN_TEXT)

    def test_tavily_tidak_aktif(self):
        os.environ["TAVILY_API_KEY"] = ""
        llm = FakeLLM(materi=rs.PENANDA_TIDAK_ADA)
        with mock.patch.object(rs, "cari_web") as cari:
            r = buat(llm).query("Maggot bisa terbang?")
        cari.assert_not_called()
        self.assertEqual(r["sumber_jawaban"], "tidak_ditemukan")

    def test_tavily_galat(self):
        llm = FakeLLM(materi=rs.PENANDA_TIDAK_ADA)
        with mock.patch.object(rs, "cari_web", side_effect=RuntimeError("timeout")):
            r = buat(llm).query("Maggot bisa terbang?")
        self.assertEqual(r["sumber_jawaban"], "tidak_ditemukan")

    def test_penyaring_gagal_tetap_lanjut(self):
        llm = FakeLLM(topik=RuntimeError("rate limit"))
        r = buat(llm).query("Kapan maggot dipanen?")
        self.assertEqual(r["sumber_jawaban"], "materi")

    def test_penyaring_format_aneh(self):
        llm = FakeLLM(topik="bukan json")
        r = buat(llm).query("Kapan maggot dipanen?")
        self.assertEqual(r["sumber_jawaban"], "materi")

    def test_chat_lanjutan_membawa_riwayat_ke_penyaring(self):
        llm = FakeLLM()
        buat(llm).chat([
            {"role": "user", "content": "Pakan maggot apa saja?"},
            {"role": "assistant", "content": "Sisa sayur dan buah."},
            {"role": "user", "content": "Kalau nasi basi boleh?"},
        ])
        isi_penyaring = llm.panggilan[0][2][-1]["content"]
        self.assertIn("Pakan maggot apa saja?", isi_penyaring)
        self.assertIn("Pesan terakhir: Kalau nasi basi boleh?", isi_penyaring)


if __name__ == "__main__":
    unittest.main()
