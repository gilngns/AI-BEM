"""
Pencarian internet lewat Tavily (https://tavily.com), dipakai bila pertanyaan soal maggot /
sampah organik tidak terjawab oleh materi PDF.

Aktif hanya bila TAVILY_API_KEY diisi di .env. Hasilnya ringkas (judul, URL, cuplikan),
jadi hemat token LLM dibanding pencarian bawaan model.
"""

import os
from typing import Dict, List

import httpx

TAVILY_URL = "https://api.tavily.com/search"
MAKS_CUPLIKAN = 1200  # karakter per hasil yang dikirim ke LLM


def web_search_aktif() -> bool:
    return bool(os.getenv("TAVILY_API_KEY", "").strip())


def cari_web(query: str, max_results: int = None, timeout: float = 15.0) -> List[Dict]:
    """
    Cari di internet. Mengembalikan list {title, url, content}; list kosong bila
    pencarian tidak aktif. Galat jaringan / API dilempar ke pemanggil.
    """
    key = os.getenv("TAVILY_API_KEY", "").strip()
    if not key or not query or not query.strip():
        return []
    n = max_results or int(os.getenv("WEB_SEARCH_MAX_RESULTS", 5))

    res = httpx.post(
        TAVILY_URL,
        json={
            "query": query.strip()[:400],
            "search_depth": "basic",
            "max_results": max(1, min(n, 10)),
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        },
        headers={"Authorization": f"Bearer {key}"},
        timeout=timeout,
    )
    res.raise_for_status()

    hasil = []
    for item in (res.json() or {}).get("results") or []:
        url = item.get("url") or ""
        if not url.startswith(("https://", "http://")):
            continue
        hasil.append({
            "title": str(item.get("title") or "")[:200],
            "url": url[:500],
            "content": str(item.get("content") or "")[:MAKS_CUPLIKAN],
        })
    return hasil
