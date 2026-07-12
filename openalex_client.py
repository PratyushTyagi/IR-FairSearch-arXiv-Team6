"""OpenAlex client: batched DOI lookups + title-search with rapidfuzz verification.

- Polite pool: every request carries mailto=...
- Rate limit: ~10 req/s (sleep-gated; respects 429 + Retry-After).
- Retries with exponential backoff on 429/5xx.
- On-disk JSON cache keyed by lookup key (doi: or title:).
"""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional

import requests
from rapidfuzz.fuzz import token_sort_ratio

OPENALEX_WORKS = "https://api.openalex.org/works"


_punct_re = re.compile(r"[^\w\s]+", re.UNICODE)
_ws_re = re.compile(r"\s+")


def norm_doi(doi):
    if not doi:
        return None
    s = str(doi).strip().lower()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s)
    return s.strip() or None


def norm_text(t):
    if not t:
        return ""
    s = unicodedata.normalize("NFKC", str(t)).lower()
    s = _punct_re.sub(" ", s)
    s = _ws_re.sub(" ", s).strip()
    return s


@dataclass
class MatchResult:
    arxiv_id: str
    openalex_id: Optional[str]
    matched_doi: Optional[str]
    match_method: Optional[str]            # "doi" | "title" | None
    cited_by_count: Optional[int]
    first_author_institution_id: Optional[str]
    first_author_institution_name: Optional[str]
    title_similarity: Optional[float] = None
    raw: Optional[dict] = None             # for debug; not persisted by default

    def to_dict(self, include_raw=False):
        d = {
            "arxiv_id": self.arxiv_id,
            "openalex_id": self.openalex_id,
            "matched_doi": self.matched_doi,
            "match_method": self.match_method,
            "cited_by_count": self.cited_by_count,
            "first_author_institution_id": self.first_author_institution_id,
            "first_author_institution_name": self.first_author_institution_name,
            "title_similarity": self.title_similarity,
        }
        if include_raw:
            d["raw"] = self.raw
        return d


class OpenAlexClient:
    def __init__(
        self,
        email: str,
        cache_dir: str,
        rate_per_sec: float = 9.0,
        max_retries: int = 5,
        timeout: int = 30,
        doi_batch_size: int = 50,
        title_sim_threshold: float = 90.0,  # rapidfuzz returns 0-100
    ):
        self.email = email
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "doi"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "title"), exist_ok=True)
        self.rate_per_sec = rate_per_sec
        self.max_retries = max_retries
        self.timeout = timeout
        self.doi_batch_size = doi_batch_size
        self.title_sim_threshold = title_sim_threshold
        self._min_interval = 1.0 / rate_per_sec
        self._last_call_ts = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"kaggle-preproc/1.0 (mailto:{email})",
        })

    # ---------- low-level ----------
    def _throttle(self):
        now = time.monotonic()
        delta = now - self._last_call_ts
        if delta < self._min_interval:
            time.sleep(self._min_interval - delta)
        self._last_call_ts = time.monotonic()

    def _get(self, params):
        params = dict(params)
        params["mailto"] = self.email
        last_exc = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                r = self.session.get(OPENALEX_WORKS, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                last_exc = e
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                retry_after = r.headers.get("Retry-After")
                sleep_s = float(retry_after) if retry_after else (2 ** attempt)
                time.sleep(min(sleep_s, 30))
                last_exc = RuntimeError(f"http {r.status_code}: {r.text[:200]}")
                continue
            # non-retryable: 4xx other than 429
            raise RuntimeError(f"OpenAlex http {r.status_code}: {r.text[:200]}")
        raise RuntimeError(f"OpenAlex retries exhausted: {last_exc}")

    # ---------- DOI ----------
    def _doi_cache_path(self, doi_norm: str) -> str:
        # safe filename
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", doi_norm)[:200]
        return os.path.join(self.cache_dir, "doi", f"{safe}.json")

    def lookup_by_dois(self, dois: list[str]) -> dict[str, dict]:
        """Batch-resolve a list of normalized DOIs. Returns {doi_norm: work_dict}.
        Caches each (doi, work) pair to disk. Missing DOIs are absent from the
        returned dict (and a None marker is cached so we don't re-query).
        """
        out: dict[str, dict] = {}
        to_query: list[str] = []
        for d in dois:
            if not d:
                continue
            p = self._doi_cache_path(d)
            if os.path.exists(p):
                try:
                    cached = json.load(open(p))
                except json.JSONDecodeError:
                    cached = None
                if cached is None:
                    continue  # cached miss
                out[d] = cached
            else:
                to_query.append(d)

        # query in batches
        for i in range(0, len(to_query), self.doi_batch_size):
            batch = to_query[i : i + self.doi_batch_size]
            doi_filter = "doi:" + "|".join(batch)
            params = {
                "filter": doi_filter,
                "per-page": 100,
                "select": "id,doi,title,cited_by_count,authorships",
            }
            data = self._get(params)
            found = data.get("results") or []
            found_dois = set()
            for w in found:
                w_doi = norm_doi(w.get("doi"))
                if w_doi:
                    found_dois.add(w_doi)
                    out[w_doi] = w
                    try:
                        json.dump(w, open(self._doi_cache_path(w_doi), "w"))
                    except OSError:
                        pass
            # cache misses
            for d in batch:
                if d not in found_dois:
                    try:
                        json.dump(None, open(self._doi_cache_path(d), "w"))
                    except OSError:
                        pass
        return out

    # ---------- TITLE ----------
    def _title_cache_path(self, title_norm: str) -> str:
        import hashlib
        h = hashlib.sha1(title_norm.encode("utf-8")).hexdigest()[:16]
        return os.path.join(self.cache_dir, "title", f"{h}.json")

    def lookup_by_title(self, title: str) -> tuple[Optional[dict], float]:
        """Returns (work, similarity in 0-100). Returns (None, 0) if no match or low confidence."""
        if not title:
            return None, 0.0
        title_norm = norm_text(title)
        if not title_norm:
            return None, 0.0
        cache_p = self._title_cache_path(title_norm)
        if os.path.exists(cache_p):
            try:
                cached = json.load(open(cache_p))
            except json.JSONDecodeError:
                cached = None
            if cached is None:
                return None, 0.0
            return cached["work"], cached["similarity"]

        params = {
            "filter": f"title.search:{title}",
            "per-page": 5,
            "select": "id,doi,title,cited_by_count,authorships",
        }
        try:
            data = self._get(params)
        except RuntimeError as e:
            # title-search occasionally rejects odd characters; cache as miss
            try:
                json.dump(None, open(cache_p, "w"))
            except OSError:
                pass
            return None, 0.0

        results = data.get("results") or []
        best = None
        best_sim = 0.0
        for w in results:
            cand_norm = norm_text(w.get("title") or "")
            sim = token_sort_ratio(title_norm, cand_norm)
            if sim > best_sim:
                best = w
                best_sim = sim
        if best is None or best_sim < self.title_sim_threshold:
            try:
                json.dump(None, open(cache_p, "w"))
            except OSError:
                pass
            return None, best_sim
        try:
            json.dump({"work": best, "similarity": best_sim}, open(cache_p, "w"))
        except OSError:
            pass
        return best, best_sim


def first_author_first_institution(work: dict) -> tuple[Optional[str], Optional[str]]:
    """From an OpenAlex work, get (institution_id, institution_name) of the
    first author's first institution. Returns (None, None) if missing.
    """
    for a in (work.get("authorships") or []):
        if a.get("author_position") == "first":
            insts = a.get("institutions") or []
            if insts:
                inst = insts[0]
                return inst.get("id"), inst.get("display_name")
            return None, None
    # fallback: if no explicit "first", use index 0
    auths = work.get("authorships") or []
    if auths:
        insts = auths[0].get("institutions") or []
        if insts:
            inst = insts[0]
            return inst.get("id"), inst.get("display_name")
    return None, None
