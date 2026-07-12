"""Semantic Scholar Graph API client (BATCH endpoint, arXiv-id lookup).

We resolve title-only arXiv papers via the /paper/batch endpoint with
'ARXIV:<arxiv_id>' identifiers. The batch endpoint accepts up to 500 ids
per call, works without an API key, and returns 200 reliably (no 429s
observed in testing). Fields returned: title, authors.affiliations,
authors.name, citationCount, externalIds.

Caching: per-arxiv-id JSON files under cache/s2/.
Bounded retries on 429/5xx/network exhaust → raises BatchRetryable so the
runner can mark the batch's records as 'unresolved_retryable' and move on.
"""
from __future__ import annotations

import json
import os
import time
from typing import Iterable, Optional

import requests

BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
DEFAULT_FIELDS = "title,authors.affiliations,authors.name,citationCount,externalIds"


class BatchRetryable(Exception):
    pass


class S2BatchClient:
    def __init__(
        self,
        cache_dir: str,
        api_key: Optional[str] = None,
        batch_size: int = 500,
        rate_per_sec: float = 3.0,
        max_retries: int = 2,
        retry_cap_sec: float = 8.0,
        timeout: int = 60,
    ):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.api_key = api_key
        self.batch_size = batch_size
        self.rate_per_sec = rate_per_sec
        self.max_retries = max_retries
        self.retry_cap_sec = retry_cap_sec
        self.timeout = timeout
        self._min_interval = 1.0 / rate_per_sec
        self._last_call_ts = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "kaggle-preproc/1.0 (jain.pav@northeastern.edu)",
            "Content-Type": "application/json",
        })
        if api_key:
            self.session.headers["x-api-key"] = api_key

    def _throttle(self):
        now = time.monotonic()
        delta = now - self._last_call_ts
        if delta < self._min_interval:
            time.sleep(self._min_interval - delta)
        self._last_call_ts = time.monotonic()

    def _cache_path(self, arxiv_id: str) -> str:
        # arxiv ids are safe filenames (slash-free). Replace any '/' just in case.
        safe = arxiv_id.replace("/", "_")
        return os.path.join(self.cache_dir, f"{safe}.json")

    def _read_cache(self, arxiv_id: str):
        p = self._cache_path(arxiv_id)
        if not os.path.exists(p):
            return None, False  # (data, was_cached)
        try:
            return json.load(open(p)), True
        except (json.JSONDecodeError, OSError):
            return None, True  # treat as cached null (don't re-query)

    def _write_cache(self, arxiv_id: str, data):
        try:
            json.dump(data, open(self._cache_path(arxiv_id), "w"))
        except OSError:
            pass

    def _post_batch(self, ids: list[str], fields: str = DEFAULT_FIELDS):
        body = {"ids": ids}
        params = {"fields": fields}
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                r = self.session.post(BATCH_URL, params=params, json=body, timeout=self.timeout)
            except requests.RequestException as e:
                if attempt >= self.max_retries:
                    raise BatchRetryable(f"network: {e}")
                time.sleep(min(2 ** attempt, self.retry_cap_sec))
                continue
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError as e:
                    raise BatchRetryable(f"bad json: {e}")
            if r.status_code in (429, 500, 502, 503, 504):
                if attempt >= self.max_retries:
                    raise BatchRetryable(f"http {r.status_code}: {r.text[:200]}")
                ra = r.headers.get("Retry-After")
                sleep_s = float(ra) if ra else (2 ** attempt)
                time.sleep(min(sleep_s, self.retry_cap_sec))
                continue
            # non-retryable 4xx: signal failure but don't retry
            raise BatchRetryable(f"http {r.status_code}: {r.text[:200]}")
        raise BatchRetryable("retries exhausted")

    def lookup_by_arxiv_ids(
        self, arxiv_ids: list[str]
    ) -> tuple[dict[str, Optional[dict]], list[str]]:
        """Resolve a list of arXiv IDs via batch.

        Returns:
          results: {arxiv_id: paper_dict_or_None}  (None == not-in-S2)
          unresolved_retryable: list of arxiv_ids whose batch call failed transiently
        """
        results: dict[str, Optional[dict]] = {}
        retryable: list[str] = []

        # split cached vs uncached
        uncached: list[str] = []
        for aid in arxiv_ids:
            cached, was_cached = self._read_cache(aid)
            if was_cached:
                results[aid] = cached  # may be None (cached miss) or dict
            else:
                uncached.append(aid)

        # batch the uncached
        for i in range(0, len(uncached), self.batch_size):
            batch_ids = uncached[i : i + self.batch_size]
            payload = [f"ARXIV:{a}" for a in batch_ids]
            try:
                data = self._post_batch(payload)
            except BatchRetryable:
                retryable.extend(batch_ids)
                continue
            # S2 returns a list aligned with the input ids; None for misses
            for aid, item in zip(batch_ids, data):
                if item is None:
                    self._write_cache(aid, None)
                    results[aid] = None
                else:
                    self._write_cache(aid, item)
                    results[aid] = item
        return results, retryable


def first_author_first_affiliation(paper: dict) -> tuple[Optional[str], Optional[str]]:
    """Returns (institution_string, institution_string).
    S2 has no canonical institution IDs in the paper response, so id == name.
    A later normalization step canonicalizes for cross-source grouping.
    """
    auths = paper.get("authors") or []
    if not auths:
        return None, None
    first = auths[0]
    affs = first.get("affiliations") or []
    if not affs:
        return None, None
    aff = affs[0]
    if isinstance(aff, dict):
        aff = aff.get("name") or aff.get("displayName") or None
    if not aff:
        return None, None
    aff = str(aff).strip()
    return (aff, aff) if aff else (None, None)
