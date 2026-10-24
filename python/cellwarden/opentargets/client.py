"""Open Targets Platform GraphQL client with a file cache and an offline mode.

(docs/agents-spec.md section 6; DESIGN.md 6.2)

* ``POST`` to the GraphQL endpoint with ``requests``, a timeout, and
  ``retries`` attempts with exponential backoff on connection errors,
  timeouts, HTTP 5xx and 429.
* File cache under ``cache_dir``: one JSON file per (query, variables),
  named by the SHA-256 of the canonical JSON ``{"query": ..., "variables":
  ...}``. Entries never expire; ``refresh=True`` ignores them (and
  overwrites them with the fresh response).
* ``offline=True`` reads only the cache. A miss returns ``None`` from the
  high-level functions and is recorded in ``client.offline_misses``.
* GraphQL errors in the response raise :class:`OpenTargetsError` (they are
  not cached).

Module-level convenience functions use a default client built from
``configure(...)``; the stage passes an explicit client.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

from . import queries as Q

__all__ = [
    "OpenTargetsError",
    "OpenTargetsClient",
    "cache_key",
    "configure",
    "target_by_symbol",
    "target_by_ensembl",
    "search_entity",
]

ENTITY_NAMES = ("disease", "target", "drug")
RETRY_STATUS = {429, 500, 502, 503, 504}


class OpenTargetsError(RuntimeError):
    """The API could not be reached, returned an error, or the entity type is unknown."""


def cache_key(query: str, variables: dict[str, Any] | None) -> str:
    body = json.dumps({"query": query, "variables": variables or {}}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class OpenTargetsClient:
    def __init__(
        self,
        cache_dir: str | os.PathLike | None,
        *,
        offline: bool = False,
        timeout: float = 30.0,
        retries: int = 3,
        backoff: float = 1.0,
        endpoint: str = Q.ENDPOINT,
        refresh: bool = False,
        session: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        user_agent: str = "cellwarden (https://github.com/mukeshkbj/singlecell)",
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.offline = bool(offline)
        self.timeout = float(timeout)
        self.retries = max(0, int(retries))
        self.backoff = float(backoff)
        self.endpoint = endpoint
        self.refresh = bool(refresh)
        self.session = session
        self.sleep = sleep
        self.user_agent = user_agent
        self.cache_hits = 0
        self.cache_misses = 0
        self.network_calls = 0
        self.offline_misses: list[dict[str, Any]] = []
        self.errors: list[str] = []

    # ------------------------------------------------------------------ cache
    def _cache_path(self, key: str) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / f"{key}.json"

    def _cache_read(self, key: str) -> dict[str, Any] | None:
        p = self._cache_path(key)
        if p is None or not p.exists():
            return None
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return rec if isinstance(rec, dict) and "data" in rec else None

    def _cache_write(self, key: str, query: str, variables: dict[str, Any], data: Any) -> None:
        p = self._cache_path(key)
        if p is None:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        rec = {"query": query, "variables": variables, "fetched_utc": _utc(), "endpoint": self.endpoint, "data": data}
        tmp = p.with_name(p.name + f".tmp-{os.getpid()}")
        tmp.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, p)

    # ---------------------------------------------------------------- network
    def _post(self, query: str, variables: dict[str, Any]) -> Any:
        """One GraphQL request with retries; returns the ``data`` object."""
        payload = {"query": query, "variables": variables}
        headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": self.user_agent}
        post = self.session.post if self.session is not None else requests.post
        last: str | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                self.sleep(self.backoff * (2 ** (attempt - 1)))
            self.network_calls += 1
            try:
                r = post(self.endpoint, json=payload, headers=headers, timeout=self.timeout)
            except requests.RequestException as e:  # connection error, timeout, ...
                last = f"{type(e).__name__}: {e}"
                continue
            if r.status_code in RETRY_STATUS:
                last = f"HTTP {r.status_code}"
                continue
            if r.status_code != 200:
                raise OpenTargetsError(f"Open Targets returned HTTP {r.status_code}: {r.text[:300]}")
            try:
                body = r.json()
            except ValueError as e:
                raise OpenTargetsError(f"Open Targets returned non-JSON: {e}") from e
            if not isinstance(body, dict):
                raise OpenTargetsError("Open Targets returned an unexpected body")
            if body.get("errors"):
                msgs = "; ".join(str(e.get("message", e)) for e in body["errors"] if isinstance(e, dict))
                raise OpenTargetsError(f"GraphQL error: {msgs or body['errors']}")
            return body.get("data")
        raise OpenTargetsError(f"Open Targets unreachable after {self.retries + 1} attempt(s): {last}")

    # ------------------------------------------------------------------ query
    def query(self, query: str, variables: dict[str, Any] | None = None) -> tuple[Any, str]:
        """Return ``(data, source)`` with source ``cache`` or ``network``.

        Offline misses return ``(None, "offline-miss")`` and are recorded.
        """
        variables = dict(variables or {})
        key = cache_key(query, variables)
        if not self.refresh:
            rec = self._cache_read(key)
            if rec is not None:
                self.cache_hits += 1
                return rec["data"], "cache"
        self.cache_misses += 1
        if self.offline:
            self.offline_misses.append({"key": key, "variables": variables})
            return None, "offline-miss"
        data = self._post(query, variables)
        self._cache_write(key, query, variables, data)
        return data, "network"

    # ------------------------------------------------------------- high level
    def search_entity(self, text: str, entity: str, size: int = 10) -> list[dict[str, Any]] | None:
        """Best hits for ``text`` among ``entity`` (disease | target | drug): ``[{id, name, score, entity}]``.

        ``None`` when offline and not cached.
        """
        if entity not in ENTITY_NAMES:
            raise OpenTargetsError(f"entity must be one of {ENTITY_NAMES}, got {entity!r}")
        data, src = self.query(Q.ENTITY_SEARCH, {"text": text, "entities": [entity], "size": int(size)})
        if data is None:
            return None
        hits = ((data.get("search") or {}).get("hits")) or []
        out = []
        for h in hits:
            obj = h.get("object") or {}
            name = obj.get("approvedSymbol") or obj.get("name") or h.get("name")
            out.append({"id": h.get("id") or obj.get("id"), "name": name, "score": h.get("score"),
                        "entity": h.get("entity") or entity, "description": h.get("description")})
        return out

    def resolve_symbol(self, symbol: str) -> dict[str, Any] | None:
        """``{ensembl_id, approved_symbol, approved_name, match: exact|top}`` for a symbol, ``{}`` if no hit, ``None`` offline-miss."""
        data, _ = self.query(Q.TARGET_SEARCH, {"text": symbol})
        if data is None:
            return None
        hits = ((data.get("search") or {}).get("hits")) or []
        targets = [h for h in hits if (h.get("entity") or "target") == "target"]
        if not targets:
            return {}
        exact = [h for h in targets if ((h.get("object") or {}).get("approvedSymbol") or h.get("name") or "").upper() == symbol.upper()]
        h = exact[0] if exact else targets[0]
        obj = h.get("object") or {}
        return {"ensembl_id": obj.get("id") or h.get("id"), "approved_symbol": obj.get("approvedSymbol") or h.get("name"),
                "approved_name": obj.get("approvedName"), "match": "exact" if exact else "top"}

    def target_by_ensembl(self, ensembl_id: str, *, n_drugs: int = 50, n_diseases: int = 10) -> dict[str, Any] | None:
        """Evidence for a target (see :func:`shape_target`); ``{}`` if unknown, ``None`` offline-miss."""
        data, _ = self.query(Q.TARGET_EVIDENCE, {"id": ensembl_id, "diseases": int(n_diseases)})
        if data is None:
            return None
        t = data.get("target")
        if not t:
            return {}
        return shape_target(t, n_drugs=n_drugs)

    def target_by_symbol(self, symbol: str, **kw) -> dict[str, Any] | None:
        """Symbol -> search -> target evidence. ``{}`` when the symbol has no target, ``None`` offline-miss."""
        r = self.resolve_symbol(symbol)
        if r is None:
            return None
        if not r:
            return {}
        t = self.target_by_ensembl(r["ensembl_id"], **kw)
        if t is None:
            return None
        if t:
            t["symbol_match"] = r["match"]
            t["query_symbol"] = symbol
        return t

    def stats(self) -> dict[str, Any]:
        return {"cache_hits": self.cache_hits, "cache_misses": self.cache_misses, "network_calls": self.network_calls,
                "offline_misses": len(self.offline_misses), "offline": self.offline,
                "cache_dir": str(self.cache_dir) if self.cache_dir else None, "endpoint": self.endpoint}


_STAGE_PHASE = {"PHASE_1": 1, "PHASE_1_2": 1.5, "PHASE_2": 2, "PHASE_2_3": 2.5, "PHASE_3": 3, "PHASE_4": 4, "APPROVAL": 4}


def _known_drugs(t: dict[str, Any], ens: Any, n: int) -> tuple[list[dict[str, Any]], int]:
    """(drug x disease rows, number of drugs) from ``drugAndClinicalCandidates`` (current API, one row per
    drug with its diseases) or ``knownDrugs`` (the API before October 2026, one row per drug x disease)."""
    kd: list[dict[str, Any]] = []
    if "drugAndClinicalCandidates" in t:
        block = t.get("drugAndClinicalCandidates") or {}
        rows = sorted(block.get("rows") or [], key=lambda r: -(_STAGE_PHASE.get(str(r.get("maxClinicalStage")), 0)))
        per_drug = []
        for row in rows:
            drug = row.get("drug") or {}
            stage = row.get("maxClinicalStage")
            moa = [m.get("mechanismOfAction") for m in ((drug.get("mechanismsOfAction") or {}).get("rows") or [])
                   if m.get("mechanismOfAction")]
            diseases = [d.get("disease") for d in row.get("diseases") or [] if d.get("disease")] or [{}]
            per_drug.append([{"drug": drug.get("name"), "drug_id": drug.get("id"), "phase": _STAGE_PHASE.get(str(stage)),
                              "status": str(stage).lower() if stage else None, "disease": dis.get("name"),
                              "disease_id": dis.get("id"), "mechanism": moa[0] if moa else None,
                              "evidence_id": f"OT:{ens}:drug:{drug.get('id')}:{dis.get('id')}"} for dis in diseases])
        # every drug's first disease, then the further ones, without repeating a drug x disease pair
        seen: set[str] = set()
        for k in range(max((len(x) for x in per_drug), default=0)):
            for entries in per_drug:
                if k < len(entries) and len(kd) < n and entries[k]["evidence_id"] not in seen:
                    seen.add(entries[k]["evidence_id"])
                    kd.append(entries[k])
        return kd, int(block.get("count") or len(rows))
    for row in ((t.get("knownDrugs") or {}).get("rows")) or []:
        drug = row.get("drug") or {}
        dis = row.get("disease") or {}
        kd.append({
            "drug": drug.get("name"), "drug_id": drug.get("id"), "phase": row.get("phase"), "status": row.get("status"),
            "disease": dis.get("name"), "disease_id": dis.get("id"), "mechanism": row.get("mechanismOfAction"),
            "evidence_id": f"OT:{ens}:drug:{drug.get('id')}:{dis.get('id')}",
        })
    return kd, int((t.get("knownDrugs") or {}).get("count", len(kd)))


def shape_target(t: dict[str, Any], n_drugs: int = 50) -> dict[str, Any]:
    """Flatten a ``target`` GraphQL object to the spec's shape.

    ``{ensembl_id, approved_symbol, approved_name, biotype, tractability:[{modality,id,value}],
    known_drugs:[{drug, drug_id, phase, status, disease, disease_id, mechanism, evidence_id}],
    associated_diseases:[{efo_id, name, score, evidence_id}]}``.
    ``evidence_id`` values (``OT:<ensembl>:drug:<chembl>:<efo>`` and
    ``OT:<ensembl>:disease:<efo>``) are what the G4 agent cites.
    """
    ens = t.get("id")
    kd, n_kd = _known_drugs(t, ens, n_drugs)
    ad = []
    for row in ((t.get("associatedDiseases") or {}).get("rows")) or []:
        dis = row.get("disease") or {}
        ad.append({"efo_id": dis.get("id"), "name": dis.get("name"), "score": row.get("score"),
                   "evidence_id": f"OT:{ens}:disease:{dis.get('id')}"})
    # tractability `label` (current API) was `id` before October 2026
    tract = [{"modality": x.get("modality"), "id": x.get("label") or x.get("id"), "value": x.get("value")}
             for x in (t.get("tractability") or [])]
    return {
        "ensembl_id": ens, "approved_symbol": t.get("approvedSymbol"), "approved_name": t.get("approvedName"),
        "biotype": t.get("biotype"), "tractability": tract,
        "tractability_summary": sorted({x["modality"] for x in tract if x.get("value") and x.get("modality")}),
        "known_drugs": kd, "known_drugs_count": n_kd,
        "associated_diseases": ad, "associated_diseases_count": (t.get("associatedDiseases") or {}).get("count", len(ad)),
    }


# ---------------------------------------------------------------- module API
_DEFAULT: OpenTargetsClient | None = None


def configure(cache_dir: str | os.PathLike | None, **kw) -> OpenTargetsClient:
    """Create the module's default client (used by the convenience functions below)."""
    global _DEFAULT
    _DEFAULT = OpenTargetsClient(cache_dir, **kw)
    return _DEFAULT


def _client() -> OpenTargetsClient:
    if _DEFAULT is None:
        raise OpenTargetsError("cellwarden.opentargets is not configured; call configure(cache_dir, offline=...) first")
    return _DEFAULT


def target_by_symbol(symbol: str, **kw) -> dict[str, Any] | None:
    return _client().target_by_symbol(symbol, **kw)


def target_by_ensembl(ensembl_id: str, **kw) -> dict[str, Any] | None:
    return _client().target_by_ensembl(ensembl_id, **kw)


def search_entity(text: str, entity: str, size: int = 10) -> list[dict[str, Any]] | None:
    return _client().search_entity(text, entity, size)
