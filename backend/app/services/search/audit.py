############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# search/audit.py: first-class audit logging for outbound
#     web-search provider calls.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Record every outbound web search as a first-class audit entity.

A web search is the one place MindRouter sends user text to a THIRD PARTY, so
"what left the building, to whom, and what came back" is an audit question in
its own right — not a footnote on an inference request. Several searches have
no inference request behind them at all (the admin test query, catalog
enrichment), which is the other reason this is its own table.

DESIGN NOTES

* **Own session, always.** The row is written through
  ``get_async_db_context()`` rather than the caller's session. A caller whose
  own transaction later rolls back must not take the audit record with it, and
  a failure to log must never fail the search. This mirrors the DLP worker and
  the API-key maintenance job.

* **Never raises.** Every entry point swallows its own errors and logs them.
  An audit subsystem that can break the feature it observes is worse than no
  audit subsystem.

* **Credentials are redacted here, centrally.** Providers hand over exactly
  what they sent, including the subscription token, so no provider has to
  remember which of its own headers are secret — the rule lives in one place
  and a new provider inherits it.
"""

from __future__ import annotations

import re
import time
import uuid as _uuid
from typing import Any, Optional

from backend.app.logging_config import get_logger
from backend.app.services.search.base import (
    accepts_kwarg,
    SearchExchange,
    SearchResult,
    exchange_from_exception,
)

logger = get_logger(__name__)

# Config keys (seeded by migration 081) and their defaults.
AUDIT_ENABLED_KEY = "search.audit.enabled"
AUDIT_STORE_BODY_KEY = "search.audit.store_response_body"
AUDIT_MAX_BODY_KEY = "search.audit.max_body_chars"

DEFAULT_AUDIT_ENABLED = True
DEFAULT_STORE_BODY = True
DEFAULT_MAX_BODY_CHARS = 200_000

# Hard ceiling regardless of configuration: response_body is MEDIUMTEXT (16MB)
# and a runaway value would be a denial-of-service on the audit table itself.
ABSOLUTE_MAX_BODY_CHARS = 4_000_000

# A query longer than this is truncated for storage; the audit row records the
# search that ran, and no legitimate query approaches it.
MAX_QUERY_CHARS = 4000
MAX_ERROR_CHARS = 2000

REDACTED = "***REDACTED***"

# Param/header names whose VALUE is a credential. Matched case-insensitively
# as a substring, so "X-Subscription-Token" and "api_key" both hit.
_SECRET_NAME_RE = re.compile(
    r"(token|key|secret|password|passwd|auth|credential|cookie|session)",
    re.IGNORECASE,
)


def redact_mapping(data: Optional[dict]) -> Optional[dict]:
    """Copy ``data`` with credential-bearing values replaced.

    Redacts on the NAME, never on the value's shape: a heuristic that tried to
    spot key-looking strings would both miss short keys and mangle a query
    that happens to look like one.
    """
    if not data:
        return None
    out: dict = {}
    for k, v in data.items():
        if isinstance(k, str) and _SECRET_NAME_RE.search(k):
            out[k] = REDACTED
        elif isinstance(v, dict):
            out[k] = redact_mapping(v)
        else:
            out[k] = v
    return out


def _truncate(value: Optional[str], limit: int) -> tuple[Optional[str], bool]:
    """(value capped to limit, whether it was cut)."""
    if value is None:
        return None, False
    if limit <= 0 or len(value) <= limit:
        return value, False
    return value[:limit], True


async def load_audit_config(db) -> dict:
    """Read the audit settings, tolerating missing or malformed values."""
    from backend.app.db import crud

    def _int(raw, default):
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    try:
        enabled = bool(await crud.get_config_json(db, AUDIT_ENABLED_KEY, DEFAULT_AUDIT_ENABLED))
        store_body = bool(await crud.get_config_json(db, AUDIT_STORE_BODY_KEY, DEFAULT_STORE_BODY))
        max_body = _int(
            await crud.get_config_json(db, AUDIT_MAX_BODY_KEY, DEFAULT_MAX_BODY_CHARS),
            DEFAULT_MAX_BODY_CHARS,
        )
    except Exception:
        logger.warning("web_search_audit_config_unreadable", exc_info=True)
        return {
            "enabled": DEFAULT_AUDIT_ENABLED,
            "store_body": DEFAULT_STORE_BODY,
            "max_body_chars": DEFAULT_MAX_BODY_CHARS,
        }
    return {
        "enabled": enabled,
        "store_body": store_body,
        "max_body_chars": max(0, min(max_body, ABSOLUTE_MAX_BODY_CHARS)),
    }


def _row_status(error: Optional[BaseException], blocked: bool) -> str:
    """success / error / blocked — blocked meaning nothing was ever sent."""
    if blocked:
        return "blocked"
    return "error" if error is not None else "success"


def _results_payload(results: Optional[list]) -> Optional[list]:
    """Normalized results as plain dicts, for the JSON column."""
    if not results:
        return None
    out = []
    for r in results:
        if isinstance(r, SearchResult):
            out.append(r.to_dict())
        elif isinstance(r, dict):
            out.append(r)
        else:  # pragma: no cover — a provider returning something exotic
            out.append({"value": str(r)[:500]})
    return out


async def record_search(
    *,
    query: str,
    source: str,
    provider: str,
    exchange: Optional[SearchExchange] = None,
    error: Optional[BaseException] = None,
    latency_ms: Optional[int] = None,
    max_results: Optional[int] = None,
    user_id: Optional[int] = None,
    api_key_id: Optional[int] = None,
    request_id: Optional[int] = None,
    client_ip: Optional[str] = None,
    audit_config: Optional[dict] = None,
    screen: Any = None,
    blocked: bool = False,
) -> Optional[str]:
    """Persist one web-search audit row. Returns its search_uuid, or None.

    ``screen`` is the pre-send DLP QueryScreen when screening ran; ``blocked``
    marks a search that never reached the provider because of it — the most
    important row in the table, so it is written exactly like the others.

    Never raises: a logging failure is logged and swallowed.
    """
    try:
        from backend.app.db.models import WebSearchLog
        from backend.app.db.session import get_async_db_context

        # Config is read in the same short-lived session when not supplied.
        async with get_async_db_context() as db:
            cfg = audit_config or await load_audit_config(db)
            if not cfg.get("enabled", True):
                return None

            if exchange is None and error is not None:
                # A provider that failed before/around the HTTP call still
                # attaches whatever it had — URL, params, status, body.
                exchange = exchange_from_exception(error)
            exchange = exchange or SearchExchange()

            body = exchange.response_body if cfg.get("store_body", True) else None
            body, truncated = _truncate(body, int(cfg.get("max_body_chars", DEFAULT_MAX_BODY_CHARS)))

            stored_query, _ = _truncate(query or "", MAX_QUERY_CHARS)
            # The caller's pre-screening text, kept only when screening
            # actually changed or refused the query — an unmodified query would
            # just duplicate `query`, and the operator can switch it off.
            original_raw = None
            if screen is not None and getattr(screen, "scanned", False):
                candidate = getattr(screen, "original_query", "") or ""
                first = (screen.passes[0] if getattr(screen, "passes", None) else {}) or {}
                if candidate and candidate != (query or "") and first.get("text_stored", True):
                    original_raw, _ = _truncate(candidate, MAX_QUERY_CHARS)
            err_msg = None
            err_type = None
            if error is not None:
                err_type = type(error).__name__[:100]
                err_msg, _ = _truncate(str(error), MAX_ERROR_CHARS)

            # Built once, in a closure: the retry below differs only in
            # request_id, and two hand-maintained copies of a 20-field row is
            # how a new column ends up on one path and not the other.
            def _build(link_request: bool):
                return WebSearchLog(
                    search_uuid=str(_uuid.uuid4()),
                    source=(source or "other")[:32],
                    user_id=user_id,
                    api_key_id=api_key_id,
                    request_id=request_id if link_request else None,
                    client_ip=(client_ip or None),
                    provider=(provider or "unknown")[:32],
                    query=stored_query or "",
                    query_original=original_raw,
                    max_results=max_results,
                    request_url=(exchange.request_url or None),
                    request_params=redact_mapping(exchange.request_params),
                    request_headers=redact_mapping(exchange.request_headers),
                    status=_row_status(error, blocked),
                    http_status=exchange.http_status,
                    latency_ms=latency_ms,
                    result_count=len(exchange.results or []),
                    results=_results_payload(exchange.results),
                    response_body=body,
                    response_truncated=truncated,
                    response_meta=exchange.response_meta,
                    error_type=err_type,
                    error_message=err_msg,
                    dlp_action=(
                        screen.action
                        if screen is not None and getattr(screen, "scanned", False)
                        else None
                    ),
                    dlp_detail=(screen.audit_detail() if screen is not None else None),
                )

            row = _build(True)
            db.add(row)
            try:
                await db.commit()
            except Exception:
                # The commonest cause is the request_id FK: a caller can pass
                # the id of a request row its own uncommitted transaction
                # created. An audit row without the back-link is far better
                # than none, so retry once detached.
                await db.rollback()
                if request_id is None:
                    raise
                logger.warning("web_search_audit_fk_retry", request_id=request_id)
                row = _build(False)
                db.add(row)
                await db.commit()

            logger.info(
                "web_search_logged",
                search_uuid=row.search_uuid,
                provider=row.provider,
                source=row.source,
                status=row.status,
                dlp_action=row.dlp_action,
                http_status=row.http_status,
                latency_ms=row.latency_ms,
                results=row.result_count,
            )
            return row.search_uuid
    except Exception:
        logger.error("web_search_audit_failed", exc_info=True)
        return None


async def run_logged_search(
    db,
    query: str,
    *,
    source: str,
    max_results: Optional[int] = None,
    config: Optional[dict] = None,
    provider: Any = None,
    extra_snippets: bool = False,
    user_id: Optional[int] = None,
    api_key_id: Optional[int] = None,
    request_id: Optional[int] = None,
    client_ip: Optional[str] = None,
) -> list:
    """Run a search through the configured provider and audit the round-trip.

    Drop-in for ``provider.search(...)`` at every call site: it returns the
    same ``list[SearchResult]`` and re-raises the same exceptions, so error
    handling upstream is unchanged — the only difference is that the call is
    now on the record.

    ``config`` and ``provider`` may be passed by a caller that already loaded
    them (most have), avoiding a second config read.
    """
    from backend.app.services.search.registry import PROVIDERS, get_search_config

    if config is None:
        config = await get_search_config(db)
    if max_results is None:
        max_results = int(config.get("search.max_results", 10) or 10)

    provider_key = config.get("search.provider", "brave")
    if provider is None:
        provider = PROVIDERS.get(provider_key)
    else:
        provider_key = getattr(provider, "provider_key", provider_key)

    audit_config = await load_audit_config(db)

    if provider is None:
        err = ValueError(f"Unknown search provider: {provider_key}")
        # No screen here on purpose: this fails before any query is screened,
        # so the row records "not scanned" rather than a misleading outcome.
        await record_search(
            query=query, source=source, provider=str(provider_key), error=err,
            latency_ms=0, max_results=max_results, user_id=user_id,
            api_key_id=api_key_id, request_id=request_id, client_ip=client_ip,
            audit_config=audit_config,
        )
        raise err

    # --- Pre-send DLP screening -------------------------------------
    # Runs before ANY provider is contacted, so a blocked query never leaves
    # the building. Inert unless the operator set dlp.websearch.enabled.
    from backend.app.services.search.dlp_gate import (
        WebSearchBlockedError,
        screen_query,
    )

    screen = await screen_query(db, query)
    if not screen.allowed:
        logger.warning(
            "websearch_blocked_by_dlp",
            source=source, provider=provider_key,
            severity=screen.severity, second_severity=screen.second_severity,
            categories=screen.categories, reason=screen.reason,
        )
        # `query` records the furthest-masked form (how far screening got);
        # `query_original` records what the caller typed. status=blocked and a
        # NULL request_url are what say nothing was actually sent.
        await record_search(
            query=screen.outbound_text(), source=source, provider=provider_key,
            latency_ms=0, max_results=max_results, user_id=user_id,
            api_key_id=api_key_id, request_id=request_id, client_ip=client_ip,
            audit_config=audit_config, screen=screen, blocked=True,
        )
        raise WebSearchBlockedError(
            screen.reason or "Search blocked: the query contains sensitive data",
            screen=screen,
        )
    # From here on the REDACTED query is the query: it is what gets sent and
    # what the audit row records, so the log never holds the unredacted text.
    query = screen.query or query

    started = time.monotonic()
    try:
        kwargs: dict = {"max_results": max_results, "config": config}
        # Forwarded only when asked for AND the provider's search_exchange
        # takes it: this is the one door every surface uses, so a keyword an
        # older provider's signature lacks would break that provider for
        # every caller, not just the one that set the flag.
        if extra_snippets and accepts_kwarg(provider.search_exchange, "extra_snippets"):
            kwargs["extra_snippets"] = True
        exchange = await provider.search_exchange(query, **kwargs)
    except Exception as e:
        elapsed = int((time.monotonic() - started) * 1000)
        await record_search(
            query=query, source=source, provider=provider_key, error=e,
            latency_ms=elapsed, max_results=max_results, user_id=user_id,
            api_key_id=api_key_id, request_id=request_id, client_ip=client_ip,
            audit_config=audit_config,
        )
        raise

    elapsed = int((time.monotonic() - started) * 1000)
    await record_search(
        query=query, source=source, provider=provider_key, exchange=exchange,
        latency_ms=elapsed, max_results=max_results, user_id=user_id,
        api_key_id=api_key_id, request_id=request_id, client_ip=client_ip,
        audit_config=audit_config, screen=screen,
    )
    return exchange.results
