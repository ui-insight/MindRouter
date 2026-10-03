############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_worker.py: Background DLP scanning worker
#
# Drains an asyncio queue of request IDs, loads request
# data from DB, runs DLP scanners, and creates alerts.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Background DLP worker: async queue consumer for post-hoc scanning."""

import asyncio
import random
import re
import time
import zoneinfo
from datetime import datetime, timedelta, timezone

from backend.app.logging_config import get_logger

logger = get_logger(__name__)

# Module-level queue — bounded to 10,000 to prevent unbounded memory growth
_dlp_queue: asyncio.Queue = asyncio.Queue(maxsize=10_000)

# Scans dropped on queue overflow, per process — exposed via /metrics so
# coverage loss under load is observable (it used to be a log line only).
_queue_dropped_total: int = 0

# Concurrent scan consumers per worker process. One serial consumer let scan
# lag reach p95 ~77s under GLiNER bursts (measured); a few concurrent scans
# parallelize across the thread-pool executor (torch releases the GIL).
# Admin-tunable via app_config dlp.worker.concurrency, re-read every
# CONCURRENCY_CHECK_SECONDS, clamped to [1, MAX] (each GLiNER scan can pin a
# CPU thread — the cap protects the inference API sharing this process).
DLP_WORKER_DEFAULT_CONCURRENCY = 3
DLP_WORKER_MAX_CONCURRENCY = 8
CONCURRENCY_CHECK_SECONDS = 60.0

# Cap on masked snippets stored per alert (see _process_one).
MAX_STORED_ENTITIES = 50

# Alert-email flood guard: at most EMAIL_BURST messages per severity per
# EMAIL_WINDOW_SECONDS, per process.  severity -> (window_start, count, suppressed)
EMAIL_BURST = 10
EMAIL_WINDOW_SECONDS = 300.0
_email_budget: dict = {}

# Alert de-duplication (admin-toggleable — dlp.dedup.enabled / .window_seconds).
# The same sensitive value re-appears on every conversation turn and across a
# client's auxiliary model calls (web-search classifier, title generation), so
# one exposure of one SSN can mint a dozen alerts+emails in seconds.  Suppress
# a repeat alert whose (user, masked value-set) matches one already raised
# within the window.  Per-process, best-effort — a flood guard, not a ledger.
DEDUP_DEFAULT_WINDOW = 300.0
_DEDUP_MAX_KEYS = 20_000
_dedup_seen: dict = {}  # key -> first-seen monotonic time

# Scanner-failure surfacing (F53): a scanner that errors must never look like
# clean traffic.  Log at ERROR every time, and raise a visible dashboard alert
# at most once per scanner per window so a permanently-broken scanner does not
# flood while an operator is still told DLP is degraded.
SCANNER_ERROR_ALERT_WINDOW = 3600.0
_scanner_error_seen: dict = {}

# Per-severity email delivery modes (admin -> DLP panel).  Each severity picks
# how its alerts are emailed:
#   "immediate" — email now, to that severity's recipients (flood-guarded)
#   "digest"    — roll up into the periodic digest report (dlp_digest_loop)
#   "off"       — no email; the alert is still created and logged
_DELIVERY_MODES = ("immediate", "digest", "off")
_SEVERITIES = ("minor", "moderate", "major")

# Digest schedule: how often the rolled-up report is sent.
DIGEST_FREQUENCIES = {
    "hourly": 3600,
    "6h": 21600,
    "12h": 43200,
    "daily": 86400,
}
# How often the digest loop wakes to check whether a report is due.
DIGEST_CHECK_INTERVAL = 300.0
# Per-wake random extra sleep.  Every uvicorn worker runs its own digest loop
# and they all start within milliseconds of each other on a restart; without
# jitter they tick in lockstep forever and race on "is a report due?".  The
# compare-and-set claim below is what makes a duplicate impossible — jitter
# just keeps the workers from contending every tick.
DIGEST_CHECK_JITTER = 60.0
# Cap alerts enumerated in one digest email so a busy window can't produce a
# multi-megabyte message; the count line still reports the true total.
DIGEST_MAX_ROWS = 500


def _dedup_is_duplicate(key, window: float) -> bool:
    """True if ``key`` was first seen within ``window`` seconds (a duplicate).

    First (or post-expiry) sighting records the time and returns False; repeats
    inside the window return True.  One alert per value per window.
    """
    now = time.monotonic()
    if len(_dedup_seen) > _DEDUP_MAX_KEYS:
        # Bound memory: drop expired keys, then hard-clear if still oversized.
        cutoff = now - window
        for k in [k for k, t in list(_dedup_seen.items()) if t < cutoff]:
            _dedup_seen.pop(k, None)
        if len(_dedup_seen) > _DEDUP_MAX_KEYS:
            _dedup_seen.clear()
    first = _dedup_seen.get(key)
    if first is not None and (now - first) < window:
        return True
    _dedup_seen[key] = now
    return False


def get_dlp_queue() -> asyncio.Queue:
    """Return the module-level DLP queue."""
    return _dlp_queue


def get_queue_dropped_total() -> int:
    """Scans dropped on overflow in this process (for /metrics)."""
    return _queue_dropped_total


async def enqueue_for_dlp(request_id: int) -> None:
    """Enqueue a request ID for DLP scanning. Non-blocking, drops if full."""
    global _queue_dropped_total
    try:
        _dlp_queue.put_nowait(request_id)
    except asyncio.QueueFull:
        _queue_dropped_total += 1
        logger.warning("dlp_queue_full", dropped_request_id=request_id,
                       dropped_total=_queue_dropped_total)


async def _read_worker_concurrency(fallback: int = DLP_WORKER_DEFAULT_CONCURRENCY) -> int:
    """Target consumer count from app_config, clamped.

    On any error returns ``fallback`` (the caller passes its last-known-good
    target): one transient failed poll must not snap an admin-raised pool back
    to the default and cancel live consumers mid-scan.
    """
    try:
        from backend.app.db import crud
        from backend.app.db.session import get_async_db_context
        async with get_async_db_context() as db:
            raw = await crud.get_config_json(
                db, "dlp.worker.concurrency", DLP_WORKER_DEFAULT_CONCURRENCY)
        return max(1, min(DLP_WORKER_MAX_CONCURRENCY, int(raw)))
    except Exception as e:
        logger.warning("dlp_concurrency_poll_failed", error=type(e).__name__,
                       keeping=fallback)
        return fallback


async def _consume_loop(worker_idx: int) -> None:
    """One scan consumer. Exits only via cancellation (resize/shutdown)."""
    while True:
        try:
            request_id = await _dlp_queue.get()
            try:
                await _process_one(request_id)
            except Exception:
                logger.exception("dlp_process_failed", request_id=request_id)
            finally:
                _dlp_queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("dlp_worker_error", worker=worker_idx)
            await asyncio.sleep(1)


async def dlp_worker_loop() -> None:
    """DLP worker supervisor. Runs as a background task during app lifespan.

    Keeps dlp.worker.concurrency consumer tasks draining the shared queue and
    hot-resizes when the config changes (checked every CONCURRENCY_CHECK_SECONDS).
    Shrinking cancels the newest consumer; a scan cancelled mid-item is lost,
    which matches the queue's existing best-effort contract. Per-process state
    the consumers share (_dedup_seen, _email_budget, _scanner_error_seen) is
    mutated only in synchronous sections, so concurrent consumers on one event
    loop cannot interleave inside a check-and-set.
    """
    logger.info("dlp_worker_started")
    consumers: list = []
    target = DLP_WORKER_DEFAULT_CONCURRENCY
    try:
        while True:
            target = await _read_worker_concurrency(fallback=target)
            # Restart any consumer that died (should not happen — belt and braces).
            for i, t in enumerate(consumers):
                if t.done():
                    logger.error("dlp_consumer_died", worker=i)
                    consumers[i] = asyncio.create_task(_consume_loop(i))
            while len(consumers) < target:
                consumers.append(asyncio.create_task(_consume_loop(len(consumers))))
                logger.info("dlp_consumer_started", total=len(consumers))
            while len(consumers) > target:
                t = consumers.pop()
                t.cancel()
                logger.info("dlp_consumer_stopped", total=len(consumers))
            await asyncio.sleep(CONCURRENCY_CHECK_SECONDS)
    except asyncio.CancelledError:
        logger.info("dlp_worker_cancelled")
        for t in consumers:
            t.cancel()
        await asyncio.gather(*consumers, return_exceptions=True)


async def _process_one(request_id: int) -> None:
    """Process a single request for DLP scanning."""
    from backend.app.db import crud
    from backend.app.db.session import get_async_db_context
    from backend.app.services.dlp_scanner import (
        extract_scannable_text,
        run_dlp_scan,
    )

    async with get_async_db_context() as db:
        # Load DLP master toggle
        enabled = await crud.get_config_json(db, "dlp.enabled", False)
        if not enabled:
            return

        # Load the request
        from backend.app.db.models import Request, Response
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        result = await db.execute(
            select(Request).where(Request.id == request_id)
        )
        req = result.scalar_one_or_none()
        if req is None:
            return

        # No self-loop guard is needed: the LLM scanner dispatches straight to a
        # backend (_internal_chat) and never re-enters the gateway, so a scan
        # creates no request row of its own.

        # Load response content
        resp_result = await db.execute(
            select(Response).where(Response.request_id == request_id)
        )
        resp = resp_result.scalar_one_or_none()
        response_content = resp.content if resp else None

        # Extract scannable text
        text = extract_scannable_text(
            messages=req.messages,
            prompt=req.prompt,
            response_content=response_content,
            modality=req.modality if hasattr(req, "modality") else None,
        )
        if not text:
            return

        # Build config dict from DB
        config = await _load_dlp_config(db)

        # Run the scan
        scan_result = await run_dlp_scan(text, config)
        if scan_result is None:
            return

        # F53: a scanner that errored must not masquerade as clean traffic —
        # surface it (loud log + rate-limited dashboard alert) so an operator
        # knows DLP may be MISSING sensitive data.
        if scan_result.scanner_errors:
            await _surface_scanner_errors(db, req, request_id, scan_result)
            if not scan_result.findings:
                return

        # Build entities list.  Snippets are MASKED, not merely truncated: an
        # alert row is metadata about sensitive data and must not become a
        # second, longer-lived copy of it.  Cap the count so a pathological
        # match (a 1-char keyword against a large prompt) can't write a
        # multi-megabyte JSON column.
        from backend.app.services.dlp_scanner import mask_snippet

        entities = []
        categories_set = set()
        max_confidence = 0.0
        for f in scan_result.findings:
            if len(entities) < MAX_STORED_ENTITIES:
                entities.append({
                    "scanner": f.scanner,
                    "category": f.category,
                    "text": mask_snippet(f.text),
                    "confidence": f.confidence,
                })
            categories_set.add(f.category)
            if f.confidence > max_confidence:
                max_confidence = f.confidence
        if len(scan_result.findings) > MAX_STORED_ENTITIES:
            entities.append({
                "scanner": "-",
                "category": "(truncated)",
                "text": f"+{len(scan_result.findings) - MAX_STORED_ENTITIES} more",
                "confidence": 0.0,
            })

        # De-duplicate: the same user re-sending the same masked value-set within
        # the window (conversation history, or a client's classifier/title calls)
        # collapses to a single alert+email.  Admin-toggleable; window <= 0 or the
        # toggle off restores the pre-dedup one-alert-per-request behavior.
        if config.get("dedup.enabled", True) and entities:
            window = config.get("dedup.window_seconds", DEDUP_DEFAULT_WINDOW)
            try:
                window = float(window)
            except (TypeError, ValueError):
                window = DEDUP_DEFAULT_WINDOW
            if window > 0:
                masked = tuple(sorted(e["text"] for e in entities if e.get("text")))
                if masked and _dedup_is_duplicate((req.user_id, masked), window):
                    logger.info(
                        "dlp_alert_deduplicated",
                        request_id=request_id,
                        user_id=req.user_id,
                        categories=sorted(categories_set),
                    )
                    return

        # Create DLP alert
        alert = await crud.create_dlp_alert(
            db,
            request_id=request_id,
            user_id=req.user_id,
            severity=scan_result.severity,
            scanner=scan_result.scanner,
            categories=list(categories_set),
            entities=entities,
            confidence=max_confidence,
            scan_latency_ms=scan_result.scan_latency_ms,
            detail=scan_result.detail,
        )
        await db.commit()

        logger.info(
            "dlp_alert_created",
            alert_id=alert.id,
            request_id=request_id,
            severity=scan_result.severity,
            findings=len(scan_result.findings),
            latency_ms=scan_result.scan_latency_ms,
        )

        # Send email notification if configured
        await _maybe_send_email(db, alert, scan_result)


async def _load_dlp_config(db) -> dict:
    """Load all DLP configuration from the database."""
    from backend.app.db import crud

    config = {}
    config["regex.enabled"] = await crud.get_config_json(db, "dlp.regex.enabled", True)
    config["regex.patterns"] = await crud.get_config_json(db, "dlp.regex.patterns", [])
    config["regex.keywords"] = await crud.get_config_json(db, "dlp.regex.keywords", [])
    config["gliner.enabled"] = await crud.get_config_json(db, "dlp.gliner.enabled", False)
    config["gliner.threshold"] = await crud.get_config_json(db, "dlp.gliner.threshold", 0.5)
    config["gliner.categories"] = await crud.get_config_json(db, "dlp.gliner.categories", None)
    from backend.app.services.dlp_scanner import GLINER_DEFAULT_MAX_CHARS
    # Global scan window for every scanner (0 = no limit, scan the whole
    # document).  Read per scan so an admin change takes effect immediately.
    from backend.app.services.dlp_scanner import MAX_SCAN_CHARS
    config["max_scan_chars"] = await crud.get_config_json(
        db, "dlp.max_scan_chars", MAX_SCAN_CHARS
    )
    config["gliner.max_scan_chars"] = await crud.get_config_json(
        db, "dlp.gliner.max_scan_chars", GLINER_DEFAULT_MAX_CHARS
    )

    # Off-host GLiNER service (optional).  Disabled by default: on-host GLiNER
    # stays the behavior when these are absent.  fallback="local" means a remote
    # failure quietly reruns the in-process scanner; "skip" surfaces it degraded.
    config["gliner.remote.enabled"] = await crud.get_config_json(db, "dlp.gliner.remote.enabled", False)
    # Pool of off-host endpoints (scale-out + failover); legacy single .url is
    # still honored by parse_remote_endpoints when the list is empty.
    config["gliner.remote.endpoints"] = await crud.get_config_json(db, "dlp.gliner.remote.endpoints", [])
    config["gliner.remote.url"] = await crud.get_config_json(db, "dlp.gliner.remote.url", "")
    config["gliner.remote.key"] = await crud.get_config_json(db, "dlp.gliner.remote.key", "")
    config["gliner.remote.timeout"] = await crud.get_config_json(db, "dlp.gliner.remote.timeout", 10.0)
    config["gliner.remote.fallback"] = await crud.get_config_json(db, "dlp.gliner.remote.fallback", "local")
    config["gliner.remote.verify_tls"] = await crud.get_config_json(db, "dlp.gliner.remote.verify_tls", True)

    # Authoritative auto-discovery of the off-host GLiNER pool.
    #
    # When the off-host scanner is enabled but the endpoint list is empty,
    # treat healthy engine=dlp backends in the fleet registry as the pool.
    # AUTHORITATIVE health: only backends the fleet currently reports HEALTHY
    # are used, so a downed DLP node drops out on the next config load and a
    # recovered one rejoins — no manual edit. A hand-entered endpoint list
    # ALWAYS wins and bypasses discovery entirely (explicit operator override).
    # This block NEVER raises: on any error the configured endpoints are left
    # exactly as loaded so DLP scanning keeps working. Reuses the caller's db
    # session — no new connection is opened.
    try:
        from backend.app.services.dlp_scanner import parse_remote_endpoints
        manual = parse_remote_endpoints(
            config["gliner.remote.endpoints"], legacy_url=config["gliner.remote.url"]
        )
        if config["gliner.remote.enabled"] and not manual:
            from backend.app.db.models import BackendEngine
            backends = await crud.get_backends_by_engine(
                db, BackendEngine.DLP, healthy_only=True
            )
            config["gliner.remote.endpoints"] = [b.url for b in backends]
            logger.info("dlp_remote_autodiscovered", count=len(backends))
    except Exception:
        logger.exception("dlp_remote_autodiscovery_failed")

    config["llm.enabled"] = await crud.get_config_json(db, "dlp.llm.enabled", False)
    config["llm.model"] = await crud.get_config_json(db, "dlp.llm.model", "")
    config["llm.system_prompt"] = await crud.get_config_json(db, "dlp.llm.system_prompt", "")
    config["severity_rules"] = await crud.get_config_json(db, "dlp.severity_rules", {})
    # True once the admin page has saved the rule list: dlp.regex.patterns then
    # holds the built-ins too (possibly edited/removed) and is authoritative.
    config["regex.builtins_in_list"] = await crud.get_config_json(
        db, "dlp.regex.builtins_in_list", False
    )

    # Alert de-duplication (admin -> DLP panel toggle).
    config["dedup.enabled"] = await crud.get_config_json(db, "dlp.dedup.enabled", True)
    config["dedup.window_seconds"] = await crud.get_config_json(
        db, "dlp.dedup.window_seconds", int(DEDUP_DEFAULT_WINDOW)
    )

    # The LLM scanner dispatches through this callable rather than holding an
    # API key.  Nothing credential-shaped is stored for DLP anywhere.
    config["llm.complete"] = _internal_chat if config["llm.enabled"] else None

    return config


async def _surface_scanner_errors(db, req, request_id, scan_result) -> None:
    """Make a DLP scanner failure VISIBLE (F53).

    Logs at ERROR every time (for monitoring), and raises a dashboard alert at
    most once per scanner per SCANNER_ERROR_ALERT_WINDOW so an operator learns
    DLP is degraded — rather than a broken scanner silently passing sensitive
    data — without a permanently-broken scanner flooding the alert list.
    """
    from backend.app.db import crud

    now = time.monotonic()
    for err in scan_result.scanner_errors:
        scanner = (err.split(":", 1)[0] or "scanner").strip()
        logger.error(
            "dlp_scanner_failed_open",
            scanner=scanner,
            request_id=request_id,
            detail=err[:300],
        )
        last = _scanner_error_seen.get(scanner)
        if last is not None and (now - last) < SCANNER_ERROR_ALERT_WINDOW:
            continue
        _scanner_error_seen[scanner] = now
        try:
            await crud.create_dlp_alert(
                db,
                request_id=request_id,
                user_id=req.user_id,
                severity="major",
                scanner=scanner,
                categories=["dlp_scanner_error"],
                entities=[],
                confidence=0.0,
                scan_latency_ms=scan_result.scan_latency_ms,
                detail=(
                    f"DLP scanner '{scanner}' failed — scans may be MISSING "
                    f"sensitive data until this is resolved. {err[:200]}"
                ),
            )
            await db.commit()
        except Exception:
            logger.exception("dlp_scanner_error_alert_failed", scanner=scanner)


async def _internal_chat(model: str, messages: list) -> str:
    """Run one chat completion for the LLM scanner, straight to a backend.

    Deliberately bypasses the gateway's own /v1 endpoint: routing internally
    means DLP needs no API key, creates no request/response rows (which would
    re-store the very content it flagged), and consumes no user's quota.
    Mirrors services/image_policy._call_judge, which does the same on the
    image-moderation path.

    Trade-off: no scheduler admission, no retry/failover.  Scans are post-hoc
    and best-effort, so a failed dispatch costs one scan, not a request.
    """
    import httpx

    from backend.app.core.telemetry.registry import get_registry
    from backend.app.db.models import BackendEngine

    registry = get_registry()
    resolved, _ = registry.resolve_alias(model)

    backend = await registry.pick_available_backend(resolved)
    if backend is None:
        logger.warning("dlp_llm_no_backend", model=model, resolved=resolved)
        raise RuntimeError(f"no healthy backend serving DLP model {resolved!r}")

    payload = {
        "model": resolved,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1024,
        "stream": False,
    }
    # Reasoning is off fleet-wide on the normal inference path; dispatching
    # directly skips that policy, so re-apply it here or the scanner's JSON
    # answer arrives wrapped in <think> blocks.  gpt-oss rejects the kwarg.
    if backend.engine != BackendEngine.OLLAMA and "gpt-oss" not in resolved.lower():
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{backend.url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

    choices = data.get("choices") or [{}]
    msg = choices[0].get("message") or {}
    return msg.get("content") or ""


def _email_allowed(severity: str) -> tuple:
    """Token-bucket guard: at most EMAIL_BURST alerts per severity per window.

    DLP scans every completed request, so a single chatty user can trip the same
    rule hundreds of times a minute.  Without a cap, enabling notifications for
    a low severity mails the admin into oblivion.  Per-process state (each
    uvicorn worker keeps its own), which is fine — this is a flood guard, not
    an exactly-once ledger.

    Returns (allowed, suppressed_since_last_send).
    """
    now = time.monotonic()
    window_start, count, suppressed = _email_budget.get(severity, (now, 0, 0))
    if now - window_start >= EMAIL_WINDOW_SECONDS:
        # Reset the window's send count but CARRY the suppressed tally: it is
        # cleared only by an actual send, so the next email that goes out can
        # tell the admin how many alerts were dropped while the budget was
        # exhausted.  Zeroing it here made that report permanently unreachable.
        window_start, count = now, 0
    if count >= EMAIL_BURST:
        _email_budget[severity] = (window_start, count, suppressed + 1)
        return False, suppressed + 1
    _email_budget[severity] = (window_start, count + 1, 0)
    return True, suppressed


async def _alert_recipients(db, severity: str, alert) -> list:
    """Resolve the email recipients for a severity's alert.

    The configured recipient list (comma/newline separated) plus — when the
    severity's notify-user flag is on — the requesting user's own address.
    Order-preserving de-duplication.
    """
    from backend.app.db import crud

    raw = await crud.get_config_json(db, f"dlp.email.{severity}_recipients", "")
    recips = [r.strip() for r in re.split(r"[,\n\r]+", str(raw)) if r.strip()]

    if await crud.get_config_json(db, f"dlp.email.{severity}.notify_user", False):
        uid = getattr(alert, "user_id", None)
        if uid:
            user = await crud.get_user_by_id(db, uid)
            email = getattr(user, "email", None) if user else None
            if email:
                recips.append(email)

    seen: set = set()
    ordered = []
    for r in recips:
        if r not in seen:
            seen.add(r)
            ordered.append(r)
    return ordered


async def _maybe_send_email(db, alert, scan_result) -> None:
    """Send an IMMEDIATE email for this alert, per its severity's action config.

    Two gates: the severity's Alert action must be ON (Admin -> DLP -> Detection
    Action), and its delivery mode must be "immediate" ("digest" defers to
    dlp_digest_loop). Recipients are the configured list plus, optionally, the
    requesting user. Defaults preserve prior behavior (alert on, immediate).
    """
    from backend.app.db import crud

    severity = scan_result.severity

    # Action model: no email at all unless this severity's Alert action is on.
    if not await crud.get_config_json(db, f"dlp.action.{severity}.alert", True):
        return

    mode = await crud.get_config_json(db, f"dlp.email.{severity}.mode", "immediate")
    if mode != "immediate":
        # "digest" is picked up by dlp_digest_loop.
        return

    recipients = await _alert_recipients(db, severity, alert)
    if not recipients:
        return

    allowed, suppressed = _email_allowed(severity)
    if not allowed:
        logger.info("dlp_email_suppressed", severity=severity, suppressed=suppressed)
        return

    try:
        from backend.app.services import email_service

        smtp_config = await email_service.get_smtp_config(db)
        if not email_service.is_smtp_configured(smtp_config):
            logger.warning("dlp_email_smtp_not_configured")
            return

        categories = ", ".join(sorted(alert.categories or [])) or "unknown"
        subject = f"[MindRouter DLP] {severity.upper()} alert — {categories}"

        # Body carries metadata only.  The matched values stay out of mail:
        # an inbox is the last place a DLP alert should reproduce them.
        base_url = await email_service.get_base_url(db)
        suppressed_note = (
            f"<p><em>{suppressed} further {severity} alert(s) were suppressed by the "
            f"notification rate limit.</em></p>" if suppressed else ""
        )
        body_html = (
            f"<p><strong>DLP Alert: {severity.upper()}</strong></p>"
            f"<p>Scanner: {scan_result.scanner}<br>"
            f"Categories: {categories}<br>"
            f"Findings: {len(scan_result.findings)}<br>"
            f"Request ID: {alert.request_id if alert.request_id else 'n/a'}<br>"
            f"Scan latency: {scan_result.scan_latency_ms}ms</p>"
            f"{suppressed_note}"
            f'<p><a href="{base_url}/admin/dlp">Review in Admin &rarr; DLP</a></p>'
            f"<p><small>Matched values are not included in this email. "
            f"Open the alert in the admin console to review.</small></p>"
        )

        sent = await email_service.send_notification_email(
            config=smtp_config,
            recipients=recipients,
            subject=subject,
            body_html=body_html,
            base_url=base_url,
        )
        logger.info("dlp_email_sent", severity=severity, recipients=len(recipients), sent=sent)
    except Exception:
        logger.exception("dlp_email_failed")


# ===================================================================
# Digest report — a rolled-up email of digest-mode alerts on a schedule
# ===================================================================

async def dlp_digest_loop() -> None:
    """Background task: periodically send the DLP digest report.

    Wakes every DIGEST_CHECK_INTERVAL and sends a report when one is due per the
    configured frequency.  Runs alongside dlp_worker_loop for the app lifespan.
    """
    logger.info("dlp_digest_loop_started")
    while True:
        try:
            await asyncio.sleep(DIGEST_CHECK_INTERVAL + random.uniform(0, DIGEST_CHECK_JITTER))
            await _maybe_send_digest()
        except asyncio.CancelledError:
            logger.info("dlp_digest_loop_cancelled")
            break
        except Exception:
            logger.exception("dlp_digest_loop_error")
            await asyncio.sleep(DIGEST_CHECK_INTERVAL + random.uniform(0, DIGEST_CHECK_JITTER))


async def _maybe_send_digest() -> None:
    """Send the digest report if one is due, then advance the watermark."""
    from backend.app.db import crud
    from backend.app.db.session import get_async_db_context

    async with get_async_db_context() as db:
        recipients_str = await crud.get_config_json(db, "dlp.digest.recipients", "")
        recipients = [r.strip() for r in str(recipients_str).split(",") if r.strip()]
        if not recipients:
            return  # digest not configured

        digest_sevs = [
            s for s in _SEVERITIES
            if await crud.get_config_json(db, f"dlp.action.{s}.alert", True)
            and await crud.get_config_json(db, f"dlp.email.{s}.mode", "immediate") == "digest"
        ]
        if not digest_sevs:
            return  # no severity routes to the digest

        freq = await crud.get_config_json(db, "dlp.digest.frequency", "daily")
        interval = DIGEST_FREQUENCIES.get(freq, DIGEST_FREQUENCIES["daily"])

        now = datetime.now(timezone.utc)
        last_sent_raw = await crud.get_config_json(db, "dlp.digest.last_sent_at", None)
        last_sent = _parse_iso(last_sent_raw)
        # First run establishes the watermark without emailing a backfill.
        if last_sent is None:
            await crud.set_config(db, "dlp.digest.last_sent_at", now.isoformat())
            await db.commit()
            return
        if (now - last_sent).total_seconds() < interval:
            return  # not due yet

        # Claim the window BEFORE sending.  Every worker process runs this
        # loop; a read-then-send-then-write sequence let several of them see
        # the same stale watermark and each email the same report (4 identical
        # digests after the 2.9.46 restart).  The claim is a single
        # compare-and-set UPDATE, so exactly one worker's statement matches the
        # old value; the rest see rowcount 0 and stand down.
        if not await _claim_digest_window(db, last_sent_raw, now.isoformat()):
            logger.info("dlp_digest_claimed_elsewhere")
            return

        # Gather digest-mode alerts scanned since the last report.
        alerts = await _digest_alerts_since(db, last_sent, digest_sevs)
        if alerts:
            try:
                await _send_digest_email(db, recipients, alerts, since=last_sent, until=now)
            except Exception:
                # Do NOT keep the advanced watermark on send failure: restore
                # the previous one so the next cycle retries the same window
                # instead of dropping alerts.  (Restore, not "never advance",
                # because the claim above already moved it.)
                logger.exception("dlp_digest_send_failed")
                try:
                    await crud.set_config(db, "dlp.digest.last_sent_at", last_sent_raw)
                    await db.commit()
                except Exception:
                    logger.exception("dlp_digest_watermark_restore_failed")
                return

        logger.info("dlp_digest_sent", alerts=len(alerts), recipients=len(recipients), frequency=freq)


async def _claim_digest_window(db, previous_raw, now_iso: str) -> bool:
    """Atomically advance the digest watermark from ``previous_raw`` to ``now_iso``.

    A compare-and-set on the app_config row: the UPDATE only matches while the
    stored value still equals the one this worker read, so among N workers
    that all saw the same stale watermark exactly one gets rowcount 1.  Values
    are compared in their stored JSON encoding (set_config json.dumps them).
    Returns True only for the worker that owns this digest window.
    """
    import json as _json
    from sqlalchemy import update
    from backend.app.db.models import AppConfig

    result = await db.execute(
        update(AppConfig)
        .where(
            AppConfig.key == "dlp.digest.last_sent_at",
            AppConfig.value == _json.dumps(previous_raw),
        )
        .values(value=_json.dumps(now_iso))
    )
    await db.commit()
    return (getattr(result, "rowcount", 0) or 0) == 1


def _parse_iso(value):
    """Parse an ISO timestamp from config, tolerant of None/garbage."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


async def _digest_alerts_since(db, since, severities):
    """Fetch digest-mode alerts scanned after `since`, oldest first, capped."""
    from sqlalchemy import select
    from backend.app.db.models import DlpAlert

    result = await db.execute(
        select(DlpAlert)
        .where(DlpAlert.scanned_at > since, DlpAlert.severity.in_(severities))
        .order_by(DlpAlert.scanned_at.asc())
        .limit(DIGEST_MAX_ROWS + 1)
    )
    return list(result.scalars().all())


DEFAULT_DISPLAY_TZ = "America/Los_Angeles"


async def _display_tz(db) -> zoneinfo.ZoneInfo:
    """The admin-configured display timezone (app.timezone), same one the
    dashboard's ``localtime`` filter uses, so an email and the console agree.
    Falls back to U.S. Pacific on a missing or unknown name."""
    from backend.app.db import crud

    name = DEFAULT_DISPLAY_TZ
    try:
        name = await crud.get_config_json(db, "app.timezone", DEFAULT_DISPLAY_TZ) or DEFAULT_DISPLAY_TZ
        return zoneinfo.ZoneInfo(str(name))
    except Exception:
        logger.warning("dlp_digest_bad_timezone", timezone=str(name)[:60])
        return zoneinfo.ZoneInfo(DEFAULT_DISPLAY_TZ)


def _fmt_local(dt, tz: zoneinfo.ZoneInfo, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Format a datetime in ``tz``.  Alert timestamps come back from MariaDB
    naive; they were written in UTC (server_default now() on a UTC host), so
    a naive value is UTC — never the local wall clock."""
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz).strftime(fmt)


def _tz_label(tz: zoneinfo.ZoneInfo, at: datetime) -> str:
    """Abbreviation in force at ``at`` (PDT vs PST), falling back to the key."""
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    return at.astimezone(tz).tzname() or getattr(tz, "key", "local")


async def _send_digest_email(db, recipients, alerts, since, until) -> None:
    """Build and send the digest report. Matched values stay masked/absent.

    Times are rendered in the configured display timezone (Admin → Settings →
    Timezone; U.S. Pacific by default) with the zone abbreviation, not UTC.
    """
    from backend.app.services import email_service

    smtp_config = await email_service.get_smtp_config(db)
    if not email_service.is_smtp_configured(smtp_config):
        logger.warning("dlp_digest_smtp_not_configured")
        raise RuntimeError("SMTP not configured")

    base_url = await email_service.get_base_url(db)
    tz = await _display_tz(db)
    tz_label = _tz_label(tz, until)
    truncated = len(alerts) > DIGEST_MAX_ROWS
    shown = alerts[:DIGEST_MAX_ROWS]

    # Severity + category tallies for the summary.
    sev_counts: dict = {}
    cat_counts: dict = {}
    for a in shown:
        sev_counts[a.severity] = sev_counts.get(a.severity, 0) + 1
        for c in (a.categories or []):
            cat_counts[c] = cat_counts.get(c, 0) + 1

    def _esc(s):
        return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    summary = ", ".join(f"{n} {_esc(s)}" for s, n in sorted(sev_counts.items())) or "0"
    cats = ", ".join(f"{_esc(c)} ({n})" for c, n in sorted(cat_counts.items(), key=lambda kv: -kv[1])) or "none"

    rows = []
    for a in shown:
        when = _fmt_local(a.scanned_at, tz)
        cat = _esc(", ".join(a.categories or []) or "unknown")
        rows.append(
            f"<tr><td>{when}</td><td>{_esc(a.severity)}</td><td>{_esc(a.scanner)}</td>"
            f"<td>{cat}</td><td>{a.request_id if a.request_id else 'n/a'}</td></tr>"
        )
    trunc_note = (
        f"<p><em>Showing the first {DIGEST_MAX_ROWS}; more alerts exist in this "
        f"window — review them in the admin console.</em></p>" if truncated else ""
    )

    body_html = (
        f"<p><strong>MindRouter DLP digest</strong></p>"
        f"<p>Window: {_fmt_local(since, tz)} &rarr; {_fmt_local(until, tz)} {_esc(tz_label)}<br>"
        f"Alerts: {len(shown)}{'+' if truncated else ''}<br>"
        f"By severity: {summary}<br>"
        f"Top categories: {cats}</p>"
        f'<table border="1" cellpadding="4" cellspacing="0">'
        f"<tr><th>Time ({_esc(tz_label)})</th><th>Severity</th><th>Scanner</th><th>Categories</th><th>Request</th></tr>"
        f"{''.join(rows)}</table>"
        f"{trunc_note}"
        f'<p><a href="{base_url}/admin/dlp">Review in Admin &rarr; DLP</a></p>'
        f"<p><small>Matched values are masked in the console and never included here.</small></p>"
    )

    await email_service.send_notification_email(
        config=smtp_config,
        recipients=recipients,
        subject=f"[MindRouter DLP] Digest — {len(shown)}{'+' if truncated else ''} alert(s)",
        body_html=body_html,
        base_url=base_url,
    )


# NOTE: ensure_internal_api_key() was removed in 2.9.9.  It minted a
# never-expiring API key owned by whichever row `SELECT id FROM users LIMIT 1`
# returned (in practice the bootstrap admin, so the key carried admin rights)
# and stored the RAW key in app_config.dlp.internal_api_key_raw — the only
# unhashed key in a system that otherwise persists Argon2 + SHA-256 only.
# The LLM scanner now dispatches straight to a backend via _internal_chat and
# needs no credential at all.  Migration 071 revokes any key it created and
# drops both config rows.
