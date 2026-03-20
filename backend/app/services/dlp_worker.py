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
from typing import Optional

from backend.app.logging_config import get_logger

logger = get_logger(__name__)

# Module-level queue — bounded to 10,000 to prevent unbounded memory growth
_dlp_queue: asyncio.Queue = asyncio.Queue(maxsize=10_000)


def get_dlp_queue() -> asyncio.Queue:
    """Return the module-level DLP queue."""
    return _dlp_queue


async def enqueue_for_dlp(request_id: int) -> None:
    """Enqueue a request ID for DLP scanning. Non-blocking, drops if full."""
    try:
        _dlp_queue.put_nowait(request_id)
    except asyncio.QueueFull:
        logger.warning("dlp_queue_full", dropped_request_id=request_id)


async def dlp_worker_loop() -> None:
    """Main DLP worker loop. Runs as a background task during app lifespan."""
    logger.info("dlp_worker_started")
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
            logger.info("dlp_worker_cancelled")
            break
        except Exception:
            logger.exception("dlp_worker_error")
            await asyncio.sleep(1)


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

        # Self-loop prevention: skip requests from the DLP internal API key
        internal_key_id = await crud.get_config_json(db, "dlp.internal_api_key_id", None)
        if internal_key_id is not None and req.api_key_id == internal_key_id:
            return

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

        # Build entities list (redacted snippets)
        entities = []
        categories_set = set()
        max_confidence = 0.0
        for f in scan_result.findings:
            entities.append({
                "scanner": f.scanner,
                "category": f.category,
                "text": f.text[:50],  # truncate for storage
                "confidence": f.confidence,
            })
            categories_set.add(f.category)
            if f.confidence > max_confidence:
                max_confidence = f.confidence

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
    config["llm.enabled"] = await crud.get_config_json(db, "dlp.llm.enabled", False)
    config["llm.model"] = await crud.get_config_json(db, "dlp.llm.model", "")
    config["llm.system_prompt"] = await crud.get_config_json(db, "dlp.llm.system_prompt", "")
    config["severity_rules"] = await crud.get_config_json(db, "dlp.severity_rules", {})

    # Load internal API key for LLM scanner (raw key stored in config)
    if config["llm.enabled"]:
        config["llm.api_key"] = await crud.get_config_json(db, "dlp.internal_api_key_raw", "") or ""
    else:
        config["llm.api_key"] = ""

    config["llm.base_url"] = "http://localhost:8000"

    return config


async def _maybe_send_email(db, alert, scan_result) -> None:
    """Send email notification if configured for this severity level."""
    from backend.app.db import crud

    severity = scan_result.severity
    key = f"dlp.email.{severity}_recipients"
    recipients_str = await crud.get_config_json(db, key, "")
    if not recipients_str:
        return

    recipients = [r.strip() for r in recipients_str.split(",") if r.strip()]
    if not recipients:
        return

    try:
        from backend.app.services import email_service

        smtp_config = await email_service.get_smtp_config(db)
        if not email_service.is_smtp_configured(smtp_config):
            logger.warning("dlp_email_smtp_not_configured")
            return

        subject = f"[MindRouter DLP] {severity.upper()} alert — {', '.join(scan_result.findings[0].category if scan_result.findings else ['unknown'])}"
        body = (
            f"DLP Alert: {severity.upper()}\n\n"
            f"Scanner: {scan_result.scanner}\n"
            f"Findings: {len(scan_result.findings)}\n"
            f"Detail: {scan_result.detail}\n"
            f"Request ID: {alert.request_id}\n"
            f"Scan latency: {scan_result.scan_latency_ms}ms\n\n"
            f"Review in Admin: /admin/dlp"
        )

        for recipient in recipients:
            await email_service.send_email(
                smtp_config=smtp_config,
                to_email=recipient,
                subject=subject,
                body=body,
            )

        logger.info("dlp_email_sent", severity=severity, recipients=len(recipients))
    except Exception:
        logger.exception("dlp_email_failed")


async def ensure_internal_api_key(db) -> Optional[int]:
    """Ensure the DLP system API key exists. Returns the key ID.

    The raw key is stored in AppConfig so the LLM scanner can use it
    for self-routing. The DB is encrypted at rest via TDE.
    """
    from backend.app.db import crud

    existing_id = await crud.get_config_json(db, "dlp.internal_api_key_id", None)
    if existing_id is not None:
        # Verify it still exists
        from backend.app.db.models import ApiKey
        from sqlalchemy import select
        result = await db.execute(
            select(ApiKey.id).where(ApiKey.id == existing_id)
        )
        if result.scalar_one_or_none() is not None:
            return existing_id

    # Generate a proper API key
    from backend.app.security.api_keys import generate_api_key
    full_key, key_hash, key_prefix = generate_api_key()

    # Find a system user (user_id=1 is typically admin)
    from backend.app.db.models import User
    from sqlalchemy import select as sa_select
    result = await db.execute(sa_select(User.id).limit(1))
    system_user_id = result.scalar_one_or_none() or 1

    api_key = await crud.create_api_key(
        db,
        user_id=system_user_id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name="DLP Internal Scanner",
    )
    await crud.set_config(db, "dlp.internal_api_key_id", api_key.id)
    # Store raw key in config so LLM scanner can authenticate
    await crud.set_config(db, "dlp.internal_api_key_raw", full_key)
    await db.commit()

    logger.info("dlp_internal_api_key_created", key_id=api_key.id)
    return api_key.id
