"""Voice Chat routes — full-duplex voice conversation via PersonaPlex proxy."""

import asyncio
import json
import os
import ssl
from typing import Optional
from urllib.parse import urlencode, urlparse, urlunparse

from fastapi import APIRouter, Depends, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db import crud
from backend.app.db.session import get_async_db
from backend.app.dashboard.routes import get_session_user_id, get_effective_user_id
from backend.app.logging_config import get_logger

logger = get_logger(__name__)

voice_chat_router = APIRouter(tags=["voice-chat"])

# Setup templates (same directory as other dashboard templates)
templates_path = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_path)
from backend.app.dashboard.routes import _get_voice_chat_enabled
templates.env.globals["voice_chat_enabled"] = _get_voice_chat_enabled


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_session_user_id_from_ws(websocket: WebSocket) -> Optional[int]:
    """Extract user ID from the mindrouter_session cookie on a WebSocket."""
    from itsdangerous import URLSafeTimedSerializer
    from backend.app.settings import get_settings

    session_data = websocket.cookies.get("mindrouter_session")
    if not session_data:
        return None
    try:
        settings = get_settings()
        serializer = URLSafeTimedSerializer(settings.secret_key, salt="session")
        user_id = serializer.loads(session_data, max_age=86400 * 7)
        return int(user_id)
    except Exception:
        return None


def _build_personaplex_url(base_url: str, voice_prompt: str, text_prompt: str) -> str:
    """Build the PersonaPlex WebSocket URL with voice/text prompt query params.

    PersonaPlex expects connections to /api/chat with query parameters:
      ?voice_prompt=NATF2.pt&text_prompt=You+are+a+friendly+assistant
    """
    parsed = urlparse(base_url)
    # Ensure path ends with /api/chat
    path = parsed.path.rstrip("/")
    if not path.endswith("/api/chat"):
        path = path + "/api/chat"
    query = urlencode({
        "voice_prompt": voice_prompt,
        "text_prompt": text_prompt,
    })
    return urlunparse((parsed.scheme, parsed.netloc, path, "", query, ""))


# ---------------------------------------------------------------------------
# Page render
# ---------------------------------------------------------------------------

@voice_chat_router.get("/voice-chat", response_class=HTMLResponse)
async def voice_chat_page(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Render the Voice Chat page."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    # Check feature enabled
    enabled = await crud.get_config_json(db, "voice_chat.enabled", False)
    if not enabled:
        return RedirectResponse(url="/chat", status_code=302)

    # Load personas
    personas = await crud.get_config_json(db, "voice_chat.personas", [])

    # Load user's preferred persona
    effective_id = await get_effective_user_id(request, db)
    user_persona = await crud.get_config_json(
        db, f"user.{effective_id}.voice_chat_persona", None
    )

    return templates.TemplateResponse(
        "voice_chat.html",
        {
            "request": request,
            "user": user,
            "personas": personas,
            "user_persona": user_persona,
        },
    )


# ---------------------------------------------------------------------------
# Save persona preference
# ---------------------------------------------------------------------------

@voice_chat_router.post("/voice-chat/api/save-persona")
async def save_voice_chat_persona(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Save the user's preferred voice chat persona."""
    user_id = get_session_user_id(request)
    if not user_id:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    effective_id = await get_effective_user_id(request, db)
    body = await request.json()
    persona_name = body.get("persona", "")

    config_key = f"user.{effective_id}.voice_chat_persona"
    if persona_name:
        await crud.set_config(db, config_key, persona_name)
    await db.commit()

    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# WebSocket proxy — browser ↔ MindRouter ↔ PersonaPlex
# ---------------------------------------------------------------------------

@voice_chat_router.websocket("/voice-chat/ws")
async def voice_chat_ws(websocket: WebSocket):
    """Bidirectional WebSocket proxy to PersonaPlex backend.

    PersonaPlex protocol (binary frames):
      0x00 = handshake (server → client, signals ready)
      0x01 + opus_data = audio (bidirectional)
      0x02 + utf8_text = text token (server → client)

    Persona config format:
      {
        "name": "Nova",
        "url": "wss://host:8998",
        "voice_prompt": "NATF2.pt",
        "text_prompt": "You are a friendly assistant...",
        "description": "Friendly assistant"
      }
    """
    import websockets

    # 1. Auth
    user_id = _get_session_user_id_from_ws(websocket)
    if not user_id:
        await websocket.close(code=4001, reason="Not authenticated")
        return

    # 2. Get config from DB
    from backend.app.db.session import get_async_db_context
    async with get_async_db_context() as db:
        enabled = await crud.get_config_json(db, "voice_chat.enabled", False)
        if not enabled:
            await websocket.close(code=4003, reason="Voice chat disabled")
            return

        personas = await crud.get_config_json(db, "voice_chat.personas", [])

    # 3. Find the requested persona
    persona_name = websocket.query_params.get("persona", "")
    persona = None
    if personas:
        for p in personas:
            if p.get("name") == persona_name:
                persona = p
                break
        # Fallback to first persona
        if not persona:
            persona = personas[0] if personas else None

    if not persona or not persona.get("url"):
        await websocket.close(code=4002, reason="No PersonaPlex persona configured")
        return

    # 4. Build the PersonaPlex WebSocket URL with voice/text prompt params
    pp_url = _build_personaplex_url(
        persona["url"],
        persona.get("voice_prompt", "NATF2.pt"),
        persona.get("text_prompt", "You enjoy having a good conversation."),
    )

    # 5. Accept browser WebSocket
    await websocket.accept()
    logger.info(
        "voice_chat_ws_connected",
        user_id=user_id,
        persona=persona.get("name", "unknown"),
        backend_url=persona["url"],
    )

    # 6. Connect to PersonaPlex backend
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    connect_kwargs = {"ssl": ssl_ctx} if pp_url.startswith("wss://") else {}
    # PersonaPlex processes system prompts before sending handshake — allow up to 120s
    connect_kwargs["open_timeout"] = 120
    connect_kwargs["close_timeout"] = 10
    connect_kwargs["ping_timeout"] = 60

    try:
        async with websockets.connect(pp_url, **connect_kwargs) as pp_ws:
            # 7. Bridge: forward all frames bidirectionally
            # PersonaPlex uses binary frames exclusively (0x00/0x01/0x02 prefixed)
            async def browser_to_pp():
                try:
                    while True:
                        msg = await websocket.receive()
                        if msg.get("type") == "websocket.disconnect":
                            break
                        if "bytes" in msg and msg["bytes"]:
                            await pp_ws.send(msg["bytes"])
                        elif "text" in msg and msg["text"]:
                            await pp_ws.send(msg["text"])
                except Exception:
                    pass
                finally:
                    await pp_ws.close()

            async def pp_to_browser():
                try:
                    async for msg in pp_ws:
                        if isinstance(msg, bytes):
                            await websocket.send_bytes(msg)
                        else:
                            await websocket.send_text(msg)
                except Exception:
                    pass

            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(browser_to_pp()),
                    asyncio.create_task(pp_to_browser()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    except Exception as exc:
        logger.warning("voice_chat_ws_error", error=str(exc), user_id=user_id)
        try:
            await websocket.close(code=4000, reason=str(exc)[:120])
        except Exception:
            pass
    finally:
        logger.info("voice_chat_ws_disconnected", user_id=user_id)
