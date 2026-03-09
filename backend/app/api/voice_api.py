############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# voice_api.py: Public TTS/STT API endpoints (/v1/audio/*)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Public voice API endpoints (TTS and STT) with API key authentication."""

import time
from typing import Optional, Tuple

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.auth import authenticate_request
from backend.app.db import crud
from backend.app.db.models import ApiKey, Modality, RequestStatus, User
from backend.app.db.session import get_async_db
from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

logger = get_logger(__name__)
router = APIRouter(prefix="/v1/audio", tags=["voice"])


class TTSRequest(BaseModel):
    """OpenAI-compatible TTS request body."""

    model: str = "kokoro"
    input: str
    voice: str = "af_heart"
    response_format: str = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _check_quota(db: AsyncSession, user: User):
    """Check if user has sufficient quota."""
    await crud.reset_quota_if_needed(db, user.id)
    quota = await crud.get_user_quota(db, user.id)
    group_budget = user.group.token_budget if user.group else 0
    if quota and group_budget > 0 and quota.tokens_used >= group_budget:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Token quota exceeded",
        )


async def _record_and_complete(
    db: AsyncSession,
    user: User,
    api_key: ApiKey,
    http_request: Request,
    endpoint: str,
    modality: Modality,
    token_cost: int,
    model: str,
    error_message: Optional[str] = None,
):
    """Create a request record, mark it completed, update quota, and commit."""
    client_ip = http_request.client.host if http_request.client else None
    user_agent = http_request.headers.get("user-agent")

    db_request = await crud.create_request(
        db=db,
        user_id=user.id,
        api_key_id=api_key.id,
        endpoint=endpoint,
        model=model,
        modality=modality,
        client_ip=client_ip,
        user_agent=user_agent,
    )

    if error_message:
        await crud.update_request_failed(db, db_request.id, error_message)
    else:
        await crud.update_request_completed(
            db, db_request.id,
            prompt_tokens=token_cost,
            completion_tokens=0,
            tokens_estimated=True,
        )
        await crud.update_quota_usage(db, user.id, token_cost)

    await db.commit()

    if not error_message:
        await crud.incr_quota_redis(user.id, token_cost)


# ---------------------------------------------------------------------------
# POST /v1/audio/speech   (TTS)
# ---------------------------------------------------------------------------

@router.post("/speech")
async def tts_speech(
    request: Request,
    body: TTSRequest,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """
    OpenAI-compatible text-to-speech endpoint.

    Proxies to the configured TTS service (Kokoro).
    Requires API key authentication.
    """
    user, api_key = auth
    settings = get_settings()

    await _check_quota(db, user)

    if not body.input.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    # Read TTS config from DB (same as chat.py)
    tts_enabled = await crud.get_config_json(db, "voice.tts_enabled", False)
    if not tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is not enabled")

    tts_url = await crud.get_config_json(db, "voice.tts_url", None)
    if not tts_url:
        raise HTTPException(status_code=500, detail="TTS service URL not configured")

    tts_api_key = await crud.get_config_json(db, "voice.tts_api_key", None)

    headers = {"Content-Type": "application/json"}
    if tts_api_key:
        headers["Authorization"] = f"Bearer {tts_api_key}"

    payload = {
        "model": body.model,
        "input": body.input,
        "voice": body.voice,
        "speed": body.speed,
        "response_format": body.response_format,
    }

    token_cost = settings.tts_quota_tokens

    async def stream_audio():
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream(
                    "POST",
                    f"{tts_url.rstrip('/')}/v1/audio/speech",
                    json=payload,
                    headers=headers,
                ) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        logger.warning("tts_api_proxy_error", status=resp.status_code, body=error_body[:500])
                        return
                    async for chunk in resp.aiter_bytes(4096):
                        yield chunk
        except Exception as e:
            logger.warning("tts_api_proxy_error", error=str(e))

    # Record the request and deduct quota
    await _record_and_complete(
        db, user, api_key, request,
        endpoint="/v1/audio/speech",
        modality=Modality.TTS,
        token_cost=token_cost,
        model=body.model,
    )

    content_type = "audio/mpeg" if body.response_format == "mp3" else f"audio/{body.response_format}"
    return StreamingResponse(stream_audio(), media_type=content_type)


# ---------------------------------------------------------------------------
# POST /v1/audio/transcriptions   (STT)
# ---------------------------------------------------------------------------

@router.post("/transcriptions")
async def stt_transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: Optional[str] = None,
    language: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """
    OpenAI-compatible speech-to-text endpoint.

    Proxies to the configured STT service (Whisper).
    Requires API key authentication.
    """
    user, api_key = auth
    settings = get_settings()

    await _check_quota(db, user)

    # Read STT config from DB (same as chat.py)
    stt_enabled = await crud.get_config_json(db, "voice.stt_enabled", False)
    if not stt_enabled:
        raise HTTPException(status_code=404, detail="STT is not enabled")

    stt_url = await crud.get_config_json(db, "voice.stt_url", None)
    if not stt_url:
        raise HTTPException(status_code=500, detail="STT service URL not configured")

    stt_model = model or await crud.get_config_json(db, "voice.stt_model", "whisper-large-v3-turbo")
    stt_api_key = await crud.get_config_json(db, "voice.stt_api_key", None)

    headers = {}
    if stt_api_key:
        headers["Authorization"] = f"Bearer {stt_api_key}"

    audio_data = await file.read()

    token_cost = settings.stt_quota_tokens

    data = {"model": stt_model}
    if language:
        data["language"] = language

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{stt_url.rstrip('/')}/v1/audio/transcriptions",
                files={"file": (file.filename or "audio.webm", audio_data, file.content_type or "application/octet-stream")},
                data=data,
                headers=headers,
            )
            if resp.status_code != 200:
                logger.warning("stt_api_proxy_error", status=resp.status_code, body=resp.text[:500])
                await _record_and_complete(
                    db, user, api_key, request,
                    endpoint="/v1/audio/transcriptions",
                    modality=Modality.STT,
                    token_cost=0,
                    model=stt_model,
                    error_message=f"STT service error: {resp.status_code}",
                )
                raise HTTPException(status_code=502, detail="STT service error")

            result = resp.json()

    except httpx.TimeoutException as e:
        logger.warning("stt_api_proxy_timeout", error=str(e))
        await _record_and_complete(
            db, user, api_key, request,
            endpoint="/v1/audio/transcriptions",
            modality=Modality.STT,
            token_cost=0,
            model=stt_model,
            error_message=f"STT timeout: {e}",
        )
        raise HTTPException(status_code=502, detail="STT service timed out (model may be loading)")
    except httpx.HTTPError as e:
        logger.warning("stt_api_proxy_error", error=str(e))
        await _record_and_complete(
            db, user, api_key, request,
            endpoint="/v1/audio/transcriptions",
            modality=Modality.STT,
            token_cost=0,
            model=stt_model,
            error_message=f"STT unavailable: {e}",
        )
        raise HTTPException(status_code=502, detail=f"STT service unavailable: {e}")

    # Success — record and deduct quota
    await _record_and_complete(
        db, user, api_key, request,
        endpoint="/v1/audio/transcriptions",
        modality=Modality.STT,
        token_cost=token_cost,
        model=stt_model,
    )

    return JSONResponse({"text": result.get("text", "")})
