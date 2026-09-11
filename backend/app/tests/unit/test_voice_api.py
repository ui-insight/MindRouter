############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_voice_api.py: Unit tests for public TTS/STT API endpoints
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for the public voice API (TTS and STT).

Covers:
- TTSRequest Pydantic model validation
- _check_quota helper: passes, rejects, handles missing group
- _record_and_complete helper: success path, error path, Redis sync
- TTS endpoint: happy path, empty text, TTS disabled, URL not configured,
  upstream API key forwarding, response_format content-type mapping
- STT endpoint: happy path, STT disabled, URL not configured,
  upstream error, upstream timeout, upstream HTTP error, language passthrough,
  custom model parameter
- Modality enum values exist for TTS and STT
"""

import importlib
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

# ----------------------------------------------------------------
# Direct-load voice_api module to avoid the DB import chain.
# ----------------------------------------------------------------

_api_dir = Path(__file__).resolve().parents[2] / "api"
_services_dir = Path(__file__).resolve().parents[2] / "services"
_models_dir = Path(__file__).resolve().parents[2] / "db"


def _load_voice_modules():
    """Direct-load voice_api.py + the REAL services/voice_router.py against
    FORCED stubs, then restore sys.modules.

    Two lessons are baked in here:
    - `setdefault` stubbing binds this file's modules to whatever happened to
      be imported first, so behavior depended on suite order (43/43 alone,
      20 failures after files that imported the real db chain). Stubs are
      installed unconditionally now — loading is deterministic — and every
      touched key is snapshotted and restored, so nothing leaks into other
      files' collection either (the leaked stubs previously broke five other
      files' collection outright).
    - voice_router is spec-loaded from source (not a stand-in) so these tests
      exercise the production resolution logic.
    """
    _KEYS = (
        "backend.app.api", "backend.app.api.auth",
        "backend.app.db", "backend.app.db.crud", "backend.app.db.models",
        "backend.app.db.session", "backend.app.db.base",
        "backend.app.logging_config",
        "backend.app.services", "backend.app.services.voice_router",
    )
    saved = {k: sys.modules.get(k) for k in _KEYS}
    try:
        sys.modules["backend.app.api"] = MagicMock()
        sys.modules["backend.app.api.auth"] = MagicMock()
        db_stub = MagicMock()
        sys.modules["backend.app.db"] = db_stub
        sys.modules["backend.app.db.crud"] = db_stub.crud
        sys.modules["backend.app.db.models"] = MagicMock()
        sys.modules["backend.app.db.session"] = MagicMock()
        sys.modules["backend.app.logging_config"] = MagicMock(
            get_logger=MagicMock(return_value=MagicMock()))

        vr_spec = importlib.util.spec_from_file_location(
            "backend.app.services.voice_router", _services_dir / "voice_router.py",
            submodule_search_locations=[],
        )
        vr_mod = importlib.util.module_from_spec(vr_spec)
        vr_spec.loader.exec_module(vr_mod)

        services_stub = MagicMock()
        services_stub.__path__ = [str(_services_dir)]
        services_stub.voice_router = vr_mod
        sys.modules["backend.app.services"] = services_stub
        sys.modules["backend.app.services.voice_router"] = vr_mod

        voice_spec = importlib.util.spec_from_file_location(
            "voice_api_under_test", _api_dir / "voice_api.py",
            submodule_search_locations=[],
        )
        voice_mod = importlib.util.module_from_spec(voice_spec)
        voice_spec.loader.exec_module(voice_mod)

        # Real Modality enum from db/models.py, with Base stubbed out. Forced
        # (not setdefault): with the real declarative Base already imported,
        # re-executing models.py would collide on its MetaData.
        sys.modules["backend.app.db.base"] = MagicMock(
            Base=type("Base", (), {"__tablename__": "", "metadata": MagicMock()}),
            TimestampMixin=type("TimestampMixin", (), {}),
            SoftDeleteMixin=type("SoftDeleteMixin", (), {}),
        )
        models_spec = importlib.util.spec_from_file_location(
            "models_enum", _models_dir / "models.py",
            submodule_search_locations=[],
        )
        models_mod = importlib.util.module_from_spec(models_spec)
        try:
            models_spec.loader.exec_module(models_mod)
            modality = models_mod.Modality
        except Exception:
            # Fallback if models.py can't load due to SQLAlchemy deps
            modality = None
        return voice_mod, vr_mod, db_stub, modality
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


_voice_mod, _vr_mod, _db_stub, Modality = _load_voice_modules()

# Pull out testable items
TTSRequest = _voice_mod.TTSRequest
_check_quota = _voice_mod._check_quota
_record_and_complete = _voice_mod._record_and_complete
tts_speech = _voice_mod.tts_speech
stt_transcriptions = _voice_mod.stt_transcriptions

# Also get module-level references so we can patch them
_crud = _voice_mod.crud


@pytest.fixture(autouse=True)
def _pin_call_time_imports():
    """voice_api and voice_router resolve their collaborators with imports
    INSIDE the request handlers (voice_api.py `from ...voice_router import`,
    voice_router.py `from backend.app.db import crud` / registry), so what
    they bind depends on sys.modules at CALL time, not at load. Pin those
    keys to this file's stubs for the duration of every test — suite order
    and other files' leftovers can no longer rebind them — and restore
    afterwards so nothing leaks out of this file."""
    registry_stub = MagicMock()
    registry_stub.get_registry = MagicMock(side_effect=RuntimeError(
        "registry stubbed — voice tests exercise the config fallback"))
    overrides = {
        "backend.app.db": _db_stub,
        "backend.app.db.crud": _crud,
        "backend.app.services.voice_router": _vr_mod,
        "backend.app.core.telemetry.registry": registry_stub,
    }
    # Ancestor packages must exist in sys.modules or resolving the keys
    # above would import the real (heavy) packages mid-test.
    import types as _types
    for anc in ("backend.app.core", "backend.app.core.telemetry",
                "backend.app.services"):
        if anc not in sys.modules:
            stub = _types.ModuleType(anc)
            stub.__path__ = []
            overrides[anc] = stub
    saved = {k: sys.modules.get(k) for k in overrides}
    sys.modules.update(overrides)
    # Re-pin the attribute path too: `from backend.app.db import crud`
    # resolves via getattr on the parent, and another test once rebound it.
    _db_stub.crud = _crud
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# ================================================================
# Helpers
# ================================================================

def _make_mock_user(tokens_used=0, token_budget=1000000):
    """Create a mock user with group."""
    group = MagicMock()
    group.token_budget = token_budget
    user = MagicMock()
    user.id = 1
    user.group = group
    return user


def _make_mock_api_key():
    """Create a mock API key."""
    api_key = MagicMock()
    api_key.id = 42
    return api_key


def _make_mock_request():
    """Create a mock FastAPI Request."""
    request = MagicMock()
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    request.headers = {"user-agent": "test-client/1.0"}
    return request


def _make_mock_db():
    """Create a mock async DB session."""
    db = AsyncMock()
    db.commit = AsyncMock()
    return db


# ================================================================
# TTSRequest Pydantic model
# ================================================================


class TestTTSRequest:
    """Test TTSRequest Pydantic model validation."""

    def test_defaults(self):
        """Default values are applied correctly."""
        req = TTSRequest(input="Hello world")
        assert req.model == "kokoro"
        assert req.voice == "af_heart"
        assert req.response_format == "mp3"
        assert req.speed == 1.0

    def test_custom_values(self):
        """Custom values override defaults."""
        req = TTSRequest(
            model="custom-tts",
            input="Test",
            voice="en_male",
            response_format="wav",
            speed=1.5,
        )
        assert req.model == "custom-tts"
        assert req.voice == "en_male"
        assert req.response_format == "wav"
        assert req.speed == 1.5

    def test_speed_min_boundary(self):
        """Speed at minimum boundary (0.25) is valid."""
        req = TTSRequest(input="Test", speed=0.25)
        assert req.speed == 0.25

    def test_speed_max_boundary(self):
        """Speed at maximum boundary (4.0) is valid."""
        req = TTSRequest(input="Test", speed=4.0)
        assert req.speed == 4.0

    def test_speed_below_min_rejected(self):
        """Speed below 0.25 is rejected."""
        with pytest.raises(Exception):
            TTSRequest(input="Test", speed=0.1)

    def test_speed_above_max_rejected(self):
        """Speed above 4.0 is rejected."""
        with pytest.raises(Exception):
            TTSRequest(input="Test", speed=5.0)

    def test_empty_input_accepted_by_model(self):
        """Empty string is accepted by Pydantic (endpoint validates it)."""
        req = TTSRequest(input="")
        assert req.input == ""

    def test_missing_input_rejected(self):
        """Missing input field is rejected."""
        with pytest.raises(Exception):
            TTSRequest()


# ================================================================
# _check_quota helper
# ================================================================


class TestCheckQuota:
    """Test the _check_quota helper function."""

    @pytest.mark.asyncio
    async def test_quota_passes_when_under_budget(self):
        """No exception when tokens_used < budget."""
        db = _make_mock_db()
        user = _make_mock_user(tokens_used=500, token_budget=1000000)
        quota = MagicMock()
        # real column is nullable; a bare MagicMock reads as an override
        quota.token_budget_override = None
        quota.tokens_used = 500
        quota.rpm_limit = 0  # real column is an int; bare MagicMock breaks `> 0`

        with patch.object(_crud, "reset_quota_if_needed", new_callable=AsyncMock), \
             patch.object(_crud, "get_user_quota", new_callable=AsyncMock, return_value=quota):
            await _check_quota(db, user)  # Should not raise

    @pytest.mark.asyncio
    async def test_quota_rejected_when_exceeded(self):
        """HTTPException 429 when tokens_used >= budget."""
        db = _make_mock_db()
        user = _make_mock_user(token_budget=1000)
        quota = MagicMock()
        # real column is nullable; a bare MagicMock reads as an override
        quota.token_budget_override = None
        quota.tokens_used = 1000

        with patch.object(_crud, "reset_quota_if_needed", new_callable=AsyncMock), \
             patch.object(_crud, "get_user_quota", new_callable=AsyncMock, return_value=quota):
            with pytest.raises(HTTPException) as exc:
                await _check_quota(db, user)
            assert exc.value.status_code == 429

    @pytest.mark.asyncio
    async def test_quota_passes_when_no_group(self):
        """No exception when user has no group (budget=0)."""
        db = _make_mock_db()
        user = MagicMock()
        user.id = 1
        user.group = None
        quota = MagicMock()
        # real column is nullable; a bare MagicMock reads as an override
        quota.token_budget_override = None
        quota.tokens_used = 99999
        quota.rpm_limit = 0  # real column is an int; bare MagicMock breaks `> 0`

        with patch.object(_crud, "reset_quota_if_needed", new_callable=AsyncMock), \
             patch.object(_crud, "get_user_quota", new_callable=AsyncMock, return_value=quota):
            await _check_quota(db, user)  # Should not raise

    @pytest.mark.asyncio
    async def test_quota_passes_when_no_quota_record(self):
        """No exception when user has no quota record yet."""
        db = _make_mock_db()
        user = _make_mock_user()

        with patch.object(_crud, "reset_quota_if_needed", new_callable=AsyncMock), \
             patch.object(_crud, "get_user_quota", new_callable=AsyncMock, return_value=None):
            await _check_quota(db, user)  # Should not raise


# ================================================================
# _record_and_complete helper
# ================================================================


class TestRecordAndComplete:
    """Test the _record_and_complete helper function."""

    @pytest.mark.asyncio
    async def test_success_path_creates_record_and_updates_quota(self):
        """On success: create_request, update_completed, update_quota, commit, incr_redis."""
        db = _make_mock_db()
        user = _make_mock_user()
        api_key = _make_mock_api_key()
        http_request = _make_mock_request()

        mock_db_req = MagicMock()
        mock_db_req.id = 99

        # Use a mock Modality value
        mock_modality = MagicMock()

        with patch.object(_crud, "create_request", new_callable=AsyncMock, return_value=mock_db_req) as cr, \
             patch.object(_crud, "update_request_completed", new_callable=AsyncMock) as uc, \
             patch.object(_crud, "update_quota_usage", new_callable=AsyncMock) as uq, \
             patch.object(_crud, "incr_quota_redis", new_callable=AsyncMock) as ir:

            await _record_and_complete(
                db, user, api_key, http_request,
                endpoint="/v1/audio/speech",
                modality=mock_modality,
                token_cost=100,
                model="kokoro",
            )

            cr.assert_called_once()
            uc.assert_called_once_with(db, 99, prompt_tokens=100, completion_tokens=0, tokens_estimated=True)
            uq.assert_called_once_with(db, user.id, 100)
            db.commit.assert_called_once()
            ir.assert_called_once_with(user.id, 100)

    @pytest.mark.asyncio
    async def test_error_path_records_failure_no_quota_deduction(self):
        """On error: create_request, update_failed, commit, NO quota update."""
        db = _make_mock_db()
        user = _make_mock_user()
        api_key = _make_mock_api_key()
        http_request = _make_mock_request()

        mock_db_req = MagicMock()
        mock_db_req.id = 99
        mock_modality = MagicMock()

        with patch.object(_crud, "create_request", new_callable=AsyncMock, return_value=mock_db_req) as cr, \
             patch.object(_crud, "update_request_failed", new_callable=AsyncMock) as uf, \
             patch.object(_crud, "update_request_completed", new_callable=AsyncMock) as uc, \
             patch.object(_crud, "update_quota_usage", new_callable=AsyncMock) as uq, \
             patch.object(_crud, "incr_quota_redis", new_callable=AsyncMock) as ir:

            await _record_and_complete(
                db, user, api_key, http_request,
                endpoint="/v1/audio/transcriptions",
                modality=mock_modality,
                token_cost=0,
                model="whisper",
                error_message="STT service error: 500",
            )

            cr.assert_called_once()
            uf.assert_called_once_with(db, 99, "STT service error: 500")
            uc.assert_not_called()
            uq.assert_not_called()
            db.commit.assert_called_once()
            ir.assert_not_called()

    @pytest.mark.asyncio
    async def test_client_ip_and_user_agent_captured(self):
        """Client IP and user-agent are passed to create_request."""
        db = _make_mock_db()
        user = _make_mock_user()
        api_key = _make_mock_api_key()
        http_request = _make_mock_request()

        mock_db_req = MagicMock()
        mock_db_req.id = 1

        with patch.object(_crud, "create_request", new_callable=AsyncMock, return_value=mock_db_req) as cr, \
             patch.object(_crud, "update_request_completed", new_callable=AsyncMock), \
             patch.object(_crud, "update_quota_usage", new_callable=AsyncMock), \
             patch.object(_crud, "incr_quota_redis", new_callable=AsyncMock):

            await _record_and_complete(
                db, user, api_key, http_request,
                endpoint="/v1/audio/speech",
                modality=MagicMock(),
                token_cost=100,
                model="kokoro",
            )

            call_kwargs = cr.call_args[1]
            assert call_kwargs["client_ip"] == "127.0.0.1"
            assert call_kwargs["user_agent"] == "test-client/1.0"


# ================================================================
# TTS endpoint
# ================================================================


def _make_tts_upstream(status=200, chunks=(b"ID3AUDIO",), send_exc=None,
                       error_body=b'{"detail":"upstream exploded"}'):
    """Mock httpx client for the TTS proxy.

    The endpoint calls build_request/send(stream=True) and consumes the
    response with aiter_bytes, so the mock must model that shape rather than
    the older `client.stream()` context manager.
    """
    resp = MagicMock()
    resp.status_code = status
    resp.aread = AsyncMock(return_value=error_body)
    resp.aclose = AsyncMock()

    async def _aiter_bytes(_n=4096):
        for c in chunks:
            yield c

    resp.aiter_bytes = _aiter_bytes

    client = MagicMock()
    client.build_request = MagicMock(return_value=MagicMock(name="request"))
    client.send = AsyncMock(side_effect=send_exc) if send_exc else AsyncMock(return_value=resp)
    client.aclose = AsyncMock()
    return client, resp


_TTS_CONFIG = {
    ("voice.tts_enabled", False): True,
    ("voice.tts_url", None): "http://tts-service:8080",
    ("voice.tts_api_key", None): None,
    ("voice_api.tts_quota_tokens", 100): 100,
}


def _tts_patches(client, config_map=None, record=None):
    cfg = config_map or _TTS_CONFIG
    return (
        patch.object(_voice_mod, "_check_quota", new_callable=AsyncMock),
        patch.object(_crud, "get_config_json", new_callable=AsyncMock,
                     side_effect=lambda db, key, default: cfg.get((key, default), default)),
        patch.object(_voice_mod, "_record_and_complete",
                     new=(record or AsyncMock())),
        patch.object(_voice_mod.httpx, "AsyncClient", return_value=client),
    )


class TestTTSSpeechEndpoint:
    """Test the POST /v1/audio/speech endpoint."""

    @pytest.mark.asyncio
    async def test_tts_happy_path_returns_streaming_response(self):
        """Successful TTS request returns StreamingResponse with audio/mpeg."""
        db = _make_mock_db()
        user = _make_mock_user()
        api_key = _make_mock_api_key()
        http_request = _make_mock_request()
        body = TTSRequest(input="Hello world")

        client, _ = _make_tts_upstream(status=200)
        p1, p2, p3, p4 = _tts_patches(client)
        with p1, p2, p3, p4:
            result = await tts_speech(http_request, body, db=db, auth=(user, api_key))
            assert result.media_type == "audio/mpeg"
            # the audio actually flows through
            got = b"".join([c async for c in result.body_iterator])
            assert got == b"ID3AUDIO"

    @pytest.mark.asyncio
    async def test_tts_empty_text_rejected(self):
        """Empty input text raises 400."""
        db = _make_mock_db()
        user = _make_mock_user()
        api_key = _make_mock_api_key()
        http_request = _make_mock_request()
        body = TTSRequest(input="   ")

        with patch.object(_voice_mod, "_check_quota", new_callable=AsyncMock):
            with pytest.raises(HTTPException) as exc:
                await tts_speech(http_request, body, db=db, auth=(user, api_key))
            assert exc.value.status_code == 400
            assert "No text" in exc.value.detail

    @pytest.mark.asyncio
    async def test_tts_disabled_returns_404(self):
        """TTS disabled in config returns 404."""
        db = _make_mock_db()
        user = _make_mock_user()
        api_key = _make_mock_api_key()
        http_request = _make_mock_request()
        body = TTSRequest(input="Hello")

        config_map = {
            ("voice.tts_enabled", False): False,
        }

        with patch.object(_voice_mod, "_check_quota", new_callable=AsyncMock), \
             patch.object(_crud, "get_config_json", new_callable=AsyncMock,
                          side_effect=lambda db, key, default: config_map.get((key, default), default)):
            with pytest.raises(HTTPException) as exc:
                await tts_speech(http_request, body, db=db, auth=(user, api_key))
            assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_tts_url_not_configured_returns_500(self):
        """TTS enabled but no URL configured returns 500."""
        db = _make_mock_db()
        user = _make_mock_user()
        api_key = _make_mock_api_key()
        http_request = _make_mock_request()
        body = TTSRequest(input="Hello")

        config_map = {
            ("voice.tts_enabled", False): True,
            ("voice.tts_url", None): None,
        }

        with patch.object(_voice_mod, "_check_quota", new_callable=AsyncMock), \
             patch.object(_crud, "get_config_json", new_callable=AsyncMock,
                          side_effect=lambda db, key, default: config_map.get((key, default), default)):
            with pytest.raises(HTTPException) as exc:
                await tts_speech(http_request, body, db=db, auth=(user, api_key))
            assert exc.value.status_code == 500

    @pytest.mark.asyncio
    async def test_tts_wav_format_returns_audio_wav(self):
        """response_format=wav returns audio/wav content type."""
        db = _make_mock_db()
        user = _make_mock_user()
        api_key = _make_mock_api_key()
        http_request = _make_mock_request()
        body = TTSRequest(input="Hello", response_format="wav")

        client, _ = _make_tts_upstream(status=200)
        p1, p2, p3, p4 = _tts_patches(client)
        with p1, p2, p3, p4:
            result = await tts_speech(http_request, body, db=db, auth=(user, api_key))
            assert result.media_type == "audio/wav"

    @pytest.mark.asyncio
    async def test_tts_records_request_with_correct_params(self):
        """TTS endpoint calls _record_and_complete with correct modality and cost."""
        db = _make_mock_db()
        user = _make_mock_user()
        api_key = _make_mock_api_key()
        http_request = _make_mock_request()
        body = TTSRequest(input="Hello", model="custom-tts")

        cfg = {
            ("voice.tts_enabled", False): True,
            ("voice.tts_url", None): "http://tts:8080",
            ("voice.tts_api_key", None): None,
            ("voice_api.tts_quota_tokens", 100): 150,
        }

        client, _ = _make_tts_upstream(status=200)
        record = AsyncMock()
        p1, p2, p3, p4 = _tts_patches(client, config_map=cfg, record=record)
        with p1, p2, p3, p4:
            await tts_speech(http_request, body, db=db, auth=(user, api_key))

            record.assert_called_once()
            call_kwargs = record.call_args[1]
            assert call_kwargs["endpoint"] == "/v1/audio/speech"
            assert call_kwargs["token_cost"] == 150
            assert call_kwargs["model"] == "custom-tts"

    @pytest.mark.asyncio
    async def test_tts_upstream_api_key_forwarded(self):
        """When voice.tts_api_key is set, it's included in the upstream headers."""
        db = _make_mock_db()
        user = _make_mock_user()
        api_key = _make_mock_api_key()
        http_request = _make_mock_request()
        body = TTSRequest(input="Hello")

        cfg = {
            ("voice.tts_enabled", False): True,
            ("voice.tts_url", None): "http://tts:8080",
            ("voice.tts_api_key", None): "sk-upstream-key",
            ("voice_api.tts_quota_tokens", 100): 100,
        }

        client, _ = _make_tts_upstream(status=200)
        p1, p2, p3, p4 = _tts_patches(client, config_map=cfg)
        with p1, p2, p3, p4:
            await tts_speech(http_request, body, db=db, auth=(user, api_key))

        # Actually assert the header reached the upstream request. The previous
        # version of this test asserted only `result is not None`, which passed
        # whether or not the key was forwarded.
        client.build_request.assert_called_once()
        headers = client.build_request.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer sk-upstream-key"


class TestTTSUpstreamFailureIsNotBilled:
    """A failing TTS upstream must produce an error, not a billed empty 200.

    Before this fix the request was recorded COMPLETED and the caller charged
    the full token cost *before* the upstream was ever contacted — the response
    generator dialled out lazily, so a dead service returned HTTP 200 with a
    zero-byte body while the audit trail showed success. Every test in this
    class fails against that implementation.
    """

    async def _run(self, **upstream):
        db = _make_mock_db()
        user = _make_mock_user()
        api_key = _make_mock_api_key()
        http_request = _make_mock_request()
        body = TTSRequest(input="Hello world")

        client, _ = _make_tts_upstream(**upstream)
        record = AsyncMock()
        p1, p2, p3, p4 = _tts_patches(client, record=record)
        with p1, p2, p3, p4:
            with pytest.raises(HTTPException) as exc:
                await tts_speech(http_request, body, db=db, auth=(user, api_key))
        return exc.value, record, client

    @pytest.mark.asyncio
    async def test_upstream_500_raises_502_and_bills_nothing(self):
        exc, record, _ = await self._run(status=500)
        assert exc.status_code == 502
        record.assert_called_once()
        kw = record.call_args[1]
        assert kw["token_cost"] == 0, "a failed TTS request must not be billed"
        assert kw["error_message"], "the failure must be recorded, not silent"

    @pytest.mark.asyncio
    async def test_upstream_timeout_raises_502_and_bills_nothing(self):
        exc, record, _ = await self._run(send_exc=_voice_mod.httpx.TimeoutException("timed out"))
        assert exc.status_code == 502
        assert record.call_args[1]["token_cost"] == 0

    @pytest.mark.asyncio
    async def test_upstream_unreachable_raises_502_and_bills_nothing(self):
        exc, record, _ = await self._run(send_exc=ConnectionError("no route to host"))
        assert exc.status_code == 502
        assert record.call_args[1]["token_cost"] == 0

    @pytest.mark.asyncio
    async def test_failure_closes_the_upstream_connection(self):
        """No leaked httpx client/response on the error path."""
        _, _, client = await self._run(status=503)
        client.aclose.assert_awaited()

    @pytest.mark.asyncio
    async def test_error_body_is_not_logged(self):
        """An upstream validation error can echo the caller's input text back,
        so the body must never reach the logs."""
        db = _make_mock_db()
        user = _make_mock_user()
        api_key = _make_mock_api_key()
        http_request = _make_mock_request()
        body = TTSRequest(input="my social security number is 123-45-6789")

        sentinel = b'{"detail":"bad input: my social security number is 123-45-6789"}'
        client, _ = _make_tts_upstream(status=422, error_body=sentinel)

        emitted = []

        class _CaptureLogger:
            def warning(self, event, **kw):
                emitted.append((event, kw))

            def __getattr__(self, _):
                return lambda *a, **k: None

        p1, p2, p3, p4 = _tts_patches(client)
        with p1, p2, p3, p4, patch.object(_voice_mod, "logger", _CaptureLogger()):
            with pytest.raises(HTTPException):
                await tts_speech(http_request, body, db=db, auth=(user, api_key))

        assert emitted, "the upstream error should still be logged"
        blob = repr(emitted)
        assert "123-45-6789" not in blob, f"caller content leaked into logs: {blob}"

    @pytest.mark.asyncio
    async def test_success_still_bills_exactly_once(self):
        """The fix must not double-charge or skip charging on the happy path."""
        db = _make_mock_db()
        user = _make_mock_user()
        api_key = _make_mock_api_key()
        http_request = _make_mock_request()
        body = TTSRequest(input="Hello")

        client, _ = _make_tts_upstream(status=200)
        record = AsyncMock()
        p1, p2, p3, p4 = _tts_patches(client, record=record)
        with p1, p2, p3, p4:
            result = await tts_speech(http_request, body, db=db, auth=(user, api_key))
            b"".join([c async for c in result.body_iterator])

        record.assert_called_once()
        kw = record.call_args[1]
        assert kw["token_cost"] == 100
        assert kw.get("error_message") is None


# ================================================================
# STT endpoint
# ================================================================


class TestSTTTranscriptionsEndpoint:
    """Test the POST /v1/audio/transcriptions endpoint."""

    def _make_upload_file(self, filename="test.mp3", content=b"fake-audio-data"):
        """Create a mock UploadFile."""
        upload = MagicMock()
        upload.filename = filename
        upload.content_type = "audio/mpeg"
        upload.read = AsyncMock(return_value=content)
        return upload

    @pytest.mark.asyncio
    async def test_stt_happy_path(self):
        """Successful STT returns JSON with text."""
        db = _make_mock_db()
        user = _make_mock_user()
        api_key = _make_mock_api_key()
        http_request = _make_mock_request()
        file = self._make_upload_file()

        config_map = {
            ("voice.stt_enabled", False): True,
            ("voice.stt_url", None): "http://stt:8080",
            ("voice.stt_model", "whisper-large-v3-turbo"): "whisper-large-v3-turbo",
            ("voice.stt_api_key", None): None,
            ("voice_api.stt_quota_tokens", 200): 200,
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Hello world"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(_voice_mod, "_check_quota", new_callable=AsyncMock), \
             patch.object(_crud, "get_config_json", new_callable=AsyncMock,
                          side_effect=lambda db, key, default: config_map.get((key, default), default)), \
             patch.object(_voice_mod, "_record_and_complete", new_callable=AsyncMock) as record, \
             patch.object(_voice_mod.httpx, "AsyncClient", return_value=mock_client):

            result = await stt_transcriptions(
                http_request, file=file, model=None, language=None,
                db=db, auth=(user, api_key),
            )

            assert result.status_code == 200
            import json
            body = json.loads(result.body.decode())
            assert body["text"] == "Hello world"

            # Verify request was recorded as success
            record.assert_called_once()
            call_kwargs = record.call_args[1]
            assert call_kwargs["endpoint"] == "/v1/audio/transcriptions"
            assert call_kwargs["token_cost"] == 200
            assert call_kwargs.get("error_message") is None

    @pytest.mark.asyncio
    async def test_stt_disabled_returns_404(self):
        """STT disabled in config returns 404."""
        db = _make_mock_db()
        user = _make_mock_user()
        api_key = _make_mock_api_key()
        http_request = _make_mock_request()
        file = self._make_upload_file()

        config_map = {
            ("voice.stt_enabled", False): False,
        }

        with patch.object(_voice_mod, "_check_quota", new_callable=AsyncMock), \
             patch.object(_crud, "get_config_json", new_callable=AsyncMock,
                          side_effect=lambda db, key, default: config_map.get((key, default), default)):
            with pytest.raises(HTTPException) as exc:
                await stt_transcriptions(
                    http_request, file=file, model=None, language=None,
                    db=db, auth=(user, api_key),
                )
            assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_stt_url_not_configured_returns_500(self):
        """STT enabled but no URL configured returns 500."""
        db = _make_mock_db()
        user = _make_mock_user()
        api_key = _make_mock_api_key()
        http_request = _make_mock_request()
        file = self._make_upload_file()

        config_map = {
            ("voice.stt_enabled", False): True,
            ("voice.stt_url", None): None,
        }

        with patch.object(_voice_mod, "_check_quota", new_callable=AsyncMock), \
             patch.object(_crud, "get_config_json", new_callable=AsyncMock,
                          side_effect=lambda db, key, default: config_map.get((key, default), default)):
            with pytest.raises(HTTPException) as exc:
                await stt_transcriptions(
                    http_request, file=file, model=None, language=None,
                    db=db, auth=(user, api_key),
                )
            assert exc.value.status_code == 500

    @pytest.mark.asyncio
    async def test_stt_upstream_error_returns_502_and_records_failure(self):
        """Upstream STT returns non-200 → 502 + failure recorded."""
        db = _make_mock_db()
        user = _make_mock_user()
        api_key = _make_mock_api_key()
        http_request = _make_mock_request()
        file = self._make_upload_file()

        config_map = {
            ("voice.stt_enabled", False): True,
            ("voice.stt_url", None): "http://stt:8080",
            ("voice.stt_model", "whisper-large-v3-turbo"): "whisper-large-v3-turbo",
            ("voice.stt_api_key", None): None,
            ("voice_api.stt_quota_tokens", 200): 200,
        }

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(_voice_mod, "_check_quota", new_callable=AsyncMock), \
             patch.object(_crud, "get_config_json", new_callable=AsyncMock,
                          side_effect=lambda db, key, default: config_map.get((key, default), default)), \
             patch.object(_voice_mod, "_record_and_complete", new_callable=AsyncMock) as record, \
             patch.object(_voice_mod.httpx, "AsyncClient", return_value=mock_client):

            with pytest.raises(HTTPException) as exc:
                await stt_transcriptions(
                    http_request, file=file, model=None, language=None,
                    db=db, auth=(user, api_key),
                )
            assert exc.value.status_code == 502

            # Failure should be recorded with zero token cost
            record.assert_called_once()
            call_kwargs = record.call_args[1]
            assert call_kwargs["token_cost"] == 0
            assert "error_message" in call_kwargs
            assert call_kwargs["error_message"] is not None

    @pytest.mark.asyncio
    async def test_stt_timeout_returns_502(self):
        """Upstream STT timeout → 502."""
        db = _make_mock_db()
        user = _make_mock_user()
        api_key = _make_mock_api_key()
        http_request = _make_mock_request()
        file = self._make_upload_file()

        config_map = {
            ("voice.stt_enabled", False): True,
            ("voice.stt_url", None): "http://stt:8080",
            ("voice.stt_model", "whisper-large-v3-turbo"): "whisper-large-v3-turbo",
            ("voice.stt_api_key", None): None,
            ("voice_api.stt_quota_tokens", 200): 200,
        }

        import httpx as real_httpx

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=real_httpx.ReadTimeout("timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(_voice_mod, "_check_quota", new_callable=AsyncMock), \
             patch.object(_crud, "get_config_json", new_callable=AsyncMock,
                          side_effect=lambda db, key, default: config_map.get((key, default), default)), \
             patch.object(_voice_mod, "_record_and_complete", new_callable=AsyncMock) as record, \
             patch.object(_voice_mod.httpx, "AsyncClient", return_value=mock_client):

            with pytest.raises(HTTPException) as exc:
                await stt_transcriptions(
                    http_request, file=file, model=None, language=None,
                    db=db, auth=(user, api_key),
                )
            assert exc.value.status_code == 502
            assert "timed out" in exc.value.detail

    @pytest.mark.asyncio
    async def test_stt_http_error_returns_502(self):
        """Upstream STT connection error → 502."""
        db = _make_mock_db()
        user = _make_mock_user()
        api_key = _make_mock_api_key()
        http_request = _make_mock_request()
        file = self._make_upload_file()

        config_map = {
            ("voice.stt_enabled", False): True,
            ("voice.stt_url", None): "http://stt:8080",
            ("voice.stt_model", "whisper-large-v3-turbo"): "whisper-large-v3-turbo",
            ("voice.stt_api_key", None): None,
            ("voice_api.stt_quota_tokens", 200): 200,
        }

        import httpx as real_httpx

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=real_httpx.ConnectError("connection refused")
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(_voice_mod, "_check_quota", new_callable=AsyncMock), \
             patch.object(_crud, "get_config_json", new_callable=AsyncMock,
                          side_effect=lambda db, key, default: config_map.get((key, default), default)), \
             patch.object(_voice_mod, "_record_and_complete", new_callable=AsyncMock), \
             patch.object(_voice_mod.httpx, "AsyncClient", return_value=mock_client):

            with pytest.raises(HTTPException) as exc:
                await stt_transcriptions(
                    http_request, file=file, model=None, language=None,
                    db=db, auth=(user, api_key),
                )
            assert exc.value.status_code == 502
            assert "unavailable" in exc.value.detail

    @pytest.mark.asyncio
    async def test_stt_custom_model_parameter(self):
        """Custom model parameter is used instead of DB default."""
        db = _make_mock_db()
        user = _make_mock_user()
        api_key = _make_mock_api_key()
        http_request = _make_mock_request()
        file = self._make_upload_file()

        config_map = {
            ("voice.stt_enabled", False): True,
            ("voice.stt_url", None): "http://stt:8080",
            ("voice.stt_api_key", None): None,
            ("voice_api.stt_quota_tokens", 200): 200,
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Transcribed"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(_voice_mod, "_check_quota", new_callable=AsyncMock), \
             patch.object(_crud, "get_config_json", new_callable=AsyncMock,
                          side_effect=lambda db, key, default: config_map.get((key, default), default)), \
             patch.object(_voice_mod, "_record_and_complete", new_callable=AsyncMock) as record, \
             patch.object(_voice_mod.httpx, "AsyncClient", return_value=mock_client):

            await stt_transcriptions(
                http_request, file=file, model="whisper-small", language=None,
                db=db, auth=(user, api_key),
            )

            call_kwargs = record.call_args[1]
            assert call_kwargs["model"] == "whisper-small"

    @pytest.mark.asyncio
    async def test_stt_language_passthrough(self):
        """Language parameter is forwarded to upstream service."""
        db = _make_mock_db()
        user = _make_mock_user()
        api_key = _make_mock_api_key()
        http_request = _make_mock_request()
        file = self._make_upload_file()

        config_map = {
            ("voice.stt_enabled", False): True,
            ("voice.stt_url", None): "http://stt:8080",
            ("voice.stt_model", "whisper-large-v3-turbo"): "whisper-large-v3-turbo",
            ("voice.stt_api_key", None): None,
            ("voice_api.stt_quota_tokens", 200): 200,
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Bonjour"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(_voice_mod, "_check_quota", new_callable=AsyncMock), \
             patch.object(_crud, "get_config_json", new_callable=AsyncMock,
                          side_effect=lambda db, key, default: config_map.get((key, default), default)), \
             patch.object(_voice_mod, "_record_and_complete", new_callable=AsyncMock), \
             patch.object(_voice_mod.httpx, "AsyncClient", return_value=mock_client):

            result = await stt_transcriptions(
                http_request, file=file, model=None, language="fr",
                db=db, auth=(user, api_key),
            )

            # Verify the upstream call included language in data
            post_call = mock_client.post.call_args
            assert post_call[1]["data"]["language"] == "fr"

    @pytest.mark.asyncio
    async def test_stt_upstream_api_key_forwarded(self):
        """When voice.stt_api_key is set, it's forwarded upstream."""
        db = _make_mock_db()
        user = _make_mock_user()
        api_key = _make_mock_api_key()
        http_request = _make_mock_request()
        file = self._make_upload_file()

        config_map = {
            ("voice.stt_enabled", False): True,
            ("voice.stt_url", None): "http://stt:8080",
            ("voice.stt_model", "whisper-large-v3-turbo"): "whisper-large-v3-turbo",
            ("voice.stt_api_key", None): "sk-stt-key",
            ("voice_api.stt_quota_tokens", 200): 200,
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Hello"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(_voice_mod, "_check_quota", new_callable=AsyncMock), \
             patch.object(_crud, "get_config_json", new_callable=AsyncMock,
                          side_effect=lambda db, key, default: config_map.get((key, default), default)), \
             patch.object(_voice_mod, "_record_and_complete", new_callable=AsyncMock), \
             patch.object(_voice_mod.httpx, "AsyncClient", return_value=mock_client):

            await stt_transcriptions(
                http_request, file=file, model=None, language=None,
                db=db, auth=(user, api_key),
            )

            post_call = mock_client.post.call_args
            assert post_call[1]["headers"]["Authorization"] == "Bearer sk-stt-key"

    # --- 2.8.43: OpenAI model-name aliasing + actionable upstream-404 ------

    def _stt_config(self, default_model="deepdml/faster-whisper-large-v3-turbo-ct2"):
        return {
            ("voice.stt_enabled", False): True,
            ("voice.stt_url", None): "http://stt:8080",
            ("voice.stt_model", "whisper-large-v3-turbo"): default_model,
            ("voice.stt_api_key", None): None,
            ("voice_api.stt_quota_tokens", 200): 200,
        }

    def _stt_client(self, status=200, body=None, text=""):
        mock_response = MagicMock()
        mock_response.status_code = status
        mock_response.json.return_value = body or {"text": "ok"}
        mock_response.text = text
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        return mock_client

    @pytest.mark.asyncio
    async def test_stt_openai_model_names_alias_to_configured_model(self):
        """whisper-1/whisper/etc. must map to the operator-configured model —
        forwarding them verbatim 404s the Whisper server (opaque 502 pre-2.8.43,
        broke every OpenAI-conventional client while the web mic worked)."""
        for alias in ("whisper-1", "Whisper", "gpt-4o-transcribe", "default-stt"):
            client = self._stt_client()
            with patch.object(_voice_mod, "_check_quota", new_callable=AsyncMock), \
                 patch.object(_crud, "get_config_json", new_callable=AsyncMock,
                              side_effect=lambda db, key, default: self._stt_config().get((key, default), default)), \
                 patch.object(_voice_mod, "_record_and_complete", new_callable=AsyncMock), \
                 patch.object(_voice_mod.httpx, "AsyncClient", return_value=client):
                await stt_transcriptions(
                    _make_mock_request(), file=self._make_upload_file(), model=alias,
                    language=None, db=_make_mock_db(), auth=(_make_mock_user(), _make_mock_api_key()),
                )
                sent = client.post.call_args[1]["data"]["model"]
                assert sent == "deepdml/faster-whisper-large-v3-turbo-ct2", alias

    @pytest.mark.asyncio
    async def test_stt_explicit_nonalias_model_passes_through(self):
        """A real (non-alias) model name is forwarded verbatim, so a second
        installed model stays reachable."""
        client = self._stt_client()
        with patch.object(_voice_mod, "_check_quota", new_callable=AsyncMock), \
             patch.object(_crud, "get_config_json", new_callable=AsyncMock,
                          side_effect=lambda db, key, default: self._stt_config().get((key, default), default)), \
             patch.object(_voice_mod, "_record_and_complete", new_callable=AsyncMock), \
             patch.object(_voice_mod.httpx, "AsyncClient", return_value=client):
            await stt_transcriptions(
                _make_mock_request(), file=self._make_upload_file(),
                model="Systran/faster-whisper-small", language=None,
                db=_make_mock_db(), auth=(_make_mock_user(), _make_mock_api_key()),
            )
            assert client.post.call_args[1]["data"]["model"] == "Systran/faster-whisper-small"

    @pytest.mark.asyncio
    async def test_stt_upstream_404_returns_actionable_400(self):
        """Upstream 'model not installed' 404 → 400 naming the configured
        default, not an opaque 502."""
        client = self._stt_client(status=404, text='{"detail":"Model not installed"}')
        with patch.object(_voice_mod, "_check_quota", new_callable=AsyncMock), \
             patch.object(_crud, "get_config_json", new_callable=AsyncMock,
                          side_effect=lambda db, key, default: self._stt_config().get((key, default), default)), \
             patch.object(_voice_mod, "_record_and_complete", new_callable=AsyncMock), \
             patch.object(_voice_mod.httpx, "AsyncClient", return_value=client):
            with pytest.raises(HTTPException) as exc:
                await stt_transcriptions(
                    _make_mock_request(), file=self._make_upload_file(),
                    model="Systran/faster-whisper-large-v3", language=None,
                    db=_make_mock_db(), auth=(_make_mock_user(), _make_mock_api_key()),
                )
            assert exc.value.status_code == 400
            assert "deepdml/faster-whisper-large-v3-turbo-ct2" in exc.value.detail
            assert "Systran/faster-whisper-large-v3" in exc.value.detail


# ================================================================
# Modality enum
# ================================================================


class TestModalityEnum:
    """Test that TTS and STT modality values exist."""

    @pytest.mark.skipif(Modality is None, reason="Could not load Modality enum")
    def test_tts_modality_exists(self):
        assert Modality.TTS.value == "tts"

    @pytest.mark.skipif(Modality is None, reason="Could not load Modality enum")
    def test_stt_modality_exists(self):
        assert Modality.STT.value == "stt"

    @pytest.mark.skipif(Modality is None, reason="Could not load Modality enum")
    def test_modality_values_complete(self):
        """All expected modality values are present."""
        values = {m.value for m in Modality}
        assert "tts" in values
        assert "stt" in values
        assert "chat" in values
        assert "embedding" in values
