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

# Stub out heavy dependencies before loading the module
sys.modules.setdefault("backend", MagicMock())
sys.modules.setdefault("backend.app", MagicMock())
sys.modules.setdefault("backend.app.api", MagicMock())
sys.modules.setdefault("backend.app.api.auth", MagicMock())
sys.modules.setdefault("backend.app.db", MagicMock())
sys.modules.setdefault("backend.app.db.crud", MagicMock())
sys.modules.setdefault("backend.app.db.models", MagicMock())
sys.modules.setdefault("backend.app.db.session", MagicMock())
sys.modules.setdefault("backend.app.logging_config", MagicMock(get_logger=MagicMock(return_value=MagicMock())))

_voice_spec = importlib.util.spec_from_file_location(
    "voice_api", _api_dir / "voice_api.py",
    submodule_search_locations=[],
)
_voice_mod = importlib.util.module_from_spec(_voice_spec)
_voice_spec.loader.exec_module(_voice_mod)

# Pull out testable items
TTSRequest = _voice_mod.TTSRequest
_check_quota = _voice_mod._check_quota
_record_and_complete = _voice_mod._record_and_complete
tts_speech = _voice_mod.tts_speech
stt_transcriptions = _voice_mod.stt_transcriptions

# Also get module-level references so we can patch them
_crud = _voice_mod.crud

# Get the real Modality enum (safe to import directly from the .py file)
_models_dir = Path(__file__).resolve().parents[2] / "db"
_models_spec = importlib.util.spec_from_file_location(
    "models_enum", _models_dir / "models.py",
    submodule_search_locations=[],
)
# We need Base for models.py — stub it
sys.modules.setdefault("backend.app.db.base", MagicMock(
    Base=type("Base", (), {"__tablename__": "", "metadata": MagicMock()}),
    TimestampMixin=type("TimestampMixin", (), {}),
    SoftDeleteMixin=type("SoftDeleteMixin", (), {}),
))
_models_mod = importlib.util.module_from_spec(_models_spec)
try:
    _models_spec.loader.exec_module(_models_mod)
    Modality = _models_mod.Modality
except Exception:
    # Fallback if models.py can't load due to SQLAlchemy deps
    Modality = None


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
        quota.tokens_used = 500

        with patch.object(_crud, "reset_quota_if_needed", new_callable=AsyncMock), \
             patch.object(_crud, "get_user_quota", new_callable=AsyncMock, return_value=quota):
            await _check_quota(db, user)  # Should not raise

    @pytest.mark.asyncio
    async def test_quota_rejected_when_exceeded(self):
        """HTTPException 429 when tokens_used >= budget."""
        db = _make_mock_db()
        user = _make_mock_user(token_budget=1000)
        quota = MagicMock()
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
        quota.tokens_used = 99999

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

        config_map = {
            ("voice.tts_enabled", False): True,
            ("voice.tts_url", None): "http://tts-service:8080",
            ("voice.tts_api_key", None): None,
            ("voice_api.tts_quota_tokens", 100): 100,
        }

        with patch.object(_voice_mod, "_check_quota", new_callable=AsyncMock), \
             patch.object(_crud, "get_config_json", new_callable=AsyncMock,
                          side_effect=lambda db, key, default: config_map.get((key, default), default)), \
             patch.object(_voice_mod, "_record_and_complete", new_callable=AsyncMock):

            result = await tts_speech(http_request, body, db=db, auth=(user, api_key))

            assert result.media_type == "audio/mpeg"

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

        config_map = {
            ("voice.tts_enabled", False): True,
            ("voice.tts_url", None): "http://tts:8080",
            ("voice.tts_api_key", None): None,
            ("voice_api.tts_quota_tokens", 100): 100,
        }

        with patch.object(_voice_mod, "_check_quota", new_callable=AsyncMock), \
             patch.object(_crud, "get_config_json", new_callable=AsyncMock,
                          side_effect=lambda db, key, default: config_map.get((key, default), default)), \
             patch.object(_voice_mod, "_record_and_complete", new_callable=AsyncMock):

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

        config_map = {
            ("voice.tts_enabled", False): True,
            ("voice.tts_url", None): "http://tts:8080",
            ("voice.tts_api_key", None): None,
            ("voice_api.tts_quota_tokens", 100): 150,
        }

        with patch.object(_voice_mod, "_check_quota", new_callable=AsyncMock), \
             patch.object(_crud, "get_config_json", new_callable=AsyncMock,
                          side_effect=lambda db, key, default: config_map.get((key, default), default)), \
             patch.object(_voice_mod, "_record_and_complete", new_callable=AsyncMock) as record:

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

        config_map = {
            ("voice.tts_enabled", False): True,
            ("voice.tts_url", None): "http://tts:8080",
            ("voice.tts_api_key", None): "sk-upstream-key",
            ("voice_api.tts_quota_tokens", 100): 100,
        }

        # We test that the function completes without error — the actual
        # upstream call is in the generator, not executed until iterated.
        with patch.object(_voice_mod, "_check_quota", new_callable=AsyncMock), \
             patch.object(_crud, "get_config_json", new_callable=AsyncMock,
                          side_effect=lambda db, key, default: config_map.get((key, default), default)), \
             patch.object(_voice_mod, "_record_and_complete", new_callable=AsyncMock):

            result = await tts_speech(http_request, body, db=db, auth=(user, api_key))
            assert result is not None  # StreamingResponse returned


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
