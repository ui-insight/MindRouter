############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_responses_store.py: Unit tests for the Responses API
# server-side store (persistence, chains, image offload).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Covers:
- Item id stamping
- Image offload to disk + out-of-band map; re-inflation; path
  containment rejection (the arbitrary-file-read guard)
- Chain rebuild ordering, item_reference resolution, oversized-row
  refusal
- persist_response: delta-only storage, payload cap, per-user
  row-cap eviction, never-raises contract
- CRUD owner-scoping and migration wiring (source-text assertions)
"""

import base64
import importlib.util
import sys
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class _FakeStoredResponseStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"
    CANCELLED = "cancelled"


# ----------------------------------------------------------------
# Direct-load responses_store.py with heavy leaves stubbed.  The
# module iterates StoredResponseStatus at import time, so the models
# stub must carry a real enum replica.
# ----------------------------------------------------------------

_models_stub = MagicMock()
_models_stub.StoredResponseStatus = _FakeStoredResponseStatus
_models_stub.StoredResponse = MagicMock

_STUB_NAMES = {
    "backend.app.db": MagicMock(),
    "backend.app.db.crud": MagicMock(),
    "backend.app.db.models": _models_stub,
    "backend.app.db.session": MagicMock(),
    "backend.app.logging_config": MagicMock(
        get_logger=MagicMock(return_value=MagicMock())
    ),
    "backend.app.settings": MagicMock(),
}
_added = []
for _name, _stub in _STUB_NAMES.items():
    if _name not in sys.modules:
        sys.modules[_name] = _stub
        _added.append(_name)

_svc_dir = Path(__file__).resolve().parents[2] / "services"
_spec = importlib.util.spec_from_file_location(
    "responses_store", _svc_dir / "responses_store.py",
    submodule_search_locations=[],
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

for _name in _added:
    sys.modules.pop(_name, None)


def _settings(tmp_path, max_payload=5242880, max_rows=1000):
    s = MagicMock()
    s.artifact_storage_path = str(tmp_path)
    s.responses_store_max_payload_bytes = max_payload
    s.responses_store_max_rows_per_user = max_rows
    return s


def _big_data_uri(media="image/png", size=2048):
    return f"data:{media};base64," + base64.b64encode(b"x" * size).decode()


def _make_stored(response_id="resp_r1", input_items=None, output_items=None,
                 offloaded_images=None, parameters=None, previous=None):
    stored = MagicMock()
    stored.response_id = response_id
    stored.user_id = 1
    stored.api_key_id = 42
    stored.model = "m"
    stored.input_items = input_items or []
    stored.output_items = output_items or []
    stored.offloaded_images = offloaded_images
    stored.parameters = parameters or {}
    stored.previous_response_id = previous
    stored.instructions = None
    return stored


class TestStampItemIds:
    def test_stamps_by_type(self):
        items = _mod.stamp_item_ids([
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "function_call", "call_id": "call_1", "name": "f",
             "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
            {"role": "user", "content": "typeless"},
        ])
        assert items[0]["id"].startswith("msg_")
        assert items[1]["id"].startswith("fc_")
        assert items[2]["id"].startswith("fco_")
        assert items[3]["id"].startswith("msg_")

    def test_preserves_existing_ids(self):
        items = _mod.stamp_item_ids([{"type": "message", "id": "msg_keep"}])
        assert items[0]["id"] == "msg_keep"


class TestImageOffload:
    def test_offload_and_reinflate_round_trip(self, tmp_path):
        uri = _big_data_uri()
        items = [{
            "type": "message", "role": "user",
            "content": [
                {"type": "input_text", "text": "look:"},
                {"type": "input_image", "image_url": uri},
            ],
        }]
        with patch.object(_mod, "get_settings", return_value=_settings(tmp_path)):
            out_items, offload_map = _mod.offload_images(items, "resp_t1")
            # Placeholder replaced the URI; map holds the real location
            part = out_items[0]["content"][1]
            assert part["image_url"] == "__mindrouter_offloaded__"
            assert offload_map == {
                "0.1": {"file": "0.png", "media_type": "image/png"}
            }
            saved = tmp_path / "responses_store" / "resp_t1" / "0.png"
            assert saved.exists()

            stored = _make_stored(
                "resp_t1", input_items=out_items, offloaded_images=offload_map
            )
            restored = _mod.reinflate_images(stored)
            assert restored[0]["content"][1]["image_url"] == uri
            # Text part untouched
            assert restored[0]["content"][0]["text"] == "look:"

    def test_small_and_non_data_images_not_offloaded(self, tmp_path):
        items = [{
            "type": "message", "role": "user",
            "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,QUJD"},
                {"type": "input_image", "image_url": "https://x.example/i.png"},
            ],
        }]
        with patch.object(_mod, "get_settings", return_value=_settings(tmp_path)):
            out_items, offload_map = _mod.offload_images(items, "resp_t2")
        assert offload_map is None
        assert out_items[0]["content"][0]["image_url"].startswith("data:")

    def test_reinflate_rejects_path_escape(self, tmp_path):
        # A forged/corrupted map entry must never read outside the row's
        # own artifact directory.
        secret = tmp_path / "secret.txt"
        secret.write_text("password")
        stored = _make_stored(
            "resp_t3",
            input_items=[{
                "type": "message", "role": "user",
                "content": [{"type": "input_image",
                             "image_url": "__mindrouter_offloaded__"}],
            }],
            offloaded_images={
                "0.0": {"file": "../../secret.txt", "media_type": "image/png"}
            },
        )
        with patch.object(_mod, "get_settings", return_value=_settings(tmp_path)):
            restored = _mod.reinflate_images(stored)
        # Placeholder remains; secret content never inlined
        assert restored[0]["content"][0]["image_url"] == "__mindrouter_offloaded__"

    def test_artifact_dir_rejects_traversal_ids(self, tmp_path):
        with patch.object(_mod, "get_settings", return_value=_settings(tmp_path)):
            with pytest.raises(ValueError):
                _mod._artifact_dir("../escape")


class TestNormalizeInput:
    def test_string(self):
        items = _mod.normalize_input_to_items("hello")
        assert items == [{"type": "message", "role": "user", "content": "hello"}]

    def test_list_and_none(self):
        assert _mod.normalize_input_to_items(None) == []
        items = _mod.normalize_input_to_items([{"role": "user", "content": "x"}])
        assert items == [{"role": "user", "content": "x"}]

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _mod.normalize_input_to_items(42)


class TestChainRebuild:
    def test_root_to_leaf_concatenation(self, tmp_path):
        root = _make_stored(
            "resp_root",
            input_items=[{"id": "msg_1", "type": "message", "role": "user",
                          "content": "turn 1"}],
            output_items=[{"id": "msg_2", "type": "message", "role": "assistant",
                           "content": [{"type": "output_text", "text": "answer 1"}]}],
        )
        leaf = _make_stored(
            "resp_leaf",
            input_items=[{"id": "msg_3", "type": "message", "role": "user",
                          "content": "turn 2"}],
            output_items=[{"id": "msg_4", "type": "message", "role": "assistant",
                           "content": [{"type": "output_text", "text": "answer 2"}]}],
            previous="resp_root",
        )
        new_items = [{"type": "message", "role": "user", "content": "turn 3"}]
        with patch.object(_mod, "get_settings", return_value=_settings(tmp_path)):
            combined = _mod.rebuild_input_from_chain([root, leaf], new_items)
        ids = [i.get("id") for i in combined]
        assert ids == ["msg_1", "msg_2", "msg_3", "msg_4", None]
        assert combined[-1]["content"] == "turn 3"

    def test_item_reference_resolution(self, tmp_path):
        root = _make_stored(
            "resp_root",
            input_items=[{"id": "msg_ref", "type": "message", "role": "user",
                          "content": "referenced"}],
        )
        with patch.object(_mod, "get_settings", return_value=_settings(tmp_path)):
            combined = _mod.rebuild_input_from_chain(
                [root], [{"type": "item_reference", "id": "msg_ref"}]
            )
        assert combined[-1]["content"] == "referenced"

        with patch.object(_mod, "get_settings", return_value=_settings(tmp_path)):
            with pytest.raises(ValueError):
                _mod.rebuild_input_from_chain(
                    [root], [{"type": "item_reference", "id": "msg_ghost"}]
                )

    def test_oversized_row_refuses_replay(self, tmp_path):
        big = _make_stored("resp_big", parameters={"payload_too_large": True})
        with patch.object(_mod, "get_settings", return_value=_settings(tmp_path)):
            with pytest.raises(ValueError):
                _mod.rebuild_input_from_chain([big], [])


class TestPersistResponse:
    def _ctx(self):
        ctx = MagicMock()
        ctx.response_id = "resp_p1"
        ctx.user_id = 1
        ctx.api_key_id = 42
        ctx.model = "m"
        ctx.previous_response_id = None
        ctx.instructions = None
        ctx.to_stored_parameters = MagicMock(return_value={"model": "m"})
        return ctx

    def _db_context(self, db):
        @asynccontextmanager
        async def ctx_mgr():
            yield db
        return ctx_mgr

    async def test_persists_delta_items(self, tmp_path):
        db = AsyncMock()
        crud = MagicMock()
        crud.count_stored_responses_for_user = AsyncMock(return_value=0)
        crud.create_stored_response = AsyncMock()
        with patch.object(_mod, "get_settings", return_value=_settings(tmp_path)), \
             patch.object(_mod, "crud", crud), \
             patch.object(_mod, "get_async_db_context", self._db_context(db)):
            await _mod.persist_response(
                self._ctx(),
                [{"type": "message", "role": "user", "content": "hi"}],
                [{"type": "message", "role": "assistant",
                  "content": [{"type": "output_text", "text": "yo"}]}],
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                "completed",
            )
        kwargs = crud.create_stored_response.await_args.kwargs
        assert kwargs["response_id"] == "resp_p1"
        assert kwargs["status"] == _FakeStoredResponseStatus.COMPLETED
        assert kwargs["input_items"][0]["id"].startswith("msg_")
        assert kwargs["output_items"][0]["content"][0]["text"] == "yo"
        db.commit.assert_awaited()

    async def test_payload_cap_drops_items_and_flags(self, tmp_path):
        db = AsyncMock()
        crud = MagicMock()
        crud.count_stored_responses_for_user = AsyncMock(return_value=0)
        crud.create_stored_response = AsyncMock()
        with patch.object(_mod, "get_settings",
                          return_value=_settings(tmp_path, max_payload=10)), \
             patch.object(_mod, "crud", crud), \
             patch.object(_mod, "get_async_db_context", self._db_context(db)):
            await _mod.persist_response(
                self._ctx(),
                [{"type": "message", "role": "user", "content": "x" * 100}],
                [], None, "completed",
            )
        kwargs = crud.create_stored_response.await_args.kwargs
        assert kwargs["input_items"] is None
        assert kwargs["output_items"] is None
        assert kwargs["parameters"]["payload_too_large"] is True

    async def test_per_user_cap_evicts_oldest(self, tmp_path):
        db = AsyncMock()
        old_row = MagicMock()
        old_row.response_id = "resp_old"
        crud = MagicMock()
        crud.count_stored_responses_for_user = AsyncMock(return_value=5)
        crud.get_oldest_stored_responses_for_user = AsyncMock(return_value=[old_row])
        crud.create_stored_response = AsyncMock()
        with patch.object(_mod, "get_settings",
                          return_value=_settings(tmp_path, max_rows=5)), \
             patch.object(_mod, "crud", crud), \
             patch.object(_mod, "get_async_db_context", self._db_context(db)):
            await _mod.persist_response(self._ctx(), [], [], None, "completed")
        crud.get_oldest_stored_responses_for_user.assert_awaited_once()
        db.delete.assert_awaited_once_with(old_row)

    async def test_never_raises(self, tmp_path):
        crud = MagicMock()
        crud.count_stored_responses_for_user = AsyncMock(
            side_effect=RuntimeError("db down")
        )
        with patch.object(_mod, "get_settings", return_value=_settings(tmp_path)), \
             patch.object(_mod, "crud", crud), \
             patch.object(_mod, "get_async_db_context",
                          self._db_context(AsyncMock())):
            # Must swallow — called from streaming finally blocks.
            await _mod.persist_response(self._ctx(), [], [], None, "completed")


class TestSourceContracts:
    """Source-text assertions (house style: test_model_enrichment.py)."""

    _crud_src = (Path(__file__).resolve().parents[2] / "db" / "crud.py").read_text()

    def test_crud_owner_scoping(self):
        assert "StoredResponse.user_id == user_id" in self._crud_src
        assert "async def get_stored_response(" in self._crud_src
        assert "async def get_stored_response_chain(" in self._crud_src

    def test_chain_walk_has_cycle_and_depth_guards(self):
        assert "chain contains a cycle" in self._crud_src
        assert "exceeds maximum depth" in self._crud_src

    def test_migration_wiring(self):
        versions = Path(__file__).resolve().parents[2] / "db" / "migrations" / "versions"
        mig = next(versions.glob("*_062_add_stored_responses.py")).read_text()
        assert 'revision = "062"' in mig
        assert 'down_revision = "061"' in mig
        assert "stored_responses" in mig
        assert "ix_stored_responses_created_at" in mig

    def test_retention_sweep_registered(self):
        retention_src = (
            Path(__file__).resolve().parents[2] / "services" / "retention.py"
        ).read_text()
        assert '"retention.responses_store_days": 30' in retention_src
        assert "async def cleanup_expired_stored_responses(" in retention_src
        assert "cleanup_expired_stored_responses(" in retention_src.split(
            "async def run_retention_cycle"
        )[1]
