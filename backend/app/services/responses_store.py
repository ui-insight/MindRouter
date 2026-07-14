############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# responses_store.py: Server-side state for the OpenAI Responses
# API (store=true / previous_response_id chains).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Persistence and chain reconstruction for /v1/responses.

Design constraints (see docs/responses-api-plan.md §2):

- Each row stores only the request's OWN delta input items plus its
  output items.  Chains are rebuilt at request time by walking
  previous_response_id root->leaf and concatenating; storing rebuilt
  input would duplicate ancestors once per generation (O(N^2)).
- Large inline images are offloaded to the artifact disk, and the
  mapping lives in a server-controlled column (offloaded_images) —
  NEVER in the item JSON itself, so a client cannot forge an offload
  reference and read arbitrary server files.  Re-inflation validates
  that the resolved path stays inside this row's own artifact
  directory.
- Hard caps: max stored payload bytes per response (oversized rows are
  persisted without item JSON and flagged, so billing/GET still work
  but chaining fails loudly), and max stored rows per user (oldest
  rows are evicted at insert time).
- Persistence always opens its own DB session and commits; the
  streaming path calls it from a finally block under asyncio.shield
  (dashboard/chat.py precedent) so client disconnects still leave a
  row for any planned chain.
"""

import base64
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.app.db import crud
from backend.app.db.models import StoredResponse, StoredResponseStatus
from backend.app.db.session import get_async_db_context
from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

logger = get_logger(__name__)

_ID_PREFIX_BY_TYPE = {
    "message": "msg",
    "function_call": "fc",
    "function_call_output": "fco",
    "reasoning": "rs",
}

_MEDIA_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
}

_STATUS_BY_NAME = {s.value: s for s in StoredResponseStatus}


def _artifact_dir(rel_key: str, subdir: str = "responses_store") -> Path:
    """Artifact directory for one stored object (a stored response, or a
    conversation item as ``<conv_id>/<item_id>``).

    Keys are always server-generated, but validate anyway so a crafted
    id can never traverse outside the store root.
    """
    root = Path(get_settings().artifact_storage_path) / subdir
    target = (root / rel_key).resolve()
    if not str(target).startswith(str(root.resolve()) + "/"):
        raise ValueError(f"invalid artifact key path: {rel_key!r}")
    return target


def remove_artifacts(rel_key: str, subdir: str = "responses_store") -> None:
    """Best-effort removal of a stored object's offloaded files."""
    try:
        target = _artifact_dir(rel_key, subdir)
        if target.exists():
            shutil.rmtree(target)
    except Exception as e:
        logger.warning("responses_store_artifact_cleanup_failed",
                       rel_key=rel_key, error=str(e))


# ---------------------------------------------------------------------------
# Item id stamping + image offload
# ---------------------------------------------------------------------------

def stamp_item_ids(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure every item has a stable id (used by input_items pagination
    and item_reference resolution)."""
    stamped = []
    for item in items:
        item = dict(item)
        if not item.get("id"):
            item_type = item.get("type") or ("message" if "role" in item else "item")
            prefix = _ID_PREFIX_BY_TYPE.get(item_type, "item")
            item["id"] = f"{prefix}_{uuid.uuid4().hex}"
        stamped.append(item)
    return stamped


def offload_images(
    items: List[Dict[str, Any]],
    rel_key: str,
    subdir: str = "responses_store",
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Extract large inline data-URI images from input items to disk.

    Returns (items_with_placeholders, offload_map).  The item JSON gets
    a bare placeholder string; the authoritative mapping (placeholder
    position -> filename + media type) lives only in the returned map,
    which is persisted in the server-controlled offloaded_images
    column.  Re-inflation reads exclusively from that map.
    """
    offload_map: Dict[str, Any] = {}
    out_items: List[Dict[str, Any]] = []
    img_idx = 0
    images_dir: Optional[Path] = None

    for i, item in enumerate(items):
        content = item.get("content")
        if not isinstance(content, list):
            out_items.append(item)
            continue
        new_content = []
        item_copy = dict(item)
        for j, part in enumerate(content):
            if (
                isinstance(part, dict)
                and part.get("type") == "input_image"
                and isinstance(part.get("image_url"), str)
                and part["image_url"].startswith("data:")
                and len(part["image_url"]) > 1024
            ):
                try:
                    header, b64data = part["image_url"].split(",", 1)
                    media_type = header.split(":")[1].split(";")[0]
                    ext = _MEDIA_EXT.get(media_type, "bin")
                    if images_dir is None:
                        images_dir = _artifact_dir(rel_key, subdir)
                        images_dir.mkdir(parents=True, exist_ok=True)
                    filename = f"{img_idx}.{ext}"
                    (images_dir / filename).write_bytes(
                        base64.b64decode(b64data, validate=False)
                    )
                    part = dict(part)
                    part["image_url"] = "__mindrouter_offloaded__"
                    offload_map[f"{i}.{j}"] = {
                        "file": filename,
                        "media_type": media_type,
                    }
                    img_idx += 1
                except Exception as e:
                    logger.warning("responses_store_offload_failed", error=str(e))
            new_content.append(part)
        item_copy["content"] = new_content
        out_items.append(item_copy)

    return out_items, (offload_map or None)


def reinflate_item_images(
    items: List[Dict[str, Any]],
    offload_map: Optional[Dict[str, Any]],
    rel_key: str,
    subdir: str = "responses_store",
) -> List[Dict[str, Any]]:
    """Restore offloaded images in an item list as data URIs.

    Reads ONLY from the server-written offload map and refuses any
    resolved path outside the object's own artifact directory.
    """
    items = [dict(i) for i in items]
    if not offload_map:
        return items

    base_dir = _artifact_dir(rel_key, subdir)
    for key, meta in offload_map.items():
        try:
            i_str, j_str = key.split(".", 1)
            i, j = int(i_str), int(j_str)
            filename = str(meta.get("file", ""))
            resolved = (base_dir / filename).resolve()
            if not str(resolved).startswith(str(base_dir.resolve()) + "/"):
                logger.warning(
                    "responses_store_offload_path_rejected",
                    rel_key=rel_key, file=filename,
                )
                continue
            data = base64.b64encode(resolved.read_bytes()).decode("ascii")
            media_type = meta.get("media_type", "image/png")
            part = dict(items[i]["content"][j])
            part["image_url"] = f"data:{media_type};base64,{data}"
            content = list(items[i]["content"])
            content[j] = part
            items[i] = dict(items[i])
            items[i]["content"] = content
        except Exception as e:
            logger.warning(
                "responses_store_reinflate_failed",
                rel_key=rel_key, key=key, error=str(e),
            )
    return items


def reinflate_images(stored: StoredResponse) -> List[Dict[str, Any]]:
    """Return a stored response row's input items with offloaded images
    restored as data URIs."""
    return reinflate_item_images(
        list(stored.input_items or []),
        stored.offloaded_images,
        stored.response_id,
    )


# ---------------------------------------------------------------------------
# Chain reconstruction
# ---------------------------------------------------------------------------

def normalize_input_to_items(input_value: Any) -> List[Dict[str, Any]]:
    """Normalize the polymorphic ``input`` field to an item list."""
    if input_value is None:
        return []
    if isinstance(input_value, str):
        return [{"type": "message", "role": "user", "content": input_value}]
    if isinstance(input_value, list):
        return [dict(i) for i in input_value if isinstance(i, dict)]
    raise ValueError("'input' must be a string or an array of items")


def rebuild_input_from_chain(
    chain: List[StoredResponse],
    new_input_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Concatenate each chain row's delta input + output items
    (root->leaf), then the new request's items.

    Per spec, instructions/tools/params are NOT inherited from the
    chain — the current request's own values apply.  item_reference
    items in the new input are resolved against the chain's stamped
    item ids.
    """
    history: List[Dict[str, Any]] = []
    by_id: Dict[str, Dict[str, Any]] = {}

    for stored in chain:
        if stored.parameters and stored.parameters.get("payload_too_large"):
            raise ValueError(
                f"Previous response '{stored.response_id}' exceeded the "
                "storage payload limit and cannot be replayed."
            )
        for item in reinflate_images(stored):
            history.append(item)
            if item.get("id"):
                by_id[item["id"]] = item
        for item in stored.output_items or []:
            history.append(item)
            if item.get("id"):
                by_id[item["id"]] = item

    resolved_new: List[Dict[str, Any]] = []
    for item in new_input_items:
        if item.get("type") == "item_reference":
            ref = by_id.get(item.get("id") or "")
            if ref is None:
                raise ValueError(
                    f"item_reference '{item.get('id')}' not found in the "
                    "previous_response_id chain"
                )
            resolved_new.append(ref)
        else:
            resolved_new.append(item)

    return history + resolved_new


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

async def persist_response(
    ctx,
    input_items: List[Dict[str, Any]],
    output_items: List[Dict[str, Any]],
    usage: Optional[Dict[str, Any]],
    status: str,
    error: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist one completed/failed response in its own DB session.

    Safe to call from a streaming generator's finally block under
    asyncio.shield.  Never raises.
    """
    settings = get_settings()
    try:
        input_items = stamp_item_ids(input_items)
        output_items = stamp_item_ids(output_items or [])
        input_items, offload_map = offload_images(input_items, ctx.response_id)

        parameters = ctx.to_stored_parameters()

        payload_bytes = len(json.dumps(input_items)) + len(json.dumps(output_items))
        max_bytes = settings.responses_store_max_payload_bytes
        if max_bytes and payload_bytes > max_bytes:
            logger.warning(
                "responses_store_payload_too_large",
                response_id=ctx.response_id,
                payload_bytes=payload_bytes,
                max_bytes=max_bytes,
            )
            input_items, output_items = None, None
            parameters["payload_too_large"] = True
            remove_artifacts(ctx.response_id)
            offload_map = None

        async with get_async_db_context() as db:
            # Per-user cap: evict oldest rows to make room.
            max_rows = settings.responses_store_max_rows_per_user
            if max_rows:
                count = await crud.count_stored_responses_for_user(db, ctx.user_id)
                if count >= max_rows:
                    evict = await crud.get_oldest_stored_responses_for_user(
                        db, ctx.user_id, count - max_rows + 1
                    )
                    for old in evict:
                        old_id = old.response_id
                        await db.delete(old)
                        remove_artifacts(old_id)
                    logger.info(
                        "responses_store_evicted",
                        user_id=ctx.user_id, count=len(evict),
                    )

            await crud.create_stored_response(
                db,
                response_id=ctx.response_id,
                user_id=ctx.user_id,
                api_key_id=ctx.api_key_id,
                model=ctx.model,
                status=_STATUS_BY_NAME.get(status, StoredResponseStatus.COMPLETED),
                previous_response_id=ctx.previous_response_id,
                input_items=input_items,
                output_items=output_items,
                instructions=ctx.instructions,
                parameters=parameters,
                usage=usage,
                error=error,
                offloaded_images=offload_map,
            )
            await db.commit()
        logger.info("responses_store_persisted", response_id=ctx.response_id,
                    status=status)
    except Exception as e:
        logger.warning(
            "responses_store_persist_failed",
            response_id=getattr(ctx, "response_id", "?"), error=str(e),
        )
