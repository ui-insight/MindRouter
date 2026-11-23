"""Unit tests for the GLB writer/validator, the software preview renderer and
input-image decoding (mock deps only)."""

import io
import json
import struct

import numpy as np
import pytest
from PIL import Image

from config import WorkerConfig
from conftest import b64, image_bytes
from glb import GLBError, inspect_glb, parse_glb, write_glb
from inputs import RequestError, decode_image, open_image, validate_request
from preview import face_colors_from_texture, render_preview

TRI = (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32),
       np.array([[0, 1, 2], [1, 3, 2]]))


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── GLB writer ────────────────────────────────────────────────────────────

def test_write_glb_layout_and_counts():
    pos, faces = TRI
    data = write_glb(pos, faces, uvs=np.zeros((4, 2)), texture_png=_png(Image.new("RGB", (4, 4))))
    magic, version, length = struct.unpack_from("<III", data, 0)
    assert (magic, version, length) == (0x46546C67, 2, len(data))
    json_len, _ = struct.unpack_from("<II", data, 12)
    assert json_len % 4 == 0 and data[20 + json_len - 1:20 + json_len] in (b" ", b"}")
    gltf, binary = parse_glb(data)
    assert len(binary) % 4 == 0
    for view in gltf["bufferViews"]:
        assert view["byteOffset"] % 4 == 0
    pos_acc = gltf["accessors"][gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    assert pos_acc["min"] == [0.0, 0.0, 0.0] and pos_acc["max"] == [1.0, 1.0, 0.0]
    assert gltf["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] == {"index": 0, "texCoord": 0}
    assert inspect_glb(data) == {"faces": 2, "vertices": 4}


def test_write_glb_is_deterministic():
    pos, faces = TRI
    assert write_glb(pos, faces) == write_glb(pos, faces)


@pytest.mark.parametrize("bad", [
    lambda: write_glb(np.zeros((0, 3)), np.zeros((0, 3))),
    lambda: write_glb(TRI[0], np.array([0, 1])),
    lambda: write_glb(TRI[0], np.array([[0, 1, 9]])),
    lambda: write_glb(TRI[0], TRI[1], texture_png=b"x"),     # texture without uvs
])
def test_write_glb_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        bad()


def _rebuild(gltf: dict, binary: bytes) -> bytes:
    js = json.dumps(gltf).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(binary)
    return (struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(binary), 0x004E4942) + binary)


def test_inspect_glb_rejects_corrupt_files():
    good = write_glb(*TRI)
    gltf, binary = parse_glb(good)
    with pytest.raises(GLBError):
        inspect_glb(b"glTF")                                      # too short
    with pytest.raises(GLBError):
        inspect_glb(b"XXXX" + good[4:])                           # bad magic
    with pytest.raises(GLBError):
        inspect_glb(good[:-4])                                    # length mismatch
    with pytest.raises(GLBError):
        inspect_glb(good[:12] + struct.pack("<II", 4, 0x004E4942) + good[20:])  # BIN first
    overrun = json.loads(json.dumps(gltf))
    overrun["bufferViews"][0]["byteLength"] = len(binary) * 10
    with pytest.raises(GLBError):
        inspect_glb(_rebuild(overrun, binary))
    no_tris = json.loads(json.dumps(gltf))
    no_tris["meshes"] = []
    with pytest.raises(GLBError):
        inspect_glb(_rebuild(no_tris, binary))
    bad_acc = json.loads(json.dumps(gltf))
    bad_acc["meshes"][0]["primitives"][0]["indices"] = 99
    with pytest.raises(GLBError):
        inspect_glb(_rebuild(bad_acc, binary))


# ── preview ───────────────────────────────────────────────────────────────

def test_render_preview_is_512_png_with_transparent_background():
    from engines.mock import _box

    pos, _uv, faces = _box(8, 0.3)
    png = render_preview(pos, faces)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    img = Image.open(io.BytesIO(png))
    assert img.size == (512, 512) and img.mode == "RGBA"
    assert img.getpixel((2, 2))[3] == 0                  # background
    assert img.getpixel((256, 256))[3] == 255            # object


def test_render_preview_handles_empty_and_degenerate_meshes():
    for verts, faces in ((np.zeros((0, 3)), np.zeros((0, 3), int)),
                         (np.zeros((3, 3)), np.array([[0, 1, 2]]))):
        img = Image.open(io.BytesIO(render_preview(verts, faces)))
        assert img.size == (512, 512)


def test_face_colors_sample_the_texture_top_left_origin():
    tex = Image.new("RGB", (2, 2))
    tex.putpixel((0, 0), (255, 0, 0))     # top-left
    tex.putpixel((1, 1), (0, 0, 255))     # bottom-right
    uvs = np.array([[0.1, 0.1], [0.2, 0.1], [0.1, 0.2], [0.9, 0.9], [0.8, 0.9], [0.9, 0.8]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    cols = face_colors_from_texture(faces, uvs, tex)
    assert cols[0].tolist() == [255, 0, 0] and cols[1].tolist() == [0, 0, 255]


# ── input decoding ────────────────────────────────────────────────────────

def test_alpha_detection():
    cfg = WorkerConfig(mode="mock")
    *_, has_alpha = decode_image(b64(image_bytes(alpha=True)), cfg)
    assert has_alpha is True
    *_, has_alpha = decode_image(b64(image_bytes(alpha=False)), cfg)   # opaque RGBA
    assert has_alpha is False
    *_, has_alpha = decode_image(b64(image_bytes(fmt="JPEG", mode="RGB", alpha=False)), cfg)
    assert has_alpha is False
    pal = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    pal.paste((255, 0, 0, 255), (16, 16, 48, 48))
    *_, has_alpha = decode_image(b64(_png(pal.convert("P"))), cfg)
    assert has_alpha is True


def test_open_image_hands_engines_rgba_or_rgb():
    cfg = WorkerConfig(mode="mock")
    assert open_image(validate_request({"image": b64(image_bytes(alpha=True))}, cfg)).mode == "RGBA"
    assert open_image(validate_request({"image": b64(image_bytes(alpha=False))}, cfg)).mode == "RGB"


def test_decompression_bomb_is_a_client_error():
    cfg = WorkerConfig(mode="mock", max_image_pixels=100 * 100)
    with pytest.raises(RequestError):
        decode_image(b64(image_bytes(size=(200, 200))), cfg)


def test_seed_defaults_to_random_when_omitted():
    cfg = WorkerConfig(mode="mock")
    seeds = {validate_request({"image": b64(image_bytes())}, cfg).seed for _ in range(5)}
    assert len(seeds) > 1
    assert validate_request({"image": b64(image_bytes()), "seed": 0}, cfg).seed == 0
