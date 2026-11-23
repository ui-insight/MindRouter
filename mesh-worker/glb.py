############################################################
#
# mindrouter mesh-worker - async image-to-3D mesh service
#
# glb.py: Minimal glTF 2.0 binary (GLB) writer + reader, numpy/struct only.
#
#   write_glb()  — the mock engine's output (and a fallback for engines that
#                  hand back plain arrays).
#   inspect_glb() — structural validation of EVERY engine's output before a
#                  job is marked completed (header, chunk layout, JSON, the
#                  accessors referenced by each triangle primitive), and the
#                  face/vertex counts reported on the status object.
#
# Luke Sheneman — University of Idaho RCDS — sheneman@uidaho.edu
#
############################################################

"""GLB (binary glTF 2.0) writing and validation."""

import json
import struct
from typing import Any, Dict, Optional

import numpy as np

GLB_MAGIC = 0x46546C67      # b"glTF"
CHUNK_JSON = 0x4E4F534A     # b"JSON"
CHUNK_BIN = 0x004E4942      # b"BIN\0"

_FLOAT = 5126
_UINT32 = 5125
_UINT16 = 5123
_UINT8 = 5121
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963
_TRIANGLES = 4

_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_COMPONENT_BYTES = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}


class GLBError(ValueError):
    """The bytes are not a structurally valid GLB."""


def _pad(data: bytes, fill: bytes) -> bytes:
    return data + fill * ((4 - len(data) % 4) % 4)


def write_glb(
    positions: np.ndarray,
    indices: np.ndarray,
    *,
    normals: Optional[np.ndarray] = None,
    uvs: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    texture_png: Optional[bytes] = None,
    base_color: tuple = (1.0, 1.0, 1.0, 1.0),
    metallic: float = 0.0,
    roughness: float = 0.8,
    generator: str = "mindrouter mesh-worker",
) -> bytes:
    """Serialize one triangle mesh to GLB bytes, deterministically.

    positions (N,3) float, indices (M,3) int, optional normals (N,3), uvs (N,2),
    colors (N,3|4) float in [0,1], texture_png (embedded baseColorTexture,
    needs uvs).
    """
    positions = np.ascontiguousarray(positions, dtype="<f4")
    indices = np.ascontiguousarray(indices, dtype="<u4").reshape(-1)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise ValueError("positions must be a non-empty (N, 3) array")
    if len(indices) == 0 or len(indices) % 3:
        raise ValueError("indices must hold a whole number of triangles")
    if int(indices.max()) >= len(positions):
        raise ValueError("index out of range")
    if texture_png is not None and uvs is None:
        raise ValueError("a texture needs uvs")

    blob = bytearray()
    views, accessors, attributes = [], [], {}

    def add_view(data: bytes, target: Optional[int]) -> int:
        offset = len(blob)
        blob.extend(_pad(data, b"\x00"))
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        views.append(view)
        return len(views) - 1

    def add_accessor(arr: np.ndarray, ctype: int, atype: str, target: Optional[int], minmax=False) -> int:
        view = add_view(arr.tobytes(), target)
        acc = {"bufferView": view, "componentType": ctype, "count": int(arr.shape[0]), "type": atype}
        if minmax:  # required for POSITION
            acc["min"] = [float(x) for x in arr.min(axis=0)]
            acc["max"] = [float(x) for x in arr.max(axis=0)]
        accessors.append(acc)
        return len(accessors) - 1

    attributes["POSITION"] = add_accessor(positions, _FLOAT, "VEC3", _ARRAY_BUFFER, minmax=True)
    if normals is not None:
        attributes["NORMAL"] = add_accessor(np.ascontiguousarray(normals, dtype="<f4"), _FLOAT, "VEC3", _ARRAY_BUFFER)
    if uvs is not None:
        attributes["TEXCOORD_0"] = add_accessor(np.ascontiguousarray(uvs, dtype="<f4"), _FLOAT, "VEC2", _ARRAY_BUFFER)
    if colors is not None:
        cols = np.ascontiguousarray(colors, dtype="<f4")
        attributes["COLOR_0"] = add_accessor(cols, _FLOAT, "VEC4" if cols.shape[1] == 4 else "VEC3", _ARRAY_BUFFER)
    index_acc = add_accessor(indices, _UINT32, "SCALAR", _ELEMENT_ARRAY_BUFFER)

    pbr: Dict[str, Any] = {
        "baseColorFactor": [float(c) for c in base_color],
        "metallicFactor": float(metallic),
        "roughnessFactor": float(roughness),
    }
    gltf: Dict[str, Any] = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "mesh"}],
        "meshes": [{"primitives": [{"attributes": attributes, "indices": index_acc,
                                    "material": 0, "mode": _TRIANGLES}]}],
        "materials": [{"pbrMetallicRoughness": pbr, "doubleSided": True, "name": "material"}],
    }
    if texture_png is not None:
        image_view = add_view(texture_png, None)
        gltf["images"] = [{"bufferView": image_view, "mimeType": "image/png"}]
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}]
        gltf["textures"] = [{"sampler": 0, "source": 0}]
        pbr["baseColorTexture"] = {"index": 0, "texCoord": 0}
    gltf["accessors"] = accessors
    gltf["bufferViews"] = views
    gltf["buffers"] = [{"byteLength": len(blob)}]

    json_chunk = _pad(json.dumps(gltf, sort_keys=True, separators=(",", ":")).encode("utf-8"), b" ")
    bin_chunk = bytes(blob)
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    return b"".join([
        struct.pack("<III", GLB_MAGIC, 2, total),
        struct.pack("<II", len(json_chunk), CHUNK_JSON), json_chunk,
        struct.pack("<II", len(bin_chunk), CHUNK_BIN), bin_chunk,
    ])


def parse_glb(data: bytes) -> tuple:
    """Split GLB bytes into (gltf_json: dict, bin_chunk: bytes|None).
    Raises GLBError on any header / chunk inconsistency."""
    if len(data) < 20:
        raise GLBError("file too short for a GLB header")
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        raise GLBError("bad magic (not binary glTF)")
    if version != 2:
        raise GLBError(f"unsupported glTF container version {version}")
    if length != len(data):
        raise GLBError(f"header length {length} != file size {len(data)}")
    offset, gltf, binary = 12, None, None
    while offset < length:
        if offset + 8 > length:
            raise GLBError("truncated chunk header")
        clen, ctype = struct.unpack_from("<II", data, offset)
        offset += 8
        if offset + clen > length:
            raise GLBError("chunk overruns the file")
        chunk = data[offset:offset + clen]
        offset += clen
        if gltf is None:
            if ctype != CHUNK_JSON:
                raise GLBError("first chunk must be JSON")
            try:
                gltf = json.loads(chunk.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise GLBError(f"JSON chunk does not parse: {exc}")
        elif ctype == CHUNK_BIN and binary is None:
            binary = chunk
        # Unknown chunk types must be ignored per the spec.
    if not isinstance(gltf, dict):
        raise GLBError("missing JSON chunk")
    return gltf, binary


def inspect_glb(data: bytes) -> Dict[str, int]:
    """Validate the structure and count triangles/vertices over every
    triangle primitive. Returns {"faces", "vertices"}."""
    gltf, binary = parse_glb(data)
    if str(gltf.get("asset", {}).get("version", "")).split(".")[0] != "2":
        raise GLBError("asset.version must be 2.x")
    accessors = gltf.get("accessors") or []
    views = gltf.get("bufferViews") or []
    buffers = gltf.get("buffers") or []
    bin_len = len(binary) if binary is not None else 0

    def check_accessor(idx: Any) -> Dict[str, Any]:
        if not isinstance(idx, int) or not (0 <= idx < len(accessors)):
            raise GLBError(f"accessor {idx!r} out of range")
        acc = accessors[idx]
        count = acc.get("count")
        if not isinstance(count, int) or count < 0:
            raise GLBError(f"accessor {idx} has a bad count")
        if "bufferView" in acc:
            vidx = acc["bufferView"]
            if not isinstance(vidx, int) or not (0 <= vidx < len(views)):
                raise GLBError(f"accessor {idx} bufferView out of range")
            view = views[vidx]
            if view.get("buffer", 0) != 0 or not buffers or "uri" in buffers[0]:
                raise GLBError("only the embedded GLB buffer is supported")
            end = view.get("byteOffset", 0) + view.get("byteLength", 0)
            if end > bin_len:
                raise GLBError(f"bufferView {vidx} overruns the BIN chunk")
            elem = _COMPONENTS.get(acc.get("type"), 0) * _COMPONENT_BYTES.get(acc.get("componentType"), 0)
            if elem == 0:
                raise GLBError(f"accessor {idx} has an unknown type")
            stride = view.get("byteStride") or elem
            need = acc.get("byteOffset", 0) + (stride * (count - 1) + elem if count else 0)
            if need > view.get("byteLength", 0):
                raise GLBError(f"accessor {idx} overruns its bufferView")
        return acc

    faces = vertices = 0
    seen_positions = set()
    for mesh in gltf.get("meshes") or []:
        for prim in mesh.get("primitives") or []:
            pos_idx = (prim.get("attributes") or {}).get("POSITION")
            if pos_idx is None:
                raise GLBError("primitive without POSITION")
            pos = check_accessor(pos_idx)
            if prim.get("mode", _TRIANGLES) != _TRIANGLES:
                continue
            if pos_idx not in seen_positions:
                seen_positions.add(pos_idx)
                vertices += pos["count"]
            if "indices" in prim:
                faces += check_accessor(prim["indices"])["count"] // 3
            else:
                faces += pos["count"] // 3
    if faces == 0:
        raise GLBError("no triangles")
    return {"faces": faces, "vertices": vertices}
