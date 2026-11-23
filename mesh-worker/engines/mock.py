############################################################
#
# mindrouter mesh-worker - async image-to-3D mesh service
#
# engines/mock.py: No-GPU engine for dev, CI and gateway load tests.
#
#   Deterministic from (seed, image, texture, texture_size, target_faces): a
#   subdivided cube "pillowed" toward a sphere by a seed-dependent amount.
#   texture=true wraps the (alpha-cropped) input image onto every side as an
#   embedded PNG baseColorTexture; texture=false uses seeded vertex colours.
#   Walks the same stages as the real engines, with an optional per-stage
#   delay (MESH_WORKER_MOCK_STAGE_DELAY) so progress + cancel can be tested.
#
# Luke Sheneman — University of Idaho RCDS — sheneman@uidaho.edu
#
############################################################

"""Mock mesh engine (numpy + Pillow only)."""

import colorsys
import io
import time
from typing import Any, Dict

import numpy as np
from PIL import Image

from config import WorkerConfig
from engines.base import StageTracker
from glb import write_glb
from preview import face_colors_from_texture, face_colors_from_vertices, render_preview

# 12 * 64^2 = 49,152 faces: keeps the mock (and its software preview) fast.
MAX_SUBDIVISIONS = 64
# Embedded mock textures are capped; the real bake honours texture_size.
MAX_MOCK_TEXTURE = 512

# Cube sides: (normal axis, sign, u axis, v axis).
_SIDES = [(2, 1, 0, 1), (2, -1, 0, 1), (0, 1, 2, 1), (0, -1, 2, 1), (1, 1, 0, 2), (1, -1, 0, 2)]


def subdivisions_for(target_faces: int) -> int:
    return int(min(MAX_SUBDIVISIONS, max(1, round((target_faces / 12) ** 0.5))))


def _box(n: int, roundness: float):
    """Six n x n grids (separate vertices per side so each side has its own UVs).
    Returns positions (6(n+1)^2, 3), uvs, faces (12 n^2, 3)."""
    lin = np.linspace(-1.0, 1.0, n + 1)
    gu, gv = np.meshgrid(lin, lin, indexing="xy")          # (n+1, n+1)
    uv = np.stack([(gu + 1) / 2, 1 - (gv + 1) / 2], -1).reshape(-1, 2)
    pos_all, uv_all, faces_all = [], [], []
    for side, (axis, sign, ua, va) in enumerate(_SIDES):
        p = np.zeros(((n + 1) ** 2, 3))
        p[:, axis] = sign
        p[:, ua] = gu.reshape(-1) * (sign if axis != 1 else 1)
        p[:, va] = gv.reshape(-1)
        base = side * (n + 1) ** 2
        idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1) + base
        a, b = idx[:-1, :-1].reshape(-1), idx[:-1, 1:].reshape(-1)
        c, d = idx[1:, :-1].reshape(-1), idx[1:, 1:].reshape(-1)
        quads = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
        pos_all.append(p)
        uv_all.append(uv)
        faces_all.append(quads)
    pos = np.concatenate(pos_all)
    sphere = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    pos = (1 - roundness) * pos + roundness * sphere
    faces = np.concatenate(faces_all)
    # Orient every face outward (the grids above are not consistently wound).
    tri = pos[faces]
    n_out = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = np.einsum("ij,ij->i", n_out, tri.mean(axis=1)) < 0
    faces[flip] = faces[flip][:, ::-1]
    return pos * 0.5, np.concatenate(uv_all), faces


def _vertex_normals(pos: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = pos[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(pos)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    norm = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.where(norm > 0, norm, 1)


def _subject(image: Image.Image) -> Image.Image:
    """Crop to the alpha foreground (like the real pipelines), flatten on white."""
    if image.mode == "RGBA":
        bbox = image.getchannel("A").point(lambda a: 255 if a > 204 else 0).getbbox()
        if bbox:
            image = image.crop(bbox)
        flat = Image.new("RGB", image.size, (255, 255, 255))
        flat.paste(image, mask=image.getchannel("A"))
        return flat
    return image.convert("RGB")


class MockEngine:
    def __init__(self, config: WorkerConfig):
        self.config = config

    def load(self) -> None:
        return None

    def _sleep(self, tracker: StageTracker) -> None:
        delay = self.config.mock_stage_delay
        if delay > 0:
            time.sleep(delay)

    def generate(self, spec, image: Image.Image, glb_path: str, preview_path: str,
                 tracker: StageTracker) -> Dict[str, Any]:
        rng = np.random.default_rng(spec.seed)

        tracker.enter("preprocess", 5.0)
        subject = _subject(image)
        self._sleep(tracker)

        tracker.enter("shape", 25.0)
        n = subdivisions_for(spec.target_faces)
        pos, uvs, faces = _box(n, roundness=float(rng.uniform(0.1, 0.7)))
        self._sleep(tracker)

        texture_png = None
        colors = None
        if spec.texture:
            tracker.enter("texture", 55.0)
            side = min(spec.texture_size, MAX_MOCK_TEXTURE)
            tex = subject.resize((side, side), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            tex.save(buf, format="PNG", optimize=False)
            texture_png = buf.getvalue()
            self._sleep(tracker)
        else:
            hue = float(rng.uniform(0, 1))
            base = np.array(colorsys.hsv_to_rgb(hue, 0.55, 0.85))
            shade = 0.75 + 0.25 * (pos[:, 1:2] / 0.5 + 1) / 2   # lighter toward the top
            colors = np.clip(base[None, :] * shade, 0, 1)

        tracker.enter("export", 85.0)
        data = write_glb(
            pos, faces,
            normals=_vertex_normals(pos, faces),
            uvs=uvs if texture_png is not None else None,
            colors=colors,
            texture_png=texture_png,
            generator=f"mindrouter mesh-worker mock ({self.config.model_id})",
        )
        with open(glb_path, "wb") as fh:
            fh.write(data)
        if texture_png is not None:
            fc = face_colors_from_texture(faces, uvs, tex)
        else:
            fc = face_colors_from_vertices(faces, colors)
        png = render_preview(pos, faces, fc)
        with open(preview_path, "wb") as fh:
            fh.write(png)
        self._sleep(tracker)
        return {"faces": int(len(faces)), "vertices": int(len(pos))}
