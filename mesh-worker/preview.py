############################################################
#
# mindrouter mesh-worker - async image-to-3D mesh service
#
# preview.py: 512x512 PNG thumbnail of a finished mesh — a small software
#     renderer (numpy + Pillow, no GPU, no pyrender/OpenGL), shared by every
#     engine so one code path produces every preview.
#
#   Orthographic 3/4 view (glTF is Y-up), painter's algorithm (far faces
#   first), two-sided Lambert shading, 2x supersampling, transparent
#   background. Face colour comes from the base-colour texture sampled at
#   each face's UV centroid, from vertex colours, or a neutral grey.
#
# Luke Sheneman — University of Idaho RCDS — sheneman@uidaho.edu
#
############################################################

"""Software preview renderer."""

import io
import math
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

PREVIEW_SIZE = 512
_SUPERSAMPLE = 2
_GREY = np.array([170, 174, 180], dtype=np.float32)


def face_colors_from_texture(faces: np.ndarray, uvs: np.ndarray, texture: Image.Image) -> np.ndarray:
    """(M,3) uint8 colours: the texture sampled at each face's UV centroid
    (glTF UV convention: v=0 is the TOP row of the image)."""
    tex = np.asarray(texture.convert("RGB"), dtype=np.uint8)
    h, w = tex.shape[:2]
    centroid = uvs[faces].mean(axis=1)
    u = np.clip(np.mod(centroid[:, 0], 1.0), 0.0, 1.0)
    v = np.clip(np.mod(centroid[:, 1], 1.0), 0.0, 1.0)
    x = np.minimum((u * w).astype(np.int64), w - 1)
    y = np.minimum((v * h).astype(np.int64), h - 1)
    return tex[y, x]


def face_colors_from_vertices(faces: np.ndarray, colors: np.ndarray) -> np.ndarray:
    cols = np.asarray(colors, dtype=np.float32)[:, :3]
    if cols.max(initial=0.0) <= 1.0:
        cols = cols * 255.0
    return np.clip(cols[faces].mean(axis=1), 0, 255).astype(np.uint8)


def render_preview(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_colors: Optional[np.ndarray] = None,
    size: int = PREVIEW_SIZE,
    azimuth_deg: float = 35.0,
    elevation_deg: float = 25.0,
) -> bytes:
    """Render the mesh and return PNG bytes (size x size, RGBA)."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    canvas = size * _SUPERSAMPLE
    img = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))

    if len(v) and len(f):
        # Centre + fit the bounding box.
        lo, hi = v.min(axis=0), v.max(axis=0)
        v = v - (lo + hi) / 2.0
        # Rotate: yaw about +Y, then pitch about +X (camera looks down -Z).
        az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
        ry = np.array([[math.cos(az), 0, math.sin(az)], [0, 1, 0], [-math.sin(az), 0, math.cos(az)]])
        rx = np.array([[1, 0, 0], [0, math.cos(el), -math.sin(el)], [0, math.sin(el), math.cos(el)]])
        v = v @ (rx @ ry).T
        extent = float(np.abs(v[:, :2]).max()) or 1.0
        scale = 0.45 * canvas / extent
        xy = np.empty((len(v), 2))
        xy[:, 0] = canvas / 2 + v[:, 0] * scale
        xy[:, 1] = canvas / 2 - v[:, 1] * scale  # image y grows downward

        tri = v[f]                                  # (M,3,3) view space
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        norm = np.linalg.norm(normals, axis=1)
        keep = norm > 1e-12                         # drop degenerate faces
        normals = normals[keep] / norm[keep, None]
        f = f[keep]
        depth = tri[keep][:, :, 2].mean(axis=1)

        light = np.array([-0.35, 0.55, 0.75])
        light /= np.linalg.norm(light)
        # Two-sided: generated meshes do not always have consistent winding.
        shade = 0.38 + 0.62 * np.abs(normals @ light)
        if face_colors is None:
            base = np.broadcast_to(_GREY, (len(f), 3))
        else:
            base = np.asarray(face_colors, dtype=np.float32)[keep][:, :3]
        rgb = np.clip(base * shade[:, None], 0, 255).astype(np.uint8)

        order = np.argsort(depth)                   # far (-z) first
        draw = ImageDraw.Draw(img)
        pts = xy[f[order]]
        cols = rgb[order]
        for p, c in zip(pts.tolist(), cols.tolist()):
            colour = (c[0], c[1], c[2], 255)
            draw.polygon([tuple(p[0]), tuple(p[1]), tuple(p[2])], fill=colour, outline=colour)

    img = img.resize((size, size), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=False)
    return out.getvalue()
