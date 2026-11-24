############################################################
#
# mindrouter mesh-worker - async image-to-3D mesh service
#
# engines/hunyuan3d21.py: Tencent Hunyuan3D 2.1, shape (hy3dshape DiT flow
#     matching) then optional PBR paint (hy3dpaint multiview diffusion +
#     RealESRGAN + UV bake), IN-PROCESS. We never run upstream api_server.py:
#     it has no auth, blocks its event loop and ignores seed/face/texture.
#
#   Source read for this adapter (pinned in docker/hunyuan3d21/Dockerfile):
#     Tencent-Hunyuan/Hunyuan3D-2.1 @ 82920d643c0dc2f7bfd7255f45f62d386edfe60c
#       hy3dshape/hy3dshape/{pipelines,postprocessors,rembg}.py,
#       hy3dpaint/textureGenPipeline.py, hy3dpaint/utils/multiview_utils.py,
#       hy3dpaint/DifferentiableRenderer/MeshRender.py, gradio_app.py
#
#   Upstream facts this adapter works around:
#     - paint has NO seed: it reseeds the global RNGs to 0 on every call, so
#       only the shape is seed-dependent.
#     - paint takes a mesh FILE path and writes .obj/.mtl/.jpg next to it, so
#       each job gets its own scratch dir (removed afterwards).
#     - use_remesh=True re-decimates to a hard-coded 40k faces; we only use it
#       when target_faces <= 40k, so larger budgets are honoured.
#     - the bake atlas (MeshRender texture_size, default 4096) is halved on
#       save; we set it to 2x the requested size per job, then resize the
#       saved maps to exactly texture_size.
#     - upstream GLB conversion needs Blender (bpy) or writes into the CWD;
#       we assemble the PBR GLB ourselves with trimesh.
#     - hy3dshape post-processors leak NamedTemporaryFile(delete=False) .ply
#       files; we delete the ones a job created.
#     - DifferentiableRenderer/mesh_utils.py imports bpy at MODULE level, so
#       bpy must be installed even though we never call it.
#
#   Every call into upstream ran on an H200 in the Nov 2026 bake-off
#   (18/18 inputs at 100k faces / 2048 px, plus 8k-1M faces, 4096 px).
#
# Luke Sheneman — University of Idaho RCDS — sheneman@uidaho.edu
#
############################################################

"""Hunyuan3D 2.1 engine."""

import glob
import logging
import os
import shutil
import sys
import tempfile
from typing import Any, Dict

import numpy as np
from PIL import Image

from config import WorkerConfig
from engines.base import EngineInputError, StageTracker
from engines.common import (
    fast_decimate,
    jpeg_textures,
    pull_push_fill_gpu,
    release_cuda_memory,
    stage_hooks,
    trimesh_preview_png,
)

log = logging.getLogger(__name__)

PAINT_REMESH_FACES = 40_000      # hard-coded in hy3dpaint remesh_mesh()
MAX_BAKE_ATLAS = 8192


def paint_plan(target_faces: int, texture_size: int) -> Dict[str, int]:
    """Per-job paint settings (pure function, unit-tested)."""
    return {
        "use_remesh": int(target_faces <= PAINT_REMESH_FACES),
        "bake_atlas": min(MAX_BAKE_ATLAS, 2 * int(texture_size)),
    }


class Hunyuan3D21Engine:
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.shape = None
        self.paint = None
        self.rembg = None
        self._post = None
        self._torch = None
        self._trimesh = None

    def capabilities(self) -> Dict[str, Any]:
        return {"supports_texture": True}

    # -- loading -----------------------------------------------------------
    def _sys_path(self) -> None:
        """Same precedence as upstream gradio_app.py (run from the repo root):
        hy3dpaint, then hy3dshape, then the repo root."""
        repo = self.config.hunyuan_repo
        for p in (repo, os.path.join(repo, "hy3dshape"), os.path.join(repo, "hy3dpaint")):
            if p in sys.path:
                sys.path.remove(p)
            sys.path.insert(0, p)

    def load(self) -> None:  # pragma: no cover - GPU node only
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        self._sys_path()
        import torch
        import trimesh

        self._torch = torch
        self._trimesh = trimesh
        repo = self.config.hunyuan_repo
        ckpt = self.config.checkpoint

        # Verified on the GPU (bake-off, Nov 2026): basicsr (RealESRGAN) imports the removed
        # torchvision.transforms.functional_tensor; upstream patches it first.
        from torchvision_fix import apply_fix
        apply_fix()

        # Verified on the GPU (bake-off, Nov 2026): import paths + from_pretrained signature at the pinned
        # commit. ckpt is an absolute dir holding hunyuan3d-dit-v2-1/, or the
        # repo id resolved from the HF cache (HF_HUB_OFFLINE=1).
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dshape.postprocessors import DegenerateFaceRemover, FaceReducer, FloaterRemover
        from hy3dshape.rembg import BackgroundRemover

        log.info("Loading Hunyuan3D 2.1 shape from %s…", ckpt)
        self.shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            ckpt, subfolder="hunyuan3d-dit-v2-1", use_safetensors=False, device="cuda",
        )
        self._post = (FloaterRemover(), DegenerateFaceRemover(), FaceReducer())
        # rembg u2net (ONNX, CPU) for RGB inputs only; u2net.onnx must be
        # pre-staged in $U2NET_HOME or rembg downloads it on first use.
        self.rembg = BackgroundRemover()

        # Verified on the GPU (bake-off, Nov 2026): paint config attributes. Every default path upstream is
        # relative to the CWD, so all are set absolute here. The multiview
        # weights are fetched with snapshot_download(repo_id=...), which needs a
        # REPO ID (HF cache, offline), never a local dir.
        from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

        conf = Hunyuan3DPaintConfig(6, 512)          # (max_num_view, resolution): gradio defaults
        conf.multiview_cfg_path = os.path.join(repo, "hy3dpaint", "cfgs", "hunyuan-paint-pbr.yaml")
        conf.multiview_pretrained_path = ckpt if not os.path.isdir(ckpt) else "tencent/Hunyuan3D-2.1"
        conf.dino_ckpt_path = self.config.hunyuan_dino
        conf.realesrgan_ckpt_path = self.config.realesrgan_ckpt or os.path.join(
            repo, "hy3dpaint", "ckpt", "RealESRGAN_x4plus.pth")
        log.info("Loading Hunyuan3D 2.1 paint (multiview %s)…", conf.multiview_pretrained_path)
        self.paint = Hunyuan3DPaintPipeline(conf)
        if self.config.hunyuan_fast_post:
            import fast_simplification  # noqa: F401  (fail at startup, not mid-job)
            self._install_fast_inpaint()
            log.info("Hunyuan3D fast post-processing on: fast-simplification + GPU pull-push inpainting")

    def _install_fast_inpaint(self) -> None:  # pragma: no cover - GPU node only
        """Replace the paint renderer's uv_inpaint on THIS instance: keep
        upstream's mesh-aware vertex inpainting (it fills texels inside the UV
        charts from neighbouring painted vertices), then fill the rest of the
        atlas with a GPU pull-push instead of OpenCV's Navier-Stokes over the
        whole atlas (5-8 s per map at 4096 px -> ~0.02 s)."""
        import types

        import torch
        from DifferentiableRenderer import MeshRender as mesh_render_module

        vertex_inpaint = mesh_render_module.meshVerticeInpaint

        def uv_inpaint(render, texture, mask, vertex_inpaint_on=True, method="NS", return_float=False):
            if isinstance(texture, torch.Tensor):
                texture_np = texture.cpu().numpy()
            elif isinstance(texture, np.ndarray):
                texture_np = texture
            else:
                texture_np = np.array(texture) / 255.0
            if isinstance(mask, torch.Tensor):
                mask = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
            if vertex_inpaint_on:
                vtx_pos, pos_idx, vtx_uv, uv_idx = render.get_mesh()
                texture_np, mask = vertex_inpaint(texture_np, mask, vtx_pos, vtx_uv, pos_idx, uv_idx)
            return pull_push_fill_gpu((np.clip(texture_np, 0, 1) * 255).astype(np.uint8), np.asarray(mask) > 0)

        render = self.paint.render
        render.uv_inpaint = types.MethodType(uv_inpaint, render)

    # -- per job -----------------------------------------------------------
    def _rgba(self, image: Image.Image) -> Image.Image:  # pragma: no cover
        """Shape conditioning uses alpha as the object mask. Keep the gateway's
        alpha; matte RGB inputs (check the mode BEFORE converting — upstream's
        demo converts to RGBA first and so never runs rembg)."""
        if image.mode == "RGBA":
            return image
        out = self.rembg(image.convert("RGB"))
        if out.mode != "RGBA":
            out = out.convert("RGBA")
        if out.getchannel("A").getextrema()[1] <= 204:
            raise EngineInputError("no foreground object found in the image")
        return out

    def _shape(self, image: Image.Image, spec) -> Any:  # pragma: no cover
        torch = self._torch
        # Verified on the GPU (bake-off, Nov 2026): __call__ kwargs; returns a list of trimesh.Trimesh (an
        # entry is None when surface extraction fails).
        generator = torch.Generator().manual_seed(int(spec.seed))
        opts = spec.options   # validated in inputs.py; defaults from the worker env
        meshes = self.shape(
            image=image,
            num_inference_steps=int(opts.get("steps", self.config.hunyuan_steps)),
            guidance_scale=float(opts.get("guidance_scale", self.config.hunyuan_guidance)),
            octree_resolution=int(opts.get("octree_resolution", self.config.hunyuan_octree)),
            num_chunks=8000,
            generator=generator,
            output_type="trimesh",
        )
        mesh = meshes[0] if meshes else None
        if mesh is None:
            raise EngineInputError("surface extraction produced no mesh for this image")
        floaters, degenerate, reducer = self._post
        mesh = floaters(mesh)
        mesh = degenerate(mesh)
        if self.config.hunyuan_fast_post:
            # 868k -> 40k faces in ~0.6 s instead of ~38 s (single-core pymeshlab).
            vertices, faces = fast_decimate(mesh.vertices, mesh.faces, int(spec.target_faces))
            return self._trimesh.Trimesh(vertices, faces, process=False)
        return reducer(mesh, max_facenum=int(spec.target_faces))

    def _paint(self, mesh: Any, image: Image.Image, spec, workdir: str,
               tracker: StageTracker) -> Any:  # pragma: no cover
        trimesh = self._trimesh
        plan = paint_plan(spec.target_faces, spec.texture_size)
        white = os.path.join(workdir, "white.obj")
        mesh.export(white)
        render = self.paint.render
        # Runs on the GPU without error. Its effect on the internal bake
        # resolution was not measured separately: the saved maps are resized
        # to texture_size below either way. (Texture buffers are created per
        # load_mesh/bake, so it applies to the next bake.)
        render.set_default_texture_resolution(plan["bake_atlas"])
        hooks = {"bake_from_multiview": ("texture", 70.0)}
        with stage_hooks(self.paint.view_processor, tracker, hooks):
            # Verified on the GPU (bake-off, Nov 2026): returns the textured OBJ path; writes <base>.jpg,
            # <base>_metallic.jpg, <base>_roughness.jpg beside it.
            obj_path = self.paint(
                mesh_path=white, image_path=image, output_mesh_path=os.path.join(workdir, "textured.obj"),
                use_remesh=bool(plan["use_remesh"]), save_glb=False,
            )
        tracker.enter("export", 85.0)
        base = os.path.splitext(obj_path)[0]
        size = (int(spec.texture_size), int(spec.texture_size))

        def load_map(path: str, mode: str) -> Image.Image:
            with Image.open(path) as im:
                return im.convert(mode).resize(size, Image.Resampling.LANCZOS)

        albedo = load_map(base + ".jpg", "RGB")
        metallic = load_map(base + "_metallic.jpg", "L")
        roughness = load_map(base + "_roughness.jpg", "L")
        # glTF metallicRoughness: G = roughness, B = metallic (R unused).
        mr = Image.merge("RGB", (Image.new("L", size, 255), roughness, metallic))
        # Verified on the GPU (bake-off, Nov 2026): the OBJ loads as one Trimesh with per-vertex UVs.
        textured = trimesh.load(obj_path, force="mesh", process=False)
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=albedo, metallicRoughnessTexture=mr,
            metallicFactor=1.0, roughnessFactor=1.0, doubleSided=False,
        )
        textured.visual = trimesh.visual.TextureVisuals(uv=textured.visual.uv, material=material)
        return textured

    def generate(self, spec, image: Image.Image, glb_path: str, preview_path: str,
                 tracker: StageTracker) -> Dict[str, Any]:  # pragma: no cover - GPU only
        if self.shape is None:
            self.load()
        workdir = tempfile.mkdtemp(prefix="hy21-", dir=self.config.output_dir)
        tmp_root = tempfile.gettempdir()
        ply_before = set(glob.glob(os.path.join(tmp_root, "*.ply")))
        try:
            tracker.enter("preprocess", 5.0)
            rgba = self._rgba(image)
            tracker.enter("shape", 10.0)
            mesh = self._shape(rgba, spec)
            if spec.texture:
                tracker.enter("texture", 45.0)
                out = self._paint(mesh, rgba, spec, workdir, tracker)
            else:
                out = mesh
                tracker.enter("export", 85.0)
            if self.config.glb_jpeg and not self.config.glb_webp:
                jpeg_textures(out)
            out.export(glb_path, extension_webp=bool(self.config.glb_webp))
            tracker.progress("export", 95.0)
            with open(preview_path, "wb") as fh:
                fh.write(trimesh_preview_png(out))
            return {"faces": int(len(out.faces)), "vertices": int(len(out.vertices))}
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
            for leaked in set(glob.glob(os.path.join(tmp_root, "*.ply"))) - ply_before:
                try:
                    os.remove(leaked)
                except OSError:
                    pass
            release_cuda_memory()
