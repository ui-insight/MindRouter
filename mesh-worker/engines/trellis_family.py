############################################################
#
# mindrouter mesh-worker - async image-to-3D mesh service
#
# engines/trellis_family.py: Microsoft TRELLIS.2-4B and Tencent ARC Pixal3D.
#
#   Both are the TRELLIS.2 stack (O-Voxel sparse structure -> shape SLat ->
#   texture SLat, exported to GLB by o_voxel.postprocess.to_glb with CuMesh
#   decimation + UV unwrap + PBR bake). Pixal3D vendors the code as its own
#   `pixal3d` package and adds pixel-aligned (camera-projected) conditioning.
#
#   Source read for this adapter (pinned in docker/trellis2/Dockerfile):
#     microsoft/TRELLIS.2 @ 75fbf0183001ed9876c8dbb35de6b68552ee08bd
#       trellis2/pipelines/{base,trellis2_image_to_3d}.py, rembg/BiRefNet.py,
#       o-voxel/o_voxel/postprocess.py, example.py, app.py
#     TencentARC/Pixal3D @ f7cf38429b0bd264f1995f0f8743a88b1c728b94 (master)
#       pixal3d/pipelines/pixal3d_image_to_3d.py, inference.py,
#       pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py
#     HF pipeline.json: microsoft/TRELLIS.2-4B @ af44b45f, TencentARC/Pixal3D @ b0cb2e1b
#
#   LICENSE GUARD: both pipeline.json files name briaai/RMBG-2.0 (CC BY-NC)
#   as the background remover. load() swaps the rembg factory for the MIT
#   ZhengPeng7/BiRefNet (same architecture, same upstream wrapper class) and
#   refuses to start if the pipeline built any other remover. Inputs that
#   already carry an alpha mask (the gateway's matting output) never touch it:
#   upstream preprocess_image() uses a non-opaque alpha as-is.
#
#   Every call into upstream code ran on an H200 in the Nov 2026 bake-off
#   (aspen1 GPU2: 18/18 inputs per engine at 100k faces / 2048 px, plus
#   1M-face / 4096 px runs); the comments below record what was checked.
#
# Luke Sheneman — University of Idaho RCDS — sheneman@uidaho.edu
#
############################################################

"""TRELLIS.2 and Pixal3D engines."""

import importlib
import logging
import math
import os
import tempfile
from typing import Any, Dict

from PIL import Image, ImageDraw

from config import DEFAULT_PIPELINE_TYPE, ENGINE_OPTIONS, WorkerConfig
from engines.base import EngineInputError, StageTracker
from engines.common import (
    check_rembg_allowed, gltf_axes, jpeg_textures, patched_attr, release_cuda_memory, stage_hooks,
    trimesh_preview_png,
)

log = logging.getLogger(__name__)

AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
MAX_NUM_TOKENS = 49152  # upstream run() default


class _TrellisFamilyEngine:
    package = ""                     # python package: trellis2 | pixal3d
    pipeline_class = ""
    pipeline_types: tuple = ()
    default_pipeline_type = ""

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.pipeline_type = config.pipeline_type or self.default_pipeline_type
        if self.pipeline_type not in self.pipeline_types:
            raise ValueError(
                f"MESH_WORKER_PIPELINE_TYPE for {config.mode} must be one of "
                f"{', '.join(self.pipeline_types)}, not {self.pipeline_type!r}"
            )
        check_rembg_allowed(config.birefnet)
        self.pipeline = None
        self._o_voxel = None
        self._torch = None

    # -- loading -----------------------------------------------------------
    def _build_pipeline(self):
        """from_pretrained with the background remover forced to BiRefNet (MIT).

        Upstream from_pretrained does
            pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])
        with args = {"name": "BiRefNet", "args": {"model_name": "briaai/RMBG-2.0"}}.
        We replace the module attribute for the duration of the call, so the
        RMBG-2.0 weights are never even downloaded/loaded.
        """
        pipelines = importlib.import_module(f"{self.package}.pipelines")
        rembg_mod = importlib.import_module(f"{self.package}.pipelines.rembg")
        pipeline_cls = getattr(pipelines, self.pipeline_class)
        birefnet = self.config.birefnet
        built = []

        def safe_birefnet(*args, **kwargs):
            requested = kwargs.get("model_name", args[0] if args else None)
            if requested != birefnet:
                log.info("Background remover %r replaced by %r (license guard).", requested, birefnet)
            remover = original(model_name=birefnet)
            built.append(remover)
            return remover

        with patched_attr(rembg_mod, "BiRefNet", safe_birefnet) as original, self._extra_load_patches():
            # Verified on the GPU (bake-off, Nov 2026): from_pretrained(local_dir) resolves "ckpts/..." under
            # the dir; TRELLIS.2's sparse_structure_decoder lives in ANOTHER
            # repo ("microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16")
            # and is found either at <ckpt>/microsoft/TRELLIS-image-large/ckpts/…
            # or in the HF cache (README: staging).
            pipeline = pipeline_cls.from_pretrained(self.config.checkpoint)

        remover = getattr(pipeline, "rembg_model", None)
        if remover is not None and not any(remover is b for b in built):
            raise RuntimeError(
                f"{self.package} built a background remover other than BiRefNet "
                f"({type(remover).__name__}); pipeline.json changed upstream — review its license before serving"
            )
        return pipeline

    def _extra_load_patches(self):
        import contextlib
        return contextlib.nullcontext()

    def load(self) -> None:  # pragma: no cover - GPU node only
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        import o_voxel

        self._torch = torch
        self._o_voxel = o_voxel
        log.info("Loading %s from %s (pipeline_type=%s, low_vram=%s)…", self.package,
                 self.config.checkpoint, self.pipeline_type, self.config.low_vram)
        pipeline = self._build_pipeline()
        # Verified on the GPU (bake-off, Nov 2026): low_vram=False + cuda() keeps every sub-model resident
        # (Pipeline.to() only moves models when low_vram is False).
        pipeline.low_vram = bool(self.config.low_vram)
        pipeline.cuda()
        self.pipeline = pipeline
        self._after_load()
        if self.config.warmup:
            self._warm_up()

    def _after_load(self) -> None:
        pass

    def _warm_up(self) -> None:  # pragma: no cover - GPU node only
        """One small textured job so CUDA kernels / FlexGEMM autotune are warm."""
        from inputs import MeshSpec

        img = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        ImageDraw.Draw(img).ellipse((96, 96, 416, 416), fill=(200, 80, 60, 255))
        spec = MeshSpec(model=self.config.model_id, image_bytes=b"", image_format="PNG", width=512,
                        height=512, has_alpha=True, seed=1, texture=True,
                        texture_size=min(self.config.texture_sizes), target_faces=self.config.min_faces * 10)
        tmp = tempfile.mkdtemp(prefix="warmup-", dir=self.config.output_dir)
        tracker = StageTracker(lambda stage, progress: None, lambda: False)
        try:
            self.generate(spec, img, os.path.join(tmp, "w.glb"), os.path.join(tmp, "w.png"), tracker)
            log.info("Warm-up job done.")
        except Exception:
            log.warning("Warm-up job failed; the first real job pays the warm-up.", exc_info=True)
        finally:
            for name in os.listdir(tmp):
                try:
                    os.remove(os.path.join(tmp, name))
                except OSError:
                    pass
            try:
                os.rmdir(tmp)
            except OSError:
                pass
            release_cuda_memory()

    # -- per job -----------------------------------------------------------
    def capabilities(self) -> Dict[str, Any]:
        return {"supports_texture": True, "pipeline_type": self.pipeline_type}

    def _preprocess(self, image: Image.Image) -> Image.Image:  # pragma: no cover
        # Verified on the GPU (bake-off, Nov 2026): upstream preprocess_image(): RGBA with a non-opaque
        # alpha is used as-is; otherwise BiRefNet mattes it. Then crops to the
        # alpha>0.8 bbox (raises ValueError on an empty bbox) and flattens.
        try:
            return self.pipeline.preprocess_image(image)
        except ValueError as exc:
            raise EngineInputError(f"no foreground object found in the image ({exc})")

    def _post_transform(self, mesh: Any) -> None:
        """Engine-specific final transform of the exported trimesh."""

    def _export_textured(self, mesh: Any, res: int, spec, glb_path: str):  # pragma: no cover
        # Verified on the GPU (bake-off, Nov 2026): same call as TRELLIS.2 app.py extract_glb / Pixal3D
        # inference.py; decimation_target is a FACE count (app.py docstring).
        glb = self._o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
            coords=mesh.coords, attr_layout=self.pipeline.pbr_attr_layout,
            grid_size=res, aabb=AABB,
            decimation_target=int(spec.target_faces),
            texture_size=int(spec.texture_size),
            remesh=True, remesh_band=1, remesh_project=0,
        )
        self._post_transform(glb)
        # Verified on the GPU with WebP off (the default; upstream uses
        # extension_webp=True). Opaque textures are written as JPEG
        # (MESH_WORKER_GLB_JPEG, default on): encoding them as PNG was 5-10%
        # of the job. MESH_WORKER_GLB_WEBP=1 has not been exercised yet.
        if self.config.glb_jpeg and not self.config.glb_webp:
            jpeg_textures(glb)
        glb.export(glb_path, extension_webp=bool(self.config.glb_webp))
        return glb

    def _export_shape_only(self, mesh: Any, spec, glb_path: str):  # pragma: no cover
        import trimesh

        # Verified on the GPU (bake-off, Nov 2026): Mesh.simplify(target) is CuMesh quadric decimation to a
        # face count (same call to_glb makes).
        mesh.simplify(int(spec.target_faces))
        vertices = gltf_axes(mesh.vertices.detach().float().cpu().numpy())
        faces = mesh.faces.detach().cpu().numpy()
        out = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        self._post_transform(out)
        out.export(glb_path)
        return out

    def _sample(self, image: Image.Image, spec, tracker: StageTracker):  # pragma: no cover
        """Return (mesh, res); mesh is MeshWithVoxel when textured."""
        raise NotImplementedError

    def generate(self, spec, image: Image.Image, glb_path: str, preview_path: str,
                 tracker: StageTracker) -> Dict[str, Any]:  # pragma: no cover - GPU only
        if self.pipeline is None:
            self.load()
        torch = self._torch
        try:
            tracker.enter("preprocess", 5.0)
            prepared = self._preprocess(image)
            tracker.enter("shape", 10.0)
            with torch.no_grad():
                mesh, res = self._sample(prepared, spec, tracker)
            tracker.enter("export", 85.0)
            if spec.texture:
                out = self._export_textured(mesh, res, spec, glb_path)
            else:
                out = self._export_shape_only(mesh, spec, glb_path)
            tracker.progress("export", 95.0)
            with open(preview_path, "wb") as fh:
                fh.write(trimesh_preview_png(out))
            return {"faces": int(len(out.faces)), "vertices": int(len(out.vertices))}
        finally:
            release_cuda_memory()


class Trellis2Engine(_TrellisFamilyEngine):
    package = "trellis2"
    pipeline_class = "Trellis2ImageTo3DPipeline"
    pipeline_types = tuple(ENGINE_OPTIONS["trellis2"]["pipeline_type"]["choices"])
    default_pipeline_type = DEFAULT_PIPELINE_TYPE["trellis2"]   # pipeline.json default_pipeline_type

    def _extra_load_patches(self):
        """Point the DINOv3 extractor at MESH_WORKER_DINOV3 (an admin-staged
        local dir of the GATED facebook/dinov3-vitl16-pretrain-lvd1689m)."""
        import contextlib

        ife = importlib.import_module("trellis2.modules.image_feature_extractor")
        dinov3 = self.config.dinov3
        stack = contextlib.ExitStack()

        def dinov3_factory(*args, **kwargs):
            kwargs.pop("model_name", None)
            return original(dinov3, *args[1:], **kwargs)

        # Verified on the GPU (bake-off, Nov 2026): pipeline.json names DinoV3FeatureExtractor(model_name=...).
        original = stack.enter_context(patched_attr(ife, "DinoV3FeatureExtractor", dinov3_factory))
        return stack

    def _sample(self, image, spec, tracker):  # pragma: no cover - GPU only
        p = self.pipeline
        pt = spec.options.get("pipeline_type") or self.pipeline_type
        if spec.texture:
            hooks = {
                "sample_sparse_structure": ("shape", 15.0),
                "sample_shape_slat": ("shape", 30.0),
                "sample_shape_slat_cascade": ("shape", 30.0),
                "sample_tex_slat": ("texture", 60.0),
                "decode_latent": ("texture", 75.0),
            }
            # Verified on the GPU (bake-off, Nov 2026): run() signature at the pinned commit.
            with stage_hooks(p, tracker, hooks):
                meshes, (_shape, _tex, res) = p.run(
                    image, seed=int(spec.seed), preprocess_image=False,
                    pipeline_type=pt, return_latent=True, max_num_tokens=MAX_NUM_TOKENS,
                )
            return meshes[0], res

        # Shape only: run() always samples texture, so drive the same public
        # stage methods run() uses and stop before the texture SLat (saves the
        # texture sampling + the UV bake). Mirrors run() at the pinned commit.
        # Verified on the GPU (bake-off, Nov 2026): every call below.
        torch = self._torch
        torch.manual_seed(int(spec.seed))
        cond_512 = p.get_cond([image], 512)
        cond_1024 = p.get_cond([image], 1024) if pt != "512" else None
        ss_res = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}[pt]
        tracker.enter("shape", 15.0)
        coords = p.sample_sparse_structure(cond_512, ss_res, 1, {})
        tracker.enter("shape", 30.0)
        if pt == "512":
            slat = p.sample_shape_slat(cond_512, p.models["shape_slat_flow_model_512"], coords, {})
            res = 512
        elif pt == "1024":
            slat = p.sample_shape_slat(cond_1024, p.models["shape_slat_flow_model_1024"], coords, {})
            res = 1024
        else:
            slat, res = p.sample_shape_slat_cascade(
                cond_512, cond_1024,
                p.models["shape_slat_flow_model_512"], p.models["shape_slat_flow_model_1024"],
                512, 1024 if pt == "1024_cascade" else 1536,
                coords, {}, MAX_NUM_TOKENS,
            )
        tracker.enter("shape", 70.0)
        meshes, _subs = p.decode_shape_slat(slat, res)
        mesh = meshes[0]
        mesh.fill_holes()   # decode_latent does this before export
        return mesh, res


# Pixal3D image-conditioning models: copied from inference.py IMAGE_COND_CONFIGS
# at the pinned commit, with the third-party "camenduru/dinov3-..." mirror
# replaced by MESH_WORKER_DINOV3 (the official, gated weights).
PIXAL3D_IMAGE_COND = {
    "image_cond_model_ss": {"image_size": 512, "grid_resolution": 16},
    "image_cond_model_shape_512": {"image_size": 512, "grid_resolution": 32,
                                   "use_naf_upsample": True, "naf_target_size": 512},
    "image_cond_model_shape_1024": {"image_size": 1024, "grid_resolution": 64,
                                    "use_naf_upsample": True, "naf_target_size": 512},
    "image_cond_model_tex_1024": {"image_size": 1024, "grid_resolution": 64,
                                  "use_naf_upsample": True, "naf_target_size": 1024},
}


def pixal3d_camera_from_fov(camera_angle_x: float, mesh_scale: float = 1.0,
                            extend_pixel: int = 0, image_resolution: int = 512) -> Dict[str, float]:
    """Pixal3D inference.py distance_from_fov()/compute_f_pixels() (MIT,
    Tencent), in plain math. Camera distance that frames the unit object for
    a given horizontal FOV."""
    focal_length = 16.0 / math.tan(camera_angle_x / 2.0)
    f_pixels = focal_length * image_resolution / 32.0
    # grid_point (-1, 0, 0) rotated by [[1,0,0],[0,0,-1],[0,1,0]] -> (-1, 0, 0)
    xw, yw = -1.0 / mesh_scale / 2, 0.0
    xt = float(0 - extend_pixel)
    x_ndc = xt - image_resolution / 2.0
    distance = f_pixels * xw / x_ndc - yw
    return {"camera_angle_x": float(camera_angle_x), "distance": float(distance), "mesh_scale": float(mesh_scale)}


class Pixal3DEngine(_TrellisFamilyEngine):
    package = "pixal3d"
    pipeline_class = "Pixal3DImageTo3DPipeline"
    pipeline_types = tuple(ENGINE_OPTIONS["pixal3d"]["pipeline_type"]["choices"])
    default_pipeline_type = DEFAULT_PIPELINE_TYPE["pixal3d"]   # inference.py standard (non-low-VRAM) mode

    # inference.py applies UPSTREAM_ROTATION after to_glb (on top of to_glb's
    # own Y/Z swap). That leaves the object's front facing -Z, while glTF, and
    # our TRELLIS.2 and Hunyuan3D outputs, put the front at +Z (seen on the
    # GPU: a face came out looking backwards). So we also turn it 180 degrees
    # about +Y: ROTATION = Ry(180) @ UPSTREAM_ROTATION. Up stays +Y.
    UPSTREAM_ROTATION = [[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    ROTATION = [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]

    def __init__(self, config: WorkerConfig):
        super().__init__(config)
        self._moge = None

    def _post_transform(self, mesh: Any) -> None:  # pragma: no cover
        import numpy as np

        mesh.apply_transform(np.array(self.ROTATION, dtype=np.float64))

    def _after_load(self) -> None:  # pragma: no cover - GPU node only
        torch = self._torch
        # Verified on the GPU (bake-off, Nov 2026): import path + constructor at the pinned commit.
        from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor

        p = self.pipeline
        for attr, cfg in PIXAL3D_IMAGE_COND.items():
            model = DinoV3ProjFeatureExtractor(model_name=self.config.dinov3, **cfg)
            model.eval()
            setattr(p, attr, model)
            if not self.config.low_vram:
                model.cuda()
        # NAF upsampler: torch.hub.load("valeoai/NAF", "naf", pretrained=True,
        # trust_repo=True) inside _load_naf(). Load it NOW (not mid-job) and,
        # when MESH_WORKER_NAF_DIR is set, from a local clone so no GitHub call
        # is attempted. Its pretrained weights come from $TORCH_HOME/hub/checkpoints.
        # Verified on the GPU (bake-off, Nov 2026): hub entrypoint name + kwargs.
        naf_dir = self.config.naf_dir
        original_hub_load = torch.hub.load

        def hub_load(repo_or_dir, model, *args, **kwargs):
            if naf_dir and repo_or_dir == "valeoai/NAF":
                kwargs.pop("trust_repo", None)
                kwargs["source"] = "local"
                return original_hub_load(naf_dir, model, *args, **kwargs)
            return original_hub_load(repo_or_dir, model, *args, **kwargs)

        with patched_attr(torch.hub, "load", hub_load):
            for attr in PIXAL3D_IMAGE_COND:
                m = getattr(p, attr)
                if getattr(m, "use_naf_upsample", False):
                    m._load_naf()
        if self.config.pixal3d_fov <= 0:
            # MoGe-2 API (moge.model.v2.MoGeModel) as in inference.py. Its
            # from_pretrained() takes a checkpoint FILE or a repo id; given a
            # staged local repo directory it fails with IsADirectoryError
            # (found in the 2026-11-23 bake-off), so resolve it to model.pt.
            from moge.model.v2 import MoGeModel

            source = self.config.moge
            if os.path.isdir(source):
                source = os.path.join(source, "model.pt")
            self._moge = MoGeModel.from_pretrained(source).to("cuda").eval()

    def _camera_params(self, image: Image.Image) -> Dict[str, float]:  # pragma: no cover
        if self.config.pixal3d_fov > 0:
            return pixal3d_camera_from_fov(self.config.pixal3d_fov)
        import numpy as np

        torch = self._torch
        # Verified on the GPU (bake-off, Nov 2026): inference.py get_camera_params_wild_moge(), on the
        # preprocessed image (it saves that image to disk and re-reads it).
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to("cuda")
        with torch.no_grad():
            output = self._moge.infer(tensor)
        intrinsics = output["intrinsics"].squeeze().cpu().numpy()
        width = image.size[0]
        fx = intrinsics[0, 0] * width
        return pixal3d_camera_from_fov(2 * math.atan(width / (2 * fx)))

    def _sample(self, image, spec, tracker):  # pragma: no cover - GPU only
        camera = self._camera_params(image)
        tex_stage = "texture" if spec.texture else "shape"
        hooks = {
            "sample_sparse_structure": ("shape", 15.0),
            "sample_shape_slat": ("shape", 25.0),
            "get_proj_cond_shape": ("shape", 35.0),
            "sample_tex_slat": (tex_stage, 60.0),
            "decode_latent": (tex_stage, 75.0),
        }
        # Pixal3D's run() inlines its HR shape stage, so a shape-only job still
        # samples the texture latent (only the UV bake is skipped).
        # Verified on the GPU (bake-off, Nov 2026): run() signature at the pinned commit.
        with stage_hooks(self.pipeline, tracker, hooks):
            meshes, (_shape, _tex, res) = self.pipeline.run(
                image, camera_params=camera, seed=int(spec.seed), preprocess_image=False,
                return_latent=True, pipeline_type=spec.options.get("pipeline_type") or self.pipeline_type,
                max_num_tokens=MAX_NUM_TOKENS,
            )
        return meshes[0], res
