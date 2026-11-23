# syntax=docker/dockerfile:1.6
############################################################
#
# mindrouter mesh-worker — Hunyuan3D 2.1 image (H200 / sm_90).
# Serves MESH_WORKER_MODE=hunyuan3d21 (tencent/Hunyuan3D-2.1), shape + PBR
# paint in-process. Upstream's api_server.py is NOT used (no auth).
#
# Build ON THE GPU NODE (never on a laptop — disk rule), context = mesh-worker/:
#   docker build -f docker/hunyuan3d21/Dockerfile -t mindrouter-mesh-hunyuan3d21:0.1.0 .
#
# Weights are NOT baked in: mount the HF cache read-only at /weights (see
# README "Staging weights"). HF_HUB_OFFLINE=1 — no outbound network needed.
#
# Versions follow upstream README ("Python 3.10 and PyTorch 2.5.1+cu124") and
# its requirements.txt @ the pinned commit, minus its Tencent/Aliyun mirror
# index lines and the packages only the demo/training use.
#
# LICENSE: Tencent Hunyuan 3D 2.1 Community License — NOT licensed in the EU,
# UK or South Korea; outputs may not be used to improve other AI models. See
# README before deploying.
#
# Luke Sheneman — University of Idaho RCDS — sheneman@uidaho.edu
#
############################################################

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG HUNYUAN_COMMIT=82920d643c0dc2f7bfd7255f45f62d386edfe60c
ARG MAX_JOBS=16

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="9.0" \
    MAX_JOBS=${MAX_JOBS} \
    CUDA_HOME=/usr/local/cuda \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libGL/X libs: opencv-python (non-headless) and the bpy module link them even
# headless; python3-dev provides the python3-config compile_mesh_painter.sh uses.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-dev git build-essential \
        libgl1 libglib2.0-0 libegl1 libsm6 libxrender1 libxi6 libxxf86vm1 libxfixes3 \
        libxkbcommon0 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:${PATH}
RUN pip install --upgrade pip setuptools wheel packaging

RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /opt/Hunyuan3D-2.1 \
    && cd /opt/Hunyuan3D-2.1 && git checkout ${HUNYUAN_COMMIT}

# Upstream requirements.txt, filtered:
#   - its --extra-index-url mirror lines (we install from PyPI only)
#   - gradio (demo UI), cupy-cuda12x (pinned but never imported),
#     deepspeed (training), pythreejs (notebook viewer)
# Kept on purpose: bpy==4.0 (DifferentiableRenderer/mesh_utils.py imports it
# at module level even though we never call it), trimesh 4.4.7 + open3d
# (remesh relies on that trimesh's simplify_quadric_decimation signature),
# tb_nightly (basicsr install dependency).
RUN sed -e '/^--extra-index-url/d' -e '/^gradio==/d' -e '/^cupy-cuda12x==/d' \
        -e '/^deepspeed/d' -e '/^pythreejs/d' \
        /opt/Hunyuan3D-2.1/requirements.txt > /tmp/requirements-upstream.txt \
    && pip install --extra-index-url https://download.blender.org/pypi/ -r /tmp/requirements-upstream.txt
# ^ bpy 4.0.0 for Python 3.10 is published only on Blender's official index
#   (PyPI's bpy starts at cp311); upstream reached it through the Tencent and
#   Aliyun PyPI mirrors stripped above.
COPY docker/hunyuan3d21/requirements-engine.txt /tmp/requirements-engine.txt
RUN pip install -r /tmp/requirements-engine.txt

# Compiled pieces of the paint pipeline (README install steps).
RUN cd /opt/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer && pip install -e . --no-build-isolation
RUN cd /opt/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh \
    && ls mesh_inpaint_processor*.so
# Fail the build (not the first job) if the extensions or bpy are missing:
# MeshRender otherwise only prints "InPaint Function CAN NOT BE Imported!!!".
# torch first: the rasterizer extension needs libc10.so already loaded.
RUN cd /opt/Hunyuan3D-2.1/hy3dpaint && python -c "import torch, bpy, custom_rasterizer; from DifferentiableRenderer import mesh_inpaint_processor"

# pytorch_lightning (imported by the paint model) still needs pkg_resources,
# which recent setuptools no longer ships, and the base step upgrades
# setuptools to the latest. Pin it back, then import the paint model's own
# modules so a missing dependency fails the build, not the first job.
RUN pip install "setuptools<80" \
    && cd /opt/Hunyuan3D-2.1/hy3dpaint \
    && python -c "import torch, pkg_resources, pytorch_lightning; from hunyuanpaintpbr.unet.modules import Dino_v2; import textureGenPipeline; print('paint imports ok')"

COPY app.py config.py worker.py inputs.py glb.py preview.py /opt/mesh-worker/
COPY engines /opt/mesh-worker/engines
RUN pip freeze > /opt/mesh-worker/pip-freeze.txt \
    && python -c "import sys; sys.path.insert(0, '/opt/mesh-worker'); import app"

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin mesh \
    && mkdir -p /data /weights && chown mesh:mesh /data

# Runtime: offline, weights read-only under /weights, scratch under /data.
# The paint pipeline resolves its multiview weights with
# snapshot_download(repo_id=...), so the Hunyuan weights MUST be staged as an
# HF cache (HF_HOME), and MESH_WORKER_CKPT stays the repo id.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/weights/hf \
    U2NET_HOME=/weights/u2net \
    MESH_WORKER_REALESRGAN_CKPT=/weights/realesrgan/RealESRGAN_x4plus.pth \
    MESH_WORKER_HUNYUAN_REPO=/opt/Hunyuan3D-2.1 \
    MESH_WORKER_CKPT=tencent/Hunyuan3D-2.1 \
    MESH_WORKER_OUTPUT_DIR=/data/out \
    TMPDIR=/data/tmp \
    XDG_CACHE_HOME=/data/cache \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    MESH_WORKER_MODE=hunyuan3d21

USER mesh
WORKDIR /data
EXPOSE 18400
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
    CMD curl -fsS http://127.0.0.1:18400/health || exit 1
# /data is usually a fresh host volume: create the scratch dirs, then exec.
# TMPDIR is private to this process: the worker deletes the .ply files the
# hy3dshape post-processors leak there.
CMD ["sh", "-c", "mkdir -p /data/out /data/cache /data/tmp && exec uvicorn app:app --app-dir /opt/mesh-worker --host 0.0.0.0 --port 18400 --log-level info"]
