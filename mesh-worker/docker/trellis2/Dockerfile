# syntax=docker/dockerfile:1.6
############################################################
#
# mindrouter mesh-worker — TRELLIS.2 stack image (H200 / sm_90).
# Serves MESH_WORKER_MODE=trellis2 (microsoft/TRELLIS.2-4B) and
# MESH_WORKER_MODE=pixal3d (TencentARC/Pixal3D, built on the same stack).
#
# Build ON THE GPU NODE (never on a laptop — disk rule), context = mesh-worker/:
#   docker build -f docker/trellis2/Dockerfile -t mindrouter-mesh-trellis2:0.1.0 .
#
# Weights are NOT baked in: mount them read-only at /weights (see README
# "Staging weights"). HF_HUB_OFFLINE=1 — the container needs no outbound
# network at runtime.
#
# Versions follow upstream: TRELLIS.2 README/setup.sh (python 3.10, torch
# 2.6.0 + CUDA 12.4, flash-attn 2.7.3, nvdiffrast v0.4.0) and Pixal3D README
# (natten 0.21.0, MoGe, its utils3d wheel). Every git dependency is pinned to
# the commit read on 2026-11-23.
#
# Luke Sheneman — University of Idaho RCDS — sheneman@uidaho.edu
#
############################################################

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG TRELLIS2_COMMIT=75fbf0183001ed9876c8dbb35de6b68552ee08bd
ARG PIXAL3D_COMMIT=f7cf38429b0bd264f1995f0f8743a88b1c728b94
ARG CUMESH_COMMIT=12289e1062f0603f2f0d0771b02e1395d247f26f
ARG FLEXGEMM_COMMIT=6dd94a859c26ee8246888502eada3dd8ad85532e
ARG NVDIFFRAST_COMMIT=253ac4fcea7de5f396371124af597e6cc957bfae
ARG NVDIFFREC_COMMIT=b296927cc7fd01c2ac1087c8065c4d7248f72da4
ARG UTILS3D_COMMIT=9a4eb15e4021b67b12c460c7057d642626897ec8
ARG MOGE_COMMIT=74fbce054ebed49800de42d0ad0e83495065719a
ARG NAF_COMMIT=37f2dfc180f2de53d98bd601109c0da0dd6b0f43
ARG PIXAL3D_UTILS3D_WHEEL=https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl
ARG MAX_JOBS=16

# H200 = Hopper = sm_90 only (keeps the extension builds short).
ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="9.0" \
    NATTEN_CUDA_ARCH="9.0" \
    NATTEN_N_WORKERS=${MAX_JOBS} \
    MAX_JOBS=${MAX_JOBS} \
    CUDA_HOME=/usr/local/cuda \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev git build-essential \
        libjpeg-dev libglib2.0-0 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:${PATH}
RUN pip install --upgrade pip setuptools wheel packaging

# torch first: every extension below builds against it (--no-build-isolation).
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

COPY docker/trellis2/requirements-engine.txt /tmp/requirements-engine.txt
RUN pip install -r /tmp/requirements-engine.txt
# setup.sh --flash-attn (TRELLIS.2 attention backend; ATTN_BACKEND below).
RUN pip install flash-attn==2.7.3 --no-build-isolation

WORKDIR /opt/build
# TRELLIS.2 itself (o-voxel needs the eigen submodule).
RUN git clone https://github.com/microsoft/TRELLIS.2.git /opt/TRELLIS.2 \
    && cd /opt/TRELLIS.2 && git checkout ${TRELLIS2_COMMIT} \
    && git submodule update --init --recursive
# setup.sh --nvdiffrast / --nvdiffrec / --cumesh / --flexgemm / --o-voxel
RUN git clone https://github.com/NVlabs/nvdiffrast.git nvdiffrast \
    && cd nvdiffrast && git checkout ${NVDIFFRAST_COMMIT} \
    && pip install . --no-build-isolation
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git nvdiffrec \
    && cd nvdiffrec && git checkout ${NVDIFFREC_COMMIT} \
    && pip install . --no-build-isolation
RUN git clone --recursive https://github.com/JeffreyXiang/CuMesh.git CuMesh \
    && cd CuMesh && git checkout ${CUMESH_COMMIT} && git submodule update --init --recursive \
    && pip install . --no-build-isolation
RUN git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git FlexGEMM \
    && cd FlexGEMM && git checkout ${FLEXGEMM_COMMIT} && git submodule update --init --recursive \
    && pip install . --no-build-isolation
RUN cp -r /opt/TRELLIS.2/o-voxel o-voxel && pip install ./o-voxel --no-build-isolation
RUN pip install "git+https://github.com/EasternJournalist/utils3d.git@${UTILS3D_COMMIT}"

# Pixal3D (README steps 2-4, in order: requirements incl. MoGe, natten, its
# utils3d wheel). Its utils3d replaces TRELLIS.2's pin; generation + GLB export
# never import utils3d (only renderers / training do), so TRELLIS.2 serving is
# unaffected. UNVERIFIED until the bake-off runs both modes in this image.
RUN git clone https://github.com/TencentARC/Pixal3D.git /opt/Pixal3D \
    && cd /opt/Pixal3D && git checkout ${PIXAL3D_COMMIT}
RUN pip install "git+https://github.com/microsoft/MoGe.git@${MOGE_COMMIT}"
# natten builds with CMake (+ Ninja); neither is in the CUDA devel base image.
RUN pip install cmake ninja && pip install natten==0.21.0 --no-build-isolation
RUN pip install "${PIXAL3D_UTILS3D_WHEEL}"
# NAF feature upsampler (Apache-2.0), loaded via torch.hub source="local" so
# no GitHub call happens at runtime (MESH_WORKER_NAF_DIR).
RUN git clone https://github.com/valeoai/NAF.git /opt/NAF && cd /opt/NAF && git checkout ${NAF_COMMIT}

# The git installs above (MoGe, utils3d) pull huggingface_hub unpinned and
# upgrade it past 1.0, which transformers 4.57 refuses to import with
# ("huggingface-hub>=0.34.0,<1.0 is required"). Pin it back last; the import
# check below fails the build if anything still conflicts.
RUN pip install "huggingface_hub>=0.34.0,<1.0"

# Expose ONLY the two packages (not the repo roots: both have an app.py that
# would shadow the worker's).
RUN mkdir -p /opt/engine-pkgs \
    && ln -s /opt/TRELLIS.2/trellis2 /opt/engine-pkgs/trellis2 \
    && ln -s /opt/Pixal3D/pixal3d /opt/engine-pkgs/pixal3d \
    && echo /opt/engine-pkgs > /opt/venv/lib/python3.10/site-packages/mesh-engines.pth \
    && rm -rf /opt/build

# The worker itself (explicit list: nothing else from the context).
COPY app.py config.py worker.py inputs.py glb.py preview.py /opt/mesh-worker/
COPY engines /opt/mesh-worker/engines
RUN pip freeze > /opt/mesh-worker/pip-freeze.txt \
    && python -c "import sys; sys.path.insert(0, '/opt/mesh-worker'); import app" \
    && python -c "import transformers, diffusers, huggingface_hub; from transformers import AutoModelForImageSegmentation; print('deps ok', huggingface_hub.__version__)"

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin mesh \
    && mkdir -p /data /weights && chown mesh:mesh /data

# Runtime: offline, weights read-only under /weights, scratch under /data.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/weights/hf \
    HF_MODULES_CACHE=/data/cache/hf_modules \
    TORCH_HOME=/weights/torch \
    MESH_WORKER_NAF_DIR=/opt/NAF \
    MESH_WORKER_OUTPUT_DIR=/data/out \
    FLEX_GEMM_AUTOTUNE_CACHE_PATH=/data/cache/flex_gemm_autotune.json \
    TRITON_CACHE_DIR=/data/cache/triton \
    XDG_CACHE_HOME=/data/cache \
    ATTN_BACKEND=flash_attn \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    MESH_WORKER_MODE=trellis2

USER mesh
WORKDIR /data
EXPOSE 18400
# Generous start period: weights load before the port opens.
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
    CMD curl -fsS http://127.0.0.1:18400/health || exit 1
# /data is usually a fresh host volume: create the scratch dirs, then exec.
CMD ["sh", "-c", "mkdir -p /data/out /data/cache && exec uvicorn app:app --app-dir /opt/mesh-worker --host 0.0.0.0 --port 18400 --log-level info"]
