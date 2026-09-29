# =============================================================================
# Production image: Ubuntu 24.04, multi-stage (2.9.35 base switch).
#
# Why Ubuntu, why multi-stage (evaluated 2026-08-18, full-image Snyk scans of
# built images, OS/dpkg layer, deduped by CVE ID):
#   python:3.11-slim single-stage : 292 CVEs (3 critical / 11 high)
#   ubuntu:24.04     single-stage :  46 CVEs (0 critical / 0 high)
#   ubuntu:24.04     multi-stage  :  28 CVEs (0 critical / 0 high)  <- this file
# The drop is Ubuntu's vendor security triage+patching (honored by scanners),
# not suppression. Multi-stage ships no compiler toolchain: build-essential/
# binutils was the single largest CVE contributor. There is no smaller
# apt-capable base: ubuntu:24.04 IS Canonical's minimal Ubuntu Base rootfs
# (no `-minimal` tag exists), and chiselled images have no apt/shell so they
# cannot host the LibreOffice/poppler/xmlsec stack.
#
# Ubuntu-port specifics (each is load-bearing, do not "simplify" away):
#   - Python 3.12 (noble's distro python; pyproject requires-python >=3.11)
#   - PEP 668: noble's python is externally-managed, so everything pip touches
#     lives in /opt/venv. The venv is first on PATH, so `pip`, `python`,
#     `uvicorn`, and the docker-compose `command:` override all resolve into it.
#   - noble's time_t 64-bit ABI transition renamed the xmlsec runtime package:
#     libxmlsec1-openssl -> libxmlsec1t64-openssl (the -dev name is unchanged).
#   - ubuntu:24.04 ships NO ca-certificates (pip/httpx TLS fails without it),
#     no locale env (LANG=C.UTF-8 matches the old python:3.11-slim), and a
#     stock `ubuntu` user owning UID 1000 (removed so appuser keeps UID 1000
#     for bind-mount permission parity).
#   - fonts-liberation is a deliberate addition (libreoffice-common's
#     first-choice recommended font; improves document-conversion fidelity).
#
# Runtime-stage apt rationale:
#   python3.12             interpreter backing /opt/venv
#   ca-certificates        outbound TLS (httpx to backends, SSO IdPs)
#   curl                   HEALTHCHECK
#   libxmlsec1t64-openssl  shared libs the pip-built xmlsec binding links
#                          against (pulls libxmlsec1t64/libxml2/libxslt);
#                          the -dev packages stay in the builder
#   poppler-utils          pdf2image/pdfplumber shell out to pdftoppm
#   libreoffice-* + fonts  document conversion
# Dropped vs the Debian-era image: default-libmysqlclient-dev — pymysql/
# aiomysql are pure Python; nothing in the image links libmysqlclient.
# =============================================================================
FROM ubuntu:24.04 AS builder

ENV LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    pkg-config \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    default-libmysqlclient-dev \
    libxmlsec1-dev \
    libxmlsec1t64-openssl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency lock: pip may only pick versions listed in constraints.txt
# (exact pins that have run in production; regenerate with `make lock`).
# PIP_CONSTRAINT is read by every pip invocation below, including the
# pip/setuptools/wheel upgrade, so the whole venv is reproducible.
COPY constraints.txt ./
ENV PIP_CONSTRAINT=/app/constraints.txt

# PEP 668: all Python packages go into this venv, never the distro python
RUN python3.12 -m venv /opt/venv && \
    pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY backend/__init__.py backend/
COPY backend/app/__init__.py backend/app/

# Install Python dependencies (editable install: the __editable__*.pth files
# in the venv reference /app, which the runtime stage recreates at the same
# path — both /opt/venv and /app must be copied across together)
RUN pip install -e .[saml]

# TrustMark invisible image watermarking (Adobe, MIT). Installed --no-deps:
# its numpy<2 pin conflicts with this image's numpy 2.x and is empirically
# unnecessary; its real runtime deps (torchvision/einops/omegaconf/lightning)
# are in pyproject. The bake downloads the Q-model weights (MD5-verified)
# into the venv's site-packages so nothing is fetched at runtime. The proving
# roundtrip runs in the RUNTIME stage below, where it doubles as a check that
# the slimmed-down runtime package set still satisfies torch/PIL.
RUN pip install --no-cache-dir --no-deps trustmark==0.9.1 && \
    python -c "\
from trustmark import TrustMark; \
tm = TrustMark(verbose=True, model_type='Q', loadRemover=False, device='cpu'); \
assert tm.encoder is not None and tm.decoder is not None, 'TrustMark weights missing'"

# =============================================================================
FROM ubuntu:24.04

ENV LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    python3.12 \
    libxmlsec1t64-openssl \
    poppler-utils \
    libreoffice-core \
    libreoffice-writer \
    libreoffice-calc \
    libreoffice-impress \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Reclaim UID 1000 from the stock ubuntu user, then create the app user with
# the same UID the Debian image uses (predictable bind mount permissions)
RUN (userdel -r ubuntu 2>/dev/null || true) && \
    useradd --create-home --shell /bin/bash --uid 1000 appuser

WORKDIR /app

# Venv + the editable-install anchor it references (see builder comment)
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

# TrustMark encode/decode roundtrip, run HERE so it proves the runtime stage:
# TrustMark swallows download failures and constructs a half-built model
# instead of raising, and a missing runtime shared library would only surface
# at import time — this catches both classes of breakage at build time.
RUN python -c "\
import xmlsec; \
from trustmark import TrustMark; \
from PIL import Image; \
tm = TrustMark(verbose=True, model_type='Q', loadRemover=False, device='cpu'); \
assert tm.encoder is not None and tm.decoder is not None, 'TrustMark weights missing'; \
s, ok, _ = tm.decode(tm.encode(Image.new('RGB', (256, 256), (128, 64, 32)), 'BAKETEST')); \
assert ok and s == 'BAKETEST', f'TrustMark roundtrip failed: {ok} {s!r}'; \
print('TrustMark runtime-stage roundtrip verified')"

# Copy application code
COPY backend/ backend/
COPY scripts/ scripts/
COPY alembic.ini ./

# Create directories
RUN mkdir -p /data/artifacts /data/chat_files /data/branding /var/log/mindrouter && \
    chown -R appuser:appuser /app /data /var/log/mindrouter

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Run the application
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--timeout-graceful-shutdown", "60"]
