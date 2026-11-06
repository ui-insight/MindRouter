############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_malloc_arena.py: the app runs with MALLOC_ARENA_MAX=2
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Worker memory: two glibc malloc arenas instead of up to 8 per core.

Prod's eight uvicorn workers held 4-6 GB each after weeks (33 GB of the
host's 62, 2026-11-06): memory freed by threads (OCR page conversion and
other asyncio.to_thread work) stays in per-thread arenas and is not returned.
The runtime image sets MALLOC_ARENA_MAX=2, and docker-compose passes it
through (prod starts the app from compose's `command:`, so the image's
environment is what reaches the workers unless compose overrides it).
"""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]


def test_the_runtime_image_sets_two_arenas():
    dockerfile = (ROOT / "Dockerfile").read_text()
    # The last stage is the image that runs; the builder's env is discarded.
    runtime = dockerfile[dockerfile.rindex("\nFROM "):]
    assert re.search(r"^ENV MALLOC_ARENA_MAX=2$", runtime, re.M)


def test_compose_passes_it_through_with_the_same_default():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    env = compose["services"]["app"]["environment"]
    assert "MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}" in env
