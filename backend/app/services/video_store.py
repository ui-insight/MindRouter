############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# video_store.py: On-disk layout for rendered video artifacts.
#
# Range-capable streaming delivery is added with the content endpoint;
# for now this just owns the path layout under settings.video_storage_path.
# See docs/video-generation-plan.md.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Filesystem layout for generated video artifacts."""

import os


def job_output_path(storage_root: str, user_id: int, job_uuid: str) -> str:
    """Absolute path for a job's final MP4, creating the parent dir.

    Layout: <root>/<user_id>/<job_uuid>.mp4 — sharded by user so retention and
    per-user storage accounting are simple directory walks.
    """
    user_dir = os.path.join(storage_root, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    return os.path.join(user_dir, f"{job_uuid}.mp4")
