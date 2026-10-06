############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# matting_service/__main__.py: uvicorn entrypoint.
#
#   python -m matting_service
#
# Reads configuration from the environment (see server.py /
# README.md), builds the FastAPI app, and serves it with
# uvicorn on MATTING_HOST:MATTING_PORT.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""uvicorn entrypoint for the matting service."""

from __future__ import annotations

import logging

from .server import ServiceConfig, create_app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = ServiceConfig.from_env()
    app = create_app(config)

    import uvicorn

    # access_log off: keep request logging in one place (ours, which never
    # logs anything about a picture).
    uvicorn.run(app, host=config.host, port=config.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
