############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# clef_service/__main__.py: uvicorn entrypoint.
#
#   python -m clef_service
#
# Reads configuration from the environment (see server.py /
# README.md), builds the FastAPI app, and serves it with
# uvicorn on CLEF_HOST:CLEF_PORT.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""uvicorn entrypoint for the Clef System One service."""

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

    # access_log off: uvicorn's access line is harmless, but keeping request
    # logging in one place (ours, which never logs content) is simpler to trust.
    uvicorn.run(app, host=config.host, port=config.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
