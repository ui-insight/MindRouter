############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# __init__.py: Services package exports
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Services for MindRouter."""

from backend.app.services.inference import InferenceService

__all__ = ["InferenceService"]
