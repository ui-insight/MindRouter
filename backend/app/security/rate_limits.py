############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# rate_limits.py: Rate limiting
#
# RPM enforcement is now handled via Redis in
# backend/app/core/redis_client.py (check_rpm).
# This module is kept as a placeholder.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Rate limiting — RPM enforcement via Redis.

See :func:`backend.app.core.redis_client.check_rpm` for the
cross-worker implementation used by inference endpoints.
"""
