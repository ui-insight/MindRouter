############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# search_api.py: Public web search API endpoints
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Public web search API endpoints with API key authentication.

Endpoints:
    POST /v1/search   — OpenAI-style path
    POST /api/search   — Ollama-style path (same handler)
"""

import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.auth import authenticate_request
from backend.app.db import crud
from backend.app.db.models import ApiKey, Modality, RequestStatus, User
from backend.app.db.session import get_async_db
from backend.app.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["search"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Web search request body."""

    query: str = Field(..., description="The search query string", min_length=1, max_length=2000)
    max_results: Optional[int] = Field(
        None, description="Maximum results to return (default from config)", ge=1, le=50
    )


class SearchResultItem(BaseModel):
    """A single search result."""

    title: str
    url: str
    snippet: str
    published: Optional[str] = None


class SearchResponse(BaseModel):
    """Web search response."""

    query: str
    provider: str
    results: list[SearchResultItem]
    total_results: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _check_search_quota(db: AsyncSession, user: User, config: dict):
    """Check if user has sufficient quota for a search request."""
    await crud.reset_quota_if_needed(db, user.id)
    quota = await crud.get_user_quota(db, user.id)
    group_budget = user.group.token_budget if user.group else 0
    if quota and group_budget > 0 and quota.tokens_used >= group_budget:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Token quota exceeded",
        )


async def _deduct_search_tokens(
    db: AsyncSession, user: User, api_key: ApiKey, config: dict, latency_ms: int
):
    """Deduct fixed token cost for a search request and log it."""
    token_cost = int(config.get("search.quota_tokens_per_request", 50))
    if token_cost <= 0:
        return

    # Record as a lightweight request for audit/quota tracking
    request_record = await crud.create_request(
        db,
        user_id=user.id,
        api_key_id=api_key.id,
        model="web_search",
        endpoint="/v1/search",
        modality=Modality.CHAT,
    )

    await crud.update_request_completed(
        db,
        request_record.id,
        prompt_tokens=token_cost,
        completion_tokens=0,
    )

    # Update quota (DB + Redis)
    await crud.update_quota_usage(db, user.id, token_cost)
    await db.commit()
    await crud.incr_quota_redis(user.id, token_cost)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


async def _do_search(
    body: SearchRequest,
    request: Request,
    db: AsyncSession,
    auth: tuple,
):
    """Shared handler for both /v1/search and /api/search."""
    from backend.app.services.search.registry import get_search_config, PROVIDERS

    user, api_key = auth
    config = await get_search_config(db)

    # Check if search is enabled
    if not config.get("search.enabled", True):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Web search is not enabled on this server",
        )

    # Check quota
    await _check_search_quota(db, user, config)

    # Resolve provider
    provider_key = config.get("search.provider", "brave")
    provider = PROVIDERS.get(provider_key)
    if not provider:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Search provider '{provider_key}' is not available",
        )

    max_results = body.max_results or int(config.get("search.max_results", 10))

    # Execute search
    t0 = time.monotonic()
    try:
        results = await provider.search(
            body.query, max_results=max_results, config=config
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception:
        logger.exception("search_provider_error", provider=provider_key)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Search provider returned an error",
        )
    latency_ms = int((time.monotonic() - t0) * 1000)

    logger.info(
        "search_request",
        user_id=user.id,
        provider=provider_key,
        query_len=len(body.query),
        results=len(results),
        latency_ms=latency_ms,
    )

    # Deduct quota tokens
    await _deduct_search_tokens(db, user, api_key, config, latency_ms)

    return SearchResponse(
        query=body.query,
        provider=provider.display_name,
        results=[
            SearchResultItem(
                title=r.title,
                url=r.url,
                snippet=r.snippet,
                published=r.published,
            )
            for r in results
        ],
        total_results=len(results),
    )


@router.post("/v1/search", response_model=SearchResponse)
async def v1_search(
    body: SearchRequest,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth: tuple = Depends(authenticate_request),
):
    """Search the web via MindRouter's configured search provider.

    Authenticates with your MindRouter API key. Deducts a small fixed
    token cost from your quota per request.
    """
    return await _do_search(body, request, db, auth)


@router.post("/api/search", response_model=SearchResponse)
async def api_search(
    body: SearchRequest,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth: tuple = Depends(authenticate_request),
):
    """Search the web (Ollama-style path). Same as POST /v1/search."""
    return await _do_search(body, request, db, auth)
