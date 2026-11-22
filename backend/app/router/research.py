from fastapi import APIRouter, HTTPException
from market_research.schemas.research import ResearchResponse, ResearchRequest
from market_research.agents.orchestrator import run_pipeline
from market_research.config import ConfigError, get_settings
import asyncio
import logging
from pathlib import Path


logger = logging.getLogger(__name__)

router = APIRouter(tags=["Research"])


@router.post("/api/research", response_model=ResearchResponse)
async def research(request: ResearchRequest) -> ResearchResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be empty.")

    try:
        get_settings()
    except ConfigError as error:
        raise HTTPException(
            status_code=503,
            detail=f"{error} Copy .env.example to .env, fill it in and restart.",
        ) from error

    research_path = (
        Path(request.research_path).resolve()
        if request.research_path
        else None
    )

    try:
        def on_progress(message: str) -> None:
            logger.info("Research progress: %s", message)

        state = await asyncio.to_thread(
            run_pipeline,
            query,
            on_progress,
            research_path,
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"The research workflow could not be completed: {error}",
        ) from error

    return ResearchResponse(state=state.model_dump(mode="json"))