"""HTTP surface for the Recursive Self-Learning engine."""

from fastapi import APIRouter, Request

from app.core.auth import resolve_request_context
from app.schemas.chat import ChatRequest

router = APIRouter()


@router.post("/v1/rsl/chat")
async def rsl_chat(
    body: ChatRequest,
    request: Request,
    explore: bool = True,
    learn: bool = True,
):
    """Run one RSL cycle: route -> generate -> judge -> (revise) -> learn."""
    await resolve_request_context(request)
    engine = getattr(request.app.state, "rsl", None)
    if engine is None:
        return {"error": "RSL engine is not enabled"}
    # explore=false pins the policy to the current best arm, which is what an
    # evaluation run needs (otherwise exploration noise hides the improvement).
    return await engine.chat(
        body,
        epsilon_override=None if explore else 0.0,
        learn=learn,
    )


@router.get("/v1/rsl/stats")
async def rsl_stats(request: Request, limit: int = 50):
    """Inspect the learned Q-table and recent learning events."""
    await resolve_request_context(request)
    engine = getattr(request.app.state, "rsl", None)
    if engine is None:
        return {"error": "RSL engine is not enabled"}
    return await engine.stats(limit=limit)


@router.post("/v1/rsl/reset")
async def rsl_reset(request: Request):
    """Clear learned policy and events (useful before an A/B experiment)."""
    await resolve_request_context(request)
    engine = getattr(request.app.state, "rsl", None)
    if engine is None:
        return {"error": "RSL engine is not enabled"}
    await engine.reset()
    return {"status": "reset"}
