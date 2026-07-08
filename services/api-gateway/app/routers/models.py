"""Model selector endpoint."""

from fastapi import APIRouter, Depends

from app.config import AVAILABLE_MODELS, settings
from app.dependencies import get_current_session
from app.models.schemas import ModelOption, ModelsResponse

router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
async def list_models(session: dict = Depends(get_current_session)) -> ModelsResponse:
    """Return the models available in the frontend selector and the default."""
    return ModelsResponse(
        available=[ModelOption(**m) for m in AVAILABLE_MODELS],
        default=settings.default_model_frontend,
    )
