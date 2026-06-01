"""Model selector endpoint."""

from fastapi import APIRouter

from app.config import AVAILABLE_MODELS, settings
from app.models.schemas import ModelOption, ModelsResponse

router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """Return the models available in the frontend selector and the default."""
    return ModelsResponse(
        available=[ModelOption(**m) for m in AVAILABLE_MODELS],
        default=settings.default_model_frontend,
    )
