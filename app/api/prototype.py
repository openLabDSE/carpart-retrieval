"""Prototype endpoints: part type with an "I don't know" rule, and the demo page for the garage owner."""
import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse

from pipeline.preprocessor import validate_image_full
from pipeline.type_identifier import MODES, TypeIdentifier

logger = logging.getLogger(__name__)
router = APIRouter()

ASSETS = Path(__file__).resolve().parents[2] / "indexes" / "typeproto"
PAGE = Path(__file__).resolve().parents[1] / "static" / "demo.html"
_identifier: TypeIdentifier | None = None


def get_identifier() -> TypeIdentifier:
    global _identifier
    if _identifier is None:
        if not (ASSETS / "gallery.npz").exists():
            raise HTTPException(503, "Prototype assets are missing. Run scripts/proto/build_type_prototype.py first.")
        _identifier = TypeIdentifier.load(ASSETS)
    return _identifier


@router.post("/identify-type")
async def identify_type(file: UploadFile = File(...), mode: str = Query("balanced")):
    """Name the part type in a photo, or say it is not sure. Brand is not available yet."""
    if mode not in MODES:
        raise HTTPException(400, f"mode must be one of {list(MODES)}")
    result = validate_image_full(await file.read())
    if not result.valid:
        raise HTTPException(result.http_status, result.error)
    ident = get_identifier()
    out = await asyncio.get_running_loop().run_in_executor(None, ident.identify, result.image, mode)
    out["warning"] = result.warning
    out["brand"] = {"status": "coming_soon"}
    return out


async def demo_page() -> HTMLResponse:
    return HTMLResponse(PAGE.read_text(encoding="utf-8"))
