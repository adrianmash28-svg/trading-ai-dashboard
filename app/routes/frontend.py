from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse


BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"


router = APIRouter(tags=["frontend"])


@router.get("/strategy-lab")
def strategy_lab_page():
    return FileResponse(FRONTEND_DIR / "strategy_lab.html")
