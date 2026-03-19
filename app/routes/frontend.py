from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse


BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"


router = APIRouter(tags=["frontend"])


@router.get("/")
def home_page():
    return FileResponse(FRONTEND_DIR / "home.html")


@router.get("/command-center")
def command_center_page():
    return FileResponse(FRONTEND_DIR / "command_center.html")


@router.get("/market")
def market_page():
    return FileResponse(FRONTEND_DIR / "market.html")


@router.get("/strategy-lab")
def strategy_lab_page():
    return FileResponse(FRONTEND_DIR / "strategy_lab.html")
