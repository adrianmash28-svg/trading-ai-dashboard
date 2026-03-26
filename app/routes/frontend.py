from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"
TEMPLATES_DIR = FRONTEND_DIR / "templates"


router = APIRouter(tags=["frontend"])
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def render_frontend_page(request: Request, template_name: str, page_title: str, active_page: str):
    return templates.TemplateResponse(
        request=request,
        name=template_name,
        context={
            "request": request,
            "page_title": page_title,
            "active_page": active_page,
        },
    )


@router.get("/")
def home_page(request: Request):
    return render_frontend_page(request, "home.html", "Mash Terminal", "home")


@router.get("/command-center")
def command_center_page(request: Request):
    return render_frontend_page(request, "command_center.html", "Mash Terminal Command Center", "command-center")


@router.get("/market")
def market_page(request: Request):
    return render_frontend_page(request, "market.html", "Mash Terminal Market", "market")


@router.get("/strategy-lab")
def strategy_lab_page(request: Request):
    return render_frontend_page(request, "strategy_lab.html", "Mash Terminal Strategy Lab", "strategy-lab")
