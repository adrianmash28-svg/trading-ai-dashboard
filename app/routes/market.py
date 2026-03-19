from fastapi import APIRouter

from shared.state import get_market_overview_snapshot


router = APIRouter(prefix="/market", tags=["market"])


@router.get("/overview")
def market_overview():
    return get_market_overview_snapshot()
