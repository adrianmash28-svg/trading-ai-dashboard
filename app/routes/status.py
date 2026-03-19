from fastapi import APIRouter

from shared.state import get_api_status_snapshot


router = APIRouter()


@router.get("/status")
def api_status():
    return get_api_status_snapshot()
