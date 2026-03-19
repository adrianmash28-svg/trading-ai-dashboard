from fastapi import APIRouter


router = APIRouter()


@router.get("/status")
def api_status():
    return {
        "app": "Mash Terminal",
        "status": "ready",
        "mode": "skeleton",
    }

