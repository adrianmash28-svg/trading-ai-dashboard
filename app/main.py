from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routes.frontend import router as frontend_router
from app.routes.strategy_lab import router as strategy_lab_router
from app.routes.status import router as status_router


app = FastAPI(
    title="Mash Terminal API",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(frontend_router)
app.include_router(status_router, prefix="/api")
app.include_router(strategy_lab_router, prefix="/api")
