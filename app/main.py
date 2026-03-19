from fastapi import FastAPI

from app.routes.status import router as status_router


app = FastAPI(
    title="Mash Terminal API",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(status_router, prefix="/api")

