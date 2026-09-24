from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.api.webhook import router as webhook_router


app = FastAPI(
    title="Codebase Onboarding Assistant",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:3003",
        "http://localhost:3004",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002",
        "http://127.0.0.1:3003",
        "http://127.0.0.1:3004",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(api_router)

app.include_router(
    webhook_router,
    prefix="/webhook",
)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
    }


@app.on_event("startup")
async def startup_event():
    print("Application startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    print("Application shutdown")
