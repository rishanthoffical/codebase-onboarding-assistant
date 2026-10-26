import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.api.webhook import router as webhook_router
from app.services.status_service import StatusService
from app.core.logging import configure_logging


configure_logging()

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
    """
    Lightweight liveness check.

    This endpoint confirms that the FastAPI application
    process is running. It does not require Redis or Chroma.
    """
    return {
        "status": "healthy",
    }


@app.get("/readiness")
def readiness():
    """
    Readiness check for Redis and Chroma.
    """
    print("READINESS: entered", flush=True)

    try:
        redis_status = StatusService.check_redis_connection()
        print("READINESS: Redis =", redis_status, flush=True)

        chroma_status = StatusService.check_chroma_connection()
        print("READINESS: Chroma =", chroma_status, flush=True)

        if not redis_status or not chroma_status:
            unavailable = []

            if not redis_status:
                unavailable.append("Redis")

            if not chroma_status:
                unavailable.append("Chroma")

            raise HTTPException(
                status_code=503,
                detail=(
                    "Service is not ready: "
                    + ", ".join(unavailable)
                    + " unavailable."
                ),
            )

        result = {"status": "ready"}

        print("READINESS: result =", result, flush=True)

        return result

    except HTTPException:
        raise

    except Exception as exc:
        print("READINESS: error =", repr(exc), flush=True)

        raise HTTPException(
            status_code=503,
            detail="Service readiness check failed.",
        )