import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI()

async def gen():
    yield "event: test\ndata: hello\n\n"
    await asyncio.sleep(1)
    yield "event: complete\ndata: {}\n\n"

@app.get("/test-stream")
async def test_stream():
    return StreamingResponse(gen(), media_type="text/event-stream")

uvicorn.run(app, host="127.0.0.1", port=8010)
