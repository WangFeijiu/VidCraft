"""Voice Studio Backend — FastAPI application entry point."""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import socketio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from loguru import logger

from app.config import Settings

settings = Settings()

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    ping_timeout=600,
    ping_interval=120,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Voice Studio starting up...")
    settings.ensure_dirs()

    if sys.platform == "win32":
        from app.utils.gpu import setup_cuda_dlls
        setup_cuda_dlls(settings)

    logger.info(f"Data directory: {settings.DATA_DIR.resolve()}")
    logger.info(f"Projects: {settings.projects_dir.resolve()}")
    yield
    logger.info("Voice Studio shutting down.")


app = FastAPI(title="Voice Studio", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.routes import projects, sentences, voice, compose, llm_config, img2vid, tools, video_edit

app.include_router(projects.router)
app.include_router(sentences.router)
app.include_router(voice.router)
app.include_router(compose.router)
app.include_router(llm_config.router)
app.include_router(img2vid.router)
app.include_router(tools.router)
app.include_router(video_edit.router)

socket_app = socketio.ASGIApp(sio, other_asgi_app=app)


@app.get("/")
async def index():
    html_path = Path(settings.FRONTEND_DIR) / "index.html"
    if not html_path.exists():
        return {"error": "Frontend not found", "path": str(html_path)}
    return FileResponse(html_path, media_type="text/html")


def create_app():
    return socket_app


if __name__ == "__main__":
    logger.info(f"Voice Studio -> http://{settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "app.main:socket_app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
