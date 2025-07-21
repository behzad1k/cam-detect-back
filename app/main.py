from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.core.config import settings
from app.api.routes import router
from app.websocket.handlers import websocket_endpoint
from app.utils.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.TITLE,
    root_path=settings.ROOT_PATH,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    debug=settings.DEBUG
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_route(websocket: WebSocket):
    await websocket_endpoint(websocket)

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {settings.TITLE} v{settings.VERSION}")
    logger.info(f"Device: {settings.DEVICE}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Available models: {list(settings.MODEL_PATHS.keys())}")
