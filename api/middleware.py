# api/middleware.py - Custom middleware
import time
import logging
import traceback
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import uvicorn

from core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


class ErrorHandlerMiddleware:
    """Global error handling middleware"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Can modify response here if needed
                pass
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unhandled error: {e}")
            logger.error(traceback.format_exc())

            response = JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": (
                        str(e) if settings.debug else "An unexpected error occurred"
                    ),
                },
            )
            await response(scope, receive, send)


class LoggingMiddleware:
    """Request/response logging middleware"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        start_time = time.time()

        # Process request
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Log request/response
                process_time = time.time() - start_time
                logger.info(
                    f"{request.method} {request.url.path} - "
                    f"Status: {message['status']} - "
                    f"Time: {process_time:.3f}s"
                )
            await send(message)

        await self.app(scope, receive, send_wrapper)
