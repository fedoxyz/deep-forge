import os
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

PASSWORD = os.environ.get("FORGE_PASSWORD", "")

class PasswordMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not PASSWORD:
            return await call_next(request)
        if request.method == "OPTIONS":
            return await call_next(request)
        token = (
            request.headers.get("X-Forge-Password") or
            request.query_params.get("password")
        )
        if token != PASSWORD:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)
