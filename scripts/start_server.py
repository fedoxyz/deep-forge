#!/usr/bin/env python3
"""Start the FastAPI backend server."""
import os, sys, uvicorn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == '__main__':
    uvicorn.run(
        "backend.api.server:app",
        host="0.0.0.0",
        port=int(os.environ.get("API_PORT", 8000)),
        reload=os.environ.get("DEV_MODE", "false").lower() == "true",
    )
