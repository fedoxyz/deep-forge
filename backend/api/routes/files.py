from fastapi import APIRouter
from fastapi.responses import FileResponse
import os

files_router = APIRouter(prefix="/api/files", tags=["files"])

@files_router.get("/{file_path:path}")
async def serve_file(file_path: str):
    '''Serve a file from disk (for sample images, etc).'''
    # Security: only allow serving from outputs/samples directories
    allowed_prefixes = ['./outputs/', '/data/outputs/', './logs/']
    
    decoded = file_path
    if not any(decoded.startswith(p) for p in allowed_prefixes):
        # Also allow absolute paths that are in outputs
        if not os.path.isabs(decoded):
            decoded = os.path.join('./outputs', decoded)
    
    if not os.path.exists(decoded):
        raise HTTPException(404, "File not found")
    
    return FileResponse(decoded)
