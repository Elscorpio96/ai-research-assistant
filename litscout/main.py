"""ASGI entry point: `uvicorn litscout.main:app`."""

from .app import create_app

app = create_app()
