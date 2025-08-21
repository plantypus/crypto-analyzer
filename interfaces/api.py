"""FastAPI minimal app for crypto analyzer"""
try:
    from fastapi import FastAPI
except ImportError:  # pragma: no cover - FastAPI may not be installed
    FastAPI = None  # type: ignore

app = FastAPI(title="Crypto Analyzer API") if FastAPI else None

if FastAPI:
    @app.get("/")
    def read_root() -> dict:
        return {"status": "ok"}
