"""HTTP API and web UI.

    uvicorn omnisearch.api.server:app            # serves ./collections
    OMNISEARCH_ROOT=/path/to/collections uvicorn omnisearch.api.server:app

Endpoints
    GET    /api/collections
    GET    /api/collections/{name}                 stats, schema, facets
    POST   /api/collections/{name}/query           {"query": "<OmniQL>"}
    POST   /api/collections/{name}/search          form: text, image, filters, fusion, ...
    POST   /api/collections/{name}/documents       form: file, metadata (JSON)
    GET    /api/collections/{name}/documents/{id}
    DELETE /api/collections/{name}/documents/{id}
    GET    /api/collections/{name}/media/{id}/{field}
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from omnisearch.api.builder import build_query
from omnisearch.engine import Engine
from omnisearch.index.attribute import CategoryIndex
from omnisearch.query import PlanError, QuerySyntaxError

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 25 * 1024 * 1024


def create_app(root: str | Path = "collections") -> FastAPI:
    engine = Engine(root)
    uploads = Path(tempfile.mkdtemp(prefix="omnisearch-uploads-"))
    app = FastAPI(title="OmniSearch", version="0.1.0")
    app.state.engine = engine

    def coll(name):
        try:
            c = engine.collection(name)
        except KeyError:
            raise HTTPException(404, f"no collection {name!r}") from None
        c.query_path_roots = [uploads]
        return c

    def save_upload(f: UploadFile) -> Path:
        suffix = Path(f.filename or "").suffix.lower()[:8]
        dest = uploads / f"{uuid.uuid4().hex}{suffix}"
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out, length=1 << 20)
        if dest.stat().st_size > MAX_UPLOAD:
            dest.unlink()
            raise HTTPException(413, "upload too large")
        return dest

    def run(c, text):
        try:
            return c.query(text).to_dict()
        except (QuerySyntaxError, PlanError, ValueError, KeyError, PermissionError, FileNotFoundError) as e:
            raise HTTPException(400, str(e)) from None

    @app.get("/api/collections")
    def list_collections():
        out = []
        for name in engine.names():
            c = coll(name)
            out.append({"name": name, "docs": c.n_alive})
        return out

    @app.get("/api/collections/{name}")
    def describe(name: str):
        c = coll(name)
        facets, ranges = {}, {}
        for f in c.schema.attribute_fields():
            if isinstance(c.attrs[f.name], CategoryIndex):
                facets[f.name] = c.facets(f.name)
            else:
                ranges[f.name] = c.numeric_range(f.name)
        return {"stats": c.stats(), "schema": c.schema.to_dict(), "facets": facets, "ranges": ranges}

    @app.post("/api/collections/{name}/query")
    def query(name: str, body: dict):
        if not isinstance(body.get("query"), str):
            raise HTTPException(400, "body must be {\"query\": \"SEARCH ...\"}")
        return run(coll(name), body["query"])

    @app.post("/api/collections/{name}/search")
    def search(name: str, text: str = Form(""), filters: str = Form("{}"), fusion: str = Form("rrf"),
               limit: int = Form(12), strategy: str = Form("auto"), topk: str = Form("auto"),
               image: UploadFile | None = File(None)):
        c = coll(name)
        img = save_upload(image) if image is not None and image.filename else None
        if fusion not in ("rrf", "weighted", "auto"):
            raise HTTPException(400, "fusion must be rrf, weighted or auto")
        try:
            q = build_query(c, text, str(img) if img else None, json.loads(filters or "{}"), fusion, limit,
                            {"strategy": strategy, "topk": topk if fusion == "weighted" else None})
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            raise HTTPException(400, f"bad search form: {e}") from None
        result = run(c, q)
        result["omniql"] = q
        return result

    @app.post("/api/collections/{name}/documents")
    def add_document(name: str, metadata: str = Form("{}"), file: UploadFile | None = File(None)):
        c = coll(name)
        try:
            rec = json.loads(metadata)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"metadata must be JSON: {e}") from None
        if file is not None and file.filename:
            path = save_upload(file)
            kinds = {m.kind: m.name for m in c.schema.media_fields()}
            ext = path.suffix
            kind = "video" if ext in (".mp4", ".mov", ".webm", ".mkv") else "audio" if ext in (".wav", ".mp3", ".flac", ".ogg") else "image"
            if kind not in kinds:
                raise HTTPException(400, f"collection has no {kind} field")
            rec[kinds[kind]] = str(path)
        try:
            ids = c.add([rec])
            c.save()
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(400, str(e)) from None
        return {"ids": ids}

    @app.get("/api/collections/{name}/documents/{doc_id}")
    def get_document(name: str, doc_id: str):
        rec = coll(name).get(doc_id)
        if rec is None:
            raise HTTPException(404, "not found")
        return rec

    @app.delete("/api/collections/{name}/documents/{doc_id}")
    def delete_document(name: str, doc_id: str):
        c = coll(name)
        if not c.delete([doc_id]):
            raise HTTPException(404, "not found")
        c.save()
        return {"deleted": doc_id}

    @app.get("/api/collections/{name}/media/{doc_id}/{field}")
    def media(name: str, doc_id: str, field: str):
        c = coll(name)
        rec = c.get(doc_id)
        if rec is None or not rec.get(field) or c.schema.media_kind(field) is None:
            raise HTTPException(404, "no such media")
        try:
            return FileResponse(c.media.resolve(rec[field]))
        except ValueError:
            raise HTTPException(404, "no such media") from None

    @app.exception_handler(HTTPException)
    def http_error(_, exc: HTTPException):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def __getattr__(name):
    # `uvicorn omnisearch.api.server:app` builds the app lazily, so importing
    # this module (e.g. in tests) has no side effects.
    if name == "app":
        globals()["app"] = create_app(os.environ.get("OMNISEARCH_ROOT", "collections"))
        return globals()["app"]
    raise AttributeError(name)
