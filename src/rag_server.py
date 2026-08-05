from __future__ import annotations

import ctypes
import os
from typing import Any

import chromadb
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BASE_DIR = os.path.dirname(__file__)
REPO_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
CHROMA_PATH = os.getenv(
	"GATHERPOINT_CHROMA_PATH",
	os.path.abspath(os.path.join(REPO_DIR, "gatherpoint_db")),
)
COLLECTION_NAME = os.getenv("GATHERPOINT_CHROMA_COLLECTION", "friend_profiles")
LIBINTERSECT_PATH = os.getenv(
	"GATHERPOINT_LIBINTERSECT_PATH",
	os.path.join(BASE_DIR, "libintersect.so"),
)
LOCAL_MODEL_PATH = os.getenv(
	"GATHERPOINT_EMBEDDER_PATH",
	os.path.join(REPO_DIR, "all-MiniLM-L6-v2"),
)
TOP_K = 3


class RetrievalRequest(BaseModel):
	query: str = Field(..., min_length=1, description="Natural-language query")


class IntersectionRequest(BaseModel):
	latitudes: list[float] = Field(..., min_length=1)
	longitudes: list[float] = Field(..., min_length=1)


def _load_intersection_library() -> ctypes.CDLL:
	library = ctypes.CDLL(LIBINTERSECT_PATH)
	library.calculate_center.argtypes = [
		ctypes.POINTER(ctypes.c_float),
		ctypes.POINTER(ctypes.c_float),
		ctypes.c_int,
		ctypes.POINTER(ctypes.c_float),
		ctypes.POINTER(ctypes.c_float),
	]
	# calculate_center returns 0 on success, or a non-zero hipError_t code.
	library.calculate_center.restype = ctypes.c_int
	return library


def _load_embedder(device: str) -> SentenceTransformer:
	# Prefer local files and avoid any Hugging Face network access by default.
	os.environ.setdefault("HF_HUB_OFFLINE", "1")
	os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

	candidates = [
		LOCAL_MODEL_PATH,
		os.path.join(BASE_DIR, "all-MiniLM-L6-v2"),
		os.path.join(os.path.dirname(REPO_DIR), "all-MiniLM-L6-v2"),
	]

	for model_path in candidates:
		if os.path.isdir(model_path):
			return SentenceTransformer(model_path, device=device)

	raise RuntimeError(
		"Embedding model directory not found. Expected one of: "
		f"{candidates}. Place the all-MiniLM-L6-v2 folder on disk or set "
		"GATHERPOINT_EMBEDDER_PATH to its absolute path."
	)


def _build_app() -> FastAPI:
	app = FastAPI(title="GatherPoint RAG Service")

	device = "cuda" if torch.cuda.is_available() else "cpu"
	client = chromadb.PersistentClient(path=CHROMA_PATH)
	collection = client.get_or_create_collection(name=COLLECTION_NAME)
	embedder = _load_embedder(device=device)
	try:
		intersection_library = _load_intersection_library()
	except OSError:
		intersection_library = None

	@app.get("/health")
	def healthcheck() -> dict[str, str]:
		return {
			"status": "ok",
			"device": device,
			"collection": COLLECTION_NAME,
		}

	@app.post("/retrieve_profiles")
	def retrieve_profiles(payload: RetrievalRequest) -> dict[str, Any]:
		query_text = payload.query.strip()

		if not query_text:
			raise HTTPException(status_code=400, detail="query must not be empty")

		query_embedding = embedder.encode([query_text]).tolist()
		results = collection.query(
			query_embeddings=query_embedding,
			n_results=TOP_K,
		)

		documents = results.get("documents", [[]])
		metadatas = results.get("metadatas", [[]])
		ids = results.get("ids", [[]])
		distances = results.get("distances", [[]])

		matches = []
		for index, document in enumerate(documents[0] if documents else []):
			matches.append(
				{
					"id": ids[0][index] if ids and ids[0] else None,
					"document": document,
					"metadata": metadatas[0][index] if metadatas and metadatas[0] else None,
					"distance": distances[0][index] if distances and distances[0] else None,
				}
			)

		return {
			"query": query_text,
			"matches": matches,
		}

	@app.post("/calculate_intersection")
	def calculate_intersection(payload: IntersectionRequest) -> dict[str, float]:
		if len(payload.latitudes) != len(payload.longitudes):
			raise HTTPException(
				status_code=400,
				detail="latitudes and longitudes must have the same length",
			)

		if intersection_library is None:
			raise HTTPException(
				status_code=503,
				detail=f"Shared library unavailable at {LIBINTERSECT_PATH}",
			)

		size = len(payload.latitudes)
		lat_array = (ctypes.c_float * size)(*payload.latitudes)
		lon_array = (ctypes.c_float * size)(*payload.longitudes)
		out_lat = ctypes.c_float()
		out_lon = ctypes.c_float()

		try:
			status = intersection_library.calculate_center(
				lat_array,
				lon_array,
				size,
				ctypes.byref(out_lat),
				ctypes.byref(out_lon),
			)
		except OSError as exc:
			raise HTTPException(
				status_code=500,
				detail=f"Intersection library call failed: {exc}",
			) from exc

		if status != 0:
			raise HTTPException(
				status_code=500,
				detail=f"HIP kernel failed with error code {status}",
			)

		return {
			"latitude": float(out_lat.value),
			"longitude": float(out_lon.value),
		}

	return app


app = _build_app()


if __name__ == "__main__":
	uvicorn.run(app, host="0.0.0.0", port=8000)
