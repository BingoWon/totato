import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine import TokenEngine

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit",
)

engine = TokenEngine(MODEL_PATH)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print(f"Loading model: {MODEL_PATH}")
    engine.load_model()
    print("Model loaded.")
    yield


app = FastAPI(title="Token Explorer", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class InitRequest(BaseModel):
    prompt: str
    temperature: float = 1.0
    top_k: int = 200


class StepRequest(BaseModel):
    token_id: int
    probability: float
    rank: int
    temperature: float = 1.0
    top_k: int = 200


@app.get("/api/health")
def health():
    return {"status": "ready" if engine.loaded else "loading"}


@app.get("/api/model")
def model_info():
    if not engine.loaded:
        raise HTTPException(503, "Model not loaded")
    return engine.model_info


@app.post("/api/init")
def init_session(req: InitRequest):
    if not engine.loaded:
        raise HTTPException(503, "Model not loaded")
    dist = engine.init_session(req.prompt, req.temperature, req.top_k)
    return {"distribution": dist, "history": engine.history}


@app.post("/api/step")
def step(req: StepRequest):
    if not engine.loaded:
        raise HTTPException(503, "Model not loaded")
    dist = engine.step(req.token_id, req.probability, req.rank, req.temperature, req.top_k)
    return {"distribution": dist, "history": engine.history}


@app.post("/api/reset")
def reset():
    engine.reset()
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
