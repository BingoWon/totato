import os
from contextlib import asynccontextmanager

from engine import TokenEngine
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit",
)


class PredictRequest(BaseModel):
    text: str
    system_prompt: str | None = None
    temperature: float = 1.0
    top_k: int = 200


class TokenizeRequest(BaseModel):
    text: str


engine = TokenEngine(MODEL_PATH)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print(f"Loading model: {MODEL_PATH}")
    engine.load_model()
    print("Model loaded.")
    yield


app = FastAPI(title="Token Explorer", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {"status": "ready" if engine.loaded else "loading"}


@app.get("/api/model")
def model_info():
    if not engine.loaded:
        raise HTTPException(503, "Model not loaded")
    return engine.model_info


@app.post("/api/tokenize")
def tokenize(req: TokenizeRequest):
    if not engine.loaded:
        raise HTTPException(503, "Model not loaded")
    return {"tokens": engine.tokenize(req.text)}


@app.post("/api/predict")
def predict(req: PredictRequest):
    if not engine.loaded:
        raise HTTPException(503, "Model not loaded")
    result = engine.predict(req.text, req.system_prompt, req.temperature, req.top_k)
    if result is None:
        raise HTTPException(409, "Superseded by newer request")
    return {"distribution": result}


@app.post("/api/reset")
def reset():
    engine.reset()
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)
