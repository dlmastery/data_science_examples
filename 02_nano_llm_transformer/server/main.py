# FastAPI Streaming Server for NanoLlama LLM

import os
import sys
import json
import time
import asyncio
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add core folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../core')))
from inference import engine
from train import train_nanollama

app = FastAPI(
    title="NanoLlama SOTA LLM Microservice",
    description="High-performance streaming inference, attention visualization, and training telemetry API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schemas
class AttentionInspectRequest(BaseModel):
    prompt: str = Field(..., example="What is RoPE?")

class TokenizeRequest(BaseModel):
    text: str = Field(..., example="Hello! Who are you?")

class RetrainRequest(BaseModel):
    epochs: Optional[int] = Field(20, ge=5, le=50)
    batch_size: Optional[int] = Field(16, ge=4, le=64)
    lr: Optional[float] = Field(0.003, ge=0.0001, le=0.01)

# Preset Prompt Templates
PRESET_PROMPTS = [
    {
        "id": "intro",
        "title": "Who are you?",
        "category": "Identity",
        "system": "You are NanoLlama, a helpful, brilliant AI assistant.",
        "user": "Hello! Who are you and how were you built?"
    },
    {
        "id": "rope_explain",
        "title": "Explain RoPE Embeddings",
        "category": "AI Research",
        "system": "You are NanoLlama, an expert machine learning research assistant.",
        "user": "What is Rotary Position Embedding (RoPE) and how does it work?"
    },
    {
        "id": "swiglu_explain",
        "title": "Why SwiGLU Activations?",
        "category": "AI Research",
        "system": "You are NanoLlama, an expert machine learning research assistant.",
        "user": "Why do modern LLMs use SwiGLU instead of standard ReLU?"
    },
    {
        "id": "python_fib",
        "title": "Python Fibonacci Function",
        "category": "Coding",
        "system": "You are NanoLlama, an expert Python software engineer.",
        "user": "Write a Python function to compute the Fibonacci sequence."
    },
    {
        "id": "dragon_story",
        "title": "Baby Dragon Bedtime Story",
        "category": "Creative Story",
        "system": "You are NanoLlama, a warm, imaginative bedtime storyteller.",
        "user": "Tell me a short bedtime story about a baby dragon."
    },
    {
        "id": "math_speed",
        "title": "Math Reasoning: Speed Calculation",
        "category": "Logic & Math",
        "system": "You are NanoLlama, a precise and clear mathematical tutor.",
        "user": "If a train travels 120 miles in 2 hours, what is its average speed?"
    }
]

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": engine.model is not None,
        "parameters": engine.model.count_parameters() if engine.model else 0,
        "vocab_size": len(engine.tokenizer.vocab) if engine.tokenizer else 0,
        "telemetry": engine.telemetry
    }

@app.get("/api/prompts/presets")
def get_presets():
    return {"success": True, "presets": PRESET_PROMPTS}

@app.get("/api/chat/stream")
async def chat_stream(
    prompt: str,
    system: Optional[str] = "You are NanoLlama, a helpful, brilliant AI assistant.",
    temperature: Optional[float] = 0.7,
    top_p: Optional[float] = 0.9,
    top_k: Optional[int] = 40,
    repetition_penalty: Optional[float] = 1.1,
    max_tokens: Optional[int] = 150
):
    """Server-Sent Events (SSE) streaming token generator."""
    if not engine.model:
        engine.load()

    def event_stream():
        try:
            generator = engine.generate_stream(
                user_message=prompt,
                system_prompt=system,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty
            )
            for event in generator:
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/api/inspect/attention")
def inspect_attention(req: AttentionInspectRequest):
    """Extract multi-head attention weights for all layers for heatmap rendering."""
    try:
        if not engine.model:
            engine.load()
        result = engine.get_attention_matrix(req.prompt)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tokenize")
def tokenize_text(req: TokenizeRequest):
    """Analyze subwords, token IDs, and top-5 predicted probabilities."""
    try:
        if not engine.model:
            engine.load()
        result = engine.inspect_tokens_and_probabilities(req.text)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training/telemetry")
def get_training_telemetry():
    """Fetch pre-training and fine-tuning loss curves, perplexity, and parameter breakdown."""
    if engine.telemetry:
        return {"success": True, "data": engine.telemetry}

    meta_path = os.path.join(engine.checkpoints_dir, 'telemetry.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            engine.telemetry = data
            return {"success": True, "data": data}
    return {"success": False, "message": "Telemetry not found"}

@app.post("/api/train/live")
def trigger_live_training(req: RetrainRequest):
    """Trigger live in-memory training and reload updated model."""
    try:
        train_nanollama(
            epochs=req.epochs,
            batch_size=req.batch_size,
            lr=req.lr
        )
        engine.load()
        return {
            "success": True,
            "message": "NanoLlama model trained and checkpointed successfully!",
            "telemetry": engine.telemetry
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
