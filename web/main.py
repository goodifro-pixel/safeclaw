"""SafeClaw AI Code Generator — Web API."""

from __future__ import annotations

import os
import textwrap
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="SafeClaw AI Code Generator", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "openai/gpt-oss-20b:free"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class GenerateRequest(BaseModel):
    description: str
    mode: str = "generate"  # generate | plan | review


class GenerateResponse(BaseModel):
    result: str
    model: str
    mode: str


async def _call_llm(prompt: str, system_prompt: str) -> str:
    """Call OpenRouter LLM and return content."""
    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY not configured."

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 4000,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(ENDPOINT, json=payload, headers=headers)
    except httpx.HTTPError as e:
        return f"API Error: request failed ({e})"

    try:
        data = resp.json()
    except ValueError:
        return f"API Error: non-JSON response (status {resp.status_code})"

    if isinstance(data, dict) and "error" in data:
        err = data["error"]
        msg = err.get("message", "Unknown") if isinstance(err, dict) else str(err)
        return f"API Error: {msg}"

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return "API Error: unexpected response format"


PLAN_PROMPT = textwrap.dedent("""\
    You are a principal software architect.
    Given this project description, produce a detailed implementation plan
    with: Tech Stack, Architecture, File Tree, Data Model, Error Handling,
    Testing Strategy, Implementation Order, Security Considerations.

    Description: {description}
""")

CODE_PROMPT = textwrap.dedent("""\
    You are a senior software engineer who writes clean, production-ready code.
    Generate the complete implementation with:
    - Full type annotations
    - Docstrings on all public functions
    - Error handling (no bare except)
    - SOLID principles
    - Input validation

    Description: {description}
""")

REVIEW_PROMPT = textwrap.dedent("""\
    You are a staff engineer doing a rigorous code review.
    Review this code for: correctness, type safety, error handling,
    security, performance, maintainability, testing.
    Give a Quality Score from 1-10.

    Code to review:
    {description}
""")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the main page."""
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "model": MODEL}


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate code, plan, or review using AI."""
    mode = req.mode
    desc = req.description.strip()

    if not desc:
        return GenerateResponse(result="Please provide a description.", model=MODEL, mode=mode)

    if mode == "plan":
        prompt = PLAN_PROMPT.format(description=desc)
        system = "You are a principal software architect. Be concrete and specific."
    elif mode == "review":
        prompt = REVIEW_PROMPT.format(description=desc)
        system = "You are a staff engineer. Be thorough. Give a quality score 1-10."
    else:
        prompt = CODE_PROMPT.format(description=desc)
        system = "You are a senior engineer. Write production-ready code."

    result = await _call_llm(prompt, system)
    return GenerateResponse(result=result, model=MODEL, mode=mode)


@app.get("/api/status")
async def status() -> dict[str, Any]:
    """Return API status and capabilities."""
    return {
        "status": "online",
        "model": MODEL,
        "capabilities": ["generate", "plan", "review"],
        "api_key_configured": bool(OPENROUTER_API_KEY),
    }
