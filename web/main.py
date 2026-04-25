"""SafeClaw AI Code Generator — Web API with multilingual chat."""

from __future__ import annotations

import json
import os
import textwrap
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="SafeClaw AI Code Generator", version="2.0.0")
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


class ChatRequest(BaseModel):
    messages: list[dict[str, str]]
    language: str = "auto"


class GenerateResponse(BaseModel):
    result: str
    model: str
    mode: str


class ChatResponse(BaseModel):
    reply: str
    model: str
    detected_language: str


async def _call_llm(
    prompt: str,
    system_prompt: str,
    *,
    max_tokens: int = 4000,
    temperature: float = 0.2,
) -> str:
    """Call OpenRouter LLM and return content."""
    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY not configured."

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(ENDPOINT, json=payload, headers=headers)

        if resp.status_code != 200:
            return f"API Error: HTTP {resp.status_code}"

        data = resp.json()
    except json.JSONDecodeError:
        return "API Error: unexpected response format (non-JSON)"
    except httpx.HTTPError as exc:
        return f"API Error: {exc}"

    if "error" in data:
        err = data["error"]
        msg = err.get("message", "Unknown") if isinstance(err, dict) else str(err)
        return f"API Error: {msg}"

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return "API Error: unexpected response format"


async def _call_llm_chat(messages: list[dict[str, str]], system_prompt: str) -> str:
    """Call OpenRouter LLM with full conversation history."""
    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY not configured."

    all_messages = [{"role": "system", "content": system_prompt}]
    all_messages.extend(
        {"role": m["role"], "content": m.get("content", "")}
        for m in messages
        if m.get("role") in ("user", "assistant")
    )

    payload = {
        "model": MODEL,
        "messages": all_messages,
        "temperature": 0.4,
        "max_tokens": 4000,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(ENDPOINT, json=payload, headers=headers)

        if resp.status_code != 200:
            return f"API Error: HTTP {resp.status_code}"

        data = resp.json()
    except json.JSONDecodeError:
        return "API Error: unexpected response format (non-JSON)"
    except httpx.HTTPError as exc:
        return f"API Error: {exc}"

    if "error" in data:
        err = data["error"]
        msg = err.get("message", "Unknown") if isinstance(err, dict) else str(err)
        return f"API Error: {msg}"

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return "API Error: unexpected response format"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PLAN_PROMPT = textwrap.dedent("""\
    You are a principal software architect.
    Given this project description, produce a detailed implementation plan
    with: Tech Stack, Architecture, File Tree, Data Model, Error Handling,
    Testing Strategy, Implementation Order, Security Considerations.

    Description: {description}
""")

CODE_PROMPT = textwrap.dedent("""\
    You are a world-class senior software engineer who can write production-ready
    code in ANY programming language and framework. You have mastered:

    Languages: Python, JavaScript, TypeScript, Go, Rust, C, C++, Java, Kotlin,
    Swift, Ruby, PHP, C#, Scala, Haskell, Elixir, Dart, Lua, R, MATLAB,
    Assembly, SQL, Bash, PowerShell, Perl, and more.

    Frameworks: React, Vue, Angular, Next.js, Django, Flask, FastAPI, Express,
    Spring Boot, Rails, Laravel, ASP.NET, Gin, Actix-web, Rocket, SwiftUI,
    Flutter, Jetpack Compose, Qt, Electron, Tauri, and more.

    Tools: Docker, Kubernetes, Terraform, AWS, GCP, Azure, CI/CD pipelines,
    databases (PostgreSQL, MongoDB, Redis, etc.), message queues, GraphQL, gRPC.

    Rules:
    - Detect the requested language/framework from the description
    - If no language is specified, default to Python
    - Write complete, runnable code — not snippets
    - Full type annotations (where the language supports them)
    - Docstrings / comments on all public functions
    - Error handling (no bare except, proper error types)
    - SOLID principles
    - Input validation
    - Follow idiomatic conventions for the target language

    IMPORTANT: Always respond in the same language the user writes in.
    If the user writes in Ukrainian, respond in Ukrainian.
    If in English, respond in English. And so on.

    Description: {description}
""")

REVIEW_PROMPT = textwrap.dedent("""\
    You are a staff engineer doing a rigorous code review.
    You can review code in ANY programming language.
    Review this code for: correctness, type safety, error handling,
    security, performance, maintainability, testing.
    Give a Quality Score from 1-10.

    IMPORTANT: Respond in the same language the user writes in.

    Code to review:
    {description}
""")

CHAT_SYSTEM = textwrap.dedent("""\
    You are SafeClaw AI — a world-class programming assistant that can write
    code in ANY programming language and framework. You are helpful, precise,
    and always provide working code examples.

    Key capabilities:
    - Write code in 50+ programming languages
    - Design system architectures
    - Debug and fix code
    - Explain concepts clearly
    - Review code for quality
    - Help with DevOps, databases, APIs, frontend, backend, mobile, ML/AI

    CRITICAL RULE: Always respond in the SAME LANGUAGE the user writes in.
    If the user writes in Ukrainian (українська), respond in Ukrainian.
    If in English, respond in English.
    If in Spanish, respond in Spanish.
    If in German, respond in German.
    If in French, respond in French.
    If in Japanese, respond in Japanese.
    If in Chinese, respond in Chinese.
    If in Polish, respond in Polish.
    And so on for any language.

    When providing code, always use proper markdown code blocks with the
    language specified (```python, ```javascript, ```go, etc.).

    Be concise but thorough. If the user asks a simple question, give a
    simple answer. If they ask for complex code, provide complete implementations.
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
        return GenerateResponse(
            result="Please provide a description.", model=MODEL, mode=mode
        )

    if mode == "plan":
        prompt = PLAN_PROMPT.format(description=desc)
        system = "You are a principal software architect. Be concrete and specific. Respond in the same language the user writes in."
    elif mode == "review":
        prompt = REVIEW_PROMPT.format(description=desc)
        system = "You are a staff engineer. Be thorough. Give a quality score 1-10. Respond in the same language the user writes in."
    else:
        prompt = CODE_PROMPT.format(description=desc)
        system = "You are a world-class senior engineer. Write production-ready code in any language. Respond in the same language the user writes in."

    result = await _call_llm(prompt, system)
    return GenerateResponse(result=result, model=MODEL, mode=mode)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Chat with AI assistant — multilingual, multi-turn conversation."""
    if not req.messages:
        return ChatResponse(
            reply="Please send a message.", model=MODEL, detected_language="unknown"
        )

    last_msg = req.messages[-1].get("content", "").strip()
    if not last_msg:
        return ChatResponse(
            reply="Empty message.", model=MODEL, detected_language="unknown"
        )

    reply = await _call_llm_chat(req.messages, CHAT_SYSTEM)

    return ChatResponse(reply=reply, model=MODEL, detected_language=req.language)


@app.get("/api/status")
async def status() -> dict[str, Any]:
    """Return API status and capabilities."""
    return {
        "status": "online",
        "model": MODEL,
        "capabilities": ["generate", "plan", "review", "chat"],
        "api_key_configured": bool(OPENROUTER_API_KEY),
        "supported_languages": [
            "Python", "JavaScript", "TypeScript", "Go", "Rust", "C", "C++",
            "Java", "Kotlin", "Swift", "Ruby", "PHP", "C#", "Scala",
            "Haskell", "Elixir", "Dart", "Lua", "R", "SQL", "Bash",
        ],
    }
