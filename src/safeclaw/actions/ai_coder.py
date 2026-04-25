"""
SafeClaw AI Coder Action — autonomous code generation from natural-language specs.

Generates complete, multi-file projects using free LLM providers (OpenRouter
free-tier models, Ollama, or any other configured provider).  Output is written
to disk and optionally fed into the Bug Finder for review.

CLI surface:
    ai-code generate <description>       — generate code from a description
    ai-code generate <desc> --lang py    — target a specific language
    ai-code iterate <path> <feedback>    — refine existing code
    ai-code plan <description>           — generate a plan before coding
    ai-code help                         — show usage
"""

from __future__ import annotations

import logging
import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

from safeclaw.actions.base import BaseAction

if TYPE_CHECKING:
    from safeclaw.core.engine import SafeClaw

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_PLAN_PROMPT = textwrap.dedent("""\
    You are a senior software architect.  Given the following project
    description, produce a concise implementation plan.

    Description:
    {description}

    Respond with:
    1. Recommended tech stack / language (pick the best free option).
    2. File tree (list every file that will be created).
    3. Key design decisions.
    4. Step-by-step implementation order.

    Keep it practical — no vague advice.
""")

_CODE_PROMPT = textwrap.dedent("""\
    You are an expert software engineer.  Generate **complete, production-ready
    code** for the following request.

    {language_hint}

    Description:
    {description}

    RULES:
    • Output ONLY code — no conversational commentary.
    • When producing multiple files, wrap each one in a fenced block whose
      info-string is the relative file path, e.g.:
        ```path/to/file.py
        <code>
        ```
    • Include all imports, dependencies, and boilerplate needed to run.
    • Add a requirements.txt / package.json when external libraries are used.
    • Write clear docstrings and type hints where applicable.
""")

_ITERATE_PROMPT = textwrap.dedent("""\
    You are an expert software engineer.  The user wants to improve the code
    shown below based on their feedback.

    === Current code ({path}) ===
    ```
    {code}
    ```

    === Feedback ===
    {feedback}

    Produce the complete updated file.  Output ONLY the new code.
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(
    r"```(?P<path>[^\n`]+)\n(?P<code>.*?)```",
    re.DOTALL,
)


def _parse_code_blocks(text: str) -> list[tuple[str, str]]:
    """Extract ``(file_path, code)`` pairs from fenced code blocks.

    If the info-string looks like a file path (contains a dot or a slash) the
    block is treated as a named file.  Otherwise we fall back to a single
    ``main`` file with the dominant language extension.
    """
    files: list[tuple[str, str]] = []
    for m in _FENCE_RE.finditer(text):
        info = m.group("path").strip()
        code = m.group("code")
        # Heuristic: contains dot or slash → treat as path
        if "." in info or "/" in info:
            files.append((info, code))
        else:
            ext = _lang_to_ext(info)
            files.append((f"main{ext}", code))
    if not files:
        # No fenced blocks — treat the whole response as a single file
        files.append(("main.py", text))
    return files


def _lang_to_ext(lang: str) -> str:
    mapping = {
        "python": ".py", "py": ".py",
        "javascript": ".js", "js": ".js",
        "typescript": ".ts", "ts": ".ts",
        "rust": ".rs", "go": ".go",
        "java": ".java", "kotlin": ".kt",
        "ruby": ".rb", "php": ".php",
        "html": ".html", "css": ".css",
        "bash": ".sh", "sh": ".sh",
        "sql": ".sql", "yaml": ".yaml",
        "json": ".json", "toml": ".toml",
    }
    return mapping.get(lang.lower(), ".py")


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class AiCoderAction(BaseAction):
    """Generate complete code projects from natural-language descriptions."""

    name = "ai_coder"
    description = "AI-powered code generation from descriptions"

    async def execute(
        self,
        params: dict[str, Any],
        user_id: str,
        channel: str,
        engine: SafeClaw,
    ) -> str:
        raw = params.get("raw_input", "").strip()
        lower = raw.lower()

        if not raw or "ai-code help" in lower or lower in ("ai-code", "ai code"):
            return self._help()

        if lower.startswith("ai-code plan") or lower.startswith("ai code plan"):
            desc = raw.split("plan", 1)[-1].strip()
            return await self._plan(desc, engine)

        if lower.startswith("ai-code iterate") or lower.startswith("ai code iterate"):
            return await self._iterate(raw, engine)

        if lower.startswith("ai-code generate") or lower.startswith("ai code generate"):
            desc = raw.split("generate", 1)[-1].strip()
            return await self._generate(desc, engine)

        # Default: treat entire input as a generation request
        return await self._generate(raw, engine)

    # ------------------------------------------------------------------
    # Subcommands
    # ------------------------------------------------------------------

    async def _plan(self, description: str, engine: SafeClaw) -> str:
        if not description:
            return "Provide a project description: `ai-code plan <description>`"

        ai_writer, label = self._get_writer(engine)
        if not ai_writer:
            return self._no_llm_msg()

        prompt = _PLAN_PROMPT.format(description=description)
        resp = await ai_writer.generate(
            prompt=prompt,
            provider_label=label,
            system_prompt="You are a senior software architect.",
            temperature=0.3,
        )
        if resp.error:
            return f"Planning failed: {resp.error}"
        return f"**Implementation Plan** *(via {resp.provider}/{resp.model})*\n\n{resp.content}"

    async def _generate(self, description: str, engine: SafeClaw) -> str:
        if not description:
            return "Provide a description: `ai-code generate <what to build>`"

        # Detect --lang flag
        lang_hint = ""
        if "--lang" in description:
            parts = description.split("--lang")
            description = parts[0].strip()
            lang = parts[1].strip().split()[0] if parts[1].strip() else ""
            lang_hint = f"Use {lang} as the primary language."

        ai_writer, label = self._get_writer(engine)
        if not ai_writer:
            return self._no_llm_msg()

        prompt = _CODE_PROMPT.format(
            description=description,
            language_hint=lang_hint,
        )
        resp = await ai_writer.generate(
            prompt=prompt,
            provider_label=label,
            system_prompt=(
                "You are an expert software engineer who writes clean, "
                "production-ready code."
            ),
            temperature=0.2,
            max_tokens=4000,
        )
        if resp.error:
            return f"Code generation failed: {resp.error}"

        files = _parse_code_blocks(resp.content)

        # Determine output directory
        output_dir = Path.cwd() / "ai_generated"
        output_dir.mkdir(parents=True, exist_ok=True)

        written: list[str] = []
        for rel_path, code in files:
            dest = output_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(code)
            written.append(str(dest))

        file_list = "\n".join(f"  - `{p}`" for p in written)
        return (
            f"**Code Generated** *(via {resp.provider}/{resp.model})*\n\n"
            f"Files written to `{output_dir}`:\n{file_list}\n\n"
            f"Next steps:\n"
            f"  `ai-code iterate {output_dir}/<file> <feedback>` — refine\n"
            f"  `bug-find {output_dir}` — scan for bugs\n"
            f"  `deploy {output_dir}` — deploy to free hosting"
        )

    async def _iterate(self, raw: str, engine: SafeClaw) -> str:
        parts = raw.split("iterate", 1)[-1].strip().split(None, 1)
        if len(parts) < 2:
            return "Usage: `ai-code iterate <file_path> <feedback>`"

        path_str, feedback = parts
        path = Path(path_str).expanduser()
        if not path.exists():
            return f"File not found: {path}"

        code = path.read_text(errors="ignore")
        ai_writer, label = self._get_writer(engine)
        if not ai_writer:
            return self._no_llm_msg()

        prompt = _ITERATE_PROMPT.format(
            path=path, code=code[:6000], feedback=feedback,
        )
        resp = await ai_writer.generate(
            prompt=prompt,
            provider_label=label,
            system_prompt="You are an expert software engineer.",
            temperature=0.2,
            max_tokens=4000,
        )
        if resp.error:
            return f"Iteration failed: {resp.error}"

        # Write back
        new_code = resp.content
        # Strip outer fenced block if present
        m = re.match(r"```[^\n]*\n(.*?)```", new_code, re.DOTALL)
        if m:
            new_code = m.group(1)
        path.write_text(new_code)

        return (
            f"**Updated** `{path}` *(via {resp.provider}/{resp.model})*\n\n"
            f"Run `bug-find {path}` to check for issues."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_writer(engine: SafeClaw):
        from safeclaw.core.ai_writer import AIWriter

        task_providers = engine.config.get("task_providers", {})
        label = task_providers.get("coding")
        ai_writer = AIWriter.from_config(engine.config)
        if not ai_writer.providers:
            return None, None
        return ai_writer, label

    @staticmethod
    def _no_llm_msg() -> str:
        return (
            "No LLM configured.  Add a **free** provider in `config/config.yaml`:\n\n"
            "```yaml\nai_providers:\n"
            '  - label: "openrouter-free"\n'
            '    provider: "openrouter"\n'
            '    api_key: "${OPENROUTER_API_KEY}"\n'
            '    model: "meta-llama/llama-3.1-8b-instruct:free"\n'
            "```\n\n"
            "Or install Ollama locally (100 % free, no API key):\n"
            "  `curl -fsSL https://ollama.com/install.sh | sh && ollama pull llama3.1`"
        )

    @staticmethod
    def _help() -> str:
        return textwrap.dedent("""\
            **AI Coder** — generate complete code from descriptions

            Commands:
              `ai-code generate <description>`         — generate code
              `ai-code generate <desc> --lang python`  — specify language
              `ai-code plan <description>`             — architecture plan first
              `ai-code iterate <file> <feedback>`      — refine existing code
              `ai-code help`                           — this message

            Requires an LLM provider in config.yaml (Ollama = free, local).
        """)
