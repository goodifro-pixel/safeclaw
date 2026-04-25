"""
SafeClaw AI Coder Action — professional-grade autonomous code generation.

Generates complete, production-quality multi-file projects using free LLM
providers.  Enforces SOLID principles, type safety, error handling, tests,
documentation, and proper project structure.

Pipeline per generation:
  1. Architecture plan (tech stack, file tree, design decisions)
  2. Code generation (strict rules: types, docstrings, error handling)
  3. Test generation (unit tests for every module)
  4. Self-review (the LLM critiques its own output)
  5. Auto-fix (apply review feedback in a refinement pass)
  6. Scaffold (README, .gitignore, config files)

CLI surface:
    ai-code generate <description>       — full professional pipeline
    ai-code generate <desc> --lang py    — target a specific language
    ai-code plan <description>           — architecture plan only
    ai-code iterate <path> <feedback>    — refine existing code
    ai-code review <path>                — AI code review
    ai-code test <path>                  — generate tests for existing code
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
# Prompt templates — professional-grade
# ---------------------------------------------------------------------------

_PLAN_PROMPT = textwrap.dedent("""\
    You are a principal software architect with 20+ years of experience
    designing scalable, maintainable systems.

    Given this project description, produce a detailed implementation plan.

    Description:
    {description}

    Your plan MUST include:

    1. **Tech Stack** — language, framework, database, testing framework.
       Pick the best free, production-proven option for the task.
    2. **Architecture** — design pattern (MVC, hexagonal, microservice, etc.)
       and why it fits.  Name the layers/modules.
    3. **File Tree** — every file that will be created, with a one-line
       purpose comment.
    4. **Data Model** — key entities / types / schemas.
    5. **Error Handling Strategy** — how errors propagate, what gets logged.
    6. **Testing Strategy** — unit, integration, and edge-case categories.
    7. **Implementation Order** — numbered steps with dependencies.
    8. **Security Considerations** — input validation, auth, secrets management.

    Be concrete.  No vague advice like "consider using a database" — state
    which one and why.
""")

_CODE_PROMPT = textwrap.dedent("""\
    You are a senior software engineer who writes clean, production-ready,
    enterprise-grade code.  Generate the complete implementation.

    {language_hint}

    Description:
    {description}

    {plan_context}

    STRICT RULES — violations are unacceptable:

    1. **Type Safety** — every function has full type annotations (parameters
       AND return type).  Use generics where appropriate.  No `Any` unless
       truly unavoidable.
    2. **Docstrings** — every public class/function has a concise docstring
       explaining purpose, parameters, return value, and exceptions raised.
    3. **Error Handling** — never use bare `except:`.  Catch specific
       exceptions.  Raise custom exceptions for domain errors.  Always
       include helpful error messages.
    4. **SOLID Principles** — Single Responsibility, Open/Closed, Liskov
       Substitution, Interface Segregation, Dependency Inversion.  Each
       module does one thing well.
    5. **Clean Code** — short functions (< 30 lines), descriptive names,
       no magic numbers, no code duplication.  Constants are named.
    6. **Project Structure** — proper separation: `src/` for code,
       `tests/` for tests.  Use `__init__.py` files.
    7. **Input Validation** — validate all external inputs.  Never trust
       user data.  Use guard clauses.
    8. **Logging** — use stdlib `logging`, not `print()`.  Log at
       appropriate levels (DEBUG, INFO, WARNING, ERROR).
    9. **Configuration** — no hardcoded values.  Use environment variables
       or config files for anything that could change.
    10. **Dependencies** — include `requirements.txt` or `pyproject.toml`
        with pinned versions.

    OUTPUT FORMAT:
    • Wrap each file in a fenced code block whose info-string is the
      relative file path:
        ```path/to/file.py
        <code>
        ```
    • Output ONLY code blocks — no conversational text.
    • Include ALL imports, boilerplate, and entry points needed to run.
""")

_TEST_PROMPT = textwrap.dedent("""\
    You are a QA engineer who writes thorough, professional test suites.

    Generate comprehensive unit tests for the following code.

    {file_contents}

    RULES:
    1. Use `pytest` as the test framework.
    2. Test every public function and class method.
    3. Include:
       - **Happy path** tests (normal expected behavior)
       - **Edge cases** (empty input, None, boundary values, max/min)
       - **Error cases** (invalid input that should raise exceptions)
    4. Use descriptive test names: `test_<function>_<scenario>_<expected>`.
    5. Use `pytest.raises` for exception testing.
    6. Use fixtures for shared setup when it reduces duplication.
    7. Mock external dependencies (network, filesystem, APIs).
    8. Each test should be independent — no test should depend on another.
    9. Add type annotations to test functions.
    10. Aim for >= 90% code coverage of the source files.

    OUTPUT FORMAT:
    • Wrap each test file in a fenced code block:
        ```tests/test_<module>.py
        <code>
        ```
    • Output ONLY code blocks — no commentary.
""")

_REVIEW_PROMPT = textwrap.dedent("""\
    You are a staff engineer performing a rigorous code review.

    Review the following code for production readiness.

    {file_contents}

    Evaluate EACH of these dimensions.  For EVERY issue found, state:
      - **File:Line** — exact location
      - **Severity** — CRITICAL / HIGH / MEDIUM / LOW
      - **Issue** — what's wrong
      - **Fix** — concrete code change (not vague advice)

    Dimensions:
    1. **Correctness** — logic bugs, off-by-one, race conditions
    2. **Type Safety** — missing annotations, incorrect types, unsafe casts
    3. **Error Handling** — bare excepts, swallowed errors, missing validation
    4. **Security** — injection, hardcoded secrets, unsafe deserialization
    5. **Performance** — O(n²) where O(n) is possible, memory leaks, N+1 queries
    6. **Maintainability** — god classes, long functions, unclear naming
    7. **Testing** — untestable code, missing edge cases, tight coupling

    After listing all issues, provide:
    - **Quality Score**: 1-10 (10 = production ready)
    - **Summary**: one paragraph overall assessment
    - **Top 3 Priorities**: most impactful fixes
""")

_FIX_PROMPT = textwrap.dedent("""\
    You are a senior engineer fixing code based on a code review.

    === Current code ({path}) ===
    ```
    {code}
    ```

    === Code Review Findings ===
    {review_feedback}

    Apply ALL the fixes from the review.  Output the COMPLETE updated file.

    RULES:
    • Fix every issue mentioned in the review.
    • Do NOT introduce new issues.
    • Preserve all existing functionality.
    • Add missing type annotations.
    • Add missing error handling.
    • Add missing docstrings.
    • Output ONLY the fixed code — no commentary.
""")

_ITERATE_PROMPT = textwrap.dedent("""\
    You are a senior engineer refining code based on user feedback.

    === Current code ({path}) ===
    ```
    {code}
    ```

    === Feedback ===
    {feedback}

    Apply the requested changes.  Output the COMPLETE updated file.

    RULES:
    • Maintain all existing type annotations and docstrings.
    • Maintain or improve error handling.
    • Do NOT introduce regressions.
    • Follow SOLID principles.
    • Output ONLY the updated code — no commentary.
""")

_SCAFFOLD_TEMPLATES: dict[str, str] = {
    ".gitignore": textwrap.dedent("""\
        # Python
        __pycache__/
        *.py[cod]
        *$py.class
        *.so
        dist/
        build/
        *.egg-info/
        .eggs/
        venv/
        .venv/
        env/

        # IDE
        .idea/
        .vscode/
        *.swp
        *.swo
        *~

        # OS
        .DS_Store
        Thumbs.db

        # Environment
        .env
        .env.local

        # Testing
        .coverage
        htmlcov/
        .pytest_cache/
        .mypy_cache/
    """),
}


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
        if "." in info or "/" in info:
            files.append((info, code))
        else:
            ext = _lang_to_ext(info)
            files.append((f"main{ext}", code))
    if not files:
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


def _collect_source_files(path: Path, max_files: int = 10) -> str:
    """Read source files from a path and format them for a prompt."""
    if path.is_file():
        content = path.read_text(errors="ignore")[:8000]
        return f"=== {path.name} ===\n```\n{content}\n```"

    parts: list[str] = []
    extensions = {".py", ".js", ".ts", ".go", ".rs", ".java", ".rb"}
    count = 0
    for f in sorted(path.rglob("*")):
        if f.is_file() and f.suffix in extensions and count < max_files:
            content = f.read_text(errors="ignore")[:6000]
            rel = f.relative_to(path)
            parts.append(f"=== {rel} ===\n```\n{content}\n```")
            count += 1
    return "\n\n".join(parts) if parts else "(no source files found)"


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class AiCoderAction(BaseAction):
    """Professional AI code generation with built-in quality assurance."""

    name = "ai_coder"
    description = "Professional AI-powered code generation"

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

        if lower.startswith("ai-code review") or lower.startswith("ai code review"):
            path_str = raw.split("review", 1)[-1].strip()
            return await self._review(path_str, engine)

        if lower.startswith("ai-code test") or lower.startswith("ai code test"):
            path_str = raw.split("test", 1)[-1].strip()
            return await self._generate_tests(path_str, engine)

        if lower.startswith("ai-code iterate") or lower.startswith("ai code iterate"):
            return await self._iterate(raw, engine)

        if lower.startswith("ai-code generate") or lower.startswith("ai code generate"):
            desc = raw.split("generate", 1)[-1].strip()
            return await self._generate(desc, engine)

        return await self._generate(raw, engine)

    # ------------------------------------------------------------------
    # Subcommands
    # ------------------------------------------------------------------

    async def _plan(self, description: str, engine: SafeClaw) -> str:
        """Generate a detailed architecture plan."""
        if not description:
            return "Provide a project description: `ai-code plan <description>`"

        ai_writer, label = self._get_writer(engine)
        if not ai_writer:
            return self._no_llm_msg()

        prompt = _PLAN_PROMPT.format(description=description)
        resp = await ai_writer.generate(
            prompt=prompt,
            provider_label=label,
            system_prompt=(
                "You are a principal software architect. "
                "Be concrete and specific — no vague advice."
            ),
            temperature=0.3,
        )
        if resp.error:
            return f"Planning failed: {resp.error}"
        return (
            f"**Implementation Plan** *(via {resp.provider}/{resp.model})*\n\n"
            f"{resp.content}"
        )

    async def _generate(self, description: str, engine: SafeClaw) -> str:
        """Full professional generation pipeline: plan → code → tests → review → fix → scaffold."""
        if not description:
            return "Provide a description: `ai-code generate <what to build>`"

        lang_hint = ""
        if "--lang" in description:
            parts = description.split("--lang")
            description = parts[0].strip()
            lang = parts[1].strip().split()[0] if parts[1].strip() else ""
            lang_hint = f"Use {lang} as the primary language."

        ai_writer, label = self._get_writer(engine)
        if not ai_writer:
            return self._no_llm_msg()

        output_dir = Path.cwd() / "ai_generated"
        output_dir.mkdir(parents=True, exist_ok=True)
        report: list[str] = []

        # ── Step 1: Architecture plan ──
        plan_prompt = _PLAN_PROMPT.format(description=description)
        plan_resp = await ai_writer.generate(
            prompt=plan_prompt,
            provider_label=label,
            system_prompt="You are a principal software architect.",
            temperature=0.3,
        )
        plan_context = ""
        if not plan_resp.error:
            plan_context = f"Follow this architecture plan:\n\n{plan_resp.content}"
            plan_file = output_dir / "ARCHITECTURE.md"
            plan_file.write_text(f"# Architecture Plan\n\n{plan_resp.content}")
            report.append("1. Architecture plan created")

        # ── Step 2: Generate code ──
        code_prompt = _CODE_PROMPT.format(
            description=description,
            language_hint=lang_hint,
            plan_context=plan_context,
        )
        code_resp = await ai_writer.generate(
            prompt=code_prompt,
            provider_label=label,
            system_prompt=(
                "You are a senior engineer. Write production-grade code. "
                "Every function must have type annotations and docstrings."
            ),
            temperature=0.2,
            max_tokens=4000,
        )
        if code_resp.error:
            return f"Code generation failed: {code_resp.error}"

        files = _parse_code_blocks(code_resp.content)
        written: list[str] = []
        for rel_path, code in files:
            dest = output_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(code)
            written.append(str(dest.relative_to(output_dir)))
        report.append(f"2. Generated {len(written)} file(s)")

        # ── Step 3: Generate tests ──
        source_content = _collect_source_files(output_dir)
        test_prompt = _TEST_PROMPT.format(file_contents=source_content)
        test_resp = await ai_writer.generate(
            prompt=test_prompt,
            provider_label=label,
            system_prompt=(
                "You are a QA engineer. Write thorough pytest tests."
            ),
            temperature=0.2,
            max_tokens=4000,
        )
        test_files_written = 0
        if not test_resp.error:
            test_files = _parse_code_blocks(test_resp.content)
            for rel_path, code in test_files:
                if not rel_path.startswith("test"):
                    rel_path = f"tests/{rel_path}"
                dest = output_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(code)
                test_files_written += 1
                written.append(str(dest.relative_to(output_dir)))
        report.append(f"3. Generated {test_files_written} test file(s)")

        # ── Step 4: Self-review ──
        review_content = _collect_source_files(output_dir)
        review_prompt = _REVIEW_PROMPT.format(file_contents=review_content)
        review_resp = await ai_writer.generate(
            prompt=review_prompt,
            provider_label=label,
            system_prompt=(
                "You are a staff engineer. Be thorough and specific. "
                "Give a quality score from 1-10."
            ),
            temperature=0.3,
        )
        quality_score = "N/A"
        review_text = ""
        if not review_resp.error:
            review_text = review_resp.content
            score_match = re.search(r"Quality Score[:\s]*(\d+)", review_text)
            if score_match:
                quality_score = score_match.group(1) + "/10"
            review_file = output_dir / "CODE_REVIEW.md"
            review_file.write_text(f"# Code Review\n\n{review_text}")
            report.append(f"4. Self-review complete — Quality: {quality_score}")

        # ── Step 5: Auto-fix based on review ──
        fix_count = 0
        if review_text and quality_score != "N/A":
            score_val = int(re.search(r"\d+", quality_score).group())
            if score_val < 9:
                source_files = [
                    f for f in sorted(output_dir.rglob("*"))
                    if f.is_file() and f.suffix in (".py", ".js", ".ts", ".go")
                    and "test" not in f.name.lower()
                    and f.name != "CODE_REVIEW.md"
                    and f.name != "ARCHITECTURE.md"
                ]
                for src_file in source_files[:5]:
                    code_text = src_file.read_text(errors="ignore")
                    fix_prompt = _FIX_PROMPT.format(
                        path=src_file.name,
                        code=code_text[:6000],
                        review_feedback=review_text[:3000],
                    )
                    fix_resp = await ai_writer.generate(
                        prompt=fix_prompt,
                        provider_label=label,
                        system_prompt=(
                            "You are a senior engineer fixing code review "
                            "findings. Output ONLY the complete fixed file."
                        ),
                        temperature=0.1,
                        max_tokens=4000,
                    )
                    if not fix_resp.error:
                        fixed_code = fix_resp.content
                        m = re.match(
                            r"```[^\n]*\n(.*?)```", fixed_code, re.DOTALL,
                        )
                        if m:
                            fixed_code = m.group(1)
                        src_file.write_text(fixed_code)
                        fix_count += 1
        report.append(f"5. Auto-fixed {fix_count} file(s)")

        # ── Step 6: Scaffold ──
        scaffold_count = 0
        for filename, content in _SCAFFOLD_TEMPLATES.items():
            target = output_dir / filename
            if not target.exists():
                target.write_text(content)
                scaffold_count += 1
                written.append(filename)

        readme = output_dir / "README.md"
        if not readme.exists():
            readme_content = self._generate_readme(description, written)
            readme.write_text(readme_content)
            scaffold_count += 1
            written.append("README.md")
        report.append(f"6. Added {scaffold_count} scaffold file(s)")

        # ── Final report ──
        file_list = "\n".join(f"  - `{p}`" for p in written)
        pipeline_report = " → ".join(report)
        return (
            f"**Code Generated** *(via {code_resp.provider}/{code_resp.model})*\n\n"
            f"Quality Score: **{quality_score}**\n\n"
            f"Pipeline: {pipeline_report}\n\n"
            f"Files in `{output_dir}`:\n{file_list}\n\n"
            f"Next steps:\n"
            f"  `ai-code review {output_dir}` — detailed code review\n"
            f"  `ai-code iterate {output_dir}/<file> <feedback>` — refine\n"
            f"  `bug-find {output_dir}` — static analysis\n"
            f"  `deploy {output_dir}` — deploy to free hosting"
        )

    async def _review(self, path_str: str, engine: SafeClaw) -> str:
        """Perform a professional AI code review."""
        if not path_str:
            return "Provide a file or directory: `ai-code review <path>`"

        path = Path(path_str).expanduser()
        if not path.exists():
            return f"Path not found: {path}"

        ai_writer, label = self._get_writer(engine)
        if not ai_writer:
            return self._no_llm_msg()

        source_content = _collect_source_files(path)
        prompt = _REVIEW_PROMPT.format(file_contents=source_content)
        resp = await ai_writer.generate(
            prompt=prompt,
            provider_label=label,
            system_prompt=(
                "You are a staff engineer. Be thorough and specific."
            ),
            temperature=0.3,
        )
        if resp.error:
            return f"Review failed: {resp.error}"

        quality_score = "N/A"
        score_match = re.search(r"Quality Score[:\s]*(\d+)", resp.content)
        if score_match:
            quality_score = score_match.group(1) + "/10"

        return (
            f"**Code Review** *(via {resp.provider}/{resp.model})*\n"
            f"Quality Score: **{quality_score}**\n\n"
            f"{resp.content}"
        )

    async def _generate_tests(self, path_str: str, engine: SafeClaw) -> str:
        """Generate professional test suite for existing code."""
        if not path_str:
            return "Provide a file or directory: `ai-code test <path>`"

        path = Path(path_str).expanduser()
        if not path.exists():
            return f"Path not found: {path}"

        ai_writer, label = self._get_writer(engine)
        if not ai_writer:
            return self._no_llm_msg()

        source_content = _collect_source_files(path)
        prompt = _TEST_PROMPT.format(file_contents=source_content)
        resp = await ai_writer.generate(
            prompt=prompt,
            provider_label=label,
            system_prompt="You are a QA engineer. Write thorough pytest tests.",
            temperature=0.2,
            max_tokens=4000,
        )
        if resp.error:
            return f"Test generation failed: {resp.error}"

        test_files = _parse_code_blocks(resp.content)
        output_dir = path if path.is_dir() else path.parent
        test_dir = output_dir / "tests"
        test_dir.mkdir(parents=True, exist_ok=True)

        written: list[str] = []
        for rel_path, code in test_files:
            name = Path(rel_path).name
            if not name.startswith("test_"):
                name = f"test_{name}"
            dest = test_dir / name
            dest.write_text(code)
            written.append(str(dest))

        file_list = "\n".join(f"  - `{p}`" for p in written)
        return (
            f"**Tests Generated** *(via {resp.provider}/{resp.model})*\n\n"
            f"Files:\n{file_list}\n\n"
            f"Run: `cd {output_dir} && pytest tests/ -v`"
        )

    async def _iterate(self, raw: str, engine: SafeClaw) -> str:
        """Refine existing code based on feedback."""
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
            system_prompt=(
                "You are a senior engineer. Maintain code quality. "
                "Output ONLY the complete updated file."
            ),
            temperature=0.2,
            max_tokens=4000,
        )
        if resp.error:
            return f"Iteration failed: {resp.error}"

        new_code = resp.content
        m = re.match(r"```[^\n]*\n(.*?)```", new_code, re.DOTALL)
        if m:
            new_code = m.group(1)
        path.write_text(new_code)

        return (
            f"**Updated** `{path}` *(via {resp.provider}/{resp.model})*\n\n"
            f"Run `ai-code review {path}` to check quality."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_readme(description: str, files: list[str]) -> str:
        """Generate a professional README.md."""
        file_tree = "\n".join(f"    {f}" for f in sorted(files))
        return textwrap.dedent(f"""\
            # Project

            {description}

            ## Quick Start

            ```bash
            # Install dependencies
            pip install -r requirements.txt

            # Run
            python main.py

            # Test
            pytest tests/ -v
            ```

            ## Project Structure

            ```
            {file_tree}
            ```

            ## Development

            ```bash
            # Install dev dependencies
            pip install -r requirements.txt pytest

            # Run tests
            pytest tests/ -v --cov

            # Lint
            ruff check .
            ```

            ## License

            MIT
        """)

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
            **AI Coder** — professional-grade code generation

            Commands:
              `ai-code generate <description>`         — full pipeline (plan → code → tests → review → fix)
              `ai-code generate <desc> --lang python`  — specify language
              `ai-code plan <description>`             — architecture plan only
              `ai-code review <path>`                  — AI code review with quality score
              `ai-code test <path>`                    — generate test suite
              `ai-code iterate <file> <feedback>`      — refine existing code
              `ai-code help`                           — this message

            Pipeline per generation:
              1. Architecture plan (tech stack, file tree, design)
              2. Code generation (types, docstrings, error handling)
              3. Test generation (pytest, edge cases, mocks)
              4. Self-review (quality score 1-10)
              5. Auto-fix (apply review findings)
              6. Scaffold (README, .gitignore, config)

            Requires an LLM provider in config.yaml (Ollama = free, local).
        """)
