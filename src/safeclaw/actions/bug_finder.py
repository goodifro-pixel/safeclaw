"""
SafeClaw Bug Finder Action — static analysis + AI-powered code review.

Non-LLM features (always free):
  - Python AST analysis (unused imports, bare excepts, mutable defaults, …)
  - Pattern-based security checks (eval, exec, SQL injection hints, …)
  - Complexity estimation (nested depth, long functions)
  - Duplicate code detection (simple hash-based)

LLM features (optional):
  - Deep semantic code review via configured provider
  - Suggested fixes with diffs

CLI surface:
    bug-find <path>                   — full analysis (static + AI if available)
    bug-find static <path>            — static analysis only (no LLM)
    bug-find security <path>          — security-focused scan
    bug-find ai-review <path>         — LLM-only deep review
    bug-find help                     — show usage
"""

from __future__ import annotations

import ast
import logging
import re
import textwrap
from hashlib import md5
from pathlib import Path
from typing import TYPE_CHECKING, Any

from safeclaw.actions.base import BaseAction

if TYPE_CHECKING:
    from safeclaw.core.engine import SafeClaw

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Security patterns (regex-based, language-agnostic)
# ---------------------------------------------------------------------------

_SECURITY_PATTERNS: list[tuple[str, str, str]] = [
    (r"\beval\s*\(", "eval() usage", "HIGH — arbitrary code execution risk"),
    (r"\bexec\s*\(", "exec() usage", "HIGH — arbitrary code execution risk"),
    (r"\b__import__\s*\(", "dynamic __import__", "MEDIUM — hidden dependency"),
    (r"subprocess\.\w+\(.*shell\s*=\s*True", "shell=True in subprocess", "HIGH — command injection risk"),
    (r"os\.system\s*\(", "os.system() usage", "HIGH — prefer subprocess"),
    (r"pickle\.loads?\s*\(", "pickle deserialization", "HIGH — untrusted data risk"),
    (r"yaml\.load\s*\([^)]*\)", "yaml.load without SafeLoader", "MEDIUM — use yaml.safe_load"),
    (r"SELECT\s+.*\+\s*['\"]", "SQL string concatenation", "HIGH — SQL injection risk"),
    (r"password\s*=\s*['\"][^'\"]+['\"]", "hardcoded password", "HIGH — use env vars"),
    (r"api_key\s*=\s*['\"][A-Za-z0-9]{16,}['\"]", "hardcoded API key", "HIGH — use env vars"),
    (r"\.format\(.*input", "format with user input", "MEDIUM — potential injection"),
    (r"DEBUG\s*=\s*True", "DEBUG mode enabled", "LOW — disable in production"),
    (r"CORS\(.*allow_all", "CORS allow all origins", "MEDIUM — restrict origins"),
    (r"verify\s*=\s*False", "SSL verification disabled", "HIGH — MITM risk"),
]

# ---------------------------------------------------------------------------
# AST-based checks (Python only)
# ---------------------------------------------------------------------------


class _PythonAnalyzer(ast.NodeVisitor):
    """Walk a Python AST and collect issues."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.issues: list[dict[str, Any]] = []
        self._imported: set[str] = set()
        self._used: set[str] = set()
        self._function_depths: list[int] = []

    def run(self) -> list[dict[str, Any]]:
        try:
            tree = ast.parse(self.source)
        except SyntaxError as exc:
            return [{"line": exc.lineno or 0, "severity": "ERROR",
                      "message": f"SyntaxError: {exc.msg}"}]
        self.visit(tree)
        self._check_unused_imports()
        return self.issues

    # -- visitors --

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name
            self._imported.add(name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            name = alias.asname or alias.name
            self._imported.add(name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self._used.add(node.id)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.issues.append({
                "line": node.lineno,
                "severity": "WARN",
                "message": "Bare except — catches all exceptions including KeyboardInterrupt",
            })
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_function(node)
        self.generic_visit(node)

    # -- checks --

    def _check_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Mutable default arguments
        for default in node.args.defaults + node.args.kw_defaults:
            if default is not None and isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self.issues.append({
                    "line": node.lineno,
                    "severity": "WARN",
                    "message": f"Mutable default argument in `{node.name}()` — use None instead",
                })

        # Long functions
        body_lines = (node.end_lineno or node.lineno) - node.lineno
        if body_lines > 80:
            self.issues.append({
                "line": node.lineno,
                "severity": "INFO",
                "message": f"`{node.name}()` is {body_lines} lines — consider splitting",
            })

        # Deep nesting
        depth = self._max_depth(node, 0)
        if depth > 5:
            self.issues.append({
                "line": node.lineno,
                "severity": "WARN",
                "message": f"`{node.name}()` has nesting depth {depth} — simplify logic",
            })

    @staticmethod
    def _max_depth(node: ast.AST, current: int) -> int:
        max_d = current
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                max_d = max(max_d, _PythonAnalyzer._max_depth(child, current + 1))
            else:
                max_d = max(max_d, _PythonAnalyzer._max_depth(child, current))
        return max_d

    def _check_unused_imports(self) -> None:
        unused = self._imported - self._used
        # filter out common false positives
        false_pos = {"annotations", "__future__", "TYPE_CHECKING"}
        for name in sorted(unused - false_pos):
            self.issues.append({
                "line": 0,
                "severity": "INFO",
                "message": f"Possibly unused import: `{name}`",
            })


# ---------------------------------------------------------------------------
# Duplicate detection (simple line-hash approach)
# ---------------------------------------------------------------------------

def _find_duplicates(path: Path, min_lines: int = 6) -> list[dict[str, Any]]:
    """Find duplicate code blocks across files in *path*."""
    blocks: dict[str, list[tuple[str, int]]] = {}

    for fp in (path.rglob("*.py") if path.is_dir() else [path]):
        if not fp.is_file():
            continue
        try:
            lines = fp.read_text(errors="ignore").splitlines()
        except Exception:
            continue
        for start in range(len(lines) - min_lines + 1):
            chunk = "\n".join(lines[start : start + min_lines]).strip()
            if not chunk or chunk.count("\n") < min_lines - 2:
                continue
            h = md5(chunk.encode()).hexdigest()
            entry = (str(fp), start + 1)
            blocks.setdefault(h, []).append(entry)

    dupes: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for h, locations in blocks.items():
        if len(locations) > 1 and h not in seen_hashes:
            seen_hashes.add(h)
            dupes.append({
                "severity": "INFO",
                "message": f"Duplicate {min_lines}-line block found in {len(locations)} places",
                "locations": locations[:5],
            })
    return dupes[:20]


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

_REVIEW_PROMPT = textwrap.dedent("""\
    You are a senior code reviewer.  Analyse the following code and report:
    1. Bugs or logic errors
    2. Security vulnerabilities
    3. Performance issues
    4. Code-quality improvements

    For each issue give: severity (HIGH/MEDIUM/LOW), line number(s), and a
    one-line fix suggestion.

    ```{lang}
    {code}
    ```
""")


class BugFinderAction(BaseAction):
    """Find bugs via static analysis and optional AI review."""

    name = "bug_finder"
    description = "Static analysis + AI-powered bug finding"

    async def execute(
        self,
        params: dict[str, Any],
        user_id: str,
        channel: str,
        engine: SafeClaw,
    ) -> str:
        raw = params.get("raw_input", "").strip()
        lower = raw.lower()

        if not raw or "bug-find help" in lower or lower in ("bug-find", "bugfind"):
            return self._help()

        if lower.startswith("bug-find static") or lower.startswith("bugfind static"):
            path_str = raw.split("static", 1)[-1].strip()
            return self._static_analysis(path_str)

        if lower.startswith("bug-find security") or lower.startswith("bugfind security"):
            path_str = raw.split("security", 1)[-1].strip()
            return self._security_scan(path_str)

        if lower.startswith("bug-find ai-review") or lower.startswith("bugfind ai-review"):
            path_str = raw.split("ai-review", 1)[-1].strip()
            return await self._ai_review(path_str, engine)

        # Default: full analysis
        path_str = raw.replace("bug-find", "").replace("bugfind", "").strip()
        return await self._full_analysis(path_str, engine)

    # ------------------------------------------------------------------

    async def _full_analysis(self, path_str: str, engine: SafeClaw) -> str:
        """Run static checks, security scan, duplication check, and AI review."""
        parts: list[str] = []

        static = self._static_analysis(path_str)
        if static:
            parts.append(static)

        sec = self._security_scan(path_str)
        if sec and "No security" not in sec:
            parts.append(sec)

        # Duplicates
        path = Path(path_str).expanduser()
        if path.exists():
            dupes = _find_duplicates(path)
            if dupes:
                lines = ["**Duplicate Code**", ""]
                for d in dupes:
                    locs = ", ".join(f"`{loc[0]}:{loc[1]}`" for loc in d["locations"])
                    lines.append(f"  - {d['message']}: {locs}")
                parts.append("\n".join(lines))

        # AI review (best-effort)
        ai = await self._ai_review(path_str, engine)
        if ai and "No LLM" not in ai:
            parts.append(ai)

        if not parts:
            return f"No issues found in `{path_str}`."
        return "\n\n---\n\n".join(parts)

    def _static_analysis(self, path_str: str) -> str:
        """Run Python AST analysis on *path_str*."""
        path = Path(path_str).expanduser()
        if not path.exists():
            return f"Path not found: {path}"

        py_files = list(path.rglob("*.py")) if path.is_dir() else [path]
        py_files = [f for f in py_files if f.is_file()
                     and ".git" not in f.parts
                     and "__pycache__" not in f.parts
                     and "venv" not in f.parts]

        if not py_files:
            return f"No Python files found in `{path}`"

        all_issues: list[str] = []
        for fp in py_files[:50]:
            try:
                source = fp.read_text(errors="ignore")
            except Exception:
                continue
            analyzer = _PythonAnalyzer(source)
            issues = analyzer.run()
            for iss in issues:
                line = iss.get("line", "?")
                sev = iss["severity"]
                msg = iss["message"]
                all_issues.append(f"  [{sev}] `{fp}:{line}` — {msg}")

        if not all_issues:
            return f"**Static Analysis**: no issues in `{path}`"

        header = f"**Static Analysis** ({len(all_issues)} issues)\n"
        return header + "\n".join(all_issues[:60])

    def _security_scan(self, path_str: str) -> str:
        """Regex-based security scan."""
        path = Path(path_str).expanduser()
        if not path.exists():
            return f"Path not found: {path}"

        files = list(path.rglob("*")) if path.is_dir() else [path]
        code_exts = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rb", ".php",
                     ".java", ".rs", ".yaml", ".yml", ".sh"}
        files = [f for f in files if f.is_file() and f.suffix in code_exts
                 and ".git" not in f.parts and "node_modules" not in f.parts]

        findings: list[str] = []
        for fp in files[:100]:
            try:
                content = fp.read_text(errors="ignore")
            except Exception:
                continue
            for pattern, name, severity in _SECURITY_PATTERNS:
                for m in re.finditer(pattern, content):
                    line_no = content[:m.start()].count("\n") + 1
                    findings.append(f"  [{severity}] `{fp}:{line_no}` — {name}")
                    if len(findings) >= 50:
                        break
            if len(findings) >= 50:
                break

        if not findings:
            return f"**Security Scan**: No security issues in `{path}`"

        header = f"**Security Scan** ({len(findings)} findings)\n"
        return header + "\n".join(findings)

    async def _ai_review(self, path_str: str, engine: SafeClaw) -> str:
        """LLM-powered deep code review."""
        path = Path(path_str).expanduser()
        if not path.exists():
            return f"Path not found: {path}"

        # Read code
        if path.is_file():
            code = path.read_text(errors="ignore")[:8000]
            lang = path.suffix.lstrip(".")
        else:
            # Concatenate first few files
            parts: list[str] = []
            for fp in sorted(path.rglob("*.py"))[:5]:
                if ".git" in fp.parts or "__pycache__" in fp.parts:
                    continue
                try:
                    parts.append(f"# --- {fp.relative_to(path)} ---\n" + fp.read_text(errors="ignore"))
                except Exception:
                    continue
            code = "\n\n".join(parts)[:8000]
            lang = "python"

        if not code.strip():
            return "No code to review."

        from safeclaw.core.ai_writer import AIWriter

        task_providers = engine.config.get("task_providers", {})
        label = task_providers.get("coding")
        ai_writer = AIWriter.from_config(engine.config)
        if not ai_writer.providers:
            return "No LLM configured — run `bug-find static` for free analysis."

        prompt = _REVIEW_PROMPT.format(lang=lang, code=code)
        resp = await ai_writer.generate(
            prompt=prompt,
            provider_label=label,
            system_prompt="You are a meticulous senior code reviewer.",
            temperature=0.1,
            max_tokens=3000,
        )
        if resp.error:
            return f"AI review failed: {resp.error}"

        return (
            f"**AI Code Review** *(via {resp.provider}/{resp.model})*\n\n"
            f"{resp.content}"
        )

    @staticmethod
    def _help() -> str:
        return textwrap.dedent("""\
            **Bug Finder** — static analysis + AI code review

            Commands:
              `bug-find <path>`              — full analysis (static + AI)
              `bug-find static <path>`       — Python AST analysis only (free)
              `bug-find security <path>`     — security-focused scan (free)
              `bug-find ai-review <path>`    — LLM-powered deep review
              `bug-find help`                — this message

            Static and security scans require no LLM — always free.
        """)
