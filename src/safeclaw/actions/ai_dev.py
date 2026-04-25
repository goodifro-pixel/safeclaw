"""
SafeClaw AI Dev Pipeline — professional autonomous code→review→fix→deploy.

Orchestrates the AI Coder, Bug Finder, and Deployer into a single command
that takes a natural-language project description and produces deployed,
professionally reviewed code — all using free resources.

Pipeline stages:
  1. **Plan**      — architect the project (detailed tech stack, file tree)
  2. **Code**      — generate all files (types, docstrings, error handling)
  3. **Tests**     — generate comprehensive test suite (pytest)
  4. **Review**    — static analysis + AI code review with quality score
  5. **Fix**       — auto-fix all issues (iterative refinement)
  6. **Verify**    — re-review to confirm quality improvement
  7. **Deploy**    — push to GitHub + generate CI

Quality gate: pipeline reports a quality score (1-10) and only deploys
if the score meets the threshold (default: 6).

CLI surface:
    ai-dev <description>                       — run full pipeline
    ai-dev <description> --repo owner/repo     — include GitHub push
    ai-dev <description> --no-deploy           — skip deployment
    ai-dev <description> --quality 8           — set quality threshold
    ai-dev status                              — pipeline status
    ai-dev help                                — show usage
"""

from __future__ import annotations

import logging
import re
import textwrap
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from safeclaw.actions.ai_coder import AiCoderAction
from safeclaw.actions.base import BaseAction
from safeclaw.actions.bug_finder import BugFinderAction
from safeclaw.actions.deployer import DeployerAction

if TYPE_CHECKING:
    from safeclaw.core.engine import SafeClaw

logger = logging.getLogger(__name__)

_DEFAULT_QUALITY_THRESHOLD = 6
_MAX_FIX_ITERATIONS = 2


class AiDevAction(BaseAction):
    """Professional AI development pipeline with quality gates."""

    name = "ai_dev"
    description = "Professional autonomous AI development pipeline"

    def __init__(self) -> None:
        self._coder = AiCoderAction()
        self._finder = BugFinderAction()
        self._deployer = DeployerAction()

    async def execute(
        self,
        params: dict[str, Any],
        user_id: str,
        channel: str,
        engine: SafeClaw,
    ) -> str:
        raw = params.get("raw_input", "").strip()
        lower = raw.lower()

        if not raw or lower in ("ai-dev help", "ai dev help", "ai-dev", "ai dev"):
            return self._help()

        if lower in ("ai-dev status", "ai dev status"):
            return self._deployer._status()

        # Parse flags
        repo = ""
        skip_deploy = False
        quality_threshold = _DEFAULT_QUALITY_THRESHOLD
        description = raw

        for prefix in ("ai-dev", "ai dev"):
            if description.lower().startswith(prefix):
                description = description[len(prefix):].strip()

        if "--repo" in description:
            parts = description.split("--repo")
            description = parts[0].strip()
            repo = parts[1].strip().split()[0] if parts[1].strip() else ""

        if "--no-deploy" in description:
            description = description.replace("--no-deploy", "").strip()
            skip_deploy = True

        if "--quality" in description:
            parts = description.split("--quality")
            description = parts[0].strip()
            q_val = parts[1].strip().split()[0] if parts[1].strip() else ""
            if q_val.isdigit():
                quality_threshold = int(q_val)

        if not description:
            return "Provide a project description: `ai-dev <what to build>`"

        # ── Run pipeline ──────────────────────────────────────────────
        report: list[str] = [
            "# AI Dev Pipeline Report",
            "",
            f"> **Project:** {description}",
            f"> **Quality threshold:** {quality_threshold}/10",
            "",
        ]
        start = time.time()
        output_dir = Path.cwd() / "ai_generated"

        # Stage 1: Plan
        report.append("## Stage 1: Architecture Planning")
        plan_result = await self._coder.execute(
            {"raw_input": f"ai-code plan {description}"},
            user_id, channel, engine,
        )
        report.append(plan_result)
        report.append("")

        if "No LLM" in plan_result or "failed" in plan_result.lower():
            report.append("**Pipeline aborted** — LLM not available.")
            return "\n\n".join(report)

        # Stage 2: Code generation (includes tests + self-review + fix + scaffold)
        report.append("## Stage 2: Code Generation (Professional Pipeline)")
        gen_result = await self._coder.execute(
            {"raw_input": f"ai-code generate {description}"},
            user_id, channel, engine,
        )
        report.append(gen_result)
        report.append("")

        # Stage 3: Static analysis (bug finder)
        report.append("## Stage 3: Static Analysis")
        if output_dir.exists():
            static_result = await self._finder.execute(
                {"raw_input": f"bug-find static {output_dir}"},
                user_id, channel, engine,
            )
            report.append(static_result)

            security_result = await self._finder.execute(
                {"raw_input": f"bug-find security {output_dir}"},
                user_id, channel, engine,
            )
            if "No security" not in security_result:
                report.append(security_result)
        else:
            report.append("No output directory — skipping.")
        report.append("")

        # Stage 4: Iterative fix loop (if quality below threshold)
        report.append("## Stage 4: Quality Assurance")
        quality_score = self._extract_quality_score(gen_result)
        report.append(f"Initial quality score: **{quality_score}/10**")

        iteration = 0
        while (
            quality_score < quality_threshold
            and iteration < _MAX_FIX_ITERATIONS
            and output_dir.exists()
        ):
            iteration += 1
            report.append(f"\n### Fix Iteration {iteration}")

            source_files = [
                f for f in sorted(output_dir.rglob("*"))
                if f.is_file()
                and f.suffix in (".py", ".js", ".ts", ".go")
                and "test" not in f.name.lower()
                and f.name not in ("CODE_REVIEW.md", "ARCHITECTURE.md")
            ]

            fix_count = 0
            for src_file in source_files[:5]:
                fix_result = await self._coder.execute(
                    {
                        "raw_input": (
                            f"ai-code iterate {src_file} "
                            f"Apply all fixes from the code review. "
                            f"Add missing type annotations, docstrings, "
                            f"and error handling. Target quality 10/10."
                        ),
                    },
                    user_id, channel, engine,
                )
                if "Updated" in fix_result:
                    fix_count += 1

            report.append(f"Fixed {fix_count} file(s)")

            # Re-review after fixes
            review_result = await self._coder.execute(
                {"raw_input": f"ai-code review {output_dir}"},
                user_id, channel, engine,
            )
            new_score = self._extract_quality_score(review_result)
            report.append(f"Quality after fix: **{new_score}/10**")

            if new_score > quality_score:
                quality_score = new_score
            else:
                report.append("No improvement — stopping iterations.")
                break

        report.append("")

        # Stage 5: Quality Gate
        report.append("## Stage 5: Quality Gate")
        passed = quality_score >= quality_threshold
        status = "PASSED" if passed else "BELOW THRESHOLD"
        report.append(
            f"Score: **{quality_score}/10** | "
            f"Threshold: **{quality_threshold}/10** | "
            f"Status: **{status}**"
        )
        report.append("")

        # Stage 6: Deploy
        report.append("## Stage 6: Deployment")
        if skip_deploy:
            report.append("Deployment skipped (`--no-deploy`).")
        elif not output_dir.exists():
            report.append("No output to deploy.")
        elif not passed:
            report.append(
                f"Deployment skipped — quality {quality_score}/10 "
                f"below threshold {quality_threshold}/10.\n\n"
                f"Run `ai-code iterate <file> <feedback>` to improve, "
                f"then `ai-dev <desc> --quality {quality_score}` to retry."
            )
        else:
            ci_result = await self._deployer.execute(
                {"raw_input": f"deploy ci {output_dir}"},
                user_id, channel, engine,
            )
            report.append(ci_result)

            if repo:
                gh_result = await self._deployer.execute(
                    {"raw_input": f"deploy github {output_dir} --repo {repo}"},
                    user_id, channel, engine,
                )
                report.append(gh_result)
        report.append("")

        # ── Summary ──
        elapsed = time.time() - start
        report.extend([
            "---",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Time | {elapsed:.1f}s |",
            f"| Quality | {quality_score}/10 |",
            f"| Gate | {status} |",
            f"| Fix iterations | {iteration} |",
            "",
            "Next steps:",
            f"  `ai-code review {output_dir}` — detailed review",
            f"  `ai-code iterate {output_dir}/<file> <feedback>` — refine",
            f"  `bug-find {output_dir}` — static analysis",
        ])
        if not skip_deploy and not repo and passed:
            report.append(
                "  `deploy github <path> --repo owner/repo` — push to GitHub"
            )

        return "\n\n".join(report)

    @staticmethod
    def _extract_quality_score(text: str) -> int:
        """Extract quality score from pipeline or review output."""
        match = re.search(r"Quality(?:\s+Score)?[:\s]*\**(\d+)/10", text)
        if match:
            return int(match.group(1))
        match = re.search(r"\*\*(\d+)/10\*\*", text)
        if match:
            return int(match.group(1))
        return 0

    @staticmethod
    def _help() -> str:
        return textwrap.dedent("""\
            **AI Dev Pipeline** — professional autonomous development

            Usage:
              `ai-dev <description>`                    — full pipeline
              `ai-dev <description> --repo owner/repo`  — include GitHub push
              `ai-dev <description> --no-deploy`        — skip deployment
              `ai-dev <description> --quality 8`        — set quality threshold (default: 6)
              `ai-dev status`                           — deployment status
              `ai-dev help`                             — this message

            Pipeline stages:
              1. Architecture planning (tech stack, design patterns, file tree)
              2. Professional code generation (types, docstrings, SOLID)
              3. Auto test generation (pytest, edge cases, mocking)
              4. Static analysis + security scan
              5. Quality assurance (iterative fix loop until score met)
              6. Quality gate (deploy only if score >= threshold)
              7. Deployment (GitHub + CI/CD)

            Quality scoring: 1-10 scale.  Default deploy threshold: 6/10.
            The pipeline iterates up to 2 times to improve quality.

            All stages use free resources:
              - LLM: OpenRouter free models / Ollama (local)
              - Hosting: GitHub Pages / Fly.io free tier
              - CI/CD: GitHub Actions (free for public repos)
        """)
