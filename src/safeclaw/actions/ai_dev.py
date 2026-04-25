"""
SafeClaw AI Dev Pipeline — fully autonomous code→review→fix→deploy cycle.

Orchestrates the AI Coder, Bug Finder, and Deployer into a single command
that takes a natural-language project description and produces deployed,
reviewed code — all using free resources.

Pipeline stages:
  1. **Plan**   — architect the project (AI Coder)
  2. **Code**   — generate all files (AI Coder)
  3. **Review** — static analysis + AI review (Bug Finder)
  4. **Fix**    — auto-fix issues found in review (AI Coder iterate)
  5. **Deploy** — push to GitHub + generate CI (Deployer)

CLI surface:
    ai-dev <description>                       — run full pipeline
    ai-dev <description> --repo owner/repo     — include GitHub push
    ai-dev <description> --no-deploy           — skip deployment
    ai-dev status                              — pipeline status
    ai-dev help                                — show usage
"""

from __future__ import annotations

import logging
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


class AiDevAction(BaseAction):
    """Full AI development pipeline: describe → code → review → fix → deploy."""

    name = "ai_dev"
    description = "Autonomous AI development pipeline"

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

        if not description:
            return "Provide a project description: `ai-dev <what to build>`"

        # ── Run pipeline ──────────────────────────────────────────────
        report: list[str] = ["# AI Dev Pipeline Report", ""]
        start = time.time()

        # Stage 1: Plan
        report.append("## 1. Planning")
        plan_result = await self._coder.execute(
            {"raw_input": f"ai-code plan {description}"},
            user_id, channel, engine,
        )
        report.append(plan_result)
        report.append("")

        # Stage 2: Generate code
        report.append("## 2. Code Generation")
        gen_result = await self._coder.execute(
            {"raw_input": f"ai-code generate {description}"},
            user_id, channel, engine,
        )
        report.append(gen_result)
        report.append("")

        # Determine output directory
        output_dir = Path.cwd() / "ai_generated"

        # Stage 3: Bug review
        report.append("## 3. Code Review")
        if output_dir.exists():
            review_result = await self._finder.execute(
                {"raw_input": f"bug-find {output_dir}"},
                user_id, channel, engine,
            )
            report.append(review_result)
        else:
            report.append("No output directory found — skipping review.")
        report.append("")

        # Stage 4: Auto-fix (iterate on files with issues)
        report.append("## 4. Auto-Fix")
        if output_dir.exists() and "issues" in review_result.lower():
            fix_count = 0
            for py_file in sorted(output_dir.rglob("*.py"))[:5]:
                fix_result = await self._coder.execute(
                    {
                        "raw_input": (
                            f"ai-code iterate {py_file} "
                            f"Fix all bugs and issues found in code review. "
                            f"Ensure production quality."
                        ),
                    },
                    user_id, channel, engine,
                )
                if "Updated" in fix_result:
                    fix_count += 1
                    report.append(f"  - Fixed: `{py_file.name}`")
            if fix_count == 0:
                report.append("No auto-fixes applied.")
        else:
            report.append("No critical issues — skipping auto-fix.")
        report.append("")

        # Stage 5: Deploy
        report.append("## 5. Deployment")
        if skip_deploy:
            report.append("Deployment skipped (`--no-deploy`).")
        elif output_dir.exists():
            # Generate CI
            ci_result = await self._deployer.execute(
                {"raw_input": f"deploy ci {output_dir}"},
                user_id, channel, engine,
            )
            report.append(ci_result)

            # Push to GitHub if repo specified
            if repo:
                gh_result = await self._deployer.execute(
                    {"raw_input": f"deploy github {output_dir} --repo {repo}"},
                    user_id, channel, engine,
                )
                report.append(gh_result)
        else:
            report.append("No output to deploy.")

        elapsed = time.time() - start
        report.extend([
            "",
            "---",
            f"**Pipeline completed in {elapsed:.1f}s**",
            "",
            "Next steps:",
            f"  `bug-find {output_dir}` — re-scan for remaining issues",
            f"  `ai-code iterate {output_dir}/<file> <feedback>` — refine code",
        ])
        if not skip_deploy and not repo:
            report.append("  `deploy github <path> --repo owner/repo` — push to GitHub")

        return "\n\n".join(report)

    @staticmethod
    def _help() -> str:
        return textwrap.dedent("""\
            **AI Dev Pipeline** — autonomous code generation + review + deployment

            Usage:
              `ai-dev <description>`                    — full pipeline
              `ai-dev <description> --repo owner/repo`  — include GitHub push
              `ai-dev <description> --no-deploy`        — skip deployment
              `ai-dev status`                           — deployment status
              `ai-dev help`                             — this message

            Pipeline stages:
              1. Plan architecture (AI)
              2. Generate code (AI)
              3. Review for bugs (static + AI)
              4. Auto-fix issues (AI)
              5. Deploy (GitHub + CI/CD)

            All stages use free resources:
              - LLM: OpenRouter free models / Ollama (local)
              - Hosting: GitHub Pages / Fly.io free tier
              - CI/CD: GitHub Actions (free for public repos)
        """)
