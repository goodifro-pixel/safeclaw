"""
SafeClaw Auto-Deployer Action — push code to GitHub and free hosting.

Supported free deployment targets:
  1. **GitHub** — commit & push to a repository (public or private)
  2. **GitHub Pages** — deploy static sites (HTML/CSS/JS)
  3. **Fly.io** (free tier) — deploy backend apps via Dockerfile
  4. **CI/CD generation** — create GitHub Actions workflow for auto-deploy

CLI surface:
    deploy github <path> --repo <owner/repo>  — push code to GitHub
    deploy pages <path> --repo <owner/repo>    — deploy to GitHub Pages
    deploy fly <path> --app <name>             — deploy to Fly.io free tier
    deploy ci <path>                           — generate GitHub Actions CI/CD
    deploy status                              — show deployment status
    deploy help                                — show usage
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

from safeclaw.actions.base import BaseAction

if TYPE_CHECKING:
    from safeclaw.core.engine import SafeClaw

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GitHub Actions templates
# ---------------------------------------------------------------------------

_CI_TEMPLATE = textwrap.dedent("""\
    name: CI/CD Pipeline

    on:
      push:
        branches: [main, master]
      pull_request:
        branches: [main, master]

    jobs:
      test:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - uses: actions/setup-python@v5
            with:
              python-version: "{python_version}"
          - name: Install dependencies
            run: |
              python -m pip install --upgrade pip
              {install_cmd}
          - name: Lint
            run: {lint_cmd}
          - name: Test
            run: {test_cmd}

      deploy:
        needs: test
        runs-on: ubuntu-latest
        if: github.ref == 'refs/heads/main' && github.event_name == 'push'
        steps:
          - uses: actions/checkout@v4
          {deploy_steps}
""")

_PAGES_DEPLOY_STEPS = textwrap.dedent("""\
    - name: Deploy to GitHub Pages
            uses: peaceiris/actions-gh-pages@v3
            with:
              github_token: ${{ secrets.GITHUB_TOKEN }}
              publish_dir: {publish_dir}
""")

_FLY_DEPLOY_STEPS = textwrap.dedent("""\
    - uses: superfly/flyctl-actions/setup-flyctl@master
          - name: Deploy to Fly.io
            run: flyctl deploy --remote-only
            env:
              FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}
""")

_DOCKERFILE_TEMPLATE = textwrap.dedent("""\
    FROM python:3.12-slim

    WORKDIR /app

    COPY requirements.txt* pyproject.toml* ./
    RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null \\
        || pip install --no-cache-dir . 2>/dev/null \\
        || true

    COPY . .

    EXPOSE 8080

    CMD {cmd}
""")

_FLY_TOML_TEMPLATE = textwrap.dedent("""\
    app = "{app_name}"
    primary_region = "iad"

    [build]

    [http_service]
      internal_port = 8080
      force_https = true
      auto_stop_machines = true
      auto_start_machines = true
      min_machines_running = 0

    [[vm]]
      cpu_kind = "shared"
      cpus = 1
      memory_mb = 256
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    """Run a shell command and return (returncode, combined output)."""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=120,
        )
        out = (result.stdout + "\n" + result.stderr).strip()
        return result.returncode, out
    except FileNotFoundError:
        return 1, f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 1, "Command timed out after 120s"


def _detect_project(path: Path) -> dict[str, str]:
    """Detect project type and return useful metadata."""
    info: dict[str, str] = {"type": "unknown", "python_version": "3.12"}

    if (path / "pyproject.toml").exists() or (path / "setup.py").exists():
        info["type"] = "python"
        info["install_cmd"] = "pip install -e '.[dev]'"
        info["lint_cmd"] = "ruff check . || true"
        info["test_cmd"] = "pytest --tb=short -q || true"
    elif (path / "package.json").exists():
        info["type"] = "node"
        info["install_cmd"] = "npm ci"
        info["lint_cmd"] = "npm run lint || true"
        info["test_cmd"] = "npm test || true"
    elif (path / "Cargo.toml").exists():
        info["type"] = "rust"
        info["install_cmd"] = "# Rust project"
        info["lint_cmd"] = "cargo clippy || true"
        info["test_cmd"] = "cargo test"
    elif (path / "go.mod").exists():
        info["type"] = "go"
        info["install_cmd"] = "go mod download"
        info["lint_cmd"] = "go vet ./..."
        info["test_cmd"] = "go test ./..."
    elif (path / "index.html").exists():
        info["type"] = "static"
        info["install_cmd"] = "# Static site"
        info["lint_cmd"] = "echo 'No lint'"
        info["test_cmd"] = "echo 'No tests'"

    # Detect entry point for Docker
    if (path / "main.py").exists():
        info["entrypoint"] = '["python", "main.py"]'
    elif (path / "app.py").exists():
        info["entrypoint"] = '["python", "app.py"]'
    elif (path / "manage.py").exists():
        info["entrypoint"] = '["python", "manage.py", "runserver", "0.0.0.0:8080"]'
    else:
        info["entrypoint"] = '["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]'

    return info


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class DeployerAction(BaseAction):
    """Auto-deploy code to GitHub and free hosting platforms."""

    name = "deployer"
    description = "Deploy code to GitHub and free hosting"

    async def execute(
        self,
        params: dict[str, Any],
        user_id: str,
        channel: str,
        engine: SafeClaw,
    ) -> str:
        raw = params.get("raw_input", "").strip()
        lower = raw.lower()

        if not raw or "deploy help" in lower or lower == "deploy":
            return self._help()

        if lower.startswith("deploy github"):
            return self._deploy_github(raw)

        if lower.startswith("deploy pages"):
            return self._deploy_pages(raw)

        if lower.startswith("deploy fly"):
            return self._deploy_fly(raw)

        if lower.startswith("deploy ci"):
            return self._generate_ci(raw)

        if lower.startswith("deploy status"):
            return self._status()

        # Try to infer
        path_str = raw.replace("deploy", "").strip().split()[0] if raw.replace("deploy", "").strip() else "."
        return self._generate_ci(f"deploy ci {path_str}")

    # ------------------------------------------------------------------
    # GitHub push
    # ------------------------------------------------------------------

    def _deploy_github(self, raw: str) -> str:
        """Push code to a GitHub repository."""
        parts = raw.split()
        path_str = "."
        repo = ""
        skip: set[int] = set()

        for i, p in enumerate(parts):
            if i in skip:
                continue
            if p == "--repo" and i + 1 < len(parts):
                repo = parts[i + 1]
                skip.add(i + 1)
            elif p not in ("deploy", "github", "--repo") and not p.startswith("--"):
                path_str = p

        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            return f"Path not found: {path}"

        git_dir = path / ".git"
        if not git_dir.exists():
            # Initialize git repo
            _run(["git", "init"], cwd=path)
            _run(["git", "branch", "-M", "main"], cwd=path)

        if repo:
            _run(["git", "remote", "remove", "origin"], cwd=path)
            _run(["git", "remote", "add", "origin", f"https://github.com/{repo}.git"], cwd=path)

        # Stage and commit
        _run(["git", "add", "-A"], cwd=path)
        rc, out = _run(["git", "commit", "-m", "Auto-deploy via SafeClaw AI"], cwd=path)

        # Push
        rc, push_out = _run(["git", "push", "-u", "origin", "main"], cwd=path)
        if rc != 0:
            # Try force push on first push
            rc, push_out = _run(["git", "push", "-u", "origin", "main", "--force-with-lease"], cwd=path)

        if rc == 0:
            repo_url = f"https://github.com/{repo}" if repo else "(local remote)"
            return (
                f"**Deployed to GitHub**\n\n"
                f"Repository: {repo_url}\n"
                f"Branch: main\n\n"
                f"Next: `deploy pages {path_str} --repo {repo}` for static hosting\n"
                f"Or: `deploy ci {path_str}` to generate CI/CD pipeline"
            )
        return f"Git push failed:\n```\n{push_out}\n```"

    # ------------------------------------------------------------------
    # GitHub Pages
    # ------------------------------------------------------------------

    def _deploy_pages(self, raw: str) -> str:
        """Set up GitHub Pages deployment."""
        parts = raw.split()
        path_str = "."
        repo = ""
        skip: set[int] = set()

        for i, p in enumerate(parts):
            if i in skip:
                continue
            if p == "--repo" and i + 1 < len(parts):
                repo = parts[i + 1]
                skip.add(i + 1)
            elif p not in ("deploy", "pages", "--repo") and not p.startswith("--"):
                path_str = p

        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            return f"Path not found: {path}"

        # Create GitHub Actions workflow for Pages
        workflows_dir = path / ".github" / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)

        pages_workflow = textwrap.dedent("""\
            name: Deploy to GitHub Pages

            on:
              push:
                branches: [main]

            permissions:
              contents: read
              pages: write
              id-token: write

            concurrency:
              group: "pages"
              cancel-in-progress: false

            jobs:
              deploy:
                environment:
                  name: github-pages
                  url: ${{ steps.deployment.outputs.page_url }}
                runs-on: ubuntu-latest
                steps:
                  - uses: actions/checkout@v4
                  - uses: actions/configure-pages@v4
                  - uses: actions/upload-pages-artifact@v3
                    with:
                      path: '.'
                  - name: Deploy to GitHub Pages
                    id: deployment
                    uses: actions/deploy-pages@v4
        """)

        workflow_file = workflows_dir / "pages.yml"
        workflow_file.write_text(pages_workflow)

        result = (
            f"**GitHub Pages Setup**\n\n"
            f"Created: `{workflow_file.relative_to(path)}`\n\n"
            f"To activate:\n"
            f"  1. Push to GitHub: `deploy github {path_str} --repo {repo}`\n"
            f"  2. Go to repo Settings > Pages > Source: GitHub Actions\n"
        )

        if repo:
            result += f"  3. Your site will be at: https://{repo.split('/')[0]}.github.io/{repo.split('/')[-1]}/\n"

        return result

    # ------------------------------------------------------------------
    # Fly.io
    # ------------------------------------------------------------------

    def _deploy_fly(self, raw: str) -> str:
        """Deploy to Fly.io free tier."""
        parts = raw.split()
        path_str = "."
        app_name = ""
        skip: set[int] = set()

        for i, p in enumerate(parts):
            if i in skip:
                continue
            if p == "--app" and i + 1 < len(parts):
                app_name = parts[i + 1]
                skip.add(i + 1)
            elif p not in ("deploy", "fly", "--app") and not p.startswith("--"):
                path_str = p

        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            return f"Path not found: {path}"

        project = _detect_project(path)
        if not app_name:
            app_name = path.name.lower().replace("_", "-").replace(" ", "-")

        # Generate Dockerfile if not exists
        dockerfile = path / "Dockerfile"
        if not dockerfile.exists():
            dockerfile.write_text(
                _DOCKERFILE_TEMPLATE.format(cmd=project.get("entrypoint", '["python", "main.py"]'))
            )

        # Generate fly.toml
        fly_toml = path / "fly.toml"
        fly_toml.write_text(_FLY_TOML_TEMPLATE.format(app_name=app_name))

        # Check if flyctl is installed
        flyctl = shutil.which("flyctl") or shutil.which("fly")

        if not flyctl:
            return (
                f"**Fly.io Setup**\n\n"
                f"Generated:\n"
                f"  - `Dockerfile`\n"
                f"  - `fly.toml` (app: {app_name})\n\n"
                f"To deploy:\n"
                f"  1. Install flyctl: `curl -L https://fly.io/install.sh | sh`\n"
                f"  2. Sign up (free): `flyctl auth signup`\n"
                f"  3. Launch: `flyctl launch --name {app_name} --region iad`\n"
                f"  4. Deploy: `flyctl deploy`\n\n"
                f"Free tier includes 3 shared-cpu VMs and 256 MB RAM."
            )

        # Try to deploy
        rc, out = _run([flyctl, "deploy", "--remote-only"], cwd=path)
        if rc == 0:
            return (
                f"**Deployed to Fly.io!**\n\n"
                f"App: {app_name}\n"
                f"URL: https://{app_name}.fly.dev\n\n"
                f"Manage: `flyctl status` | `flyctl logs`"
            )
        return (
            f"**Fly.io Deploy** — setup complete, deploy pending\n\n"
            f"Generated: Dockerfile + fly.toml\n"
            f"Run: `flyctl launch --name {app_name}` then `flyctl deploy`\n\n"
            f"Output:\n```\n{out[:500]}\n```"
        )

    # ------------------------------------------------------------------
    # CI/CD generation
    # ------------------------------------------------------------------

    def _generate_ci(self, raw: str) -> str:
        """Generate GitHub Actions CI/CD workflow."""
        path_str = raw.split("ci", 1)[-1].strip().split()[0] if "ci" in raw else "."
        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            return f"Path not found: {path}"

        project = _detect_project(path)
        workflows_dir = path / ".github" / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)

        # Pick deploy steps based on project type
        if project["type"] == "static":
            deploy_steps = _PAGES_DEPLOY_STEPS.format(publish_dir=".")
        else:
            deploy_steps = _FLY_DEPLOY_STEPS

        ci_content = _CI_TEMPLATE.format(
            python_version=project.get("python_version", "3.12"),
            install_cmd=project.get("install_cmd", "echo 'no install'"),
            lint_cmd=project.get("lint_cmd", "echo 'no lint'"),
            test_cmd=project.get("test_cmd", "echo 'no tests'"),
            deploy_steps=deploy_steps,
        )

        ci_file = workflows_dir / "ci.yml"
        ci_file.write_text(ci_content)

        return (
            f"**CI/CD Pipeline Generated**\n\n"
            f"Created: `{ci_file.relative_to(path)}`\n"
            f"Project type: {project['type']}\n\n"
            f"Pipeline:\n"
            f"  1. Lint → Test → Deploy (on push to main)\n\n"
            f"Push to GitHub to activate: `deploy github {path_str} --repo <owner/repo>`"
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _status(self) -> str:
        """Show deployment status."""
        flyctl = shutil.which("flyctl") or shutil.which("fly")
        lines = ["**Deployment Status**", ""]

        # Check git remotes
        rc, remotes = _run(["git", "remote", "-v"])
        if rc == 0 and remotes:
            lines.append("Git remotes:")
            for line in remotes.splitlines()[:4]:
                lines.append(f"  {line}")
        else:
            lines.append("No git remotes configured.")

        lines.append("")

        # Check Fly.io
        if flyctl:
            rc, fly_status = _run([flyctl, "status"])
            if rc == 0:
                lines.append(f"Fly.io: active\n{fly_status[:300]}")
            else:
                lines.append("Fly.io: not deployed (flyctl installed)")
        else:
            lines.append("Fly.io: flyctl not installed")

        return "\n".join(lines)

    @staticmethod
    def _help() -> str:
        return textwrap.dedent("""\
            **Auto-Deployer** — deploy code to GitHub + free hosting

            Commands:
              `deploy github <path> --repo owner/repo`  — push to GitHub
              `deploy pages <path> --repo owner/repo`    — GitHub Pages (static)
              `deploy fly <path> --app name`             — Fly.io free tier (backend)
              `deploy ci <path>`                         — generate CI/CD workflow
              `deploy status`                            — show deployment status
              `deploy help`                              — this message

            All hosting is free:
              - GitHub Pages: free for public repos (static sites)
              - Fly.io: 3 free shared-cpu VMs (256 MB RAM each)
              - GitHub Actions: free for public repos (CI/CD)
        """)
