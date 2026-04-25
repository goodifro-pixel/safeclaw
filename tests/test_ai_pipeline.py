"""Tests for the AI Dev Pipeline actions: ai_coder, bug_finder, deployer, ai_dev."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from safeclaw.actions.ai_coder import AiCoderAction, _parse_code_blocks
from safeclaw.actions.bug_finder import BugFinderAction, _PythonAnalyzer, _find_duplicates
from safeclaw.actions.deployer import DeployerAction, _detect_project


# ---------------------------------------------------------------------------
# _parse_code_blocks
# ---------------------------------------------------------------------------

class TestParseCodeBlocks:
    def test_single_file_with_path(self):
        text = "```app.py\nprint('hello')\n```"
        result = _parse_code_blocks(text)
        assert len(result) == 1
        assert result[0][0] == "app.py"
        assert "print" in result[0][1]

    def test_multiple_files(self):
        text = "```main.py\nprint('a')\n```\n```utils/helpers.py\ndef f(): pass\n```"
        result = _parse_code_blocks(text)
        assert len(result) == 2
        assert result[0][0] == "main.py"
        assert result[1][0] == "utils/helpers.py"

    def test_language_block_without_path(self):
        text = "```python\nprint('hi')\n```"
        result = _parse_code_blocks(text)
        assert len(result) == 1
        assert result[0][0] == "main.py"

    def test_no_fences_returns_raw(self):
        text = "just some code\nno fences"
        result = _parse_code_blocks(text)
        assert len(result) == 1
        assert result[0][0] == "main.py"
        assert "just some code" in result[0][1]


# ---------------------------------------------------------------------------
# _PythonAnalyzer
# ---------------------------------------------------------------------------

class TestPythonAnalyzer:
    def test_syntax_error(self):
        issues = _PythonAnalyzer("def f(\n").run()
        assert any("SyntaxError" in i["message"] for i in issues)

    def test_bare_except(self):
        source = textwrap.dedent("""\
            try:
                x = 1
            except:
                pass
        """)
        issues = _PythonAnalyzer(source).run()
        assert any("Bare except" in i["message"] for i in issues)

    def test_mutable_default(self):
        source = "def f(x=[]):\n    return x\n"
        issues = _PythonAnalyzer(source).run()
        assert any("Mutable default" in i["message"] for i in issues)

    def test_clean_code_no_issues(self):
        source = textwrap.dedent("""\
            import os
            x = os.getcwd()
            print(x)
        """)
        issues = _PythonAnalyzer(source).run()
        # Should have no errors (unused import is possible but OK)
        assert not any(i["severity"] == "ERROR" for i in issues)


# ---------------------------------------------------------------------------
# _detect_project
# ---------------------------------------------------------------------------

class TestDetectProject:
    def test_python_project(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        info = _detect_project(tmp_path)
        assert info["type"] == "python"
        assert "pip" in info["install_cmd"]

    def test_node_project(self, tmp_path: Path):
        (tmp_path / "package.json").write_text("{}")
        info = _detect_project(tmp_path)
        assert info["type"] == "node"

    def test_static_project(self, tmp_path: Path):
        (tmp_path / "index.html").write_text("<html></html>")
        info = _detect_project(tmp_path)
        assert info["type"] == "static"

    def test_unknown_project(self, tmp_path: Path):
        info = _detect_project(tmp_path)
        assert info["type"] == "unknown"


# ---------------------------------------------------------------------------
# _find_duplicates
# ---------------------------------------------------------------------------

class TestFindDuplicates:
    def test_detects_identical_blocks(self, tmp_path: Path):
        block = "\n".join(f"line_{i} = {i}" for i in range(8))
        (tmp_path / "a.py").write_text(block + "\n\nother_code = 1\n")
        (tmp_path / "b.py").write_text("header = True\n" + block + "\n")
        dupes = _find_duplicates(tmp_path, min_lines=6)
        assert len(dupes) >= 1

    def test_no_duplicates(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("x = 1\ny = 2\n")
        (tmp_path / "b.py").write_text("z = 3\nw = 4\n")
        dupes = _find_duplicates(tmp_path, min_lines=6)
        assert len(dupes) == 0


# ---------------------------------------------------------------------------
# BugFinderAction (static analysis subcommand)
# ---------------------------------------------------------------------------

class TestBugFinderAction:
    @pytest.fixture
    def action(self):
        return BugFinderAction()

    def test_help(self, action: BugFinderAction):
        result = action._help()
        assert "Bug Finder" in result

    def test_static_analysis_on_file(self, action: BugFinderAction, tmp_path: Path):
        src = tmp_path / "bad.py"
        src.write_text("try:\n    x = 1\nexcept:\n    pass\n")
        result = action._static_analysis(str(src))
        assert "Bare except" in result

    def test_security_scan(self, action: BugFinderAction, tmp_path: Path):
        src = tmp_path / "insecure.py"
        src.write_text("import os\nos.system('rm -rf /')\n")
        result = action._security_scan(str(src))
        assert "os.system" in result

    def test_security_scan_clean(self, action: BugFinderAction, tmp_path: Path):
        src = tmp_path / "clean.py"
        src.write_text("x = 1\nprint(x)\n")
        result = action._security_scan(str(src))
        assert "No security" in result


# ---------------------------------------------------------------------------
# AiCoderAction
# ---------------------------------------------------------------------------

class TestAiCoderAction:
    @pytest.fixture
    def action(self):
        return AiCoderAction()

    def test_help(self, action: AiCoderAction):
        result = action._help()
        assert "AI Coder" in result

    def test_no_llm_msg(self, action: AiCoderAction):
        result = action._no_llm_msg()
        assert "No LLM" in result
        assert "openrouter" in result.lower() or "ollama" in result.lower()


# ---------------------------------------------------------------------------
# DeployerAction
# ---------------------------------------------------------------------------

class TestDeployerAction:
    @pytest.fixture
    def action(self):
        return DeployerAction()

    def test_help(self, action: DeployerAction):
        result = action._help()
        assert "Auto-Deployer" in result

    def test_generate_ci(self, action: DeployerAction, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        result = action._generate_ci(f"deploy ci {tmp_path}")
        assert "CI/CD Pipeline Generated" in result
        ci_file = tmp_path / ".github" / "workflows" / "ci.yml"
        assert ci_file.exists()

    def test_deploy_pages_creates_workflow(self, action: DeployerAction, tmp_path: Path):
        result = action._deploy_pages(f"deploy pages {tmp_path}")
        assert "GitHub Pages" in result
        pages_file = tmp_path / ".github" / "workflows" / "pages.yml"
        assert pages_file.exists()


# ---------------------------------------------------------------------------
# AiDevAction
# ---------------------------------------------------------------------------

class TestAiDevAction:
    def test_help(self):
        from safeclaw.actions.ai_dev import AiDevAction
        action = AiDevAction()
        result = action._help()
        assert "AI Dev Pipeline" in result
        assert "pipeline" in result.lower()
