"""Tests for the AI Dev Pipeline actions: ai_coder, bug_finder, deployer, ai_dev."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from safeclaw.actions.ai_coder import (
    AiCoderAction,
    _collect_source_files,
    _parse_code_blocks,
)
from safeclaw.actions.bug_finder import BugFinderAction, _find_duplicates, _PythonAnalyzer
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
        assert "professional" in result.lower() or "pipeline" in result.lower()

    def test_help_mentions_review(self, action: AiCoderAction):
        result = action._help()
        assert "review" in result.lower()

    def test_help_mentions_test(self, action: AiCoderAction):
        result = action._help()
        assert "test" in result.lower()

    def test_no_llm_msg(self, action: AiCoderAction):
        result = action._no_llm_msg()
        assert "No LLM" in result
        assert "openrouter" in result.lower() or "ollama" in result.lower()

    def test_generate_readme(self, action: AiCoderAction):
        readme = action._generate_readme("A test project", ["main.py", "tests/test_main.py"])
        assert "test project" in readme.lower()
        assert "main.py" in readme
        assert "pytest" in readme.lower()


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

class TestCollectSourceFiles:
    def test_single_file(self, tmp_path: Path):
        src = tmp_path / "app.py"
        src.write_text("print('hello')\n")
        result = _collect_source_files(src)
        assert "app.py" in result
        assert "hello" in result

    def test_directory(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("x = 1\n")
        (tmp_path / "utils.py").write_text("y = 2\n")
        result = _collect_source_files(tmp_path)
        assert "main.py" in result
        assert "utils.py" in result

    def test_skips_non_source_files(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("x = 1\n")
        (tmp_path / "readme.txt").write_text("hello\n")
        result = _collect_source_files(tmp_path)
        assert "main.py" in result
        assert "readme.txt" not in result

    def test_empty_directory(self, tmp_path: Path):
        result = _collect_source_files(tmp_path)
        assert "no source files" in result.lower()


class TestScaffoldTemplates:
    def test_gitignore_template(self):
        from safeclaw.actions.ai_coder import _SCAFFOLD_TEMPLATES
        assert ".gitignore" in _SCAFFOLD_TEMPLATES
        gitignore = _SCAFFOLD_TEMPLATES[".gitignore"]
        assert "__pycache__" in gitignore
        assert ".env" in gitignore
        assert ".pytest_cache" in gitignore


class TestAiDevAction:
    def test_help(self):
        from safeclaw.actions.ai_dev import AiDevAction
        action = AiDevAction()
        result = action._help()
        assert "AI Dev Pipeline" in result
        assert "pipeline" in result.lower()

    def test_help_mentions_quality(self):
        from safeclaw.actions.ai_dev import AiDevAction
        action = AiDevAction()
        result = action._help()
        assert "quality" in result.lower()

    def test_help_mentions_threshold(self):
        from safeclaw.actions.ai_dev import AiDevAction
        action = AiDevAction()
        result = action._help()
        assert "threshold" in result.lower() or "6" in result

    def test_extract_quality_score(self):
        from safeclaw.actions.ai_dev import AiDevAction
        action = AiDevAction()
        assert action._extract_quality_score("Quality Score: 7/10") == 7
        assert action._extract_quality_score("Quality: **8/10**") == 8
        assert action._extract_quality_score("no score here") == 0
        assert action._extract_quality_score("**3/10**") == 3

    def test_extract_quality_score_edge_cases(self):
        from safeclaw.actions.ai_dev import AiDevAction
        action = AiDevAction()
        assert action._extract_quality_score("") == 0
        assert action._extract_quality_score("Quality Score: 10/10") == 10
        assert action._extract_quality_score("Quality: **1/10**") == 1
