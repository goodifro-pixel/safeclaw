# Testing SafeClaw AI Dev Pipeline

## Overview
The AI Dev Pipeline adds 4 CLI commands: `ai-code`, `bug-find`, `deploy`, `ai-dev`. Testing should be split into **non-LLM tests** (always work) and **LLM tests** (require valid API key).

## Devin Secrets Needed
- `OPENROUTER_API_KEY` — Must be a valid OpenRouter API key (prefix `sk-or-`). Get one at https://openrouter.ai/keys. Do NOT use a HuggingFace token here.
- `HF_TOKEN` — HuggingFace token (optional, for other features)

## Setup
```bash
cd /home/ubuntu/repos/safeclaw
pip install -e ".[dev]"
safeclaw --version  # should show version
safeclaw --help     # verify ai-code, bug-find, deploy, ai-dev are listed
```

## Non-LLM Tests (always free, no API key needed)

### bug-find --static
Create a file with known Python issues (bare except, mutable default args, unused imports, deep nesting) and verify the AST analyzer finds them:
```bash
safeclaw bug-find /path/to/buggy_file.py --static
```
**Expected:** Output contains "Static Analysis" with WARN/INFO severity issues.

### bug-find --security
Create a file with security issues (eval(), os.system(), pickle, hardcoded passwords) and verify regex patterns catch them:
```bash
safeclaw bug-find /path/to/insecure_file.py --security
```
**Expected:** Output contains "Security Scan" with HIGH severity findings.

### deploy ci
Create a temp directory with a `pyproject.toml` to test CI workflow generation:
```bash
safeclaw deploy ci /path/to/project
```
**Expected:** Creates `.github/workflows/ci.yml` with valid YAML, no un-interpolated placeholders like `{python_version}`.

### deploy pages (without --repo)
```bash
safeclaw deploy pages /path/to/static/site
```
**Expected:** Creates `.github/workflows/pages.yml` with `actions/deploy-pages@v4`.

## LLM Tests (require valid OPENROUTER_API_KEY)

### ai-code generate
```bash
safeclaw ai-code "a Python function that checks if a number is prime"
```
**Expected:** Output contains "Code Generated", files appear in `./ai_generated/`.

### ai-code --plan
```bash
safeclaw ai-code "a REST API with user auth" --plan
```
**Expected:** Output contains "Implementation Plan" with provider info. No files written.

### ai-dev (full pipeline)
```bash
safeclaw ai-dev "a calculator app" --no-deploy
```
**Expected:** Runs plan → code → review → fix stages sequentially.

## Known Issues

### deploy pages/github --repo parser bug
The `--repo` flag value overwrites the path argument in `_deploy_pages()` and `_deploy_github()`. The parser loop doesn't skip the token after `--repo`. Workaround: test `deploy pages` without `--repo` flag.

### OpenRouter 401 errors
If you see `HTTP 401: Missing Authentication header`, the `OPENROUTER_API_KEY` is either missing, empty, or contains a non-OpenRouter token. Verify with:
```bash
echo "${OPENROUTER_API_KEY:0:5}"
# Should start with 'sk-or' for OpenRouter
```

## Test Approach
- All testing is CLI/shell-based — no browser recording needed
- Filter noisy INFO logs: `safeclaw <cmd> 2>&1 | grep -v 'INFO\|WARNING\|DEBUG'`
- Run non-LLM tests first to validate core functionality before attempting LLM tests
- The engine loads all plugins on startup (produces many INFO lines) — this is normal
