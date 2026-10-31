"""Workflow shell safety: under `set -euo pipefail` a `$(find|ls ... | ...)` lookup
must not abort the step on a harmless non-zero exit (permission-denied entries,
no match, SIGPIPE from head). Regression test for the 30 Oct 2026 failure of the
ImageMagick policy lookup (run 41070589645)."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WORKFLOWS = sorted((Path(__file__).resolve().parents[1] / ".github" / "workflows").glob("*.yml"))


def _run_blocks(text):
    """(start_line, body) for every `run: |` block."""
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        m = re.match(r"^(\s*)(- )?run: \|\s*$", lines[i])
        if not m:
            i += 1
            continue
        start = i + 1
        body = []
        i += 1
        indent = None
        while i < len(lines):
            line = lines[i]
            if line.strip():
                ind = len(line) - len(line.lstrip())
                if indent is None:
                    indent = ind
                if ind < indent:
                    break
            body.append(line)
            i += 1
        out.append((start, "\n".join(body)))
    return out


def risky_substitutions(body):
    """Command substitutions with a pipeline or find/ls that can exit non-zero,
    without `|| true`, inside a block that runs with errexit + pipefail."""
    if not re.search(r"set -[a-z]*e[a-z]*o pipefail|set -euo pipefail", body):
        return []
    bad = []
    for sub in re.findall(r"\$\(((?:[^()]|\([^()]*\))*)\)", body):
        if not re.search(r"\b(find|ls|grep)\b", sub):
            continue
        if re.search(r"\|\|\s*true\b", sub):
            continue
        bad.append(sub.strip())
    return bad


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_no_spurious_pipefail_lookups(path):
    problems = []
    for start, body in _run_blocks(path.read_text(encoding="utf-8")):
        for sub in risky_substitutions(body):
            problems.append(f"{path.name}:{start}: $({sub})")
    assert not problems, "Unguarded lookups under set -euo pipefail:\n" + "\n".join(problems)


def test_lint_flags_the_old_imagemagick_line():
    old = 'set -euo pipefail\nPOLICY_FILE=$(find /etc -name "policy.xml" 2>/dev/null | grep -i "ImageMagick" | head -n 1)\n'
    assert risky_substitutions(old)


@pytest.mark.skipif(not shutil.which("bash"), reason="bash required")
def test_lookup_survives_unreadable_dirs_and_no_match(tmp_path):
    etc = tmp_path / "etc"
    (etc / "ImageMagick-6").mkdir(parents=True)
    (etc / "ImageMagick-6" / "policy.xml").write_text("<policymap/>")
    locked = etc / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        def run(snippet):
            return subprocess.run(["bash", "-c", "set -euo pipefail\n" + snippet + '\necho "RESULT=[$X]"'],
                                  capture_output=True, text=True)
        old = run(f'X=$(find {etc} -name "policy.xml" 2>/dev/null | grep -i "ImageMagick" | head -n 1)')
        new = run(f'X=$( (find {etc} -name "policy.xml" 2>/dev/null || true) | grep -i "ImageMagick" | head -n 1 || true)')
        none = run(f'X=$( (find {etc} -name "nomatch.xml" 2>/dev/null || true) | grep -i "ImageMagick" | head -n 1 || true)')
        if old.returncode == 0:
            pytest.skip("running as root: unreadable directories are readable")
        assert "RESULT=" not in old.stdout  # the pre-fix line aborts the step
        assert new.returncode == 0 and "ImageMagick-6/policy.xml" in new.stdout
        assert none.returncode == 0 and "RESULT=[]" in none.stdout
    finally:
        locked.chmod(0o755)
