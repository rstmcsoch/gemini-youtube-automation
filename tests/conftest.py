"""Shared test fixtures."""
import pytest


@pytest.fixture(autouse=True)
def _no_real_provider_keys(monkeypatch):
    """Hermetic: a real NVIDIA key in the shell must never reach a test (NVIDIA FLUX
    fallback, judge). Tests that need one set a fake key themselves."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("DYPOL_JOB_ID", raising=False)
    monkeypatch.setenv("DYPOL_MUSIC_ENABLED", "0")
    monkeypatch.setenv("DYPOL_NVIDIA_FLUX_MIN_INTERVAL", "0")


@pytest.fixture(autouse=True)
def _visual_judge_disabled_by_default():
    """Tests never call NVIDIA: each test starts with a disabled process-wide judge
    (as if NVIDIA_API_KEY were unset); judge tests install their own mocked judge."""
    from src import visual_judge
    visual_judge.reset_judge(visual_judge.VisualJudge(env={}))
    yield
    visual_judge.reset_judge(None)
