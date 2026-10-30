"""Shared test fixtures."""
import pytest


@pytest.fixture(autouse=True)
def _visual_judge_disabled_by_default():
    """Tests never call NVIDIA: each test starts with a disabled process-wide judge
    (as if NVIDIA_API_KEY were unset); judge tests install their own mocked judge."""
    from src import visual_judge
    visual_judge.reset_judge(visual_judge.VisualJudge(env={}))
    yield
    visual_judge.reset_judge(None)
