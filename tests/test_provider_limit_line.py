"""Engine A prints the shared machine-readable limit line once per provider."""
from src import visuals


def test_trip_breaker_prints_dypol_provider_limit_once(capsys):
    visuals.reset_breakers()
    try:
        assert visuals.trip_breaker("flux", "HTTP 429") is True
        assert visuals.trip_breaker("flux", "HTTP 429") is False
        out = capsys.readouterr().out
        assert out.count("DYPOL_PROVIDER_LIMIT provider=flux") == 1
        assert "flux quota/limit hit" in out  # human line kept for older console builds
    finally:
        visuals.reset_breakers()
