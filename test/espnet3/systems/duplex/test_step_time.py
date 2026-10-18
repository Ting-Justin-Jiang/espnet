"""Tests for the frame-time metric of the Moshi system."""

from espnet3.systems.duplex.metrics.step_time import StepTime


def test_mean_frame_time_and_real_time_factor(tmp_path):
    """Each conversation's mean frame time is averaged, then divided by 80 ms."""
    scp = tmp_path / "step_ms.scp"
    scp.write_text("a 20.0\nb 40.0\nc 60.0\n")
    result = StepTime()({"step_ms": scp}, "test")
    assert result == {"step_ms": 40.0, "real_time_factor": 0.5}
