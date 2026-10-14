"""Small helpers shared by the Newton simulator, its ROS bridge and its sensors."""


def log(logger, level: str, message: str, tag_logger: bool = True) -> None:
    """Send *message* to a ROS-style logger at *level* ("info", "warn"), or print it.

    The printed fallback is always tagged "[LEVEL] "; *tag_logger* adds the same tag
    to the logger line as well.
    """
    tagged = f"[{level.upper()}] {message}"
    if logger is not None:
        getattr(logger, level)(tagged if tag_logger else message)
    else:
        print(tagged)


def step_interval(frequency_hz: float, dt: float) -> int:
    """Physics steps between two events at *frequency_hz*, rounded, at least 1."""
    return max(1, int(round(1.0 / (frequency_hz * dt))))


if __name__ == "__main__":
    assert step_interval(10.0, 0.004) == 25
    assert step_interval(15.0, 0.004) == 17  # 16.67 rounds up
    assert step_interval(1000.0, 0.004) == 1  # faster than the physics never goes below 1

    class _Recorder:
        def info(self, message):
            self.line = message

    recorder = _Recorder()
    log(recorder, "info", "hi")
    assert recorder.line == "[INFO] hi"
    log(recorder, "info", "hi", tag_logger=False)
    assert recorder.line == "hi"
    print("utils self-check OK")
