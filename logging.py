import logging as _logging

_logging.basicConfig(
    level=_logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def get_logger(name: str) -> _logging.Logger:
    """Return a logger with the given name."""
    return _logging.getLogger(name)
