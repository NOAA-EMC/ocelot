import logging
import sys
import os

# Get log level from environment variable, default to INFO
LOG_LEVEL = os.getenv('OCELOT_LOG_LEVEL', 'INFO').upper()
level = getattr(logging, LOG_LEVEL, logging.INFO)


def setup_logger(name='ocelot', level=logging.INFO):
    """
    Configures and returns a logger instance.
    """
    # Get the logger and set the level
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding duplicate handlers
    if not logger.handlers:
        # Create a handler to write to standard output
        handler = logging.StreamHandler(sys.stdout)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(handler)

    return logger


logger = setup_logger(level=level)
