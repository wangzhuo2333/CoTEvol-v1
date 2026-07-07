import os
import sys
from loguru import logger


def formatter(record):
    """
    Custom log format based on the log record's attributes.

    Args:
        record: Log record object containing log attributes.

    Returns:
        str: Formatted log string.

    """
    # default format
    time_format = "<green>{time:MM-DD/HH:mm:ss}</>"
    lvl_format = "<lvl><i>{level:^5}</></>"
    rcd_format = "<cyan>{file}:{line:}</>"
    msg_format = "<lvl>{message}</>"

    if record["level"].name in ["WARNING", "CRITICAL"]:
        lvl_format = "<l>" + lvl_format + "</>"

    return "|".join([time_format, lvl_format, rcd_format, msg_format]) + "\n"


def setup_logger(main_process, save_dir):
    """
    Configures and initializes the logger based on the provided training arguments.
    Returns:
        logger: Configured and initialized logger object.
    """
    logger.remove()
    if main_process:
        logger.add(
            sys.stderr, format=formatter,
            colorize=True, enqueue=False
        )
    else:
        logger.add(
            os.path.join(save_dir, "log.txt"), format=formatter,
            colorize=True, enqueue=False
        )

    # registry.register("logger", logger)
    return logger
