import os
import logging


def setup_log2file(path: str):
    # disable logging to console
    logger = logging.getLogger("file")
    logger.addHandler(logging.FileHandler(path, encoding="utf-8"))


format = "%(levelname).4s [%(asctime).19s] %(message)s"

if os.getenv("debug"):
    # cannot set it to debug coz some libraries use it
    logging.basicConfig(level=logging.INFO, format=format)
else:
    logging.basicConfig(level=logging.WARN, format=format)

# Default logger
log2cons = logging.getLogger()
log2file = logging.getLogger("file")
