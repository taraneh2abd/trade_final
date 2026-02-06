import logging
from typing import Optional


def get_logger(name: str = "project_ci", level: int = logging.INFO, logfile: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # جلوگیری از تکرار handler

    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if logfile:
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
