"""Logs estruturados em JSON, um registro por evento."""
import json
import logging
import sys
from datetime import datetime, timezone

from src.config import LOGS_DIR

_run_id: str | None = None


def set_run_id(run_id: str | None) -> None:
    """Passa a anexar `run_id` a todo log emitido, para correlacionar uma execução."""
    global _run_id
    _run_id = run_id


_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        if _run_id:
            payload["run_id"] = _run_id
        payload.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        return json.dumps(payload, ensure_ascii=False, default=str)


def get_logger(name: str = "pipeline") -> logging.Logger:
    """Logger com saída em stdout e em logs/pipeline.jsonl (sem dados pessoais)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    LOGS_DIR.mkdir(exist_ok=True)
    fmt = JsonFormatter()
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(LOGS_DIR / "pipeline.jsonl", encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.propagate = False
    return logger
