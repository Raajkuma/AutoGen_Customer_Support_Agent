"""Central configuration. Everything comes from the environment (.env); nothing secret lives in code."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


# --- paths -----------------------------------------------------------------------------------
KB_DIR = Path(os.getenv("KB_DIR", ROOT / "knowledge_base"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "data" / "index"))
DB_FILE = Path(os.getenv("DB_FILE", ROOT / "data" / "shop.db"))
LOG_FILE = Path(os.getenv("LOG_FILE", ROOT / "logs" / "tickets.csv"))  # CSV export of the ticket table
TRACE_DIR = Path(os.getenv("TRACE_DIR", ROOT / "logs" / "traces"))
APP_LOG = Path(os.getenv("APP_LOG", ROOT / "logs" / "app.log"))

# --- local LLM (Docker Model Runner, OpenAI-compatible) ----------------------------------------
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:12434/engines/v1")
MODEL = os.getenv("LLM_MODEL", "ai/qwen3-vl:2B-BF16")
API_KEY = os.getenv("LLM_API_KEY", "docker")  # Docker Model Runner ignores the key
LLM_TIMEOUT = _float("LLM_TIMEOUT", 180)
MAX_TOKENS = _int("LLM_MAX_TOKENS", 600)

# --- retrieval ---------------------------------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "ai/qwen3-embedding")  # multilingual; verify the tag you pulled
EMBED_ENABLED = os.getenv("EMBED_ENABLED", "true").lower() == "true"
CHUNK_SIZE = _int("CHUNK_SIZE", 120)        # words per chunk window
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 20)   # words shared between consecutive windows
RETRIEVAL_K = _int("RETRIEVAL_K", 6)        # candidates kept after rank fusion
RERANK_TOP_N = _int("RERANK_TOP_N", 3)      # passages handed to the Resolution agent
RRF_K = _int("RRF_K", 60)
MIN_MASS_ROUTED = _float("MIN_MASS_ROUTED", 0.5)   # idf mass of shared terms needed inside the routed document
MIN_MASS_OPEN = _float("MIN_MASS_OPEN", 3.0)       # ...and outside it (category unknown / wrong)
MIN_COVERAGE_OPEN = _float("MIN_COVERAGE_OPEN", 0.5)
DENSE_MIN = _float("DENSE_MIN", 0.55)        # embedding-similarity gate (calibrate with the eval script)

# --- app / safety ------------------------------------------------------------------------------
STAFF_PIN = os.getenv("STAFF_PIN", "")
MAX_COMPLAINT_CHARS = _int("MAX_COMPLAINT_CHARS", 1000)
SUBMIT_COOLDOWN_SECONDS = _int("SUBMIT_COOLDOWN_SECONDS", 8)
MAX_TEAM_MESSAGES = _int("MAX_TEAM_MESSAGES", 12)  # safety net on top of the termination words
