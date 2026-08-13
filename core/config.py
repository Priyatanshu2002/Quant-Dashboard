"""Central configuration — loads .env from the project root once."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")


def get(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


# ── LLM (provider-agnostic: Nous Portal or OpenRouter) ─────────────
# Project Agonistes is a standalone app → bearer-token auth against an
# OpenAI-compatible endpoint. Two supported providers:
#   nous      → https://inference-api.nousresearch.com/v1  (NOUS_API_KEY from
#               portal.nousresearch.com → API Keys; serves deepseek-v4-flash-0731
#               plus Hermes-4 models)
#   openrouter→ https://openrouter.ai/api/v1               (OPENROUTER_API_KEY)
# Provider is auto-detected from whichever key is set; override with LLM_PROVIDER.
NOUS_API_KEY = get("NOUS_API_KEY", "")
OPENROUTER_API_KEY = get("OPENROUTER_API_KEY", "")

LLM_PROVIDER = (get("LLM_PROVIDER", "") or
                ("nous" if NOUS_API_KEY else "openrouter" if OPENROUTER_API_KEY else "none"))

if LLM_PROVIDER == "nous":
    LLM_BASE_URL = "https://inference-api.nousresearch.com/v1"
    LLM_API_KEY = NOUS_API_KEY
elif LLM_PROVIDER == "openrouter":
    LLM_BASE_URL = "https://openrouter.ai/api/v1"
    LLM_API_KEY = OPENROUTER_API_KEY
else:
    LLM_BASE_URL = "https://openrouter.ai/api/v1"
    LLM_API_KEY = ""

# Default model: the plan's deepseek-v4-flash-0731 (served by BOTH providers).
# This project uses ONLY deepseek/deepseek-v4-flash-0731.
# `or` so an empty LLM_MODEL env var falls back to the default (same pattern
# as LLM_PROVIDER above).
LLM_MODEL = get("LLM_MODEL") or "deepseek/deepseek-v4-flash-0731"

# ── Storage ─────────────────────────────────────────────────────────
# sqlite:///data/agonistes_dev.db  (dev mode, default)
# postgresql://agonistes:pass@localhost:5432/agonistes  (TimescaleDB)
DATABASE_URL = get("DATABASE_URL", "sqlite:///data/agonistes_dev.db")

# ── External API keys (optional — feeds degrade gracefully) ─────────
FRED_API_KEY = get("FRED_API_KEY", "")
BLS_API_KEY = get("BLS_API_KEY", "")
DUNE_API_KEY = get("DUNE_API_KEY", "")

# ── Runtime ─────────────────────────────────────────────────────────
DEV_MODE = get("DEV_MODE", "true").lower() in ("1", "true", "yes")
LOG_LEVEL = get("LOG_LEVEL", "INFO")

# ── Screener defaults (overridable via screener/screener_config.yaml)
N_CANDIDATES = int(get("N_CANDIDATES", "10"))
MIN_SCORE_THRESHOLD = float(get("MIN_SCORE_THRESHOLD", "60.0"))
MAX_PER_ASSET_CLASS = int(get("MAX_PER_ASSET_CLASS", "3"))
