"""Asset universe — all tradable instruments across asset classes.

Loaded from screener_config.yaml; each entry is a symbol + asset class.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.logging import get_logger

log = get_logger(__name__)

CONFIG_PATH = Path(__file__).parent / "screener_config.yaml"


@dataclass
class UniverseEntry:
    symbol: str
    asset_class: str
    name: str = ""
    exchange: str = ""
    sector: str = ""


class AssetUniverse:
    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or CONFIG_PATH
        self.entries: list[UniverseEntry] = []
        self.weights: dict[str, dict[str, float]] = {}
        self.settings: dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        with open(self.config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        self.weights = cfg.get("signal_weights", {})
        self.settings = cfg.get("settings", {})
        self.entries = []
        for asset_class, symbols in cfg.get("universe", {}).items():
            for item in symbols:
                if isinstance(item, str):
                    self.entries.append(UniverseEntry(symbol=item, asset_class=asset_class))
                else:
                    self.entries.append(UniverseEntry(asset_class=asset_class, **item))
        log.info("Universe loaded: %d instruments across %d classes",
                 len(self.entries), len(cfg.get("universe", {})))

    def symbols(self, asset_class: str | None = None) -> list[str]:
        return [e.symbol for e in self.entries
                if asset_class is None or e.asset_class == asset_class]

    def asset_class_of(self, symbol: str) -> str | None:
        for e in self.entries:
            if e.symbol == symbol:
                return e.asset_class
        return None


_universe: AssetUniverse | None = None
_universe_mtime: float | None = None


def get_universe() -> AssetUniverse:
    """Return the universe, reloading if the config file changed on disk.

    This makes adding/removing tickers in screener_config.yaml take effect
    WITHOUT a server restart — the next call picks up the change. (The market
    screener payload is separately TTL-cached for 15 min, so edits appear
    within that window, not on the next request.)
    """
    global _universe, _universe_mtime
    try:
        mtime = CONFIG_PATH.stat().st_mtime
    except OSError:
        mtime = None
    if _universe is None or (mtime is not None and mtime != _universe_mtime):
        _universe = AssetUniverse()
        _universe_mtime = mtime
    return _universe
