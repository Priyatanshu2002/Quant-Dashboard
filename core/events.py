"""Lightweight async event bus connecting data feeds to downstream layers.

Feed writers publish normalized events (price bar, filing, sentiment hit);
the screener / feature pipeline / debate layer subscribe and react.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Awaitable, Callable

Handler = Callable[[str, dict], Awaitable[None] | None]

EVENT_PRICE_BAR = "price_bar"
EVENT_FUNDAMENTAL = "fundamental_snapshot"
EVENT_SENTIMENT = "sentiment_event"
EVENT_MACRO = "macro_snapshot"
EVENT_ONCHAIN = "onchain_snapshot"


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[Handler]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Handler) -> None:
        self._subscribers[event_type].append(handler)

    async def publish(self, event_type: str, payload: dict) -> None:
        for handler in list(self._subscribers.get(event_type, [])):
            result = handler(event_type, payload)
            if asyncio.iscoroutine(result):
                await result

    def subscriber_count(self, event_type: str) -> int:
        return len(self._subscribers.get(event_type, []))


# Process-wide singleton so feeds and consumers share one bus.
bus = EventBus()
