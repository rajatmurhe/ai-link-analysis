"""Simple in-memory cache + rate limiter.

If the same url comes in twice at once only one llm call runs, the other waits for it.
Would need redis if we ever run more than one instance.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Awaitable, Callable, Generic, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    def __init__(self, ttl_seconds: int, max_entries: int) -> None:
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._data: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._inflight: dict[str, asyncio.Future[T]] = {}

    def get(self, key: str) -> T | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires, value = item
        if expires < time.monotonic():
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: T) -> None:
        if self.ttl <= 0:
            return
        self._data[key] = (time.monotonic() + self.ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    async def get_or_compute(self, key: str, compute: Callable[[], Awaitable[T]], *, refresh: bool = False) -> tuple[T, bool]:
        """returns (value, was_cached)"""
        if not refresh:
            cached = self.get(key)
            if cached is not None:
                return cached, True
            pending = self._inflight.get(key)
            if pending is not None:
                return await asyncio.shield(pending), True

        loop = asyncio.get_running_loop()
        future: asyncio.Future[T] = loop.create_future()
        self._inflight[key] = future
        try:
            value = await compute()
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
                future.exception()  # otherwise asyncio warns 'exception never retrieved'
            raise
        else:
            self.set(key, value)
            future.set_result(value)
            return value, False
        finally:
            if self._inflight.get(key) is future:
                self._inflight.pop(key, None)

    def __len__(self) -> int:
        return len(self._data)


class RateLimiter:
    """per ip, per minute. mainly so nobody burns through the llm quota on the demo"""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, tuple[int, int]] = {}

    def allow(self, client_id: str) -> tuple[bool, int]:
        if self.per_minute <= 0:
            return True, 0
        window = int(time.time() // 60)
        win, count = self._hits.get(client_id, (window, 0))
        if win != window:
            win, count = window, 0
        if count >= self.per_minute:
            return False, 60 - int(time.time() % 60)
        self._hits[client_id] = (win, count + 1)
        if len(self._hits) > 10_000:  # don't let this grow forever
            self._hits = {k: v for k, v in self._hits.items() if v[0] == window}
        return True, 0
