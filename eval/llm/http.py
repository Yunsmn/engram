"""HTTP JSON POST with rate limiting, retry and exponential backoff."""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import deque
from typing import Any, Mapping

import httpx

log = logging.getLogger(__name__)

RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
MAX_ATTEMPTS = 8
MAX_ATTEMPTS_RATE_LIMITED = 12  # 429s are transient: be more patient than with other errors
BASE_DELAY_S = 2.0
MAX_DELAY_S = 90.0
TIMEOUT_S = 180.0
WINDOW_S = 60.0


class RateLimiter:
    """Sliding-window limiter on requests/minute and (estimated) tokens/minute. 0 disables a limit."""

    def __init__(self, rpm: int = 0, tpm: int = 0) -> None:
        self.rpm = rpm
        self.tpm = tpm
        self._lock = threading.Lock()
        self._events: deque[tuple[float, int]] = deque()

    def acquire(self, tokens: int = 0) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._events and now - self._events[0][0] >= WINDOW_S:
                    self._events.popleft()
                used_tokens = sum(t for _, t in self._events)
                rpm_ok = not self.rpm or len(self._events) < self.rpm
                tpm_ok = not self.tpm or used_tokens + tokens <= self.tpm or not self._events
                if rpm_ok and tpm_ok:
                    self._events.append((now, tokens))
                    return
                wait = WINDOW_S - (now - self._events[0][0]) + 0.05
            time.sleep(max(wait, 0.05))


class HTTPError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    try:
        return float(raw) if raw is not None else None
    except ValueError:
        return None


def post_json_with_retry(
    url: str,
    body: Mapping[str, Any],
    headers: Mapping[str, str],
    limiter: RateLimiter | None = None,
    est_tokens: int = 0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    owned = client is None
    http = client or httpx.Client(timeout=TIMEOUT_S)
    try:
        for attempt in range(1, MAX_ATTEMPTS_RATE_LIMITED + 1):
            if limiter is not None:
                limiter.acquire(est_tokens)
            try:
                response = http.post(url, json=body, headers=dict(headers))
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= MAX_ATTEMPTS:
                    raise
                delay = _backoff(attempt)
                log.warning("transport error %s; retry %d in %.1fs", exc, attempt, delay)
                time.sleep(delay)
                continue
            if response.status_code < 400:
                return response.json()
            limit = MAX_ATTEMPTS_RATE_LIMITED if response.status_code == 429 else MAX_ATTEMPTS
            if response.status_code not in RETRY_STATUSES or attempt >= limit:
                raise HTTPError(response.status_code, response.text)
            delay = _retry_after(response) or _backoff(attempt)
            log.warning("HTTP %d from %s; retry %d in %.1fs", response.status_code, url, attempt, delay)
            time.sleep(min(delay, MAX_DELAY_S))
        raise RuntimeError("unreachable")
    finally:
        if owned:
            http.close()


def _backoff(attempt: int) -> float:
    return min(MAX_DELAY_S, BASE_DELAY_S * 2 ** (attempt - 1)) * (0.5 + random.random() / 2)
