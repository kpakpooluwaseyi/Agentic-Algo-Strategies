"""rbi_core/ai/swarm_sync.py — Async pull-based weight sync from Dell Nanobot."""
import json
import threading
import time
from typing import Callable, Optional
import urllib.request
import urllib.error

# Use requests under the hood for retry logic if available, else fall back to urllib
try:
    import requests as _req
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry as _Retry
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

class SwarmWeightSync:
    """
    Periodically polls Dell Nanobot for updated RL parameter bounds.
    Uses epoch-versioned polling: only fetches if Dell has a newer epoch.
    Gracefully degrades on failure (sets swarm_degraded flag on RL agent).
    """

    def __init__(
        self,
        nanobot_url: str,           # e.g., "http://192.168.1.50:8080"
        on_weights_updated: Callable[[dict], None],  # CorrectiveRLAgent.load_weights_from_swarm
        on_degraded: Callable[[bool], None],          # sets agent.swarm_degraded
        poll_interval_s: float = 60.0,
        staleness_threshold_s: float = 600.0,  # 10 min — reduced false-positives over Tailscale
    ):
        self.nanobot_url = nanobot_url.rstrip('/')
        self.on_weights_updated = on_weights_updated
        self.on_degraded = on_degraded
        self.poll_interval_s = poll_interval_s
        self.staleness_threshold_s = staleness_threshold_s
        self._current_epoch: int = 0
        self._last_success_ts: float = time.time()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        # Build a persistent session with retry/backoff (requests preferred)
        if _HAS_REQUESTS:
            retry = _Retry(
                total=3,
                backoff_factor=1.0,
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=["GET"],
                raise_on_status=False,
            )
            self._session = _req.Session()
            self._session.mount("http://",  HTTPAdapter(max_retries=retry))
            self._session.mount("https://", HTTPAdapter(max_retries=retry))
        else:
            self._session = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._poll_once()
            except Exception as e:
                print(f"[SwarmSync] Poll error: {e}")
            self._check_staleness()
            time.sleep(self.poll_interval_s)

    def _poll_once(self) -> None:
        url = f"{self.nanobot_url}/weights/latest?since_epoch={self._current_epoch}"
        try:
            if self._session is not None:
                resp = self._session.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
            else:
                # Fallback: raw urllib
                req = urllib.request.Request(url, method='GET')
                req.add_header('Accept', 'application/json')
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"[SwarmSync] Network error contacting {url}: {e}")
            return

        new_epoch = data.get('epoch', 0)
        if new_epoch > self._current_epoch:
            weights = data.get('params', {})
            self.on_weights_updated(weights)
            self._current_epoch = new_epoch
            self._last_success_ts = time.time()
            self.on_degraded(False)  # Clear degraded flag
            print(f"[SwarmSync] Updated to epoch {new_epoch}")

    def _check_staleness(self) -> None:
        elapsed = time.time() - self._last_success_ts
        if elapsed > self.staleness_threshold_s:
            self.on_degraded(True)
            print(f"[SwarmSync] WARNING: Stale for {elapsed:.0f}s. Swarm degraded.")
