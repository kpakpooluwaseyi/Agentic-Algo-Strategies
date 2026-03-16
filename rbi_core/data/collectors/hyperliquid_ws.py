"""rbi_core/data/collectors/hyperliquid_ws.py — Hyperliquid WebSocket tick feed.

Subscribes to trades and l2Book channels for configured symbols.
Merges trade price/volume with best bid/ask from orderbook into unified tick dicts.
Auto-reconnects on disconnect. Thread-safe callback to TickBuffer.
"""
import json
import threading
import time
from collections import defaultdict
from typing import Callable, Optional

import ssl
import websocket  # websocket-client
import certifi


def with_retry(max_retries: int = 3, backoff: float = 1.5):
    """Exponential backoff retry decorator for network calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(backoff ** attempt)
            return None
        return wrapper
    return decorator


class HyperliquidWSFeed:
    """
    Connects to Hyperliquid WebSocket and pushes ticks to a TickBuffer.

    Subscribes to:
    - trades: Real-time executed trades (price, volume, side, timestamp)
    - l2Book: Level 2 orderbook snapshots (best bid, best ask)

    Each trade message produces a tick dict:
        {
            'timestamp': float,     # unix epoch seconds
            'symbol': str,          # e.g., "BTC"
            'price': float,         # trade price
            'volume': float,        # trade volume
            'bid': float,           # best bid from last l2Book update
            'ask': float,           # best ask from last l2Book update
            'atr': float,           # placeholder 0.0 (computed downstream)
            'raw_json': str         # raw trade message JSON
        }
    """

    WS_URL = "wss://api.hyperliquid.xyz/ws"
    PING_INTERVAL = 50  # seconds — HLP requires ping within 60s

    def __init__(self, symbols: list[str], on_tick: Callable[[dict], None],
                 on_book: Optional[Callable[[dict], None]] = None):
        """
        Args:
            symbols: List of symbols to subscribe (e.g., ["BTC", "ETH"])
            on_tick: Callback invoked with each tick dict. Typically TickBuffer.append_tick.
            on_book: Optional callback for raw l2Book data (for microstructure engine).
        """
        self.symbols = symbols
        self.on_tick = on_tick
        self.on_book = on_book
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._ws: Optional[websocket.WebSocketApp] = None

        # Latest bid/ask per symbol from l2Book updates
        self._book_lock = threading.Lock()
        self._best_bid: dict[str, float] = defaultdict(float)
        self._best_ask: dict[str, float] = defaultdict(float)

    def start(self) -> None:
        """Start WebSocket listener in background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="hlp_ws")
        self._thread.start()
        print(f"[HLP_WS] Started for symbols: {self.symbols}")

    def stop(self) -> None:
        """Stop WebSocket and background thread."""
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=10.0)
        print("[HLP_WS] Stopped")

    def _run_loop(self) -> None:
        """Outer reconnection loop. Retries indefinitely while self._running."""
        while self._running:
            try:
                self._connect_and_listen()
            except Exception as e:
                if self._running:
                    print(f"[HLP_WS] Connection error: {e}. Reconnecting in 3s...")
                    time.sleep(3.0)

    def _connect_and_listen(self) -> None:
        """Establish WebSocket connection and block until disconnected."""
        self._ws = websocket.WebSocketApp(
            self.WS_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        # run_forever blocks. ping_interval keeps connection alive.
        # sslopt uses certifi CA bundle to fix macOS SSL cert verification.
        self._ws.run_forever(
            ping_interval=self.PING_INTERVAL,
            ping_timeout=10,
            sslopt={"ca_certs": certifi.where()},
        )

    def _on_open(self, ws) -> None:
        """Subscribe to trades and l2Book for each symbol."""
        print(f"[HLP_WS] Connected to {self.WS_URL}")
        for symbol in self.symbols:
            # Subscribe to trades
            ws.send(json.dumps({
                "method": "subscribe",
                "subscription": {"type": "trades", "coin": symbol}
            }))
            # Subscribe to l2Book (top-of-book updates)
            ws.send(json.dumps({
                "method": "subscribe",
                "subscription": {"type": "l2Book", "coin": symbol}
            }))
            print(f"[HLP_WS] Subscribed to trades + l2Book for {symbol}")

    def _on_message(self, ws, message: str) -> None:
        """Route incoming messages to trades or l2Book handlers."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        channel = data.get("channel")
        msg_data = data.get("data")
        if not channel or msg_data is None:
            return

        if channel == "trades":
            self._handle_trades(msg_data, message)
        elif channel == "l2Book":
            self._handle_l2book(msg_data)

    def _handle_trades(self, trades_data: list, raw_msg: str) -> None:
        """
        Process trade messages. Each trade becomes a tick.
        HLP trades format: [{"coin": "BTC", "side": "B", "px": "95432.1",
                             "sz": "0.01", "time": 1700000000000, "hash": "..."}]
        """
        if not isinstance(trades_data, list):
            return

        for trade in trades_data:
            symbol = trade.get("coin", "")
            try:
                price = float(trade.get("px", 0))
                volume = float(trade.get("sz", 0))
                ts_ms = trade.get("time", 0)
                timestamp = ts_ms / 1000.0 if ts_ms > 1e12 else float(ts_ms)
            except (ValueError, TypeError):
                continue

            # Get latest bid/ask for this symbol
            with self._book_lock:
                bid = self._best_bid.get(symbol, 0.0)
                ask = self._best_ask.get(symbol, 0.0)

            tick = {
                'timestamp': timestamp,
                'symbol': symbol,
                'price': price,
                'volume': volume,
                'bid': bid,
                'ask': ask,
                'atr': 0.0,  # Computed downstream by strategy/combiner
                'raw_json': json.dumps(trade),
            }
            self.on_tick(tick)

    def _handle_l2book(self, book_data: dict) -> None:
        """
        Update best bid/ask from l2Book snapshots.
        HLP l2Book format: {"coin": "BTC", "levels": [
            [{"px": "95430.0", "sz": "1.2", "n": 3}, ...],  # bids
            [{"px": "95432.0", "sz": "0.8", "n": 2}, ...]   # asks
        ]}
        """
        coin = book_data.get("coin", "")
        levels = book_data.get("levels", [])

        if len(levels) < 2:
            return

        bids = levels[0]
        asks = levels[1]

        try:
            best_bid = float(bids[0]["px"]) if bids else 0.0
            best_ask = float(asks[0]["px"]) if asks else 0.0
        except (IndexError, KeyError, ValueError, TypeError):
            return

        with self._book_lock:
            self._best_bid[coin] = best_bid
            self._best_ask[coin] = best_ask

        # Fire book callback for microstructure engine
        if self.on_book:
            self.on_book(book_data)

    def _on_error(self, ws, error) -> None:
        print(f"[HLP_WS] Error: {error}")

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        print(f"[HLP_WS] Closed: code={close_status_code} msg={close_msg}")

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._ws.sock is not None and self._ws.sock.connected
