"""rbi_core/swarm_integration/picoclaw_ingest.py — HTTP listener for PicoClaw regime scores."""
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Callable, Optional


class RegimeHandler(BaseHTTPRequestHandler):
    """Handles POST /regime with JSON body: {"score": 0-100, "label": "trending|ranging|volatile"}"""

    callback: Optional[Callable] = None  # Set by PicoClawIngest

    def do_POST(self):
        if self.path != '/regime':
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
            score = float(data.get('score', 50))
            # Normalize: score 0-100 -> weight 0.0-2.0
            # 50 = neutral (1.0), 100 = max aggression (2.0), 0 = full defense (0.0)
            weight = score / 50.0
            if self.callback:
                self.callback(weight)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        except (json.JSONDecodeError, ValueError) as e:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(f'{{"error":"{e}"}}'.encode())

    def do_GET(self):
        """Health-check endpoint used by Dell PicoClaw monitor agent."""
        if self.path == '/health':
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress default logging


class PicoClawIngest:
    """
    Runs a lightweight HTTP server to receive regime score webhooks
    from PicoClaw Research agent on the Dell.
    """

    def __init__(self, port: int, on_regime_update: Callable[[float], None]):
        """
        Args:
            port: Port to listen on (e.g., 9090).
            on_regime_update: Callback with new regime weight. Typically combiner.set_regime_weight.
        """
        self.port = port
        self.on_regime_update = on_regime_update
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        RegimeHandler.callback = self.on_regime_update
        self._server = HTTPServer(('0.0.0.0', self.port), RegimeHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[PicoClawIngest] Listening on :{self.port}/regime")

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
