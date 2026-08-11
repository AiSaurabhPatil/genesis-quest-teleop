from __future__ import annotations

import argparse
import logging
import time

from .config import load_config, project_root
from .diagnostics import Diagnostics
from .state_store import LatestQuestStateStore
from .transport.local_ipc import LocalStatePublisher
from .transport.webrtc_server import WebRTCServer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persistent Quest HTTPS/WebRTC ingress service"
    )
    parser.add_argument("--config", default=str(project_root() / "config/default.yaml"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper())
    config = load_config(args.config)
    store = LatestQuestStateStore()
    diagnostics = Diagnostics(config)
    publisher = LocalStatePublisher(config)
    server = WebRTCServer(config, store, diagnostics, publisher)

    try:
        server.start_in_thread()
        logging.info(
            "Quest ingress is persistent; restart Genesis with --external-ingress"
        )
        while True:
            diagnostics.maybe_log()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        publisher.close()
