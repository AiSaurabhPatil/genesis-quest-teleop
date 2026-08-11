from __future__ import annotations

import argparse
import logging

from .app import GenesisTeleopApp
from .config import load_config, project_root


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(project_root() / "config/default.yaml"))
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument(
        "--external-ingress",
        action="store_true",
        help="receive Quest state from a separately running genesis-quest-ingress",
    )
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=a.log_level.upper())
    c = load_config(a.config)
    if a.cpu:
        c["sim"]["backend"] = "cpu"
    if a.headless:
        c["sim"]["show_viewer"] = False
    app = GenesisTeleopApp(c, external_ingress=a.external_ingress)
    try:
        app.setup()
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()
