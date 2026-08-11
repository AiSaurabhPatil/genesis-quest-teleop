from __future__ import annotations

import logging
import time


class Diagnostics:
    def __init__(self, config):
        self.values = {
            name: 0
            for name in [
                "packets_received",
                "packets_rejected",
                "out_of_order_packets",
                "peer_connections",
                "peer_disconnects",
                "stale_holds",
                "tracking_loss_holds",
                "clutch_engages",
                "clutch_releases",
                "diffik_failures",
                "safety_clamps",
                "control_updates",
            ]
        }
        self._period = config["diagnostics"]["log_period_s"]
        self._last = 0

    def increment(self, name, amount=1):
        self.values[name] = self.values.get(name, 0) + amount

    def set_value(self, name, value):
        self.values[name] = value

    def maybe_log(self):
        if time.monotonic() - self._last >= self._period:
            logging.info("teleop %s", self.values)
            self._last = time.monotonic()
