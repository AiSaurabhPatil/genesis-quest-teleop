from __future__ import annotations

import logging
import socket
import threading

from ..protocol import parse_quest_packet

LOGGER = logging.getLogger(__name__)
DISCONNECT_MESSAGE = b"GENESIS_QUEST_TELEOP_DISCONNECTED"


class LocalStatePublisher:
    """Publishes validated Quest packets to a restartable local simulator."""

    def __init__(self, config):
        ipc = config["ipc"]
        self.address = (ipc["host"], int(ipc["port"]))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)

    def _send(self, payload: bytes) -> None:
        try:
            self.socket.sendto(payload, self.address)
        except (BlockingIOError, OSError):
            # Controller packets are ephemeral. If the simulator is restarting or
            # its local socket is busy, dropping this sample is the correct policy.
            return

    def publish(self, raw: str | bytes) -> None:
        payload = raw.encode("utf-8") if isinstance(raw, str) else raw
        self._send(payload)

    def mark_disconnected(self) -> None:
        self._send(DISCONNECT_MESSAGE)

    def close(self) -> None:
        self.socket.close()


class LocalStateReceiver:
    """Receives only the newest local packet into a LatestQuestStateStore."""

    def __init__(self, config, state_store, diagnostics):
        ipc = config["ipc"]
        self.address = (ipc["host"], int(ipc["port"]))
        self.max_bytes = int(config["webrtc"]["max_message_bytes"])
        self.store = state_store
        self.diagnostics = diagnostics
        self.socket = None
        self.thread = None
        self.stopping = threading.Event()

    def start(self) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.address)
        self.socket.settimeout(0.5)
        self.thread = threading.Thread(
            target=self._run,
            name="quest-local-state-receiver",
            daemon=True,
        )
        self.thread.start()
        LOGGER.info("Receiving Quest state from local ingress on %s:%s", *self.address)

    def _run(self) -> None:
        while not self.stopping.is_set():
            try:
                raw, _ = self.socket.recvfrom(self.max_bytes + 1)
            except TimeoutError:
                continue
            except OSError:
                break

            if raw == DISCONNECT_MESSAGE:
                self.store.mark_disconnected()
                continue

            try:
                packet = parse_quest_packet(raw, self.max_bytes)
            except (UnicodeDecodeError, ValueError):
                self.diagnostics.increment("packets_rejected")
                continue

            if self.store.replace(packet):
                self.diagnostics.increment("packets_received")
                self.diagnostics.set_value("active_session_id", packet.session_id)
                self.diagnostics.set_value("last_sequence", packet.sequence)
            else:
                self.diagnostics.increment("out_of_order_packets")

    def stop(self) -> None:
        self.stopping.set()
        if self.socket:
            self.socket.close()
        if self.thread:
            self.thread.join(timeout=2)
