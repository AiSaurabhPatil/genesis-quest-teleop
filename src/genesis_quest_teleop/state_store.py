from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .protocol import QuestStatePacket


@dataclass(frozen=True)
class ReceivedQuestState:
    packet: QuestStatePacket
    receive_monotonic_ns: int
    receive_epoch_ms: float


class LatestQuestStateStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = None
        self._session = None
        self._sequence = -1
        self._connected = False

    def replace(self, packet):
        with self._lock:
            if packet.session_id == self._session and packet.sequence <= self._sequence:
                return False
            self._session, self._sequence, self._connected = (
                packet.session_id,
                packet.sequence,
                True,
            )
            self._state = ReceivedQuestState(
                packet, time.monotonic_ns(), time.time() * 1000
            )
            return True

    def snapshot(self):
        with self._lock:
            return self._state

    def mark_disconnected(self, session_id=None):
        with self._lock:
            if session_id is None or session_id == self._session:
                self._connected = False

    def is_connected(self):
        with self._lock:
            return self._connected
