from __future__ import annotations

import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ControllerState:
    position_xyz: tuple[float, float, float] | None
    orientation_xyzw: tuple[float, float, float, float] | None
    trigger: float
    squeeze: float
    thumbstick_x: float
    thumbstick_y: float
    button_a_x: bool
    button_b_y: bool
    thumbstick_click: bool


@dataclass(frozen=True)
class QuestStatePacket:
    schema_version: int
    session_id: str
    sequence: int
    timestamp: float | None
    client_epoch_ms: float | None
    controllers: dict[str, ControllerState]


def _number(value, *, required=False):
    if value is None and not required:
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a number")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError("invalid numeric value") from None
    if not math.isfinite(value):
        raise ValueError("non-finite numeric value")
    return value


def _pose(data, keys):
    value = data.get("position" if len(keys) == 3 else "orientation")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("pose must be object or null")
    result = tuple(_number(value.get(k), required=True) for k in keys)
    if len(keys) == 4 and sum(x * x for x in result) < 1e-12:
        return None
    return result


def _controller(data):
    if not isinstance(data, dict):
        raise ValueError("controller must be object")
    clamp = lambda x: max(0.0, min(1.0, _number(x) or 0.0))
    return ControllerState(
        _pose(data, ("x", "y", "z")),
        _pose(data, ("x", "y", "z", "w")),
        clamp(data.get("trigger")),
        clamp(data.get("squeeze")),
        _number(data.get("thumbstick_x")) or 0.0,
        _number(data.get("thumbstick_y")) or 0.0,
        bool(data.get("button_a_x", False)),
        bool(data.get("button_b_y", False)),
        bool(data.get("thumbstick_click", False)),
    )


def parse_quest_packet(raw: str | bytes, max_bytes: int) -> QuestStatePacket:
    if isinstance(raw, bytes):
        if len(raw) > max_bytes:
            raise ValueError("packet too large")
        raw = raw.decode("utf-8")
    elif not isinstance(raw, str) or len(raw.encode()) > max_bytes:
        raise ValueError("packet too large or not text")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("unsupported schema version")
    session = data.get("session_id")
    seq = data.get("sequence")
    if not isinstance(session, str) or not session:
        raise ValueError("session_id required")
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
        raise ValueError("invalid sequence")
    controllers = data.get("controllers", {})
    if not isinstance(controllers, dict):
        raise ValueError("controllers must be object")
    return QuestStatePacket(
        1,
        session,
        seq,
        _number(data.get("timestamp")),
        _number(data.get("client_epoch_ms")),
        {h: _controller(v) for h, v in controllers.items() if h in ("left", "right")},
    )
