# Genesis Quest Teleoperation — Codex Implementation Plan

> Execution target: Codex / coding LLM. Optimize for implementation correctness and low token ambiguity. Do not treat this as end-user documentation.

## 0. EXECUTION CONTRACT

Implement a new standalone project named `genesis-quest-teleop` as a sibling of the existing repositories.

Expected local layout:

```text
/home/saurabh/Developement/
├── genesis-world/
├── telesim/
└── genesis-quest-teleop/      # create/modify only this project
```

Hard constraints:

1. DO NOT modify `../genesis-world`.
2. DO NOT modify `../telesim`.
3. DO NOT make TeleSim a runtime dependency.
4. DO inspect TeleSim for WebXR input conventions, packet schema, low-latency latest-sample behavior, clutch semantics, and TLS setup.
5. DO use the local Genesis World clone as an editable `uv` path dependency. Declare `genesis-world` in `[project].dependencies` and map it with `[tool.uv.sources] genesis-world = { path = "../genesis-world", editable = true }`. DO NOT install Genesis with ad-hoc `pip install` commands.
6. Quest controller state MUST travel over a WebRTC RTCDataChannel.
7. DO NOT use ROS, ROS 2, `rclpy`, Isaac Sim, Lula, Omni APIs, deferred rendering, dataset recording, or camera recording in the MVP.
8. WebSocket is NOT allowed as the controller-data transport. Signaling should use HTTPS `POST /api/offer`; no persistent signaling socket is required for the LAN MVP.
9. Genesis simulation/viewer MUST remain on the main process/thread. Network/WebRTC async work MUST NOT call Genesis APIs directly.
10. Use a latest-state buffer. Never queue old controller pose packets for simulation consumption.
11. Default robot for the first complete end-to-end implementation: Genesis built-in Franka Panda.
12. Capture BOTH Quest controllers, but map only the configured hand (`right` by default) to the single-arm Franka MVP.
13. Use squeeze/grip as the motion clutch and trigger as gripper command.
14. Clutch engagement MUST be zero-jump: capture current controller pose and measured current robot end-effector pose on the clutch rising edge.
15. Default realtime arm controller MUST be Genesis-native Jacobian damped differential IK, based on the official `examples/rigid/diffik_controller.py` pattern.
16. Full `robot.inverse_kinematics(...)` may be used only for initialization, explicit recovery, or debug fallback; do not solve full iterative IK on every control frame by default.
17. Network loss, controller tracking loss, stale packets, invalid values, IK numerical failure, or peer disconnect MUST result in HOLD, never continued extrapolated motion.
18. Keep scope minimal: Quest input -> WebRTC -> coordinate mapping -> clutch -> EE target -> Genesis DiffIK -> gripper -> `scene.step()`.
19. Do not add a broad unit-test suite. Validate with syntax/compile checks plus deterministic local smoke modes and manual end-to-end checks described below.
20. Preserve modular boundaries so bimanual/OpenArm support can be added later without rewriting transport or mapping.

---

## 1. SOURCE OF TRUTH TO INSPECT BEFORE EDITING

### 1.1 Genesis World

Repository:

```text
https://github.com/Genesis-Embodied-AI/genesis-world
```

Reference baseline inspected for this plan:

```text
version: 1.3.2
commit: 8e70e94ae50f0a36d1a9e85e1b52e65758febe39
```

Codex MUST inspect the local checkout first because the local clone is authoritative.

Required reference files:

```text
../genesis-world/examples/rigid/diffik_controller.py
../genesis-world/examples/rigid/ik_franka.py
../genesis-world/examples/tutorials/control_your_robot.py
../genesis-world/examples/ipc/ipc_robot_cloth_teleop.py
../genesis-world/genesis/engine/entities/rigid_entity/rigid_entity.py
../genesis-world/pyproject.toml
```

Relevant Genesis APIs verified in 1.3.2:

```python
robot.get_link(name)
link.get_pos()
link.get_quat()
robot.get_qpos()
robot.get_dofs_position(...)
robot.get_dofs_limit(...)
robot.get_jacobian(link=...)
robot.inverse_kinematics(..., dofs_idx_local=...)
robot.control_dofs_position(position, dofs_idx_local=...)
robot.set_qpos(...)
robot.set_dofs_position(...)
scene.step()
scene.draw_debug_frame(...)
scene.update_debug_objects(...)
```

Important API behavior:

- `robot.get_jacobian(link=...)` returns a Jacobian for ALL entity DOFs. Slice its columns to `arm_dofs_idx` before DiffIK.
- `robot.inverse_kinematics(..., dofs_idx_local=...)` supports selecting arm DOFs and respects joint limits by default.
- `control_dofs_position()` is the normal PD target API for actual simulation control.
- Genesis quaternions are handled as `wxyz` in the relevant examples/utilities.
- WebXR controller orientation arrives as `xyzw`; conversion is mandatory.

### 1.2 TeleSim

Repository:

```text
https://github.com/AiSaurabhPatil/telesim
```

Reference branch:

```text
clutch_teleop
```

Inspect these files only as implementation references:

```text
../telesim/web/webxr_streamer.html
../telesim/src/launch/webxr_bridge.py
../telesim/src/quest_ingress/message_types.py
../telesim/src/quest_ingress/metrics.py
../telesim/config/config.yaml
../telesim/src/robot_adapters/bimanual_lula.py
```

Reuse concepts, not Isaac/ROS implementation:

- WebXR `local-floor` reference space.
- Quest Touch gamepad index mapping.
- controller packet fields and sequence/timestamp fields.
- latest-sample / drop-stale design.
- TLS configuration pattern.
- clutch edge semantics.
- zero-jump takeover anchored to measured robot EE state.
- transport observability concepts.

DO NOT copy or import:

```text
rclpy
geometry_msgs
sensor_msgs
Isaac Sim launch/runtime code
Lula solvers
USD/Omniverse articulation adapters
TeleSim recording/deferred-rendering code
```

---

## 2. MVP OUTCOME

After implementation:

1. Run one Python command on the Genesis workstation.
2. Genesis starts a Franka scene and an HTTPS/WebRTC service.
3. Open `https://<PC_LAN_IP>:8443` in Meta Quest Browser.
4. Start immersive AR; fallback to immersive VR when AR is unavailable.
5. Browser creates an unordered/unreliable WebRTC DataChannel named `quest-teleop`.
6. Quest sends latest left/right controller poses and button state at XR frame rate.
7. PC stores only the latest valid packet.
8. Franka remains still while right squeeze/grip is released.
9. On right squeeze rising edge, controller and actual EE poses are anchored without a jump.
10. While squeeze is held, relative controller motion moves the Franka EE.
11. Right trigger controls gripper opening/closing.
12. Release squeeze -> arm target freezes immediately.
13. Disconnect/stale tracking -> arm target freezes immediately.
14. Genesis viewer shows the robot plus a debug frame for the current target pose.

No headset video stream from Genesis is required in this milestone. The Quest browser may remain passthrough/blank while controllers are used as input. Genesis visual feedback is on the PC viewer.

---

## 3. TARGET PROJECT TREE

Create exactly this initial structure unless an equivalent existing structure is already present:

```text
genesis-quest-teleop/
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock                  # generated by `uv lock` / `uv sync`; commit it
├── README.md
├── config/
│   └── default.yaml
├── web/
│   ├── index.html
│   └── quest_client.js
├── src/
│   └── genesis_quest_teleop/
│       ├── __init__.py
│       ├── main.py
│       ├── config.py
│       ├── protocol.py
│       ├── state_store.py
│       ├── diagnostics.py
│       ├── transport/
│       │   ├── __init__.py
│       │   └── webrtc_server.py
│       ├── input/
│       │   ├── __init__.py
│       │   ├── frames.py
│       │   └── clutch.py
│       ├── control/
│       │   ├── __init__.py
│       │   └── diffik.py
│       ├── robots/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── franka.py
│       └── app.py
└── scripts/
    └── check.sh
```

Do not create more abstractions/files during MVP unless required by an actual API issue.

---

## 4. DATA FLOW — DO NOT DEVIATE

```text
Meta Quest Browser
    WebXR frame
      -> controller gripSpace pose + gamepad state
      -> JSON QuestStatePacket
      -> WebRTC RTCDataChannel (unordered, maxRetransmits=0)
            |
            v
Python aiortc network thread
      -> parse/validate packet
      -> LatestQuestStateStore.replace(packet)
            |
            | lock-protected snapshot only
            v
Genesis main thread
      -> read newest packet
      -> stale/tracking checks
      -> WebXR -> Genesis frame mapping
      -> clutch edge detector
      -> zero-jump relative EE target
      -> workspace + motion clamps
      -> Franka DiffIK
      -> position-control arm DOFs
      -> trigger -> finger position target
      -> scene.step()
```

Critical rule:

```text
Network callback NEVER calls robot.*, scene.*, link.*, Genesis renderer, or viewer APIs.
```

---

## 5. WIRE PROTOCOL

Preserve TeleSim-compatible controller field names where practical so later interoperability remains easy.

### 5.1 Controller state packet

Browser sends JSON text:

```json
{
  "schema_version": 1,
  "session_id": "uuid-string",
  "sequence": 1234,
  "timestamp": 481292.331,
  "client_epoch_ms": 1786420012345,
  "controllers": {
    "left": {
      "position": {"x": 0.0, "y": 1.2, "z": -0.4},
      "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
      "trigger": 0.0,
      "squeeze": 0.0,
      "thumbstick_x": 0.0,
      "thumbstick_y": 0.0,
      "button_a_x": false,
      "button_b_y": false,
      "thumbstick_click": false
    },
    "right": {
      "position": {"x": 0.0, "y": 1.2, "z": -0.4},
      "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
      "trigger": 0.0,
      "squeeze": 0.0,
      "thumbstick_x": 0.0,
      "thumbstick_y": 0.0,
      "button_a_x": false,
      "button_b_y": false,
      "thumbstick_click": false
    }
  }
}
```

Rules:

- Missing controller is allowed.
- `position=null` or `orientation=null` means not currently tracked.
- Reject non-finite numeric pose values.
- Normalize quaternion before use.
- Reject quaternion norm `< 1e-6`.
- Clamp analog inputs to `[0, 1]`.
- Sequence is monotonic only inside one `session_id`.
- On a new `session_id`, accept sequence restart from zero.
- Do not deserialize directly into Genesis tensors in network callback.

### 5.2 Future control messages

Parser may ignore unknown top-level fields. Do not implement a complex bidirectional protocol now.

The server may send only optional informational messages, e.g.:

```json
{"type":"server_hello","schema_version":1}
```

Controller-state messages remain the dominant one-way path.

---

## 6. WEBRTC DESIGN

### 6.1 Browser peer

Browser creates:

```javascript
const pc = new RTCPeerConnection(rtcConfig);
const dataChannel = pc.createDataChannel("quest-teleop", {
  ordered: false,
  maxRetransmits: 0
});
```

Reason: controller poses are ephemeral. Late pose packets are worse than dropped packets.

Do not use reliable ordered transport for the pose stream.

### 6.2 LAN signaling

Use one HTTPS endpoint:

```text
POST /api/offer
Content-Type: application/json
```

Request:

```json
{"sdp":"...","type":"offer"}
```

Response:

```json
{"sdp":"...","type":"answer"}
```

Single-shot non-trickle ICE is sufficient for this LAN milestone.

Browser flow:

```text
create RTCPeerConnection
create DataChannel
createOffer
setLocalDescription
wait for ICE gathering state == complete
POST localDescription to /api/offer
receive answer
setRemoteDescription(answer)
wait for DataChannel open
start sending XR frames
```

Python flow:

```text
receive offer
create RTCPeerConnection
register pc.on("datachannel") before/around remote setup
setRemoteDescription(offer)
createAnswer
setLocalDescription(answer)
return pc.localDescription
retain pc in a set until closed/failed
```

### 6.3 HTTPS

WebXR on a headset must be served from a secure origin.

Configuration MUST support explicit certificate/key paths. Do not generate or commit private keys automatically.

Allow paths to point at already-working TeleSim certificates without copying them:

```yaml
tls:
  cert_file: "../telesim/certs/cert.pem"
  key_file: "../telesim/certs/key.pem"
```

Also allow environment overrides:

```text
GENESIS_TELEOP_CERT
GENESIS_TELEOP_KEY
```

Validate files before starting the server. Fail with a clear error if HTTPS is enabled and paths are missing.

### 6.4 Remote/NAT scope

Do NOT implement TURN in MVP.

Structure config to allow later:

```yaml
webrtc:
  ice_servers: []
```

For LAN, empty ICE server list is acceptable.

---

## 7. WEBXR INPUT MAPPING

Use TeleSim `web/webxr_streamer.html` as the behavioral reference.

Request:

```javascript
navigator.xr.requestSession(mode, {
  requiredFeatures: ["local-floor"]
});
```

Prefer:

```text
immersive-ar
```

Fallback:

```text
immersive-vr
```

Each XR frame:

1. Iterate `xrSession.inputSources`.
2. Ignore sources without `gamepad`.
3. Use `inputSource.handedness` as `left` / `right`.
4. Read pose using `frame.getPose(inputSource.gripSpace, xrRefSpace)`.
5. Keep raw WebXR pose. Do frame conversion on Python side.
6. Use Quest Touch mappings consistent with TeleSim:

```text
gamepad.buttons[0] -> trigger
gamepad.buttons[1] -> squeeze/grip
gamepad.buttons[3] -> thumbstick_click
gamepad.buttons[4] -> A/X primary
gamepad.buttons[5] -> B/Y secondary
gamepad.axes[2]    -> thumbstick_x
gamepad.axes[3]    -> thumbstick_y
```

Defensively guard missing indices.

### 7.1 Send policy

Send at XR animation-frame rate only while:

```text
XR session active AND RTCDataChannel.readyState == "open"
```

Do not accumulate packets if channel is unavailable.

Optionally use:

```javascript
if (dataChannel.bufferedAmount > configuredLimit) return;
```

Default `bufferedAmount` skip threshold: `65536` bytes.

This is a drop-new protection only for severe backpressure. Normal operation should stay near zero.

---

## 8. COORDINATE SYSTEM CONVERSION

### 8.1 Conventions

WebXR `local-floor`:

```text
+X = right
+Y = up
-Z = forward
right-handed
quaternion = xyzw
```

Use this Genesis teleop convention:

```text
+X = robot/operator forward
+Y = operator left
+Z = up
right-handed
quaternion used by Genesis control = wxyz
```

Default basis transform from WebXR vectors into Genesis vectors:

```python
R_G_XR = np.array([
    [ 0.0, 0.0, -1.0],
    [-1.0, 0.0,  0.0],
    [ 0.0, 1.0,  0.0],
])
```

Therefore:

```text
Genesis X = -WebXR Z
Genesis Y = -WebXR X
Genesis Z =  WebXR Y
```

For position:

```python
p_g = R_G_XR @ p_xr
```

For orientation:

```python
R_ctrl_g = R_G_XR @ R_ctrl_xr @ R_G_XR.T
```

Then convert `R_ctrl_g` to quaternion and reorder to Genesis `wxyz`.

Do NOT transform quaternion components by simply permuting axes. Use rotation matrices or mathematically correct quaternion basis conversion.

Use `scipy.spatial.transform.Rotation` in `frames.py` for reliability; add `scipy` as a project dependency.

### 8.2 Quaternion continuity

Before computing orientation delta, keep sign continuity:

```python
if np.dot(q_now_wxyz, q_prev_wxyz) < 0:
    q_now_wxyz = -q_now_wxyz
```

Normalize every quaternion after transform/composition.

---

## 9. ZERO-JUMP CLUTCH

Configured hand default: `right`.

Default controls:

```text
squeeze >= 0.55  -> arm clutch engaged
trigger           -> gripper command
```

Use hysteresis:

```text
engage_threshold = 0.55
release_threshold = 0.45
```

This prevents analog chatter.

### 9.1 State machine

States:

```text
DISENGAGED
ENGAGED
```

Track:

```python
controller_anchor_pos_g
controller_anchor_quat_g
robot_anchor_pos_g
robot_anchor_quat_g
last_target_pos_g
last_target_quat_g
```

### 9.2 Engage rising edge

On `DISENGAGED -> ENGAGED`:

1. Verify controller position/orientation valid.
2. Read actual current EE pose from Genesis:

```python
robot_anchor_pos = ee_link.get_pos().cpu().numpy()
robot_anchor_quat = ee_link.get_quat().cpu().numpy()
```

3. Save mapped controller pose as controller anchor.
4. Set target pose equal to measured EE pose.
5. Do NOT use configured home pose as the takeover anchor.
6. Do NOT use previously commanded target as the takeover anchor.

Result: clutch can be released, repositioned physically, and re-engaged with no robot jump.

### 9.3 While engaged

Translation:

```python
delta_p = controller_pos_g - controller_anchor_pos_g
raw_target_pos = robot_anchor_pos_g + translation_scale * delta_p
```

Default:

```yaml
translation_scale: 1.0
```

Orientation should apply controller WORLD rotation delta to the anchored robot orientation.

Conceptually:

```text
R_delta = R_controller_now * inverse(R_controller_anchor)
R_target = R_delta * R_robot_anchor
```

Do not use Euler-angle subtraction.

### 9.4 Release falling edge

On `ENGAGED -> DISENGAGED`:

- preserve the last valid target pose;
- command HOLD through the existing joint target;
- clear only controller/robot clutch anchors;
- do not reset robot position;
- next engagement creates fresh anchors.

### 9.5 Tracking loss while engaged

If active controller pose becomes invalid/missing:

- transition to disengaged/hold;
- require squeeze to be released and pressed again before motion resumes;
- do not auto-reanchor while squeeze remains held.

Implement a `requires_rearm` latch.

---

## 10. SAFETY AND TARGET LIMITING

Apply safety before DiffIK.

### 10.1 Packet stale timeout

Default:

```yaml
stale_timeout_ms: 150
```

Use PC monotonic receive time, not client timestamps, to decide staleness.

If stale:

```text
arm -> HOLD
gripper -> HOLD last command
clutch -> force disengaged + require rearm
```

### 10.2 Workspace clamp

Default Franka workspace in Genesis world coordinates:

```yaml
workspace:
  x: [0.20, 0.80]
  y: [-0.55, 0.55]
  z: [0.08, 0.90]
```

Clamp target position componentwise.

Keep these values config-driven.

### 10.3 Per-control-tick target clamps

After workspace clamp, limit target movement from previous valid target.

Defaults:

```yaml
max_target_translation_step_m: 0.025
max_target_rotation_step_rad: 0.20
```

Do not let a malformed pose create a large instantaneous target jump.

### 10.4 DiffIK joint delta clamp

Default:

```yaml
max_joint_delta_rad: 0.08
joint_limit_margin_rad: 0.03
```

Use `robot.get_dofs_limit(arm_dofs_idx)` once after build. Clip finite limits with margin.

Do not clamp unlimited/infinite DOF boundaries to arbitrary values.

### 10.5 Numerical failure

If any of these occur:

```text
NaN/Inf error vector
NaN/Inf Jacobian
linear solve failure
NaN/Inf dq
invalid robot q
```

then:

```text
return previous q command
increment diagnostic counter
DO NOT move the robot
```

---

## 11. GENESIS DIFFERENTIAL IK

Base implementation directly on the Genesis official DiffIK example.

### 11.1 Pose error

At each control update:

```python
current_pos = ee_link.get_pos().cpu().numpy()
current_quat = ee_link.get_quat().cpu().numpy()  # wxyz

error_pos = target_pos - current_pos
error_quat = gs.transform_quat_by_quat(gs.inv_quat(current_quat), target_quat)
error_rotvec = gs.quat_to_rotvec(error_quat)
error = np.concatenate([error_pos, error_rotvec])
```

Use the Genesis orientation-error convention above to stay aligned with the official implementation.

### 11.2 Jacobian

```python
jac_full = robot.get_jacobian(link=ee_link).cpu().numpy()
jac = jac_full[:, arm_dofs_idx]
```

Do not assume `get_jacobian` accepts `dofs_idx_local`; Genesis 1.3.2 returns all entity DOFs.

### 11.3 Damped least squares

Implement:

```python
lambda2 = damping * damping
A = jac @ jac.T + lambda2 * np.eye(6)
dq = jac.T @ np.linalg.solve(A, error_scaled)
```

Recommended config:

```yaml
diffik:
  damping: 0.02
  position_gain: 0.7
  rotation_gain: 0.5
  max_joint_delta_rad: 0.08
```

Compute:

```python
error_scaled = np.concatenate([
    position_gain * error_pos,
    rotation_gain * error_rotvec,
])
```

### 11.4 Command

```python
q_arm = robot.get_dofs_position(arm_dofs_idx).cpu().numpy()
q_cmd = q_arm + dq
q_cmd = apply_joint_delta_clamp(q_arm, q_cmd)
q_cmd = apply_joint_limit_clip(q_cmd)
robot.control_dofs_position(q_cmd, dofs_idx_local=arm_dofs_idx)
```

Never use `robot.set_qpos()` for live teleoperation. `set_qpos()` changes state directly and bypasses physical PD response.

### 11.5 Full IK fallback

Implement an explicit method, not automatic per-frame behavior:

```python
recover_to_target(target_pos, target_quat)
```

It may call:

```python
robot.inverse_kinematics(
    link=ee_link,
    pos=target_pos,
    quat=target_quat,
    dofs_idx_local=arm_dofs_idx,
    respect_joint_limit=True,
    max_solver_iters=20,
    damping=0.01,
)
```

Do not invoke recovery continuously. A failure in DiffIK should HOLD and log; recovery is manual/debug only in MVP.

---

## 12. GRIPPER CONTROL

Use right trigger analog value.

Franka finger target mapping:

```text
trigger = 0.0 -> open
trigger = 1.0 -> closed
```

Config:

```yaml
gripper:
  open_position: 0.04
  closed_position: 0.0
  trigger_deadzone: 0.05
```

Map continuously:

```python
t = clamp((trigger - deadzone) / (1.0 - deadzone), 0.0, 1.0)
finger_target = open_position + t * (closed_position - open_position)
```

Command both finger DOFs:

```python
robot.control_dofs_position(
    np.array([finger_target, finger_target]),
    dofs_idx_local=finger_dofs_idx,
)
```

When active packet is stale/disconnected, hold the last finger target. Do not snap open.

---

## 13. SIMULATION TIMING

Use fixed physics timestep.

Default:

```yaml
sim:
  backend: gpu
  dt: 0.008333333333333333   # 120 Hz
  control_hz: 60
  show_viewer: true
```

Control every `round((1/dt)/control_hz)` physics steps.

With defaults:

```text
physics = 120 Hz
control = 60 Hz
control every 2 physics steps
```

Do not sleep to artificially enforce realtime unless Genesis is running faster than realtime AND realtime pacing is required for human control. Prefer viewer/simulator pacing first. If explicit pacing is necessary, implement only in `app.py` and base it on monotonic wall time.

Important low-latency rule:

```text
Each control tick consumes the newest packet currently available.
No interpolation queue.
No jitter buffer in MVP.
No pose prediction in MVP.
```

Prediction can be added later after end-to-end latency is measured.

---

## 13.5 ENVIRONMENT + DEPENDENCY BOOTSTRAP — USE UV ONLY

Codex MUST use `uv` for Python interpreter selection, virtual-environment creation, dependency resolution, locking, installation, and command execution for this project.

Do NOT use:

```text
python -m venv
pip install ...
python -m pip ...
conda create ...
poetry install
```

### Required Python version

Use Python **3.11** for the MVP. Genesis World 1.3.2 accepts Python `>=3.10,<3.14`; Python 3.11 is the fixed project interpreter to reduce dependency ambiguity.

Create `.python-version` containing exactly:

```text
3.11
```

### Verify/install uv

Before creating the project environment:

```bash
command -v uv
uv --version
```

Require `uv >= 0.5.3` because the PyTorch index/source configuration used by this plan depends on modern uv PyTorch integration. If `uv` is missing or older, upgrade it before creating the project environment. Do not fall back to pip-managed project setup.

### Create the virtual environment

From the new project root:

```bash
cd /home/saurabh/Developement/genesis-quest-teleop
uv python install 3.11
uv venv --python 3.11 .venv
```

Activation is optional for `uv run`, but for interactive shell use:

```bash
source .venv/bin/activate
```

Verify interpreter:

```bash
uv run python -c "import sys; print(sys.executable); print(sys.version)"
```

Expected interpreter path must resolve inside:

```text
.../genesis-quest-teleop/.venv/...
```

### Dependency model

All direct runtime dependencies MUST be declared in `pyproject.toml`:

```text
torch           # CUDA-enabled PyTorch from the official cu128 index; required for RTX 5070 Ti / Blackwell
genesis-world   # editable local sibling dependency through tool.uv.sources
aiortc          # WebRTC peer connection + RTCDataChannel
aiohttp         # HTTPS static server + POST /api/offer signaling
numpy           # control/state math
scipy           # quaternion/rotation utilities
PyYAML          # YAML configuration
```

Development dependency:

```text
ruff             # syntax/static lint check
```

Do not manually install transitive packages such as `av`, `cryptography`, `pylibsrtp`, `aioice`, or `cffi`; let `uv` resolve them from the declared direct dependencies. Genesis World's own dependencies must also be resolved transitively from `../genesis-world/pyproject.toml`.

### RTX 5070 Ti / Blackwell CUDA requirement

This workstation uses an NVIDIA GeForce RTX 5070 Ti. Treat GPU compatibility as a hard prerequisite, not an optional optimization. The RTX 5070 Ti is NVIDIA Blackwell with CUDA compute capability 12.0 (`sm_120`). Do not allow `uv` to resolve a CPU-only PyTorch build or an older CUDA wheel accidentally.

Use the official PyTorch CUDA 12.8 wheel index explicitly. CUDA 12.8 is the minimum project wheel target for this Blackwell GPU. Do not use `cu118`, `cu121`, `cu124`, or `cu126` for this project. A newer Blackwell-compatible CUDA wheel may only replace `cu128` after verifying Genesis compatibility; otherwise keep `cu128` as the deterministic baseline.

The host NVIDIA driver must be new enough to run the selected CUDA runtime. Do NOT require a separately installed system CUDA Toolkit merely to run PyTorch/Genesis wheels. Check the driver with `nvidia-smi`; PyTorch wheels provide their CUDA runtime.

### Local Genesis + PyTorch dependency sources

The local sibling Genesis checkout is authoritative. `pyproject.toml` MUST contain:

```toml
[project]
dependencies = [
    "torch",
    "genesis-world",
    "aiortc",
    "aiohttp",
    "numpy",
    "scipy",
    "PyYAML",
]

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
genesis-world = { path = "../genesis-world", editable = true }

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

`explicit = true` is mandatory so only packages explicitly mapped to the PyTorch index are resolved there; normal dependencies must continue to resolve from PyPI.

This replaces `pip install -e ../genesis-world`. Do not mix `uv sync` with manual pip installation because `uv sync` performs an exact project sync and can remove undeclared packages.

### Resolve and install

After `pyproject.toml` exists:

```bash
uv lock
uv sync
```

The `dev` dependency group is synced by default. Commit `uv.lock` so Codex and future runs resolve the same dependency graph.

Then verify the environment before writing simulator integration code. First verify the host driver:

```bash
nvidia-smi
```

Failure to detect the RTX 5070 Ti is a blocking environment error. Do not continue by silently falling back to CPU.

Verify imports and the actual PyTorch CUDA build:

```bash
uv run python - <<'PY'
import aiohttp
import aiortc
import genesis as gs
import numpy as np
import scipy
import torch
import yaml

print("dependency-imports: ok")
print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("genesis:", getattr(gs, "__version__", "unknown"))
print("numpy:", np.__version__)
print("scipy:", scipy.__version__)
print("aiohttp:", aiohttp.__version__)
print("aiortc:", aiortc.__version__)

assert torch.cuda.is_available(), "BLOCKING: PyTorch cannot access CUDA"
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
print("GPU:", name)
print("compute capability:", capability)
assert "5070 Ti" in name, f"Expected RTX 5070 Ti, got: {name}"
assert capability == (12, 0), f"Expected Blackwell sm_120, got: {capability}"

# Actual kernel execution is mandatory; import/CUDA discovery alone is insufficient.
a = torch.randn((1024, 1024), device="cuda")
b = torch.randn((1024, 1024), device="cuda")
c = a @ b
torch.cuda.synchronize()
assert c.is_cuda
print("torch-cuda-kernel: ok")
PY
```

Then verify Genesis itself resolves to CUDA and can step a scene. Run this as a separate Python process because `gs.init()` is process-global:

```bash
uv run python - <<'PY'
import genesis as gs

gs.init(backend=gs.cuda, logging_level="warning")
print("Genesis backend:", gs.backend)
print("Genesis device:", gs.device)
assert gs.backend == gs.cuda, f"Genesis did not resolve CUDA backend: {gs.backend}"

scene = gs.Scene(show_viewer=False)
scene.add_entity(gs.morphs.Plane())
scene.build()
for _ in range(10):
    scene.step()
print("genesis-cuda-smoke: ok")
PY
```

If either CUDA smoke test fails, STOP. Fix the NVIDIA driver / PyTorch CUDA wheel / Genesis environment before writing the realtime teleoperation implementation. Never change the runtime to `gs.cpu` merely to make validation pass.

Also verify dependency consistency:

```bash
uv pip check
uv tree
```

If `import genesis` fails, resolve the local editable source/path first. Do not continue by installing a different `genesis-world` release from PyPI.

### Command execution rule

Prefer:

```bash
uv run <command>
```

over relying on shell activation. Examples:

```bash
uv run genesis-quest-teleop --config config/default.yaml
uv run python -m compileall -q src
uv run ruff check src
```

---

## 14. FILE-BY-FILE IMPLEMENTATION

Execute in this exact order.

### STEP 1 — `.gitignore`

Create:

```gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
*.egg-info/
dist/
build/
.env
certs/*.pem
certs/*.key
```

Never commit TLS private keys.

---

### STEP 2 — `.python-version` + `pyproject.toml` + `uv.lock`

Create `.python-version`:

```text
3.11
```

Create `pyproject.toml` with project metadata, runtime dependencies, local Genesis source, dev dependencies, CLI entry point, and setuptools `src` discovery. Use this structure:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "genesis-quest-teleop"
version = "0.1.0"
description = "Low-latency Meta Quest teleoperation backend for Genesis World"
requires-python = ">=3.11,<3.12"
dependencies = [
    "torch",
    "genesis-world",
    "aiortc",
    "aiohttp",
    "numpy",
    "scipy",
    "PyYAML",
]

[project.scripts]
genesis-quest-teleop = "genesis_quest_teleop.main:main"

[dependency-groups]
dev = [
    "ruff",
]

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
genesis-world = { path = "../genesis-world", editable = true }

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.setuptools.packages.find]
where = ["src"]
```

Rules:

- `torch` MUST resolve from the explicit official `pytorch-cu128` index.
- Do not accept a CPU-only torch build.
- RTX 5070 Ti CUDA capability validation MUST report `(12, 0)`.
- `genesis-world` MUST resolve from `../genesis-world`, editable.
- Do NOT use a PyPI Genesis build for this project.
- Do NOT declare TeleSim as a dependency.
- Do NOT duplicate Genesis transitive dependencies in this project's direct dependency list unless this project imports/uses them directly.
- Keep `ruff` in the `dev` dependency group; do not add pytest for the MVP.

After creating the file, run:

```bash
uv lock
uv sync
uv pip check
nvidia-smi
# Then run the PyTorch + Genesis CUDA smoke tests defined in Section 13.5.
```

`uv sync` MUST install:

```text
current project -> editable
torch -> official PyTorch cu128 index
../genesis-world -> editable
aiortc
aiohttp
numpy
scipy
PyYAML
ruff
Genesis/transitive dependencies
WebRTC/transitive dependencies
```

Commit the generated `uv.lock`; do not hand-edit it.

---

### STEP 3 — `config/default.yaml`

Create one compact config containing all behavior that needs tuning.

Required structure:

```yaml
server:
  host: "0.0.0.0"
  port: 8443

web:
  directory: "web"

webrtc:
  data_channel_label: "quest-teleop"
  ice_servers: []
  max_message_bytes: 16384

# May point at already-working TeleSim certs. Do not copy keys.
tls:
  cert_file: "../telesim/certs/cert.pem"
  key_file: "../telesim/certs/key.pem"

sim:
  backend: "gpu"
  dt: 0.008333333333333333
  control_hz: 60
  show_viewer: true
  warmup_steps: 60

robot:
  type: "franka"
  mjcf_file: "xml/franka_emika_panda/panda.xml"
  end_effector_link: "hand"
  arm_joint_names:
    - joint1
    - joint2
    - joint3
    - joint4
    - joint5
    - joint6
    - joint7
  finger_joint_names:
    - finger_joint1
    - finger_joint2
  home_qpos: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]

teleop:
  active_hand: "right"
  translation_scale: 1.0
  clutch_engage_threshold: 0.55
  clutch_release_threshold: 0.45
  stale_timeout_ms: 150
  max_target_translation_step_m: 0.025
  max_target_rotation_step_rad: 0.20
  workspace:
    x: [0.20, 0.80]
    y: [-0.55, 0.55]
    z: [0.08, 0.90]

diffik:
  damping: 0.02
  position_gain: 0.7
  rotation_gain: 0.5
  max_joint_delta_rad: 0.08
  joint_limit_margin_rad: 0.03

gripper:
  open_position: 0.04
  closed_position: 0.0
  trigger_deadzone: 0.05

diagnostics:
  log_period_s: 2.0
  debug_target_frame: true
```

Do not add dozens of knobs beyond these in MVP.

---

### STEP 4 — `src/genesis_quest_teleop/config.py`

Implement YAML loading and path resolution.

Required functions:

```python
def load_config(path: str | Path) -> dict: ...
def project_root() -> Path: ...
def resolve_project_path(value: str | Path) -> Path: ...
def apply_env_overrides(config: dict) -> dict: ...
```

Behavior:

- project root is found relative to installed source, not current working directory only;
- relative `web.directory` resolves against project root;
- relative TLS paths resolve against project root;
- environment cert/key overrides take precedence;
- validate `sim.dt > 0`, `control_hz > 0`, thresholds ordered, workspace min < max;
- fail early with concise `ValueError` messages.

Environment variables:

```text
GENESIS_TELEOP_CERT
GENESIS_TELEOP_KEY
GENESIS_TELEOP_HOST
GENESIS_TELEOP_PORT
```

---

### STEP 5 — `src/genesis_quest_teleop/protocol.py`

Implement pure Python dataclasses. No Genesis imports.

Required dataclasses:

```python
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
```

Required parser:

```python
def parse_quest_packet(raw: str | bytes, max_bytes: int) -> QuestStatePacket: ...
```

Validation:

- reject payload bigger than `max_bytes`;
- decode UTF-8 if bytes;
- JSON object only;
- `schema_version == 1` for MVP;
- `session_id` non-empty string;
- `sequence >= 0` integer;
- controller hand key only needs to support `left`/`right`; ignore unknown hands;
- invalid/missing pose becomes `None`, not exception, unless values are non-finite malicious/structurally wrong;
- clamp analog values;
- preserve no arbitrary `extra` dict in MVP.

Keep parser deterministic and dependency-free.

---

### STEP 6 — `src/genesis_quest_teleop/state_store.py`

Implement a lock-protected latest-state register.

Required model:

```python
@dataclass(frozen=True)
class ReceivedQuestState:
    packet: QuestStatePacket
    receive_monotonic_ns: int
    receive_epoch_ms: float
```

Required class:

```python
class LatestQuestStateStore:
    def replace(self, packet: QuestStatePacket) -> bool: ...
    def snapshot(self) -> ReceivedQuestState | None: ...
    def mark_disconnected(self, session_id: str | None = None) -> None: ...
    def is_connected(self) -> bool: ...
```

Rules:

- `replace()` uses `time.monotonic_ns()` for local freshness.
- maintain current `session_id` + last accepted sequence.
- same session: reject sequence `<= last_sequence`.
- new session: reset sequence tracking and accept.
- store exactly one state object.
- no deque, queue, asyncio queue, ROS-like buffer, or history list.
- snapshot can return immutable object safely.
- mark disconnected changes connection flag but may preserve last packet for diagnostics; application must HOLD while disconnected.

---

### STEP 7 — `src/genesis_quest_teleop/diagnostics.py`

Implement lightweight counters only.

Track at minimum:

```text
packets_received
packets_rejected
out_of_order_packets
peer_connections
peer_disconnects
stale_holds
tracking_loss_holds
clutch_engages
clutch_releases
diffik_failures
safety_clamps
control_updates
```

Track latest:

```text
packet_age_ms
packet_rate_hz
active_session_id
last_sequence
```

Expose:

```python
class Diagnostics:
    def increment(self, name: str, amount: int = 1) -> None: ...
    def set_value(self, name: str, value: object) -> None: ...
    def maybe_log(self) -> None: ...
```

Do not build Prometheus/GUI infrastructure.

---

### STEP 8 — `src/genesis_quest_teleop/transport/webrtc_server.py`

This file owns HTTP serving, signaling, aiortc peers, and network thread lifecycle.

Required class:

```python
class WebRTCServer:
    def __init__(self, config: dict, state_store: LatestQuestStateStore, diagnostics: Diagnostics): ...
    def start_in_thread(self) -> None: ...
    def stop(self) -> None: ...
```

Internal methods:

```python
async def _run(self) -> None: ...
async def _handle_index(self, request): ...
async def _handle_static(self, request): ...
async def _handle_offer(self, request): ...
async def _handle_health(self, request): ...
async def _close_peer(self, pc) -> None: ...
def _on_datachannel(self, pc, channel) -> None: ...
def _on_message(self, channel, message) -> None: ...
```

Routes:

```text
GET  /
GET  /quest_client.js
GET  /health
POST /api/offer
```

`/health` returns JSON only:

```json
{"status":"ok"}
```

DataChannel rules:

- accept only label from config, default `quest-teleop`;
- unexpected channel label: close/ignore channel;
- on message: parse, replace state, update diagnostics;
- never block callback on simulation;
- never call Genesis;
- on channel/peer close: mark disconnected;
- keep strong references to active `RTCPeerConnection` objects in a `set`;
- remove/close peers on failed/closed/disconnected states;
- close all peers during `stop()`.

Threading:

- create a dedicated daemon/non-daemon network thread with its own asyncio event loop;
- `start_in_thread()` must return after server startup succeeds or surfaces startup exception;
- provide a thread-safe stop signal;
- do not rely on interpreter teardown to close peers.

HTTPS:

```python
ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain(cert_file, key_file)
```

Use `aiohttp.web.AppRunner` + `TCPSite`.

Do not introduce FastAPI/Uvicorn.

---

### STEP 9 — `web/index.html`

Keep UI minimal.

Required elements:

```text
PC status
WebRTC status
XR status
Start XR button
Stop XR button
small text debug area
```

Load only:

```html
<script src="/quest_client.js"></script>
```

No framework, bundler, npm, React, Three.js, or WebSocket client.

---

### STEP 10 — `web/quest_client.js`

Implement the entire Quest browser runtime.

Required state:

```javascript
let pc = null;
let dataChannel = null;
let xrSession = null;
let xrRefSpace = null;
let sequence = 0;
let sessionId = crypto.randomUUID();
```

Required functions:

```javascript
async function connectWebRTC() {}
async function waitForIceGatheringComplete(pc) {}
async function startXR() {}
async function stopXR() {}
function onXRFrame(time, frame) {}
function controllerToPayload(inputSource, frame) {}
function updateStatus(...) {}
```

`connectWebRTC()`:

1. Close previous peer/channel if any.
2. Generate new `sessionId` and reset sequence.
3. Create `RTCPeerConnection`.
4. Create data channel BEFORE offer.
5. Data channel options: unordered, zero retransmits.
6. Create/set offer.
7. Wait ICE complete.
8. POST SDP to `/api/offer` using same HTTPS origin.
9. Set answer as remote description.
10. Update status on open/close/error.

`startXR()`:

1. Ensure WebRTC connection initialized.
2. Check `navigator.xr`.
3. prefer `immersive-ar` if supported;
4. fallback `immersive-vr`;
5. request `local-floor`;
6. start requestAnimationFrame loop.

No XR WebGL scene is required merely to read controller state unless browser requires an XR-compatible layer. If Meta Quest Browser requires a base layer for stable XR frames, reuse TeleSim's minimal transparent WebGL canvas/layer approach. Do not render 3D content.

`onXRFrame()`:

- schedule next frame first;
- construct packet;
- iterate input sources;
- get `gripSpace` pose;
- fill controller state exactly per protocol;
- if channel open and buffered amount under threshold, send JSON;
- do not enqueue/retry packet.

On XR end:

- clear XR state;
- stop sending;
- keep peer alive or close it cleanly; prefer close so next Start creates a fresh session ID.

---

### STEP 11 — `src/genesis_quest_teleop/input/frames.py`

No Genesis imports except optional type-free helpers; keep numpy/scipy only.

Implement:

```python
R_G_XR = np.array([...])

def normalize_quat_xyzw(q) -> np.ndarray: ...
def normalize_quat_wxyz(q) -> np.ndarray: ...
def xyzw_to_wxyz(q) -> np.ndarray: ...
def wxyz_to_xyzw(q) -> np.ndarray: ...
def map_webxr_position_to_genesis(position_xyz) -> np.ndarray: ...
def map_webxr_quat_to_genesis_wxyz(quat_xyzw) -> np.ndarray: ...
def quat_multiply_wxyz(a, b) -> np.ndarray: ...
def quat_inverse_wxyz(q) -> np.ndarray: ...
def quat_angle_wxyz(a, b) -> float: ...
def slerp_wxyz(a, b, t) -> np.ndarray: ...
```

Use SciPy `Rotation` only for basis conversion and optionally Slerp. Keep quaternion ordering explicit at every boundary.

Unit comments must say `xyzw` or `wxyz`; never call a variable just `quat` when crossing API boundaries.

---

### STEP 12 — `src/genesis_quest_teleop/input/clutch.py`

Implement a pure state machine plus target calculation.

Required dataclasses:

```python
@dataclass
class Pose:
    position: np.ndarray     # xyz Genesis world
    quaternion_wxyz: np.ndarray

@dataclass
class ClutchOutput:
    engaged: bool
    target_pose: Pose | None
    just_engaged: bool
    just_released: bool
    requires_rearm: bool
```

Required class:

```python
class ClutchController:
    def __init__(self, config: dict): ...
    def force_hold(self) -> None: ...
    def update(
        self,
        controller_pose: Pose | None,
        squeeze: float,
        measured_ee_pose: Pose,
    ) -> ClutchOutput: ...
```

Important state machine details:

- initial state disengaged;
- engage threshold and release threshold hysteresis;
- if controller pose missing while engaged -> force hold + `requires_rearm=True`;
- while `requires_rearm`, ignore squeeze until it first drops below release threshold;
- next valid squeeze rising edge may engage;
- on engage: anchor controller + measured EE;
- while engaged: compute relative translation + rotation target;
- on release: preserve last target externally but clear anchors.

Do not perform workspace or step clamping here; that belongs in app/control safety path.

---

### STEP 13 — `src/genesis_quest_teleop/control/diffik.py`

Implement a simulator-facing DiffIK helper but keep robot-specific joint naming outside.

Required class:

```python
class DifferentialIKController:
    def __init__(
        self,
        robot,
        ee_link,
        arm_dofs_idx: list[int],
        config: dict,
    ): ...

    def compute_command(self, target_pos: np.ndarray, target_quat_wxyz: np.ndarray) -> np.ndarray | None: ...
```

Initialization:

- get arm lower/upper DOF limits once;
- convert tensors to numpy;
- preallocate `np.eye(6)`;
- retain last valid q command.

`compute_command()` exact order:

1. read measured EE pos/quaternion;
2. calculate position error;
3. calculate rotation-vector error using Genesis quaternion utilities as in official DiffIK example;
4. build scaled 6D error;
5. read full Jacobian;
6. slice Jacobian columns to arm DOFs;
7. validate shape exactly `(6, len(arm_dofs_idx))`;
8. solve damped least squares;
9. read current arm DOF positions;
10. clamp per-joint delta;
11. clamp to finite joint limits with margin;
12. validate finite output;
13. save and return command;
14. on failure return `None`, never partially valid q.

Do NOT call `robot.control_dofs_position()` inside this class. App owns command application.

---

### STEP 14 — `src/genesis_quest_teleop/robots/base.py`

Define only a small protocol/interface. Do not build a plugin framework.

Required abstract surface:

```python
class RobotAdapter(ABC):
    @property
    def entity(self): ...
    @property
    def ee_link(self): ...
    @property
    def arm_dofs_idx(self) -> list[int]: ...
    @property
    def finger_dofs_idx(self) -> list[int]: ...

    def build(self, scene) -> None: ...
    def initialize_after_scene_build(self) -> None: ...
    def get_ee_pose(self) -> Pose: ...
    def apply_arm_position(self, q_arm: np.ndarray) -> None: ...
    def apply_gripper_trigger(self, trigger: float) -> None: ...
    def hold(self) -> None: ...
```

No Isaac/Lula abstractions.

---

### STEP 15 — `src/genesis_quest_teleop/robots/franka.py`

Implement `FrankaAdapter` using Genesis built-in MJCF.

`build(scene)`:

```python
self._robot = scene.add_entity(
    gs.morphs.MJCF(file=config["mjcf_file"])
)
```

Do not call `scene.build()` inside adapter.

`initialize_after_scene_build()`:

1. resolve arm DOF local indices from configured joint names:

```python
idx = robot.get_joint(name).dofs_idx_local[0]
```

2. resolve finger indices similarly;
3. resolve EE link with `robot.get_link(end_effector_link)`;
4. set configured home qpos with `set_qpos(..., zero_velocity=True)` if qpos length matches robot generalized coordinates;
5. apply matching position targets so PD does not pull toward stale defaults;
6. initialize gripper to open;
7. do not override arm KP/KV unless explicitly present in config;
8. expose current measured EE pose.

`apply_gripper_trigger()`:

- map trigger continuously to finger target;
- store last target;
- issue `control_dofs_position` for finger DOFs.

`hold()`:

- do not freeze state via `set_qpos`;
- ensure last desired arm/finger positions remain the position-control targets.

---

### STEP 16 — `src/genesis_quest_teleop/app.py`

This is the orchestration file and the ONLY place combining transport state with Genesis control.

Required class:

```python
class GenesisTeleopApp:
    def __init__(self, config: dict): ...
    def setup(self) -> None: ...
    def run(self) -> None: ...
    def shutdown(self) -> None: ...
```

`setup()` exact sequence:

1. create state store + diagnostics;
2. start WebRTC server thread;
3. initialize Genesis backend;
4. create scene;
5. add plane;
6. optionally add one small rigid cube for grasp validation;
7. create Franka adapter and add entity;
8. `scene.build()`;
9. robot adapter post-build init;
10. execute configured warmup steps;
11. create DiffIK controller;
12. create clutch controller;
13. capture current EE pose as initial target;
14. create debug target frame if enabled;
15. calculate control-step divisor.

Recommended scene:

```python
gs.init(backend=gs.gpu if backend == "gpu" else gs.cpu)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=dt),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(1.4, -1.4, 1.1),
        camera_lookat=(0.45, 0.0, 0.35),
        camera_fov=40,
    ),
    show_viewer=show_viewer,
    show_FPS=False,
)
```

Add plane:

```python
scene.add_entity(gs.morphs.Plane())
```

Optional MVP object:

```python
scene.add_entity(
    gs.morphs.Box(
        pos=(0.55, 0.0, 0.04),
        size=(0.07, 0.07, 0.08),
    )
)
```

Do not add Nyx, camera sensors, cloth, FEM, IPC coupling, or dataset recording in the first milestone.

### 16.1 Main loop

Pseudo-code to implement closely:

```python
step_idx = 0
while viewer_alive_or_headless_running:
    if step_idx % control_divisor == 0:
        state = state_store.snapshot()
        now_ns = time.monotonic_ns()

        if state is None or not state_store.is_connected():
            enter_hold("disconnected")
        elif packet_age_ms(now_ns, state) > stale_timeout_ms:
            enter_hold("stale")
        else:
            ctrl = state.packet.controllers.get(active_hand)
            if ctrl is None or ctrl.position_xyz is None or ctrl.orientation_xyzw is None:
                enter_hold("tracking")
            else:
                controller_pose = map_controller_pose(ctrl)
                measured_ee_pose = robot.get_ee_pose()
                clutch_out = clutch.update(controller_pose, ctrl.squeeze, measured_ee_pose)

                robot.apply_gripper_trigger(ctrl.trigger)

                if clutch_out.engaged and clutch_out.target_pose is not None:
                    target = safety_limit_target(clutch_out.target_pose)
                    q_cmd = diffik.compute_command(target.position, target.quaternion_wxyz)
                    if q_cmd is not None:
                        robot.apply_arm_position(q_cmd)
                    else:
                        robot.hold()
                else:
                    robot.hold()

                update_debug_target_frame(last_valid_target)

        diagnostics.maybe_log()

    scene.step()
    step_idx += 1
```

### 16.2 HOLD semantics

Implement one helper inside app:

```python
def _enter_hold(reason: str) -> None:
```

Behavior:

- `clutch.force_hold()` for disconnect/stale/tracking reasons;
- `robot.hold()`;
- do not change the target pose;
- do not run DiffIK on stale input;
- increment corresponding diagnostic only on transition into reason, not every physics frame.

### 16.3 Target safety limiting

Implement app-local/private helpers or compact dedicated functions, not another module unless needed:

```python
_clamp_workspace(...)
_limit_translation_step(...)
_limit_rotation_step(...)
```

Rotation step limiting:

- compute angular distance between previous target and candidate;
- if angle <= max -> use candidate;
- else slerp from previous toward candidate by `max_angle / angle`.

Save last valid limited target.

### 16.4 Debug frame

Follow Genesis IPC teleop example:

```python
import genesis.utils.geom as gu

frame = scene.draw_debug_frame(
    T=gu.trans_quat_to_T(target_pos, target_quat_wxyz),
    axis_length=0.12,
    origin_size=0.008,
    axis_radius=0.005,
)
```

Update rather than recreate every frame:

```python
scene.update_debug_objects((frame,), (pose_matrix,))
```

If debug API differs in local Genesis, adapt to local 1.3.x API; do not modify Genesis.

---

### STEP 17 — `src/genesis_quest_teleop/main.py`

Implement CLI.

Arguments:

```text
--config PATH       default config/default.yaml
--cpu               override backend to cpu
--headless          disable viewer
--log-level LEVEL   optional
```

Flow:

```python
def main():
    config = load_config(...)
    apply CLI overrides
    app = GenesisTeleopApp(config)
    try:
        app.setup()
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()
```

Do not hide stack traces for setup failures.

---

### STEP 18 — `scripts/check.sh`

Implement LLM-friendly cheap verification only.

```bash
#!/usr/bin/env bash
set -euo pipefail
uv sync --locked
uv pip check
nvidia-smi >/dev/null
uv run python -m compileall -q src
uv run python - <<'PY'
import aiohttp
import aiortc
import genesis
import numpy
import scipy
import torch
import yaml
from genesis_quest_teleop.protocol import parse_quest_packet
from genesis_quest_teleop.input.frames import map_webxr_position_to_genesis

assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.get_device_capability(0) == (12, 0), torch.cuda.get_device_capability(0)
x = torch.ones(256, device="cuda")
y = x * 2.0
torch.cuda.synchronize()
assert y.is_cuda
print("imports-and-torch-cuda: ok")
PY
uv run python - <<'PY'
import genesis as gs

gs.init(backend=gs.cuda, logging_level="warning")
assert gs.backend == gs.cuda, f"Genesis CUDA backend failed: {gs.backend}"
scene = gs.Scene(show_viewer=False)
scene.add_entity(gs.morphs.Plane())
scene.build()
for _ in range(10):
    scene.step()
print("genesis-cuda-smoke: ok")
PY
uv run ruff check src
```

`ruff` is mandatory as a dev dependency. `check.sh` must fail if the lockfile is stale, dependencies are inconsistent, runtime imports fail, source compilation fails, or ruff reports an error.

---

### STEP 19 — `README.md`

Keep README operational and short.

Include only:

1. project purpose;
2. prerequisites (`uv`, NVIDIA driver + RTX 5070 Ti visible in `nvidia-smi`, local `../genesis-world`, Meta Quest Browser);
3. uv environment creation;
4. dependency sync from `pyproject.toml`/`uv.lock`, including PyTorch from the explicit `cu128` index;
5. dependency verification, PyTorch `sm_120` kernel smoke test, and Genesis CUDA scene smoke test;
6. configure cert paths;
7. run command;
8. Quest URL;
9. controls;
10. troubleshooting for uv/Genesis import/HTTPS/WebRTC/Quest tracking.

Example setup:

```bash
cd /home/saurabh/Developement/genesis-quest-teleop
uv python install 3.11
uv venv --python 3.11 .venv
uv sync
uv pip check
```

Optional shell activation:

```bash
source .venv/bin/activate
```

Run without depending on activation:

```bash
uv run genesis-quest-teleop --config config/default.yaml
```

Controls:

```text
Right squeeze/grip: clutch arm motion
Right trigger: gripper
Release squeeze: hold arm
```

---

## 15. IMPLEMENTATION DETAILS THAT MUST NOT BE MISSED

### 15.1 Main-thread rule

Genesis viewer and simulation loop remain on main thread.

Allowed:

```text
network thread -> parse packet -> state store
main thread -> read state -> Genesis APIs
```

Forbidden:

```text
aiortc callback -> robot.control_dofs_position(...)
aiortc callback -> scene.step()
aiortc callback -> viewer update
```

### 15.2 Latest-state rule

Do not port TeleSim's forwarding queue to this local WebRTC design.

For live control:

```text
new pose replaces old pose
```

No FIFO control packet queue.

### 15.3 No prediction in MVP

TeleSim has transport prediction tuning. Do not copy it initially.

Reason for implementation order:

```text
measure raw WebRTC + Genesis control latency first
then add prediction only if measured need exists
```

### 15.4 No camera workload

Do not add Genesis cameras or Nyx render capture until baseline controller-to-robot latency is proven.

This prevents repeating TeleSim's camera-induced teleoperation-latency coupling.

### 15.5 No direct state teleport for live arm

Forbidden live control:

```python
robot.set_qpos(...)
robot.set_dofs_position(...)
```

Allowed for initialization/reset only.

Live arm uses:

```python
robot.control_dofs_position(...)
```

### 15.6 Use measured EE for every clutch engagement

Do not anchor to `last_target_pose` because physical simulation/contact may differ from command target.

Always anchor to:

```python
ee_link.get_pos()
ee_link.get_quat()
```

This is a direct carry-over of the robust TeleSim intervention/clutch principle.

---

## 16. MANUAL VALIDATION SEQUENCE

Codex should implement and validate in this order. Do not attempt full Quest teleop before each lower layer works.

### VALIDATION A — Python/package

```bash
cd /home/saurabh/Developement/genesis-quest-teleop
source .venv/bin/activate
./scripts/check.sh
```

Pass condition:

```text
all Python files compile
basic imports succeed
```

### VALIDATION B — Genesis-only control

Temporarily provide a debug mode or small internal target motion, without Quest/network.

Goal:

- Franka scene starts;
- Jacobian exists;
- DiffIK moves EE toward a slowly changing target;
- gripper commands work;
- viewer remains stable.

Do not leave a separate production demo architecture; a `--debug-target` flag or small temporary code path is enough.

Pass condition:

```text
no NaNs
no exploding robot
EE follows target frame
arm DOFs stay within limits
```

### VALIDATION C — HTTPS/Web page

With simulator running:

```bash
curl -k https://127.0.0.1:8443/health
```

Expected:

```json
{"status":"ok"}
```

Open page from desktop browser for signaling sanity.

### VALIDATION D — WebRTC without XR motion

Connect browser and confirm:

```text
peer_connections increments
data channel opens
server receives packet sequence
packet rate is nonzero when XR session active
```

### VALIDATION E — Frame mapping

Physically move right Quest controller with clutch released.

Check mapped debug logs, not robot:

```text
move controller physically forward -> mapped Genesis +X increases
move controller physically right   -> mapped Genesis Y decreases
move controller physically up      -> mapped Genesis +Z increases
```

If this fails, fix only `frames.py`; do not compensate by changing robot workspace axes elsewhere.

### VALIDATION F — Zero-jump clutch

1. Arm stationary.
2. Move physical controller to arbitrary pose while squeeze released.
3. Press squeeze without moving controller.
4. Observe robot.

Pass condition:

```text
no visible EE jump at engagement
```

Then move controller 5-10 cm and confirm robot follows relative motion.

### VALIDATION G — Release/re-engage

1. Move arm with clutch.
2. Release squeeze.
3. Move physical controller far away.
4. Re-engage squeeze while stationary.

Pass condition:

```text
robot remains in current location at re-engagement
subsequent relative motion resumes naturally
```

### VALIDATION H — stale/disconnect safety

During motion:

- end XR session OR disable headset Wi-Fi OR close browser.

Pass condition:

```text
within stale_timeout_ms + one control period, new arm motion stops
no queued old packets continue driving the arm
reconnect requires fresh clutch engagement
```

### VALIDATION I — gripper

With arm clutch released:

- vary right trigger from 0 to 1.

Pass condition:

```text
fingers move smoothly from open to closed
no arm movement caused by trigger
```

### VALIDATION J — basic manipulation

Use cube in scene.

Pass condition:

```text
operator can approach cube, close gripper, lift/reposition it at interactive latency
```

Do not tune contact-rich/deformable behavior until this rigid-body baseline passes.

---

## 17. ACCEPTANCE CRITERIA

MVP is complete only when all are true:

```text
[ ] New standalone sibling repository; vendor repos untouched.
[ ] `.python-version` pins Python 3.11.
[ ] `.venv` was created with `uv venv --python 3.11`.
[ ] `uv.lock` exists and is committed.
[ ] `uv sync` succeeds from a clean environment.
[ ] `uv pip check` reports no dependency conflicts.
[ ] `nvidia-smi` detects the NVIDIA GeForce RTX 5070 Ti.
[ ] `torch` resolves from the official `pytorch-cu128` index, not CPU-only PyPI wheels.
[ ] `torch.cuda.is_available()` is true.
[ ] `torch.cuda.get_device_capability(0)` returns `(12, 0)` for the RTX 5070 Ti.
[ ] A real CUDA tensor/matmul kernel executes and synchronizes successfully.
[ ] Local `../genesis-world` resolves as the editable `genesis-world` dependency.
[ ] Runtime imports (`torch`, `genesis`, `aiortc`, `aiohttp`, `numpy`, `scipy`, `yaml`) succeed via `uv run`.
[ ] `gs.init(backend=gs.cuda)` resolves `gs.backend == gs.cuda`.
[ ] A headless Genesis scene builds and steps for at least 10 steps on CUDA without CPU fallback.
[ ] HTTPS page opens on Quest.
[ ] WebRTC DataChannel opens without WebSocket controller transport.
[ ] Quest packets include sequence/session/controller poses/buttons.
[ ] Server stores only latest packet.
[ ] Both controllers are parsed.
[ ] Configured right controller maps to Franka.
[ ] WebXR->Genesis translation mapping is correct.
[ ] WebXR xyzw -> Genesis wxyz orientation mapping is correct.
[ ] Squeeze clutch has hysteresis.
[ ] Clutch engagement is zero-jump from measured EE pose.
[ ] Tracking loss forces hold/rearm.
[ ] Packet stale timeout forces hold/rearm.
[ ] DiffIK uses Genesis Jacobian and damped least squares.
[ ] Jacobian is sliced to arm DOFs.
[ ] Live control uses control_dofs_position, not state teleport.
[ ] Joint delta and joint limit safety are active.
[ ] Workspace and target step clamps are active.
[ ] Trigger controls Franka gripper.
[ ] Debug target frame updates without being recreated every frame.
[ ] Network thread never calls Genesis APIs.
[ ] `python -m compileall -q src` succeeds.
[ ] End-to-end rigid cube manipulation works interactively.
```

---

## 18. DO NOT IMPLEMENT YET

Explicit non-goals for this pass:

```text
TeleSim integration
ROS/ROS2
Isaac Sim
Lula IK
OpenArm asset integration
bimanual Genesis robot control
policy inference
DAgger/interventions
recording datasets
camera capture
Nyx camera streaming
headset video from Genesis
WebRTC video tracks
TURN server
remote internet teleoperation
pose prediction
jitter buffer
shared-memory recorder
replay/deferred mode
cloth/FEM/IPC task integration
force feedback/haptics
hand tracking
```

These are future layers. Do not add placeholders that complicate MVP unless the existing architecture naturally exposes the extension point.

---

## 19. PHASE 2 — OPENARM/BIMANUAL EXTENSION AFTER MVP PASSES

Do not start this phase until the Franka acceptance checklist passes.

The architecture must make this possible without changing WebRTC, protocol, state store, or WebXR frame mapping.

### 19.1 Add robot config, not simulator fork

Create later:

```text
config/openarm.yaml
src/genesis_quest_teleop/robots/openarm.py
```

Only after the Genesis-compatible OpenArm URDF/MJCF/USD asset and exact joint/link names are known.

Do not guess these names.

### 19.2 Bimanual behavior

Create one per-hand control chain:

```text
left Quest  -> left ClutchController  -> left target -> left arm DiffIK
right Quest -> right ClutchController -> right target -> right arm DiffIK
```

Each hand gets independent:

```text
anchor
clutch state
rearm state
workspace
joint indices
EE link
trigger/gripper mapping
```

Both arms read the SAME latest Quest packet snapshot per control tick.

### 19.3 Bimanual Jacobian strategy

Prefer independent arm Jacobian solves if the robot kinematic chains are separable by DOF subset.

Because Genesis `get_jacobian()` returns all entity DOFs:

```python
J_left  = J_left_full[:, left_arm_dofs]
J_right = J_right_full[:, right_arm_dofs]
```

Do not add Python thread pools for Genesis Jacobian calls until profiling proves benefit and thread safety is verified. Keep simulator API access on main thread.

### 19.4 Keep transport unchanged

The current packet already contains both controllers. No WebRTC protocol redesign should be needed.

---

## 20. PHASE 3 — DATA COLLECTION ONLY AFTER REALTIME CONTROL IS STABLE

When later adding robot-learning dataset collection, avoid coupling render/camera workload directly to human control timing.

Preferred future architecture:

```text
Realtime control loop
    -> exact applied action + simulator state snapshots
    -> asynchronous recorder/camera pipeline
```

Do not reintroduce a design where camera encoding blocks controller consumption.

This phase is intentionally outside current implementation.

---

## 21. CODEX EXECUTION ORDER SUMMARY

Follow this order exactly:

```text
1. Inspect local Genesis + TeleSim reference files.
2. Create standalone project skeleton.
3. Create `.python-version` + `pyproject.toml`.
4. Run `uv python install 3.11` and `uv venv --python 3.11 .venv`.
5. Run `uv lock`, `uv sync`, and `uv pip check`.
6. Run `nvidia-smi`, verify PyTorch uses the `cu128` CUDA build, require RTX 5070 Ti compute capability `(12, 0)`, execute a real CUDA kernel, then run the Genesis `gs.cuda` 10-step scene smoke test. Do not write runtime integration code until ALL environment/GPU checks pass.
7. Create config loader.
8. Implement protocol parser.
9. Implement latest-state store.
10. Implement diagnostics.
11. Implement HTTPS + WebRTC server.
12. Implement minimal Quest web UI + WebXR DataChannel client.
13. Implement WebXR->Genesis coordinate mapping.
14. Implement clutch state machine.
15. Implement Genesis DiffIK helper.
16. Implement robot base interface.
17. Implement Franka adapter.
18. Implement Genesis app/main-thread loop.
19. Implement CLI.
20. Implement compile/import/GPU check script.
21. Validate Genesis-only DiffIK.
22. Validate HTTPS/WebRTC.
23. Validate coordinate directions.
24. Validate clutch zero-jump/rearm.
25. Validate stale/disconnect HOLD.
26. Validate gripper.
27. Validate cube manipulation.
28. Stop. Do not expand scope until MVP passes.
```

---

## 22. REFERENCE LINKS

GPU compatibility references used for environment requirements:

```text
https://developer.nvidia.com/cuda/gpus
https://docs.astral.sh/uv/guides/integration/pytorch/
https://download.pytorch.org/whl/cu128
https://genesis-world.readthedocs.io/en/latest/user_guide/configuration/initialization.html
```

Genesis World:

```text
https://github.com/Genesis-Embodied-AI/genesis-world
https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/rigid/diffik_controller.py
https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/rigid/ik_franka.py
https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/tutorials/control_your_robot.py
https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/ipc/ipc_robot_cloth_teleop.py
https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/genesis/engine/entities/rigid_entity/rigid_entity.py
```

TeleSim references:

```text
https://github.com/AiSaurabhPatil/telesim/tree/clutch_teleop
https://github.com/AiSaurabhPatil/telesim/blob/clutch_teleop/web/webxr_streamer.html
https://github.com/AiSaurabhPatil/telesim/blob/clutch_teleop/src/launch/webxr_bridge.py
https://github.com/AiSaurabhPatil/telesim/blob/clutch_teleop/src/quest_ingress/message_types.py
https://github.com/AiSaurabhPatil/telesim/blob/clutch_teleop/config/config.yaml
```

---

## 23. FINAL IMPLEMENTATION PRINCIPLE

The system is deliberately split into three timing domains:

```text
Quest XR sampling       -> fast, packet producing
WebRTC/network thread   -> fast, latest-state replacement only
Genesis main loop       -> deterministic physics/control consumer
```

Never allow one domain to queue work into the next.

The core invariant is:

```text
At every Genesis control tick, use the newest valid Quest pose available now;
otherwise hold the robot.
```

This invariant has priority over completeness, guaranteed packet delivery, smoothing, recording, rendering quality, or future abstractions.
