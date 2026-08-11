# Genesis Quest Teleop

Low-latency Quest controller input for a Genesis Franka Panda: WebXR → unreliable WebRTC DataChannel → latest-state control → Genesis DiffIK.

Prerequisites: `uv`, a driver exposing the RTX 5070 Ti in `nvidia-smi`, the local `../genesis-world` checkout, a trusted HTTPS certificate, and Meta Quest Browser.

```bash
cd /home/saurabh/Developement/genesis-quest-teleop
uv python install 3.11
uv venv --python 3.11 .venv
uv sync
./scripts/check.sh
```

The project includes a locally generated self-signed certificate at `certs/cert.pem`
and private key at `certs/key.pem`; both are gitignored. The certificate includes
the current workstation LAN address (`192.168.29.64`). Regenerate it if that address
changes, or set `GENESIS_TELEOP_CERT` and `GENESIS_TELEOP_KEY` to use your own trusted
certificate. Trust the certificate in Quest Browser before opening the service.

Then run:

```bash
uv run genesis-quest-teleop --config config/default.yaml
```

For a persistent Quest connection while iterating on the Genesis scene, run the
transport and simulator in separate terminals. Keep the first command running:

```bash
uv run genesis-quest-ingress --config config/default.yaml
```

Open the Quest page and start XR once. Then start, stop, or restart only the
Genesis process as often as needed:

```bash
uv run genesis-quest-teleop --config config/default.yaml --external-ingress
```

The processes exchange only the latest validated controller sample over
localhost UDP (`127.0.0.1:8765`). Packets sent while Genesis is restarting are
dropped, and the first fresh packet after restart resumes the local input stream.

Open `https://<PC_LAN_IP>:8443` in Quest Browser. Right squeeze is the zero-jump arm clutch; right trigger controls the gripper; releasing squeeze holds the arm.

The default scene includes a green, dynamic 5 cm grasp cube resting at `(0.65,
0.0, 0.025)`. Its position, size, friction, colour, and enabled state are in
`scene.grasp_cube` in `config/default.yaml`; set `enabled: false` to remove it.
The two Panda finger collision links use `gripper.finger_friction` (default
`2.0`) to improve grasp retention.

If Genesis cannot import, verify the sibling checkout is present and run `uv sync`. For HTTPS/WebRTC failures, trust the certificate on the headset and ensure both devices are on the same LAN. Tracking loss, disconnect, and packets older than 150 ms always hold the robot; release and re-press squeeze after reconnecting.
