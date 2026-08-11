let pc = null;
let dataChannel = null;
let xrSession = null;
let xrRefSpace = null;
let xrGl = null;
let xrGlLayer = null;
let sequence = 0;
let sessionId = crypto.randomUUID();

const byId = (id) => document.getElementById(id);

function updateStatus(id, text) {
  byId(id).textContent = text;
  byId("debug").textContent = [
    `session: ${sessionId}`,
    `sequence: ${sequence}`,
    `peer: ${pc?.connectionState ?? "none"}`,
    `ICE: ${pc?.iceConnectionState ?? "none"}`,
    `data channel: ${dataChannel?.readyState ?? "none"}`,
  ].join("\n");
}

function initializeWebGL() {
  const canvas = byId("xr-canvas");
  xrGl = canvas.getContext("webgl2", {
    xrCompatible: true,
    alpha: true,
  });
  if (!xrGl) {
    xrGl = canvas.getContext("webgl", {
      xrCompatible: true,
      alpha: true,
    });
  }
  if (!xrGl) {
    throw new Error("An XR-compatible WebGL context is unavailable");
  }
}

async function waitForIceGatheringComplete(peer) {
  if (peer.iceGatheringState === "complete") {
    return;
  }

  await new Promise((resolve) => {
    const onStateChange = () => {
      if (peer.iceGatheringState === "complete") {
        peer.removeEventListener("icegatheringstatechange", onStateChange);
        resolve();
      }
    };
    peer.addEventListener("icegatheringstatechange", onStateChange);
  });
}

async function connectWebRTC() {
  if (pc) {
    pc.close();
  }

  sessionId = crypto.randomUUID();
  sequence = 0;
  updateStatus("rtc", "connecting");

  pc = new RTCPeerConnection({ iceServers: [] });
  pc.onconnectionstatechange = () => updateStatus("rtc", pc.connectionState);
  pc.oniceconnectionstatechange = () => updateStatus("rtc", `ICE ${pc.iceConnectionState}`);
  pc.onicecandidateerror = (event) =>
    updateStatus("rtc", `ICE error ${event.errorCode}: ${event.errorText}`);
  dataChannel = pc.createDataChannel("quest-teleop", {
    ordered: false,
    maxRetransmits: 0,
  });

  dataChannel.onopen = () => updateStatus("rtc", "open");
  dataChannel.onclose = () => updateStatus("rtc", "closed");
  dataChannel.onerror = () => updateStatus("rtc", "error");

  await pc.setLocalDescription(await pc.createOffer());
  await waitForIceGatheringComplete(pc);

  const response = await fetch("/api/offer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(pc.localDescription),
  });
  if (!response.ok) {
    throw new Error(`offer failed: ${response.status}`);
  }

  await pc.setRemoteDescription(await response.json());
  await waitForDataChannelOpen(dataChannel);
}

async function waitForDataChannelOpen(channel) {
  if (channel.readyState === "open") {
    return;
  }

  await new Promise((resolve, reject) => {
    const cleanUp = () => {
      channel.removeEventListener("open", onOpen);
      channel.removeEventListener("close", onClose);
      channel.removeEventListener("error", onError);
    };
    const onOpen = () => {
      cleanUp();
      resolve();
    };
    const onClose = () => {
      cleanUp();
      reject(new Error("WebRTC DataChannel closed before opening"));
    };
    const onError = () => {
      cleanUp();
      reject(new Error("WebRTC DataChannel failed to open"));
    };

    channel.addEventListener("open", onOpen);
    channel.addEventListener("close", onClose);
    channel.addEventListener("error", onError);
  });
}

function buttonValue(gamepad, index) {
  const button = gamepad.buttons[index];
  return button ? button.value || (button.pressed ? 1 : 0) : 0;
}

function controllerToPayload(inputSource, frame) {
  const gamepad = inputSource.gamepad;
  const pose = inputSource.gripSpace
    ? frame.getPose(inputSource.gripSpace, xrRefSpace)
    : null;

  if (!pose || !gamepad) {
    return {
      position: null,
      orientation: null,
      trigger: 0,
      squeeze: 0,
      thumbstick_x: 0,
      thumbstick_y: 0,
      button_a_x: false,
      button_b_y: false,
      thumbstick_click: false,
    };
  }

  const { position, orientation } = pose.transform;
  return {
    position: { x: position.x, y: position.y, z: position.z },
    orientation: {
      x: orientation.x,
      y: orientation.y,
      z: orientation.z,
      w: orientation.w,
    },
    trigger: buttonValue(gamepad, 0),
    squeeze: buttonValue(gamepad, 1),
    thumbstick_x: gamepad.axes[2] || 0,
    thumbstick_y: gamepad.axes[3] || 0,
    thumbstick_click: Boolean(gamepad.buttons[3]?.pressed),
    button_a_x: Boolean(gamepad.buttons[4]?.pressed),
    button_b_y: Boolean(gamepad.buttons[5]?.pressed),
  };
}

function onXRFrame(time, frame) {
  if (!xrSession) {
    return;
  }
  xrSession.requestAnimationFrame(onXRFrame);

  if (xrGlLayer) {
    xrGl.bindFramebuffer(xrGl.FRAMEBUFFER, xrGlLayer.framebuffer);
    xrGl.clearColor(0, 0, 0, 0);
    xrGl.clear(xrGl.COLOR_BUFFER_BIT | xrGl.DEPTH_BUFFER_BIT);
  }

  const controllers = {};
  for (const inputSource of xrSession.inputSources) {
    if (
      (inputSource.handedness === "left" || inputSource.handedness === "right") &&
      inputSource.gamepad
    ) {
      controllers[inputSource.handedness] = controllerToPayload(inputSource, frame);
    }
  }

  if (dataChannel?.readyState !== "open" || dataChannel.bufferedAmount > 65536) {
    return;
  }

  dataChannel.send(
    JSON.stringify({
      schema_version: 1,
      session_id: sessionId,
      sequence: sequence++,
      timestamp: time,
      client_epoch_ms: Date.now(),
      controllers,
    }),
  );
}

async function startXR() {
  try {
    if (!navigator.xr) {
      throw new Error("WebXR unavailable");
    }
    initializeWebGL();

    // This must be the first asynchronous browser action after the button click.
    // Quest Browser may reject requestSession after signaling awaits because the
    // transient user activation has expired.
    let mode = "immersive-ar";
    try {
      xrSession = await navigator.xr.requestSession(mode, {
        requiredFeatures: ["local-floor"],
      });
    } catch (arError) {
      mode = "immersive-vr";
      xrSession = await navigator.xr.requestSession(mode, {
        requiredFeatures: ["local-floor"],
      });
    }

    xrGlLayer = new XRWebGLLayer(xrSession, xrGl);
    await xrSession.updateRenderState({ baseLayer: xrGlLayer });
    xrRefSpace = await xrSession.requestReferenceSpace("local-floor");

    xrSession.addEventListener("end", () => {
      xrSession = null;
      xrRefSpace = null;
      xrGlLayer = null;
      updateStatus("xr", "ended");
      if (pc) {
        pc.close();
      }
    });

    updateStatus("xr", `${mode}; connecting WebRTC`);
    if (!pc || dataChannel?.readyState !== "open") {
      await connectWebRTC();
    }
    updateStatus("xr", `${mode}; streaming controller state`);
    xrSession.requestAnimationFrame(onXRFrame);
  } catch (error) {
    if (xrSession) {
      await xrSession.end();
    }
    updateStatus("xr", error.message);
  }
}

async function stopXR() {
  if (xrSession) {
    await xrSession.end();
  }
}

byId("start").onclick = startXR;
byId("stop").onclick = stopXR;
updateStatus("pc", "page loaded");
