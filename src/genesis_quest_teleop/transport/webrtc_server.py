from __future__ import annotations

import asyncio
import logging
import ssl
import threading
from pathlib import Path

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

from ..protocol import parse_quest_packet

LOGGER = logging.getLogger(__name__)


class WebRTCServer:
    def __init__(self, config, state_store, diagnostics, state_publisher=None):
        self.config = config
        self.store = state_store
        self.diag = diagnostics
        self.state_publisher = state_publisher
        self.peers = set()
        self.loop = None
        self.thread = None
        self.runner = None
        self.ready = threading.Event()
        self.error = None

    def start_in_thread(self):
        self.thread = threading.Thread(
            target=self._thread_main, name="webrtc-network", daemon=True
        )
        self.thread.start()
        self.ready.wait()
        if self.error:
            raise self.error

    def _thread_main(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._run())
            self.ready.set()
            self.loop.run_forever()
        except BaseException as exc:
            self.error = exc
            self.ready.set()
        finally:
            self.loop.run_until_complete(self._shutdown()) if self.runner else None
            self.loop.close()

    async def _run(self):
        app = web.Application(
            client_max_size=self.config["webrtc"]["max_message_bytes"]
        )
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/quest_client.js", self._handle_static)
        app.router.add_get("/health", self._handle_health)
        app.router.add_post("/api/offer", self._handle_offer)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        tls = self.config["tls"]
        if not Path(tls["cert_file"]).is_file() or not Path(tls["key_file"]).is_file():
            raise FileNotFoundError(
                "HTTPS certificate/key not found; set GENESIS_TELEOP_CERT and GENESIS_TELEOP_KEY"
            )
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(tls["cert_file"], tls["key_file"])
        site = web.TCPSite(
            self.runner,
            self.config["server"]["host"],
            self.config["server"]["port"],
            ssl_context=context,
        )
        await site.start()
        logging.info(
            "HTTPS/WebRTC listening on %s:%s",
            self.config["server"]["host"],
            self.config["server"]["port"],
        )

    async def _handle_index(self, request):
        return web.FileResponse(
            Path(self.config["web"]["directory"]) / "index.html",
            headers={"Cache-Control": "no-store"},
        )

    async def _handle_static(self, request):
        return web.FileResponse(
            Path(self.config["web"]["directory"]) / "quest_client.js",
            headers={"Cache-Control": "no-store"},
        )

    async def _handle_health(self, request):
        return web.json_response({"status": "ok"})

    async def _handle_offer(self, request):
        data = await request.json()
        offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
        pc = RTCPeerConnection()
        self.peers.add(pc)
        self.diag.increment("peer_connections")

        @pc.on("datachannel")
        def channel(ch):
            self._on_datachannel(pc, ch)

        @pc.on("connectionstatechange")
        async def state():
            LOGGER.info("WebRTC peer state: %s", pc.connectionState)
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await self._close_peer(pc)

        await pc.setRemoteDescription(offer)
        await pc.setLocalDescription(await pc.createAnswer())
        return web.json_response(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        )

    def _on_datachannel(self, pc, channel):
        if channel.label != self.config["webrtc"]["data_channel_label"]:
            LOGGER.warning("Rejecting unexpected data channel: %s", channel.label)
            channel.close()
            return

        LOGGER.info("WebRTC DataChannel received: %s", channel.label)
        if channel.readyState == "open":
            LOGGER.info("WebRTC DataChannel already open: %s", channel.label)

        @channel.on("open")
        def open_channel():
            LOGGER.info("WebRTC DataChannel open: %s", channel.label)

        @channel.on("message")
        def message(value):
            self._on_message(channel, value)

        @channel.on("close")
        def close():
            LOGGER.info("WebRTC DataChannel closed: %s", channel.label)
            self.store.mark_disconnected()
            if self.state_publisher:
                self.state_publisher.mark_disconnected()
            self.diag.increment("peer_disconnects")

    def _on_message(self, channel, message):
        try:
            packet = parse_quest_packet(
                message, self.config["webrtc"]["max_message_bytes"]
            )
            if self.store.replace(packet):
                if self.state_publisher:
                    self.state_publisher.publish(message)
                self.diag.increment("packets_received")
                self.diag.set_value("active_session_id", packet.session_id)
                self.diag.set_value("last_sequence", packet.sequence)
            else:
                self.diag.increment("out_of_order_packets")
        except (ValueError, UnicodeDecodeError):
            self.diag.increment("packets_rejected")

    async def _close_peer(self, pc):
        if pc in self.peers:
            self.peers.remove(pc)
            self.store.mark_disconnected()
            if self.state_publisher:
                self.state_publisher.mark_disconnected()
            self.diag.increment("peer_disconnects")
            await pc.close()

    async def _shutdown(self):
        await asyncio.gather(
            *(self._close_peer(pc) for pc in list(self.peers)), return_exceptions=True
        )
        await self.runner.cleanup()

    def stop(self):
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=5)
