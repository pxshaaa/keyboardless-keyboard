# WideCam — iPhone ultra-wide over WiFi to the Mac. FROZEN protocol.

Goal: stream the iPhone 15 Pro's ULTRA-WIDE (0.5x) camera to the Mac wirelessly,
so `phase0` preview/recorder can use it as `--backend net`. No cable, ever.

## Transport: MJPEG over HTTP (plain, no TLS), Bonjour-advertised
- Service type `_widecam._tcp`, instance name "WideCam", TXT record: `v=1`,
  `w=<width>`, `h=<height>`, `fps=<fps>`.
- Port: 8080 (fall back to any free port and advertise it in Bonjour).
- Also print the URL on the phone screen (e.g. `http://192.168.1.23:8080/stream`)
  so the Mac side works even if mDNS is blocked.

## GET /stream
`Content-Type: multipart/x-mixed-replace; boundary=frame`
Each part:
```
--frame\r\n
Content-Type: image/jpeg\r\n
Content-Length: <bytes>\r\n
X-Seq: <monotonic int, starts at 0>\r\n
X-Timestamp: <double seconds, phone clock CACurrentMediaTime() at CAPTURE, not at send>\r\n
X-Width: <int>\r\n
X-Height: <int>\r\n
\r\n
<jpeg bytes>\r\n
```
Frames must never be reordered. If the encoder falls behind, DROP frames (skip
seq numbers) rather than queue them — latency matters more than completeness.

## GET /status  -> application/json
`{"lens":"ultrawide","width":1280,"height":720,"fps":60.0,"exposure":"auto|locked",
  "clients":1,"seq":12345,"dropped":7,"battery":0.82}`

## POST /control  (query params, any subset)
- `exposure=lock|auto`  — lock current exposure (AVCaptureDevice exposureMode .locked)
- `fps=30|60`
- `preset=720p|1080p`   — 1280x720 or 1920x1080. Default 720p@60.
- `quality=0.3..0.95`   — JPEG quality, default 0.7
Returns the /status JSON after applying.

## iOS app behaviour
- Lens: `AVCaptureDevice.default(.builtInUltraWideCamera, for: .video, position: .back)`.
  Fall back to the main lens ONLY with a loud on-screen banner saying so.
- Session preset: use the format matching preset+fps; `videoOrientation` landscape;
  `alwaysDiscardsLateVideoFrames = true`.
- JPEG encode on a dedicated serial queue; if a frame arrives while one is still
  encoding, drop the new one and increment `dropped`.
- `UIApplication.shared.isIdleTimerDisabled = true` while streaming.
- Screen shows: live preview, URL, fps/clients/dropped counters, lens name, and a
  big EXPOSURE LOCK toggle. Nothing else.
- Info.plist: NSCameraUsageDescription, NSLocalNetworkUsageDescription,
  NSBonjourServices ["_widecam._tcp"]. Landscape-only.
- Bundle id: `com.pxshaa.widecam`. Deployment target: iOS 17.

## Mac side (`phase0/capture/netsource.py`)
- `NetVideoSource(url_or_name, timeout=2.0)`: same surface as `AVVideoSource`:
  isOpened/read/get/set/release/name; `read()` returns (ok, BGR uint8) and never
  the same X-Seq twice; exposes `last_pts` (phone timestamp) and `last_seq`.
- Discovery: if given "auto" or a service name, browse `_widecam._tcp` for up to
  3s (use `dns-sd -B`/`-L` via subprocess or a pure-python zeroconf); else treat
  the argument as a URL/host.
- Bandwidth sanity: 720p@60 at q0.7 is roughly 25-40 Mbit/s. Fine on 5 GHz WiFi.
