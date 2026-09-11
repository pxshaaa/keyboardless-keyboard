import Foundation
import Network
import os

/// Minimal HTTP/MJPEG server per PROTOCOL.md. Hand-parses the request line only.
final class MJPEGServer {
    weak var streamer: CameraStreamer?
    private(set) var port: UInt16 = 0
    private var listener: NWListener?
    private let queue = DispatchQueue(label: "widecam.server")
    private var clients: [ObjectIdentifier: StreamClient] = [:]
    private var depthClients: [ObjectIdentifier: StreamClient] = [:]
    private let clientsLock = OSAllocatedUnfairLock()
    private var txt = ["v": "1", "w": "1280", "h": "720", "fps": "60"]

    var clientCount: Int { clientsLock.withLock { clients.count } }
    var depthClientCount: Int { clientsLock.withLock { depthClients.count } }

    func start(streamer: CameraStreamer) {
        self.streamer = streamer
        guard listener == nil else { return }
        if !listen(on: 8080) { _ = listen(on: 0) }
    }

    private func listen(on p: UInt16) -> Bool {
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        guard let port = NWEndpoint.Port(rawValue: p),
              let l = try? NWListener(using: params, on: port) else { return false }
        l.service = NWListener.Service(name: "WideCam", type: "_widecam._tcp", txtRecord: NWTXTRecord(txt))
        l.newConnectionHandler = { [weak self] c in self?.accept(c) }
        l.stateUpdateHandler = { [weak self] st in
            guard let self else { return }
            switch st {
            case .ready:
                self.port = l.port?.rawValue ?? p
                NSLog("[WideCam] listening on port \(self.port) ips=\(ContentView.ipv4Addresses().joined(separator: ","))")
                DispatchQueue.main.async { self.streamer?.port = self.port }
            case .failed(let e):
                NSLog("[WideCam] listener failed: \(e)")
                self.listener = nil
                if p == 8080 { _ = self.listen(on: 0) }
            default: break
            }
        }
        listener = l
        l.start(queue: queue)
        return true
    }

    func updateBonjour(width: Int, height: Int, fps: Double) {
        txt = ["v": "1", "w": "\(width)", "h": "\(height)", "fps": "\(Int(fps))"]
        queue.async { [self] in
            listener?.service = NWListener.Service(name: "WideCam", type: "_widecam._tcp", txtRecord: NWTXTRecord(txt))
        }
    }

    /// Called from the encode queue. Every stream client gets the latest frame; old unsent ones are replaced.
    func publish(_ f: Frame) {
        let cs = clientsLock.withLock { Array(clients.values) }
        let d = StreamClient.part(f)
        for c in cs { c.offer(d) }
    }

    /// Called from the depth queue; same single-slot mailbox discipline as video.
    func publishDepth(_ f: DepthFrame) {
        let cs = clientsLock.withLock { Array(depthClients.values) }
        guard !cs.isEmpty else { return }
        let d = DepthPacker.part(f)
        for c in cs { c.offer(d) }
    }

    // MARK: connections

    private func accept(_ conn: NWConnection) {
        conn.stateUpdateHandler = { [weak self] st in
            if case .failed = st { self?.remove(conn) }
            if case .cancelled = st { self?.remove(conn) }
        }
        conn.start(queue: queue)
        readRequest(conn, buffer: Data())
    }

    private func readRequest(_ conn: NWConnection, buffer: Data) {
        conn.receive(minimumIncompleteLength: 1, maximumLength: 8192) { [weak self] data, _, done, err in
            guard let self else { return }
            var buf = buffer
            if let data { buf.append(data) }
            if err != nil || (done && buf.isEmpty) { conn.cancel(); return }
            if let r = buf.range(of: Data("\r\n\r\n".utf8)) {
                let head = String(decoding: buf[..<r.lowerBound], as: UTF8.self)
                self.handle(conn, requestLine: head.split(separator: "\r\n", maxSplits: 1).first.map(String.init) ?? "")
            } else if buf.count > 8192 || done {
                conn.cancel()
            } else {
                self.readRequest(conn, buffer: buf)
            }
        }
    }

    private func handle(_ conn: NWConnection, requestLine: String) {
        let parts = requestLine.split(separator: " ")
        guard parts.count >= 2 else { respond(conn, status: "400 Bad Request", body: "bad request"); return }
        let method = String(parts[0]), target = String(parts[1])
        let path = target.split(separator: "?", maxSplits: 1).first.map(String.init) ?? target
        let query = target.contains("?") ? String(target.split(separator: "?", maxSplits: 1)[1]) : ""

        switch (method, path) {
        case ("GET", "/stream"):
            let hdr = "HTTP/1.1 200 OK\r\nConnection: close\r\nCache-Control: no-cache\r\nPragma: no-cache\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n"
            let client = StreamClient(conn: conn, queue: queue) { [weak self] in self?.remove(conn) }
            clientsLock.withLock { clients[ObjectIdentifier(conn)] = client }
            client.send(Data(hdr.utf8))
            NSLog("[WideCam] stream client connected (\(clientCount))")
        case ("GET", "/depth"):
            let hdr = "HTTP/1.1 200 OK\r\nConnection: close\r\nCache-Control: no-cache\r\nPragma: no-cache\r\nContent-Type: multipart/x-mixed-replace; boundary=depth\r\n\r\n"
            let client = StreamClient(conn: conn, queue: queue) { [weak self] in self?.remove(conn) }
            clientsLock.withLock { depthClients[ObjectIdentifier(conn)] = client }
            client.send(Data(hdr.utf8))
            NSLog("[WideCam] depth client connected (\(depthClientCount))")
        case ("GET", "/devices"):
            respondJSON(conn, DepthProbe.report())
        case ("GET", "/status"):
            respondJSON(conn, streamer?.statusJSON() ?? "{}")
        case ("POST", "/control"), ("GET", "/control"):
            var exposure: String?, fps: Double?, preset: String?, quality: Double?, mode: String?, filter: String?, dev: String?
            for kv in query.split(separator: "&") {
                let p = kv.split(separator: "=", maxSplits: 1).map { String($0).removingPercentEncoding ?? String($0) }
                guard p.count == 2 else { continue }
                switch p[0] {
                case "exposure": exposure = p[1]
                case "fps": fps = Double(p[1])
                case "preset": preset = p[1]
                case "quality": quality = Double(p[1])
                case "mode": mode = p[1]
                case "filter": filter = p[1]
                case "device": dev = p[1]
                default: break
                }
            }
            streamer?.applyControl(exposure: exposure, fps: fps, preset: preset, quality: quality,
                                   mode: mode, filter: filter, device: dev)
            respondJSON(conn, streamer?.statusJSON() ?? "{}")
        default:
            respond(conn, status: "404 Not Found", body: "not found")
        }
    }

    private func respondJSON(_ conn: NWConnection, _ json: String) {
        let body = Data(json.utf8)
        let h = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n"
        conn.send(content: Data(h.utf8) + body, completion: .contentProcessed { _ in conn.cancel() })
    }

    private func respond(_ conn: NWConnection, status: String, body: String) {
        let b = Data(body.utf8)
        let h = "HTTP/1.1 \(status)\r\nContent-Type: text/plain\r\nContent-Length: \(b.count)\r\nConnection: close\r\n\r\n"
        conn.send(content: Data(h.utf8) + b, completion: .contentProcessed { _ in conn.cancel() })
    }

    private func remove(_ conn: NWConnection) {
        let removed = clientsLock.withLock {
            clients.removeValue(forKey: ObjectIdentifier(conn)) ?? depthClients.removeValue(forKey: ObjectIdentifier(conn))
        }
        if removed != nil { NSLog("[WideCam] client gone (stream=\(clientCount) depth=\(depthClientCount))") }
    }
}

/// Per-client single-slot mailbox: at most one pending frame; a newer frame replaces it.
final class StreamClient {
    private let conn: NWConnection
    private let queue: DispatchQueue
    private let onError: () -> Void
    private var pending: Data?
    private var sending = false
    private var dead = false

    init(conn: NWConnection, queue: DispatchQueue, onError: @escaping () -> Void) {
        self.conn = conn; self.queue = queue; self.onError = onError
    }

    func offer(_ d: Data) {
        queue.async { [self] in
            guard !dead else { return }
            pending = d          // replace, never queue
            pump()
        }
    }

    func send(_ d: Data) {
        queue.async { [self] in
            sending = true
            conn.send(content: d, completion: .contentProcessed { [self] e in self.done(e) })
        }
    }

    private func pump() {
        guard !sending, let d = pending else { return }
        pending = nil
        sending = true
        conn.send(content: d, completion: .contentProcessed { [self] e in self.done(e) })
    }

    private func done(_ e: NWError?) {
        // completion runs on `queue`
        sending = false
        if e != nil {
            dead = true
            conn.cancel()
            onError()
        } else {
            pump()
        }
    }

    /// Exact multipart part per PROTOCOL.md.
    static func part(_ f: Frame) -> Data {
        var d = Data(("--frame\r\n"
            + "Content-Type: image/jpeg\r\n"
            + "Content-Length: \(f.jpeg.count)\r\n"
            + "X-Seq: \(f.seq)\r\n"
            + "X-Timestamp: \(String(format: "%.6f", f.timestamp))\r\n"
            + "X-Width: \(f.width)\r\n"
            + "X-Height: \(f.height)\r\n"
            + "\r\n").utf8)
        d.append(f.jpeg)
        d.append(Data("\r\n".utf8))
        return d
    }
}
