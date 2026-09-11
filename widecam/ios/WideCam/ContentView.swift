import SwiftUI
import AVFoundation

struct PreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    final class V: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var layer_: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
    func makeUIView(context: Context) -> V {
        let v = V()
        v.layer_.session = session
        v.layer_.videoGravity = .resizeAspect
        return v
    }
    func updateUIView(_ v: V, context: Context) {}
}

struct ContentView: View {
    @ObservedObject var streamer: CameraStreamer
    @State private var ips: [String] = []
    @State private var battery: Float = UIDevice.current.batteryLevel
    private let tick = Timer.publish(every: 2, on: .main, in: .common).autoconnect()

    var body: some View {
        ZStack(alignment: .topLeading) {
            Color.black.ignoresSafeArea()
            PreviewView(session: streamer.session).ignoresSafeArea()

            VStack(alignment: .leading, spacing: 6) {
                if streamer.usingFallbackLens {
                    Text("ULTRA-WIDE UNAVAILABLE — USING MAIN LENS")
                        .font(.headline.bold()).foregroundStyle(.white)
                        .padding(10).frame(maxWidth: .infinity).background(Color.red)
                }
                if let e = streamer.errorText {
                    Text(e).font(.subheadline.bold()).foregroundStyle(.white)
                        .padding(6).background(Color.orange)
                }
                if !streamer.cameraAuthorized {
                    Text("Camera permission denied — enable in Settings").font(.subheadline.bold())
                        .foregroundStyle(.white).padding(6).background(Color.red)
                }
                ForEach(urls, id: \.self) { u in
                    Text(u).font(.system(.title3, design: .monospaced).bold())
                }
                Text("lens: \(streamer.lensName)   format: \(streamer.formatDescription)")
                Text("mode: \(streamer.mode.rawValue)   depth: \(streamer.depthDescription)")
                Text("fps: \(Int(streamer.fps))   clients: \(streamer.clients)   seq: \(streamer.seq)   dropped: \(streamer.dropped)   battery: \(batteryText)")
            }
            .font(.system(.body, design: .monospaced))
            .foregroundStyle(.white)
            .padding(12)
            .background(Color.black.opacity(0.55))
            .padding(.top, 8)
            .padding(.leading, 8)

            VStack {
                Spacer()
                HStack {
                    Spacer()
                    Button {
                        streamer.setMode(streamer.mode == .rgb ? .depth : .rgb)
                    } label: {
                        Text(streamer.mode == .depth ? "DEPTH (LiDAR)" : "RGB (ultrawide)")
                            .font(.title2.bold())
                            .padding(.horizontal, 20).padding(.vertical, 18)
                            .background(streamer.mode == .depth ? Color.blue : Color.gray.opacity(0.85))
                            .foregroundStyle(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 14))
                    }
                    Button {
                        streamer.setExposureLocked(!streamer.exposureLocked)
                    } label: {
                        Text(streamer.exposureLocked ? "EXPOSURE LOCKED" : "EXPOSURE AUTO")
                            .font(.title.bold())
                            .padding(.horizontal, 28).padding(.vertical, 18)
                            .background(streamer.exposureLocked ? Color.red : Color.green.opacity(0.85))
                            .foregroundStyle(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 14))
                    }
                    .padding(24)
                }
            }
        }
        .statusBarHidden(true)
        .persistentSystemOverlays(.hidden)
        .onAppear { refresh() }
        .onReceive(tick) { _ in refresh() }
    }

    private var urls: [String] {
        let p = streamer.port
        guard p != 0 else { return ["starting server…"] }
        return ips.isEmpty ? ["no WiFi IPv4 (port \(p))"] : ips.map { "http://\($0):\(p)/stream" }
    }

    private var batteryText: String {
        battery < 0 ? "?" : "\(Int(battery * 100))%"
    }

    private func refresh() {
        ips = Self.ipv4Addresses()
        battery = UIDevice.current.batteryLevel
    }

    /// IPv4 addresses of en0 (WiFi) first, then any other non-loopback interface.
    static func ipv4Addresses() -> [String] {
        var out: [(String, String)] = []
        var ifap: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&ifap) == 0, let first = ifap else { return [] }
        defer { freeifaddrs(ifap) }
        for p in sequence(first: first, next: { $0.pointee.ifa_next }) {
            let ifa = p.pointee
            guard let sa = ifa.ifa_addr, sa.pointee.sa_family == UInt8(AF_INET),
                  (Int32(ifa.ifa_flags) & IFF_LOOPBACK) == 0 else { continue }
            let name = String(cString: ifa.ifa_name)
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            if getnameinfo(sa, socklen_t(sa.pointee.sa_len), &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST) == 0 {
                out.append((name, String(cString: host)))
            }
        }
        return out.sorted { a, b in (a.0 == "en0" ? 0 : 1) < (b.0 == "en0" ? 0 : 1) }.map(\.1)
    }
}
