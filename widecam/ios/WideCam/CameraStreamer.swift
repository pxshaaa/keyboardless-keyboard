import AVFoundation
import CoreImage
import UIKit
import os

/// One encoded frame with capture metadata (PROTOCOL.md /stream part).
struct Frame {
    let seq: Int
    let timestamp: Double   // CACurrentMediaTime() at capture callback
    let width: Int
    let height: Int
    let jpeg: Data
}

enum CaptureMode: String {
    case rgb    // ultra-wide lens, no depth (frozen PROTOCOL.md behaviour)
    case depth  // LiDAR depth device: synchronized video + AVDepthData
}

final class CameraStreamer: NSObject, ObservableObject {
    // UI-observed
    @Published var lensName = "—"
    @Published var usingFallbackLens = false
    @Published var formatDescription = "—"
    @Published var depthDescription = "—"
    @Published var mode: CaptureMode = .rgb
    @Published var width = 1280
    @Published var height = 720
    @Published var fps: Double = 60
    @Published var exposureLocked = false
    @Published var seq = 0
    @Published var dropped = 0
    @Published var clients = 0
    @Published var port: UInt16 = 0
    @Published var cameraAuthorized = false
    @Published var errorText: String?

    let session = AVCaptureSession()
    let server = MJPEGServer()

    private var device: AVCaptureDevice?
    private var input: AVCaptureDeviceInput?
    private let output = AVCaptureVideoDataOutput()
    private let depthOutput = AVCaptureDepthDataOutput()
    private var synchronizer: AVCaptureDataOutputSynchronizer?
    private let captureQueue = DispatchQueue(label: "widecam.capture", qos: .userInteractive)
    private let encodeQueue = DispatchQueue(label: "widecam.encode", qos: .userInteractive)
    private let depthQueue = DispatchQueue(label: "widecam.depth", qos: .userInteractive)
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false, .cacheIntermediates: false])
    private let colorSpace = CGColorSpaceCreateDeviceRGB()
    private var encoding = OSAllocatedUnfairLock(initialState: false)
    private var packing = OSAllocatedUnfairLock(initialState: false)
    private var counters = OSAllocatedUnfairLock(initialState: (seq: 0, dropped: 0))
    private var quality: Double = 0.7
    private var depthFiltering = false
    private var depthDevicePref = "lidar"   // lidar | dualwide | triple | wide | ultrawide
    private var lastUIUpdate: Double = 0
    private var started = false

    override init() {
        super.init()
        server.streamer = self
    }

    // MARK: lifecycle

    func start() {
        if !started { DepthProbe.logReport(); configure(mode: mode); started = true }
        captureQueue.async { [self] in if !session.isRunning { session.startRunning() } }
        server.start(streamer: self)
    }

    func stop() {
        captureQueue.async { [self] in if session.isRunning { session.stopRunning() } }
    }

    // MARK: configuration

    private func configure(mode m: CaptureMode) {
        session.beginConfiguration()
        session.sessionPreset = .inputPriority

        synchronizer?.setDelegate(nil, queue: nil)
        synchronizer = nil
        output.setSampleBufferDelegate(nil, queue: nil)
        for i in session.inputs { session.removeInput(i) }
        for o in session.outputs { session.removeOutput(o) }

        var fallback = false
        var dev: AVCaptureDevice?
        if m == .depth {
            let t: AVCaptureDevice.DeviceType
            switch depthDevicePref {
            case "dualwide": t = .builtInDualWideCamera
            case "triple": t = .builtInTripleCamera
            case "wide": t = .builtInWideAngleCamera
            case "ultrawide": t = .builtInUltraWideCamera
            default: t = .builtInLiDARDepthCamera
            }
            dev = AVCaptureDevice.default(t, for: .video, position: .back)
            if dev == nil {
                NSLog("[WideCam] no \(t.rawValue) on this device")
                DispatchQueue.main.async { self.errorText = "No depth device \(self.depthDevicePref)" }
            }
        } else {
            dev = AVCaptureDevice.default(.builtInUltraWideCamera, for: .video, position: .back)
            if dev == nil {
                dev = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
                fallback = true
            }
        }
        guard let dev, let inp = try? AVCaptureDeviceInput(device: dev), session.canAddInput(inp) else {
            session.commitConfiguration()
            DispatchQueue.main.async { self.errorText = "No camera for mode \(m.rawValue)" }
            return
        }
        session.addInput(inp)
        device = dev; input = inp

        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        if session.canAddOutput(output) { session.addOutput(output) }
        if let conn = output.connection(with: .video), conn.isVideoRotationAngleSupported(0) {
            conn.videoRotationAngle = 0   // landscape (sensor-native, home button right)
        }

        if m == .depth {
            depthOutput.alwaysDiscardsLateDepthData = true
            depthOutput.isFilteringEnabled = depthFiltering
            if session.canAddOutput(depthOutput) { session.addOutput(depthOutput) }
            if let conn = depthOutput.connection(with: .depthData), conn.isVideoRotationAngleSupported(0) {
                conn.videoRotationAngle = 0
            }
        } else {
            output.setSampleBufferDelegate(self, queue: captureQueue)
        }

        let lens = m == .depth ? "lidar(\(dev.localizedName))" : (fallback ? "main (FALLBACK)" : "ultrawide")
        DispatchQueue.main.async {
            self.lensName = lens
            self.usingFallbackLens = fallback
            self.mode = m
        }

        if m == .depth {
            applyDepthFormat(preferredWidth: 1280, fps: 30)
            // Synchronizer must be created after the outputs are on the session.
            let s = AVCaptureDataOutputSynchronizer(dataOutputs: [output, depthOutput])
            s.setDelegate(self, queue: captureQueue)
            synchronizer = s
            session.commitConfiguration()
        } else {
            session.commitConfiguration()
            applyFormat(width: width, height: height, fps: fps)
        }
    }

    /// Switches lens+outputs. Restarts the session so the change takes effect immediately.
    func setMode(_ m: CaptureMode) {
        guard m != mode else { return }
        captureQueue.sync { reconfigureRunning(m) }
    }

    private func reconfigureRunning(_ m: CaptureMode) {
        let running = session.isRunning
        if running { session.stopRunning() }
        configure(mode: m)
        if running { session.startRunning() }
    }

    // MARK: format selection

    /// Picks the device format whose dims match and whose frame-rate ranges cover `fps`.
    @discardableResult
    func applyFormat(width w: Int, height h: Int, fps f: Double) -> Bool {
        guard let device, mode == .rgb else { return false }
        let candidates = device.formats.filter { fmt in
            let d = CMVideoFormatDescriptionGetDimensions(fmt.formatDescription)
            guard Int(d.width) == w, Int(d.height) == h else { return false }
            guard CMFormatDescriptionGetMediaSubType(fmt.formatDescription) == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
                || CMFormatDescriptionGetMediaSubType(fmt.formatDescription) == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange else { return false }
            return fmt.videoSupportedFrameRateRanges.contains { $0.minFrameRate <= f && f <= $0.maxFrameRate }
        }
        // Prefer non-binned, lowest max fps that still covers (least sensor stress), non-HDR.
        guard let fmt = candidates.sorted(by: { a, b in
            let am = a.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
            let bm = b.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
            if a.isVideoBinned != b.isVideoBinned { return !a.isVideoBinned }
            return am < bm
        }).first else {
            NSLog("[WideCam] no format for \(w)x\(h)@\(f) on \(device.localizedName)")
            DispatchQueue.main.async { self.errorText = "No \(w)x\(h)@\(Int(f)) format on this lens" }
            return false
        }
        do {
            try device.lockForConfiguration()
            device.activeFormat = fmt
            let dur = CMTime(value: 1, timescale: CMTimeScale(f))
            device.activeVideoMinFrameDuration = dur
            device.activeVideoMaxFrameDuration = dur
            device.unlockForConfiguration()
        } catch {
            NSLog("[WideCam] lockForConfiguration failed: \(error)")
            return false
        }
        let d = CMVideoFormatDescriptionGetDimensions(fmt.formatDescription)
        let maxFps = fmt.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
        let desc = "\(d.width)x\(d.height) @ \(Int(f)) fps (format max \(Int(maxFps)), binned=\(fmt.isVideoBinned))"
        NSLog("[WideCam] chosen format: \(desc) lens=\(device.localizedName)")
        DispatchQueue.main.async {
            self.width = Int(d.width); self.height = Int(d.height); self.fps = f
            self.formatDescription = desc
            self.depthDescription = "—"
            self.errorText = nil
        }
        server.updateBonjour(width: Int(d.width), height: Int(d.height), fps: f)
        return true
    }

    /// Picks a video format that carries depth, plus the smallest float depth format
    /// (the extra AVFoundation sizes are upsampled from the same sparse LiDAR return).
    @discardableResult
    func applyDepthFormat(preferredWidth pw: Int, fps f: Double) -> Bool {
        guard let device else { return false }
        let withDepth = device.formats.filter { !$0.supportedDepthDataFormats.isEmpty }
        guard !withDepth.isEmpty else {
            NSLog("[WideCam] device \(device.localizedName) has NO depth-capable formats")
            DispatchQueue.main.async { self.errorText = "No depth-capable format" }
            return false
        }
        let usable = withDepth.filter { fmt in
            fmt.videoSupportedFrameRateRanges.contains { $0.minFrameRate <= f && f <= $0.maxFrameRate }
        }
        guard let fmt = (usable.isEmpty ? withDepth : usable).min(by: { a, b in
            let ad = CMVideoFormatDescriptionGetDimensions(a.formatDescription)
            let bd = CMVideoFormatDescriptionGetDimensions(b.formatDescription)
            return abs(Int(ad.width) - pw) < abs(Int(bd.width) - pw)
        }) else { return false }
        guard let dfmt = fmt.supportedDepthDataFormats.min(by: { a, b in
            let ad = CMVideoFormatDescriptionGetDimensions(a.formatDescription)
            let bd = CMVideoFormatDescriptionGetDimensions(b.formatDescription)
            return Int(ad.width) * Int(ad.height) < Int(bd.width) * Int(bd.height)
        }) else { return false }
        do {
            try device.lockForConfiguration()
            device.activeFormat = fmt
            device.activeDepthDataFormat = dfmt
            let dur = CMTime(value: 1, timescale: CMTimeScale(f))
            device.activeVideoMinFrameDuration = dur
            device.activeVideoMaxFrameDuration = dur
            device.unlockForConfiguration()
        } catch {
            NSLog("[WideCam] depth lockForConfiguration failed: \(error)")
            return false
        }
        let vd = CMVideoFormatDescriptionGetDimensions(fmt.formatDescription)
        let dd = CMVideoFormatDescriptionGetDimensions(dfmt.formatDescription)
        let sub = DepthProbe.fourCC(CMFormatDescriptionGetMediaSubType(dfmt.formatDescription))
        let vdesc = "\(vd.width)x\(vd.height) @ \(Int(f)) fps (binned=\(fmt.isVideoBinned))"
        let ddesc = "\(dd.width)x\(dd.height) \(sub) fov=\(String(format: "%.1f", fmt.videoFieldOfView))deg filtering=\(depthFiltering)"
        NSLog("[WideCam] DEPTH MODE video=\(vdesc) depth=\(ddesc) lens=\(device.localizedName)")
        DispatchQueue.main.async {
            self.width = Int(vd.width); self.height = Int(vd.height); self.fps = f
            self.formatDescription = vdesc
            self.depthDescription = ddesc
            self.errorText = nil
        }
        server.updateBonjour(width: Int(vd.width), height: Int(vd.height), fps: f)
        return true
    }

    // MARK: controls (called from server queue or UI)

    func setExposureLocked(_ locked: Bool) {
        guard let device else { return }
        do {
            try device.lockForConfiguration()
            if locked, device.isExposureModeSupported(.locked) {
                device.exposureMode = .locked
            } else if device.isExposureModeSupported(.continuousAutoExposure) {
                device.exposureMode = .continuousAutoExposure
            }
            device.unlockForConfiguration()
            DispatchQueue.main.async { self.exposureLocked = locked }
        } catch {
            NSLog("[WideCam] exposure lock failed: \(error)")
        }
    }

    func setQuality(_ q: Double) {
        encodeQueue.async { self.quality = min(0.95, max(0.3, q)) }
    }

    func setDepthFiltering(_ on: Bool) {
        depthFiltering = on
        depthOutput.isFilteringEnabled = on
        if mode == .depth {
            let d = depthDescription.replacingOccurrences(of: "filtering=\(!on)", with: "filtering=\(on)")
            DispatchQueue.main.async { self.depthDescription = d }
        }
    }

    /// Applies any subset of /control params; synchronous so /status reflects them.
    func applyControl(exposure: String?, fps: Double?, preset: String?, quality: Double?,
                      mode m: String?, filter: String?, device dpref: String?) {
        if let dpref, dpref != depthDevicePref {
            depthDevicePref = dpref
            if mode == .depth { captureQueue.sync { reconfigureRunning(.depth) } }
        }
        if let m, let cm = CaptureMode(rawValue: m) { setMode(cm) }
        if let filter { setDepthFiltering(filter == "on" || filter == "1") }
        if let exposure { setExposureLocked(exposure == "lock") }
        if let quality { setQuality(quality) }
        if fps != nil || preset != nil {
            let f = fps ?? self.fps
            if mode == .depth {
                captureQueue.sync {
                    session.beginConfiguration()
                    applyDepthFormat(preferredWidth: preset == "1080p" ? 1920 : 1280, fps: f)
                    session.commitConfiguration()
                }
            } else {
                var w = width, h = height
                if preset == "1080p" { w = 1920; h = 1080 } else if preset == "720p" { w = 1280; h = 720 }
                captureQueue.sync {
                    session.beginConfiguration()
                    applyFormat(width: w, height: h, fps: f)
                    session.commitConfiguration()
                }
            }
        }
    }

    /// Snapshot for /status. Safe from any thread.
    func statusJSON() -> String {
        let c = counters.withLock { $0 }
        let lens = mode == .depth ? "lidar" : (usingFallbackLens ? "main" : "ultrawide")
        let exp = (device?.exposureMode == .locked) ? "locked" : "auto"
        let bat = UIDevice.current.batteryLevel   // -1 if unknown
        let dims = device.map { CMVideoFormatDescriptionGetDimensions($0.activeFormat.formatDescription) }
        let w = dims.map { Int($0.width) } ?? width
        let h = dims.map { Int($0.height) } ?? height
        let f = device.map { 1.0 / CMTimeGetSeconds($0.activeVideoMinFrameDuration) } ?? fps
        let fStr = String(format: "%.1f", f)
        let batStr = String(format: "%.2f", bat < 0 ? 0 : Double(bat))
        var dw = 0, dh = 0
        if let dfmt = device?.activeDepthDataFormat {
            let d = CMVideoFormatDescriptionGetDimensions(dfmt.formatDescription)
            dw = Int(d.width); dh = Int(d.height)
        }
        let fov = device.map { String(format: "%.2f", $0.activeFormat.videoFieldOfView) } ?? "0"
        return "{\"lens\":\"\(lens)\",\"width\":\(w),\"height\":\(h),\"fps\":\(fStr),\"exposure\":\"\(exp)\","
            + "\"clients\":\(server.clientCount),\"seq\":\(c.seq),\"dropped\":\(c.dropped),\"battery\":\(batStr),"
            + "\"mode\":\"\(mode.rawValue)\",\"device\":\"\(device?.localizedName ?? "")\",\"fov_deg\":\(fov),"
            + "\"depth_width\":\(dw),\"depth_height\":\(dh),\"depth_filtering\":\(depthFiltering),"
            + "\"depth_device_pref\":\"\(depthDevicePref)\",\"depth_clients\":\(server.depthClientCount)}"
    }

    // MARK: frame plumbing

    /// JPEG-encodes and publishes; returns false if the encoder was busy (frame dropped).
    private func encodeAndPublish(_ pb: CVPixelBuffer, seq: Int, ts: Double) {
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        encodeQueue.async { [self] in
            defer { encoding.withLock { $0 = false } }
            let img = CIImage(cvPixelBuffer: pb)
            let opts: [CIImageRepresentationOption: Any] = [
                kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: quality
            ]
            guard let jpeg = ciContext.jpegRepresentation(of: img, colorSpace: colorSpace, options: opts) else { return }
            server.publish(Frame(seq: seq, timestamp: ts, width: w, height: h, jpeg: jpeg))
            if ts - lastUIUpdate > 0.25 {
                lastUIUpdate = ts
                let c = counters.withLock { $0 }
                let n = server.clientCount
                DispatchQueue.main.async { self.seq = c.seq; self.dropped = c.dropped; self.clients = n }
            }
        }
    }

    /// Atomic try-acquire of the encoder; -1 means "busy, frame dropped".
    private func nextSeqOrDrop() -> Int {
        let acquired = encoding.withLock { busy -> Bool in
            if busy { return false }
            busy = true; return true
        }
        return counters.withLock { c in
            if acquired { defer { c.seq += 1 }; return c.seq }
            c.dropped += 1; return -1
        }
    }
}

// MARK: - capture delegates

extension CameraStreamer: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let ts = CACurrentMediaTime()   // capture timestamp, BEFORE encoding
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let seq = nextSeqOrDrop()
        guard seq >= 0 else { return }
        encodeAndPublish(pb, seq: seq, ts: ts)
    }
}

extension CameraStreamer: AVCaptureDataOutputSynchronizerDelegate {
    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer,
                                didOutput collection: AVCaptureSynchronizedDataCollection) {
        let ts = CACurrentMediaTime()
        let video = collection.synchronizedData(for: output) as? AVCaptureSynchronizedSampleBufferData
        let depth = collection.synchronizedData(for: depthOutput) as? AVCaptureSynchronizedDepthData

        // One seq per synchronized pair so the Mac can join RGB and depth exactly.
        let seq = nextSeqOrDrop()
        if seq >= 0, let video, !video.sampleBufferWasDropped,
           let pb = CMSampleBufferGetImageBuffer(video.sampleBuffer) {
            encodeAndPublish(pb, seq: seq, ts: ts)
        } else if seq >= 0 {
            encoding.withLock { $0 = false }
        }

        guard let depth, !depth.depthDataWasDropped, server.depthClientCount > 0 else { return }
        let busy = packing.withLock { b -> Bool in
            if b { return true }
            b = true; return false
        }
        guard !busy else { return }
        let dd = depth.depthData
        let filtering = depthFiltering
        depthQueue.async { [self] in
            defer { packing.withLock { $0 = false } }
            if let df = DepthPacker.pack(dd, seq: max(seq, 0), timestamp: ts, filtered: filtering) {
                server.publishDepth(df)
            }
        }
    }
}
