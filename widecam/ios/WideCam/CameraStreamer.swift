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

final class CameraStreamer: NSObject, ObservableObject {
    // UI-observed
    @Published var lensName = "—"
    @Published var usingFallbackLens = false
    @Published var formatDescription = "—"
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
    private let captureQueue = DispatchQueue(label: "widecam.capture", qos: .userInteractive)
    private let encodeQueue = DispatchQueue(label: "widecam.encode", qos: .userInteractive)
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false, .cacheIntermediates: false])
    private let colorSpace = CGColorSpaceCreateDeviceRGB()
    private var encoding = OSAllocatedUnfairLock(initialState: false)
    private var counters = OSAllocatedUnfairLock(initialState: (seq: 0, dropped: 0))
    private var quality: Double = 0.7
    private var lastUIUpdate: Double = 0
    private var started = false

    override init() {
        super.init()
        server.streamer = self
    }

    // MARK: lifecycle

    func start() {
        if !started { configure(); started = true }
        captureQueue.async { [self] in if !session.isRunning { session.startRunning() } }
        server.start(streamer: self)
    }

    func stop() {
        captureQueue.async { [self] in if session.isRunning { session.stopRunning() } }
    }

    private func configure() {
        session.beginConfiguration()
        session.sessionPreset = .inputPriority

        var dev = AVCaptureDevice.default(.builtInUltraWideCamera, for: .video, position: .back)
        var fallback = false
        if dev == nil {
            dev = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
            fallback = true
        }
        guard let dev, let inp = try? AVCaptureDeviceInput(device: dev), session.canAddInput(inp) else {
            session.commitConfiguration()
            DispatchQueue.main.async { self.errorText = "No back camera available" }
            return
        }
        session.addInput(inp)
        device = dev; input = inp

        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: captureQueue)
        if session.canAddOutput(output) { session.addOutput(output) }
        if let conn = output.connection(with: .video), conn.isVideoRotationAngleSupported(0) {
            conn.videoRotationAngle = 0   // landscape (sensor-native, home button right)
        }
        session.commitConfiguration()

        let lens = fallback ? "main (FALLBACK)" : "ultrawide"
        DispatchQueue.main.async {
            self.lensName = lens
            self.usingFallbackLens = fallback
        }
        applyFormat(width: width, height: height, fps: fps)
    }

    // MARK: format selection

    /// Picks the device format whose dims match and whose frame-rate ranges cover `fps`.
    @discardableResult
    func applyFormat(width w: Int, height h: Int, fps f: Double) -> Bool {
        guard let device else { return false }
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
            self.errorText = nil
        }
        server.updateBonjour(width: Int(d.width), height: Int(d.height), fps: f)
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

    /// Applies any subset of /control params; synchronous so /status reflects them.
    func applyControl(exposure: String?, fps: Double?, preset: String?, quality: Double?) {
        if let exposure { setExposureLocked(exposure == "lock") }
        if let quality { setQuality(quality) }
        if fps != nil || preset != nil {
            var w = width, h = height
            if preset == "1080p" { w = 1920; h = 1080 } else if preset == "720p" { w = 1280; h = 720 }
            let f = fps ?? self.fps
            captureQueue.sync {
                session.beginConfiguration()
                applyFormat(width: w, height: h, fps: f)
                session.commitConfiguration()
            }
        }
    }

    /// Snapshot for /status. Safe from any thread.
    func statusJSON() -> String {
        let c = counters.withLock { $0 }
        let lens = usingFallbackLens ? "main" : "ultrawide"
        let exp = (device?.exposureMode == .locked) ? "locked" : "auto"
        let bat = UIDevice.current.batteryLevel   // -1 if unknown
        let dims = device.map { CMVideoFormatDescriptionGetDimensions($0.activeFormat.formatDescription) }
        let w = dims.map { Int($0.width) } ?? width
        let h = dims.map { Int($0.height) } ?? height
        let f = device.map { 1.0 / CMTimeGetSeconds($0.activeVideoMinFrameDuration) } ?? fps
        let fStr = String(format: "%.1f", f)
        let batStr = String(format: "%.2f", bat < 0 ? 0 : Double(bat))
        return "{\"lens\":\"\(lens)\",\"width\":\(w),\"height\":\(h),\"fps\":\(fStr),\"exposure\":\"\(exp)\",\"clients\":\(server.clientCount),\"seq\":\(c.seq),\"dropped\":\(c.dropped),\"battery\":\(batStr)}"
    }
}

// MARK: - capture delegate

extension CameraStreamer: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let ts = CACurrentMediaTime()   // capture timestamp, BEFORE encoding
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // Atomic try-acquire of the encoder; drop if busy.
        let acquired = encoding.withLock { busy -> Bool in
            if busy { return false }
            busy = true; return true
        }
        let seq: Int = counters.withLock { c in
            if acquired { defer { c.seq += 1 }; return c.seq }
            c.dropped += 1; return -1
        }
        guard acquired else { return }

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
}
