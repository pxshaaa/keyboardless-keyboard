import AVFoundation
import Foundation

/// One depth map with the metadata the Mac side needs to turn it into millimetres.
struct DepthFrame {
    let seq: Int
    let timestamp: Double      // same clock/value as the paired Frame
    let width: Int
    let height: Int
    let accuracy: String       // "absolute" | "relative"
    let quality: String        // "high" | "low"
    let filtered: Bool
    let intrinsics: String     // "fx,fy,cx,cy" in *reference* pixels, or ""
    let intrinsicsRef: String  // "w,h" the intrinsics are expressed in, or ""
    let pixels: Data           // tightly packed row-major Float32 metres, NaN where invalid
}

enum DepthPacker {
    /// AVDepthData -> Float32 metres, rows tightly packed (bytesPerRow padding removed).
    static func pack(_ depth: AVDepthData, seq: Int, timestamp: Double, filtered: Bool) -> DepthFrame? {
        let d = depth.depthDataType == kCVPixelFormatType_DepthFloat32
            ? depth
            : ((try? depth.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)) ?? depth)
        let map = d.depthDataMap
        CVPixelBufferLockBaseAddress(map, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(map) else { return nil }
        let w = CVPixelBufferGetWidth(map), h = CVPixelBufferGetHeight(map)
        let stride = CVPixelBufferGetBytesPerRow(map)
        var out = Data(count: w * h * 4)
        out.withUnsafeMutableBytes { dst in
            guard let dp = dst.baseAddress else { return }
            for y in 0..<h {
                memcpy(dp.advanced(by: y * w * 4), base.advanced(by: y * stride), w * 4)
            }
        }
        var intr = "", ref = ""
        if let cal = d.cameraCalibrationData {
            let m = cal.intrinsicMatrix
            intr = String(format: "%.4f,%.4f,%.4f,%.4f", m.columns.0.x, m.columns.1.y, m.columns.2.x, m.columns.2.y)
            ref = "\(Int(cal.intrinsicMatrixReferenceDimensions.width)),\(Int(cal.intrinsicMatrixReferenceDimensions.height))"
        }
        let acc = d.depthDataAccuracy == .absolute ? "absolute" : "relative"
        let qual = d.depthDataQuality == .high ? "high" : "low"
        return DepthFrame(seq: seq, timestamp: timestamp, width: w, height: h,
                          accuracy: acc, quality: qual, filtered: filtered,
                          intrinsics: intr, intrinsicsRef: ref, pixels: out)
    }

    /// multipart part, boundary "depth".
    static func part(_ f: DepthFrame) -> Data {
        var d = Data(("--depth\r\n"
            + "Content-Type: application/octet-stream\r\n"
            + "Content-Length: \(f.pixels.count)\r\n"
            + "X-Seq: \(f.seq)\r\n"
            + "X-Timestamp: \(String(format: "%.6f", f.timestamp))\r\n"
            + "X-Width: \(f.width)\r\n"
            + "X-Height: \(f.height)\r\n"
            + "X-Format: float32\r\n"
            + "X-Accuracy: \(f.accuracy)\r\n"
            + "X-Quality: \(f.quality)\r\n"
            + "X-Filtered: \(f.filtered ? 1 : 0)\r\n"
            + "X-Intrinsics: \(f.intrinsics)\r\n"
            + "X-Intrinsics-Ref: \(f.intrinsicsRef)\r\n"
            + "\r\n").utf8)
        d.append(f.pixels)
        d.append(Data("\r\n".utf8))
        return d
    }
}
