import AVFoundation
import Foundation

/// On-device enumeration of every back capture device and its depth capability.
/// Exists because the Mac's own AVFoundation hides the iPhone's real device set
/// (Continuity Camera), so this is the only trustworthy answer to
/// "is the ultra-wide lens also the LiDAR depth device?".
enum DepthProbe {

    static func deviceTypes() -> [AVCaptureDevice.DeviceType] {
        var t: [AVCaptureDevice.DeviceType] = [
            .builtInUltraWideCamera,
            .builtInWideAngleCamera,
            .builtInTelephotoCamera,
            .builtInDualCamera,
            .builtInDualWideCamera,
            .builtInTripleCamera,
            .builtInTrueDepthCamera,
        ]
        t.append(.builtInLiDARDepthCamera)
        return t
    }

    /// JSON for GET /devices, and the same text goes to NSLog.
    static func report() -> String {
        let ds = AVCaptureDevice.DiscoverySession(deviceTypes: deviceTypes(),
                                                  mediaType: .video,
                                                  position: .unspecified)
        var devs: [String] = []
        for d in ds.devices {
            let depthFormats = d.formats.filter { !$0.supportedDepthDataFormats.isEmpty }
            var fmtEntries: [String] = []
            for f in depthFormats {
                let vd = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
                let maxFps = f.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
                let dfs = f.supportedDepthDataFormats.map { df -> String in
                    let dd = CMVideoFormatDescriptionGetDimensions(df.formatDescription)
                    let sub = CMFormatDescriptionGetMediaSubType(df.formatDescription)
                    let dmax = df.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
                    return "{\"w\":\(dd.width),\"h\":\(dd.height),\"pixfmt\":\"\(fourCC(sub))\",\"maxFps\":\(Int(dmax))}"
                }
                fmtEntries.append("{\"w\":\(vd.width),\"h\":\(vd.height),\"maxFps\":\(Int(maxFps)),\"binned\":\(f.isVideoBinned),\"fovDeg\":\(String(format: "%.2f", f.videoFieldOfView)),\"depthFormats\":[\(dfs.joined(separator: ","))]}")
            }
            let allDims = Set(d.formats.map { f -> String in
                let vd = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
                return "\(vd.width)x\(vd.height)"
            }).sorted()
            devs.append("""
            {"uniqueID":"\(d.uniqueID)","name":"\(d.localizedName)","type":"\(d.deviceType.rawValue)",\
            "position":\(d.position.rawValue),"fovDeg":\(String(format: "%.2f", d.activeFormat.videoFieldOfView)),\
            "minFocusDistanceMM":\(d.minimumFocusDistance),\
            "formats":\(d.formats.count),"formatsWithDepth":\(depthFormats.count),\
            "constituentDevices":[\(d.constituentDevices.map { "\"\($0.deviceType.rawValue)\"" }.joined(separator: ","))],\
            "allVideoDims":[\(allDims.map { "\"\($0)\"" }.joined(separator: ","))],\
            "depthCapableFormats":[\(fmtEntries.joined(separator: ","))]}
            """)
        }
        return "{\"devices\":[\(devs.joined(separator: ","))]}"
    }

    static func fourCC(_ c: FourCharCode) -> String {
        let bytes = [UInt8((c >> 24) & 0xff), UInt8((c >> 16) & 0xff), UInt8((c >> 8) & 0xff), UInt8(c & 0xff)]
        return String(bytes: bytes, encoding: .ascii) ?? "\(c)"
    }

    static func logReport() {
        // Chunked: NSLog truncates very long lines in the syslog relay.
        let r = report()
        NSLog("[WideCam] DEVICES len=\(r.count)")
        var i = r.startIndex
        var n = 0
        while i < r.endIndex {
            let j = r.index(i, offsetBy: 700, limitedBy: r.endIndex) ?? r.endIndex
            NSLog("[WideCam] DEVICES[\(n)] \(String(r[i..<j]))")
            i = j; n += 1
        }
    }
}
