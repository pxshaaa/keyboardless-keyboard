import SwiftUI
import AVFoundation

@main
struct WideCamApp: App {
    @StateObject private var streamer = CameraStreamer()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            ContentView(streamer: streamer)
                .onAppear {
                    UIApplication.shared.isIdleTimerDisabled = true
                    UIDevice.current.isBatteryMonitoringEnabled = true
                    AVCaptureDevice.requestAccess(for: .video) { granted in
                        DispatchQueue.main.async {
                            streamer.cameraAuthorized = granted
                            if granted { streamer.start() }
                        }
                    }
                }
        }
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .active:
                UIApplication.shared.isIdleTimerDisabled = true
                if streamer.cameraAuthorized { streamer.start() }
            case .background:
                streamer.stop()
            default: break
            }
        }
    }
}
