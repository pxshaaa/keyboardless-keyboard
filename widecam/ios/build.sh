#!/bin/sh
# Rebuild + re-sign. Run this once the phone is reachable (devicectl State=available) so
# -allowProvisioningDeviceRegistration adds udid 00008130-0016141921F2001C to the team profile.
set -e
cd "$(dirname "$0")"
xcodegen generate >/dev/null
xcodebuild -project WideCam.xcodeproj -scheme WideCam -sdk iphoneos -configuration Release \
  -allowProvisioningUpdates -allowProvisioningDeviceRegistration \
  CODE_SIGN_STYLE=Automatic DEVELOPMENT_TEAM=S3QT58BTJT -derivedDataPath build/dd build | grep -E "error:|BUILD"
rm -rf build/WideCam.app && cp -R build/dd/Build/Products/Release-iphoneos/WideCam.app build/WideCam.app
grep -q 00008130-0016141921F2001C build/WideCam.app/embedded.mobileprovision && echo "profile includes iPhone 15 Pro" || echo "WARNING: iPhone 15 Pro udid NOT in profile — install will fail"
