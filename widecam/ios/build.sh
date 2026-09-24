#!/bin/sh
# Build + sign WideCam. Settings come from the environment or a gitignored local.env (see local.env.example).
set -e
cd "$(dirname "$0")"
[ -f local.env ] && set -a && . ./local.env && set +a
: "${DEVELOPMENT_TEAM:?set DEVELOPMENT_TEAM to your Apple developer team id}"
export BUNDLE_ID_PREFIX="${BUNDLE_ID_PREFIX:-com.example}"
xcodegen generate >/dev/null
xcodebuild -project WideCam.xcodeproj -scheme WideCam -sdk iphoneos -configuration Release \
  -allowProvisioningUpdates -allowProvisioningDeviceRegistration \
  CODE_SIGN_STYLE=Automatic DEVELOPMENT_TEAM="$DEVELOPMENT_TEAM" -derivedDataPath build/dd build | grep -E "error:|BUILD"
rm -rf build/WideCam.app && cp -R build/dd/Build/Products/Release-iphoneos/WideCam.app build/WideCam.app
if [ -n "$DEVICE_UDID" ]; then
  grep -q "$DEVICE_UDID" build/WideCam.app/embedded.mobileprovision && echo "profile includes $DEVICE_UDID" || echo "WARNING: $DEVICE_UDID not in profile, install will fail"
fi
