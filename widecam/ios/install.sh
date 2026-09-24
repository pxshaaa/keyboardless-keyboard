#!/bin/sh
# Wireless install + launch via devicectl (no cable). DEVICE = id from `xcrun devicectl list devices`.
cd "$(dirname "$0")"
[ -f local.env ] && set -a && . ./local.env && set +a
: "${DEVICE:?set DEVICE to your iPhone's devicectl id}"
xcrun devicectl device install app --device "$DEVICE" build/WideCam.app && \
  xcrun devicectl device process launch --device "$DEVICE" "${BUNDLE_ID_PREFIX:-com.example}.widecam"
