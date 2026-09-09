#!/bin/sh
# Wireless install + launch of WideCam on Pasha's iPhone 15 Pro (needs devicectl tunnel up; no cable).
cd "$(dirname "$0")" && xcrun devicectl device install app --device B63707B4-119C-57FA-A7FC-2F3D0E36145A build/WideCam.app && xcrun devicectl device process launch --device B63707B4-119C-57FA-A7FC-2F3D0E36145A com.pxshaa.widecam
