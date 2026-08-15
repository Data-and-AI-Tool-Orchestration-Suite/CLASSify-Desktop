#!/bin/bash
# CLASSify Desktop — macOS .app bundle assembler + DMG builder
#
# Requires: create-dmg (brew install create-dmg) or hdiutil
# Usage: ./build_dmg.sh
set -euo pipefail

VERSION="${CLASSIFY_VERSION:-1.0.0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
DIST_DIR="$REPO_ROOT/dist/CLASSify"
OUTPUT_DIR="$REPO_ROOT/dist/installers"
APP_BUNDLE="$OUTPUT_DIR/CLASSify.app"

mkdir -p "$APP_BUNDLE/Contents/MacOS"
mkdir -p "$APP_BUNDLE/Contents/Resources"
mkdir -p "$APP_BUNDLE/Contents/Frameworks"

# Copy the PyInstaller bundle contents
cp -r "$DIST_DIR"/* "$APP_BUNDLE/Contents/MacOS/"

# Info.plist
cat > "$APP_BUNDLE/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>CLASSify Desktop</string>
    <key>CFBundleDisplayName</key>
    <string>CLASSify Desktop</string>
    <key>CFBundleIdentifier</key>
    <string>edu.uky.appliedai.classify</string>
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleExecutable</key>
    <string>CLASSify</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSAppTransportSecurity</key>
    <dict>
        <key>NSAllowsLocalNetworking</key>
        <true/>
    </dict>
</dict>
</plist>
EOF

# Icon (if available)
ICON="$SCRIPT_DIR/../assets/classify_icon.icns"
if [ -f "$ICON" ]; then
    cp "$ICON" "$APP_BUNDLE/Contents/Resources/classify_icon.icns"
    # Add icon reference to Info.plist
    /usr/libexec/PlistBuddy -c "Add :CFBundleIconFile string classify_icon" "$APP_BUNDLE/Contents/Info.plist" 2>/dev/null || true
fi

# Entitlements
cat > "$OUTPUT_DIR/CLASSify.entitlements" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.network.client</key>
    <true/>
    <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
    <true/>
    <key>com.apple.security.cs.disable-library-validation</key>
    <true/>
</dict>
</plist>
EOF

# Code sign (if cert available)
ENTITLEMENTS="$OUTPUT_DIR/CLASSify.entitlements"
if [ -n "${APPLE_SIGNING_ID:-}" ]; then
    echo "Signing app bundle..."
    codesign --deep --force --options runtime \
        --entitlements "$ENTITLEMENTS" \
        --sign "$APPLE_SIGNING_ID" \
        "$APP_BUNDLE"
    echo "Verifying signature..."
    codesign --verify --deep --strict "$APP_BUNDLE"
else
    echo "APPLE_SIGNING_ID not set — skipping code signing"
fi

# Build DMG
DMG_NAME="$OUTPUT_DIR/CLASSify-${VERSION}-universal2.dmg"
if command -v create-dmg &>/dev/null; then
    create-dmg \
        --volname "CLASSify Desktop" \
        --window-pos 200 120 \
        --window-size 600 400 \
        --icon-size 100 \
        --icon "CLASSify.app" 150 100 \
        --hide-extension "CLASSify.app" \
        --app-drop-link 450 100 \
        "$DMG_NAME" \
        "$OUTPUT_DIR"
else
    echo "create-dmg not found, using hdiutil..."
    hdiutil create -volname "CLASSify Desktop" \
        -srcfolder "$APP_BUNDLE" \
        -ov -format UDZO \
        "$DMG_NAME"
fi

echo "Built: $DMG_NAME"
