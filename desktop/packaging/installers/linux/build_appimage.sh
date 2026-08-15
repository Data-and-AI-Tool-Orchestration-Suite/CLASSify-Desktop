#!/bin/bash
# CLASSify Desktop — Linux AppImage build script
#
# Requires: appimagetool (https://github.com/AppImage/AppImageKit)
# Usage: ./build_appimage.sh
set -euo pipefail

VERSION="${CLASSIFY_VERSION:-1.0.0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
DIST_DIR="$REPO_ROOT/dist/CLASSify"
OUTPUT_DIR="$REPO_ROOT/dist/installers"
APPDIR="$OUTPUT_DIR/CLASSify.AppDir"

mkdir -p "$APPDIR/usr/bin"
mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"

# Copy the PyInstaller bundle
cp -r "$DIST_DIR"/* "$APPDIR/usr/bin/"

# AppRun
cat > "$APPDIR/AppRun" << 'EOF'
#!/bin/bash
HERE="$(dirname "$(readlink -f "${0}")")"
exec "${HERE}/usr/bin/CLASSify" "$@"
EOF
chmod +x "$APPDIR/AppRun"

# .desktop file
cat > "$APPDIR/classify.desktop" << EOF
[Desktop Entry]
Name=CLASSify Desktop
Comment=Local-first ML classification training
Exec=CLASSify
Icon=classify
Type=Application
Categories=Science;ArtificialIntelligence;
Terminal=false
EOF

# Icon (if available)
ICON="$SCRIPT_DIR/../assets/classify_icon.png"
if [ -f "$ICON" ]; then
    cp "$ICON" "$APPDIR/usr/share/icons/hicolor/256x256/apps/classify.png"
    cp "$ICON" "$APPDIR/classify.png"
fi

# Build the AppImage
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    APPIMAGETOOL_ARCH="x86_64"
elif [ "$ARCH" = "aarch64" ]; then
    APPIMAGETOOL_ARCH="aarch64"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

OUTPUT_NAME="$OUTPUT_DIR/CLASSify-${VERSION}-${APPIMAGETOOL_ARCH}.AppImage"

if command -v appimagetool &>/dev/null; then
    appimagetool "$APPDIR" "$OUTPUT_NAME"
elif [ -f "$SCRIPT_DIR/appimagetool-$APPIMAGETOOL_ARCH.AppImage" ]; then
    "$SCRIPT_DIR/appimagetool-$APPIMAGETOOL_ARCH.AppImage" "$APPDIR" "$OUTPUT_NAME"
else
    echo "appimagetool not found. Download from https://github.com/AppImage/AppImageKit/releases"
    exit 1
fi

echo "Built: $OUTPUT_NAME"
