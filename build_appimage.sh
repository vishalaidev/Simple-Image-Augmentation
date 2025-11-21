#!/usr/bin/env bash
set -euo pipefail

APP_NAME="djsvs-ai-augment"
PYTHON_BIN="${PYTHON_BIN:-python3}"
APPDIR="AppDir"

echo "Using interpreter: ${PYTHON_BIN}"

if ! command -v pyinstaller >/dev/null 2>&1; then
  echo "pyinstaller not found. Install with: ${PYTHON_BIN} -m pip install pyinstaller"
  exit 1
fi

if ! command -v appimagetool >/dev/null 2>&1; then
  echo "appimagetool not found. Install it (e.g., sudo apt-get install appimagetool or download AppImage from https://github.com/AppImage/AppImageKit/releases)"
  exit 1
fi

echo "Building standalone binary with PyInstaller..."
pyinstaller --noconfirm --noconsole --onefile --hidden-import PIL._tkinter_finder --name "${APP_NAME}" augment.py

echo "Creating AppDir layout..."
rm -rf "${APPDIR}"
mkdir -p "${APPDIR}/usr/bin" "${APPDIR}/usr/share/applications" "${APPDIR}/usr/share/icons/hicolor/256x256/apps"

echo "Copying binary..."
cp "dist/${APP_NAME}" "${APPDIR}/usr/bin/${APP_NAME}"

DESKTOP_FILE="${APPDIR}/${APP_NAME}.desktop"
cat > "${DESKTOP_FILE}" <<EOF
[Desktop Entry]
Type=Application
Name=DJSVS AI - Augment
Exec=${APP_NAME} --gui
Icon=${APP_NAME}
Categories=Utility;
Terminal=false
EOF
# Also place a copy under usr/share/applications for completeness
cp "${DESKTOP_FILE}" "${APPDIR}/usr/share/applications/${APP_NAME}.desktop"

# Create AppRun launcher
cat > "${APPDIR}/AppRun" <<'EOF'
#!/bin/bash
HERE="$(dirname "$(readlink -f "$0")")"
export PATH="$HERE/usr/bin:$PATH"
exec "${HERE}/usr/bin/djsvs-ai-augment" --gui "$@"
EOF
chmod +x "${APPDIR}/AppRun"

ICON_SRC="a.png"
ICON_DST="${APPDIR}/usr/share/icons/hicolor/256x256/apps/${APP_NAME}.png"
if [[ -f "${ICON_SRC}" ]]; then
  echo "Generating 256x256 icon from ${ICON_SRC}..."
  "${PYTHON_BIN}" - <<'PY'
from PIL import Image
src = "a.png"
dst = "tmp_icon.png"
im = Image.open(src).convert("RGBA")
im.thumbnail((256, 256))
canvas = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
canvas.paste(im, ((256 - im.width) // 2, (256 - im.height) // 2), im)
canvas.save(dst)
PY
  mv tmp_icon.png "${ICON_DST}"
  cp "${ICON_DST}" "${APPDIR}/${APP_NAME}.png"
  ln -sf "${APP_NAME}.png" "${APPDIR}/.DirIcon"
else
  echo "a.png not found, skipping icon generation."
fi

echo "Building AppImage..."
appimagetool "${APPDIR}"

echo "Done. Look for a .AppImage file in the current directory."
