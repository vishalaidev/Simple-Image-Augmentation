# DJSVS AI - Image Augmentation

Interactive data augmentation tool for vision datasets. Run the GUI to tune probabilities, preview effects on a single image, and batch-generate augmented samples.

## Requirements
- Python 3.8+ (tested on 3.13)
- Pillow (installed automatically in builds; install locally with `python3 -m pip install pillow`)
- Tkinter (usually preinstalled; on Debian/Ubuntu: `sudo apt-get install -y python3-tk`)

## Quick start (GUI)
```bash
python3 augment.py --gui
```

## CLI usage
Generate N augmented images per source in `data/`, output to `augmented/`:
```bash
python3 augment.py --num-per-image 5 --input-dir data --output-dir augmented
```
All augmentation probabilities/ranges are configurable via flags (see `--help`).

## Build binaries
### Linux binary (x86_64)
```bash
bash build_linux.sh
# binary at dist/djsvs-ai-augment
```

### Linux AppImage (x86_64)
Requires `appimagetool` in PATH.
```bash
bash build_appimage.sh
# outputs DJSVS_AI_-_Augment-x86_64.AppImage
```

### Windows .exe (run on Windows)
```powershell
pip install pyinstaller
powershell -ExecutionPolicy Bypass -File build_win.ps1
# outputs dist\djsvs-ai-augment.exe
```

### Jetson / ARM64
Run/build directly on the Jetson (arm64):
1) `sudo apt-get install -y python3-tk`
2) `python3 -m pip install pillow pyinstaller`
3) `bash build_appimage.sh` (with an arm64 `appimagetool`) or run `python3 augment.py --gui`.

## Project layout
- `augment.py` — application entry (GUI + CLI).
- `build_linux.sh` — PyInstaller binary for Linux.
- `build_appimage.sh` — AppImage packaging for Linux.
- `build_win.ps1` — PyInstaller (Windows).
- `a.png` — app/icon asset used in GUI header and AppImage icon (fallback: `logo.png`).

## Notes
- Build scripts include a hidden import for `PIL._tkinter_finder` to avoid missing-module errors in packaged apps.
- AppImage and PyInstaller outputs are not committed; regenerate via scripts.
