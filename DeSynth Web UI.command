#!/bin/zsh
set -eu

ROOT_DIR="${0:A:h}"
BUILD_DIR="$ROOT_DIR/.webui-build"
SOURCE="$ROOT_DIR/webui.swift"
BINARY="$BUILD_DIR/DeSynthWebUI"

if [[ ! -f "$ROOT_DIR/web/index.html" || ! -f "$ROOT_DIR/webui_worker.py" ]]; then
  print "DeSynth Web UI 文件不完整。"
  read -k 1 "?按任意键退出…"
  exit 1
fi

if [[ ! -x "$BINARY" || "$SOURCE" -nt "$BINARY" ]]; then
  mkdir -p "$BUILD_DIR"
  print "首次启动：正在构建本机 WebKit 界面…"
  SDK_PATH=""
  for CANDIDATE in \
    "/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk" \
    "/Library/Developer/CommandLineTools/SDKs/MacOSX15.sdk"; do
    if [[ -d "$CANDIDATE" ]]; then
      SDK_PATH="$CANDIDATE"
      break
    fi
  done
  if [[ -z "$SDK_PATH" ]]; then
    SDK_PATH="$(xcrun --sdk macosx --show-sdk-path)"
  fi
  CLANG_MODULE_CACHE_PATH="$BUILD_DIR/module-cache" xcrun swiftc "$SOURCE" \
    -sdk "$SDK_PATH" \
    -target arm64-apple-macosx14.0 \
    -swift-version 5 \
    -o "$BINARY" \
    -framework AppKit \
    -framework WebKit \
    -framework ImageIO
fi

cd "$ROOT_DIR"
exec "$BINARY" --root "$ROOT_DIR"
