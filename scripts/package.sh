#!/bin/zsh
# Build the command-line miner and zip it for a release:
# build/macotaur-<version>-macos-arm64.zip. The binary is ad-hoc signed.
set -euo pipefail
cd "$(dirname "$0")/.."

swift build -c release --product macotaur
DIR=build/macotaur
rm -rf build
mkdir -p "$DIR"
cp .build/release/macotaur LICENSE THIRD_PARTY_NOTICES "$DIR/"
# Debug info records the build machine's paths; a release doesn't need it.
strip -S -x "$DIR/macotaur"
codesign --force --sign - "$DIR/macotaur"
VERSION=$("$DIR/macotaur" --version | awk '{print $2}')
ZIP="build/macotaur-$VERSION-macos-arm64.zip"
ditto -c -k --norsrc --noextattr --keepParent "$DIR" "$ZIP"
echo "Packaged $ZIP"
