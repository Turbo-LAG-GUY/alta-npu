#!/bin/sh
# install.sh: menu entry + icon for Alta Chat, in ~/.local/share only (no root). --remove undoes it.
set -e
C="$(cd "$(dirname "$0")" && pwd)"
A=$HOME/.local/share/applications/io.github.turbolagguy.AltaChat.desktop
I=$HOME/.local/share/icons/hicolor/scalable/apps/io.github.turbolagguy.AltaChat.svg
if [ "${1:-}" = --remove ]; then rm -f "$A" "$I"; echo "removed"; exit 0; fi
mkdir -p "$(dirname "$A")" "$(dirname "$I")"
cp "$C/icons/hicolor/scalable/apps/io.github.turbolagguy.AltaChat.svg" "$I"
cat > "$A" <<D
[Desktop Entry]
Type=Application
Name=Alta Chat
Comment=Chat with SmolLM2 on the A311D's NPU
Exec=$C/alta-chat
Icon=io.github.turbolagguy.AltaChat
Terminal=false
Categories=Utility;Development;
StartupWMClass=io.github.turbolagguy.AltaChat
D
gtk-update-icon-cache -q -t "$HOME/.local/share/icons/hicolor" 2>/dev/null || true
update-desktop-database -q "$HOME/.local/share/applications" 2>/dev/null || true
echo "installed: $A"
