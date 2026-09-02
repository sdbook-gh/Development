#!/bin/bash
# wsl-guest-fix.sh - configure WSL guest eth0/route/DNS
set -euo pipefail

GATEWAY="172.19.16.1"
IP="172.19.16.2"
MASK="20"
IFACE="eth0"

echo "[guest] iface=$IFACE addr=${IP}/${MASK} gw=$GATEWAY"

echo "[guest] remove old IPv4 addresses..."
mapfile -t oldaddrs < <(ip -o -4 addr show dev "$IFACE" | awk '{print $4}')
for a in "${oldaddrs[@]}"; do
  if [ "$a" != "${IP}/${MASK}" ]; then
    ip addr del "$a" dev "$IFACE" 2>/dev/null && echo "        removed $a"
  fi
done

if ip -o -4 addr show dev "$IFACE" | awk '{print $4}' | grep -qx "${IP}/${MASK}"; then
  echo "[guest] ${IP}/${MASK} already present"
else
  echo "[guest] add ${IP}/${MASK}"
  ip addr add "${IP}/${MASK}" dev "$IFACE"
fi

echo "[guest] default route -> $GATEWAY"
ip -4 route replace default via "$GATEWAY" dev "$IFACE"

echo "[guest] DNS nameserver=$GATEWAY"
if [ -L /etc/resolv.conf ]; then
  rm -f /etc/resolv.conf
fi
printf 'nameserver %s\n' "$GATEWAY" > /etc/resolv.conf

WSLCONF="/etc/wsl.conf"
if [ -f "$WSLCONF" ]; then
  if ! grep -qE 'generateResolvConf' "$WSLCONF"; then
    printf '\n[network]\ngenerateResolvConf = false\n' >> "$WSLCONF"
    echo "[guest] appended generateResolvConf=false to $WSLCONF"
  fi
else
  printf '[network]\ngenerateResolvConf = false\n' > "$WSLCONF"
  echo "[guest] created $WSLCONF"
fi

echo "[guest] result:"
ip -4 addr show dev "$IFACE"
ip -4 route show default
echo "[guest] done"