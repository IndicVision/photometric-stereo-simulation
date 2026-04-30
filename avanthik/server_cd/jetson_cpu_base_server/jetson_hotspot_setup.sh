#!/bin/bash
# =============================================================================
#  PhotoStereo — Jetson Nano Hotspot Setup (Option C)
#  Run once on the Jetson. After this, the Jetson creates the WiFi hotspot
#  automatically on every boot. ESP32 and phone connect to it.
#
#  Network layout after setup:
#    Jetson        192.168.10.1  (gateway, hotspot host)
#    ESP32         192.168.10.2  (DHCP, static lease by MAC)
#    Phone         192.168.10.x  (DHCP, dynamic)
#
#  Usage:
#    chmod +x jetson_hotspot_setup.sh
#    sudo ./jetson_hotspot_setup.sh
# =============================================================================

set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[SETUP]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $1"; }
fail() { echo -e "${RED}[FAIL]${NC}  $1"; exit 1; }

# ── Must run as root ──────────────────────────────────────────────────────────
[ "$EUID" -eq 0 ] || fail "Please run as root: sudo ./jetson_hotspot_setup.sh"

# ── WiFi interface (hardcoded) ────────────────────────────────────────────────
WIFI_IF="wlan0"
log "WiFi interface: $WIFI_IF (hardcoded)"

# ── Configuration (edit these if needed) ─────────────────────────────────────
SSID="Photostereo_Jetson"
PASSWORD="12345678"
CHANNEL="6"
JETSON_IP="192.168.10.1"
DHCP_RANGE_START="192.168.10.10"
DHCP_RANGE_END="192.168.10.50"
SUBNET="192.168.10.0"
NETMASK="255.255.255.0"

log "Hotspot SSID     : $SSID"
log "Hotspot Password : $PASSWORD"
log "Jetson IP        : $JETSON_IP"
log "DHCP Range       : $DHCP_RANGE_START - $DHCP_RANGE_END"

# ── Install dependencies ──────────────────────────────────────────────────────
log "Installing hostapd and dnsmasq..."
apt-get update -q
apt-get install -y hostapd dnsmasq

# ── Stop NetworkManager from managing this interface ─────────────────────────
log "Configuring NetworkManager to ignore $WIFI_IF..."
NM_CONF="/etc/NetworkManager/conf.d/photostereo-unmanaged.conf"
cat > "$NM_CONF" << EOF
[keyfile]
unmanaged-devices=interface-name:$WIFI_IF
EOF
systemctl reload NetworkManager 2>/dev/null || true

# ── Assign static IP to the WiFi interface ───────────────────────────────────
log "Setting static IP $JETSON_IP on $WIFI_IF..."
IP_CONF="/etc/systemd/network/10-photostereo-ap.network"
mkdir -p /etc/systemd/network
cat > "$IP_CONF" << EOF
[Match]
Name=$WIFI_IF

[Network]
Address=$JETSON_IP/24
EOF

# Also set it immediately for this session
ip addr flush dev "$WIFI_IF" 2>/dev/null || true
ip addr add "$JETSON_IP/24" dev "$WIFI_IF" 2>/dev/null || true
ip link set "$WIFI_IF" up

# ── Configure hostapd ─────────────────────────────────────────────────────────
log "Writing hostapd config..."
cat > /etc/hostapd/hostapd.conf << EOF
# PhotoStereo WiFi Hotspot
interface=$WIFI_IF
driver=nl80211
ssid=$SSID
hw_mode=g
channel=$CHANNEL
wmm_enabled=0
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0

# WPA2 security
wpa=2
wpa_passphrase=$PASSWORD
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP
EOF

# Point hostapd at our config
sed -i 's|^#DAEMON_CONF=.*|DAEMON_CONF="/etc/hostapd/hostapd.conf"|' \
    /etc/default/hostapd 2>/dev/null || \
    echo 'DAEMON_CONF="/etc/hostapd/hostapd.conf"' >> /etc/default/hostapd

# ── Configure dnsmasq (DHCP server) ──────────────────────────────────────────
log "Writing dnsmasq config..."

# Back up original config
[ -f /etc/dnsmasq.conf ] && cp /etc/dnsmasq.conf /etc/dnsmasq.conf.backup

cat > /etc/dnsmasq.conf << EOF
# PhotoStereo DHCP server
interface=$WIFI_IF
bind-interfaces

# DHCP range and lease time
dhcp-range=$DHCP_RANGE_START,$DHCP_RANGE_END,$NETMASK,24h

# Jetson gateway
dhcp-option=3,$JETSON_IP

# DNS — point to Jetson itself (no internet needed)
dhcp-option=6,$JETSON_IP
server=8.8.8.8

# Hostname resolution
local=/$SSID/
address=/jetson.local/$JETSON_IP

# IMPORTANT: Reserve .2 for ESP32 (update MAC after first connection)
# Step 1: Connect ESP32, run: cat /var/lib/misc/dnsmasq.leases
# Step 2: Uncomment and set ESP32 MAC below, then: sudo systemctl restart dnsmasq
# dhcp-host=AA:BB:CC:DD:EE:FF,esp32,192.168.10.2,infinite

EOF

# ── Enable IP forwarding (optional, for internet sharing later) ───────────────
log "Enabling IP forwarding..."
echo "net.ipv4.ip_forward=1" > /etc/sysctl.d/photostereo.conf
sysctl -p /etc/sysctl.d/photostereo.conf 2>/dev/null || true

# ── Create startup script that brings up the hotspot in correct order ─────────
log "Creating hotspot startup script..."
STARTUP_SCRIPT="/usr/local/bin/photostereo-hotspot-start.sh"
cat > "$STARTUP_SCRIPT" << SCRIPT
#!/bin/bash
# PhotoStereo hotspot startup — runs before hostapd and dnsmasq
WIFI_IF="$WIFI_IF"
JETSON_IP="$JETSON_IP"

# Bring interface up and assign IP
ip link set \$WIFI_IF up
sleep 1
ip addr flush dev \$WIFI_IF 2>/dev/null || true
ip addr add \$JETSON_IP/24 dev \$WIFI_IF 2>/dev/null || true
echo "PhotoStereo hotspot interface ready: \$WIFI_IF @ \$JETSON_IP"
SCRIPT
chmod +x "$STARTUP_SCRIPT"

# ── Create systemd service for hotspot pre-setup ──────────────────────────────
log "Creating systemd service: photostereo-hotspot..."
cat > /etc/systemd/system/photostereo-hotspot.service << EOF
[Unit]
Description=PhotoStereo WiFi Hotspot Pre-Setup
Before=hostapd.service dnsmasq.service
After=network.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=$STARTUP_SCRIPT

[Install]
WantedBy=multi-user.target
EOF

# ── Enable all services ───────────────────────────────────────────────────────
log "Enabling services to start on boot..."
systemctl daemon-reload
systemctl unmask hostapd 2>/dev/null || true
systemctl enable photostereo-hotspot
systemctl enable hostapd
systemctl enable dnsmasq

# ── Start services now ────────────────────────────────────────────────────────
log "Starting hotspot services..."
systemctl start photostereo-hotspot || warn "pre-setup service issue — check logs"
systemctl restart hostapd  || warn "hostapd start issue — check: journalctl -u hostapd"
systemctl restart dnsmasq  || warn "dnsmasq start issue — check: journalctl -u dnsmasq"

sleep 2

# ── Verify ────────────────────────────────────────────────────────────────────
log "Verifying..."
echo ""
echo "── hostapd status ──"
systemctl is-active hostapd && echo "  ✅ hostapd running" || echo "  ❌ hostapd not running"
echo ""
echo "── dnsmasq status ──"
systemctl is-active dnsmasq && echo "  ✅ dnsmasq running" || echo "  ❌ dnsmasq not running"
echo ""
echo "── Interface ──"
ip addr show "$WIFI_IF" | grep "inet "

echo ""
echo "============================================================"
echo "  PhotoStereo Hotspot Setup Complete!"
echo "============================================================"
echo ""
echo "  SSID       : $SSID"
echo "  Password   : $PASSWORD"
echo "  Jetson IP  : $JETSON_IP"
echo "  Channel    : $CHANNEL"
echo ""
echo "  Next steps:"
echo "  1. Reboot Jetson: sudo reboot"
echo "  2. Flash updated ESP32 firmware (STA mode, connects to $SSID)"
echo "  3. Connect phone to WiFi: $SSID"
echo "  4. Start WebSocket relay: python3 photostereo/esp32_relay.py"
echo "  5. Start PS server:       python3 photostereo/jetson_server.py"
echo ""
echo "  After ESP32 connects, find its MAC for static IP:"
echo "  cat /var/lib/misc/dnsmasq.leases"
echo "  Then set dhcp-host= in /etc/dnsmasq.conf"
echo "============================================================"
