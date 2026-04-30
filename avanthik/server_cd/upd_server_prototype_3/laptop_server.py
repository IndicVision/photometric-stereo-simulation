#!/usr/bin/env python3
import os
# Restrict C-math libraries to 1 thread to prevent PyAMG cache thrashing
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
"""
PhotoStereo — Laptop Server
Replaces the Jetson server. Runs on any laptop (Windows / Mac / Linux).

mDNS  : This server announces itself as  photostereo.local
          so the phone and ESP32 can find it without knowing the IP.
ESP32 : The ESP32 announces itself as  esp32ps.local  after connecting.
          This server resolves that name to an IP before the first capture.

Endpoints (identical to the Jetson server — no Android changes needed):
  WebSocket : ws://photostereo.local:8080/ws
  Upload    : POST http://photostereo.local:8080/upload_session
  Status    : GET  http://photostereo.local:8080/status
  Result    : GET  http://photostereo.local:8080/result

Requirements (pip install):
  zeroconf
"""

import urllib.request
import urllib.error
import base64, hashlib, json, logging, os, shutil, socket, struct, sys, threading, time
import calibrate_lights   # new import

try:
    from http.server import BaseHTTPRequestHandler, HTTPServer
except ImportError:
    from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer

# ── mDNS (zeroconf) ───────────────────────────────────────────────────────────
try:
    from zeroconf import ServiceInfo, Zeroconf
    import socket as _socket
    HAS_ZEROCONF = True
except ImportError:
    HAS_ZEROCONF = False

# ─────────────────────────────────────────────────────────────────────────────
HOST       = "0.0.0.0"
PORT       = 8080
SAVE_DIR   = os.path.expanduser("~/photostereo_sessions")
NUM_LIGHTS = 4

# ESP32 discovery
ESP32_MDNS_NAME = "esp32ps.local"   # mDNS name the ESP32 announces
ESP32_PORT      = 80
_esp32_ip       = None              # resolved lazily before first capture
_esp32_ip_lock  = threading.Lock()

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("photostereo")

# ── Job state ─────────────────────────────────────────────────────────────────
_job = {"status": "idle", "stage": "", "html": None, "error": ""}
_job_lock = threading.Lock()

def set_job(status, stage="", html=None, error=""):
    with _job_lock:
        _job["status"] = status
        _job["stage"]  = stage
        _job["html"]   = html
        _job["error"]  = error
    log.info("[JOB] {} / {}".format(status, stage))

def get_job():
    with _job_lock:
        return dict(_job)

# ─────────────────────────────────────────────────────────────────────────────
# mDNS — announce this server as photostereo.local
# ─────────────────────────────────────────────────────────────────────────────
# In laptop_server.py, replace get_local_ip() with this:
def get_local_ip():
    """Return the IP on the active hotspot/WiFi interface."""
    import subprocess
    candidates = []
    try:
        # Get all IPv4 addresses on this machine
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                candidates.append(ip)
    except Exception:
        pass
    
    # Prefer hotspot subnet (Windows Mobile Hotspot is always 192.168.137.x)
    for ip in candidates:
        if ip.startswith("192.168.137."):
            return ip
    
    # Fallback: original method
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return candidates[0] if candidates else "127.0.0.1"

_zeroconf_instance = None

def start_mdns_announcement():
    global _zeroconf_instance
    if not HAS_ZEROCONF:
        log.warning("zeroconf not installed — mDNS disabled. "
                    "Install with: pip install zeroconf")
        log.warning("Phone must use manual IP entry fallback.")
        return

    local_ip = get_local_ip()
    log.info("Local IP detected: {}".format(local_ip))

    info = ServiceInfo(
        "_http._tcp.local.",
        "PhotoStereo Server._http._tcp.local.",
        addresses=[socket.inet_aton(local_ip)],
        port=PORT,
        properties={"version": "1.0", "path": "/"},
        server="photostereo.local.",
    )

    _zeroconf_instance = Zeroconf()
    _zeroconf_instance.register_service(info)
    log.info("mDNS: server announced as photostereo.local:{}".format(PORT))

def stop_mdns_announcement():
    if _zeroconf_instance:
        _zeroconf_instance.close()

# ─────────────────────────────────────────────────────────────────────────────
# ESP32 discovery — resolve esp32ps.local → IP
# ─────────────────────────────────────────────────────────────────────────────
def resolve_esp32_ip(timeout_s=10):
    """
    Try to resolve esp32ps.local via the OS DNS/mDNS stack.
    Returns IP string on success, None on failure.
    Works on Mac (Bonjour built-in) and Linux (avahi).
    On Windows, requires Bonjour for Windows or iTunes installed.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            ip = socket.gethostbyname(ESP32_MDNS_NAME)
            log.info("ESP32 resolved: {} -> {}".format(ESP32_MDNS_NAME, ip))
            return ip
        except socket.gaierror:
            time.sleep(1)
    log.warning("Could not resolve {} after {}s".format(ESP32_MDNS_NAME, timeout_s))
    return None

def get_esp32_ip():
    """Return cached ESP32 IP, resolving lazily on first call."""
    global _esp32_ip
    with _esp32_ip_lock:
        if _esp32_ip is None:
            _esp32_ip = resolve_esp32_ip()
        return _esp32_ip

# ─────────────────────────────────────────────────────────────────────────────
# ESP32 HTTP helper
# ─────────────────────────────────────────────────────────────────────────────
def esp32_get(path):
    ip = get_esp32_ip()
    if ip is None:
        log.warning("ESP32 IP unknown — cannot send: {}".format(path))
        return False
    url = "http://{}:{}{}".format(ip, ESP32_PORT, path)
    try:
        with urllib.request.urlopen(url, timeout=1.5) as response:
            resp_text = response.read().decode('utf-8', errors='ignore').strip()
            ok = response.getcode() == 200
            log.info("  ESP32 GET {}  -> HTTP {} ({})".format(path, response.getcode(), resp_text))
            return ok
    except Exception as e:
        log.warning("  ESP32 GET {}  -> FAILED ({})".format(path, e))
        return False

# ─────────────────────────────────────────────────────────────────────────────
# WebSocket (RFC 6455) — identical to Jetson version
# ─────────────────────────────────────────────────────────────────────────────
WS_MAGIC = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

def ws_handshake(conn, headers):
    key    = headers.get("Sec-WebSocket-Key", "").strip()
    accept = base64.b64encode(hashlib.sha1(
        (key + WS_MAGIC).encode()).digest()).decode()
    conn.sendall((
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\nConnection: Upgrade\r\n"
        "Sec-WebSocket-Accept: {}\r\n\r\n").format(accept).encode())

def ws_recv_frame(rfile):
    try:
        h = rfile.read(2)
        if len(h) < 2: return None, None
        opcode = h[0] & 0x0F
        masked = (h[1] & 0x80) != 0
        length = h[1] & 0x7F
        if length == 126: length = struct.unpack(">H", rfile.read(2))[0]
        elif length == 127: length = struct.unpack(">Q", rfile.read(8))[0]
        mask_key = rfile.read(4) if masked else b"\x00\x00\x00\x00"
        data = bytearray(rfile.read(length))
        if masked:
            for i in range(len(data)): data[i] ^= mask_key[i % 4]
        if opcode == 8: return None, None
        return opcode, data.decode("utf-8", errors="ignore")
    except Exception:
        return None, None

def ws_send_text(conn, text):
    payload = text.encode("utf-8")
    n = len(payload)
    if n <= 125:     hdr = struct.pack("BB", 0x81, n)
    elif n <= 65535: hdr = struct.pack(">BBH", 0x81, 126, n)
    else:            hdr = struct.pack(">BBQ", 0x81, 127, n)
    try:
        conn.sendall(hdr + payload)
        log.info("-> Phone: {}".format(text))
        return True
    except Exception as e:
        log.warning("ws_send_text failed: {}".format(e))
        return False

def handle_ws_message(text, conn):
    text = text.strip()
    log.info("<- Phone: {}".format(text))

    if text.startswith("SET_PREVIEW:"):
        light_ids = text.split(":")[1]
        esp32_get("/set_lights?ids={}".format(light_ids))

    elif text == "PREVIEW_LIGHT_OFF":
        esp32_get("/all_off")

    elif text == "START":
        esp32_get("/light_on?id=1")
        ws_send_text(conn, "TRIGGER_CAPTURE_1")

    elif text.startswith("DONE_"):
        try: n = int(text[5:])
        except ValueError: return
        esp32_get("/all_off")
        if n < NUM_LIGHTS:
            esp32_get("/light_on?id={}".format(n + 1))
            ws_send_text(conn, "TRIGGER_CAPTURE_{}".format(n + 1))
        else:
            ws_send_text(conn, "SEQUENCE_COMPLETE")
            log.info("Capture sequence complete")
    else:
        log.warning("Unknown WS message: {}".format(text))

# ─────────────────────────────────────────────────────────────────────────────
# Multipart parser — identical to Jetson version
# ─────────────────────────────────────────────────────────────────────────────
def parse_multipart(body, boundary):
    delim   = ("--" + boundary).encode()
    results = []
    for raw in body.split(delim):
        raw = raw.lstrip(b"\r\n")
        if not raw or raw.startswith(b"--"): continue
        if b"\r\n\r\n" not in raw: continue
        hdr, _, part_body = raw.partition(b"\r\n\r\n")
        part_body = part_body.rstrip(b"\r\n")
        filename  = None
        for line in hdr.decode("utf-8", errors="ignore").splitlines():
            if "Content-Disposition" in line and "filename=" in line:
                for tok in line.split(";"):
                    tok = tok.strip()
                    if tok.lower().startswith("filename="):
                        filename = tok[9:].strip('"').strip("'"); break
            if filename: break
        if filename:
            results.append((filename, part_body))
    return results

# ─────────────────────────────────────────────────────────────────────────────
# Background pipeline thread
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Background pipeline thread
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Background pipeline thread
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline_thread(png_files, json_files, save_dir):
    try:
        set_job("processing", "masking")
        from process_pipeline import run_pipeline   # matches process_pipeline.py filename
        
        # Capture both the lightweight phone zip and heavy desktop zip
        phone_zip, desktop_zip = run_pipeline(png_files, json_files, save_dir)
        
        # ─── SAVE THE HEAVY DEBUG ZIP TO WINDOWS DESKTOP ─────────────────────────
        try:
            desktop_dir = r"C:\Users\chand\OneDrive\Desktop\PhotoStereo_Scans"
            os.makedirs(desktop_dir, exist_ok=True)
            
            safe_filename = "scan_debug_full_{}.zip".format(int(time.time()))
            desktop_zip_path = os.path.join(desktop_dir, safe_filename)
            
            shutil.copy2(desktop_zip, desktop_zip_path)
            log.info("💾 AUTOMATIC SAVE: Copied heavy debug zip to {}".format(desktop_zip_path))
        except Exception as e:
            log.error("Failed to save to Desktop: {}".format(e))
        # ─────────────────────────────────────────────────────────────────────────

        # Tell the server job it is done, and give it the LIGHTWEIGHT zip to send to the phone
        set_job("done", "complete", html=str(phone_zip))
    except Exception as e:
        import traceback
        traceback.print_exc()
        set_job("error", "failed", error=str(e))

# ─────────────────────────────────────────────────────────────────────────────
# HTTP handler
# ─────────────────────────────────────────────────────────────────────────────
class PhotoStereoHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args): pass

    def do_GET(self):
        if self.path == "/ws":
            self.handle_websocket()
        elif self.path == "/status":
            self.handle_status()
        elif self.path == "/result":
            self.handle_result()
        elif self.path == "/calibration_result":
            result_path = os.path.join(SAVE_DIR, "calibration", "calibration_result.json")
            if os.path.exists(result_path):
                with open(result_path) as f:
                    body = f.read().encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json_response(404, {"status": "not_ready"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/upload_calibration":
            self._handle_calibration_upload()
            return
        elif self.path == "/upload_session":
            self.handle_upload()
        else:
            self._json(404, {"error": "not found"})

    def _handle_calibration_upload(self):
        """
        Receives the four white-paper PNG files from the phone,
        saves them to ~/photostereo_sessions/calibration/,
        runs calibrate_lights.py, and returns the factors as JSON.
        No masking. No reconstruction. Completely separate from the scan pipeline.
        """
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._json_response(400, {"status": "error", "message": "Empty body"})
            return
        body = self.rfile.read(length)

        # Wipe and recreate calibration subfolder
        calib_dir = os.path.join(SAVE_DIR, "calibration")
        if os.path.exists(calib_dir):
            shutil.rmtree(calib_dir)
        os.makedirs(calib_dir)

        # Parse multipart — reuse existing parse_multipart helper
        ct = self.headers.get("Content-Type", "")
        boundary = next((t.strip()[9:] for t in ct.split(";") if t.strip().startswith("boundary=")), None)
        if not boundary:
            self._json_response(400, {"status": "error", "message": "No boundary"})
            return
            
        parts = parse_multipart(body, boundary)
        png_files = parts   # existing function
        
        if len(png_files) < 4:
            self._json_response(400, {
                "status": "error",
                "message": f"Expected 4 PNG files, got {len(png_files)}"
            })
            return

        # Save PNGs
        saved = []
        for name, data in png_files:
            dest = os.path.join(calib_dir, name)
            with open(dest, "wb") as f:
                f.write(data)
            saved.append(dest)
        
        log.info(f"Calibration: saved {len(saved)} images to {calib_dir}")

        # Run calibration — synchronous, fast (no GPU, no masking)
        try:
            from pathlib import Path
            result = calibrate_lights.run(Path(calib_dir))
        except Exception as e:
            log.error(f"Calibration failed: {e}")
            self._json_response(500, {"status": "error", "message": str(e)})
            return
            
        self._json_response(200, result)

    def _json_response(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_websocket(self):
        if self.headers.get("Upgrade", "").lower() != "websocket":
            self.send_response(400); self.end_headers(); return
        log.info("Phone connected (WebSocket)")
        conn = self.request
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        ws_handshake(conn, self.headers)
        while True:
            opcode, text = ws_recv_frame(self.rfile)
            if opcode is None: break
            if opcode == 1 and text: handle_ws_message(text, conn)
        log.info("Phone disconnected")
        esp32_get("/all_off")

    def handle_upload(self):
        ct = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ct:
            self._json(400, {"status": "ERROR", "message": "Expected multipart/form-data"}); return
        boundary = next((t.strip()[9:] for t in ct.split(";") if t.strip().startswith("boundary=")), None)
        if not boundary:
            self._json(400, {"status": "ERROR", "message": "No boundary"}); return

        cl = int(self.headers.get("Content-Length", 0))
        if cl > 0:
            log.info("Reading {:.1f} MB...".format(cl / 1024 ** 2))
            body = self._read_exact(cl)
        else:
            chunks = []
            while True:
                c = self.rfile.read(65536)
                if not c: break
                chunks.append(c)
            body = b"".join(chunks)

        log.info("Received {:.1f} MB".format(len(body) / 1024 ** 2))
        if not body:
            self._json(400, {"status": "ERROR", "message": "Empty body"}); return

        if os.path.exists(SAVE_DIR): shutil.rmtree(SAVE_DIR)
        os.makedirs(SAVE_DIR, exist_ok=True)

        parts = parse_multipart(body, boundary)
        if not parts:
            self._json(400, {"status": "ERROR", "message": "No files in body"}); return

        png_files = []; json_files = []
        for filename, data in parts:
            dest = os.path.join(SAVE_DIR, os.path.basename(filename))
            with open(dest, "wb") as f: f.write(data)
            log.info("  Saved: {}  ({:.1f} KB)".format(filename, len(data) / 1024))
            if filename.endswith(".png"):   png_files.append(dest)
            elif filename.endswith(".json"): json_files.append(dest)

        log.info("Upload complete: {} PNG(s), {} JSON(s)".format(len(png_files), len(json_files)))

        set_job("processing", "starting")
        threading.Thread(
            target=run_pipeline_thread,
            args=(png_files, json_files, SAVE_DIR),
            daemon=True
        ).start()

        self._json(200, {
            "status":  "FILES_RECEIVED",
            "message": "Processing started. Poll /status for progress.",
            "files":   [f for f, _ in parts]
        })

    def handle_status(self):
        self._json(200, get_job())

    def handle_result(self):
        job = get_job()
        if job["status"] != "done":
            self._json(409, {"error": "not ready", "status": job["status"]}); return

        zip_path = job.get("html")
        if not zip_path or not os.path.exists(zip_path):
            self._json(404, {"error": "ZIP file not found"}); return

        size = os.path.getsize(zip_path)
        self.send_response(200)
        self.send_header("Content-Type",        "application/zip")
        self.send_header("Content-Length",      str(size))
        self.send_header("Content-Disposition", "attachment; filename=results.zip")
        self.end_headers()
        with open(zip_path, "rb") as f:
            shutil.copyfileobj(f, self.wfile)
        log.info("ZIP sent: {:.1f} KB".format(size / 1024))

    def _read_exact(self, n):
        buf = bytearray()
        rem = n
        while rem > 0:
            c = self.rfile.read(min(65536, rem))
            if not c: break
            buf.extend(c); rem -= len(c)
        return bytes(buf)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ThreadedHTTPServer(HTTPServer):
    allow_reuse_address = True
    def process_request(self, req, addr):
        threading.Thread(target=self._h, args=(req, addr), daemon=True).start()
    def _h(self, req, addr):
        try: self.finish_request(req, addr)
        except Exception as e: log.warning("Conn error {}: {}".format(addr, e))
        finally: self.shutdown_request(req)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    log.info("=" * 60)
    log.info("  PhotoStereo Laptop Server  (Python {})".format(sys.version.split()[0]))
    log.info("  WebSocket : ws://photostereo.local:{}/ws".format(PORT))
    log.info("  Upload    : POST http://photostereo.local:{}/upload_session".format(PORT))
    log.info("  Status    : GET  http://photostereo.local:{}/status".format(PORT))
    log.info("  Result    : GET  http://photostereo.local:{}/result".format(PORT))
    log.info("  Save dir  : {}".format(SAVE_DIR))
    log.info("=" * 60)

    start_mdns_announcement()

    server = ThreadedHTTPServer((HOST, PORT), PhotoStereoHandler)
    log.info("Listening on port {}... (Ctrl+C to stop)".format(PORT))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down.")
        stop_mdns_announcement()
        server.server_close()


if __name__ == "__main__":
    main()