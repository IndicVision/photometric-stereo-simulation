#!/usr/bin/env python3
"""
PhotoStereo — Jetson Server  (Python 3.6.9 / 3.7.5)
Zero external dependencies — stdlib only.

WebSocket : ws://192.168.10.1:8080/ws
Upload    : POST http://192.168.10.1:8080/upload_session
Status    : GET  http://192.168.10.1:8080/status
Result    : GET  http://192.168.10.1:8080/result

FIX vs old version:
  - Pipeline runs in background thread — upload returns 200 immediately
  - /status endpoint lets phone poll without timeout
  - /result endpoint sends HTML back when ready
"""

import base64, hashlib, json, logging, os, shutil, socket, struct, sys, threading, time
try:
    from http.server import BaseHTTPRequestHandler, HTTPServer
except ImportError:
    from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer

# ─────────────────────────────────────────────────────────────────────────────
ESP32_IP   = "192.168.10.2"   # confirmed from arp -a
ESP32_PORT = 80
HOST       = "0.0.0.0"
PORT       = 8080
SAVE_DIR   = os.path.expanduser("~/session_data")
NUM_LIGHTS = 4

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("photostereo")

# ── Job state ─────────────────────────────────────────────────────────────────
# { "status": "idle"|"processing"|"done"|"error",
#   "stage":  "masking"|"reconstruction"|"complete"|"",
#   "html":   path or None,
#   "error":  message or "" }
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
# ESP32 helper
# ─────────────────────────────────────────────────────────────────────────────
def esp32_get(path):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        s.connect((ESP32_IP, ESP32_PORT))
        s.sendall("GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n".format(
            path, ESP32_IP).encode())
        resp = b""
        while True:
            c = s.recv(1024)
            if not c: break
            resp += c
        s.close()
        first = resp.split(b"\r\n")[0].decode("utf-8", errors="ignore")
        ok = "200" in first
        log.info("  ESP32 GET {}  -> {}".format(path, "OK" if ok else first))
        return ok
    except Exception as e:
        log.warning("  ESP32 GET {}  -> FAILED ({})".format(path, e))
        return False

# ─────────────────────────────────────────────────────────────────────────────
# WebSocket (RFC 6455)
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
    if text == "PREVIEW_LIGHT_ON":
        esp32_get("/preview_on")
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
            esp32_get("/light_on?id={}".format(n+1))
            ws_send_text(conn, "TRIGGER_CAPTURE_{}".format(n+1))
        else:
            ws_send_text(conn, "SEQUENCE_COMPLETE")
            log.info("Sequence complete")
    else:
        log.warning("Unknown: {}".format(text))

# ─────────────────────────────────────────────────────────────────────────────
# Multipart parser
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
def run_pipeline_thread(png_files, json_files, save_dir):
    try:
        set_job("processing", "masking")
        from process_pipeline import run_pipeline
        html_path = run_pipeline(png_files, json_files, save_dir)
        set_job("done", "complete", html=str(html_path))
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
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/upload_session":
            self.handle_upload()
        else:
            self._json(404, {"error": "not found"})

    # ── WebSocket ─────────────────────────────────────────────────────────────
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

    # ── Upload — saves files then returns 200 IMMEDIATELY ─────────────────────
    def handle_upload(self):
        ct = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ct:
            self._json(400, {"status":"ERROR","message":"Expected multipart/form-data"}); return
        boundary = next((t.strip()[9:] for t in ct.split(";") if t.strip().startswith("boundary=")), None)
        if not boundary:
            self._json(400, {"status":"ERROR","message":"No boundary"}); return

        cl = int(self.headers.get("Content-Length", 0))
        if cl > 0:
            log.info("Reading {:.1f} MB...".format(cl / 1024**2))
            body = self._read_exact(cl)
        else:
            chunks = []
            while True:
                c = self.rfile.read(65536)
                if not c: break
                chunks.append(c)
            body = b"".join(chunks)

        log.info("Received {:.1f} MB".format(len(body) / 1024**2))
        if not body:
            self._json(400, {"status":"ERROR","message":"Empty body"}); return

        if os.path.exists(SAVE_DIR): shutil.rmtree(SAVE_DIR)
        os.makedirs(SAVE_DIR, exist_ok=True)

        parts = parse_multipart(body, boundary)
        if not parts:
            self._json(400, {"status":"ERROR","message":"No files in body"}); return

        png_files = []; json_files = []
        for filename, data in parts:
            dest = os.path.join(SAVE_DIR, os.path.basename(filename))
            with open(dest, "wb") as f: f.write(data)
            log.info("  Saved: {}  ({:.1f} KB)".format(filename, len(data)/1024))
            if filename.endswith(".png"):  png_files.append(dest)
            elif filename.endswith(".json"): json_files.append(dest)

        log.info("Upload complete: {} PNG(s), {} JSON(s)".format(len(png_files), len(json_files)))

        # ── Start pipeline in background — respond immediately ─────────────────
        set_job("processing", "starting")
        threading.Thread(
            target=run_pipeline_thread,
            args=(png_files, json_files, SAVE_DIR),
            daemon=True
        ).start()

        # 200 returned right away — phone polls /status
        self._json(200, {
            "status":  "FILES_RECEIVED",
            "message": "Processing started. Poll /status for progress.",
            "files":   [f for f, _ in parts]
        })

    # ── Status — phone polls this while pipeline runs ─────────────────────────
    def handle_status(self):
        job = get_job()
        self._json(200, job)

    # ── Result — phone downloads HTML when status == done ─────────────────────
    def handle_result(self):
        job = get_job()
        if job["status"] != "done":
            self._json(409, {"error": "not ready", "status": job["status"]}); return
        html_path = job.get("html")
        if not html_path or not os.path.exists(html_path):
            self._json(404, {"error": "HTML file not found"}); return
        size = os.path.getsize(html_path)
        self.send_response(200)
        self.send_header("Content-Type",        "text/html; charset=utf-8")
        self.send_header("Content-Length",      str(size))
        self.send_header("Content-Disposition", "attachment; filename=surface_3d.html")
        self.end_headers()
        with open(html_path, "rb") as f:
            shutil.copyfileobj(f, self.wfile)
        log.info("HTML sent: {:.1f} KB".format(size/1024))

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

# ─────────────────────────────────────────────────────────────────────────────
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
    log.info("="*56)
    log.info("  PhotoStereo Jetson Server  (Python {})".format(sys.version.split()[0]))
    log.info("  WebSocket : ws://{}:{}/ws".format(HOST, PORT))
    log.info("  Upload    : POST http://{}:{}/upload_session".format(HOST, PORT))
    log.info("  Status    : GET  http://{}:{}/status".format(HOST, PORT))
    log.info("  Result    : GET  http://{}:{}/result".format(HOST, PORT))
    log.info("  ESP32     : http://{}:{}".format(ESP32_IP, ESP32_PORT))
    log.info("  Save dir  : {}".format(SAVE_DIR))
    log.info("="*56)
    server = ThreadedHTTPServer((HOST, PORT), PhotoStereoHandler)
    log.info("Listening... (Ctrl+C to stop)")
    try: server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down."); server.server_close()

if __name__ == "__main__":
    main()
