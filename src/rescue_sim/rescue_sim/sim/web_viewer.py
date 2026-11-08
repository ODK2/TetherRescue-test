"""브라우저 뷰어: 오프스크린(CPU) 렌더 → JPEG 스트림(MJPEG) + 키/마우스 입력.

WSLg 창이 안 보이는 환경용. WSL2 는 localhost 를 Windows 와 공유 → Windows 브라우저에서 열림.
반드시 mujoco import 전에 `setup_cpu_gl()` 를 불러야 한다 (MUJOCO_GL 은 import 시점에 정해짐).
"""
from __future__ import annotations

import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import parse_qs, urlparse


def setup_cpu_gl():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")   # llvmpipe (CPU)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px monospace}
#v{display:block;width:100vw;height:100vh;object-fit:contain;cursor:grab;user-select:none}
#s{position:fixed;top:8px;left:8px;background:#000a;padding:8px;white-space:pre;pointer-events:none}</style>
</head><body><img id="v" src="/stream" draggable="false"><div id="s">connecting...</div><script>
const keys={ArrowUp:'up',ArrowDown:'down',ArrowLeft:'left',ArrowRight:'right',Home:'stop',s:'stop',S:'stop',
 r:'reset',R:'reset',c:'cam',C:'cam'};
addEventListener('keydown',e=>{const k=keys[e.key];if(k){e.preventDefault();fetch('/key?k='+k)}});
let drag=null;const v=document.getElementById('v');
v.onmousedown=e=>{drag=[e.clientX,e.clientY]};addEventListener('mouseup',()=>drag=null);
addEventListener('mousemove',e=>{if(!drag)return;const dx=e.clientX-drag[0],dy=e.clientY-drag[1];
 drag=[e.clientX,e.clientY];fetch(`/cam?az=${-dx*0.3}&el=${-dy*0.3}`)});
v.onwheel=e=>{e.preventDefault();fetch('/cam?zoom='+(e.deltaY>0?1.1:0.9))};
setInterval(async()=>{try{const r=await fetch('/status');document.getElementById('s').textContent=
 (await r.json()).text}catch(e){}},200);
</script></body></html>"""


class WebViewer:
    """viewer = WebViewer(cam, on_key, port); 루프에서 viewer.publish(rgb, status_text).

    cam 은 mujoco.MjvCamera. 키/마우스 콜백은 HTTP 스레드에서 오므로 `viewer.lock` 으로 감싸서 호출한다.
    시뮬레이션 루프도 같은 lock 안에서 스텝 → 경쟁 상태 없음.
    """

    def __init__(self, cam, on_key: Callable[[str], None], port: int = 8000, title: str = "rescue_sim"):
        self.cam = cam
        self.lock = threading.Lock()
        self._cond = threading.Condition()
        self._jpeg, self._seq, self._status = b"", 0, ""
        viewer = self
        page = PAGE.replace("__TITLE__", title).encode()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, body: bytes, ctype: str):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                url = urlparse(self.path)
                q = {k: v[0] for k, v in parse_qs(url.query).items()}
                if url.path == "/":
                    self._send(page, "text/html; charset=utf-8")
                elif url.path == "/key":
                    with viewer.lock:
                        on_key(q.get("k", ""))
                    self._send(b"ok", "text/plain")
                elif url.path == "/cam":
                    with viewer.lock:
                        c = viewer.cam
                        c.azimuth += float(q.get("az", 0))
                        c.elevation = min(-2.0, max(-89.0, c.elevation + float(q.get("el", 0))))
                        c.distance = min(80.0, max(2.0, c.distance * float(q.get("zoom", 1))))
                    self._send(b"ok", "text/plain")
                elif url.path == "/status":
                    self._send(json.dumps({"text": viewer._status}).encode(), "application/json")
                elif url.path == "/stream":
                    self._stream()
                else:
                    self.send_error(404)

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                seq = -1
                try:
                    while True:
                        with viewer._cond:
                            viewer._cond.wait_for(lambda: viewer._seq != seq, timeout=2.0)
                            seq, jpg = viewer._seq, viewer._jpeg
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        print(f"브라우저에서 열기 →  http://localhost:{port}   (Ctrl+C 로 종료)", flush=True)

    def publish(self, rgb, status: str, quality: int = 80):
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
        with self._cond:
            self._jpeg, self._status = buf.getvalue(), status
            self._seq += 1
            self._cond.notify_all()

    def close(self):
        self.server.shutdown()
