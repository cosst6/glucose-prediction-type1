"""Cloud forecast server: the same model folders, but called by the phone instead of running on the laptop.

  POST /forecast   header  Authorization: Bearer <Firebase ID token of the signed-in app user>
                   -> reads that user's glucose/events/watch from Firestore, runs every model folder,
                      writes forecasts/latest (+ per model + history) and returns the main forecast as JSON.
  GET  /           health check ("ok" + installed models)

Settings come from environment variables (set as secrets on the host):
  FIREBASE_API_KEY, FIREBASE_PROJECT_ID   your Firebase project
  ALLOWED_EMAILS   comma-separated emails allowed to use the server (your own; blocks strangers)
  ACTIVE_MODEL     model id shown first (default: GLU-B v10 glucometer)
  MODELS           optional comma-separated ids to run (default: every folder in models/)
  TIMEZONE         default Asia/Kolkata
  PORT             default 7860 (Hugging Face); Cloud Run sets 8080 itself
Not a dosing tool.
"""
import json, os, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import forecaster
from firebase_rest import TokenFirebase, verify_id_token

KEY, PID = os.environ.get("FIREBASE_API_KEY", ""), os.environ.get("FIREBASE_PROJECT_ID", "")
ALLOWED = {e.strip().lower() for e in os.environ.get("ALLOWED_EMAILS", "").split(",") if e.strip()}
CFG = {"timezone": os.environ.get("TIMEZONE", "Asia/Kolkata"),
       "active_model": os.environ.get("ACTIVE_MODEL", "GLU-B_glucometer-watch15-steps-cal-PLANNED2H-FIBER-EXERCISE_v10_2026-11-01"),
       "models": [m.strip() for m in os.environ.get("MODELS", "").split(",") if m.strip()]}
LOCK = threading.Lock()          # one forecast at a time (free machines are small)
CACHES = {}                      # uid -> forecaster.Cache, so repeat calls only download new documents
LAST = {}                        # uid -> (time, result): calls within 30 s get the same answer


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj, default=float).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._send(200, {"ok": True, "models": [m["id"] for m in forecaster.enabled(CFG)], "active": CFG["active_model"]})

    def do_POST(self):
        if self.path.rstrip("/") != "/forecast": return self._send(404, {"error": "not found"})
        tok = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        try:
            uid, email = verify_id_token(KEY, tok)
        except Exception as e:
            return self._send(401, {"error": str(e)})
        if ALLOWED and email.lower() not in ALLOWED: return self._send(403, {"error": "this account is not allowed on this server"})
        with LOCK:
            t, res = LAST.get(uid, (0, None))
            if res is not None and time.time() - t < 30: return self._send(200, res)
            try:
                fb = TokenFirebase(KEY, PID, tok, uid, CFG)
                cache = CACHES.get(uid) or forecaster.Cache(fb); cache.fb = fb; CACHES[uid] = cache
                res = forecaster.run_once(fb, cache) or {"status": "error", "message": "no models installed"}
                LAST[uid] = (time.time(), res)
                return self._send(200, res)
            except Exception as e:
                traceback.print_exc()
                return self._send(500, {"status": "error", "message": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *a): print(time.strftime("%H:%M:%S"), self.address_string(), fmt % a, flush=True)


if __name__ == "__main__":
    if not KEY or not PID: raise SystemExit("Set FIREBASE_API_KEY and FIREBASE_PROJECT_ID")
    port = int(os.environ.get("PORT", "7860"))
    print("Glucose forecast server on port", port, "models:", [m["id"] for m in forecaster.enabled(CFG)], flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
