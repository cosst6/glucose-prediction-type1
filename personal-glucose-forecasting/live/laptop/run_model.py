"""Runs ONE model folder in its own Python process, so models with different code versions never clash.
   stdin: {"glucose": [...], "events": [...], "watch": [...], "now_ms": ..., "tz": "Asia/Kolkata"}
   stdout: the forecast dict (JSON)."""
import json, os, sys, traceback, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
folder = os.path.abspath(sys.argv[1])
spec = json.load(open(os.path.join(folder, "model.json"), encoding="utf-8"))
for k, v in spec.get("env", {}).items(): os.environ[k] = str(v)
sys.path[:0] = [folder, HERE]
os.chdir(folder)
from zoneinfo import ZoneInfo
try:
    import adapter
    d = json.load(sys.stdin)
    out = adapter.forecast(d["glucose"], d["events"], d["watch"], d["now_ms"], ZoneInfo(d["tz"]), spec)
except Exception as e:
    out = {"status": "error", "message": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
out.update({"model": spec["id"], "model_name": spec.get("name", spec["id"])})
sys.stdout.write(json.dumps(out, default=float))
