"""Runs every model in the models/ folder on the latest cloud data and writes the forecasts back.

  python forecaster.py once     forecast now with every enabled model
  python forecaster.py loop     forecast every 15 min, AND within ~1 min after you log something in the app
  python forecaster.py list     show the installed models

Adding a model = copy its folder into models/ (code + .joblib + model.json + adapter.py). Nothing else changes.
Pick which one the phone shows first with "active_model" in config.json (the others are shown below it).
Each model runs in its own Python process, so models built with different code versions never clash.

Cloud (users/<uid>/...):  in: glucose, events, watch    out: forecasts/latest, forecasts/<model id>, forecast_history/...
Not a dosing tool.
"""
import json, os, subprocess, sys, time
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
H = 3600 * 1000


def installed():
    out = []
    for d in sorted(os.listdir(MODELS)):
        p = os.path.join(MODELS, d, "model.json")
        if os.path.exists(p):
            s = json.load(open(p, encoding="utf-8")); s["folder"] = os.path.join(MODELS, d); out.append(s)
    return out


def enabled(cfg):
    ms = installed(); want = cfg.get("models")        # list of ids, or missing = all installed
    ms = [m for m in ms if not want or m["id"] in want]
    act = cfg.get("active_model") or (ms[0]["id"] if ms else None)
    ms.sort(key=lambda m: m["id"] != act)               # active one first
    return ms


def run_model(spec, data, timeout=240):
    py = sys.executable.replace("pythonw.exe", "python.exe")
    r = subprocess.run([py, os.path.join(HERE, "run_model.py"), spec["folder"]], input=json.dumps(data),
                       capture_output=True, text=True, timeout=timeout,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        return json.loads(r.stdout)
    except Exception:
        return {"status": "error", "model": spec["id"], "model_name": spec.get("name"), "message": (r.stderr or r.stdout)[-800:]}


class Cache:
    """Keeps cloud documents in memory so each run only downloads what is new (stays far inside the free quota)."""
    def __init__(self, fb):
        self.fb = fb; self.g = {}; self.w = {}; self.last_g = self.last_w = 0

    def refresh(self, now):
        # Libre history can arrive up to 8 h late (reader holds 8 h), so re-ask from 9 h before the newest seen.
        since = max(now - 30 * H, self.last_g - 9 * H) if self.last_g else now - 30 * H
        for d in self.fb.query("glucose", since=since): self.g[d["_id"]] = d
        since = max(now - 26 * H, self.last_w - 1 * H) if self.last_w else now - 26 * H
        for d in self.fb.query("watch", since=since): self.w[d["_id"]] = d
        self.g = {k: v for k, v in self.g.items() if v["t"] >= now - 30 * H}
        self.w = {k: v for k, v in self.w.items() if v["t"] >= now - 26 * H}
        self.last_g = max([v["t"] for v in self.g.values()], default=0)
        self.last_w = max([v["t"] for v in self.w.values()], default=0)
        events = self.fb.query("events", since=now - 30 * H, until=now + 7 * H)   # small: always re-read (edits/deletes)
        return sorted(self.g.values(), key=lambda d: d["t"]), events, sorted(self.w.values(), key=lambda d: d["t"])


def run_once(fb=None, cache=None):
    from firebase_rest import Firebase
    fb = fb or Firebase(HERE); cache = cache or Cache(fb)
    tz_name = fb.cfg.get("timezone", "Asia/Kolkata"); tz = ZoneInfo(tz_name)
    now = int(time.time() * 1000)
    glucose, events, watch = cache.refresh(now)
    data = {"glucose": glucose, "events": events, "watch": watch, "now_ms": now, "tz": tz_name}
    made_local = time.strftime("%Y-%m-%d %H:%M")
    results = []
    for spec in enabled(fb.cfg):
        f = run_model(spec, data); f.update({"made_at": now, "made_local": made_local}); f.pop("trace", None) if f.get("status") != "error" else None
        results.append(f)
        print(f"[{made_local}] {f.get('model_name')}: {f.get('status')} now {f.get('g_now')} ({f.get('g_source')}) ->",
              ", ".join(f"+{p['h']}m {p['mid']} ({p['lo']}-{p['hi']})" for p in f.get("preds", [])), "|", f.get("message"), flush=True)
    if not results: print("No models in models/ folder."); return None
    main = dict(results[0])
    main["others"] = [{k: r.get(k) for k in ("model", "model_name", "status", "g_now", "g_source", "based_on_local", "preds", "spike_risk", "message")}
                      for r in results[1:]]
    docs = [("forecasts/latest", main)] + [(f"forecasts/{r['model']}", r) for r in results]
    for r in results:
        if r.get("based_on_local"):
            docs.append((f"forecast_history/{r['model'][:40]}_{r['based_on_local'].replace('-', '').replace(' ', '').replace(':', '')}", {**r, "t": now}))
    fb.put_many(docs)
    return main


def loop(every_min=15):
    from firebase_rest import Firebase
    fb = Firebase(HERE); cache = Cache(fb); last_run = 0; last_trigger = 0
    while True:
        now = int(time.time() * 1000)
        try:
            trig = (fb.get("meta/trigger") or {}).get("t", 0)   # the app and the reader uploader poke this after new data
        except Exception as e:
            print("trigger check failed:", e, flush=True); trig = 0
        if now - last_run >= every_min * 60 * 1000 or trig > last_trigger:
            last_trigger = max(last_trigger, trig)
            try:
                run_once(fb, cache); last_run = now
            except Exception as e:
                print("forecast failed:", e, flush=True)
        time.sleep(60)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "once"
    if cmd == "list":
        from firebase_rest import Firebase
        try: cfg = Firebase(HERE).cfg
        except Exception: cfg = {}
        act = cfg.get("active_model")
        for m in installed(): print(("* " if m["id"] == act else "  ") + m["id"], "-", m.get("name"), "| horizons", m.get("horizons_min"))
    elif cmd == "loop": loop()
    else: run_once()
