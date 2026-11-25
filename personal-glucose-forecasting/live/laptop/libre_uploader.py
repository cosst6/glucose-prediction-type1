"""FreeStyle Libre reader -> Firebase uploader for Windows (also works on Linux/Mac).

Commands
  python libre_uploader.py login            one-time: sign in to your Firebase project
  python libre_uploader.py sync             read the plugged-in reader once and upload
  python libre_uploader.py watch            run forever: uploads every time you plug the reader in
  python libre_uploader.py import-csv FILE  upload a LibreView CSV export (old data / backup path)

Reads the reader directly over USB with the open-source glucometerutils FreeStyle Libre driver,
so no LibreView, password or CSV download is needed. Everything already uploaded is skipped
(same reading -> same document id), so plugging in many times a day is safe.
"""
import datetime as dt, json, logging, os, sys, time
from zoneinfo import ZoneInfo

from firebase_rest import Firebase

HERE = os.path.dirname(os.path.abspath(__file__))
ABBOTT_VID, LIBRE_PID = 0x1A61, 0x3650
log = logging.getLogger("libre")


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(os.path.join(HERE, "uploader.log"), encoding="utf-8"), logging.StreamHandler()])


def doc(local_dt, mgdl, kind, src, tz, extra=None):
    """One glucose reading. kind: hist (15-min sensor history), scan (sensor scan), stick (fingerstick strip)."""
    t = int(local_dt.replace(tzinfo=tz).timestamp() * 1000)
    d = {"t": t, "local": local_dt.strftime("%Y-%m-%d %H:%M"), "mgdl": float(mgdl), "kind": kind, "src": src}
    if extra: d.update(extra)
    return f"glucose/{kind}_{local_dt:%Y%m%d%H%M}", d


def event_doc(local_dt, etype, tz, text="", units=None, src="reader"):
    t = int(local_dt.replace(tzinfo=tz).timestamp() * 1000)
    d = {"t": t, "local": local_dt.strftime("%Y-%m-%d %H:%M"), "type": etype, "text": text, "units": units,
         "planned": False, "deleted": False, "src": src, "created": int(time.time() * 1000)}
    return f"events/{src}_{etype}_{local_dt:%Y%m%d%H%M}", d


def notes_to_events(local_dt, comment, tz):
    """Insulin / food notes typed on the reader itself become events too."""
    import re
    out = []
    m = re.search(r"Rapid-acting insulin \(([\d.]+)\)", comment or "")
    if m: out.append(event_doc(local_dt, "rapid", tz, units=float(m.group(1))))
    m = re.search(r"Long-acting insulin \(([\d.]+)\)", comment or "")
    if m: out.append(event_doc(local_dt, "lantus", tz, units=float(m.group(1))))
    m = re.search(r"Food \((\d+) g\)", comment or "")
    if m: out.append(event_doc(local_dt, "food", tz, text=f"{m.group(1)} gm carb"))
    return out


def reader_present():
    try:
        import hid
        return len(hid.enumerate(ABBOTT_VID, LIBRE_PID)) > 0
    except Exception as e:   # hidapi missing -> fall back to just trying to open the reader
        log.debug("hid enumerate failed: %s", e); return True


def read_reader():
    from glucometerutils.drivers import fslibre
    from glucometerutils import common
    dev = fslibre.Device(None); dev.connect()
    try:
        info = dev.get_meter_info()
        unit = dev.get_glucose_unit()
        rows = list(dev.get_readings())
    finally:
        dev.disconnect()
    return info, unit, rows, common


def sync(fb, tz, beep=True):
    info, unit, rows, common = read_reader()
    log.info("Reader %s: %d records", info.serial_number, len(rows))
    docs = []
    for r in rows:
        if not isinstance(r, common.GlucoseReading): continue
        v = r.get_value_as(common.Unit.MG_DL) if unit != common.Unit.MG_DL else r.value
        c = r.comment or ""
        if r.measure_method == common.MeasurementMethod.CGM:
            kind = "hist" if "(Sensor)" in c else "scan"
        else:
            kind = "stick"
        docs.append(doc(r.timestamp, v, kind, "reader", tz, {"note": c}))
        docs += notes_to_events(r.timestamp, c, tz)
    # skip what was uploaded before (ids are deterministic, this just saves writes)
    last = fb.state.get("last_reader_t", 0)
    new = [(p, d) for p, d in docs if d["t"] > last - 6 * 3600 * 1000]
    n = fb.put_many(new) if new else 0
    if docs:
        fb.state["last_reader_t"] = max(d["t"] for _, d in docs); fb.state["last_sync"] = time.time(); fb.save_state()
    newest = max((d for _, d in docs if d.get("kind") in ("hist", "scan")), key=lambda d: d["t"], default=None)
    log.info("Uploaded %d documents. Newest sensor reading: %s", n, f"{newest['local']} = {newest['mgdl']:.0f}" if newest else "none")
    if beep:
        try:
            import winsound; winsound.MessageBeep()
        except Exception: pass
    return n


def import_csv(fb, path, tz):
    import pandas as pd
    raw = pd.read_csv(path, skiprows=1)
    raw["ts"] = pd.to_datetime(raw["Device Timestamp"], format="%d-%m-%Y %I:%M %p")
    docs = []
    for _, r in raw.iterrows():
        t = r.ts.to_pydatetime(); rt = r["Record Type"]
        if rt == 0 and pd.notna(r["Historic Glucose mg/dL"]): docs.append(doc(t, r["Historic Glucose mg/dL"], "hist", "csv", tz))
        elif rt == 1 and pd.notna(r["Scan Glucose mg/dL"]): docs.append(doc(t, r["Scan Glucose mg/dL"], "scan", "csv", tz))
        elif rt == 2 and pd.notna(r.get("Strip Glucose mg/dL")): docs.append(doc(t, r["Strip Glucose mg/dL"], "stick", "csv", tz))
    log.info("Uploading %d readings from %s", len(docs), path)
    return fb.put_many(docs)


def watch(fb, tz, poll=10):
    """Waits for the reader to be plugged in, uploads once per plug-in."""
    log.info("Watching for the FreeStyle Libre reader (plug it in any time). Ctrl+C to stop.")
    was = False
    while True:
        now = reader_present()
        if now and not was:
            time.sleep(4)                       # let Windows finish installing the HID device
            for attempt in range(3):
                try:
                    sync(fb, tz); break
                except Exception as e:
                    log.warning("Sync attempt %d failed: %s", attempt + 1, e); time.sleep(5)
            try:   # new glucose -> ask the forecaster (running in the background) to refresh within ~1 min
                fb.put("meta/trigger", {"t": int(time.time() * 1000), "by": "reader"})
            except Exception as e:
                log.warning("Forecast trigger failed: %s", e)
        was = now
        time.sleep(poll)


def main():
    setup_logging()
    fb = Firebase(HERE); tz = ZoneInfo(fb.cfg.get("timezone", "Asia/Kolkata"))
    cmd = sys.argv[1] if len(sys.argv) > 1 else "watch"
    if cmd == "login": print("Signed in, uid:", fb.login())
    elif cmd == "sync": sync(fb, tz)
    elif cmd == "watch": watch(fb, tz)
    elif cmd == "import-csv": print("uploaded", import_csv(fb, sys.argv[2], tz))
    else: print(__doc__)


if __name__ == "__main__":
    main()
