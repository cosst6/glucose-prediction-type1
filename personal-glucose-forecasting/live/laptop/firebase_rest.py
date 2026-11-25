"""Tiny Firebase client (Auth + Firestore REST). No Google SDK needed, works on Windows/Linux/Mac.

Login is email + password once; afterwards only the refresh token is kept in state.json.
All data lives under users/<uid>/ so Firestore rules can lock it to you.
"""
import json, os, time, getpass
import requests

AUTH = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={key}"
REFRESH = "https://securetoken.googleapis.com/v1/token?key={key}"
FS = "https://firestore.googleapis.com/v1/projects/{pid}/databases/(default)/documents"


def enc(v):
    """Python value -> Firestore typed value."""
    if v is None: return {"nullValue": None}
    if isinstance(v, bool): return {"booleanValue": v}
    if isinstance(v, int): return {"integerValue": str(v)}
    if isinstance(v, float): return {"doubleValue": v}
    if isinstance(v, str): return {"stringValue": v}
    if isinstance(v, (list, tuple)): return {"arrayValue": {"values": [enc(x) for x in v]}}
    if isinstance(v, dict): return {"mapValue": {"fields": {k: enc(x) for k, x in v.items()}}}
    if hasattr(v, "item"): return enc(v.item())          # numpy scalars
    raise TypeError(f"cannot store {type(v)}")


def dec(v):
    """Firestore typed value -> Python value."""
    (k, x), = v.items()
    if k == "integerValue": return int(x)
    if k == "doubleValue": return float(x)
    if k == "arrayValue": return [dec(i) for i in x.get("values", [])]
    if k == "mapValue": return {a: dec(b) for a, b in x.get("fields", {}).items()}
    if k == "nullValue": return None
    return x


class Firebase:
    def __init__(self, folder):
        self.folder = folder
        self.cfg = json.load(open(os.path.join(folder, "config.json")))
        self.state_path = os.path.join(folder, "state.json")
        self.state = json.load(open(self.state_path)) if os.path.exists(self.state_path) else {}
        self.key, self.pid = self.cfg["firebase_web_api_key"], self.cfg["firebase_project_id"]
        self._tok, self._exp = None, 0

    def save_state(self):
        json.dump(self.state, open(self.state_path, "w"), indent=1)

    # ---------------------------------------------------------------- auth
    def login(self, password=None):
        password = password or getpass.getpass(f"Firebase password for {self.cfg['email']}: ")
        r = requests.post(AUTH.format(key=self.key), json={"email": self.cfg["email"], "password": password, "returnSecureToken": True}, timeout=30)
        r.raise_for_status(); j = r.json()
        self.state.update(uid=j["localId"], refresh_token=j["refreshToken"]); self.save_state()
        self._tok, self._exp = j["idToken"], time.time() + int(j["expiresIn"]) - 120
        return j["localId"]

    def token(self):
        if self._tok and time.time() < self._exp: return self._tok
        if "refresh_token" not in self.state: raise RuntimeError("Not logged in: run  python libre_uploader.py login")
        r = requests.post(REFRESH.format(key=self.key), data={"grant_type": "refresh_token", "refresh_token": self.state["refresh_token"]}, timeout=30)
        r.raise_for_status(); j = r.json()
        self._tok, self._exp = j["id_token"], time.time() + int(j["expires_in"]) - 120
        if j.get("refresh_token") and j["refresh_token"] != self.state["refresh_token"]:
            self.state["refresh_token"] = j["refresh_token"]; self.save_state()
        return self._tok

    @property
    def uid(self): return self.state["uid"]

    def _h(self): return {"Authorization": f"Bearer {self.token()}"}

    def _base(self): return FS.format(pid=self.pid)

    def _name(self, path): return f"projects/{self.pid}/databases/(default)/documents/users/{self.uid}/{path}"

    # ---------------------------------------------------------------- writes
    def put_many(self, docs):
        """docs: list of (path under users/<uid>/, dict). Written in batches of 400 (Firestore max 500)."""
        n = 0
        for i in range(0, len(docs), 400):
            writes = [{"update": {"name": self._name(p), "fields": {k: enc(v) for k, v in d.items()}}} for p, d in docs[i:i + 400]]
            r = requests.post(self._base() + ":commit", headers=self._h(), json={"writes": writes}, timeout=60)
            if r.status_code >= 400: raise RuntimeError(f"Firestore write failed {r.status_code}: {r.text[:300]}")
            n += len(writes)
        return n

    def put(self, path, d): return self.put_many([(path, d)])

    # ---------------------------------------------------------------- reads
    def get(self, path):
        r = requests.get(f"{self._base()}/users/{self.uid}/{path}", headers=self._h(), timeout=30)
        if r.status_code == 404: return None
        r.raise_for_status()
        return {k: dec(v) for k, v in r.json().get("fields", {}).items()}

    def query(self, collection, field="t", since=None, until=None, limit=None):
        """All docs of users/<uid>/<collection> with since <= field < until, ordered by field."""
        filters = []
        if since is not None: filters.append({"fieldFilter": {"field": {"fieldPath": field}, "op": "GREATER_THAN_OR_EQUAL", "value": enc(since)}})
        if until is not None: filters.append({"fieldFilter": {"field": {"fieldPath": field}, "op": "LESS_THAN", "value": enc(until)}})
        q = {"from": [{"collectionId": collection}], "orderBy": [{"field": {"fieldPath": field}}]}
        if len(filters) == 1: q["where"] = filters[0]
        elif filters: q["where"] = {"compositeFilter": {"op": "AND", "filters": filters}}
        if limit: q["limit"] = limit
        r = requests.post(f"{self._base()}/users/{self.uid}:runQuery", headers=self._h(), json={"structuredQuery": q}, timeout=60)
        r.raise_for_status()
        out = []
        for row in r.json():
            d = row.get("document")
            if d: out.append({"_id": d["name"].rsplit("/", 1)[-1], **{k: dec(v) for k, v in d.get("fields", {}).items()}})
        return out


class TokenFirebase(Firebase):
    """Same client, but acting with a user's ID token sent by the phone (used by the cloud server).
    The server never sees a password; Firestore rules still limit it to that user's own data."""
    def __init__(self, key, pid, id_token, uid, cfg=None):
        self.folder, self.cfg, self.state = None, dict(cfg or {}), {"uid": uid}
        self.key, self.pid = key, pid
        self._tok, self._exp = id_token, time.time() + 50 * 60

    def save_state(self): pass

    def token(self): return self._tok


def verify_id_token(key, id_token):
    """Asks Firebase who owns this ID token. Returns (uid, email) or raises."""
    r = requests.post(f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={key}", json={"idToken": id_token}, timeout=20)
    if r.status_code != 200: raise PermissionError("invalid or expired sign-in token")
    u = r.json()["users"][0]
    return u["localId"], u.get("email", "")
