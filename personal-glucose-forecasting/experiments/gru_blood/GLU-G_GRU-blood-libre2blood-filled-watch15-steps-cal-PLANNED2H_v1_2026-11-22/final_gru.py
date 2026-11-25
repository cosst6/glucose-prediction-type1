import sys, os, json, numpy as np, pandas as pd, torch; sys.path.insert(0, os.path.dirname(__file__)); from gru import *
G = pd.read_pickle(sys.argv[1]); O = sys.argv[2]; N = os.path.basename(O)
a, b, n = calib(G, G.index == G.index)
gv, real, syn = glucose_series(G, a, b); A = anchors(G, gv, real, syn)
X, St, A = tensors(G, gv, real, syn, A); g0 = gv[A.i.values]
y = torch.tensor(((A.y.values - g0) / 50).astype(np.float32)); w = np.where(A.real == 1, 5.0, 1.0)
ms = fit(X, St, y, w)
torch.save({"state_dicts": [m.state_dict() for m in ms], "n_channels": X.shape[2]}, f"{O}/{N}.pt")
json.dump({"model_id": N, "trained": "2026-11-22", "type": "GRU(48) + MLP head, 3-seed ensemble", "history_min": HIST * STEP, "planned_window_min": FUT * STEP,
  "horizons_min": [60, 120], "target": "blood glucose (glucometer)", "libre_to_blood": {"a": a, "b": b, "pairs": n},
  "glucose_gap_fill": "exponential (log-linear) interpolation between past points, forward-fill to now, back-fill window start",
  "training_samples": int(len(A)), "real_stick_pairs": int(A.real.sum()), "stick_weight": 5, "event_channels": EV, "event_scale": EVS.tolist()}, open(f"{O}/{N}_card.json", "w"), indent=1)
print(a, b, n, len(A), int(A.real.sum()))
