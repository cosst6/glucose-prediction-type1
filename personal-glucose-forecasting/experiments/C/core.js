// Model C core: food text -> macros, absorption curves, features, ridge prediction. Mirrors glucoforecast.py.
const GF = (() => {
  const NUTR = ["carbs", "protein", "fat", "fiber"];
  function num(s) { const m = s.match(/(\d+(?:\.\d+)?)/); return m ? parseFloat(m[1]) : null; }
  function parseItem(text, DB) {
    const t = text.toLowerCase().trim();
    if (!t || t === "coffee") return null;
    const ex = t.match(/(\d+(?:\.\d+)?)\s*gm?\s*carb/);
    if (ex) return { item: t, food: "explicit carbs", grams: null, carbs: +ex[1], protein: 0, fat: 0, fiber: 0 };
    const row = DB.find(r => new RegExp(r.pattern).test(t));
    if (!row) return { item: t, food: "unknown", grams: null, carbs: 0, protein: 0, fat: 0, fiber: 0 };
    const n = num(t); let g;
    if (row.unit === "ml_milk") g = (n || row.default_qty) * row.grams_per_unit;
    else if (/\bml\b/.test(t)) g = n;
    else if (/\d\s*(gm|g|gms|grams)\b/.test(t)) g = n;
    else if (t.includes("cup")) g = (n || 1) * (row.key === "curd" ? 240 : (row.unit === "cup" ? row.grams_per_unit : 150));
    else if (/tbsp|table\s*spoon/.test(t)) g = (n || 1) * (row.unit === "tbsp" ? row.grams_per_unit : 15);
    else if (row.unit === "piece") g = (n || row.default_qty) * row.grams_per_unit;
    else g = ((n && row.unit !== "g") ? n : row.default_qty) * ((row.unit !== "g" && row.unit !== "ml") ? row.grams_per_unit : 1);
    const o = { item: t, food: row.key.replace(/_/g, " "), grams: Math.round(g * 10) / 10 };
    for (const k of NUTR) o[k] = Math.round(g * row[k] / 100 * 10) / 10;
    return o;
  }
  function parseMeal(text, DB) {
    const items = (text || "").split(/[;\n]/).map(x => parseItem(x, DB)).filter(Boolean);
    const tot = {}; for (const k of NUTR) tot[k] = Math.round(items.reduce((a, i) => a + i[k], 0) * 10) / 10;
    return { items, tot };
  }
  function insulinCum(m, peak = 75, dur = 300) {
    if (m < 0) return 0;
    const tau = peak * (1 - peak / dur) / (1 - 2 * peak / dur), a = 2 * tau / dur, S = 1 / (1 - a + (1 + a) * Math.exp(-dur / tau));
    const t = Math.min(Math.max(m, 0), dur);
    const iob = 1 - S * (1 - a) * ((t * t / (tau * dur * (1 - a)) - t / tau - 1) * Math.exp(-t / tau) + 1);
    return 1 - Math.min(Math.max(iob, 0), 1);
  }
  function triCum(m, dur, pf = 1 / 3) {
    if (m < 0) return 0;
    const p = dur * pf, t = Math.min(Math.max(m, 0), dur);
    return t < p ? t * t / (p * dur) : 1 - (dur - t) ** 2 / ((dur - p) * dur);
  }
  const KERNELS = { ins: ["rapid", m => insulinCum(m)], carb: ["net_carbs", m => triCum(m, 180)], fpu: ["fpu", m => triCum(m, 300, 0.5)] };
  // events: [{t: Date, rapid, lantus, carbs, protein, fat, fiber}]
  function features(now, g0, events, horizons) {
    const f = { g0 };
    const ev = events.filter(e => e.t <= now).map(e => ({ ...e, m: (now - e.t) / 60000,
      net_carbs: Math.max((e.carbs || 0) - 0.5 * (e.fiber || 0), 0), fpu: (9 * (e.fat || 0) + 4 * (e.protein || 0)) / 100 }));
    for (const [name, [col, cum]] of Object.entries(KERNELS)) {
      let ob = 0, a30 = 0; const nxt = {}; horizons.forEach(h => nxt[h] = 0);
      for (const e of ev) {
        const amt = e[col] || 0; if (!amt) continue;
        const c = cum(e.m); ob += amt * (1 - c); a30 += amt * (c - cum(e.m - 30));
        for (const h of horizons) nxt[h] += amt * (cum(e.m + h) - c);
      }
      f[name + "_ob"] = ob; f[name + "_act30"] = a30; for (const h of horizons) f[`${name}_next${h}`] = nxt[h];
    }
    const lant = ev.filter(e => e.lantus > 0).sort((a, b) => a.t - b.t);
    f.h_since_lantus = lant.length ? (now - lant[lant.length - 1].t) / 3600000 : 24;
    f.lantus_ob = ev.reduce((a, e) => a + (e.lantus || 0) * Math.min(Math.max(1 - e.m / 1440, 0), 1), 0);
    const hr = now.getHours() + now.getMinutes() / 60;
    f.tod_s = Math.sin(2 * Math.PI * hr / 24); f.tod_c = Math.cos(2 * Math.PI * hr / 24);
    return f;
  }
  function forecast(model, now, g0, events) {
    const hs = Object.keys(model.horizons).map(Number).sort((a, b) => a - b);
    const f = features(now, g0, events, hs);
    const out = hs.map(h => {
      const m = model.horizons[h];
      let y = m.intercept; m.features.forEach((c, i) => { y += (f[c] - m.mean[i]) / m.scale[i] * m.coef[i]; });
      return { h, g: Math.min(Math.max(g0 + y, 40), 400) };
    });
    return { f, out };
  }
  return { parseMeal, features, forecast, insulinCum, triCum };
})();
if (typeof module !== "undefined") module.exports = GF;
