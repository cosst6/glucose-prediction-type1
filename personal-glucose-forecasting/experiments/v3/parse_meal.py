"""Turn free-text meal notes ("3 boiled egg; 250 ml milk with coffee; 1 cup moong") into macros.

Each item is matched to the first row of food_db.csv whose pattern appears in it. Quantities:
  '<n> ml'  '<n> gm|g'  '<n> cup'  '<n> tbsp|table spoon'  bare number = pieces / default unit.
Values in food_db are per 100 g (per 100 ml for milk) except where noted.
"""
import re, os, pandas as pd

DB = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "food_db.csv"))
NUTR = ["carbs", "protein", "fat", "fiber"]
CUP_G = {"curd": 240}
TBSP_G = 15

def _num(s):
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None

def parse_item(text):
    t = text.lower().strip()
    if not t or "coffee" == t:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*gm?\s*carb", t)
    if m:  # "30 gm carb apple" -> explicit carbs
        return {"item": t, "food": "explicit_carbs", "grams": None, "carbs": float(m.group(1)), "protein": 0, "fat": 0, "fiber": 0}
    row = next((r for _, r in DB.iterrows() if re.search(r.pattern, t)), None)
    if row is None:
        return {"item": t, "food": "UNKNOWN", "grams": None, **{k: 0.0 for k in NUTR}}
    n = _num(t)
    if row.unit == "ml_milk":
        grams = (n or row.default_qty) * row.grams_per_unit
    elif re.search(r"\bml\b", t):
        grams = n
    elif re.search(r"\d\s*(gm|g|gms|grams)\b", t):
        grams = n
    elif "cup" in t:
        grams = (n or 1) * (CUP_G.get(row.key, row.grams_per_unit if row.unit == "cup" else 150))
    elif re.search(r"tbsp|table\s*spoon", t):
        grams = (n or 1) * (row.grams_per_unit if row.unit == "tbsp" else TBSP_G)
    elif row.unit in ("piece",):
        grams = (n or row.default_qty) * row.grams_per_unit
    else:
        grams = (n if (n and row.unit != "g") else row.default_qty) * (row.grams_per_unit if row.unit not in ("g", "ml") else 1)
    out = {"item": t, "food": row.key, "grams": round(grams, 1)}
    for k in NUTR:
        out[k] = round(grams * row[k] / 100, 1)
    return out

def parse_meal(text):
    if not isinstance(text, str) or not text.strip():
        return [], {k: 0.0 for k in NUTR}
    items = [p for p in (parse_item(x) for x in re.split(r"[;\n]", text)) if p]
    tot = {k: round(sum(i[k] for i in items), 1) for k in NUTR}
    return items, tot

if __name__ == "__main__":
    import sys
    items, tot = parse_meal(" ".join(sys.argv[1:]))
    for i in items: print(i)
    print("TOTAL", tot)
