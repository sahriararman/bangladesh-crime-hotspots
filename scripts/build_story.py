"""Build index.html from Bangladesh-Crime-Dataset.csv.

Usage: python -I build_story.py [path/to/Bangladesh-Crime-Dataset.csv]
Reuses the cleaning and hotspot method in build_hotspot_map.py.
"""
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy import stats

import build_hotspot_map as B

OUT = HERE.parent / "index.html"
WEEK = ["saturday", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
LAT, LON = 23.77, 90.40   # central Dhaka, for the city close-up


def smooth_ring(members, cells, s):
    """Outer boundary of contiguous hex cells, rounded with Chaikin corner cutting, as [lon, lat]."""
    corners = [(1, -1), (1, 1), (0, 2), (-1, 1), (-1, -1), (0, -2)]
    edges = {}
    for i in members:
        X, Y = 2 * cells[i][0] + cells[i][1], 3 * cells[i][1]
        pts = [(X + a, Y + b) for a, b in corners]
        for k in range(6):
            a, b = pts[k], pts[(k + 1) % 6]
            if (b, a) in edges:
                del edges[(b, a)]
            else:
                edges[(a, b)] = 1
    nxt = {a: b for a, b in edges}
    w, rings = math.sqrt(3) * s, []
    while nxt:
        start = next(iter(nxt))
        at, ring = start, []
        while True:
            ring.append((at[0] * w / 2, at[1] * s / 2))
            at = nxt.pop(at)
            if at == start:
                break
        rings.append(ring)
    def area(r):
        return abs(sum(p[0] * r[(i + 1) % len(r)][1] - r[(i + 1) % len(r)][0] * p[1] for i, p in enumerate(r)))
    pts = max(rings, key=area)
    for _ in range(3):
        out = []
        for i, p in enumerate(pts):
            n = pts[(i + 1) % len(pts)]
            out += [(.75 * p[0] + .25 * n[0], .75 * p[1] + .25 * n[1]), (.25 * p[0] + .75 * n[0], .25 * p[1] + .75 * n[1])]
        pts = out
    coords = [[round(B.REF_LON + x / B.KM_X, 5), round(B.REF_LAT + y / B.KM_Y, 5)] for x, y in pts]
    return coords + [coords[0]]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    df, n_raw, moved = B.load()
    n = len(df)
    s = B.HEX_FLAT_KM / math.sqrt(3)
    df.loc[df.incident_division == "gazipur", "incident_division"] = "dhaka"   # Gazipur is a district of Dhaka division

    keys = sorted(set(zip(df.latitude, df.longitude)))
    xy = np.array([B.to_km(*k) for k in keys])
    cells = B.build_grid(xy, s)
    cell_id = {(c[0], c[1]): i for i, c in enumerate(cells)}
    site_cell = {k: cell_id[B.hex_round(*B.to_km(*k), s)] for k in keys}
    df["cell"] = [site_cell[k] for k in zip(df.latitude, df.longitude)]
    x = np.bincount(df.cell, minlength=len(cells)).astype(float)

    def run(band, v=x):
        nb = B.neighbours(cells, band)
        major = B.clusters(B.gi_pass(v, nb, np.ones(len(v), bool)), cells, cell_id, v)
        active = np.ones(len(v), bool)
        for m in major:
            active[m] = False
        regional = B.clusters(B.gi_pass(v, nb, active), cells, cell_id, v)
        by = lambda c: -v[c].sum()
        return sorted(major, key=by), sorted(regional, key=by)

    # Sites
    site_rows = []
    grouped = df.groupby(["latitude", "longitude"])
    order = np.random.default_rng(7).permutation(len(keys))
    for o, ((lat, lon), d) in zip(order, grouped):
        top_place = d.incident_place.value_counts().index[0]
        city = (lat, lon) in B.CITY_LEVEL
        site_rows.append({"lat": lat, "lon": lon, "n": len(d), "o": int(o), "city": city,
                          "name": B.nice(top_place), "cell": site_cell[(lat, lon)],
                          "labelable": (not city) or "city" in top_place})
    sites = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
         "properties": {"n": r["n"], "o": r["o"]}} for r in site_rows]}

    def top_sites(cond, k):
        rows = sorted((r for r in site_rows if r["labelable"] and cond(r)), key=lambda r: -r["n"])[:k]
        return [{"name": r["name"], "n": r["n"], "lon": r["lon"], "lat": r["lat"]} for r in rows]

    # Hotspots at the story's 20 km radius
    major, regional = run(B.BAND_KM)
    hotspots = []
    for tier, group in (("major", major), ("regional", regional)):
        for m in group:
            d = df[df.cell.isin(m)]
            ring = smooth_ring(m, cells, s)
            lons, lats = [p[0] for p in ring], [p[1] for p in ring]
            cellset = set(m)
            hotspots.append({"tier": tier, "n": len(d), "cells": len(m), "ring": ring,
                             "name": B.nice(d.incident_district.value_counts().index[0]),
                             "bbox": [[min(lons), min(lats)], [max(lons), max(lats)]],
                             "top": top_sites(lambda r: r["cell"] in cellset, 5)})
    inside = sum(h["n"] for h in hotspots)
    hot_cells = sum(h["cells"] for h in hotspots)

    # How stable each hotspot is across radii
    stability = {}
    for band in (15.0, 20.0, 30.0):
        maj, reg = run(band)
        for tier, group in (("major", maj), ("regional", reg)):
            for m in group:
                name = B.nice(df[df.cell.isin(m)].incident_district.value_counts().index[0])
                row = stability.setdefault(name, {"name": name, "major": False, "v": {}})
                row["v"][str(int(band))] = int(x[m].sum())
                row["major"] = row["major"] or tier == "major"
    stability = sorted(stability.values(), key=lambda r: (-len(r["v"]), -max(r["v"].values())))

    # One map per crime, using the same test on that crime's counts
    crime_maps = []
    for crime in B.CRIMES:
        d = df[df.crime == crime]
        v = np.bincount(d.cell, minlength=len(cells)).astype(float)
        maj, reg = run(B.BAND_KM, v)
        hot = []
        for tier, group in (("major", maj), ("regional", reg)):
            for m in group:
                name = B.nice(d[d.cell.isin(m)].incident_district.value_counts().index[0])
                hot.append({"tier": tier, "name": "Greater Dhaka" if name == "Dhaka" else name, "n": int(v[m].sum()),
                            "ring": smooth_ring(m, cells, s)[::6]})
        at = d.groupby(["latitude", "longitude"]).agg(n=("crime", "size"), place=("incident_place", lambda q: q.value_counts().index[0])).reset_index()
        dhaka_n = next((h["n"] for h in hot if h["name"] == "Greater Dhaka"), 0)
        crime_maps.append({"key": crime, "name": B.nice(crime), "n": len(d), "dhakaPct": round(100 * dhaka_n / len(d)),
                           "hot": hot, "sites": [[r.longitude, r.latitude, int(r.n), B.nice(r.place)] for r in at.itertuples()]})
    crime_maps.sort(key=lambda c: -c["dhakaPct"])
    cm = {c["key"]: c for c in crime_maps}
    uni = df.incident_place.str.contains("university")
    uni_assault = int((uni & (df.crime == "assault")).sum())
    uni_other_pct = 100 * (uni & (df.crime != "assault")).sum() / (df.crime != "assault").sum()

    # Each division's crime mix against the national mix
    ct = pd.crosstab(df.incident_division, df.crime)
    chi2, _, _, exp = stats.chi2_contingency(ct)
    row_share = ct.sum(axis=1).values[:, None] / ct.values.sum()
    col_share = ct.sum(axis=0).values[None, :] / ct.values.sum()
    adj = (ct.values - exp) / np.sqrt(exp * (1 - row_share) * (1 - col_share))
    z_cut = stats.norm.isf(.025 / adj.size)   # Bonferroni: 5% across every cell in the grid
    cramer = math.sqrt(chi2 / ct.values.sum() / (min(ct.shape) - 1))
    finger_cols = [c["key"] for c in crime_maps]
    finger = {"cols": [cm[c]["name"] for c in finger_cols], "rows": [
        {"name": B.nice(div), "v": [{"r": round(float(ct.loc[div, c] / exp[i, list(ct.columns).index(c)]), 2),
                                     "n": int(ct.loc[div, c]), "sig": bool(abs(adj[i, list(ct.columns).index(c)]) >= z_cut)} for c in finger_cols]}
        for i, div in enumerate(ct.index)]}
    ratio = lambda div, c: f'{next(r for r in finger["rows"] if r["name"] == div)["v"][finger_cols.index(c)]["r"]:.1f}'

    # Population units and rates
    df["unit"] = df.incident_district + "|" + df.total_population.astype(str)
    units = []
    for _, d in df.groupby("unit"):
        counts = d.incident_place.value_counts()
        pop, density = int(d.total_population.iloc[0]), int(d.density_per_kmsq.iloc[0])
        metro = len(counts) >= 5 and density >= 10000
        name = f"{B.nice(d.incident_district.iloc[0])} metro" if metro else B.nice(counts.index[0])
        site = d.groupby(["latitude", "longitude"]).size().idxmax()
        units.append({"name": name, "district": B.nice(d.incident_district.iloc[0]), "n": len(d), "pop": pop,
                      "density": density, "rate": 1e5 * len(d) / pop, "lat": site[0], "lon": site[1]})
    ranked = sorted((u for u in units if u["n"] >= 15), key=lambda u: -u["rate"])
    rate_dots = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [u["lon"], u["lat"]]},
         "properties": {"rate": round(u["rate"], 1)}} for u in ranked]}
    pick = ranked[:7]
    for u in ranked:
        if u["name"] in ("Dhaka metro", "Chattogram metro"):
            pick.append(u)
    rate_list = [{"name": u["name"], "district": u["district"], "rate": round(u["rate"], 1), "n": u["n"],
                  "pop": u["pop"], "ref": u["name"].endswith("metro")} for u in pick]
    rate_labels = [{"name": u["name"], "n": f'{u["rate"]:.1f}', "lon": u["lon"], "lat": u["lat"]} for u in ranked[:5]]
    rho = stats.spearmanr([u["rate"] for u in units], [u["density"] for u in units]).statistic
    scatter = [{"d": u["density"], "r": round(u["rate"], 2), "l": u["name"], "n": u["n"]} for u in units]

    # Charts
    counts = df.crime.value_counts()
    mix = [{"name": B.nice(c), "n": int(v), "pct": round(100 * v / n, 1)} for c, v in counts.items()]
    dh = df.assign(d=df.incident_district == "dhaka").groupby("crime").d.mean().sort_values(ascending=False)
    dhaka_all = (df.incident_district == "dhaka").mean()
    dhaka_share = [{"name": B.nice(c), "pct": round(100 * v, 1)} for c, v in dh.items()]
    ct = pd.crosstab(df.crime, df.part_of_the_day, normalize="index") * 100
    ct = ct[B.PARTS].sort_values("night", ascending=False)
    clock = {"parts": [B.nice(p) for p in B.PARTS],
             "rows": [{"name": B.nice(c), "v": [int(round(v)) for v in row]} for c, row in zip(ct.index, ct.values)]}
    m = df.incident_month.value_counts().sort_index()
    months = [{"name": MONTHS[i - 1], "n": int(m[i])} for i in m.index]
    wd = df.incident_weekday.value_counts()
    week = [{"name": B.nice(w), "n": int(wd[w])} for w in WEEK]
    other = np.mean([m[i] for i in range(1, 11)])
    late = np.mean([m[11], m[12]])

    top = lambda name: next(h for h in hotspots if h["name"] == name)
    night = (df.part_of_the_day == "night").mean()
    V = {
        "rob_pct": str(cm["robbery"]["dhakaPct"]), "mur_pct": str(cm["murder"]["dhakaPct"]),
        "uni_assault": str(uni_assault), "uni_assault_pct": f'{100 * uni_assault / cm["assault"]["n"]:.0f}', "uni_other_pct": f"{uni_other_pct:.1f}",
        "mym_mur": ratio("Mymensingh", "murder"), "dh_rob_r": ratio("Dhaka", "robbery"), "bar_ass": ratio("Barisal", "assault"),
        "syl_body": ratio("Sylhet", "bodyfound"), "ran_mur": ratio("Rangpur", "murder"), "cramer": f"{cramer:.2f}",
        "n": f"{n:,}", "raw": f"{n_raw:,}", "dupes": f"{n_raw - n:,}", "moved": f"{moved:,}",
        "sites": f"{len(keys):,}", "cells": f"{len(cells):,}", "hot": str(len(hotspots)),
        "inside_pct": f"{100 * inside / n:.0f}", "area_pct": f"{100 * hot_cells / len(cells):.0f}",
        "dhaka_n": f'{top("Dhaka")["n"]:,}', "dhaka_pct": f'{100 * top("Dhaka")["n"] / n:.0f}',
        "ctg_n": f'{top("Chattogram")["n"]:,}', "jashore_n": str(top("Jashore")["n"]), "mym_n": str(top("Mymensingh")["n"]),
        "savar_n": str(next(r["n"] for r in site_rows if r["name"] == "Savar")),
        "ctg_city_n": str(next(r["n"] for r in site_rows if r["name"] == "Chattogram City")),
        "top_mix": mix[0]["name"].lower(), "top_mix_pct": f'{mix[0]["pct"]:.0f}', "mix2": mix[1]["name"].lower(), "mix2_pct": f'{mix[1]["pct"]:.0f}',
        "dhaka_all": f"{100 * dhaka_all:.0f}", "dh_rob": f'{dhaka_share[0]["pct"]:.0f}', "dh_mur": f'{dhaka_share[-1]["pct"]:.0f}',
        "top_rate_name": ranked[0]["name"], "top_rate": f'{ranked[0]["rate"]:.1f}',
        "dhaka_rate": f'{next(u["rate"] for u in ranked if u["name"] == "Dhaka metro"):.1f}',
        "ctg_rate": f'{next(u["rate"] for u in ranked if u["name"] == "Chattogram metro"):.1f}',
        "rho": f"{rho:.2f}".replace("-", "−"), "units": str(len(units)),
        "night_pct": f"{100 * night:.0f}",
        "body_morning": str(next(r for r in clock["rows"] if r["name"] == "Body found")["v"][0]),
        "late_drop": f"{100 * (1 - late / other):.0f}", "other_avg": f"{other:.0f}", "late_avg": f"{late:.0f}",
        "city_n": str(sum(r["n"] for r in site_rows if r["city"])),
        "wk_hi": f'{wd.idxmax().capitalize()} ({wd.max()})', "wk_lo": f'{wd.idxmin().capitalize()} ({wd.min()})',
    }
    data = {"sites": sites, "rates": rate_dots, "hotspots": hotspots, "rateLabels": rate_labels,
            "cityLabels": top_sites(lambda r: math.hypot((r["lat"] - LAT) * 111, (r["lon"] - LON) * 102) < 16, 6),
            "mix": mix, "dhakaShare": dhaka_share, "dhakaAll": round(100 * dhaka_all, 1), "clock": clock,
            "crimeMaps": crime_maps, "finger": finger, "months": months, "week": week, "rateList": rate_list, "scatter": scatter, "stability": stability}

    t = (HERE / "story_template.html").read_text(encoding="utf-8")
    t = t.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    t = re.sub(r"\{\{(\w+)\}\}", lambda g: V[g.group(1)], t)
    OUT.write_text(t, encoding="utf-8")
    for k in ("inside_pct", "area_pct", "dhaka_n", "ctg_n", "jashore_n", "mym_n", "top_rate_name", "top_rate", "dhaka_rate", "rho", "late_drop", "dh_rob", "dh_mur", "dhaka_all", "city_n"):
        print(f"{k}: {V[k]}")
    print("uni", V["uni_assault"], V["uni_assault_pct"], V["uni_other_pct"], "cramer", V["cramer"], "z_cut", round(float(z_cut), 2))
    print("crime maps", [(c["name"], c["n"], c["dhakaPct"], [(h["tier"][0], h["name"], h["n"]) for h in c["hot"]]) for c in crime_maps])
    print("finger sig", [(r["name"], finger["cols"][i], x_["r"]) for r in finger["rows"] for i, x_ in enumerate(r["v"]) if x_["sig"]])
    print("stability", [(r["name"], r["v"]) for r in stability])
    print("rate list", [(r["name"], r["rate"]) for r in rate_list])
    print("city labels", [(r["name"], r["n"]) for r in data["cityLabels"]])
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
