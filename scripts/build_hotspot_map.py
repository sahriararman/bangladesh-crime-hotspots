"""Build map.html from Bangladesh-Crime-Dataset.csv.

Usage: python -I build_hotspot_map.py [path/to/Bangladesh-Crime-Dataset.csv]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data" / "Bangladesh-Crime-Dataset.csv"
OUT = HERE.parent / "map.html"

HEX_FLAT_KM = 10.0   # hex width, flat to flat
BAND_KM = 20.0       # Gi* neighbourhood radius
BUFFER_KM = 25.0     # grid cells kept within this distance of any recorded location
REF_LAT, REF_LON = 23.7, 90.0
KM_X = 111.32 * math.cos(math.radians(REF_LAT))
KM_Y = 110.57

MIN_CLUSTER = 10
HEX_DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

CRIMES = ["murder", "bodyfound", "rape", "assault", "kidnap", "robbery"]
PARTS = ["morning", "noon", "afternoon", "evening", "night"]
SEASONS = ["hot", "rainy", "winter"]

# (place, district) -> corrected (lat, lon). Approximate upazila/neighbourhood centres.
# These rows carried a shared fallback or a wrong point, and each sat more than ~10 km
# from where the place is. Their population column already names the right upazila.
FIXES = {
    ("anwara", "chattogram"): (22.22, 91.90),
    ("banshkhali", "chattogram"): (22.03, 91.95),
    ("boalkhali", "chattogram"): (22.38, 91.92),
    ("chandanaish", "chattogram"): (22.22, 92.01),
    ("fatikchhari", "chattogram"): (22.69, 91.79),
    ("lohagara", "chattogram"): (22.00, 92.10),
    ("mirsharai", "chattogram"): (22.77, 91.57),
    ("patiya", "chattogram"): (22.30, 91.98),
    ("rangunia", "chattogram"): (22.47, 92.08),
    ("raozan", "chattogram"): (22.53, 91.92),
    ("sandwip", "chattogram"): (22.49, 91.44),
    ("satkania", "chattogram"): (22.08, 92.05),
    ("sitakunda", "chattogram"): (22.62, 91.66),
    ("chougachha", "jashore"): (23.26, 89.03),
    ("dhaka university", "dhaka"): (23.7337, 90.3928),
    ("sir salimullah medical college", "dhaka"): (23.7115, 90.4015),
    ("uttarkhan", "dhaka"): (23.87, 90.43),
    ("babubazar bridge", "dhaka"): (23.7125, 90.4020),
    ("fulbaria", "dhaka"): (23.7265, 90.4095),
    ("shonir akhra", "dhaka"): (23.7014, 90.4624),
    ("goalanda", "rajbari"): (23.74, 89.76),
    ("shibganj", "chapainawabganj"): (24.68, 88.17),
    ("senbug", "noakhali"): (22.98, 91.22),
    ("syedpur", "nilphamari"): (25.78, 88.90),
    ("baghaichari", "rangamati"): (23.19, 92.19),
    ("mohalchari", "khagrachhari"): (23.22, 91.99),
}
# Population says Dhaka metro, but the point is far away and the true spot is unknown.
TO_DHAKA_FALLBACK = {("saidpur", "dhaka"), ("jamtala", "dhaka")}
DHAKA_FALLBACK = (23.784506, 90.403409)
# Points shared by many unrelated neighbourhoods: city-level at best.
CITY_LEVEL = {(23.72891, 90.41714), (23.78451, 90.40341), (23.79441, 90.41374),
              (22.31667, 91.81667), (22.35963, 91.83017)}

LABELS = {"coxsbazar": "Cox's Bazar", "bodyfound": "Body found"}


def nice(s):
    return LABELS.get(s, " ".join(w.capitalize() for w in s.split()))


def load():
    raw = pd.read_csv(CSV, index_col=0)
    df = raw.drop_duplicates().copy()
    keys = list(zip(df.incident_place, df.incident_district))
    moved = 0
    for i, k in enumerate(keys):
        if k in FIXES or k in TO_DHAKA_FALLBACK:
            lat, lon = FIXES.get(k, DHAKA_FALLBACK)
            df.iat[i, df.columns.get_loc("latitude")] = lat
            df.iat[i, df.columns.get_loc("longitude")] = lon
            moved += 1
    df["latitude"] = df.latitude.round(5)
    df["longitude"] = df.longitude.round(5)
    return df, len(raw), moved


def to_km(lat, lon):
    return (lon - REF_LON) * KM_X, (lat - REF_LAT) * KM_Y


def hex_round(x, y, s):
    """Nearest pointy-top hex for km coordinates; s is the centre-to-vertex size."""
    q = (math.sqrt(3) / 3 * x - y / 3) / s
    r = (2 / 3 * y) / s
    cx, cz, cy = q, r, -q - r
    rx, rz, ry = round(cx), round(cz), round(cy)
    dx, dz, dy = abs(rx - cx), abs(rz - cz), abs(ry - cy)
    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy <= dz:
        rz = -rx - ry
    return int(rx), int(rz)


def build_grid(site_xy, s):
    w = math.sqrt(3) * s
    xs, ys = site_xy[:, 0], site_xy[:, 1]
    rmin, rmax = int((ys.min() - BUFFER_KM) / (1.5 * s)) - 1, int((ys.max() + BUFFER_KM) / (1.5 * s)) + 1
    cells = []
    for r in range(rmin, rmax + 1):
        qmin = int((xs.min() - BUFFER_KM) / w - r / 2) - 1
        qmax = int((xs.max() + BUFFER_KM) / w - r / 2) + 1
        for q in range(qmin, qmax + 1):
            cx, cy = w * (q + r / 2), 1.5 * s * r
            if np.min((xs - cx) ** 2 + (ys - cy) ** 2) <= BUFFER_KM ** 2:
                cells.append((q, r, cx, cy))
    return cells


def neighbours(cells, band):
    xy = np.array([(c[2], c[3]) for c in cells])
    out = []
    for i in range(len(xy)):
        d2 = ((xy - xy[i]) ** 2).sum(axis=1)
        out.append(np.nonzero(d2 <= (band + .01) ** 2)[0].tolist())
    return out


def bh(p, testable):
    q = np.ones(len(p))
    order = testable[np.argsort(p[testable], kind="stable")]
    prev, m = 1.0, len(order)
    for rank in range(m, 0, -1):
        prev = min(prev, p[order[rank - 1]] * m / rank)
        q[order[rank - 1]] = prev
    return q


def gi_pass(x, nb, active):
    """One analytic Gi* pass over the active cells, with Benjamini-Hochberg q-values."""
    idx = np.nonzero(active)[0]
    n, mean, sd = len(idx), x[idx].mean(), x[idx].std()
    p, testable = np.ones(len(x)), []
    for i in idx:
        lst = [j for j in nb[i] if active[j]]
        w, total = len(lst), x[lst].sum()
        if total > 0:
            p[i] = stats.norm.sf((total - mean * w) / (sd * math.sqrt((n * w - w * w) / (n - 1))))
            testable.append(i)
    return bh(p, np.array(testable, dtype=int))


def clusters(q, cells, cell_id, x):
    """Two or more contiguous significant cells holding at least MIN_CLUSTER incidents."""
    seen, out = set(), []
    for i in np.nonzero(q < .05)[0]:
        if i in seen:
            continue
        stack, members = [i], []
        seen.add(i)
        while stack:
            c = stack.pop()
            members.append(c)
            for dq, dr in HEX_DIRS:
                j = cell_id.get((cells[c][0] + dq, cells[c][1] + dr))
                if j is not None and q[j] < .05 and j not in seen:
                    seen.add(j)
                    stack.append(j)
        if len(members) >= 2 and x[members].sum() >= MIN_CLUSTER:
            out.append(members)
    return out


def main():
    df, n_raw, moved = load()
    s = HEX_FLAT_KM / math.sqrt(3)

    # Sites
    site_key = df[["latitude", "longitude"]].drop_duplicates().sort_values(["latitude", "longitude"])
    site_key = [tuple(v) for v in site_key.values]
    site_id = {k: i for i, k in enumerate(site_key)}
    site_xy = np.array([to_km(*k) for k in site_key])

    # Grid
    cells = build_grid(site_xy, s)
    cell_id = {(c[0], c[1]): i for i, c in enumerate(cells)}
    w = math.sqrt(3) * s
    site_cell = []
    for x, y in site_xy:
        k = hex_round(x, y, s)
        if k not in cell_id:
            raise RuntimeError(f"site outside grid: {k}")
        site_cell.append(cell_id[k])

    # Compact incident rows
    places = sorted(df.incident_place.unique())
    place_id = {p: i for i, p in enumerate(places)}
    df["unit_key"] = df.incident_district + "|" + df.total_population.astype(str)
    units = sorted(df.unit_key.unique())
    unit_id = {u: i for i, u in enumerate(units)}
    unit_info = []
    for u in units:
        d = df[df.unit_key == u]
        district = d.incident_district.iloc[0]
        top = d.incident_place.value_counts()
        is_metro = len(top) >= 5 and d.density_per_kmsq.iloc[0] >= 10000
        label = f"{nice(district)} metro" if is_metro else nice(top.index[0])
        unit_info.append({"label": label, "district": nice(district), "pop": int(d.total_population.iloc[0])})

    rows = []
    grouped = df.groupby(["latitude", "longitude", "incident_place", "incident_district", "unit_key",
                          "crime", "part_of_the_day", "season"]).size()
    districts = sorted(df.incident_district.unique())
    district_id = {d: i for i, d in enumerate(districts)}
    for (lat, lon, place, dist, unit, crime, part, season), n in grouped.items():
        rows.append([site_id[(lat, lon)], place_id[place], district_id[dist], unit_id[unit],
                     CRIMES.index(crime), PARTS.index(part), SEASONS.index(season), int(n)])

    data = {
        "constants": {"s": s, "kmX": KM_X, "kmY": KM_Y, "refLat": REF_LAT, "refLon": REF_LON,
                      "hexKm": HEX_FLAT_KM, "bandKm": BAND_KM, "minCluster": MIN_CLUSTER},
        "crimes": [nice(c) for c in CRIMES],
        "parts": [nice(p) for p in PARTS],
        "seasons": [nice(p) for p in SEASONS],
        "places": [nice(p) for p in places],
        "districts": [nice(d) for d in districts],
        "units": unit_info,
        "sites": [{"lat": k[0], "lon": k[1], "cell": site_cell[i], "city": int(k in CITY_LEVEL)}
                  for i, k in enumerate(site_key)],
        "cells": [[c[0], c[1], round(c[2], 3), round(c[3], 3)] for c in cells],
        "rows": rows,
        "meta": {"rawRows": n_raw, "incidents": int(len(df)), "dupes": n_raw - len(df),
                 "moved": moved, "sites": len(site_key),
                 "cityLevel": int(sum(n for (la, lo, *_), n in grouped.items() if (la, lo) in CITY_LEVEL))},
    }

    template = (HERE / "map_template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    OUT.write_text(template.replace("/*__DATA__*/null", payload), encoding="utf-8")

    # Build log with sensitivity check
    print(f"rows {n_raw} -> incidents {len(df)} ({n_raw - len(df)} exact duplicates removed)")
    print(f"rows moved {moved}, sites {len(site_key)}, cells {len(cells)}, city-level incidents {data['meta']['cityLevel']}")
    x = np.zeros(len(cells))
    for r in rows:
        x[site_cell[r[0]]] += r[7]
    cell_dist = np.array([site_cell[r[0]] for r in rows])
    dist_n = np.array([r[7] for r in rows])
    dist_id = np.array([r[2] for r in rows])

    def names(members):
        mask = np.isin(cell_dist, members)
        tally = np.bincount(dist_id[mask], weights=dist_n[mask], minlength=len(districts))
        return " / ".join(nice(districts[d]) for d in np.argsort(-tally)[:3] if tally[d] > 0)

    for band in (15.0, BAND_KM, 30.0):
        nbs = neighbours(cells, band)
        national = clusters(gi_pass(x, nbs, np.ones(len(x), bool)), cells, cell_id, x)
        active = np.ones(len(x), bool)
        for m in national:
            active[m] = False
        regional = clusters(gi_pass(x, nbs, active), cells, cell_id, x)
        print(f"band {band:.0f} km")
        for tag, group in (("national", national), ("regional", regional)):
            for m in sorted(group, key=lambda c: -x[c].sum()):
                print(f"  {tag:8} {int(x[m].sum()):>5} incidents {len(m):>3} cells  {names(m)}")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
