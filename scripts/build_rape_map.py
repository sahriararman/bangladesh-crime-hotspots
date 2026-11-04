"""Static print map of reported rape incidents: rape-map.png and .pdf

Usage: python -E -P build_rape_map.py [path/to/Bangladesh-Crime-Dataset.csv]
Boundaries and census come from reference_repo_bd_road_crashes (geoBoundaries, Natural Earth, BBS).
"""
import json
import math
import os
import sys
from pathlib import Path as FsPath

HERE = FsPath(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Polygon as MplPolygon, Rectangle
from matplotlib.path import Path

import build_hotspot_map as B
from build_story import smooth_ring

REF = HERE.parent.parent / "Bangladesh Historical Rainfall and Meteorological" / "reference_repo_bd_road_crashes"
BASEMAP = REF / "assets" / "basemap"
OUT = HERE.parent / "rape-map"

COS = math.cos(math.radians(23.7))
EXTENT = (87.75, 92.95, 20.45, 26.85)
DHAKA_BOX = (89.98, 90.82, 23.42, 24.2)
CLASS_COLORS = ["#fff0d9", "#fdc98b", "#f78f58", "#d8452f", "#8c1518"]
INK, MUTED, SEA, LAND = "#1c1f22", "#5b6066", "#d7e6ef", "#eeeee8"

# Dataset spelling -> boundary file spelling
GEO_NAME = {"bogura": "Bogra", "brahmanbaria": "Brahamanbaria", "chapainawabganj": "Nawabganj", "chattogram": "Chittagong",
            "coxsbazar": "Cox's Bazar", "jashore": "Jessore", "jhalakathi": "Jhalokati", "moulvibazar": "Maulvibazar", "netrokona": "Netrakona"}
# Any spelling -> one key, so the census file joins whichever form it uses
CANON = {"bogra": "bogura", "brahamanbaria": "brahmanbaria", "nawabganj": "chapainawabganj", "chapai nawabganj": "chapainawabganj", "chittagong": "chattogram",
         "cox's bazar": "coxsbazar", "cox's bazaar": "coxsbazar", "coxs bazar": "coxsbazar", "jessore": "jashore", "jhalokati": "jhalakathi", "jhalokathi": "jhalakathi", "maulvibazar": "moulvibazar",
         "netrakona": "netrokona", "barishal": "barisal", "cumilla": "comilla", "habiganj": "habiganj", "hobiganj": "habiganj", "habigonj": "habiganj"}
canon = lambda s: CANON.get(str(s).strip().lower(), str(s).strip().lower())
SHOW = {"coxsbazar": "Cox's Bazar"}
show = lambda k: SHOW.get(k, B.nice(k))


def load_geo(name):
    with open(BASEMAP / f"{name}.geojson", encoding="utf-8") as f:
        return json.load(f)["features"]


def rings(geom):
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    return polys


def compound(geom):
    """Polygon or MultiPolygon as one matplotlib path, with holes."""
    verts, codes = [], []
    for poly in rings(geom):
        for ring in poly:
            verts += [tuple(p[:2]) for p in ring]
            codes += [Path.MOVETO] + [Path.LINETO] * (len(ring) - 2) + [Path.CLOSEPOLY]
    return Path(verts, codes)


def lines(geom):
    return [geom["coordinates"]] if geom["type"] == "LineString" else geom["coordinates"]


def jenks(values, k):
    """Fisher-Jenks natural breaks; returns the upper bound of each class."""
    v = np.sort(np.asarray(values, float))
    n = len(v)
    pre, pre2 = np.r_[0, np.cumsum(v)], np.r_[0, np.cumsum(v ** 2)]
    cost = lambda i, j: (pre2[j + 1] - pre2[i]) - (pre[j + 1] - pre[i]) ** 2 / (j - i + 1)
    D = np.full((k, n), np.inf)
    S = np.zeros((k, n), int)
    for j in range(n):
        D[0, j] = cost(0, j)
    for c in range(1, k):
        for j in range(c, n):
            for i in range(c, j + 1):
                val = D[c - 1, i - 1] + cost(i, j)
                if val < D[c, j]:
                    D[c, j], S[c, j] = val, i
    ups, j = [], n - 1
    for c in range(k - 1, -1, -1):
        ups.append(v[j])
        j = S[c, j] - 1
    return ups[::-1]


def hotspots(df):
    s = B.HEX_FLAT_KM / math.sqrt(3)
    keys = sorted(set(zip(df.latitude, df.longitude)))
    xy = np.array([B.to_km(*k) for k in keys])
    cells = B.build_grid(xy, s)
    cell_id = {(c[0], c[1]): i for i, c in enumerate(cells)}
    site_cell = {k: cell_id[B.hex_round(*B.to_km(*k), s)] for k in keys}
    d = df[df.crime == "rape"].copy()
    d["cell"] = [site_cell[k] for k in zip(d.latitude, d.longitude)]
    v = np.bincount(d.cell, minlength=len(cells)).astype(float)
    nb = B.neighbours(cells, B.BAND_KM)
    major = B.clusters(B.gi_pass(v, nb, np.ones(len(v), bool)), cells, cell_id, v)
    active = np.ones(len(v), bool)
    for m in major:
        active[m] = False
    regional = B.clusters(B.gi_pass(v, nb, active), cells, cell_id, v)
    out = []
    for tier, group in (("major", major), ("regional", regional)):
        for m in sorted(group, key=lambda c: -v[c].sum()):
            name = B.nice(d[d.cell.isin(m)].incident_district.value_counts().index[0])
            out.append({"tier": tier, "name": "Greater Dhaka" if name == "Dhaka" else name, "n": int(v[m].sum()), "ring": smooth_ring(m, cells, s)})
    return d, out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    plt.rcParams["font.family"] = ["Segoe UI", "Arial", "DejaVu Sans"]
    df, _, _ = B.load()
    d, hot = hotspots(df)

    pop_rows = json.load(open(REF / "data" / "population_admin.json", encoding="utf-8"))
    pop = {canon(r["name"]): r["pop_2022"] for r in pop_rows if r["level"] == "district"}
    counts = d.incident_district.map(canon).value_counts()
    missing = [k for k in counts.index if k not in pop]
    assert not missing, f"districts with no census match: {missing}"

    districts = load_geo("districts")
    t = pd.DataFrame({"key": [canon(f["properties"]["district"]) for f in districts]})
    unmatched = [k for k in t.key if k not in pop]
    assert not unmatched, f"boundary names with no census match: {unmatched}"
    t["n"] = t.key.map(counts).fillna(0).astype(int)
    t["pop"] = t.key.map(pop)
    t["rate"] = 1e5 * t.n / t["pop"]
    assert t.n.sum() == len(d), (t.n.sum(), len(d))
    breaks = jenks(t.rate, 5)
    t["cls"] = np.searchsorted(breaks, t.rate - 1e-12)
    rate_of = dict(zip(t.key, t.rate))
    cls_of = dict(zip(t.key, t.cls))

    fig = plt.figure(figsize=(16.54, 11.69), facecolor="white")   # A3 landscape

    # ---------- main map frame ----------
    x0, x1, y0, y1 = EXTENT
    map_h = .885
    map_w = map_h * 11.69 * ((x1 - x0) * COS) / (y1 - y0) / 16.54
    ax = fig.add_axes([.028, .058, map_w, map_h], facecolor=SEA)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect(1 / COS)

    for f in load_geo("countries"):
        if f["properties"]["name"] != "Bangladesh":
            ax.add_patch(PathPatch(compound(f["geometry"]), fc=LAND, ec="#a9aca6", lw=.7, zorder=1))
    for f in districts:
        k = canon(f["properties"]["district"])
        ax.add_patch(PathPatch(compound(f["geometry"]), fc=CLASS_COLORS[cls_of[k]], ec="#ffffff", lw=.55, zorder=2))
    for f in load_geo("divisions"):
        ax.add_patch(PathPatch(compound(f["geometry"]), fc="none", ec="#3b3f44", lw=1.05, zorder=3))
    for f in load_geo("lakes"):
        ax.add_patch(PathPatch(compound(f["geometry"]), fc=SEA, ec="#7fa8c2", lw=.4, zorder=3))
    for f in load_geo("rivers"):
        for ln in lines(f["geometry"]):
            a = np.array(ln)
            ax.plot(a[:, 0], a[:, 1], color="#5f97bb", lw=.8, alpha=.75, zorder=3, solid_capstyle="round")

    ax.grid(True, color="#7c8087", lw=.4, ls=(0, (4, 4)), alpha=.55, zorder=4)
    ax.set_axisbelow(False)
    ax.set_xticks(np.arange(88, 93, 1)); ax.set_yticks(np.arange(21, 27, 1))
    ax.set_xticklabels([f"{v}°E" for v in range(88, 93)]); ax.set_yticklabels([f"{v}°N" for v in range(21, 27)])
    ax.tick_params(labelsize=9.5, length=4, color=INK, labelcolor=INK, top=True, right=True, labeltop=True, labelright=True)
    for sp in ax.spines.values():
        sp.set_linewidth(1.2); sp.set_color(INK); sp.set_zorder(20)

    for h in hot:
        a = np.array(h["ring"])
        ax.plot(a[:, 0], a[:, 1], color=INK, lw=1.9 if h["tier"] == "major" else 1.3,
                ls="-" if h["tier"] == "major" else (0, (3.5, 2.5)), zorder=6, solid_joinstyle="round")

    at = d.groupby(["latitude", "longitude"]).size().rename("n").reset_index().sort_values("n", ascending=False)
    size = lambda n: 7 + 8.5 * np.asarray(n, float)
    ax.scatter(at.longitude, at.latitude, s=size(at.n), fc="#20262b", ec="white", lw=.45, alpha=.82, zorder=7)

    halo = lambda c="white", w=2.6: [pe.withStroke(linewidth=w, foreground=c)]
    label_pos = {"Greater Dhaka": (91.02, 24.28), "Jashore": (88.32, 22.42), "Chattogram": (91.08, 21.62),
                 "Jhalakathi": (90.32, 21.5), "Lalmonirhat": (89.98, 26.42)}
    for h in hot:
        ring = np.array(h["ring"])
        xa, ya = label_pos[h["name"]]
        aim = ring[np.argmin((ring[:, 0] - xa) ** 2 + (ring[:, 1] - ya) ** 2)]
        ax.annotate(f'{h["name"]}\n{h["n"]} incidents', xy=aim, xytext=(xa, ya), fontsize=9.5, fontweight="bold", color=INK, ha="center", va="center", linespacing=1.1, zorder=9,
                    path_effects=halo(), arrowprops=dict(arrowstyle="-", color=INK, lw=.9, shrinkA=1, shrinkB=0, path_effects=halo("white", 2.2)))
    ax.text(92.38, 21.92, "Bandarban", fontsize=9, fontweight="bold", color="white", ha="center", va="center", zorder=9, path_effects=halo("#5e0d10", 2))
    ax.add_patch(Rectangle((DHAKA_BOX[0], DHAKA_BOX[2]), DHAKA_BOX[1] - DHAKA_BOX[0], DHAKA_BOX[3] - DHAKA_BOX[2], fc="none", ec="#1766a3", lw=1.2, zorder=8))
    for name, lon, lat, size_, rot in [("INDIA", 88.25, 25.1, 12, 0), ("INDIA", 92.1, 25.75, 12, 0), ("INDIA", 91.95, 23.95, 10, 0), ("MYANMAR", 92.72, 21.5, 10.5, 90)]:
        ax.text(lon, lat, name, fontsize=size_, color="#7b7f79", ha="center", va="center", rotation=rot, zorder=5)
    ax.text(90.25, 20.98, "Bay of Bengal", fontsize=14, style="italic", color="#3f7596", ha="center", zorder=5)

    # North arrow, in axes fraction
    nx, ny = .058, .885
    ax.add_patch(MplPolygon([(nx, ny + .058), (nx - .018, ny), (nx, ny + .012)], closed=True, fc=INK, ec=INK, lw=.8, transform=ax.transAxes, zorder=12))
    ax.add_patch(MplPolygon([(nx, ny + .058), (nx + .018, ny), (nx, ny + .012)], closed=True, fc="white", ec=INK, lw=.8, transform=ax.transAxes, zorder=12))
    ax.text(nx, ny + .067, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=14, fontweight="bold", color=INK, zorder=12)

    # Scale bar at 21°N, drawn in data units using the map's reference latitude
    km_deg = 111.32 * COS
    sx, sy, seg = 87.95, 20.72, 50 / km_deg
    for i in range(2):
        ax.add_patch(Rectangle((sx + i * seg, sy), seg, .035, fc=INK if i % 2 == 0 else "white", ec=INK, lw=.8, zorder=12))
    for i, lab in enumerate(["0", "50", "100 km"]):
        ax.text(sx + i * seg, sy + .055, lab, fontsize=9, ha="center", va="bottom", color=INK, zorder=12, path_effects=halo())

    # ---------- right column ----------
    cx = .028 + map_w + .028
    cw = 1 - cx - .022
    fig.text(cx, .925, "Reported rape incidents\nin Bangladesh", fontsize=25, fontweight="bold", color=INK, va="bottom", linespacing=1.08)
    fig.text(cx, .905, f"{len(d):,} incidents by district, location and statistical hotspot", fontsize=12, color=MUTED, va="center")
    fig.add_artist(Line2D([cx, cx + cw], [.888, .888], color=INK, lw=1.2))

    # Legend
    lax = fig.add_axes([cx, .535, cw * .5, .345]); lax.axis("off")
    low = [t.rate.min()] + breaks[:-1]
    class_counts = t.cls.value_counts()
    h1 = [Patch(fc=CLASS_COLORS[i], ec="#8d9096", lw=.5, label=f"{low[i]:.2f} – {breaks[i]:.2f}   ({class_counts.get(i, 0)})") for i in range(5)]
    l1 = lax.legend(handles=h1, title="Incidents per 100,000 people\n(districts in brackets)", loc="upper left", bbox_to_anchor=(0, 1), frameon=False, fontsize=10, title_fontsize=10.5,
                    alignment="left", labelspacing=.45, handlelength=1.8, handleheight=1.1, borderpad=0)
    l1.get_title().set_fontweight("bold")
    lax.add_artist(l1)
    h2 = [Line2D([], [], marker="o", ls="", mfc="#20262b", mec="white", mew=.45, ms=math.sqrt(size(n)), label=str(n)) for n in (1, 10, 25, 42)]
    l2 = lax.legend(handles=h2, title="Incidents at each location", loc="upper left", bbox_to_anchor=(0, .52), frameon=False, fontsize=10, title_fontsize=10.5, alignment="left",
                    ncol=4, columnspacing=1.3, handletextpad=.5, borderpad=0)
    l2.get_title().set_fontweight("bold")
    lax.add_artist(l2)
    h3 = [Line2D([], [], color=INK, lw=1.9, label="Major hotspot"), Line2D([], [], color=INK, lw=1.3, ls=(0, (3.5, 2.5)), label="Regional hotspot"),
          Line2D([], [], color="#3b3f44", lw=1.05, label="Division boundary"), Line2D([], [], color="#b5b7ba", lw=.9, label="District boundary"),
          Line2D([], [], color="#5f97bb", lw=1, label="Major river")]
    l3 = lax.legend(handles=h3, title="Map features", loc="upper left", bbox_to_anchor=(0, .3), frameon=False, fontsize=10, title_fontsize=10.5, alignment="left", labelspacing=.35, borderpad=0)
    l3.get_title().set_fontweight("bold")

    # Greater Dhaka detail: same data, closer in, because points overlap at national scale
    iax = fig.add_axes([cx + cw * .535, .535, cw * .465, .345], facecolor=LAND)
    for f in districts:
        k = canon(f["properties"]["district"])
        iax.add_patch(PathPatch(compound(f["geometry"]), fc=CLASS_COLORS[cls_of[k]], ec="#ffffff", lw=.9, zorder=2))
    for f in load_geo("rivers"):
        for ln in lines(f["geometry"]):
            a = np.array(ln)
            iax.plot(a[:, 0], a[:, 1], color="#5f97bb", lw=1, alpha=.75, zorder=3)
    for h in hot:
        a = np.array(h["ring"])
        iax.plot(a[:, 0], a[:, 1], color=INK, lw=1.9 if h["tier"] == "major" else 1.3, ls="-" if h["tier"] == "major" else (0, (3.5, 2.5)), zorder=4)
    iax.scatter(at.longitude, at.latitude, s=size(at.n), fc="#20262b", ec="white", lw=.5, alpha=.82, zorder=5)
    iax.set_xlim(DHAKA_BOX[0], DHAKA_BOX[1]); iax.set_ylim(DHAKA_BOX[2], DHAKA_BOX[3]); iax.set_aspect(1 / COS, adjustable="box", anchor="NE")
    iax.set_xticks([]); iax.set_yticks([])
    place = d.groupby(["latitude", "longitude"]).agg(n=("crime", "size"), name=("incident_place", lambda q: q.value_counts().index[0])).reset_index()
    place = place[place.longitude.between(*DHAKA_BOX[:2]) & place.latitude.between(*DHAKA_BOX[2:]) & place.apply(lambda r: (r.latitude, r.longitude) not in B.CITY_LEVEL, axis=1)]
    place = place.sort_values("n", ascending=False).head(4)
    for r in place.itertuples():
        iax.annotate(f"{B.nice(r.name)} {r.n}", (r.longitude, r.latitude), xytext=(7, 7), textcoords="offset points", fontsize=8.5, color=INK, fontweight="bold", zorder=6, path_effects=halo())
    iax.text(.03, .965, "Detail: Greater Dhaka", transform=iax.transAxes, fontsize=10, fontweight="bold", color=INK, va="top", zorder=7, path_effects=halo())
    for sp in iax.spines.values():
        sp.set_linewidth(1.4); sp.set_color("#1766a3")

    # Table
    top = t.sort_values("rate", ascending=False).head(10)
    tax = fig.add_axes([cx, .205, cw, .315]); tax.axis("off"); tax.set_xlim(0, 1); tax.set_ylim(0, 1)
    tax.text(0, .985, "Highest rates by district", fontsize=12, fontweight="bold", color=INK, va="top")
    hy, step = .885, .083
    cols = [(0, "left", "District"), (.56, "right", "Incidents"), (.77, "right", "Population"), (1, "right", "Per 100,000")]
    for xp, ha, lab in cols:
        tax.text(xp, hy, lab, fontsize=9.5, ha=ha, va="center", color=MUTED)
    tax.plot([0, 1], [hy - .04, hy - .04], color=INK, lw=.8)
    for i, r in enumerate(top.itertuples()):
        y = hy - .083 - i * step
        tax.text(0, y, f"{i + 1}.  {show(r.key)}", fontsize=10.5, va="center", color=INK)
        tax.text(.56, y, f"{r.n}", fontsize=10.5, ha="right", va="center", color=INK)
        tax.text(.77, y, f"{r.pop / 1e6:.2f} m", fontsize=10.5, ha="right", va="center", color=INK)
        tax.text(1, y, f"{r.rate:.2f}", fontsize=10.5, ha="right", va="center", color=INK, fontweight="bold")
        if i < 9:
            tax.plot([0, 1], [y - step / 2, y - step / 2], color="#dcdde0", lw=.5)

    # Notes
    fig.add_artist(Line2D([cx, cx + cw], [.195, .195], color="#8d9096", lw=.7))
    fig.text(cx, .186, (
        "How to read this map. These are reported incidents, not all crime: rape is widely under-reported, and differences between districts\n"
        "can reflect reporting and press coverage. Low rates do not show that a district is safer. The source file gives no year or source, so\n"
        "the rates compare incident counts from an undated file with 2022 populations and work only for comparison, not as true incidence.\n\n"
        "Points mark the centre of a named place, not an exact address, and some positions were corrected by hand. Hotspots use the\n"
        "Getis-Ord Gi* statistic on incident counts (10 km grid, 20 km radius, 5% false discovery rate); edges are smoothed and approximate.\n"
        "Regional hotspots change with the radius. Classes use natural breaks (Jenks). Totals only: no victim or case can be identified.\n\n"
        "Data: Bangladesh-Crime-Dataset.csv (duplicates removed). Boundaries: geoBoundaries (CC BY 4.0). Rivers, lakes, neighbouring\n"
        "countries: Natural Earth. Population: BBS Census 2022. Coordinates: WGS 84 (EPSG:4326), plotted with a fixed aspect at 23.7°N."),
        fontsize=8.4, color=MUTED, va="top", linespacing=1.32)

    fig.savefig(f"{OUT}.png", dpi=300, facecolor="white")
    fig.savefig(f"{OUT}.pdf", facecolor="white")

    labs = d.assign(k=d.incident_district.map(canon))
    print(f"incidents {len(d)}; breaks {[round(b, 2) for b in breaks]}; class counts {class_counts.sort_index().to_dict()}")
    print("top", [(show(r.key), r.n, round(r.rate, 2)) for r in top.itertuples()])
    print("lowest", [(show(r.key), r.n, round(r.rate, 2)) for r in t.sort_values("rate").head(5).itertuples()])
    print("hotspots", [(h["tier"], h["name"], h["n"]) for h in hot], "max at site", int(at.n.max()))
    print(f"wrote {OUT}.png and .pdf")


if __name__ == "__main__":
    main()
