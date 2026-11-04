# Bangladesh crime hotspots

A data journalism project on where reported crime concentrates in Bangladesh.

| File | What it is |
|---|---|
| `index.html` | Scrolling data story with maps and charts |
| `map.html` | Interactive map with crime, time, season and radius filters |
| `rape-map.png`, `rape-map.pdf` | Static A3 map of reported rape incidents by district |
| `scripts/` | Python build scripts and HTML templates |

The pages need an internet connection for map tiles, fonts and libraries.

## Data and sources

- **Incidents:** `Bangladesh-Crime-Dataset.csv`. The file states no source, licence or years, so it is not included in this repository.
- **Basemaps:** OpenStreetMap contributors, OpenFreeMap, OpenMapTiles, and OpenStreetMap Humanitarian tiles.
- **Static map boundaries:** geoBoundaries (CC BY 4.0) and Natural Earth.
- **Static map population:** Bangladesh Bureau of Statistics, Census 2022.

## Method

- 385 exact duplicate rows were removed from 6,574, leaving 6,189 incidents. Some could be separate victims.
- 278 rows were moved because their coordinates were wrong. The new positions are approximate and are listed in `FIXES` in `scripts/build_hotspot_map.py`.
- Hotspots use the Getis-Ord Gi* statistic on 10 km hexagons, with a Benjamini-Hochberg false discovery rate of 5%. A second pass removes the major hotspots to find regional ones.
- Hotspot outlines are smoothed, so their edges are approximate.

## Limits

- These are reported incidents, not all crime. Differences between places can reflect reporting and press coverage.
- With no years in the file, nothing here shows a trend.
- Points mark the centre of a named place, not an exact address. Some have only a city-level location.
- Regional hotspots change with the search radius, so treat them as leads to check.
- The results show totals for places and identify no victims or individual cases.

## Rebuilding

Put the CSV at `data/Bangladesh-Crime-Dataset.csv`, or pass its path as an argument. The scripts need Python with pandas, numpy, scipy and matplotlib.

```bash
python -I scripts/build_hotspot_map.py
python -I scripts/build_story.py
python -E -P scripts/build_rape_map.py
```

The static map also expects boundary and census files in a sibling folder. Set `REF` in `scripts/build_rape_map.py` to their location.
