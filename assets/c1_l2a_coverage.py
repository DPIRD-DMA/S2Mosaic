"""Measure and plot how far Collection-1 has backfilled the Sentinel-2 archive.

Renders ``assets/c1-l2a-coverage.png``: both of Element 84's L2A collections as
a share of the parent L1C product, so L1C is the 100% reference and the current
default and the candidate sit on one axis. The chart is the evidence behind
``TestAwsCollectionArchiveCoverage`` and behind issue #5, which asks to move
``Source.collection_id`` to ``sentinel-2-c1-l2a``.

Re-run it to check whether the backfill has caught up::

    uv run python assets/c1_l2a_coverage.py

Needs ``matplotlib`` (dev group) and ``requests`` (pulled in by pystac-client).
About 1200 catalogue requests, a few minutes.

Counting
--------
Acquisitions are keyed on the **datatake id** parsed from ``s2:product_uri``,
not on the ``datetime`` property. The Collection-1 reprocessing restamps an
acquisition's sensing time by minutes, so a sensing-time key counts one
overpass twice and lifts L2A above its own parent product -- it also understated
Collection-1's 2018 coverage by ten points. The datatake id is assigned at
acquisition, survives every reprocessing, and is identical between collections.

(Package deduplication has the same problem but cannot use this key, because one
datatake can yield two granules over a single tile. ``_acquisition_key`` in
``s2mosaic/stac.py`` keys on the datastrip sensing start instead, which
separates those two while still surviving reprocessing.)
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import matplotlib

# Select the headless backend before pyplot is imported, so this runs without
# a display. That is why these imports sit below a statement.
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import requests
from matplotlib.ticker import FuncFormatter

SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
OUT_PATH = Path(__file__).with_name("c1-l2a-coverage.png")

REFERENCE = "sentinel-2-l1c"
COLLECTIONS = [REFERENCE, "sentinel-2-l2a", "sentinel-2-c1-l2a"]
LABELS = {
    "sentinel-2-l2a": "sentinel-2-l2a  (current default)",
    "sentinel-2-c1-l2a": "sentinel-2-c1-l2a  (candidate)",
}
YEARS = list(range(2016, 2027))

# Sampled once from the catalogue: three tiles from each of twelve land regions,
# taken from a June 2024 search rather than chosen by hand. Fixed here so a
# re-run is comparable with the committed chart -- re-sampling would move the
# numbers for reasons unrelated to the backfill. Per-tile coverage spans only
# 70-77%, so the shortfall is temporal rather than regional.
REGIONS = {
    "W Australia": ("49HGC", "49HGD", "49HGE"),
    "E Australia": ("55HEA", "55HFA", "55HFB"),
    "W Europe": ("30TWP", "30TWQ", "30TWR"),
    "E Europe": ("34TDP", "34TDQ", "34TDR"),
    "N America W": ("10SEE", "10SEF", "10SEG"),
    "N America E": ("17SNB", "17SNC", "17SND"),
    "S America": ("20JQM", "20JQN", "20JQP"),
    "Africa N": ("31PCS", "31PCT", "31PDS"),
    "Africa S": ("34JGM", "35JKG", "35JLG"),
    "India": ("43QEC", "43QFC", "43QFD"),
    "SE Asia": ("47PNM", "47PNN", "47PNP"),
    "E Asia": ("49RGP", "49SGB", "49SGC"),
}
TILES = [tile for tiles in REGIONS.values() for tile in tiles]

MGRS_RE = re.compile(r"(\d{1,2})([A-Z])([A-Z]{2})")
DATATAKE_RE = re.compile(r"\d{8}T\d{6}")
FIELDS = {
    "include": ["properties.s2:product_uri", "properties.datetime"],
    "exclude": ["geometry", "assets", "links", "bbox", "stac_extensions", "collection"],
}

SURFACE = "#fcfcfb"
INK = "#17160f"
INK_2 = "#52514e"
INK_3 = "#78766e"
GRID = "#e6e4dc"
# Categorical slots 1 and 2; validated for contrast and colour-vision
# separation against this surface.
SERIES_COLOURS = {"sentinel-2-l2a": "#2a78d6", "sentinel-2-c1-l2a": "#eb6834"}

# Direct-label a year only where Collection-1 is materially short; above
# this the two lines sit on the reference and a label would be noise.
LABEL_BELOW_PCT = 95.0


def mgrs_query(tile: str) -> dict[str, object]:
    """Element 84 splits the MGRS id across three queryable fields."""
    match = MGRS_RE.fullmatch(tile)
    if match is None:
        raise ValueError(f"{tile!r} is not an MGRS tile id like '50HMH'")
    zone, band, square = match.groups()
    return {
        "mgrs:utm_zone": {"eq": int(zone)},
        "mgrs:latitude_band": {"eq": band},
        "mgrs:grid_square": {"eq": square},
    }


def datatake_of(properties: dict[str, object]) -> str:
    """The datatake id, third field of the product URI.

        S2B_MSIL2A_20190323T021349_N0500_R060_T50HMH_20221118T090644.SAFE
                   └── this one ──┘

    Falls back to the datetime property, which is what this script exists to
    avoid, so a fallback is reported rather than passed over.
    """
    uri = properties.get("s2:product_uri")
    if isinstance(uri, str):
        parts = uri.split("_")
        if len(parts) > 2 and DATATAKE_RE.fullmatch(parts[2]):
            return parts[2]
    return "datetime:" + str(properties.get("datetime", ""))[:19]


def count_datatakes(
    session: requests.Session, collection: str, tile: str, year: int
) -> set[str]:
    body: dict[str, object] = {
        "collections": [collection],
        "datetime": f"{year}-01-01T00:00:00Z/{year + 1}-01-01T00:00:00Z",
        "query": mgrs_query(tile),
        "limit": 100,
        "fields": FIELDS,
    }
    seen: set[str] = set()
    while body is not None:
        response = None
        for _ in range(3):
            try:
                response = session.post(SEARCH_URL, json=body, timeout=120)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                response = None
        if response is None or response.status_code != 200:
            raise RuntimeError(f"search failed for {collection} {tile} {year}")
        page = response.json()
        for feature in page.get("features", []):
            seen.add(datatake_of(feature["properties"]))
        nxt = next(
            (ln for ln in page.get("links", []) if ln.get("rel") == "next"), None
        )
        body = {**body, **nxt["body"]} if nxt and "body" in nxt else None
    return seen


def gather() -> dict[str, dict[int, int]]:
    jobs = [(c, t, y) for c in COLLECTIONS for t in TILES for y in YEARS]

    def run(job: tuple[str, str, int]) -> tuple[tuple[str, str, int], set[str]]:
        collection, tile, year = job
        return job, count_datatakes(requests.Session(), collection, tile, year)

    totals: dict[str, dict[int, int]] = {c: defaultdict(int) for c in COLLECTIONS}
    fallbacks = 0
    with ThreadPoolExecutor(max_workers=12) as pool:
        for (collection, _tile, year), datatakes in pool.map(run, jobs):
            totals[collection][year] += len(datatakes)
            fallbacks += sum(1 for d in datatakes if d.startswith("datetime:"))
    if fallbacks:
        print(f"warning: {fallbacks} acquisitions had no parseable datatake id")
    return {c: dict(v) for c, v in totals.items()}


def share(totals: dict[str, dict[int, int]], collection: str) -> list[float]:
    return [
        100.0 * totals[collection].get(y, 0) / totals[REFERENCE][y]
        if totals[REFERENCE].get(y)
        else 0.0
        for y in YEARS
    ]


def print_table(totals: dict[str, dict[int, int]]) -> None:
    print(f"\n{'year':<6}{'L1C':>8}{'L2A':>8}{'C1-L2A':>9}{'C1/L1C':>9}")
    for year in YEARS:
        l1c, l2a, c1 = (totals[c].get(year, 0) for c in COLLECTIONS)
        pct = f"{100 * c1 / l1c:.1f}%" if l1c else "--"
        print(f"{year:<6}{l1c:>8}{l2a:>8}{c1:>9}{pct:>9}")


def label_points(ax: plt.Axes, values: Iterable[float], halo: list[object]) -> None:
    """Direct-label only the years Collection-1 is materially short.

    2016 is skipped: it holds the archive's first weeks, so its point sits on
    the axis and a label there reads as noise rather than information.
    """
    for year, pct in zip(YEARS, values, strict=False):
        if pct >= LABEL_BELOW_PCT or year == YEARS[0]:
            continue
        text = f"{pct:.1f}%" if pct < 10 else f"{pct:.0f}%"
        ax.annotate(
            text,
            xy=(year, pct),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=INK_2,
            zorder=4,
            path_effects=halo,
        )


def render(totals: dict[str, dict[int, int]], measured_on: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
        }
    )
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=200)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.815, bottom=0.265)

    # Labels ride over two steep lines, so a surface halo does the separating
    # rather than nudging them away from the point they belong to.
    halo = [pe.withStroke(linewidth=3.5, foreground=SURFACE)]

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.xaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_linewidth(0.8)

    # The L1C reference. Not a threshold, so it stays solid and recessive.
    ax.axhline(100, color=INK_3, linewidth=0.9, zorder=2)
    ax.annotate(
        "L1C = 100%",
        xy=(YEARS[0], 100),
        xytext=(0, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=INK_3,
        path_effects=halo,
    )

    style = {
        "linewidth": 2.0,
        "marker": "o",
        "markersize": 5.2,
        "markeredgewidth": 1.4,
        "markeredgecolor": SURFACE,
        "clip_on": False,
        "zorder": 3,
    }
    for collection in ("sentinel-2-l2a", "sentinel-2-c1-l2a"):
        ax.plot(
            YEARS,
            share(totals, collection),
            color=SERIES_COLOURS[collection],
            label=LABELS[collection],
            **style,
        )
    label_points(ax, share(totals, "sentinel-2-c1-l2a"), halo)

    ax.set_ylim(-3, 118)
    ax.set_xlim(YEARS[0] - 0.4, YEARS[-1] + 0.4)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xticks(YEARS)
    ax.tick_params(axis="both", length=0, colors=INK_3, labelsize=9.5, pad=7)

    legend = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005),
        frameon=False,
        ncol=2,
        handlelength=1.5,
        handletextpad=0.6,
        columnspacing=2.0,
        fontsize=10,
    )
    for text in legend.get_texts():
        text.set_color(INK_2)

    fig.text(
        0.085,
        0.945,
        "Collection-1 has not backfilled the Sentinel-2 archive",
        fontsize=15,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
    )
    fig.text(
        0.085,
        0.893,
        "Unique acquisitions as a share of the parent L1C product"
        f"  ·  {len(TILES)} MGRS tiles worldwide  ·  {measured_on}",
        fontsize=9.5,
        color=INK_3,
        ha="left",
        va="top",
    )
    footnote = "\n".join(
        [
            (
                "Counts are unique acquisitions keyed on the datatake id, not STAC"
                " items and not sensing time \u2014 the Collection-1 reprocessing"
                " restamps an"
            ),
            (
                "acquisition's sensing time by minutes, so a sensing-time key counts"
                " one overpass twice and lifts L2A above its own parent product."
                " 2016 is"
            ),
            (
                "the archive's first months, only 55 L1C acquisitions across all 36"
                " tiles. ESA holds the data missing from 2022: Dataspace lists 150"
                " L2A"
            ),
            (
                "granules for 50HMH against Earth Search's 13, so this is an Element"
                " 84 ingestion backlog rather than missing upstream reprocessing."
            ),
        ]
    )
    fig.text(
        0.085,
        0.185,
        footnote,
        fontsize=8,
        color=INK_3,
        ha="left",
        va="top",
        linespacing=1.7,
    )

    fig.savefig(OUT_PATH, dpi=200)
    print(f"wrote {OUT_PATH}")


def main() -> None:
    print(
        f"querying {len(COLLECTIONS)} collections"
        f" x {len(TILES)} tiles x {len(YEARS)} years"
    )
    totals = gather()
    print_table(totals)
    render(totals, date.today().strftime("%-d %b %Y"))


if __name__ == "__main__":
    main()
