#!/usr/bin/env python
"""Manuscript Figure 6, sized and formatted for JGR.

Five-degree maps of summed sampled FSOI, conventional versus satellite sources,
one row per verification network, each panel on its own colour scale. Coastlines
are read straight from the bundled Natural Earth 110 m shapefile, so the figure
needs no mapping library.

Drawn at final print size so no text is shrunk by the page layout. The six
repeated panel titles become column headers and row labels, and ticks are drawn
only on the outer edges, which is what buys the room at 170 mm.

AGU graphics requirements: two-column width 105-170 mm, height at most 228 mm,
Arial or Helvetica text no smaller than 8 pt at final print size, raster at
300-600 ppi. Saved as vector PDF plus 600 ppi TIFF and PNG.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

SATELLITE = {"atms", "amsua", "ssmis", "avhrr", "ascat", "seviri_asr", "seviri_csr"}


def _coastlines_110m():
    """Natural Earth 110 m coastlines read straight from cartopy's download cache.

    Used only when cartopy itself is not importable. The maps are plain
    latitude-longitude axes, identical to PlateCarree, so the polylines need no
    reprojection. Returns an empty list when the cache is absent.
    """
    import struct
    shp = (Path.home() / ".local" / "share" / "cartopy" / "shapefiles" / "natural_earth"
           / "physical" / "ne_110m_coastline.shp")
    if not shp.exists():
        return []
    data, pos, lines = shp.read_bytes(), 100, []   # 100-byte file header
    while pos + 8 <= len(data):
        _, words = struct.unpack(">2i", data[pos:pos + 8])
        content, pos = data[pos + 8:pos + 8 + 2 * words], pos + 8 + 2 * words
        if struct.unpack("<i", content[:4])[0] != 3:     # 3 = PolyLine
            continue
        n_parts, n_points = struct.unpack("<2i", content[36:44])
        starts = list(struct.unpack(f"<{n_parts}i", content[44:44 + 4 * n_parts])) + [n_points]
        offset = 44 + 4 * n_parts
        xy = np.frombuffer(content[offset:offset + 16 * n_points], dtype="<f8").reshape(n_points, 2)
        lines += [xy[a:b] for a, b in zip(starts[:-1], starts[1:])]
    return lines

# The 6.5 in text column inside 1 in margins; still inside AGU's
# 105-170 mm range for a two-column figure.


WIDTH_IN = 165.1 / 25.4
BASE_PT = 8.0
INK = "#1a1a1a"
ROWS = [("radiosonde", "Radiosonde"), ("aircraft", "Aircraft"), ("surface_obs", "Surface")]
COLS = [("conventional", "Conventional sources"), ("satellite", "Satellite sources")]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    # Keep math glyphs in Arial too, so the figure carries one font.
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "font.size": BASE_PT,
    "axes.labelsize": BASE_PT,
    "axes.titlesize": BASE_PT,
    "xtick.labelsize": BASE_PT,
    "ytick.labelsize": BASE_PT,
    "axes.linewidth": 0.5,
    "axes.edgecolor": INK,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,          # Type 42, not matplotlib's Type 3 default
    "ps.fonttype": 42,
})


def build(src: Path, out_dir: Path, dpi: int) -> list[Path]:
    d = pd.read_csv(src / "fsoi_grid_5deg.csv")
    d["group"] = np.where(d.instrument.isin(SATELLITE), "satellite", "conventional")
    rows = [(k, lbl) for k, lbl in ROWS if k in set(d.target)]
    lats, lons = np.arange(-90, 90, 5), np.arange(-180, 180, 5)
    coast = _coastlines_110m()

    fig, axes = plt.subplots(len(rows), 2, figsize=(WIDTH_IN, 1.30 * len(rows) + 0.42),
                             squeeze=False, layout="constrained")
    fig.get_layout_engine().set(h_pad=0.03, w_pad=0.03, hspace=0.02, wspace=0.02)

    limits = []
    for i, (target, row_label) in enumerate(rows):
        sub = d[d.target.eq(target)]
        for j, (group, col_label) in enumerate(COLS):
            ax = axes[i][j]
            grid = np.full((len(lats), len(lons)), np.nan)
            g = sub[sub.group.eq(group)].groupby(["ilat", "ilon"], as_index=False).fsoi_sum.sum()
            for ilat, ilon, value in zip(g.ilat, g.ilon, g.fsoi_sum):
                grid[(ilat + 90) // 5, (ilon + 180) // 5] = value
            limit = float(np.nanpercentile(np.abs(g.fsoi_sum.values), 99)) or 1e-12
            # Scale the panel explicitly rather than relying on a formatter offset:
            # the exponent then travels in the colorbar label, where it cannot be
            # dropped or pushed out of place by the layout engine.
            power = 0 if limit >= 1e-2 else int(np.floor(np.log10(limit)))
            scale = 10.0 ** power
            limits.append({"target": target, "group": group, "colour_limit": limit,
                           "label_power": power})
            mesh = ax.pcolormesh(lons, lats, grid / scale, cmap="RdBu_r",
                                 vmin=-limit / scale, vmax=limit / scale,
                                 shading="auto", rasterized=True)
            for line in coast:
                ax.plot(line[:, 0], line[:, 1], color="#2b2b2b", linewidth=0.3)
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_aspect("equal")
            ax.set_xticks([-180, -90, 0, 90, 180])
            ax.set_yticks([-60, 0, 60])
            # Outer edges only: interior labels would not fit at this width.
            if i == len(rows) - 1:
                ax.set_xlabel("Longitude (°)", labelpad=1.5)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(f"{row_label}\nLatitude (°)", labelpad=1.5)
            else:
                ax.set_yticklabels([])
            if i == 0:
                ax.set_title(col_label, pad=3)
            ax.tick_params(pad=1.5)

            cb = fig.colorbar(mesh, ax=ax, shrink=0.88, pad=0.012, aspect=11)
            cb.locator = MaxNLocator(3, symmetric=True)
            cb.update_ticks()
            # The rotated label runs along a colorbar only ~1.1 in tall, so it has to
            # stay short; the caption already says the contributions are summed.
            label = "FSOI" if not power else f"FSOI ($\\times 10^{{{power}}}$)"
            cb.set_label(label, size=BASE_PT + 1.0, labelpad=1.5)
            cb.ax.tick_params(labelsize=BASE_PT, width=0.5, length=2.0, pad=1.0)
            cb.outline.set_linewidth(0.5)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "figure6_fsoi_maps_jgr"
    written = []
    for path, kw in [(out_dir / f"{stem}.pdf", {}),
                     (out_dir / f"{stem}.png", {"dpi": dpi}),
                     (out_dir / f"{stem}.tif", {"dpi": dpi, "pil_kwargs": {"compression": "tiff_lzw"}})]:
        fig.savefig(path, **kw)
        written.append(path)
    plt.close(fig)
    pd.DataFrame(limits).to_csv(out_dir / f"{stem}_colour_limits.csv", index=False)
    written.append(out_dir / f"{stem}_colour_limits.csv")
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=Path("FSOI/fsoi_outputs/paper_figures_v2"))
    p.add_argument("--output", type=Path, default=Path("FSOI/fsoi_outputs/paper_figures_v2"))
    p.add_argument("--dpi", type=int, default=600)
    a = p.parse_args()
    for path in build(a.src, a.output, a.dpi):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
