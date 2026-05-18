#!/usr/bin/env python
"""Build initialization lists from available GFS cycles.

Supported sampling modes:

paired_cycles:
    Select N dates and emit all requested cycles for each date.
    Example: --num-dates 15 --cycles 00,12 -> 30 init times.

balanced_distinct:
    Select NUM_INITS distinct date-cycle pairs, balanced across cycles.
    Example: --num-inits 30 --cycles 00,12 -> 15 00Z + 15 12Z,
    preferably on distinct dates.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _has_required_files(root: Path, ymd: str, hh: str, required_fhrs: list[int]) -> bool:
    for fhr in required_fhrs:
        p = root / ymd / f"gfs.{ymd}.t{hh}z.pgrb2.0p25.f{fhr:03d}"
        if not p.exists():
            return False
    return True


def _evenly_select(items: list[str], n: int) -> list[str]:
    if n <= 0:
        return []
    if len(items) < n:
        raise ValueError(f"Need {n} items, only found {len(items)}")
    if n == 1:
        return [items[0]]

    idx = [round(i * (len(items) - 1) / (n - 1)) for i in range(n)]
    out = [items[i] for i in idx]

    # Guard against duplicate indices from rounding.
    if len(set(out)) != len(out):
        out = []
        for item in items:
            out.append(item)
            if len(out) == n:
                break

    return out


def _parse_cycles(cycles_str: str) -> list[str]:
    return [c.strip().zfill(2) for c in cycles_str.split(",") if c.strip()]


def _parse_fhrs(fhrs_str: str) -> list[int]:
    return [int(x) for x in fhrs_str.split(",") if x.strip()]


def _available_dates(root: Path, year: int, start_date: str, end_date: str) -> list[str]:
    return sorted(
        p.name
        for p in root.iterdir()
        if (
            p.is_dir()
            and p.name.isdigit()
            and p.name.startswith(str(year))
            and start_date <= p.name <= end_date
        )
    )


def _make_paired_cycles(
    root: Path,
    dates: list[str],
    cycles: list[str],
    required_fhrs: list[int],
    num_dates: int,
) -> list[str]:
    usable_dates = [
        ymd
        for ymd in dates
        if all(_has_required_files(root, ymd, hh, required_fhrs) for hh in cycles)
    ]

    selected_dates = _evenly_select(usable_dates, num_dates)
    return [f"{ymd}{hh}" for ymd in selected_dates for hh in cycles]


def _make_balanced_distinct(
    root: Path,
    dates: list[str],
    cycles: list[str],
    required_fhrs: list[int],
    num_inits: int,
) -> list[str]:
    if num_inits <= 0:
        return []

    ncycles = len(cycles)
    base = num_inits // ncycles
    remainder = num_inits % ncycles

    per_cycle_counts = {
        hh: base + (1 if i < remainder else 0)
        for i, hh in enumerate(cycles)
    }

    selected_inits: list[str] = []
    used_dates: set[str] = set()

    for hh in cycles:
        valid_dates_for_cycle = [
            ymd
            for ymd in dates
            if _has_required_files(root, ymd, hh, required_fhrs)
        ]

        # Prefer dates not already used by another cycle, so 00Z and 12Z
        # come from different calendar days when possible.
        unused_dates = [ymd for ymd in valid_dates_for_cycle if ymd not in used_dates]

        need = per_cycle_counts[hh]
        if len(unused_dates) >= need:
            chosen_dates = _evenly_select(unused_dates, need)
        else:
            # Fallback: allow same-date reuse only if necessary.
            chosen_dates = _evenly_select(valid_dates_for_cycle, need)

        used_dates.update(chosen_dates)
        selected_inits.extend(f"{ymd}{hh}" for ymd in chosen_dates)

    return sorted(selected_inits)


def _make_all_inits(
    root: Path,
    dates: list[str],
    cycles: list[str],
    required_fhrs: list[int],
) -> list[str]:
    return [
        f"{ymd}{hh}"
        for ymd in dates
        for hh in cycles
        if _has_required_files(root, ymd, hh, required_fhrs)
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gfs-root",
        default="/scratch3/NCEPDEV/da/Mu-Chieh.Ko/JEDI-nudging/gfs-rt25",
    )
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument(
        "--start-date",
        default="20250108",
        help="Earliest initialization date to consider, YYYYMMDD.",
    )
    ap.add_argument(
        "--end-date",
        default="20251231",
        help="Latest initialization date to consider, YYYYMMDD.",
    )

    # Backward-compatible sampled options.
    ap.add_argument("--num-dates", type=int, default=None)
    ap.add_argument("--num-inits", type=int, default=None)
    ap.add_argument(
        "--all-inits",
        action="store_true",
        help="Emit every available date-cycle pair in the requested date window.",
    )

    ap.add_argument("--cycles", default="00,12")
    ap.add_argument("--required-fhrs", default="0,1,2,3,4,5,6,7,8,9,10,11,12")
    ap.add_argument(
        "--sample-mode",
        choices=["paired_cycles", "balanced_distinct"],
        default="paired_cycles",
    )
    ap.add_argument(
        "--format",
        choices=["plain", "bash"],
        default="plain",
        help="plain: one init per line; bash: INIT_TIMES=(...)",
    )

    args = ap.parse_args()

    root = Path(args.gfs_root)
    cycles = _parse_cycles(args.cycles)
    required_fhrs = _parse_fhrs(args.required_fhrs)

    dates = _available_dates(
        root=root,
        year=args.year,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    if args.all_inits:
        inits = _make_all_inits(
            root=root,
            dates=dates,
            cycles=cycles,
            required_fhrs=required_fhrs,
        )

    elif args.sample_mode == "paired_cycles":
        if args.num_dates is not None:
            num_dates = args.num_dates
        elif args.num_inits is not None:
            if args.num_inits % len(cycles) != 0:
                raise ValueError(
                    f"paired_cycles requires num_inits divisible by number of cycles. "
                    f"Got num_inits={args.num_inits}, cycles={cycles}"
                )
            num_dates = args.num_inits // len(cycles)
        else:
            num_dates = 60

        inits = _make_paired_cycles(
            root=root,
            dates=dates,
            cycles=cycles,
            required_fhrs=required_fhrs,
            num_dates=num_dates,
        )

    elif args.sample_mode == "balanced_distinct":
        if args.num_inits is None:
            raise ValueError("balanced_distinct requires --num-inits")

        inits = _make_balanced_distinct(
            root=root,
            dates=dates,
            cycles=cycles,
            required_fhrs=required_fhrs,
            num_inits=args.num_inits,
        )

    else:
        raise ValueError(f"Unknown sample mode: {args.sample_mode}")

    if args.format == "bash":
        print("INIT_TIMES=(")
        for init in inits:
            print(f"  {init}")
        print(")")
    else:
        print("\n".join(inits))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
