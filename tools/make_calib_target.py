#!/usr/bin/env python3
"""
Generate a printable checkerboard calibration target as an SVG (real-world mm).

Defaults match tools/calibrate.py: 9x6 INNER corners (= 10x7 squares), 25 mm each.
OpenCV's board size is inner corners, so an NxM-corner board has (N+1)x(M+1) squares.

    python tools/make_calib_target.py                 # 9x6 corners, 25mm -> tools/calib_*.svg
    python tools/make_calib_target.py --cols 9 --rows 6 --square-mm 25

PRINTING (critical): print at 100% / "Actual size" — NOT "Fit to page", which rescales
and silently corrupts the square size the calibrator trusts. After printing, verify with
a ruler against the 50 mm scale bar. Mount the sheet on something rigid and flat.
"""
import argparse
from pathlib import Path

MM = "mm"


def make_svg(cols: int, rows: int, square_mm: float, margin_mm: float = 15.0) -> str:
    sq_cols, sq_rows = cols + 1, rows + 1          # squares from inner-corner count
    board_w, board_h = sq_cols * square_mm, sq_rows * square_mm
    W, H = board_w + 2 * margin_mm, board_h + 2 * margin_mm + 18.0  # extra for label/scale

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}{MM}" height="{H}{MM}" '
        f'viewBox="0 0 {W} {H}">',
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="white"/>',
    ]
    # checkerboard: fill the "black" squares (top-left square is black by convention)
    for r in range(sq_rows):
        for c in range(sq_cols):
            if (r + c) % 2 == 0:
                x = margin_mm + c * square_mm
                y = margin_mm + r * square_mm
                parts.append(
                    f'<rect x="{x:.3f}" y="{y:.3f}" width="{square_mm}" '
                    f'height="{square_mm}" fill="black"/>')
    # thin frame around the board so you can see the full extent / check for print clipping
    parts.append(
        f'<rect x="{margin_mm:.3f}" y="{margin_mm:.3f}" width="{board_w:.3f}" '
        f'height="{board_h:.3f}" fill="none" stroke="black" stroke-width="0.2"/>')

    # 50 mm scale bar + label along the bottom margin
    by = margin_mm + board_h + 8.0
    parts.append(
        f'<line x1="{margin_mm:.3f}" y1="{by:.3f}" x2="{margin_mm + 50:.3f}" y2="{by:.3f}" '
        f'stroke="black" stroke-width="0.4"/>')
    for tick in (0, 50):
        parts.append(
            f'<line x1="{margin_mm + tick:.3f}" y1="{by - 1.5:.3f}" '
            f'x2="{margin_mm + tick:.3f}" y2="{by + 1.5:.3f}" stroke="black" stroke-width="0.4"/>')
    parts.append(
        f'<text x="{margin_mm + 54:.3f}" y="{by + 1.5:.3f}" font-family="sans-serif" '
        f'font-size="4">50 mm  |  {cols}x{rows} inner corners, {square_mm:g} mm squares  |  '
        f'print at 100%</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cols", type=int, default=9, help="inner corners across (default 9)")
    ap.add_argument("--rows", type=int, default=6, help="inner corners down (default 6)")
    ap.add_argument("--square-mm", type=float, default=25.0, help="square size in mm (default 25)")
    ap.add_argument("--out", type=Path, default=None, help="output .svg path")
    args = ap.parse_args()

    out = args.out or (Path(__file__).parent /
                       f"calib_checkerboard_{args.cols}x{args.rows}_{args.square_mm:g}mm.svg")
    out.write_text(make_svg(args.cols, args.rows, args.square_mm))
    sq_w = (args.cols + 1) * args.square_mm
    sq_h = (args.rows + 1) * args.square_mm
    print(f"wrote {out}")
    print(f"board: {args.cols+1}x{args.rows+1} squares = {sq_w:g}x{sq_h:g} mm "
          f"({sq_w/25.4:.1f}x{sq_h/25.4:.1f} in)")
    print("print at 100% (Actual size), landscape; verify the 50 mm bar with a ruler.")


if __name__ == "__main__":
    main()
