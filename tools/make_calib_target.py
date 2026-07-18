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
MM_TO_PT = 72.0 / 25.4  # PDF user space is 1/72 inch

# Landscape page sizes (w, h) in mm. Board is centered so "print at 100%" fits with no rescale.
PAGE_SIZES = {"letter": (279.4, 215.9), "a4": (297.0, 210.0)}


def _layout(cols, rows, square_mm, page):
    """Center the board on the page; return (page_w, page_h, board_x, board_y, board_w, board_h)
    with board_x/board_y the top-left corner in top-down mm coords."""
    page_w, page_h = PAGE_SIZES[page]
    board_w = (cols + 1) * square_mm
    board_h = (rows + 1) * square_mm
    if board_w > page_w or board_h + 12 > page_h:
        raise SystemExit(
            f"board {board_w:g}x{board_h:g} mm does not fit {page} landscape "
            f"({page_w:g}x{page_h:g} mm). Use a smaller --square-mm or fewer corners.")
    bx = (page_w - board_w) / 2
    by = (page_h - board_h) / 2
    return page_w, page_h, bx, by, board_w, board_h


def make_svg(cols: int, rows: int, square_mm: float, page: str = "letter") -> str:
    W, H, bx, by, board_w, board_h = _layout(cols, rows, square_mm, page)
    sq_cols, sq_rows = cols + 1, rows + 1

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}{MM}" height="{H}{MM}" '
        f'viewBox="0 0 {W} {H}">',
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="white"/>',
    ]
    # checkerboard: fill the "black" squares (top-left square is black by convention)
    for r in range(sq_rows):
        for c in range(sq_cols):
            if (r + c) % 2 == 0:
                x = bx + c * square_mm
                y = by + r * square_mm
                parts.append(
                    f'<rect x="{x:.3f}" y="{y:.3f}" width="{square_mm}" '
                    f'height="{square_mm}" fill="black"/>')
    # thin frame around the board so you can see the full extent / check for print clipping
    parts.append(
        f'<rect x="{bx:.3f}" y="{by:.3f}" width="{board_w:.3f}" '
        f'height="{board_h:.3f}" fill="none" stroke="black" stroke-width="0.2"/>')

    # 50 mm scale bar + label just below the board
    sy = by + board_h + 8.0
    parts.append(
        f'<line x1="{bx:.3f}" y1="{sy:.3f}" x2="{bx + 50:.3f}" y2="{sy:.3f}" '
        f'stroke="black" stroke-width="0.4"/>')
    for tick in (0, 50):
        parts.append(
            f'<line x1="{bx + tick:.3f}" y1="{sy - 1.5:.3f}" '
            f'x2="{bx + tick:.3f}" y2="{sy + 1.5:.3f}" stroke="black" stroke-width="0.4"/>')
    parts.append(
        f'<text x="{bx + 54:.3f}" y="{sy + 1.5:.3f}" font-family="sans-serif" '
        f'font-size="4">50 mm  |  {cols}x{rows} inner corners, {square_mm:g} mm squares  |  '
        f'print at 100% ({page})</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def make_pdf(cols: int, rows: int, square_mm: float, page: str = "letter") -> bytes:
    """Minimal single-page PDF, dependency-free. All geometry in points (1/72"), so the
    printed square size is exact as long as the printer does not rescale (print at 100%)."""
    W_mm, H_mm, bx_mm, by_mm, board_w, board_h = _layout(cols, rows, square_mm, page)
    W, H = W_mm * MM_TO_PT, H_mm * MM_TO_PT
    sq_pt = square_mm * MM_TO_PT
    sq_cols, sq_rows = cols + 1, rows + 1
    bx = bx_mm * MM_TO_PT
    # PDF origin is bottom-left; board top edge in PDF y:
    board_top = H - by_mm * MM_TO_PT

    ops = ["0 0 0 rg"]  # black fill
    for r in range(sq_rows):
        for c in range(sq_cols):
            if (r + c) % 2 == 0:
                x = bx + c * sq_pt
                y = board_top - (r + 1) * sq_pt      # top-down
                ops.append(f"{x:.3f} {y:.3f} {sq_pt:.3f} {sq_pt:.3f} re f")
    # board outline
    ops.append("0.2 w")
    ops.append(f"{bx:.3f} {board_top - board_h * MM_TO_PT:.3f} "
               f"{board_w * MM_TO_PT:.3f} {board_h * MM_TO_PT:.3f} re S")
    # 50 mm scale bar just below the board
    sy = board_top - (board_h + 8.0) * MM_TO_PT
    x0, x1 = bx, bx + 50 * MM_TO_PT
    ops.append("0.4 w")
    ops.append(f"{x0:.3f} {sy:.3f} m {x1:.3f} {sy:.3f} l S")
    for tx in (x0, x1):
        ops.append(f"{tx:.3f} {sy - 1.5 * MM_TO_PT:.3f} m {tx:.3f} {sy + 1.5 * MM_TO_PT:.3f} l S")
    label = (f"50 mm  |  {cols}x{rows} inner corners, {square_mm:g} mm squares  |  "
             f"print at 100% ({page})")
    ops.append(f"BT /F1 {4 * MM_TO_PT:.2f} Tf {x1 + 4 * MM_TO_PT:.3f} {sy - 1.4 * MM_TO_PT:.3f} Td "
               f"({label}) Tj ET")
    content = "\n".join(ops).encode("latin-1")

    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        (f"<</Type/Page/Parent 2 0 R/MediaBox[0 0 {W:.3f} {H:.3f}]"
         f"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>").encode("latin-1"),
        b"<</Length " + str(len(content)).encode() + b">>\nstream\n" + content + b"\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<</Size {len(objs) + 1}/Root 1 0 R>>\n"
            f"startxref\n{xref_pos}\n%%EOF\n").encode()
    return bytes(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cols", type=int, default=9, help="inner corners across (default 9)")
    ap.add_argument("--rows", type=int, default=6, help="inner corners down (default 6)")
    ap.add_argument("--square-mm", type=float, default=25.0, help="square size in mm (default 25)")
    ap.add_argument("--format", choices=("pdf", "svg", "both"), default="pdf",
                    help="output format (default pdf)")
    ap.add_argument("--page", choices=tuple(PAGE_SIZES), default="letter",
                    help="page size, landscape (default letter)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output path; extension overrides --format for a single file")
    args = ap.parse_args()

    stem = Path(__file__).parent / f"calib_checkerboard_{args.cols}x{args.rows}_{args.square_mm:g}mm"
    written = []
    if args.out is not None:
        fmt = args.out.suffix.lstrip(".").lower() or args.format
        if fmt == "svg":
            args.out.write_text(make_svg(args.cols, args.rows, args.square_mm, args.page))
        else:
            args.out.write_bytes(make_pdf(args.cols, args.rows, args.square_mm, args.page))
        written.append(args.out)
    else:
        if args.format in ("pdf", "both"):
            p = stem.with_suffix(".pdf")
            p.write_bytes(make_pdf(args.cols, args.rows, args.square_mm, args.page)); written.append(p)
        if args.format in ("svg", "both"):
            p = stem.with_suffix(".svg")
            p.write_text(make_svg(args.cols, args.rows, args.square_mm, args.page)); written.append(p)

    sq_w = (args.cols + 1) * args.square_mm
    sq_h = (args.rows + 1) * args.square_mm
    for p in written:
        print(f"wrote {p}")
    print(f"board: {args.cols+1}x{args.rows+1} squares = {sq_w:g}x{sq_h:g} mm "
          f"({sq_w/25.4:.1f}x{sq_h/25.4:.1f} in)")
    print("print at 100% (Actual size), landscape; verify the 50 mm bar with a ruler.")


if __name__ == "__main__":
    main()
