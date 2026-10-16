"""Make the printable ArUco marker sheet (step 1 of the pipeline).

Writes an A4 landscape PDF with four 5 cm markers near the corners (layout in calib.PageLayout).
Print it at 100% / "actual size", then measure one marker with a ruler and pass that number to
track.py as --marker-cm (printers often shrink pages slightly; calib.PageLayout.scaled corrects for it).

Run:
    python scripts/make_markers.py --out aruco_a4.pdf
"""
import argparse

from PIL import Image

from hand2panda.calib import PageLayout, make_marker_page


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="aruco_a4.pdf", help="PDF file to write")
    p.add_argument("--dpi", type=int, default=300, help="print resolution, dots per inch")
    a = p.parse_args()

    page = make_marker_page(PageLayout(), dpi=a.dpi)
    # saving with the same dpi fixes the physical page size at exactly A4
    Image.fromarray(page).save(a.out, resolution=a.dpi)
    print("saved", a.out, "- print at 100% and measure one marker")


if __name__ == "__main__":
    main()
