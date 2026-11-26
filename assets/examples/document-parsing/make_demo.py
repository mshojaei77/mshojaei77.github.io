"""Generate a small, entirely synthetic three-page PDF."""
from pathlib import Path
import sys
import pymupdf


def make_demo(path):
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=300)
        page.insert_text((40, 50), "Project guide", fontsize=18)
        page.insert_text((40, 80), "Use the latest manual.", fontsize=14)

        page = doc.new_page(width=400, height=300)
        # 3 rows, 2 columns, with drawn grid lines.
        xs, ys = [40, 200, 360], [50, 85, 120, 155]
        for x in xs:
            page.draw_line((x, ys[0]), (x, ys[-1]))
        for y in ys:
            page.draw_line((xs[0], y), (xs[-1], y))
        for i, row in enumerate([("Item", "Price"), ("Notebook", "12.50"),
                                 ("Pen", "2.00")]):
            for j, value in enumerate(row):
                page.insert_text((xs[j] + 8, ys[i] + 22), value, fontsize=14)

        with pymupdf.open() as source:
            p = source.new_page(width=400, height=300)
            p.insert_text((40, 80), "Scan example", fontsize=22)
            p.insert_text((40, 120), "Reference code: AB123", fontsize=18)
            pix = p.get_pixmap(dpi=200, alpha=False)
            page = doc.new_page(width=400, height=300)
            page.insert_image(page.rect, pixmap=pix)
        doc.save(str(path))


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "demo.pdf")
    if target.exists():
        raise SystemExit("Choose a new output file")
    make_demo(target)
    print(target)
