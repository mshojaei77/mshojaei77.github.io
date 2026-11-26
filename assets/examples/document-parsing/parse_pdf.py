"""Teaching baseline: inspect PDF pages, keep evidence, and route text/OCR.

Not a complete layout parser. All output is unverified until reviewed.
Use only on local, trusted inputs; see the tutorial for production safeguards.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pymupdf

SCHEMA_VERSION = "1.0"


def word_records(page, textpage=None):
    words = page.get_text("words", textpage=textpage, sort=True)
    return [
        {"bbox": list(w[:4]), "text": w[4], "block": w[5],
         "line": w[6], "word": w[7]}
        for w in words
    ]


def parse_page(page, *, ocr=False, lang="eng", dpi=300,
               force_ocr=False, tessdata=None):
    native = page.get_text("text", sort=False)
    images = page.get_image_info()
    result = {
        "page": page.number + 1,
        "size_pt": [page.cropbox.width, page.cropbox.height],
        "rotation": page.rotation,
        "native_text": native,
        "text": "",
        "words": [],
        "tables": [],
        "image_regions": [list(i["bbox"]) for i in images],
        "method": None,
        "status": "candidate",
        "warnings": [],
    }
    warnings = result["warnings"]
    if page.rotation:
        warnings.append("rotated_page_check_coordinate_mapping")
    # Illustrative routing rule, not a calibrated classifier.
    suspicious = not native.strip() or "\ufffd" in native
    if suspicious or force_ocr:
        result["status"] = "review_required"
        if not ocr:
            warnings.append("ocr_or_blank_page_review_required")
            return result
        kwargs = {"language": lang, "dpi": dpi, "full": True}
        if tessdata:
            kwargs["tessdata"] = tessdata
        tp = page.get_textpage_ocr(**kwargs)
        result["method"] = "ocr"
        result["text"] = page.get_text("text", textpage=tp, sort=True)
        result["words"] = word_records(page, tp)
        warnings.append("ocr_candidate_check_against_page_image")
        if not result["text"].strip():
            warnings.append("ocr_empty_or_blank_page")
        # OCR text is not a reconstructed table. Do not invent cells.
        warnings.append("scanned_table_structure_not_extracted")
    else:
        result["method"] = "native"
        result["text"] = page.get_text("text", sort=True)
        result["words"] = word_records(page)
        if images:
            warnings.append("images_may_contain_unread_text")
        # Simple digital tables only. A missing detection is not proof
        # of absence. Copy the results while the Page is alive.
        for table in page.find_tables().tables:
            cells = []
            for row_index, (values, row) in enumerate(
                    zip(table.extract(), table.rows)):
                for col_index, (value, bbox) in enumerate(
                        zip(values, row.cells)):
                    cells.append({
                        "row": row_index, "col": col_index,
                        "text": value,  # None is not an empty string
                        "bbox": list(bbox) if bbox is not None else None,
                    })
            result["tables"].append({
                "bbox": list(table.bbox),
                "row_count": table.row_count,
                "col_count": table.col_count,
                "cells": cells,
            })
        if result["tables"]:
            warnings.append("verify_table_cells_and_headers")
        if warnings:
            result["status"] = "review_required"
    return result


def parse_pdf(path, *, ocr=False, lang="eng", dpi=300,
              force_pages=(), tessdata=None):
    path = Path(path)
    if path.stat().st_size > 50 * 1024 * 1024:
        raise ValueError("Demo limit: 50 MiB")
    data = path.read_bytes()
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        if doc.needs_pass:
            raise ValueError("Encrypted PDF: provide an authorized decrypted copy")
        if not doc.is_pdf or not 1 <= len(doc) <= 100:
            raise ValueError("Demo accepts PDFs with 1 to 100 pages")
        if not set(force_pages).issubset(range(1, len(doc) + 1)):
            raise ValueError("Forced OCR page is outside the document")
        if force_pages and not ocr:
            raise ValueError("--ocr-page requires --ocr")
        pages = []
        for page in doc:
            try:
                pages.append(parse_page(
                    page, ocr=ocr, lang=lang, dpi=dpi,
                    force_ocr=page.number + 1 in force_pages,
                    tessdata=tessdata,
                ))
            except (RuntimeError, ValueError, OSError) as exc:
                # A failed page must remain visible in the output.
                pages.append({"page": page.number + 1, "status": "failed",
                              "error_type": type(exc).__name__})
        return {
            "schema_version": SCHEMA_VERSION,
            "source_sha256": hashlib.sha256(data).hexdigest(),
            "engine": "pymupdf",
            "engine_version": pymupdf.VersionBind,
            "config": {"ocr": ocr, "language": lang, "dpi": dpi,
                       "force_pages": list(force_pages)},
            "coordinate_system": "unrotated_page_top_left_points",
            "page_count": len(doc),
            "pages": pages,
            "status": "unverified",
        }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("input", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--ocr", action="store_true")
    cli.add_argument("--lang", default="eng")
    cli.add_argument("--dpi", type=int, default=300, choices=[150, 200, 300, 400])
    cli.add_argument("--ocr-page", type=int, action="append", default=[])
    cli.add_argument("--tessdata")
    args = cli.parse_args()
    if args.input.resolve() == args.output.resolve():
        cli.error("Input and output must be different files")
    if args.output.exists():
        cli.error("Output already exists; choose a fresh path")
    report = parse_pdf(args.input, ocr=args.ocr, lang=args.lang,
                       dpi=args.dpi, force_pages=args.ocr_page,
                       tessdata=args.tessdata)
    # Exclusive create prevents accidental overwriting of prior evidence.
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    for page in report["pages"]:
        print(page["page"], page.get("method"), page["status"])
    return 2 if any(p["status"] == "failed" for p in report["pages"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
