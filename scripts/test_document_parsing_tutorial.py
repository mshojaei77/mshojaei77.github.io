import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       "assets/examples/document-parsing"))
from make_demo import make_demo
from parse_pdf import parse_pdf


class TutorialTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "demo.pdf"
        make_demo(self.path)

    def test_native_table_and_scan_routing(self):
        result = parse_pdf(self.path)
        self.assertEqual(result["status"], "unverified")
        self.assertEqual(result["page_count"], 3)
        text, table, scan = result["pages"]
        self.assertIn("Project guide", text["text"])
        self.assertEqual(text["method"], "native")
        self.assertTrue(text["words"])
        self.assertEqual(len(result["source_sha256"]), 64)
        self.assertEqual(len(table["tables"]), 1)
        cells = {(c["row"], c["col"]): c["text"]
                 for c in table["tables"][0]["cells"]}
        self.assertEqual(cells[(1, 0)], "Notebook")
        self.assertEqual(cells[(1, 1)], "12.50")
        self.assertEqual(cells[(2, 1)], "2.00")
        self.assertEqual(scan["status"], "review_required")
        self.assertIsNone(scan["method"])
        self.assertTrue(scan["image_regions"])

    def test_ocr_candidate(self):
        result = parse_pdf(self.path, ocr=True,
                           tessdata=os.environ.get("TUTORIAL_TESSDATA"))
        scan = result["pages"][2]
        self.assertEqual(scan["method"], "ocr")
        self.assertIn("AB123", scan["text"])
        self.assertEqual(scan["status"], "review_required")
        self.assertEqual(scan["tables"], [])

    def test_force_requires_ocr(self):
        with self.assertRaises(ValueError):
            parse_pdf(self.path, force_pages=[1])


if __name__ == "__main__":
    unittest.main()
