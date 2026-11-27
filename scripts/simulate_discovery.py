from pathlib import Path
import json
# Non-secret fixtures only; disposable CI checkout, never published.
Path("_data/discovery.json").write_text(json.dumps({
    "goatcounter_site":"ci-fixture-only",
    "google_verification":"ci-fixture-google",
    "bing_verification":"ci-fixture-bing",
    "indexnow_key":"ci-indexnow-key-1234567890"
}), encoding="utf-8")
Path("ci-indexnow-key-1234567890.txt").write_bytes(b"ci-indexnow-key-1234567890")
