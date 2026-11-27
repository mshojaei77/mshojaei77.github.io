from pathlib import Path
import json
# Non-secret fixtures only; disposable CI checkout, never published.
Path("_data/discovery.json").write_text(json.dumps({
    "goatcounter_site":"ci-fixture-only",
    "google_verification":"ci-fixture-google",
    "bing_verification":"ci-fixture-bing"
}), encoding="utf-8")
