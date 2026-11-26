from pathlib import Path
import sys
import xml.etree.ElementTree as ET
root=Path(sys.argv[1])
archive=(root/"writing/index.html").read_text(encoding="utf-8")
earlier="/writing/ci-date-order-first/"
later="/writing/ci-date-order-last/"
assert archive.index(later)<archive.index(earlier), "Archive order must follow front-matter dates, not filenames"
for route,date in ((earlier,"2024-04-03"),(later,"2025-05-04")):
    page=(root/route.strip("/")/"index.html").read_text(encoding="utf-8")
    assert date in page, "Article must display its overridden date at the unchanged permalink"
    assert date in archive, "Archive must display its overridden date"
    tree=ET.parse(root/"feed.xml")
    ns={"a":"http://www.w3.org/2005/Atom"}
    entry=next(e for e in tree.findall("a:entry",ns) if any(l.get("href","").endswith(route) for l in e.findall("a:link",ns)))
    assert entry.findtext("a:published",namespaces=ns).startswith(date)
print("Verified front-matter date override, archive order, permanent routes and feed dates.")
