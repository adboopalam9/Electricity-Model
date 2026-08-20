"""Run this to inspect the raw CAISO OASIS response."""
import io
import zipfile
import xml.etree.ElementTree as ET
import requests

url = "http://oasis.caiso.com/oasisapi/SingleZip"
params = {
    "queryname":     "SLD_LOAD",
    "startdatetime": "20240115T00:00-0000",
    "enddatetime":   "20240116T00:00-0000",
    "version":       1,
}

print("Fetching CAISO OASIS SLD_LOAD...")
resp = requests.get(url, params=params, timeout=120)
print(f"HTTP status : {resp.status_code}")
print(f"Content-Type: {resp.headers.get('Content-Type')}")
print(f"Body size   : {len(resp.content):,} bytes\n")

# Is it a ZIP?
try:
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    print(f"ZIP contents: {zf.namelist()}\n")
    xml_name = next((n for n in zf.namelist() if n.endswith(".xml")), None)
    if xml_name:
        raw = zf.read(xml_name)
        print("=== First 3000 chars of XML ===")
        print(raw.decode("utf-8", errors="replace")[:3000])
        print("\n=== Parsing XML ===")
        root = ET.fromstring(raw)
        print(f"Root tag: {root.tag}")
        print("\nFirst 30 element tags found via iter():")
        for i, elem in enumerate(root.iter()):
            if i >= 30:
                break
            short_tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            print(f"  {short_tag!r:30s}  text={elem.text!r:.60s}" if elem.text else f"  {short_tag!r}")
        # Try wildcard search
        rd_nodes = list(root.iter("{*}REPORT_DATA"))
        print(f"\n{{*}}REPORT_DATA nodes: {len(rd_nodes)}")
        if rd_nodes:
            print("First node children:")
            for child in rd_nodes[0]:
                print(f"  {child.tag.split('}')[-1]}: {child.text}")
    else:
        print("No .xml file found in ZIP. Files:", zf.namelist())
        for n in zf.namelist():
            print(f"\n--- {n} ---")
            print(zf.read(n)[:1000])
except zipfile.BadZipFile:
    print("Response is NOT a ZIP file. Raw content:")
    print(repr(resp.content[:1000]))
