import json
import sys
from pathlib import Path

import pdfplumber
from pypdf import PdfReader


def plain(value):
    if value is None:
        return ""
    try:
        return str(value.get_object())
    except Exception:
        return str(value)


def rect_from_quadpoints(points):
    values = [float(x) for x in points]
    xs = values[0::2]
    ys = values[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def marked_text(page, annot):
    boxes = []
    if annot.get("/QuadPoints"):
        q = [float(x) for x in annot["/QuadPoints"]]
        boxes = [rect_from_quadpoints(q[i:i + 8]) for i in range(0, len(q), 8)]
    elif annot.get("/Rect"):
        r = [float(x) for x in annot["/Rect"]]
        boxes = [(min(r[0], r[2]), min(r[1], r[3]), max(r[0], r[2]), max(r[1], r[3]))]
    if not boxes:
        return ""

    words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
    picked = []
    seen = set()
    page_height = float(page.height)
    for box in boxes:
        x0, y0, x1, y1 = box
        top, bottom = page_height - y1, page_height - y0
        for idx, word in enumerate(words):
            wx0, wx1 = float(word["x0"]), float(word["x1"])
            wt, wb = float(word["top"]), float(word["bottom"])
            overlap_x = max(0.0, min(x1, wx1) - max(x0, wx0))
            overlap_y = max(0.0, min(bottom, wb) - max(top, wt))
            if overlap_x > 0 and overlap_y > 0 and idx not in seen:
                seen.add(idx)
                picked.append((wt, wx0, word["text"]))
    picked.sort(key=lambda item: (round(item[0], 1), item[1]))
    return " ".join(item[2] for item in picked).strip()


def main():
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    reader = PdfReader(str(source))
    records = []
    with pdfplumber.open(str(source)) as plumber:
        for page_index, pypdf_page in enumerate(reader.pages):
            for annot_ref in pypdf_page.get("/Annots", []):
                annot = annot_ref.get_object()
                subtype = plain(annot.get("/Subtype")).lstrip("/")
                if subtype in {"Link", "Popup", "Widget"}:
                    continue
                record = {
                    "page": page_index + 1,
                    "type": subtype,
                    "author": plain(annot.get("/T")),
                    "subject": plain(annot.get("/Subj")),
                    "comment": plain(annot.get("/Contents")),
                    "created": plain(annot.get("/CreationDate")),
                    "modified": plain(annot.get("/M")),
                    "marked_text": marked_text(plumber.pages[page_index], annot),
                    "rect": [float(x) for x in annot.get("/Rect", [])],
                }
                records.append(record)
    destination.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"annotations={len(records)}")
    for record in records:
        print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
