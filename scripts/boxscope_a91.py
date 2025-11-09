#!/usr/bin/env python3
"""Utility for cataloguing items in box A91 using the OpenAI API.

The original one-shot shell script relied on ad-hoc dependency installation and
on the third party ``openai`` package.  In restricted environments (including
this kata's CI) that approach fails because ``pip`` cannot reach the Python
Package Index.  This module provides a self-contained Python implementation that
uses only the standard library plus the publicly available OpenAI HTTP API.

The script keeps the behaviour of the historical tooling while staying easy to
extend.  When executed as a CLI it reads configuration from environment
variables (the same names as the shell script used) and produces JSON/CSV/HTML
summaries in the output directory.  The heavy lifting – downloading image
references, sending them to the multimodal model and aggregating the results –
now happens in Python functions that can be unit-tested.
"""
from __future__ import annotations

import argparse
import base64
import csv
import dataclasses
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from html.parser import HTMLParser
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants & configuration helpers
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.environ.get("MODEL", "gpt-4o")
DEFAULT_MAX_OBJECTS = int(os.environ.get("MAX", "80"))
DEFAULT_LIMIT = int(os.environ.get("LIMIT", "0"))
ALLOW_OCR = os.environ.get("ALLOW_OCR", "1") not in {"0", "false", "False"}
EXPORT_CSV = os.environ.get("EXPORT_CSV", "1") not in {"0", "false", "False"}
EXPORT_HTML = os.environ.get("EXPORT_HTML", "1") not in {"0", "false", "False"}

# Fallback regular expression used to filter container-like items.
DEFAULT_IGNORE_REGEX = os.environ.get(
    "IGNORE_REGEX",
    r"\b("
    r"box|cardboard box|plastic box|storage box|transparent box|clear box|bin|"
    r"plastic bin|storage bin|crate|tote|container|organizer( tray)?|tray|compartment|"
    r"divider|insert|lid|cover|shelf|drawer|basket|pouch|bag|ziploc|zip[- ]?bag|"
    r"plastic bag|bubble wrap|wrapping|packing foam|foam|padding|filler|label|barcode|"
    r"qr code|background|surface|floor|table|desk|workbench|paper( sheet)?|"
    r"boks|plastboks|eske|pappeske|kasse|plastkasse|beholder|oppbevaringsboks|"
    r"skuff|hylle|lokk|pose|zip[- ]?pose|plastpose|bakgrunn|flate|bord|"
    r"pudełko|plastikowe pudełko|przezroczyste pudełko|pokrywka|tacka|wkład|"
    r"pojemnik|organizer|przegroda|woreczek|worek|folia bąbelkowa|pianka|wypełnienie"
    r")\b"
)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MIME_BY_EXTENSION = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
OPENAI_TIMEOUT = float(os.environ.get("OPENAI_TIMEOUT", "60"))

JSON_EXAMPLE = {
    "objects": [
        {
            "en": "analog alarm clock",
            "no": "analog vekkerklokke",
            "quantity": 1,
            "color": ["silver", "black"],
            "material": ["metal", "plastic"],
            "shape": "round",
            "approx_size_cm": {"l": 12, "w": 5, "h": 12},
            "condition": "used",
            "description_en": "Small round silver-and-black analog alarm clock (~12 cm diameter) with twin bells.",
            "description_no": "Liten rund sølv- og svart analog vekkerklokke (~12 cm i diameter) med to bjeller.",
            "markings": ["ALARM", "12h"],
            "confidence": 0.92,
        }
    ]
}

SCHEMA_EXAMPLE_TEXT = json.dumps(JSON_EXAMPLE, ensure_ascii=False)

SELF_TEST_IMAGE = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z/CfAQAD"
    "9gJ/14ym8QAAAABJRU5ErkJggg=="
)

# ---------------------------------------------------------------------------
# Utility data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ImageInfo:
    """Container for basic information about a single image."""

    api_url: str
    reference: Optional[str] = None

    @property
    def name(self) -> str:
        if self.reference:
            return self.reference
        parsed = urllib.parse.urlparse(self.api_url)
        return os.path.basename(parsed.path)


@dataclasses.dataclass
class ObjectRecord:
    """Representation of a single object returned by the model."""

    en: str
    no: str
    quantity: int = 1
    color: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    material: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    shape: Optional[str] = None
    approx_size_cm: Optional[dict] = None
    condition: Optional[str] = None
    description_en: str = ""
    description_no: str = ""
    markings: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    confidence: Optional[float] = None

    def normalized_key(self) -> Tuple[str, str, str, Tuple[str, ...]]:
        primary_mat = self.material[0] if self.material else "unknown"
        colours = tuple(sorted(self.color)[:2])
        return (self.en.lower(), (self.shape or "").lower(), primary_mat, colours)


# ---------------------------------------------------------------------------
# HTML parser for remote directory listings
# ---------------------------------------------------------------------------


class _ImageLinkParser(HTMLParser):
    """Collects links to image files from a directory listing."""

    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]):
        if tag.lower() != "a":
            return
        href = None
        for key, value in attrs:
            if key.lower() == "href":
                href = value
                break
        if not href:
            return
        url = urllib.parse.urljoin(self.base_url, href)
        ext = os.path.splitext(urllib.parse.urlparse(url).path)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            self.links.append(url)


# ---------------------------------------------------------------------------
# API interaction helpers
# ---------------------------------------------------------------------------


def _api_request(path: str, payload: dict, api_key: str) -> dict:
    """Perform a POST request against the OpenAI API using standard libraries."""

    request = urllib.request.Request(
        urllib.parse.urljoin(OPENAI_BASE_URL, path),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(request, timeout=OPENAI_TIMEOUT) as response:
        body = response.read()
    return json.loads(body.decode("utf-8"))


def call_chat_completion(
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_url: str,
    api_key: str,
    max_retries: int = 3,
) -> dict:
    """Send a multimodal chat completion request and return the parsed JSON."""

    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            },
        ],
    }

    delay = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            response = _api_request("/v1/chat/completions", payload, api_key)
            content = response["choices"][0]["message"]["content"]
            if isinstance(content, list) and content:
                text = content[0].get("text", "")
            else:
                text = content or ""
            text = text.strip()
            return json.loads(text)
        except Exception as exc:  # pragma: no cover - network failure branch
            if attempt >= max_retries:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30)

    raise RuntimeError("Exhausted OpenAI retries")


# ---------------------------------------------------------------------------
# Core processing logic
# ---------------------------------------------------------------------------


def extract_box_id(url: str) -> str:
    path = urllib.parse.urlparse(url).path.rstrip("/")
    if not path:
        return "BOX"
    return os.path.basename(path).upper() or "BOX"


def gather_image_urls(base_url: str) -> List[ImageInfo]:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme in {"http", "https"}:
        ext = os.path.splitext(parsed.path)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            name = os.path.basename(parsed.path)
            return [ImageInfo(api_url=base_url, reference=name or None)]

        with urllib.request.urlopen(base_url, timeout=OPENAI_TIMEOUT) as response:
            html_text = response.read().decode("utf-8", errors="replace")
        parser = _ImageLinkParser(base_url)
        parser.feed(html_text)
        seen = set()
        images: List[ImageInfo] = []
        for url in parser.links:
            if url not in seen:
                seen.add(url)
                ref = os.path.basename(urllib.parse.urlparse(url).path) or None
                images.append(ImageInfo(api_url=url, reference=ref))
        return images

    local_path = Path(base_url).expanduser()
    if local_path.is_file():
        return [_image_from_local_path(local_path)]
    if local_path.is_dir():
        images = []
        for child in sorted(local_path.iterdir()):
            if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                images.append(_image_from_local_path(child))
        return images

    raise RuntimeError(f"Unsupported input location: {base_url}")


def _image_from_local_path(path: Path) -> ImageInfo:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {path}")
    mime = MIME_BY_EXTENSION.get(ext, "application/octet-stream")
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    data_url = f"data:{mime};base64,{encoded}"
    return ImageInfo(api_url=data_url, reference=path.name)


def _ascii_escape(text: str) -> str:
    r"""Return *text* with non-ASCII characters escaped for transport.

    Some execution environments configure ``PYTHONIOENCODING`` (or the locale)
    to ``latin-1``/``ISO-8859-1``.  When those environments forward the model
    prompt to the OpenAI API using :mod:`urllib`, any non-ASCII characters in
    the payload may trigger ``UnicodeEncodeError`` before the request is sent.

    The helper mirrors JSON's escaping rules so that the downstream model still
    receives the full expression (with ``\uXXXX`` escapes) while keeping the
    transport payload strictly ASCII.
    """

    # ``json.dumps`` with ``ensure_ascii=True`` mirrors the escaping behaviour
    # used by the API.  Stripping the surrounding quotes yields an ASCII-only
    # representation suitable for embedding in human-readable strings.
    return json.dumps(text, ensure_ascii=True)[1:-1]


def build_system_prompt(ignore_regex: str) -> str:
    safe_regex = _ascii_escape(ignore_regex)
    return (
        "You are a precise vision tagging assistant. Return ONLY valid JSON (no prose). "
        "All images/tiles/rotations belong to ONE container/box; catalog its content. "
        "List concrete, visible, physical objects (no people identification, no guessing hidden parts). "
        "Treat orientation as arbitrary (objects/text may be upside-down or rotated). "
        "Be exhaustive and include small items. "
        "CRITICAL: EXCLUDE any container or environment items (box/bin/tote/crate/container/organizer/tray/divider/"
        "insert/lid/cover/shelf/drawer/bag/ziploc/wrap/foam/padding/label/barcode/QR/background/surface/floor/table/desk/workbench/paper). "
        "Ignore items matching this regex (JSON escaped): "
        f"{safe_regex}"
    )


def build_user_prompt(max_objects: int) -> str:
    extra = (
        "Allow 'markings' as up to 4 literal tokens of visible text if clearly legible."
        if ALLOW_OCR
        else "Do NOT include any text from the image."
    )
    return (
        "You catalog items for a single box. "
        f"List up to {max_objects} distinct visible objects. "
        "Respond with ONLY JSON (no markdown) following the schema. "
        "Be orientation-invariant (objects/text may be upside-down or rotated). "
        "Be exhaustive; include small items. "
        "DO NOT include container/environment items. "
        f"{extra} "
        f"Schema example: {SCHEMA_EXAMPLE_TEXT}"
    )


def normalize_list(value: Optional[Iterable[str]]) -> Tuple[str, ...]:
    if not value:
        return tuple()
    return tuple(sorted({str(item).strip().lower() for item in value if str(item).strip()}))


def should_ignore(record: ObjectRecord, pattern: re.Pattern[str]) -> bool:
    haystack = " ".join(
        filter(
            None,
            [
                record.en,
                record.no,
                record.description_en,
                record.description_no,
                *record.markings,
            ],
        )
    ).lower()
    return bool(pattern.search(haystack))

def parse_objects(raw: dict, pattern: re.Pattern[str]) -> List[ObjectRecord]:
    objects_raw = raw.get("objects", [])
    parsed: List[ObjectRecord] = []
    for entry in objects_raw:
        if not isinstance(entry, dict):
            continue
        en = str(entry.get("en", "")).strip()
        no = str(entry.get("no", "")).strip()
        if not en or not no:
            continue
        record = ObjectRecord(
            en=en,
            no=no,
            quantity=int(entry.get("quantity", 1) or 1),
            color=normalize_list(entry.get("color")),
            material=normalize_list(entry.get("material")),
            shape=(str(entry.get("shape", "")).strip().lower() or None),
            approx_size_cm=entry.get("approx_size_cm") or None,
            condition=(str(entry.get("condition", "")).strip().lower() or None),
            description_en=str(entry.get("description_en", "")).strip(),
            description_no=str(entry.get("description_no", "")).strip(),
            markings=normalize_list(entry.get("markings")),
            confidence=(
                float(entry["confidence"])
                if entry.get("confidence") not in (None, "")
                else None
            ),
        )
        if not should_ignore(record, pattern):
            parsed.append(record)
    return parsed


def aggregate_objects(
    per_image_records: Dict[str, List[ObjectRecord]]
) -> List[dict]:
    aggregates: Dict[Tuple[str, str, str, Tuple[str, ...]], dict] = {}

    for image_name, records in per_image_records.items():
        per_key_counts: Dict[Tuple[str, str, str, Tuple[str, ...]], int] = defaultdict(int)
        per_key_payload: Dict[Tuple[str, str, str, Tuple[str, ...]], ObjectRecord] = {}
        for record in records:
            key = record.normalized_key()
            per_key_counts[key] += max(record.quantity, 1)
            per_key_payload[key] = record

        for key, quantity in per_key_counts.items():
            record = per_key_payload[key]
            agg = aggregates.get(key)
            if not agg:
                agg = {
                    "en": record.en,
                    "no": record.no,
                    "shape": record.shape,
                    "colors": set(record.color),
                    "materials": set(record.material),
                    "approx_size_cm": record.approx_size_cm,
                    "condition": record.condition or "unknown",
                    "images": set(),
                    "quantity_per_image": {},
                    "markings": set(record.markings),
                    "description_en": record.description_en,
                    "description_no": record.description_no,
                    "confidence_values": [],
                }
                aggregates[key] = agg
            agg["images"].add(image_name)
            agg["quantity_per_image"][image_name] = max(
                agg["quantity_per_image"].get(image_name, 0), quantity
            )
            agg["colors"].update(record.color)
            agg["materials"].update(record.material)
            agg["markings"].update(record.markings)
            if record.description_en and (
                len(record.description_en) > len(agg["description_en"])
            ):
                agg["description_en"] = record.description_en
            if record.description_no and (
                len(record.description_no) > len(agg["description_no"])
            ):
                agg["description_no"] = record.description_no
            if record.approx_size_cm and not agg["approx_size_cm"]:
                agg["approx_size_cm"] = record.approx_size_cm
            if record.condition and agg["condition"] in {None, "unknown"}:
                agg["condition"] = record.condition
            if record.confidence is not None:
                agg["confidence_values"].append(record.confidence)

    summary: List[dict] = []
    for agg in aggregates.values():
        counts = list(agg["quantity_per_image"].values())
        confidence_avg = (
            sum(agg["confidence_values"]) / len(agg["confidence_values"])
            if agg["confidence_values"]
            else None
        )
        summary.append(
            {
                "en": agg["en"],
                "no": agg["no"],
                "shape": agg["shape"],
                "colors": sorted(agg["colors"]),
                "materials": sorted(agg["materials"]),
                "approx_size_cm": agg["approx_size_cm"],
                "condition": agg["condition"],
                "images": sorted(agg["images"]),
                "quantity_estimate": max(counts) if counts else 0,
                "quantity_sum_upper_bound": sum(counts),
                "confidence_avg": confidence_avg,
                "markings": sorted(agg["markings"]),
                "description_en": agg["description_en"],
                "description_no": agg["description_no"],
            }
        )
    return summary


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_csv(path: str, objects: Sequence[dict]) -> None:
    headers = [
        "en",
        "no",
        "shape",
        "colors",
        "materials",
        "size_cm",
        "condition",
        "quantity_estimate",
        "quantity_sum_upper_bound",
        "confidence_avg",
        "markings",
        "images",
        "description_en",
        "description_no",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for obj in objects:
            writer.writerow(
                {
                    "en": obj.get("en", ""),
                    "no": obj.get("no", ""),
                    "shape": obj.get("shape", ""),
                    "colors": ";".join(obj.get("colors", [])),
                    "materials": ";".join(obj.get("materials", [])),
                    "size_cm": json.dumps(obj.get("approx_size_cm"), ensure_ascii=False),
                    "condition": obj.get("condition", ""),
                    "quantity_estimate": obj.get("quantity_estimate", 0),
                    "quantity_sum_upper_bound": obj.get("quantity_sum_upper_bound", 0),
                    "confidence_avg": (
                        f"{obj['confidence_avg']:.4f}"
                        if isinstance(obj.get("confidence_avg"), (float, int))
                        else ""
                    ),
                    "markings": ";".join(obj.get("markings", [])),
                    "images": ";".join(obj.get("images", [])),
                    "description_en": obj.get("description_en", ""),
                    "description_no": obj.get("description_no", ""),
                }
            )


def write_html(path: str, summary: dict) -> None:
    data_js = json.dumps(summary, ensure_ascii=False)
    template_path = os.path.join(os.path.dirname(__file__), "templates", "box_template.html")
    with open(template_path, "r", encoding="utf-8") as handle:
        template = handle.read()
    html_doc = (
        template.replace("__BOX_ID__", html.escape(summary["box_id"]))
        .replace("__SRC_URL__", html.escape(summary["source_url"]))
        .replace("__DATA__", data_js)
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(html_doc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run(
    in_url: str,
    out_dir: str,
    api_key: str,
    ignore_regex: str,
    *,
    images_override: Optional[Sequence[ImageInfo]] = None,
    call_fn=call_chat_completion,
) -> dict:
    images = list(images_override) if images_override is not None else gather_image_urls(in_url)
    if DEFAULT_LIMIT > 0 and images_override is None:
        images = images[: DEFAULT_LIMIT]

    if not images:
        raise RuntimeError(f"No images found at {in_url}")

    box_id = extract_box_id(in_url)
    system_prompt = build_system_prompt(ignore_regex)
    user_prompt = build_user_prompt(DEFAULT_MAX_OBJECTS)
    pattern = re.compile(ignore_regex, flags=re.IGNORECASE)

    per_image_records: Dict[str, List[ObjectRecord]] = {}
    processed = 0
    errors: List[str] = []

    for image in images:
        try:
            result = call_fn(
                model=DEFAULT_MODEL,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_url=image.api_url,
                api_key=api_key,
            )
            per_image_records[image.name] = parse_objects(result, pattern)
            processed += 1
        except Exception as exc:  # pragma: no cover - network failure branch
            errors.append(f"{image.name}: {exc}")

    objects_summary = aggregate_objects(per_image_records)
    timestamp = int(time.time())
    summary = {
        "box_id": box_id,
        "source_url": in_url,
        "model": DEFAULT_MODEL,
        "generated_at": timestamp,
        "total_images": len(images),
        "processed_ok": processed,
        "errors": errors,
        "objects": objects_summary,
    }

    os.makedirs(out_dir, exist_ok=True)
    index_payload = {
        "box_id": box_id,
        "model": DEFAULT_MODEL,
        "total_images": len(images),
        "processed_ok": processed,
        "errors": len(errors),
        "generated_at": timestamp,
    }
    write_json(os.path.join(out_dir, "_index.json"), index_payload)
    write_json(os.path.join(out_dir, f"box.{box_id}.json"), summary)

    if EXPORT_CSV:
        write_csv(os.path.join(out_dir, f"box.{box_id}.csv"), objects_summary)
    if EXPORT_HTML:
        write_html(os.path.join(out_dir, f"box.{box_id}.html"), summary)

    return summary


def _self_test_call(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_url: str,
    api_key: str,
) -> dict:
    _ = (model, system_prompt, user_prompt, image_url, api_key)
    return {
        "objects": [
            {
                "en": "test cube",
                "no": "test kube",
                "quantity": 1,
                "color": ["blue"],
                "material": ["plastic"],
                "shape": "cube",
                "approx_size_cm": {"l": 5, "w": 5, "h": 5},
                "condition": "new",
                "description_en": "A tiny blue plastic cube used for self-testing the pipeline.",
                "description_no": "En liten blå plastkube for å teste rørledningen.",
                "markings": ["QA"],
                "confidence": 0.99,
            }
        ]
    }


def run_self_test(out_dir: str, ignore_regex: str) -> dict:
    images = [ImageInfo(api_url=SELF_TEST_IMAGE, reference="self-test.png")]
    return run(
        in_url="self-test://A91",
        out_dir=out_dir,
        api_key="sk-self-test",
        ignore_regex=ignore_regex,
        images_override=images,
        call_fn=_self_test_call,
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BoxScope cataloguing helper")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run an offline pipeline test that skips OpenAI API calls",
    )
    parser.add_argument(
        "--ignore-regex",
        default=DEFAULT_IGNORE_REGEX,
        help="Regular expression describing container/environment objects to skip",
    )
    parser.add_argument(
        "in_url",
        nargs="?",
        help="URL to a single image or a directory with images",
    )
    parser.add_argument(
        "out_dir",
        nargs="?",
        help="Destination directory for generated artefacts",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        if not args.in_url and not args.out_dir:
            parser.error("OUT_DIR is required when using --self-test")
    else:
        if not args.in_url or not args.out_dir:
            parser.error("in_url and out_dir are required unless --self-test is used")
    return args


def _safe_console_text(text: str) -> str:
    """Return *text* encoded safely for the current stdout encoding."""

    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
        return text
    except UnicodeEncodeError:
        return text.encode(encoding, errors="backslashreplace").decode(encoding, errors="ignore")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.self_test:
        out_dir = args.out_dir or args.in_url
        summary = run_self_test(out_dir, args.ignore_regex)
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit(
                "OPENAI_API_KEY is required. Export it before running the script."
            )

        summary = run(
            in_url=args.in_url,
            out_dir=args.out_dir,
            api_key=api_key,
            ignore_regex=args.ignore_regex,
        )
    print(
        _safe_console_text(
            f"Processed {summary['processed_ok']} of {summary['total_images']} images. Errors: {len(summary['errors'])}."
        )
    )
    if summary["errors"]:
        for error in summary["errors"]:
            print(_safe_console_text(f" - {error}"))
    destination = args.out_dir or args.in_url
    print(_safe_console_text(f"Results written to {destination}"))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution path
    raise SystemExit(main())
