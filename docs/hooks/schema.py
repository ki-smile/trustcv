"""MkDocs hook: write api-schema.json from the documented API.

The inventory is the list of ``::: dotted.path`` directives in
docs/reference/api.md, so the reference page and the schema cannot disagree.
Everything is extracted statically with griffe: optional frameworks
(torch, tensorflow, ...) are never imported.
"""

import json
import re
from pathlib import Path

import griffe

API_PAGE = Path(__file__).resolve().parent.parent / "reference" / "api.md"


def _params(obj):
    out = []
    for p in obj.parameters:
        if p.name in ("self", "cls"):
            continue
        out.append(
            {
                "name": p.name,
                "kind": p.kind.value,
                "default": None if p.default is None else str(p.default),
                "annotation": None if p.annotation is None else str(p.annotation),
            }
        )
    return out


def _summary(obj):
    return obj.docstring.value.strip().splitlines()[0] if obj.docstring and obj.docstring.value.strip() else ""


def build_schema(repo_root):
    pkg = griffe.load("trustcv", search_paths=[str(repo_root)], allow_inspection=False)
    paths = re.findall(r"^::: (\S+)$", API_PAGE.read_text(encoding="utf-8"), re.M)
    version = pkg["__version__"].value.strip("\"'")
    statuses = [s.strip("\"'") for s in pkg["checks.OVERALL_STATUSES"].value.elements]
    items = []
    for path in paths:
        obj = pkg[path.removeprefix("trustcv.")]
        record = {"name": obj.name, "path": path, "kind": obj.kind.value, "summary": _summary(obj)}
        if obj.is_function:
            record["parameters"] = _params(obj)
        elif obj.is_class:
            init = obj.members.get("__init__")
            record["parameters"] = _params(init) if init is not None and init.is_function else []
            record["methods"] = [
                {"name": m.name, "parameters": _params(m), "summary": _summary(m)}
                for m in obj.members.values()
                if m.is_function and not m.name.startswith("_")
            ]
        items.append(record)
    return {
        "name": "trustcv",
        "version": version,
        "install": "pip install trustcv",
        "documentation": "https://ki-smile.github.io/trustcv/",
        "repository": "https://github.com/ki-smile/trustcv",
        "overall_status_values": statuses,
        "api": items,
    }


def on_post_build(config, **kwargs):
    repo_root = Path(config["config_file_path"]).resolve().parent
    schema = build_schema(repo_root)
    out = Path(config["site_dir"]) / "api-schema.json"
    out.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
