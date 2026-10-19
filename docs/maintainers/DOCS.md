# Maintaining the documentation site

The site at https://ki-smile.github.io/trustcv/ is built with MkDocs Material from `docs/` (config: `mkdocs.yml`) and deployed by `.github/workflows/deploy-website.yml` on pushes to `main`.

## Local preview

```bash
pip install -e ".[docs]"
mkdocs serve            # http://127.0.0.1:8000/trustcv/
```

## What is generated, not hand-written

| Output | Source |
| --- | --- |
| API reference (`reference/api.md`) | docstrings, via mkdocstrings; the `:::` list in `api.md` is also the inventory |
| `api-schema.json` | `docs/hooks/schema.py` (static griffe extraction of the `api.md` inventory) |
| `llms.txt`, `llms-full.txt` | mkdocs-llmstxt, sections in `mkdocs.yml` |
| Changelog and license pages | included from `CHANGELOG.md` and `LICENSE` |

## Checks (CI `docs` job)

- `mkdocs build --strict`: any warning, including a docstring that contradicts its signature, fails the build.
- `tests/test_docs_snippets.py`: every ```python block in the docs runs (one subprocess per page). Mark illustrative blocks with `<!-- notest: reason -->` on the line before the fence.
- `tests/test_docs_build.py`: schema matches signatures, machine-readable files exist, every old `*.html` URL redirects.
- `tests/test_public_api.py`: every public name appears in the API reference; the home page states the real splitter count.
- Before a visual change: `python scripts/devtools/site_qa.py http://127.0.0.1:8000/trustcv` (overflow, console errors, axe WCAG 2.2 AA in light and dark).

## Dependencies and upgrade path

Pinned in the `docs` extra: mkdocs 1.6, mkdocs-material 9.x, mkdocstrings 1.x with mkdocstrings-python 2.x, mkdocs-redirects, mkdocs-llmstxt, and pymdown-extensions < 11 (12.x breaks mkdocstrings highlighting).

Material for MkDocs is in maintenance mode; its authors moved to Zensical (announced November 2025), and critical maintenance for Material ends in May 2027. Before then, check whether Zensical supports mkdocstrings, redirects and llms.txt generation, and migrate in a branch using the build tests above as the acceptance check. Do not assume plugins carry over unchanged.
