"""Command line interface: ``trustcv --help | about | examples``."""

import argparse
import importlib.util
import runpy
import shutil
import sys
from importlib import resources

from . import __version__

# (module, one-line description, optional modules the recipe needs)
EXAMPLES = [
    ("all_cv_methods_demo", "Tour of all 29 CV splitters (IID, grouped, temporal, spatial)", ()),
    ("data_leakage_detection_demo", "Detect patient, temporal and feature leakage automatically", ()),
    ("heart_disease_classification", "Heart disease model with patient-grouped CV and leakage checks", ()),
    ("heart_disease_prediction", "Compare CV strategies on heart disease prediction", ()),
    ("multisite_clinical_trial", "Grouped CV for patients nested in hospitals/sites", ()),
    ("icu_patient_monitoring", "Temporal CV for ICU deterioration time series", ()),
    ("framework_agnostic_demo", "One CV runner across sklearn, PyTorch, TensorFlow", ("torch", "tensorflow")),
]

URLS = {
    "Website": "https://ki-smile.github.io/trustcv/",
    "PyPI": "https://pypi.org/project/trustcv/",
    "Source": "https://github.com/ki-smile/trustcv",
    "Issues": "https://github.com/ki-smile/trustcv/issues",
}

CITATION = """\
@software{trustcv2025,
  title = {trustcv: Trustworthy Cross-Validation Toolkit},
  author = {Abtahi, Farhad and Karbalaie, Abdolamir},
  year = {2025},
  url = {https://github.com/ki-smile/trustcv}
}"""

QUICK_START = """\
quick start:
  trustcv examples              list the numbered usage recipes
  trustcv examples 1 --run      run recipe 1
  trustcv examples 2 --copy .   copy recipe 2 here to edit
  trustcv about                 project info, links, citation

python:
  from trustcv import TrustCVValidator
  from trustcv.splitters import StratifiedGroupKFold
"""


def _missing(deps):
    return [d for d in deps if importlib.util.find_spec(d) is None]


def _path(i):
    return resources.files("trustcv.examples") / f"{EXAMPLES[i - 1][0]}.py"


def cmd_about(_):
    print(f"trustcv {__version__}  Trustworthy Cross-Validation Toolkit")
    print("Framework-agnostic CV with leakage detection, clinical metrics and reporting.\n")
    for k, v in URLS.items():
        print(f"  {k:<8} {v}")
    print(f"\nCitation:\n{CITATION}")


def cmd_examples(a):
    if a.number is None:
        print(f"{'#':<3} {'recipe':<30} {'extra deps':<18} description")
        for i, (name, desc, deps) in enumerate(EXAMPLES, 1):
            print(f"{i:<3} {name:<30} {', '.join(deps) or '-':<18} {desc}")
        print("\nUse: trustcv examples N [--run | --copy DIR]  (no flag prints the source)")
        return 0
    if not 1 <= a.number <= len(EXAMPLES):
        print(f"error: choose a number from 1 to {len(EXAMPLES)}", file=sys.stderr)
        return 2
    name, _, deps = EXAMPLES[a.number - 1]
    src = _path(a.number)
    if a.copy:
        dest = shutil.copy(str(src), a.copy)
        print(f"copied to {dest}")
    elif a.run:
        # framework recipes degrade per-section, so warn rather than refuse
        if _missing(deps):
            print(f"note: optional deps not installed ({', '.join(_missing(deps))}); "
                  "those sections will be skipped or fail.", file=sys.stderr)
        runpy.run_path(str(src), run_name="__main__")
    else:
        print(src.read_text())
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="trustcv",
        description="trustcv: trustworthy cross-validation for medical and regulated ML.",
        epilog=QUICK_START,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"trustcv {__version__}")
    sub = p.add_subparsers(dest="cmd", metavar="{about,examples}")

    sub.add_parser("about", help="project info, links, citation",
                   description="Show project info, links and the citation.")
    e = sub.add_parser(
        "examples", help=f"{len(EXAMPLES)} numbered usage recipes",
        description=f"List the {len(EXAMPLES)} bundled usage recipes, or print, run or copy one.",
        epilog="examples:\n  trustcv examples\n  trustcv examples 1 --run\n"
               "  trustcv examples 3 --copy my_heart.py\n  trustcv examples 5 > multisite.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    e.add_argument("number", nargs="?", type=int, help="recipe number (omit to list)")
    g = e.add_mutually_exclusive_group()
    g.add_argument("--run", action="store_true", help="run the recipe")
    g.add_argument("--copy", metavar="DEST", help="copy the recipe to a file or directory")

    a = p.parse_args(argv)
    if a.cmd == "about":
        return cmd_about(a) or 0
    if a.cmd == "examples":
        return cmd_examples(a)
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
