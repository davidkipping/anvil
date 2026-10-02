"""The version is a contract with downstream packages, so it is tested.

anvil is consumed by packages that report `anvil.__version__` to their
users. Shipping an API change without bumping it once cost turin a
debugging detour (VERSIONS.md records it), so the discipline is checked
rather than remembered: the top row of VERSIONS.md must name the version
the package reports.
"""

import pathlib
import re

import anvil

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_versions_table_leads_with_the_current_version():
    rows = [ln for ln in (_ROOT / "VERSIONS.md").read_text().splitlines()
            if re.match(r"^\|\s*(`?\d|\d)", ln)]
    assert rows, "VERSIONS.md has no version rows"
    top = rows[0].split("|")[1].strip().strip("`")
    assert top == anvil.__version__, (
        f"VERSIONS.md leads with {top!r} but anvil.__version__ is "
        f"{anvil.__version__!r}: bump one, or add the missing row")


def test_pyproject_reads_the_version_from_the_package():
    """One source of truth: a hard-coded version in pyproject.toml is how
    the two drifted in the first place."""
    text = (_ROOT / "pyproject.toml").read_text()
    assert 'dynamic = ["version"]' in text
    assert "[tool.hatch.version]" in text
    assert not re.search(r'^version\s*=', text, re.M)
