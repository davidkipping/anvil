"""Sphinx configuration for anvil.

Docs must build on machines without Apple Silicon (e.g. Read the Docs'
Linux builders), so MLX is mocked for autodoc: the package modules import
cleanly under the mock because no module-level code calls into MLX.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "anvil"
author = "David Kipping"
copyright = "2026, David Kipping"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

autodoc_mock_imports = ["mlx"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

myst_enable_extensions = ["colon_fence", "deflist"]

templates_path = []
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = "anvil"
html_baseurl = "https://anvil-mcmc.readthedocs.io/en/latest/"

# Version shown in the docs: read from the package source (MLX is mocked,
# so import anvil only for its version string, without executing the rest)
import re as _re

with open(os.path.join(os.path.dirname(__file__), "..", "anvil", "__init__.py")) as _f:
    release = version = _re.search(r'__version__ = "([^"]+)"', _f.read()).group(1)
html_title = f"anvil {release}"

# Strict references: a role that resolves to nothing is a warning, and the
# local build runs with -W, so a dead link fails it. Types from MLX/numpy
# and typing spellings in signatures are not anvil's to document.
nitpicky = True
nitpick_ignore_regex = [
    ("py:class", r"(mx|np|mlx|numpy)\..*"),
    ("py:class", r"(optional|callable|array|array-like|dict|tuple|list|float|int|bool|str)"),
    ("py:class", r"Literal\[.*\]|Callable.*|Any|ChainState|StepInfo|.*\.array"),
    ("py:class", r".*AdaptState|'?dict\[str.*|\"fp32_tree\"|\"fp64_anchor\"|\"fixed_point\""),
]


def _resolve_public_names(app, env, node, contnode):
    """Resolve ``anvil.X`` to where X is defined (``anvil.evidence.smc``):
    the guides and docstrings use the public names, autodoc registers each
    object under its defining module."""
    if node.get("refdomain") != "py":
        return None
    target = node.get("reftarget", "")
    if not target.startswith("anvil."):
        return None
    pkg = sys.modules.get("anvil")
    head, _, rest = target[len("anvil."):].partition(".")
    obj = getattr(pkg, head, None) if pkg is not None else None
    mod = getattr(obj, "__module__", None)
    if obj is None or mod is None or mod == "anvil":
        return None
    new = f"{mod}.{getattr(obj, '__qualname__', head)}" + (f".{rest}" if rest else "")
    if new == target:
        return None
    node["reftarget"] = new
    domain = env.get_domain("py")
    return domain.resolve_xref(env, node["refdoc"], app.builder, node["reftype"],
                               new, node, contnode)


def setup(app):
    app.connect("missing-reference", _resolve_public_names)
