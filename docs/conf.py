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
