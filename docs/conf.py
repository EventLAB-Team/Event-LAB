import os
import sys

# Allows Sphinx autodoc to import your package from the project root.
sys.path.insert(0, os.path.abspath(".."))

project = "Event-LAB"
author = "Adam D Hines"
copyright = "2026, Adam D Hines"

release = "0.1.0"
version = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

html_static_path = ["_static"]

# Lets Read the Docs set the canonical URL.
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "/")
