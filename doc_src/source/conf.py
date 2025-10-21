# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

import tomlkit

sphinx_source = Path(__file__).resolve().parent
sys.path.insert(0, sphinx_source.as_posix())
from sphinx_utils import clean_copied, copy_collections, get_poetry_version  # noqa

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, repo_root.as_posix())

with open(repo_root / "pyproject.toml", "r", encoding="utf-8") as f:
    pyproject = tomlkit.load(f)  # use tomlkit to load

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = pyproject["project"]["name"]
author = pyproject["project"].get("authors", [{"name": "William Muntean"}])[0]["name"]
copyright = f"2025, {author}"
release = get_poetry_version()
version = release
html_last_updated_fmt = ""

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "myst_parser",
    "nbsphinx",
    "sphinx_design",
]

# autodoc_mock_imports = ['packages']
autosummary_ignore_module_all = False
add_module_names = False
add_function_parentheses = False
toc_object_entries = False
autodoc_member_order = "bysource"
# numpydoc_class_members_toctree = False

templates_path = ["_templates"]
exclude_patterns = []

nbsphinx_execute = "never"
suppress_warnings = ["nbsphinx.orphan"]
myst_heading_anchors = 6

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_show_sphinx = False
html_css_files = [
    "wordwrap.css",
    "center_title.css",
]

html_theme_options = {
    "navigation_depth": 3,
    "github_url": pyproject["project"]["urls"]["repository"],
    "logo": {
        "text": project,
    },
    "navbar_start": ["navbar-logo", "version"],
    "navbar_end": ["search-field.html", "theme-switcher", "navbar-icon-links"],
    "navbar_persistent": [],
    "footer_start": ["copyright"],
    "footer_center": ["last-updated", "sphinx-version"],
    "collapse_navigation": True,
    "secondary_sidebar_items": {
        "**": ["page-toc"],
        "notebooks/*": [],
    },
}
html_sidebars = {
    "index_documentation": [],  # disables primary sidebar
}

# Define the relative collection paths
collections_config = {
    # "notebooks": {
    #     "source": f"{project}/notebooks",  # analysis, exploritory, etc
    #     "target": "notebooks",
    #     "ignore": ["*.py", "__pycache__"],
    # },
    # "assets": {
    #     "source": "assets",
    #     "target": "_static",
    #     "ignore": ["*.gitkeep"],
    # },
}

# Copy everything before Sphinx build
copy_collections(
    collections_config,
    source_base=repo_root,
    target_base=sphinx_source,
    verbose=True,
)


# Set callback
def on_build_finished(app, exception):
    """Clean up copied folders after the Sphinx build finishes."""
    clean_copied(verbose=True)


# Register the callback with Sphinx
def setup(app):
    app.connect("build-finished", on_build_finished)
