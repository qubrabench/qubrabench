# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "qubrabench"
copyright = "2023, QuBRA Benchmarking Project"
author = "QuBRA Benchmarking Project"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.napoleon", "myst_parser"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["../img"]

# -- HTML navigation bar -----------------------------------------------------
html_theme_options = {
    "github_user": "qubrabench",
    "github_repo": "qubrabench",
    "description": "A framework to benchmark the advantage of quantum algorithms.",
    "github_banner": "forkme_right_darkblue_121621.png",
    "logo": "logo.png",
    "logo_name": True,
    "github_type": "star",
    "extra_nav_links": {
        "GitHub": "https://github.com/qubrabench/qubrabench/",
        "Issue Tracker": "https://github.com/qubrabench/qubrabench/issues",
    },
}
html_sidebars = {
    "**": [
        "about.html",
        "localtoc.html",
        "navigation.html",
        "searchbox.html",
    ]
}
