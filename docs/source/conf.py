# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import time

sys.path.insert(0, os.path.abspath('../..'))
# sys.path.insert(1, os.path.abspath('../analysis'))
# sys.path.insert(2, os.path.abspath('../selection'))


# -- Project information -----------------------------------------------------

project = 'phis-scq'
author = 'Marcos Romero Lamas'
copyright = time.strftime('%Y, Marcos Romero Lamas')


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# extensions = [
#     'sphinx.ext.napoleon',
#     'sphinx.ext.autodoc',
#     'sphinx.ext.viewcode',
#     'sphinx.ext.coverage',
#     'sphinx_automodapi.automodapi',
#     'sphinx_automodapi.smart_resolver',
# ]
extensions = [
    'sphinx.ext.autodoc',
    'autoapi.extension',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    # from kovid
    'sphinx.ext.ifconfig',
    'sphinx.ext.extlinks',
    'sphinx_copybutton',
    'sphinx_inline_tabs',
    "sphinxext.opengraph",
    # 'sphinx_automodapi.automodapi',
    # 'sphinx_automodapi.smart_resolver',
    # 'sphinx_rtd_theme',
    # 'nbsphinx'
]
numpydoc_show_class_members = False

# generate autosummary even if no references
autosummary_generate = False
autosummary_imported_members = False
autoapi_keep_file = False
autoapi_type = 'python'
autoapi_dirs = [
    '../../selection',
    '../../analysis',
    '../../utils',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'
html_title = 'phis-scq'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options: Dict[str, Any] = {
#     'sidebar_hide_name': True,
#     'navigation_with_keys': True,
# }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_favicon = html_logo = '../../.logo/pelegrin_phis_square.png'
html_css_files = ['custom.css']
html_js_files = ['custom.js']
