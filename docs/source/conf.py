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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Hugin EO'
copyright = '2019, HuginEO Contributors'
author = 'Marian Neagul, Teodora Selea, Gabriel Iuhasz, Alexandru Munteanu'

try:
	import hugin
	release = hugin.__version__
except:
	with open('../src/hugin/__init__.py') as f:
		for line in f:
			if line.find("__version__") >= 0:
				version = line.split("=")[1].strip()
				version = version.strip('"')
				version = version.strip("'")
				continue

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
	'recommonmark',
	'sphinx_rtd_theme',
	'sphinx.ext.napoleon',
	'sphinx.ext.autodoc'
]

master_doc = 'index'

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
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_logo = '_static/hugin-logov1.png'
latex_logo = '_static/hugin-logov1.pdf'

latex_documents = [
    (master_doc, 'hugineo.tex', u'HuginEO Documentation',
     u'Hugin EO Contributors', 'manual'),
]

latex_elements = {
	'papersize': 'a4paper',
	'extraclassoptions': 'openany,oneside',
	'pointsize': '12pt',
	'preamble': r'''
		\usepackage{charter}
		\usepackage[defaultsans]{lato}
		\usepackage{inconsolata}
	''',
}
