import os
import sys

# Source file types:
source_suffix = ['.rst', '.md']

# -*- coding: utf-8 -*-
#
# Slugnet documentation build configuration file, created by
# sphinx-quickstart on Sun, Dec. 3rd 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
sys.path.insert(0, os.path.abspath('../'))


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.githubpages',
    # 'sphinx.ext.mathjax',
    'sphinxcontrib.tikz',
    'sphinx.ext.imgmath',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'matplotlib.sphinxext.only_directives',
    'matplotlib.sphinxext.plot_directive',
    'IPython.sphinxext.ipython_directive',
    'IPython.sphinxext.ipython_console_highlighting',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'Slugnet'
copyright = u'2017, <a href="https://jarrodkahn.com">Jarrod Kahn</a>'
author = u'Jarrod Kahn'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = u'0.0.1'
# The full version, including alpha/beta/rc tags.
release = u'0.0.1'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'slugnetdoc'


tikz_tikzlibraries = 'chains'

# -- Options for LaTeX output ---------------------------------------------

imgmath_font_size = 16
imgmath_image_format = 'svg'
imgmath_latex_preamble = """
\usepackage{graphicx}
\usepackage{amsmath, bm}
\usepackage{graphicx}
\usepackage{xfrac}
\usepackage{amssymb}
\usepackage{dsfont}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{tikz}
\usepackage{isomath}
\usetikzlibrary{decorations.pathreplacing}
\usetikzlibrary{calc}
\usetikzlibrary{positioning}
\usetikzlibrary{shadows.blur}
\usetikzlibrary{arrows.meta}
\usetikzlibrary{arrows}
\\newcommand{\\thickhat}[1]{\mathbf{\hat{\\text{$#1$}}}}
\\newcommand{\R}{\mathbb{R}}
\\renewcommand{\\thesubsection}{\\thesection.\\alph{subsection}}
\let\oldReturn\Return
\\renewcommand{\Return}{\State\oldReturn}
\DeclareMathOperator*{\\argmin}{arg\,min}

\makeatletter
\def\BState{\State\hskip-\ALG@thistlm}
\makeatother
"""

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': ('\usepackage{tkiz}\n'
    #              '\usetikzlibrary{positioning}\n'
    #              '\usepackage{bm}\n'
    #              '\usepackage{amsmath}'),

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'slugnet', u'Slugnet',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'Slugnet', u'Slugnet Documentation',
     author, 'Slugnet', 'An experimental neural networks library',
     'Miscellaneous'),
]

html_sidebars = {'**': [
    'navigation.html',
    'relations.html',
    'sourcelink.html',
    'searchbox.html'
], }

sidebar_includehidden = False

# Order autodoc docs by source order
autodoc_member_order = 'bysource'

def setup(app):
    app.add_stylesheet('caption.css')

html_theme_options = {
    'font_size': '19px',
    'github_user': 'kahnvex',
    'github_repo': 'slugnet',
    'github_button': True,
    'github_banner': True
}
