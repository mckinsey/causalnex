#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# causalnex documentation build configuration file,
# created by, sphinx-quickstart on Mon Dec 18 11:31:24 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import importlib
import re
import shutil
import sys
from distutils.dir_util import copy_tree
from inspect import getmembers, isclass, isfunction
from pathlib import Path
from typing import List

import patchy
from click import secho, style
from recommonmark.transform import AutoStructify
from sphinx.ext.autosummary.generate import generate_autosummary_docs

from causalnex import __version__ as release

# -- Project information -----------------------------------------------------

project = "causalnex"
copyright = "2020, QuantumBlack"
author = "QuantumBlack"

# The short X.Y version.
version = re.match(r"^([0-9]+\.[0-9]+).*", release).group(1)

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "recommonmark",
    "sphinx_markdown_tables",
    "sphinx_copybutton",
]

# enable autosummary plugin (table of contents for modules/classes/class
# methods)
autosummary_generate = True
autosummary_imported_members = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ["**cli*", "_build", "**.ipynb_checkpoints", "_templates"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
here = Path(__file__).parent.absolute()
# html_logo = str(here / "causalnex_logo.svg")

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "collapse_navigation": False,
    "style_external_links": True,
    # "logo_only": True
    # "github_url": "https://github.com/quantumblacklabs/causalnex"
}

html_context = {
    "display_github": True,
    "github_url": "https://github.com/quantumblacklabs/causalnex/tree/develop/docs/source",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.

# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}

html_show_sourcelink = False

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "causalnexdoc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "causalnex.tex", "causalnex Documentation", "QuantumBlack", "manual")
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "causalnex", "causalnex Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "causalnex",
        "causalnex Documentation",
        author,
        "causalnex",
        "Toolkit for causal reasoning (Bayesian Networks / Inference)",
        "Data-Science",
    )
]

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Extension configuration -------------------------------------------------

# nbsphinx_prolog = """
# see here for prolog/epilog details:
# https://nbsphinx.readthedocs.io/en/0.4.0/prolog-and-epilog.html
# """

nbsphinx_epilog = """
.. note::

     Found a bug, or didn't find what you were looking for? `🙏Please file a
     ticket <https://github.com/quantumblacklabs/causalnex/issues/new/choose>`_
"""

# -- NBconvert kernel config -------------------------------------------------
nbsphinx_kernel_name = "causalnex"


# -- causalnex specific configuration ------------------
MODULES = []


def get_classes(module):
    importlib.import_module(module)
    return [obj[0] for obj in getmembers(sys.modules[module], lambda obj: isclass(obj))]


def get_functions(module):
    importlib.import_module(module)
    return [
        obj[0] for obj in getmembers(sys.modules[module], lambda obj: isfunction(obj))
    ]


def remove_arrows_in_examples(lines):
    for i, line in enumerate(lines):
        lines[i] = line.replace(">>>", "")


def autolink_replacements(what):
    """
    Create a list containing replacement tuples of the form:
    (``regex``, ``replacement``, ``obj``) for all classes and methods which are
    imported in ``MODULES`` ``__init__.py`` files. The ``replacement``
    is a reStructuredText link to their documentation.
    For example, if the docstring reads:
        This DataSet loads and saves ...
    Then the word ``DataSet``, will be replaced by
    :class:`~causalnex.io.DataSet`
    Works for plural as well, e.g:
        These ``DataSet``s load and save
    Will convert to:
        These :class:`causalnex.io.DataSet` s load and
        save
    Args:
        what (str) : The objects to create replacement tuples for. Possible
            values ["class", "func"]
    Returns:
        List[Tuple[regex, str, str]]: A list of tuples: (regex, replacement,
        obj), for all "what" objects imported in __init__.py files of
        ``MODULES``
    """
    replacements = []
    suggestions = []
    for module in MODULES:
        if what == "class":
            objects = get_classes(module)
        elif what == "func":
            objects = get_functions(module)

        # Look for recognised class names/function names which are
        # surrounded by double back-ticks
        if what == "class":
            # first do plural only for classes
            replacements += [
                (
                    r"``{}``s".format(obj),
                    ":{}:`~{}.{}`\\\\s".format(what, module, obj),
                    obj,
                )
                for obj in objects
            ]

        # singular
        replacements += [
            (r"``{}``".format(obj), ":{}:`~{}.{}`".format(what, module, obj), obj)
            for obj in objects
        ]

        # Look for recognised class names/function names which are NOT
        # surrounded by double back-ticks, so that we can log these in the
        # terminal
        if what == "class":
            # first do plural only for classes
            suggestions += [
                (r"(?<!\w|`){}s(?!\w|`{{2}})".format(obj), "``{}``s".format(obj), obj)
                for obj in objects
            ]

        # then singular
        suggestions += [
            (r"(?<!\w|`){}(?!\w|`{{2}})".format(obj), "``{}``".format(obj), obj)
            for obj in objects
        ]

    return replacements, suggestions


def log_suggestions(lines: List[str], name: str):
    """Use the ``suggestions`` list to log in the terminal places where the
    developer has forgotten to surround with double back-ticks class
    name/function name references.

    Args:
        lines: The docstring lines.
        name: The name of the object whose docstring is contained in lines.
    """
    title_printed = False

    for i in range(len(lines)):
        if ">>>" in lines[i]:
            continue

        for existing, replacement, obj in suggestions:
            new = re.sub(existing, r"{}".format(replacement), lines[i])
            if new == lines[i]:
                continue
            if ":rtype:" in lines[i] or ":type " in lines[i]:
                continue

            if not title_printed:
                secho("-" * 50 + "\n" + name + ":\n" + "-" * 50, fg="blue")
                title_printed = True

            print(
                "["
                + str(i)
                + "] "
                + re.sub(existing, r"{}".format(style(obj, fg="magenta")), lines[i])
            )
            print(
                "["
                + str(i)
                + "] "
                + re.sub(existing, r"``{}``".format(style(obj, fg="green")), lines[i])
            )

    if title_printed:
        print("\n")


def autolink_classes_and_methods(lines):
    for i in range(len(lines)):
        if ">>>" in lines[i]:
            continue

        for existing, replacement, obj in replacements:
            lines[i] = re.sub(existing, r"{}".format(replacement), lines[i])


# Sphinx build passes six arguments
def autodoc_process_docstring(app, what, name, obj, options, lines):
    try:
        # guarded method to make sure build never fails
        log_suggestions(lines, name)
        autolink_classes_and_methods(lines)
    except Exception as e:
        print(
            style(
                "Failed to check for class name mentions that can be "
                "converted to reStructuredText links in docstring of {}. "
                "Error is: \n{}".format(name, str(e)),
                fg="red",
            )
        )

    remove_arrows_in_examples(lines)


# Sphinx build method passes six arguments
def skip(app, what, name, obj, skip, options):
    if name == "__init__":
        return False
    return skip


def _prepare_build_dir(app, config):
    """Get current working directory to the state expected
    by the ReadTheDocs builder. Shortly, it does the same as
    ./build-docs.sh script except not running `sphinx-build` step."""
    build_root = Path(app.srcdir)
    build_out = Path(app.outdir)
    copy_tree(str(here / "source"), str(build_root))
    copy_tree(str(build_root / "api_docs"), str(build_root))
    shutil.rmtree(str(build_root / "api_docs"))
    shutil.rmtree(str(build_out), ignore_errors=True)
    copy_tree(str(build_root / "css"), str(build_out / "_static" / "css"))
    copy_tree(
        str(build_root / "04_user_guide/images"), str(build_out / "04_user_guide")
    )
    shutil.rmtree(str(build_root / "css"))


def setup(app):
    app.connect("config-inited", _prepare_build_dir)
    app.connect("autodoc-process-docstring", autodoc_process_docstring)
    app.connect("autodoc-skip-member", skip)
    app.add_stylesheet("css/qb1-sphinx-rtd.css")
    # fix a bug with table wraps in Read the Docs Sphinx theme:
    # https://rackerlabs.github.io/docs-rackspace/tools/rtd-tables.html
    app.add_stylesheet("css/theme-overrides.css")
    # add "Copy" button to code snippets
    app.add_stylesheet("css/copybutton.css")
    app.add_stylesheet("css/causalnex.css")

    # when using nbsphinx, to allow mathjax render properly
    app.config._raw_config.pop("mathjax_config")
    # enable rendering RST tables in Markdown
    app.add_config_value("recommonmark_config", {"enable_eval_rst": True}, True)
    app.add_transform(AutoStructify)


def fix_module_paths():
    """
    This method fixes the module paths of all class/functions we import in the
    __init__.py file of the various causalnex submodules.
    """
    for module in MODULES:
        mod = importlib.import_module(module)
        if not hasattr(mod, "__all__"):
            mod.__all__ = get_classes(module) + get_functions(module)


# (regex, restructuredText link replacement, object) list
replacements = []

# (regex, class/function name surrounded with back-ticks, object) list
suggestions = []

try:
    # guarded code to make sure build never fails
    replacements_f, suggestions_f = autolink_replacements("func")
    replacements_c, suggestions_c = autolink_replacements("class")
    replacements = replacements_f + replacements_c
    suggestions = suggestions_f + suggestions_c
except Exception as e:
    print(
        style(
            "Failed to create list of (regex, reStructuredText link "
            "replacement) for class names and method names in docstrings. "
            "Error is: \n{}".format(str(e)),
            fg="red",
        )
    )

fix_module_paths()

patchy.patch(
    generate_autosummary_docs,
    """\
@@ -3,7 +3,7 @@ def generate_autosummary_docs(sources, output_dir=None, suffix='.rst',
                               base_path=None, builder=None, template_dir=None,
                               imported_members=False, app=None):
     # type: (List[unicode], unicode, unicode, Callable, Callable, unicode, Builder, unicode, bool, Any) -> None  # NOQA
-
+    imported_members = True
     showed_sources = list(sorted(sources))
     if len(showed_sources) > 20:
         showed_sources = showed_sources[:10] + ['...'] + showed_sources[-10:]
""",
)

patchy.patch(
    generate_autosummary_docs,
    """\
@@ -96,6 +96,21 @@ def generate_autosummary_docs(sources, output_dir=None, suffix='.rst',
                           if x in include_public or not x.startswith('_')]
                 return public, items

+            import importlib
+            def get_public_modules(obj, typ):
+                # type: (Any, str) -> List[str]
+                items = []  # type: List[str]
+                for item in getattr(obj, '__all__', []):
+                    try:
+                        importlib.import_module(name + '.' + item)
+                    except ImportError:
+                        continue
+                    finally:
+                        if item in sys.modules:
+                            sys.modules.pop(name + '.' + item)
+                    items.append(name + '.' + item)
+                return items
+
             ns = {}  # type: Dict[unicode, Any]
""",
)

patchy.patch(
    generate_autosummary_docs,
    """\
@@ -106,6 +106,9 @@ def generate_autosummary_docs(sources, output_dir=None, suffix='.rst',
                     get_members(obj, 'class', imported=imported_members)
                 ns['exceptions'], ns['all_exceptions'] = \\
                     get_members(obj, 'exception', imported=imported_members)
+                ns['public_modules'] = get_public_modules(obj, 'module')
+                ns['functions'] = [m for m in ns['functions'] if not hasattr(obj, '__all__') or m in obj.__all__]
+                ns['classes'] = [m for m in ns['classes'] if not hasattr(obj, '__all__') or m in obj.__all__]
             elif doc.objtype == 'class':
                 ns['members'] = dir(obj)
                 ns['inherited_members'] = \\
""",
)
