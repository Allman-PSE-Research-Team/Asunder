import os
import sys
import warnings

sys.path.insert(0, os.path.abspath('..'))

# sphinx-autodoc-typehints emits a Sphinx 10 deprecation warning on Sphinx 9.
# The warning is external to this project and clutters strict docs builds.
warnings.filterwarnings(
    "ignore",
    message=r".*set_application.*deprecated.*",
    module=r"sphinx_autodoc_typehints\._parser",
)

project = "Asunder"
author = "Allman Group"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "furo"
html_static_path = ["_static", "../assets"]
html_logo = "_static/sundered.gif"

autosummary_generate = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "all"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

autodoc_mock_imports = [
    "gurobipy",
    "igraph",
    "leidenalg",
    "cpnet",
    "matplotlib",
    "matplotlib.pyplot",
    "seaborn",
]
