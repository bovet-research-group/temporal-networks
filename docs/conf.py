from importlib.metadata import version as get_version

# -- Project information -------------------------------------------------------

project = 'tempnet'
copyright = '2025, Alexandre Bovet, Yasaman Asgari, Jonas I. Liechti'
author = 'Alexandre Bovet, Yasaman Asgari, Jonas I. Liechti'

release = get_version('tempnet')

# -- General configuration -----------------------------------------------------

extensions = [
    'myst_parser',
    'sphinx_design',
    'sphinx_togglebutton',
    'sphinx_copybutton',
    'autoapi.extension',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_gallery.gen_gallery',
]

# Napoleon: NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Type hints in description only (not in signatures) for readability
autodoc_typehints = 'description'

# -- Intersphinx ---------------------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# -- HTML output ---------------------------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_title = 'tempnet'
html_static_path = ['_static']
html_css_files = ['custom.css', 'custom-pst.css']

html_context = {
    'github_user': 'bovet-research-group',
    'github_repo': 'temporal-networks',
    'github_version': 'main',
    'doc_path': 'docs/',
    'default_mode': 'light',
}

html_theme_options = {
    'use_edit_page_button': True,
    'navbar_start': ['navbar-logo'],
    'navbar_center': ['navbar-nav'],
    'navbar_end': ['theme-switcher', 'navbar-icon-links'],
    'navbar_persistent': ['search-button'],
    'secondary_sidebar_items': ['page-toc', 'edit-this-page', 'sourcelink'],
    'footer_start': ['copyright', 'sphinx-version'],
}

# -- AutoAPI -------------------------------------------------------------------

autoapi_dirs = ['../src/']
autoapi_member_order = 'groupwise'
autoapi_python_class_content = 'both'  # use both class and __init__ docstring
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
]

# -- Sphinx-Gallery ------------------------------------------------------------

sphinx_gallery_conf = {
    'examples_dirs': '../example',   # path to example scripts
    'gallery_dirs': 'auto_examples',  # path to save generated gallery output
    'filename_pattern': r'plot_',     # only execute files prefixed with plot_
    'ignore_pattern': r'exmpl_',
    'plot_gallery': True,
}

# -- MyST ----------------------------------------------------------------------

myst_enable_extensions = [
    'dollarmath',
    'attrs_block',
    'amsmath',
    'deflist',
    'colon_fence',
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
