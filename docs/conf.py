"""Configuration file for the Sphinx documentation builder."""

import shutil

shutil.copyfile("../README.md", "source/README.md")
readme_f_str = open("source/README.md", "r").read()
readme_f_str = readme_f_str.replace(
    "<code>",
    '<p style="display: inline-block;"><code class="docutils literal notranslate"><span class="pre">',
).replace("</code>", "</span></code></p>")
# readme_f_str = re.sub(r"https://torchrun\.xyz/(.+?)\.html", r"./\1.md", readme_f_str)
open("source/README.md", "w").write(readme_f_str)

shutil.copyfile("../CONTRIBUTING.md", "source/contributing.md")

project = "misen"
copyright = ""
github_username = "apoorvkh"
github_repository = "misen"
html_theme = "furo"
language = "en"

extensions = [
    "sphinx.ext.autodoc",
    "myst_parser",  # support markdown
    "sphinx.ext.intersphinx",  # link to external docs
    "sphinx.ext.napoleon",  # for google style docstrings
    "sphinx.ext.linkcode",  # link to github source
    # sidebar
    "sphinx_toolbox.sidebar_links",
    "sphinx_toolbox.github",
]

maximum_signature_line_length = 90
autodoc_member_order = "bysource"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.9", None),
}

from docs.linkcode_github import generate_linkcode_resolve_fn

linkcode_resolve = generate_linkcode_resolve_fn(project, github_username, github_repository)
