import os
import re
from setuptools import setup


def read(file_path):
    with open(file_path, encoding="utf-8") as readme_file:
        readme = readme_file.read()
    return readme


# From https://packaging.python.org/en/latest/single_source_version.html
def find_version(file_path):
    version_file = read(file_path)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


HERE = os.path.abspath(os.path.dirname(__file__))

# From https://github.com/jupyterlab/jupyterlab/blob/master/setupbase.py, BSD licensed
def find_packages(top=HERE):
    """
    Find all of the packages.
    """
    packages = []
    for d, dirs, _ in os.walk(top, followlinks=True):
        if os.path.exists(os.path.join(d, "__init__.py")):
            packages.append(os.path.relpath(d, top).replace(os.path.sep, "."))
        elif d != top:
            # Do not look for packages in subfolders if current is not a package
            dirs[:] = []
    return packages


NAME                          = "deep_lincs"
DESCRIPTION                   = "A framework for deep learning with LINCS L1000 Data"
LONG_DESCRIPTION              = read("README.md")
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
PACKAGES                      = find_packages()
KEYWORDS                      = "LINCS L1000 gene expression"
AUTHOR                        = "Trevor Manz"
AUTHOR_EMAIL                  = "trevor.j.manz@gmail.com"
URL                           = "https://deep-lincs.readthedocs.io"
DOWNLOAD_URL                  = "https://github.com/manzt/deep_lincs"
LICENSE                       = "MIT"
VERSION                       = find_version("deep_lincs/__init__.py")

configuration = {
    "name": NAME,
    "description": DESCRIPTION,
    "long_description": LONG_DESCRIPTION,
    "long_description_content_type": LONG_DESCRIPTION_CONTENT_TYPE,
    "packages": PACKAGES,
    "keywords": KEYWORDS,
    "author": AUTHOR,
    "author_email": AUTHOR_EMAIL,
    "url": URL,
    "download_url": DOWNLOAD_URL,
    "license": LICENSE,
    "version": VERSION,
    "classifiers": [
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
    ],
}

setup(**configuration)
