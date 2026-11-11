"""Paths to the data files shipped with OpenDUck (parameters and templates)."""
import os

_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def data_path(*parts):
    """Absolute path of a file inside the ``duck`` package, e.g. ``data_path('parameters', 'waters', 'tip3p.prmtop')``."""
    return os.path.join(_PACKAGE_DIR, *parts)
