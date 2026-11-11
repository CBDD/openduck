"""OpenMM import shim.

OpenMM >= 7.6 is imported as ``openmm``; older releases (e.g. the 7.5 series pinned in
``environment.yaml``) only provide the ``simtk.openmm`` namespace. Import ``mm``, ``app``
and ``unit`` from here so OpenDUck works with both.
"""
try:
    import openmm as mm
    from openmm import app, unit
except ImportError:  # OpenMM < 7.6
    from simtk import openmm as mm
    from simtk.openmm import app
    from simtk import unit

__all__ = ['mm', 'app', 'unit']
