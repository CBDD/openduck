"""Small-molecule parametrization with the ``openmmforcefields`` template generators.

The ligand is read with the OpenFF toolkit (partial charges are computed with AM1-BCC by the
generator), parametrized with the requested force field and converted to a parmed Structure.
The coordinates and the residue name (``UNL``) come from an RDKit PDB export of the same file,
written to ``ligand.pdb`` (also used later by the SDF reports).
"""
import shutil

import parmed
from openff.toolkit.topology import Molecule, Topology
from rdkit import Chem

from duck.utils.compat import app


def _as_sdf(ligand_file):
    """OpenFF only reads SDF: copy ``.mol`` files to ``.sdf`` and return the path to read."""
    if ligand_file.endswith('.mol'):
        new_file = ligand_file[:-len('.mol')] + '.sdf'
        shutil.copyfile(ligand_file, new_file)
        return new_file
    return ligand_file


def generateFFStructureRDK_from_template(ligand_file, FFtemplate, **template_kwargs):
    """Parametrize a ligand with an ``openmmforcefields`` template generator.

    Args:
        ligand_file (str): ligand in MOL/SDF format (first molecule), with explicit hydrogens.
        FFtemplate (class): template generator class (e.g. ``GAFFTemplateGenerator``).
        **template_kwargs: extra arguments for the generator (e.g. ``forcefield='gaff-2.11'``).

    Returns:
        parmed.Structure: parametrized ligand with the coordinates of ``ligand_file``.
    """
    ligand_file = _as_sdf(ligand_file)
    ligand_off_molecule = Molecule.from_file(ligand_file, allow_undefined_stereo=True)
    ligand_topology = Topology.from_molecules(ligand_off_molecule).to_openmm()
    generator = FFtemplate(molecules=ligand_off_molecule, **template_kwargs)
    force_field = app.ForceField()
    force_field.registerTemplateGenerator(generator.generator)
    ligand_system = force_field.createSystem(ligand_topology)

    # Coordinates (and the UNL residue name) from an RDKit PDB export of the same molecule
    Chem.MolToPDBFile(Chem.MolFromMolFile(ligand_file, removeHs=False), 'ligand.pdb')
    ligand_pdbfile = app.PDBFile('ligand.pdb')
    return parmed.openmm.load_topology(ligand_pdbfile.topology, ligand_system, xyz=ligand_pdbfile.positions)


def generateSMIRNOFFStructureRDK(ligand_file, ff='openff-1.0.0'):
    """Parametrize a ligand with an Open Force Field (SMIRNOFF) force field, ``openff-1.0.0`` by default."""
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator
    print('Parametrizing with SMIRNOFF')
    return generateFFStructureRDK_from_template(ligand_file, SMIRNOFFTemplateGenerator, forcefield=ff)


def generateGAFFStructureRDK(ligand_file, ff='gaff-2.11'):
    """Parametrize a ligand with GAFF (``gaff-2.11`` by default)."""
    from openmmforcefields.generators import GAFFTemplateGenerator
    print('Parametrizing with GAFF2')
    return generateFFStructureRDK_from_template(ligand_file, GAFFTemplateGenerator, forcefield=ff)


def generateEspalomaFFStructureRDK(ligand_file, ff='espaloma-0.2.2'):
    """Parametrize a ligand with Espaloma (requires an ``openmmforcefields`` release that provides it)."""
    from openmmforcefields.generators import EspalomaTemplateGenerator
    print('Parametrizing with Espaloma')
    return generateFFStructureRDK_from_template(ligand_file, EspalomaTemplateGenerator, forcefield=ff)


if __name__ == '__main__':
    import sys
    generateGAFFStructureRDK(sys.argv[1])
