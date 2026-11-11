"""Convert AMBER parameters (frcmod + mol2 with charges) of a cofactor into an OpenMM XML force field."""
import argparse

import parmed as pmd


def frcmod_mol2_to_xml(frcmod, mol2, out_xml):
    """Write an OpenMM XML force field from a complete frcmod and a mol2 with partial charges.

    Args:
        frcmod (str): frcmod with all the parameters of the molecule (``parmchk2 -a Y``), not only the
            ones missing in GAFF or parm10.
        mol2 (str): mol2 file with the residue template(s) and partial charges.
        out_xml (str): output XML file (use it with ``--custom-forcefield``).
    """
    ff = pmd.openmm.OpenMMParameterSet.from_parameterset(pmd.amber.AmberParameterSet(frcmod))
    mol2 = pmd.load_file(mol2)
    # A mol2 with several residues is loaded as a ResidueTemplateContainer, otherwise as a ResidueTemplate
    if isinstance(mol2, pmd.modeller.ResidueTemplateContainer):
        ff.residues = mol2.to_library()
    else:
        ff.residues[mol2.name] = mol2
    ff.write(out_xml)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FRCMOD + mol2 to xml ff')
    parser.add_argument('--frcmod', '-f', required=True, type=str, help='Frcmod with all information, not just what is missing in gaff or parm10')
    parser.add_argument('--mol2', '-m', required=True, type=str, help='Mol2 of the molecule with the partial charges.')
    parser.add_argument('--output', '-o', required=True, type=str, help='Output xml file.')
    args = parser.parse_args()
    frcmod_mol2_to_xml(args.frcmod, args.mol2, args.output)
