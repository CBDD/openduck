"""Parametrization and solvation of the protein-ligand complex (shared by the OpenMM and AMBER routes).

Steps: ligand parametrization (SMIRNOFF or GAFF2, AM1-BCC charges), receptor parametrization
with an OpenMM protein force field (plus an optional custom XML, e.g. for cofactors), optional
structural waters, solvation in a cubic box with PDBFixer, ions (Na+/Cl-) and solvent
parametrization, and optional hydrogen mass repartitioning (HMR). The result is written as
``system_complex.prmtop``/``.inpcrd`` (``HMR_system_complex.prmtop`` with HMR) and pickled to
``complex_system.pickle`` for the OpenMM simulations.
"""
import os
import pickle
import sys
from pathlib import Path

import parmed
from pdbfixer import PDBFixer
from rdkit import Chem

from duck.utils.cal_ints import clean_up_files
from duck.utils.compat import mm, app, unit
from duck.utils.duck_stuff import apply_hmr
from duck.utils.gen_system import generateGAFFStructureRDK, generateSMIRNOFFStructureRDK
from duck.utils.resources import data_path

# ESPALOMA (generateEspalomaFFStructureRDK) is not exposed yet: it is missing from the conda release of
# openmmforcefields pinned in environment.yaml.
FF_GENERATORS = {'SMIRNOFF': generateSMIRNOFFStructureRDK, 'GAFF2': generateGAFFStructureRDK}
WATER_MODELS = ('tip3p', 'spce', 'tip4pew')


def find_box_size(input_file="complex.pdb", add_factor=20):
    """
    Find the size of a cubic box that can contain the input complex.

    Args:
        input_file (str): The path to the input PDB file containing the complex.
        add_factor (float): The additional space (in Angstroms) to add to the maximum dimension of the complex.

    Returns:
        int: The size (in Angstroms) of the cubic box that can contain the complex.
    """
    complex = parmed.load_file(input_file)
    coords = complex.coordinates
    size = max(coords.max(axis=0) - coords.min(axis=0)) + add_factor
    return int(size) + 1


def load_retained_waters(waters_to_retain):
    """Load the structural waters to keep (residues renamed to HOH), or None if no file is given or it is missing."""
    if not waters_to_retain or not Path(waters_to_retain).exists():
        return None
    print(f'waters retained from {Path(waters_to_retain)}')
    waters_retained = parmed.load_file(waters_to_retain)
    for atom in waters_retained:
        if atom.residue.name != 'HOH':
            sys.stderr.write(f'Warning: renaming {atom.residue.name}:{atom.name} to HOH:{atom.name}\n')
            atom.residue.name = 'HOH'
    return waters_retained


def prepare_system(ligand_file, protein_file, forcefield_str="amber99sb.xml", water_ff_str='tip3p',
                   hmr=False, small_molecule_ff='SMIRNOFF', box_buffer_distance=10, ionicStrength=0.1,
                   waters_to_retain="waters_to_retain.pdb", custom_forcefield=None, fix_ligand_file=False,
                   clean_up=False):
    """
    Prepares a protein-ligand system by loading and modifying molecular structures and parameters,
    solvating the system in its water box, and parameterizing ions and solvent using the specified force fields.

    Parameters
    ----------
    ligand_file : str
        The name of the file containing the ligand structure to be prepared (MOL or SDF).
    protein_file : str
        The name of the file containing the protein structure to be prepared.
    forcefield_str : str, optional
        The OpenMM force field file for the protein, by default "amber99sb.xml".
    water_ff_str : str, optional
        Water model: tip3p (default), spce or tip4pew.
    hmr : bool, optional
        Whether to apply hydrogen mass repartitioning, by default False.
    small_molecule_ff : str, optional
        The force field to use for ligand parameterization (SMIRNOFF or GAFF2), by default "SMIRNOFF".
    box_buffer_distance : float, optional
        Minimum distance between the solute and the box edge, in angstroms, by default 10.
    ionicStrength : float, optional
        The ionic strength to use for solvation, by default 0.1 M.
    waters_to_retain : str, optional
        PDB file with structural waters to keep, by default "waters_to_retain.pdb" (ignored if None or missing).
    custom_forcefield : str, optional
        Extra OpenMM XML force field (e.g. for cofactors or non-standard residues).
    fix_ligand_file : bool, optional
        Add missing hydrogens and fix the charge of tetravalent nitrogens before parametrizing.
    clean_up : bool, optional
        Remove the intermediate files.

    Returns
    -------
    list
        ``[parmed.Structure]`` with the prepared protein, ligand, ions and solvent.
    """
    if small_molecule_ff.upper() not in FF_GENERATORS:
        print(f'{small_molecule_ff} is not in the accepted small molecule forcefields. Defaulting it to SMIRNOFF')
        small_molecule_ff = 'SMIRNOFF'
    if water_ff_str not in WATER_MODELS:
        print(f'{water_ff_str} is not an available water model {WATER_MODELS}. Defaulting to tip3p')
        water_ff_str = 'tip3p'

    print("Preparing ligand")
    if fix_ligand_file:
        fixed_ligand_file = os.path.join(os.path.dirname(ligand_file), f"fixed_{os.path.basename(ligand_file)}")
        fix_ligand(ligand_file, fixed_ligand_file)
        ligand_file = fixed_ligand_file
    ligand_pmd = FF_GENERATORS[small_molecule_ff.upper()](ligand_file)

    print("Preparing protein")
    parmed.load_file(protein_file).write_pdb("fixed.pdb")
    protein = parmed.load_file("fixed.pdb")  # ions and waters in the receptor file are kept
    if custom_forcefield:
        print(f'Loading custom forcefield {custom_forcefield}')
        forcefield = app.ForceField(forcefield_str, custom_forcefield)
    else:
        forcefield = app.ForceField(forcefield_str)
    protein_system = forcefield.createSystem(protein.topology)
    protein_pmd = parmed.openmm.load_topology(protein.topology, protein_system, protein.positions)
    protein_pmd.save("protein_prepared.pdb", overwrite=True)
    waters_retained = load_retained_waters(waters_to_retain)
    if waters_retained is not None:
        prot_lig_pmd = protein_pmd + ligand_pmd + waters_retained
    else:
        prot_lig_pmd = protein_pmd + ligand_pmd
    prot_lig_pmd.save("complex.pdb", overwrite=True)

    print("Solvation")
    # Cubic box: largest solute dimension plus the buffer on both sides (the original DUck protocol used 18 A)
    box_size = find_box_size("complex.pdb", add_factor=box_buffer_distance * 2)
    fixer = PDBFixer("complex.pdb")
    fixer.addSolvent(box_size * mm.Vec3(0.1, 0.1, 0.1),  # A -> nm
                     positiveIon="Na+", negativeIon="Cl-", ionicStrength=ionicStrength * unit.molar)
    with open("complex_solvated.pdb", "w") as fh:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, fh)
    print("Solvation done")

    print("Parametrizing ions")
    complex = parmed.load_file("./complex_solvated.pdb")
    ions = complex["(:NA,CL)"]
    ions_system = app.ForceField('amber99sb.xml').createSystem(ions.topology)
    ions_pmd = parmed.openmm.load_topology(ions.topology, ions_system, ions.positions)

    print(f"Parametrizing solvent using {water_ff_str}")
    solvent = complex["(:HOH)"]
    solvent_pmd = parmed.load_file(data_path('parameters', 'waters', f'{water_ff_str}.prmtop'))
    solvent_pmd *= len(solvent.residues)
    if water_ff_str == 'tip4pew':  # TIP4P-Ew has an extra virtual site whose positions need to be generated
        modeller = app.Modeller(solvent.topology, solvent.positions)
        modeller.addExtraParticles(app.ForceField(f'{water_ff_str}.xml'))
        solvent_pmd.positions = modeller.positions
    else:
        solvent_pmd.positions = solvent.positions

    print("Merging structures")
    combined_pmd = protein_pmd + ligand_pmd + ions_pmd + solvent_pmd
    combined_pmd.box_vectors = complex.box_vectors
    combined_pmd.save("system_complex.prmtop", overwrite=True)
    combined_pmd.save("system_complex.inpcrd", overwrite=True)
    if hmr:
        print('Performing Hydrogen Mass Repartition')
        apply_hmr(combined_pmd)
        combined_pmd.save('HMR_system_complex.prmtop', overwrite=True)

    print("Writing pickle")
    with open("./complex_system.pickle", "wb") as fh:
        pickle.dump([combined_pmd], fh)
    if clean_up:
        print('Removing intermediate files')
        # Only files generated here, so files that were in the directory beforehand are not deleted.
        # ligand.pdb is kept: the SDF reports use it.
        files_to_delete = ['complex_solvated.pdb', 'fixed.pdb', 'complex.pdb', 'run.tleap', 'indice.text',
                           'no_buffer_altlocs.pdb', 'leap.log', 'protein_out.pdb', 'protein_out_no_ace_nme.pdb',
                           'protonated_protein_out.pdb',
                           os.path.splitext(os.path.basename(ligand_file))[0] + '.pdb',
                           'protein_prepared.pdb', 'chunk_leap.log']
        clean_up_files(files_to_delete=[f for f in files_to_delete if f != 'ligand.pdb'])
    return [combined_pmd]


def fix_ligand(ligand_file, fixed_ligand_file):
    """
    Add explicit hydrogens to the ligand and ensure tetravalent nitrogens are assigned a +1 charge.

    Args:
        ligand_file (str): input MOL/SDF file (first molecule is used).
        fixed_ligand_file (str): output SDF file.
    """
    with Chem.SDMolSupplier(ligand_file, sanitize=False) as supplier:
        mol = supplier[0]
    for chem_problem in Chem.DetectChemistryProblems(mol):
        if chem_problem.GetType() == 'AtomValenceException':
            atom = mol.GetAtomWithIdx(chem_problem.GetAtomIdx())
            if atom.GetSymbol() == 'N' and atom.GetFormalCharge() == 0 and atom.GetExplicitValence() == 4:
                atom.SetFormalCharge(1)
    Chem.SanitizeMol(mol)
    fixed_mol = Chem.AddHs(mol, addCoords=True)
    with Chem.SDWriter(fixed_ligand_file) as sdwriter:
        sdwriter.write(fixed_mol)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("USAGE : python parametrize.py protein.pdb ligand.mol\n")
    prepare_system(sys.argv[2], sys.argv[1], hmr=True, water_ff_str='tip3p')
