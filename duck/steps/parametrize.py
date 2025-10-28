from simtk.openmm import app
from rdkit import Chem
from simtk import unit
import parmed, pkg_resources
from simtk import openmm
from pdbfixer import PDBFixer  # for solvating
import os
import sys
import pickle
from duck.utils.gen_system import generateSMIRNOFFStructureRDK, generateGAFFStructureRDK, generateEspalomaFFStructureRDK
from pathlib import Path
from duck.utils.cal_ints import clean_up_files

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
    x_coords = [x[0] for x in complex.positions]
    y_coords = [x[1] for x in complex.positions]
    z_coords = [x[2] for x in complex.positions]
    x_size = abs(max(x_coords) - min(x_coords))
    y_size = abs(max(y_coords) - min(y_coords))
    z_size = abs(max(z_coords) - min(z_coords))
    val_in_ang = max(x_size, y_size, z_size) + add_factor * unit.angstrom
    return int(val_in_ang.value_in_unit(unit.angstrom)) + 1

def prepare_system(ligand_file, protein_file, forcefield_str="amber99sb.xml", water_ff_str = 'tip3p',
                   hmr=False, small_molecule_ff = 'SMIRNOFF', box_buffer_distance = 10, ionicStrength = 0.1,
                   waters_to_retain="waters_to_retain.pdb", custom_forcefield=None, fix_ligand_file=False, clean_up=False):
    """
    Prepares a protein-ligand system by loading and modifying molecular structures and parameters,
    solvating the system in its water box, and parameterizing ions and solvent using the specified force fields.

    Parameters
    ----------
    ligand_file : str
        The name of the file containing the ligand structure to be prepared.
    protein_file : str
        The name of the file containing the protein structure to be prepared.
    forcefield_str : str, optional
        The name of the force field file to use for protein and ions parameterization, by default "amber99sb.xml".
    water_ff_str : str, optional
        The name of the force field file to use for water parameterization, by default "tip3p".
    hmr : bool, optional
        Whether to use hydrogen mass repartitioning for simulation, by default False.
    small_molecule_ff : str, optional
        The name of the force field to use for ligand parameterization, by default "SMIRNOFF".
    box_buffer_distance : float, optional
        The buffer distance to use for box creation, in angstroms, by default 10.
    ionicStrength : float, optional
        The ionic strength to use for solvation, by default 0.1 M.
    waters_to_retain : str, optional
        The name of a PDB file containing the water molecules to retain during solvation, by default "waters_to_retain.pdb".

    Returns
    -------
    parmed.Structure
        The prepared system structure containing the protein, ligand, solvent, and ions.

    Raises
    ------
    Exception
        If an unsupported small molecule force field is specified or if an unsupported water model is specified.
    """
    #Do not put ESPALOMA yet, as it is not on the conda release of openmmforcefields yet, The function is already prepared
    #FF_generators = {'SMIRNOFF': generateSMIRNOFFStructureRDK, 'GAFF2': generateGAFFStructureRDK, 'ESPALOMA': generateEspalomaFFStructureRDK}
    FF_generators = {'SMIRNOFF': generateSMIRNOFFStructureRDK, 'GAFF2': generateGAFFStructureRDK}
    available_water_models = ['tip3p', 'spce','tip4pew']
    #Sanity checks
    if small_molecule_ff.upper() not in FF_generators:
        print(f'{small_molecule_ff} is not in the accepted small molecule forcefields. Defaulting it to SMIRNOFF')
        small_molecule_ff = 'SMIRNOFF'
    if water_ff_str not in available_water_models:
        print(f'{water_ff_str} is not available in the openmm data files. Defaulting to tip3p.xml')
        water_ff_str = 'tip3p.xml'

    print("Preparing ligand")
    if fix_ligand_file:
        fixed_ligand_file = ligand_file.replace(os.path.basename(ligand_file), f"fixed_{os.path.basename(ligand_file)}")
        fix_ligand(ligand_file, fixed_ligand_file)
        ligand_file = fixed_ligand_file
    ligand_pmd = FF_generators[small_molecule_ff.upper()](ligand_file)
    print("Fixing protein")
    protein = parmed.load_file(protein_file)
    protein.write_pdb("fixed.pdb")
    print("loading system")
   # protein = parmed.load_file("fixed.pdb")["!(:HOH,NA,CL)"]  # remove ions and water
    protein = parmed.load_file("fixed.pdb")  # don't remove ions and water
    print(f'Loading custom forcefield {custom_forcefield}')
    if custom_forcefield:
        forcefield = app.ForceField(forcefield_str, custom_forcefield)
    else:
        forcefield = app.ForceField(forcefield_str)
    protein_system = forcefield.createSystem(protein.topology)
    protein_pmd = parmed.openmm.load_topology(
        protein.topology, protein_system, protein.positions
    )
    protein_pmd.save("protein_prepared.pdb", overwrite=True)
    if Path(waters_to_retain).exists():
        print(f'waters retained from {Path(waters_to_retain)}')
        waters_retained = parmed.load_file(waters_to_retain)
        #check if waters are well named
        for atom in waters_retained:
            if atom.residue.name != 'HOH':
                sys.stderr.write(f'Warning: renaming {atom.residue.name}:{atom.name} to HOH:{atom.name}\n')
                atom.residue.name = 'HOH'
        prot_lig_pmd = protein_pmd + ligand_pmd + waters_retained
    else:
        prot_lig_pmd = protein_pmd + ligand_pmd
    prot_lig_pmd.save("complex.pdb", overwrite=True)
    print("Solvation")
    fixer = PDBFixer("complex.pdb")
    # 0.1 in Vec3 because box_size is in angstrom and fixer uses nanometer
    # scaling factor to somehow ensure no interaction with periodic image
    
    box_scaling_factor = 1.0
    # ionicStrength = 0.1 M
    # box_buffer_disance = 10

    # use the add factor as the buffer distance specified in tleap, og DUck protocol specified 18A
    box_size = find_box_size("complex.pdb",add_factor=box_buffer_distance*2) 
    fixer.addSolvent(
        box_scaling_factor * box_size * openmm.Vec3(0.1, 0.1, 0.1),
        positiveIon="Na+",
        negativeIon="Cl-",
        ionicStrength=ionicStrength * unit.molar,
    )
    # fix to use app.modeller water models, its not very pretty
    #for r in fixer.topology.residues():
    #    if r.name == 'HOH':
    #        r.name = 'WAT'
    app.PDBFile.writeFile(
        fixer.topology, fixer.positions, open("complex_solvated.pdb", "w")
    )
    print("Solvation done")
    print("Parametrizing ions")
    complex = parmed.load_file("./complex_solvated.pdb")
    ions = complex["(:NA,CL)"]
    ions_forcefield = app.ForceField('amber99sb.xml')
    ions_system = ions_forcefield.createSystem(ions.topology)
    ions_pmd = parmed.openmm.load_topology(ions.topology, ions_system, ions.positions)
    print("Parametrizing ions done")
    print(f"Parametrizing solvent using {water_ff_str}")
    solvent = complex["(:HOH)"]
    num_solvent = len(solvent.residues)
    prm_top_water_path = pkg_resources.resource_filename(
        'duck',f'parameters/waters/{water_ff_str}.prmtop'
    )
    solvent_pmd = parmed.load_file(prm_top_water_path)
    solvent_pmd *= num_solvent
    if water_ff_str == 'tip4pew': # tip4p uses an extra atom, so we need to generate the positions for this extra atom
        wat_forcefield = app.ForceField(f'{water_ff_str}.xml')
        modeller = app.Modeller(solvent.topology, solvent.positions)
        modeller.addExtraParticles(wat_forcefield)
        solvent_pmd.positions=modeller.positions
    else:
        solvent_pmd.positions=solvent.positions
    #solvent_system = wat_forcefield.createSystem(modeller.topology)
    #solvent_pmd = parmed.openmm.load_topology(modeller.topology, solvent_system, modeller.positions)
    print("Parametrizing solvent done")
    print("merge structures")
    combined_pmd = protein_pmd + ligand_pmd + ions_pmd + solvent_pmd
    #combined_pmd.get_box()
    combined_pmd.box_vectors = complex.box_vectors
    combined_pmd.save("system_complex.prmtop", overwrite=True)
    combined_pmd.save("system_complex.inpcrd", overwrite=True)
    if hmr:
        print('Performing Hydrogen Mass Repartition')
        hmass_action = parmed.tools.actions.HMassRepartition(combined_pmd)
        hmass_action.execute()
        combined_pmd.save('HMR_system_complex.prmtop', overwrite=True)

    print("merge done")
    print("writing pickle")
    complex = "./complex_system.pickle"
    pickle_out = open(complex, "wb")
    pickle.dump([combined_pmd], pickle_out)
    pickle_out.close()
    if clean_up:
        print('Removing intermediate files')
        # a list of generated files to remove, so we don't delete the files that were in the directory beforehand.
        files_to_delete = ['complex_solvated.pdb', 'fixed.pdb', 'complex.pdb', 'run.tleap', 'indice.text', 'no_buffer_altlocs.pdb', 'leap.log',
                           'protein_out.pdb', 'protein_out_no_ace_nme.pdb', 'protonated_protein_out.pdb', 'ligand.pdb', ligand_file.split('.')[0]+'.pdb',
                           'protein_prepared.pdb', 'chunk_leap.log']
        clean_up_files(files_to_delete=files_to_delete)
    return [combined_pmd]

def fix_ligand(ligand_file, fixed_ligand_file):
    """
    Add explicit hydrogens to carbon atoms and ensure tetravalent nitrogens are assigned a +1 charge
    """
    with Chem.SDMolSupplier(ligand_file, sanitize=False) as supplier:
        mol = supplier[0]
    chem_problems = Chem.DetectChemistryProblems(mol)
    for chem_problem in chem_problems:
        if chem_problem.GetType() == 'AtomValenceException':
            atom = mol.GetAtomWithIdx(chem_problem.GetAtomIdx())
            if atom.GetSymbol() == 'N' and atom.GetFormalCharge() == 0 and atom.GetExplicitValence() == 4:
                atom.SetFormalCharge(1)
    Chem.SanitizeMol(mol)
    fixed_mol = Chem.AddHs(mol, addCoords=True)
    sdwriter = Chem.SDWriter(fixed_ligand_file)
    sdwriter.write(fixed_mol)
    sdwriter.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("USAGE : python parametrize.py protein.pdb ligand.mol\n")
    ligand_file = sys.argv[2]
    protein_file = sys.argv[1]
    system = prepare_system(ligand_file, protein_file, hmr=True, water_ff_str='tip3p.xml')

