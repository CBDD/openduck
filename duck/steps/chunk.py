"""Reduction of the receptor to a pocket ("chunk") around the key interaction.

Residues within ``cutoff`` A of the receptor interaction atom are selected, single-residue gaps
between selected residues are closed, the segment ends are capped (ACE/NME) and the chunk is
protonated with tleap (ff14SB), detecting disulfide bridges.
"""
import os
import subprocess

import parmed
from parmed.geometry import distance2

from duck.utils.cal_ints import find_atom, clean_up_files
from duck.utils.resources import data_path


def return_tleap(prot_protein_chunk, out_save, disulfides=()):
    '''
    Generate the tleap input that protonates the chunk and creates the disulfide bonds.
    '''
    param_f_path = data_path('parameters', 'tleap', 'leaprc.ff14SB.redq')
    bond_list = [f"bond mol.{num[0]}.SG mol.{num[1]}.SG " for num in disulfides]
    return (f"source {param_f_path}\n"
            f"mol = loadpdb {prot_protein_chunk}\n"
            + "\n".join(bond_list)
            + f"\nsavepdb mol {out_save}\n"
            "quit")


def do_tleap(prot_protein_chunk, out_save, disulfides=()):
    '''
    Launch tleap from ambertools (log in chunk_leap.log).
    '''
    with open("run.tleap", "w") as out_f:
        out_f.write(return_tleap(prot_protein_chunk, out_save, disulfides))
    with open("chunk_leap.log", "w") as log:
        subprocess.run(["tleap", "-f", "run.tleap"], stdout=log, stderr=subprocess.STDOUT, check=False)
    if not os.path.isfile(out_save):
        raise RuntimeError(f'tleap did not write {out_save}; check chunk_leap.log.')

def add_cap(x, atom_set):
    '''Add atom ``x`` and its bonded partners (the capping atoms) to ``atom_set``.'''
    atom_set.add(x.idx)
    # Add the capping atoms
    for y in x.bond_partners:
        atom_set.add(y.idx)
    return atom_set

def find_neighbour_residues(residues):
    """
    Given a list of residues, returns information about the neighboring residues and atoms in the molecule.

    Args:
        residues (list of Residue objects): a list of residues in the molecule

    Returns:
        A tuple of three elements:
        - A dictionary where each residue in the input list is a key, and the corresponding value is a set of all residues that share a bond with at least one atom in that residue.
        - A dictionary where each residue in the input list is a key, and the corresponding value is a set of all residues that share a bond with at least one atom in a residue that shares a bond with at least one atom in the key residue.
        - A set of all atoms that are within 3 Angstroms of any atom in any residue in the input list.
    """
    single_joins = {}
    double_joins = {}
    atom_set = set()
    for resid in residues:
        residue_set = set()
        double_res_set = set()
        for atom in resid.atoms:
            for bond_1 in atom.bonds:
                if bond_1.measure() > 3.0:
                    print(
                        "Skipping too long bond (" + str(bond_1.measure()) + ") :  ",
                        bond_1.atom1,
                        bond_1.atom2,
                    )
                    continue
                if bond_1.atom1 == atom:
                    x = bond_1.atom2
                else:
                    x = bond_1.atom1
                residue_set.add(x.residue)
                for atom_two in x.residue.atoms:
                    for bond_2 in atom_two.bonds:
                        if bond_2.measure() > 3.0:
                            print(
                                "Skipping too long bond ("
                                + str(bond_2.measure())
                                + ") :  ",
                                bond_2.atom1,
                                bond_2.atom2,
                            )
                            continue
                        if bond_2.atom1 == atom:
                            conn_two = bond_2.atom2
                        elif bond_2.atom1 == atom_two:
                            conn_two = bond_2.atom2
                        else:
                            conn_two = bond_2.atom1
                        double_res_set.add(conn_two.residue)
                atom_set = add_cap(x, atom_set)
        double_res_set.difference_update(residue_set)
        single_joins[resid] = residue_set
        double_joins[resid] = double_res_set
    return single_joins, double_joins, atom_set

def find_neighbours(residues):
    """
    Given a set of residues, returns a all residues that share a bond with at least one atom in any of the input residues,
    as well as a set of all atoms that are within 3 Angstroms of any atom in any of the input residues.

    Args:
        residues (set of Residue objects): a set of residues in the molecule

    Returns:
        A tuple of two elements:
        - The residues added to ``residues`` (which is modified in place) to close single-residue gaps:
          residues bonded to two of the input residues, directly or through a second-degree bond.
        - The set of capping atoms (see :func:`find_neighbour_residues`).
    """
    single_joins, double_joins, atom_set = find_neighbour_residues(residues)
    new_residues = set()
    for resid_one in single_joins:
        for resid_two in single_joins:
            if resid_one == resid_two:
                continue
            single_join = single_joins[resid_two].intersection(single_joins[resid_one])
            # If it's just a terminal residue
            if resid_one not in single_joins[resid_two]:
                double_join = double_joins[resid_two].intersection(
                    single_joins[resid_one]
                )
            else:
                double_join = set()
            new_set = single_join.union(double_join)
            for new_res in new_set:
                residues.add(new_res)
                new_residues.add(new_res)
    return new_residues, atom_set

def convert_to_ace_nme(subset):
    """
    Given a subset of a molecule, converts some residues to ACE or NME, depending on their atom types and names. Specifically:
    - If a residue has exactly three atoms (CA, C, and O) with these names, its name is changed to ACE and its CA atom is renamed to CH3.
    - If a residue has exactly three atoms (CA, CD, and N) with these names, its name is changed to NME, its CA atom is renamed to CH3, and its CD atom is removed from the molecule.
    - If a residue has exactly two atoms (CA and N) with these names, its name is changed to NME and its CA atom is renamed to CH3.
    - If a residue has exactly two atoms (CB and SG) with these names, it is removed from the molecule.
    
    Any atoms or residues that are removed during this process are excluded from the output subset.

    Args:
        subset (parmed.Structure): the selected atoms of the chunk.

    Returns:
        A modified version of the input subset, where some residues may have been renamed to ACE or NME, and some atoms or residues may have been removed.
    """
    remove_res_ids = []
    remove_atom_ids = []
    for residue in subset.residues:
        if len(residue) == 3:
            if set([x.name for x in residue.atoms]) == set(["CA", "C", "O"]):
                residue.name = "ACE"
                for atom in residue.atoms:
                    if atom.name == "CA":
                        atom.name = "CH3"
            # If it's a proline
            if set([x.name for x in residue.atoms]) == set(["CA", "CD", "N"]):
                residue.name = "NME"
                for atom in residue.atoms:
                    if atom.name == "CA":
                        atom.name = "CH3"
                    if atom.name == "CD":
                        remove_atom_ids.append(str(atom.idx + 1))
        elif len(residue) == 2:
            if set([x.name for x in residue.atoms]) == set(["CA", "N"]):
                residue.name = "NME"
                for atom in residue.atoms:
                    if atom.name == "CA":
                        atom.name = "CH3"
            elif set([x.name for x in residue.atoms]) == set(["CB", "SG"]):
                remove_res_ids.append(str(residue.idx + 1))
    if remove_atom_ids != []:
        subset = subset["!(@" + ",".join(remove_atom_ids) + ")"]
    if remove_res_ids != []:
        subset = subset["!(:" + ",".join(remove_res_ids) + ")"]
    return subset

def remove_prot_buffers_alt_locs(prot_file):
    """
    Cleans up a protein structure PDB file by removing hydrogen atoms, solvents, and buffers. 
    Writes the modified protein structure to a new PDB file with a consistent alternate location convention. 

    Args:
        prot_file (str): Path to the protein structure PDB file.

    Returns:
        str: Path to the new PDB file with the cleaned-up protein structure.
    """

    output_file = "no_buffer_altlocs.pdb"
    solvents = ["NA", "CL", "SO4", "EDO", "FMT", "PO4", "DMS", "EPE"]
    # Remove hydrogens and solvents and buffers
    protein = parmed.load_file(prot_file)["!(:HOH," + ",".join(solvents) + ")"][
        "!(:=@H=)"
    ]
    protein.write_pdb(output_file, altlocs="first")
    return output_file

def find_disulfides(input_file, threshold=6.2):
    '''Given a PDB file, find the cysteine residues to build disulfide bonds.

    Bridged cysteines are renamed to CYX and ``input_file`` is overwritten.

    Args:
    input_file (str): Path to the protein structure PDB file.
    threshold (float): Squared SG-SG distance threshold (A^2) to define a bridge (6.2 A^2 ~ 2.5 A).

    Returns:
        list: List of tuples for the two residue numbers in each disulfide bond detected.
    '''
    structure = parmed.load_file(input_file)
    sulfurs = [x for x in structure.atoms if x.residue.name == "CYS" and x.name == "SG"]
    disulfides = []
    for atom_one in sulfurs:
        for atom_two in sulfurs:
            if atom_one.idx >= atom_two.idx:
                continue
            dist = distance2(atom_one, atom_two)
            if dist < threshold:
                atom_one.residue.name = "CYX"
                atom_two.residue.name = "CYX"
                disulfides.append((atom_one.residue.number, atom_two.residue.number))
    structure.write_pdb(input_file)
    return disulfides

def chunk_with_amber(
    mol_file="MURD-x0349.mol",
    prot_file="MURD-x0349_apo.pdb",
    interaction="A_LYS_311_N",
    out_save="protein_out.pdb",
    cutoff=9.0,
    orig_prot="MURD-x0349_apo.pdb",
):
    '''
    Chunk the protein into a smaller subset based on a cutoff radius around the given interaction.

    Args:
        mol_file (str): Ligand file. Unused (the chunk is centred on the interaction atom); kept for compatibility.
        prot_file (str): Path to the .pdb file containing the protein structure.
        interaction (str): Interaction to consider for chunking, in the format of "chain_residue_atom".
        out_save (str): Path to save the output .pdb file.
        cutoff (float): Cutoff distance (in angstroms) for selecting atoms in the protein.
        orig_prot (str): Path to the original .pdb file used to generate the protein.

    Returns a list containing the path to the output .pdb file.
    '''
    protein = parmed.load_file(prot_file)
    # get these details
    atom_idx, prot_atom = find_atom(interaction, orig_prot, protein)
    # use parmed to mask whole resudies (<:) within a cutoff distance of atom (@ atom_idx)
    mask = parmed.amber.AmberMask(
        protein, "@" + str(atom_idx[0][0] + 1) + "<:" + str(cutoff)
    )
    #select only the residues and only once each
    residues = set(
        [protein.atoms[i].residue for i, x in enumerate(mask.Selection()) if x == 1]
    )
    residues = set([x for x in residues if x.name != "UNL"])
    # # Find all the residues that are connected to two residues in this list of residues.
    new_residues, atom_set = find_neighbours(residues)
    # Collect the atoms
    atom_idx = []
    for res in residues:
        atom_idx.extend([x.idx for x in res.atoms if x.altloc in ["", "A"]])
    atom_idx.extend(atom_set)
    subset = protein[atom_idx]
    subset.write_pdb(out_save.replace(".pdb", "_no_ace_nme.pdb"))
    subset = convert_to_ace_nme(
        parmed.load_file(out_save.replace(".pdb", "_no_ace_nme.pdb"))
    )
    subset.write_pdb(out_save)
    add_ter_records(out_save, out_save)
    return [out_save]

def add_ter_records(input_file, output_file):
    '''
    Add 'TER' tag to pdb where the NME residue appears

    Args
        input_file (str): input chunked protein in .pdb to add TER tags
        output_file (str): output pdb file with the TER tags
    '''
    with open(input_file) as fh:
        lines = fh.readlines()
    with open(output_file, "w") as output_f:
        for line in lines:
            output_f.write(line)
            if "NME" in line:
                output_f.write("TER\n")
    return [output_file]

def duck_chunk(prot_file, mol_file, interaction, cutoff, output_name = 'protein_out.pdb', ignore_buffers=False, keep_all_files=False):
    """
    Performs chunking of a protein structure in the presence of a small molecule within a cutoff radius.

    Args:
        prot_file (str): Path to the protein structure PDB file.
        mol_file (str): Path to the small molecule file.
        interaction (str): Receptor interaction atom, ``chain_resname_resid_atomname``.
        cutoff (float): Residues with any atom within this distance (A) of the interaction atom are kept.
        output_name (str, optional): Name of the output protein PDB file after chunking and protonation. Defaults to 'protein_out.pdb'.
        ignore_buffers (bool, optional): Keep waters, ions, buffers, hydrogens and alternative locations (they are removed by default). Defaults to False.
        keep_all_files (bool, optional): Keep the intermediate files. Defaults to False.

    Returns:
        str: Path to the output protein PDB file after chunking and protonation.
    """
    orig_file = prot_file

    chunk_protein_prot = os.path.join(os.path.dirname(output_name), f'protonated_{os.path.basename(output_name)}')
    # Do the removal of buffer mols and alt locs
    if not ignore_buffers:
        prot_file = remove_prot_buffers_alt_locs(prot_file)
    # Do the chunking and the protonation
    # Chunk
    chunk_with_amber(mol_file,prot_file,interaction,output_name,cutoff,orig_file)
    # Protontate
    disulfides = find_disulfides(output_name)
    do_tleap(output_name, chunk_protein_prot, disulfides)
    if not keep_all_files:
        clean_up_files(files_to_delete=[output_name.replace(".pdb", "_no_ace_nme.pdb"), 'chunk_leap.log', 'leap.log', 'run.tleap', 'no_buffer_altlocs.pdb'])
    return chunk_protein_prot

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Duck chunk')
    parser.add_argument('-r', '--receptor', type=str)
    parser.add_argument('-l', '--ligand', type=str)
    parser.add_argument('-i', '--interaction', type=str)
    parser.add_argument('-c', '--cutoff', type=float)
    parser.add_argument('-o', '--output', type=str, default='protein_out.pdb')
    parser.add_argument('-b', '--ignore-buffers', action='store_true')
    args = parser.parse_args()
    duck_chunk(args.receptor, args.ligand, args.interaction, args.cutoff, args.output, args.ignore_buffers)
