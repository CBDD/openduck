"""Location of the key interaction (receptor atom - ligand atom) that DUck steers.

The receptor atom is given as ``chain_resname_resid_atomname`` (e.g. ``A_LYS_311_N``; the chain
may be empty, ``_LYS_311_N``). The ligand atom is the closest ligand (residue ``UNL``) heavy
atom of the accepted elements (N and O by default).
"""
import json
import math
import operator
import os
import pickle
import sys

import parmed
from parmed.geometry import distance2

LIGAND_RESNAME = 'UNL'


def parse_interaction(res_atom):
    """Split ``chain_resname_resid_atomname`` into (chain, resname, resid, atomname)."""
    parts = res_atom.split('_')
    if len(parts) != 4:
        raise ValueError(f'The interaction "{res_atom}" is not well formatted. '
                         'Use chain_resname_resid_atomname (e.g. A_LYS_311_N).')
    chain, res_name, res_number, atom_name = parts
    return chain, res_name, int(res_number), atom_name


def check_same(atom, chain, res_name, res_number, atom_name):
    '''
    Check if the atom object corresponds to the chain, residue_name, residue number and atom name
    '''
    return (atom.residue.name == res_name and atom.residue.number == res_number and atom.name == atom_name
            and atom.residue.chain == chain)


def is_lig(atom):
    """True for ligand heavy atoms."""
    return atom.residue.name == LIGAND_RESNAME and atom.atomic_number > 1


def is_good_atomtype(atom, elements):
    """True if the atomic number of ``atom`` is in ``elements`` (list of ints, e.g. [7, 8])."""
    elements = list(elements)
    for e in elements:
        if not isinstance(e, int):
            raise ValueError(f'Element {e} specified in the accepted elements is not an integer.')
    return atom.atomic_number in elements


def find_atom(res_atom=None, prot_file=None, combined_pmd=None):
    '''
    Find the receptor atom ``res_atom`` (``chain_resname_resid_atomname``) in ``prot_file`` and
    sort the atoms of ``combined_pmd`` by their distance to it.

    Returns:
        tuple: ([(atom index in combined_pmd, squared distance), ...] sorted by distance,
        the parmed atom found in ``prot_file``).
    '''
    chain, res_name, res_number, atom_name = parse_interaction(res_atom)
    protein = parmed.load_file(prot_file, structure=True)
    for atom in protein.atoms:
        if check_same(atom, chain, res_name, res_number, atom_name):
            prot_atom = atom
            break
    else:
        raise ValueError(f'The specified interaction atom ({res_atom}) was not found in {prot_file}.')
    distance_atom_1 = [(x.idx, distance2(x, prot_atom)) for x in combined_pmd.atoms]
    distance_atom_1.sort(key=operator.itemgetter(1))
    return distance_atom_1, prot_atom


def find_result(res_atom=None, prot_file=None, combined_pmd=None, accepted_lig_elements=(7, 8)):
    '''
    Find the system atom matching the receptor atom and the closest ligand atom of the accepted elements.

    Returns:
        tuple: (receptor atom index, ligand atom index, [receptor idx, ligand idx, distance in A],
        squared distance)
    '''
    distance_atom_1, prot_atom = find_atom(res_atom, prot_file, combined_pmd)
    distance_atom_2 = [(x.idx, distance2(x, prot_atom)) for x in combined_pmd.atoms
                       if is_lig(x) and is_good_atomtype(x, accepted_lig_elements)]
    if not distance_atom_2:
        raise ValueError(f'No ligand atom with atomic number in {list(accepted_lig_elements)} was found.')
    distance_atom_2.sort(key=operator.itemgetter(1))
    index_one = distance_atom_1[0][0]
    index_two = distance_atom_2[0][0]
    out_res = [index_one, index_two, math.sqrt(distance_atom_2[0][1])]
    return index_one, index_two, out_res, distance_atom_2[0][1]


def find_interaction(res_atom=None, prot_file=None, lig_HB_elements=(7, 8), pickle_file='complex_system.pickle'):
    '''
    Find the key interaction in the prepared system.

    Args:
        res_atom (str): receptor atom, ``chain_resname_resid_atomname`` (e.g. ``A_LYS_311_N``).
        prot_file (str): receptor PDB with the numbering used in ``res_atom``.
        lig_HB_elements (sequence of int): accepted elements of the ligand atom.
        pickle_file (str): preparation pickle with the parmed structure of the system.

    Returns:
        list: [receptor atom index, ligand atom index, distance in A] (0-based indices in the system).
        The result is also written to ``indice.text``.
    '''
    with open(pickle_file, 'rb') as fh:
        combined_pmd = pickle.load(fh)[0]
    index_one, index_two, out_res, dist = find_result(res_atom, prot_file, combined_pmd,
                                                      accepted_lig_elements=lig_HB_elements)
    with open('indice.text', 'w') as out_f:
        out_f.write(json.dumps(out_res))
    return [index_one, index_two, math.sqrt(dist)]


def find_interaction_amber_input(combined_pmd, chunk_file, res_atom):
    '''
    Find the key interaction in an AMBER-prepared system, mapping the residue through the chunk file.

    The chain is ignored, as chain information is lost when chunking.

    Args:
        combined_pmd (parmed.Structure): system loaded from the AMBER topology and coordinates.
        chunk_file (str): receptor chunk (PDB, or MOE-style mol2) with the original residue numbering.
        res_atom (str): receptor atom, ``chain_resname_resid_atomname``.

    Returns:
        list: [receptor atom index, ligand atom index, distance in A]; also written to ``indice.txt``.
    '''
    chunk = parmed.load_file(chunk_file, structure=True)
    if chunk_file.split('.')[-1] == 'mol2':
        rename_mol2_residues(chunk)
    _, res_name, res_number, atom_name = parse_interaction(res_atom)
    chain = ''

    for atom in chunk.atoms:
        if check_same(atom, chain, res_name, res_number, atom_name):
            chunk_atom = atom
            break
    else:
        raise ValueError('Cannot find the interaction atom in the chunk. Check residue labelling')

    # chunk residue number -> residue number in the prepared system
    chunk_dict = {cr.number: cd.number for cd, cr in zip(combined_pmd.residues, chunk.residues)}
    for atom in combined_pmd.atoms:
        if check_same(atom, chain, res_name, chunk_dict[chunk_atom.residue.number], atom_name):
            target_protein_atom = atom
            break
    else:
        raise ValueError('Cannot find the interaction atom in the prepared system. Check residue labelling')

    index_one = target_protein_atom.idx
    distance_atom_2 = sorted(((x.idx, distance2(x, target_protein_atom)) for x in combined_pmd.atoms if is_lig(x)),
                             key=operator.itemgetter(1))
    index_two, dist = distance_atom_2[0]
    with open('indice.txt', 'w') as out_f:
        out_f.write(json.dumps([index_one, index_two, math.sqrt(dist)]))
    return [index_one, index_two, math.sqrt(dist)]


def clean_up_files(path='./', files_to_delete=()):
    """Delete the files of ``path`` whose names are in ``files_to_delete`` (missing files are ignored)."""
    for file in os.listdir(path):
        if file in files_to_delete:
            os.remove(os.path.join(path, file))


def rename_mol2_residues(parmed_protein):
    """Split MOE-style mol2 residue names (e.g. ``LYS311``) into name and number.

    Crashes with altlocs currently.
    """
    for r in parmed_protein.residues:
        res_name = r.name[:3]
        true_res_number = int(r.name[3:])
        r.name = res_name
        r.number += true_res_number - r.number


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit('Usage: cal_ints.py chain_resname_resid_atomname receptor.pdb')
    protid, ligid, dist = find_interaction(sys.argv[1], sys.argv[2])
    print('receptor ID %s\nligand ID %s\ndistance %s' % (protid + 1, ligid + 1, dist))
