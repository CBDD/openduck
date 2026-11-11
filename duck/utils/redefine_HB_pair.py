#!/usr/bin/env python
"""Redefine the ligand atom of the steered H-bond and update the AMBER restraint files in place.

For each DUck folder matching ``--pattern``, the ligand atom closest to the receptor interaction
atom (within ``--hb-distance-cutoff`` and, optionally, close to pocket points) is selected and,
with ``--modify``, the ``iat`` atom indices of ``dist_duck.rst`` and ``dist_md.rst`` are rewritten
(a ``.BAK`` copy is kept).
"""
import argparse
import glob
import math
import os
import shutil
from fileinput import FileInput

import parmed as pmd


def find_protein_atom(system, interaction_string):
    '''Find the receptor atom ``chain_resname_resid_atomname``; resid is the 1-based residue index of the system.'''
    if len(interaction_string.split('_')) != 4:
        raise ValueError(f'The given interaction ({interaction_string}) is not well formated. It needs to be chain_resname_resid_atomname.\n')
    chain, resname, resid, atomname = interaction_string.split('_')
    if chain == '':
        chain = '1'  # parmed assigns a chain if there is none?
    for r in system.residues:
        if r.name == resname and r.idx == int(resid) - 1 and r.chain == chain:
            for atom in r.atoms:
                if atom.name == atomname:
                    return atom
    raise ValueError('Could not find protein atom in the given system.')


def find_ligand(system, ligand_name='UNL'):
    for r in system.residues:
        if r.name == ligand_name:
            return r
    raise ValueError('Could not find the ligand in the system with the given name.')


def find_interacting_atoms(ref, ligand, allowed_elements=(7, 8), distance_threshold=3.5, pocket_points=None, pocket_radius=1,
                           exclude=False):
    '''Ligand atoms of the allowed elements within ``distance_threshold`` of ``ref`` (and of the pocket points).

    Returns:
        list: [(atom, distance), ...]
    '''
    r = []
    ref_coords = (ref.xx, ref.xy, ref.xz)
    for atom in ligand.atoms:
        lig_atom_coords = (atom.xx, atom.xy, atom.xz)
        dist = math.dist(ref_coords, lig_atom_coords)
        if atom.element not in allowed_elements or dist > distance_threshold:
            continue
        if pocket_points:
            in_pocket = any(math.dist((p.xx, p.xy, p.xz), lig_atom_coords) <= pocket_radius for p in pocket_points.atoms)
            if in_pocket != exclude:
                r.append((atom, dist))
        else:
            r.append((atom, dist))
    if len(r) == 0:
        raise ValueError('No atom fits the interacting criteria.')
    return r


def modify_DUck_HB_atom_ids(prot_atom, lig_atom, files_to_mod=('dist_duck.rst', 'dist_md.rst'), backup=True):
    '''Rewrite the ``iat=`` atom indices of the AMBER restraint files.'''
    new_prot_idx = str(int(prot_atom.idx) + 1)
    new_lig_idx = str(int(lig_atom.idx) + 1)
    for file in files_to_mod:
        if backup and not os.path.exists(f'{file}.BAK'):
            shutil.copyfile(file, f"{file}.BAK")
        with FileInput(files=file, inplace=True) as f:
            for line in f:
                line = line.rstrip()
                if 'iat' in line:
                    line = ' '.join(f'iat={new_prot_idx},{new_lig_idx},' if 'iat' in part else part for part in line.split())
                print(line)


def find_and_modify_HB_interaction(system_pmd, interaction, ligand_name, distance_threshold=3.5, pocket_points=None,
                                   pocket_points_radius=0, modify=False, files_to_modify=('dist_duck.rst', 'dist_md.rst'),
                                   allowed_elements=(7, 8), exclude=False):
    '''Select the closest valid ligand atom and optionally update the restraint files.

    Returns:
        tuple: (ligand atom, distance in A)
    '''
    prot_atom = find_protein_atom(system_pmd, interaction)
    ligand = find_ligand(system_pmd, ligand_name)
    int_atoms = sorted(find_interacting_atoms(prot_atom, ligand, allowed_elements=allowed_elements, distance_threshold=distance_threshold,
                                              pocket_points=pocket_points, pocket_radius=pocket_points_radius, exclude=exclude),
                       key=lambda x: x[1])
    good_lig_atom = int_atoms[0]
    if modify:
        modify_DUck_HB_atom_ids(prot_atom, good_lig_atom[0], files_to_mod=files_to_modify, backup=True)
    return good_lig_atom


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Helper tool to redefine the hydrogen bond ligand atom and modify the amber inputs for DUck.')
    parser.add_argument('--system-pdb', '-s', type=str, required=True, help='System with the protein and ligand with the same numeration of your topology. The atom ID will be given from the numeration in this pdb.')
    parser.add_argument('--ligand-name', '-n', type=str, default='UNL', help='Ligand residue name in the pdb. Default: UNL')
    parser.add_argument('--interaction', '-i', type=str, required=True, help='Interaction string of the protein atom')
    parser.add_argument('--ligand-hb-elements', '-e', default=[7, 8], type=int, nargs='+', help='Control which elements are accepted in the ligand to define the steering interaction. Specify using the atomic number separated with a space. Default is 7 and 8 (nitrogen and oxygen)')
    parser.add_argument('--hb-distance-cutoff', '-d', default=3.5, type=float, help='Redefine the minimum distance needed to choose the ligand interacting atom.')
    parser.add_argument('--pocket-points', '-c', default=None, type=str, help='Coordinates of the points defining the allowed area of interacting ligand atoms in sdf, pdb or pharmacophore (ph4) format.')
    parser.add_argument('--pocket-points-radius', '-r', default=1, type=float, help='Tolerance radius of the pocket points to allow the interacting ligand atom.')
    parser.add_argument('--modify', '-m', default=False, action='store_true', help='Toggle to modify the amber input files with the new atom ID.')
    parser.add_argument('--exclude', '-x', default=False, action='store_true', help='Reverse pocket-point functionality. When flagged pocket-points define an exclusion where the interacting ligand atom cannot be.')
    parser.add_argument('--pattern', '-p', type=str, default='.', help='Wildcard pattern to find folders with DUck data.')
    args = parser.parse_args()

    pocket_points = pmd.load_file(args.pocket_points) if args.pocket_points else None
    basedir = os.getcwd()
    for folder in glob.glob(args.pattern):
        os.chdir(os.path.join(basedir, folder))
        sys_pmd = pmd.load_file(args.system_pdb)
        good_lig_atom, dist = find_and_modify_HB_interaction(sys_pmd, args.interaction, args.ligand_name,
                                                             distance_threshold=args.hb_distance_cutoff, pocket_points=pocket_points,
                                                             pocket_points_radius=args.pocket_points_radius, modify=args.modify,
                                                             allowed_elements=args.ligand_hb_elements, exclude=args.exclude)
        print(f'{folder}\t{good_lig_atom.residue.chain}_{good_lig_atom.residue.name}_{good_lig_atom.idx + 1}_{good_lig_atom.name}\t{dist}')
        os.chdir(basedir)
