#!/usr/bin/env
import argparse
import parmed as pmd
import glob
import math
import os
import shutil
from fileinput import FileInput

def find_protein_atom(system, interaction_string):

    if len(interaction_string.split('_')) != 4:
        raise ValueError(f'The given interaction ({interaction_string}) is not well formated. It needs to be chain_resname_resid_atomname.\n')
    chain, resname, resid, atomname = interaction_string.split('_')
    if chain == '': chain = '1' # parmed assigns a chain if there is none?
    for r in system.residues:
        if r.name == resname and r.idx == int(resid)-1 and r.chain == chain:
            for atom in r.atoms:
                if atom.name == atomname:
                    return atom
    raise ValueError('Could not find protein atom in the given system.')

def find_ligand(system, ligand_name = 'UNL'):
    for r in system.residues:
        if r.name == ligand_name:
            return r
    raise ValueError('Could not find the ligand in the system with the given name.')

def find_interacting_atoms(ref, ligand, allowed_elements = [7,8], distance_threshold = 3.5, pocket_points=None, pocket_radius=1):
    r = []
    ref_coords = (ref.xx, ref.xy, ref.xz)
    for atom in ligand.atoms:
        lig_atom_coords = (atom.xx, atom.xy, atom.xz)
        #distance to reference
        dist = math.dist(ref_coords, lig_atom_coords)
        if atom.element in allowed_elements and dist <= distance_threshold:
            # in pocket
            if pocket_points:
                for pocket_atom in pocket_points:
                    pocket_atom_coords = (pocket_atom.xx, pocket_atom.xy,pocket_atom.xz)
                    if math.dist(pocket_atom_coords, lig_atom_coords) <= pocket_radius:
                        r.append((atom, dist))
                        break
            else:
                r.append((atom, dist))
    if len(r) == 0: raise ValueError('No atom fits the interacting criteria.')
    return r
def modify_DUck_HB_atom_ids(prot_atom, lig_atom, files_to_mod = ['dist_duck.rst', 'dist_md.rst'], backup=True):
    new_prot_idx = str(int(prot_atom.idx)+1)
    new_lig_idx = str(int(lig_atom.idx)+1)
    for file in files_to_mod:
        if backup and not os.path.exists(f'{file}.BAK'):
            shutil.copyfile(file, f"{file}.BAK")
        with FileInput(files=file, inplace=True) as f:
            for line in f:
                line = line.rstrip()
                if 'iat' in line:
                    line_parts = line.split()
                    mod_parts = []
                    for part in line_parts:
                        if 'iat' in part:
                            part = f'iat={new_prot_idx},{new_lig_idx},'
                        mod_parts.append(part)
                    print(' '.join(mod_parts))
                else: print(line)
                    
def find_and_modify_HB_interaction(system_pmd, interaction, ligand_name, distance_threshold=3.5, pocket_points =None, pocket_points_radius=0, modify=False, files_to_modify = ['dist_duck.rst', 'dist_md.rst']):
    prot_atom = find_protein_atom(system_pmd, interaction)
    ligand = find_ligand(sys_pmd, ligand_name)
    int_atoms = sorted(find_interacting_atoms(prot_atom, ligand, distance_threshold=distance_threshold, pocket_points=pocket_points, pocket_radius=pocket_points_radius),key=lambda x:x[1])
    good_lig_atom = int_atoms[0]
    if args.modify: modify_DUck_HB_atom_ids(prot_atom, good_lig_atom[0], files_to_mod = files_to_modify, backup=True)
    return good_lig_atom

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Helper tool to redefine the hydrogen bond ligand atom and modify the amber inputs for DUck.')
    parser.add_argument('--system-pdb', '-s', type=str, required=True, help='System with the protein and ligand with the same numeration of your topology. The atom ID will be given from the numeration in this pdb.')
    parser.add_argument('--ligand-name', '-n', type=str, default='UNL', help='Ligand residue name in the pdb. Default: UNL')
    parser.add_argument('--interaction', '-i', type=str, required=True, help='Interaction string of the protein atom')
    parser.add_argument('--ligand-hb-elements', '-e', default=[7,8], type=int, nargs='+', help='Control which elements are accepted in the ligand to define the steering interaction. Specify using the atomic number separated with a space. Default is 7 and 8 (nitrogen and oxygen)')
    parser.add_argument('--hb-distance-cutoff', '-d', default=3.5, type=float, help='Redefine the minimum distance needed to choose the ligand interacting atom.')
    parser.add_argument('--pocket-points', '-c', default=None, type=str, help='Coordinates of the points defining the allowed area of interacting ligand atoms in sdf, pdb or pharmacophore (ph4) format.')
    parser.add_argument('--pocket-points-radius', '-r', default=1, type=float, help='Tolerance radius of the pocket points to allow the interacting ligand atom.')
    parser.add_argument('--modify', '-m', default=False, action='store_true', help='Toggle to modify the amber input files with the new atom ID.')
    parser.add_argument('--exclude', '-x', default=False, action='store_true', help='Reverse pocket-point functionality. When flagged pocket-points define an exclusion where the interacting ligand atom cannot be.')
    parser.add_argument('--pattern', '-p', type=str, default='.', help='Wildcard pattern to find folders with DUck data.')

    args = parser.parse_args()
    if args.pocket_points: args.pocket_points = pmd.load_file(args.pocket_points)
    basedir = os.getcwd()
    for dir in glob.glob(args.pattern):
        os.chdir(f'{basedir}/{dir}')
        sys_pmd = pmd.load_file(args.system_pdb)
        good_lig_atom, dist = find_and_modify_HB_interaction(sys_pmd, args.interaction, args.ligand_name, distance_threshold=args.distance_cutoff, pocket_points =args.pocket_points, pocket_points_radius=args.pocket_points_radius )
        print(f'{dir}\t{good_lig_atom.residue.chain}_{good_lig_atom.residue.name}_{good_lig_atom.idx+1}_{good_lig_atom.name}\t{dist}')
        os.chdir(basedir)
        