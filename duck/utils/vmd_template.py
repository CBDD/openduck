"""Helpers to inspect DUck SMD trajectories in VMD (used by scripts/duck_analysis_visualisation.py).

Requires MDAnalysis, which is not a dependency of the main OpenDUck workflows.
"""
from pathlib import Path

import numpy as np


def get_chunk_residue(original_pdb, chunk_pdb, target_residue_name, target_residue_number):
    """Map a residue of the original receptor to the chunk residue whose CA is closest to it.

    Returns:
        tuple: (residue name, residue id) in the chunk.
    """
    import MDAnalysis as mda
    from MDAnalysis.analysis import rms
    pdb_univ = mda.Universe(original_pdb, format='PDB')
    chunk_univ = mda.Universe(chunk_pdb, format='PDB')

    # CA atom of the target residue and all chunk CAs of the same residue type
    target_res_calpha = pdb_univ.select_atoms(f"resname {target_residue_name} and resnum {target_residue_number} and name CA")
    chunk_resis = chunk_univ.select_atoms(f"resname {target_residue_name} and name CA")

    distances = np.array([rms.rmsd(target_res_calpha[0].position, chunk_resis[i].position) for i in range(len(chunk_resis))])
    if distances[np.argmin(distances)] > 0.5:
        print(f'Alignment for {original_pdb} displaced')

    # Target chunk atom should be the closest
    target_chunk_atom = chunk_resis[np.argmin(distances)]
    return (target_chunk_atom.resname, target_chunk_atom.resid)


def vmd_template(pdb_file, dcd_file, ligand_selection, residue_selection):
    """Return a VMD state script showing the SMD trajectory with the ligand and the interacting residue."""
    lig_and_res_sel = "{" + ligand_selection + " and (" + residue_selection + ")}"
    ligand_selection = "{"+ ligand_selection + "}"
    residue_selection = "{"+ residue_selection + "}"

    print(lig_and_res_sel)

    vmd_str = """
    #!/usr/local/bin/vmd
    # VMD script written by save_state $Revision: 1.47 $
    # VMD version: 1.9.3
    set viewplist {}
    set fixedlist {}
    proc vmdrestoremymaterials {} {
      set mlist { Opaque Transparent BrushedMetal Diffuse Ghost Glass1 Glass2 Glass3 Glossy HardPlastic MetallicPastel Steel Translucent Edgy EdgyShiny EdgyGlass Goodsell AOShiny AOChalky AOEdgy BlownGlass GlassBubble RTChrome }
      set mymlist [material list]
      foreach mat $mlist {
        if { [lsearch $mymlist $mat] == -1 } { 
          material add $mat
        }
      }
    }
    vmdrestoremymaterials
    """

    vmd_str += """
    # Display settings

    mol new {0} type pdb first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all
    mol addfile {1} type dcd first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all

    mol delrep 0 top
    mol representation Licorice 0.300000 12.000000 12.000000
    mol color Name
    mol selection {2}
    mol material Opaque
    mol addrep top
    mol selupdate 0 top 0
    mol colupdate 0 top 0
    mol scaleminmax top 0 0.000000 0.000000
    mol smoothrep top 0 0
    """.format(pdb_file, dcd_file, ligand_selection)

    vmd_str += """
    mol drawframes top 0 {now}

    mol representation Lines 1.000000
    mol color Name
    mol selection {protein}
    mol material Opaque
    mol addrep top
    mol selupdate 1 top 0
    mol colupdate 1 top 0
    mol scaleminmax top 1 0.000000 0.000000
    mol smoothrep top 1 0
    mol drawframes top 1 {now}
    """

    vmd_str += """
    mol representation Licorice 0.300000 12.000000 12.000000
    mol color Name
    mol selection {0}
    mol material Opaque
    mol addrep top
    mol selupdate 2 top 0
    mol colupdate 2 top 0
    mol scaleminmax top 2 0.000000 0.000000
    mol smoothrep top 2 0
    """.format(residue_selection)

    vmd_str += """
    mol drawframes top 2 {now}
    mol representation HBonds 3.000000 20.000000 10.000000
    mol color Name
    """

    vmd_str += """
    mol selection { not water}
    mol material Opaque
    mol addrep top
    mol selupdate 3 top 0
    mol colupdate 3 top 0
    mol scaleminmax top 3 0.000000 0.000000
    mol smoothrep top 3 0
    """

    vmd_str += """
    mol drawframes top 3 {now}
    """

    vmd_str += "mol rename top {}".format(Path(pdb_file).name)
    vmd_str += """
    lappend viewplist [molinfo top]
    set topmol [molinfo top]
    # done with molecule 0
    foreach v $viewplist {
      molinfo $v set {center_matrix rotate_matrix scale_matrix global_matrix} $viewpoints($v)
    }
    foreach v $fixedlist {
      molinfo $v set fixed 1
    }
    unset viewplist
    unset fixedlist
    mol top $topmol
    unset topmol

    label textsize 1.0
    pbc wrap -center com -centersel "protein" -compound residue -all 
    pbc wrap -center com -centersel "protein" -compound residue -all 
    """
    return vmd_str
