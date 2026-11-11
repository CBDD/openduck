"""Single restrained MD run (the sampling phase between SMDs), kept for backward compatibility.

The DUck production loop uses :class:`duck.steps.production.DuckSimulation` directly so that
one OpenMM Context is reused for all cycles; this wrapper builds a one-off simulation.
"""
import os
import sys

from duck.steps.production import DuckSimulation
from duck.utils.duck_stuff import load_complex_pickle


def perform_md(checkpoint_in_file, checkpoint_out_file, csv_out_file, pdb_out_file, dcd_out_file,
               force_constant_ligand=1.0, md_len=1.0, force_constant_chunk=1.0, gpu_id=0, dt=0.002,
               pickle_file='complex_system.pickle', **simulation_kwargs):
    """Run restrained MD of the protein-ligand complex at 300 K.

    The chunk heavy atoms are restrained to the prepared coordinates and the key H-bond is
    kept formed with the flat-bottom restraint of ``dist_md.rst``.

    Args:
        checkpoint_in_file (str): checkpoint to start from.
        checkpoint_out_file (str): checkpoint written at the end. If it already exists the
            simulation is skipped.
        csv_out_file (str): StateDataReporter log.
        pdb_out_file (str): final coordinates.
        dcd_out_file (str): trajectory.
        force_constant_ligand (float): k2 of the H-bond flat-bottom restraint, kcal/mol/A^2.
        md_len (float): simulated time in ns.
        force_constant_chunk (float): chunk heavy-atom restraint, kcal/mol/A^2 (AMBER
            ``restraint_wt``; the original OpenMM protocol used 0.1).
        gpu_id (int, optional): GPU index, ``None`` for CPU.
        dt (float): time step in ps (0.004 implies HMR).
        pickle_file (str): preparation pickle with the structure and the key interaction.
        **simulation_kwargs: extra options for ``DuckSimulation`` (precision, integrator, ...).
    """
    if os.path.isfile(checkpoint_out_file):
        print(f'{checkpoint_out_file} is already calculated, skipping')
        return
    structure, key_interaction = load_complex_pickle(pickle_file)
    simulation = DuckSimulation(structure, key_interaction, gpu_id=gpu_id, hmr=dt >= 0.004, timestep=dt,
                                chunk_restraint_k=force_constant_chunk, hb_restraint_k2=force_constant_ligand,
                                **simulation_kwargs)
    simulation.load_checkpoint(checkpoint_in_file)
    simulation.run_md(md_len, csv_file=csv_out_file, dcd_file=dcd_out_file, pdb_file=pdb_out_file,
                      checkpoint_file=checkpoint_out_file)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: normal_md.py in.chk out.chk out.csv out.pdb out.dcd")
    perform_md(*sys.argv[1:6])
