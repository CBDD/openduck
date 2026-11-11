"""Single steered MD (SMD) run, kept for backward compatibility.

See :mod:`duck.steps.production` for the description of the steering protocol. The DUck
production loop uses :class:`duck.steps.production.DuckSimulation` directly; this wrapper
builds a one-off simulation from a checkpoint.
"""
import os
import sys

from duck.steps.production import DuckSimulation, speed_from_init_velocity
from duck.utils.compat import unit as u
from duck.utils.duck_stuff import load_complex_pickle


def run_steered_md(temperature, checkpoint_in_file, csv_out_file, dat_out_file, pdb_out_file, traj_out_file,
                   startdist, spring_constant=50, force_constant_chunk=1.0, init_velocity=0.00001, gpu_id=0,
                   hmr=False, pickle_file='complex_system.pickle', **simulation_kwargs):
    """Steer the key H-bond from ``startdist`` to ``startdist + 2.5`` A and write the work profile.

    Args:
        temperature (float or Quantity): temperature in K.
        checkpoint_in_file (str): checkpoint to start from (end of the preceding MD).
        csv_out_file (str): StateDataReporter log.
        dat_out_file (str): work profile in the AMBER ``jar=1`` format (r0, r, force, work).
        pdb_out_file (str): final coordinates. If it already exists the simulation is skipped.
        traj_out_file (str): trajectory.
        startdist (float): initial target distance in A.
        spring_constant (float): k of ``E = k (r - r0)^2`` in kcal/mol/A^2, AMBER convention
            (``rk2`` in ``dist_duck.rst``). The original OpenMM implementation used
            ``0.5 * k``, i.e. a spring half as stiff as AMBER's.
        force_constant_chunk (float): chunk heavy-atom restraint in kcal/mol/A^2.
        init_velocity (float): steering speed in A per 2 fs (legacy unit; 1e-5 = 5 A/ns).
        gpu_id (int, optional): GPU index, ``None`` for CPU.
        hmr (bool): 4 fs time step (the steering speed in A/ps is unchanged).
        pickle_file (str): preparation pickle with the structure and the key interaction.
        **simulation_kwargs: extra options for ``DuckSimulation`` (precision, integrator, ...).

    Returns:
        numpy.ndarray or None: the work profile, or None if the run was skipped.
    """
    if os.path.isfile(pdb_out_file):
        print(f'{pdb_out_file} is already calculated, skipping')
        return None
    if u.is_quantity(temperature):
        temperature = temperature.value_in_unit(u.kelvin)
    structure, key_interaction = load_complex_pickle(pickle_file)
    simulation = DuckSimulation(structure, key_interaction, gpu_id=gpu_id, hmr=hmr, temperature=temperature,
                                chunk_restraint_k=force_constant_chunk, smd_spring_k=spring_constant,
                                **simulation_kwargs)
    simulation.load_checkpoint(checkpoint_in_file)
    return simulation.run_smd(temperature, dat_out_file, csv_file=csv_out_file, dcd_file=traj_out_file,
                              pdb_file=pdb_out_file, start_dist=startdist,
                              speed=speed_from_init_velocity(init_velocity))


if __name__ == "__main__":
    if len(sys.argv) != 8:
        sys.exit("Usage: steered_md.py temperature in.chk out.csv out.dat out.pdb out.dcd startdist")
    run_steered_md(float(sys.argv[1]), *sys.argv[2:7], float(sys.argv[7]))
