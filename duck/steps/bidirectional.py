"""Bidirectional steered MD in OpenMM: forward and reverse pulls for the Crooks fluctuation theorem.

The protocol follows the DUck schema (chunked receptor, restraints and steering spring as in
:mod:`duck.steps.production`) and adds, after each forward pull, the reverse process that reforms
the interaction (thesis ch. 3.2.3.3 and 6.4; Miñarro et al. 2023). Each cycle ``i``:

``MD_i``
    ``md_length`` ns of restrained MD (flat-bottom H-bond restraint), continuing from the previous
    cycle as in DUck. Forward and reverse pulls branch from its final state.
``approach_i`` (optional)
    The spring is switched on at the current distance and its centre is moved to ``start_dist`` in
    ``approach_length`` ns, so the forward pull starts from (close to) equilibrium with the spring at
    ``start_dist``. Its work is written but not used. The thesis protocol used 1 ns with k = 60.
``forward_i``
    ``r0`` from ``start_dist`` to ``end_dist`` at ``speed`` (DUck: 2.5 -> 5 A at 5 A/ns, k = 50).
``reverse_i``
    From the final coordinates of ``forward_i``, with new Maxwell-Boltzmann velocities (AMBER
    ``irest=0``) and an optional ``reverse_hold`` ns with the spring fixed at ``end_dist``, ``r0`` from
    ``end_dist`` back to ``start_dist`` at the same speed.

Outputs (``save_dir``, default ``crooks_runs``): ``md_{i}.{chk,csv,pdb}``, ``approach_{i}.dat``,
``forward_{i}.{dat,pdb}``, ``reverse_{i}.{dat,pdb}`` (AMBER ``jar=1`` layout: ``r0 r force work``),
``timings.csv`` and ``protocol.json`` (parameters read by :mod:`duck.utils.crooks`). An interrupted
run is resumed by relaunching it: finished phases are detected from their outputs.
"""
import json
import os
from pathlib import Path

from duck import __version__
from duck.steps.production import DuckSimulation, append_timing
from duck.utils.crooks import OPENMM_DIR, PROTOCOL_FILE
from duck.utils.duck_stuff import getRestrainedAtoms, load_complex_pickle
from duck.utils.exceptions import EquilibrationError


# protocol.json entries that must not change when an existing run is resumed (n_cycles may grow)
PROTOCOL_KEYS = ('start_dist', 'end_dist', 'speed_A_per_ns', 'temperature', 'spring_k', 'md_length_ns',
                 'approach_length_ns', 'approach_spring_k', 'reverse_hold_ns', 'reassign_velocities', 'timestep_ps',
                 'waters_to_restrain', 'key_interaction')


def check_protocol(save_dir, **parameters):
    """Raise ValueError if ``save_dir`` holds pulls run with other protocol parameters."""
    path = Path(save_dir) / PROTOCOL_FILE
    if not path.is_file() or not any(Path(save_dir).glob('forward_*.dat')):
        return
    with open(path) as fh:
        previous = json.load(fh)
    changed = {key: (previous[key], parameters[key]) for key in PROTOCOL_KEYS
               if key in previous and key in parameters and previous[key] != parameters[key]}
    if changed:
        details = ', '.join(f'{key}: {old} -> {new}' for key, (old, new) in changed.items())
        raise ValueError(f'{save_dir} holds pulls of another protocol ({details}). Use another output folder or '
                         'the original parameters.')


def write_protocol(save_dir, **parameters):
    """Write the protocol parameters to ``save_dir/protocol.json`` (read back by the analysis)."""
    with open(Path(save_dir) / PROTOCOL_FILE, 'w') as fh:
        json.dump(dict(parameters, openduck_version=__version__), fh, indent=2)


def run_bidirectional_cycles(checkpoint, pickle_file='complex_system.pickle', n_cycles=20, md_length=0.5,
                             start_dist=2.5, end_dist=5.0, speed=0.005, temperature=300.0, approach_length=0.1,
                             approach_spring_k=None, reverse_hold=0.0, reassign_velocities=True, gpu_id=None,
                             hmr=False, save_dir=OPENMM_DIR, wqb_threshold=None, traj_interval_ps=0.0,
                             waters_to_restrain=0, **simulation_kwargs):
    """Run ``n_cycles`` x (MD + approach + forward pull + reverse pull).

    Args:
        checkpoint (str): equilibrated system checkpoint (e.g. ``equil.chk``).
        pickle_file (str): preparation pickle with the parmed structure and the key interaction.
        n_cycles (int): number of cycles (one forward and one reverse work per cycle).
        md_length (float): MD length per cycle in ns.
        start_dist, end_dist (float): ends of the pull in A (bound and quasi-bound distances).
        speed (float): steering speed in A/ps (0.005 = 5 A/ns).
        temperature (float): temperature of all phases in K (the Crooks theorem needs a single one).
        approach_length (float): length (ns) of the approach to ``start_dist`` before each forward pull;
            0 starts the forward pull directly from the MD, exactly like a DUck SMD.
        approach_spring_k (float, optional): spring constant of the approach (kcal/mol/A^2), default the
            steering spring constant.
        reverse_hold (float): ns with the spring fixed at ``end_dist`` before each reverse pull.
        reassign_velocities (bool): new velocities at the start of each reverse pull (thesis protocol).
        gpu_id (int, optional): GPU index; ``None`` runs on the CPU.
        hmr (bool): run with a 4 fs time step on a mass-repartitioned system.
        save_dir (str): output directory.
        wqb_threshold (float, optional): stop after a cycle whose forward pull yields a WQB below this value.
        traj_interval_ps (float): DCD interval of the MD and pulls (0, the default, disables trajectories).
        waters_to_restrain (int): also restrain the first N (retained) waters, e.g. the water of a steered
            water bridge (AMBER mask ``| :waters``).
        **simulation_kwargs: extra options for :class:`DuckSimulation` (``smd_spring_k``, ``restrained_atoms``,
            ``hb_restraint_distances``, ``precision``, ...).

    Returns:
        list of tuple: ``(forward .dat, reverse .dat)`` of every finished cycle.
    """
    from duck.utils.analysis_and_report import get_Wqb_value
    from duck.utils.check_system import check_if_equlibrated

    for path in (checkpoint, pickle_file):
        if not os.path.exists(path):
            raise OSError(f'{path} can not be found. Something might have gone wrong during equilibration.')
    if end_dist == start_dist:
        raise ValueError('The start and end distances of the pull must differ.')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    structure, key_interaction = load_complex_pickle(pickle_file)
    if waters_to_restrain:
        simulation_kwargs['restrained_atoms'] = getRestrainedAtoms(structure, waters_to_restrain)
    simulation = DuckSimulation(structure, key_interaction, gpu_id=gpu_id, hmr=hmr, temperature=temperature,
                                **simulation_kwargs)
    simulation.load_checkpoint(checkpoint)
    simulation.set_restraint_reference()  # restraints relative to the equilibrated system, as AMBER -ref 3_eq.rst
    device = simulation.device_name()
    timings_file = save_dir / 'timings.csv'
    protocol = dict(engine='openmm', start_dist=start_dist, end_dist=end_dist, speed_A_per_ns=speed * 1000,
                    temperature=temperature, spring_k=simulation.smd_spring_k, md_length_ns=md_length,
                    approach_length_ns=approach_length, approach_spring_k=approach_spring_k or simulation.smd_spring_k,
                    reverse_hold_ns=reverse_hold, reassign_velocities=reassign_velocities, n_cycles=n_cycles,
                    timestep_ps=simulation.dt, waters_to_restrain=waters_to_restrain,
                    key_interaction=[int(x) for x in key_interaction[:2]])
    check_protocol(save_dir, **protocol)
    write_protocol(save_dir, **protocol)
    print(f'Bidirectional SMD on {simulation.context.getPlatform().getName()} ({device}): {n_cycles} cycles, '
          f'{start_dist:g} <-> {end_dist:g} A at {speed * 1000:g} A/ns, k = {simulation.smd_spring_k:g} kcal/mol/A^2, '
          f'T = {temperature:g} K, dt = {simulation.dt * 1000:g} fs')

    def log_timing(cycle):
        append_timing(timings_file, cycle, simulation.timings[-1], device)

    pulls = []
    for i in range(n_cycles):
        md = save_dir / f'md_{i}'
        if Path(f'{md}.chk').exists():
            print(f'md_{i} already calculated, loading it')
            simulation.load_checkpoint(f'{md}.chk')
        else:
            print(f'Simulating md_{i}')
            simulation.run_md(md_length, csv_file=f'{md}.csv', dcd_file=f'{md}.dcd', pdb_file=f'{md}.pdb',
                              checkpoint_file=f'{md}.chk', temperature=temperature, traj_interval_ps=traj_interval_ps)
            log_timing(i)
            if not check_if_equlibrated(f'{md}.csv', 3):
                Path(f'{md}.chk').unlink()  # so a relaunch does not reuse the rejected MD
                raise EquilibrationError(f'md_{i} is not stable; stopping the bidirectional production.')
        md_state = simulation.get_state()

        forward, reverse = save_dir / f'forward_{i}', save_dir / f'reverse_{i}'
        if Path(f'{reverse}.dat').exists() and Path(f'{forward}.dat').exists():
            print(f'forward_{i} and reverse_{i} already calculated, skipping')
        else:
            if Path(f'{forward}.dat').exists() and Path(f'{forward}.chk').exists():
                print(f'forward_{i} already calculated, loading its final state')
                simulation.load_checkpoint(f'{forward}.chk')
            else:
                if approach_length > 0:
                    current = simulation.current_distance()
                    print(f'Simulating approach_{i} ({current:.2f} -> {start_dist:g} A)')
                    if abs(current - start_dist) < 1e-3:
                        simulation.run_hold(approach_length, start_dist, temperature, spring_k=approach_spring_k,
                                            label='approach')
                    else:
                        simulation.run_smd(temperature, save_dir / f'approach_{i}.dat', start_dist=current,
                                           distance=start_dist - current,
                                           speed=abs(start_dist - current) / (approach_length * 1000.0),
                                           spring_k=approach_spring_k, traj_interval_ps=0, label='approach')
                    log_timing(i)
                print(f'Simulating forward_{i}')
                simulation.run_smd(temperature, f'{forward}.dat', csv_file=f'{forward}.csv', dcd_file=f'{forward}.dcd',
                                   pdb_file=f'{forward}.pdb', checkpoint_file=f'{forward}.chk', start_dist=start_dist,
                                   distance=end_dist - start_dist, speed=speed, traj_interval_ps=traj_interval_ps,
                                   label='forward')
                log_timing(i)
            if reassign_velocities:
                simulation.reassign_velocities(temperature)
            if reverse_hold > 0:
                simulation.run_hold(reverse_hold, end_dist, temperature)
                log_timing(i)
            print(f'Simulating reverse_{i}')
            simulation.run_smd(temperature, f'{reverse}.dat', csv_file=f'{reverse}.csv', dcd_file=f'{reverse}.dcd',
                               pdb_file=f'{reverse}.pdb', start_dist=end_dist, distance=start_dist - end_dist,
                               speed=speed, traj_interval_ps=traj_interval_ps, label='reverse')
            log_timing(i)
            Path(f'{forward}.chk').unlink(missing_ok=True)
        pulls.append((f'{forward}.dat', f'{reverse}.dat'))

        wqb = get_Wqb_value(f'{forward}.dat')[0]
        print(f'  WQB (forward): {wqb:.2f} kcal/mol')
        if wqb_threshold is not None and wqb < wqb_threshold:
            print(f'Cycle {i} yielded a WQB of {wqb:.2f} kcal/mol, lower than the wqb_threshold ({wqb_threshold}). '
                  'Stopping the bidirectional simulations.')
            break
        simulation.set_state(md_state)  # the next MD continues from the end of this one
    return pulls
