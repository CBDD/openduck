"""OpenMM production phase of dynamic undocking: cycles of MD sampling and steered MD (SMD).

Each cycle ``i`` mirrors the AMBER DUck templates (``duck/templates/amber_inputs``):

``MD_i``
    ``md_length`` ns at 300 K starting from the previous cycle (or the equilibrated system).
    Chunk heavy atoms are restrained to the equilibrated coordinates (AMBER ``-ref 3_eq.rst``)
    and the key H-bond is kept formed by the flat-bottom restraint of ``dist_md.rst``.
``SMD_i`` at 300 K and 325 K
    Both start from the end of ``MD_i``. The H-bond distance is steered from ``start_dist`` to
    ``start_dist + 2.5`` A at constant speed (5 A/ns by default) with ``E = k (r - r0)^2``,
    ``k = 50`` kcal/mol/A^2, exactly like ``dist_duck.rst`` with ``jar=1``. After each SMD the
    WQB is computed and the run stops early if it falls below ``wqb_threshold``.

Implementation notes
    * A single OpenMM ``Context`` is created for the whole run. Switching between MD and SMD
      only changes global parameters (``hb_on``, ``smd_k``, ``smd_r0``) and the thermostat
      temperature, and branching from the MD end state uses ``Context.setState``, so the
      random number stream keeps advancing (each phase gets independent noise).
    * The steering restraint acts between the two H-bond atoms (no anchor point or steering
      vector to update). ``r0`` advances in increments every ``report_interval`` (0.1 ps by
      default, i.e. 5000 increments of 0.0005 A, the resolution of AMBER's DUMPAVE). The work
      of each increment is exact for this stepwise protocol,
      ``W += E(r; r0_new) - E(r; r0_old)``, with ``r`` recovered from the analytic derivative
      ``dE/dr0 = -2k (r - r0)`` evaluated for the steering force group only, so no coordinates
      are copied from the GPU.
    * The SMD speed is defined in A/ps, independent of the time step: with HMR (4 fs) the
      simulated SMD lasts the same 500 ps as without it.

Output files (unchanged names): ``md_{i}.{chk,csv,pdb,dcd}`` and ``smd_{i}_{T}.{dat,csv,pdb,dcd}``.
The ``.dat`` files use the AMBER ``jar=1`` DUMPAVE layout: ``r0 (A)  r (A)  force (kcal/mol/A)
work (kcal/mol)``.
"""
import csv
import os
import time
from pathlib import Path

import numpy as np

from duck.utils.compat import mm, app, unit as u
from duck.utils.duck_stuff import (SMD_FORCE_GROUP, addSteeringRestraint, applyHarmonicPositionalRestraints,
                                   applyLigandChunkRestraint, apply_hmr, getHeavyAtomsInSystem, get_platform,
                                   load_complex_pickle, structure_has_hmr)
from duck.utils.check_system import check_temperature
from duck.utils.exceptions import EquilibrationError

KJ_PER_KCAL = 4.184
LEGACY_STEP_PS = 0.002  # time step in which the legacy ``init_velocity`` (A/step) is expressed


def speed_from_init_velocity(init_velocity):
    """Convert the legacy ``init_velocity`` (A per 2 fs step) into a steering speed in A/ps."""
    return init_velocity / LEGACY_STEP_PS


def make_integrator(kind, temperature, friction, dt):
    """Create the Langevin integrator used in all DUck phases.

    Args:
        kind (str): ``'langevin'`` (leapfrog, as AMBER ``ntt=3`` and the original OpenDUck) or
            ``'langevin-middle'`` (``LangevinMiddleIntegrator``, OpenMM >= 7.5).
        temperature (float): K.
        friction (float): collision rate in 1/ps.
        dt (float): time step in ps.
    """
    kinds = {'langevin': mm.LangevinIntegrator}
    if hasattr(mm, 'LangevinMiddleIntegrator'):
        kinds['langevin-middle'] = mm.LangevinMiddleIntegrator
    if kind not in kinds:
        raise ValueError(f'Unknown integrator "{kind}". Options: {", ".join(kinds)}')
    return kinds[kind](temperature * u.kelvin, friction / u.picosecond, dt * u.picosecond)


def _state_from_foreign_checkpoint(system, checkpoint, device_index=None):
    """Read positions/velocities/box from a checkpoint written with another precision or platform.

    OpenMM checkpoints can only be loaded by a Context with the same platform and precision.
    Old OpenDUck runs wrote double-precision CUDA checkpoints, so we try the likely
    combinations with a lightweight particle-only system and return the State.
    """
    light = mm.System()
    for i in range(system.getNumParticles()):
        light.addParticle(system.getParticleMass(i))
    light.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    candidates = [('CUDA', 'double'), ('CUDA', 'mixed'), ('CUDA', 'single'),
                  ('OpenCL', 'double'), ('OpenCL', 'mixed'), ('OpenCL', 'single'),
                  ('CPU', None), ('Reference', None)]
    for platform_name, precision in candidates:
        try:
            platform = mm.Platform.getPlatformByName(platform_name)
            properties = {'Precision': precision} if precision else {}
            if precision and device_index is not None:
                properties['DeviceIndex'] = device_index
            context = mm.Context(light, mm.LangevinIntegrator(300, 1, 0.002), platform, properties)
            context.loadCheckpoint(checkpoint)
        except Exception:
            continue
        state = context.getState(getPositions=True, getVelocities=True)
        del context
        return state
    return None


class DuckSimulation:
    """MD/SMD driver that keeps one OpenMM Context alive for a whole DUck production run.

    Args:
        structure (parmed.Structure): prepared, parametrized system (from the preparation pickle).
        key_interaction (sequence): ``(protein_atom_idx, ligand_atom_idx, ...)`` of the H-bond.
        gpu_id (int, optional): GPU index; ``None`` runs on the CPU.
        hmr (bool): use a 4 fs time step. If the structure does not carry repartitioned hydrogen
            masses they are repartitioned here (in place).
        precision (str): GPU precision (``mixed``, ``single`` or ``double``).
        temperature (float): MD temperature in K.
        chunk_restraint_k (float): positional restraint on chunk heavy atoms, kcal/mol/A^2
            (AMBER ``restraint_wt``).
        smd_spring_k (float): steering spring constant ``k`` of ``E = k (r - r0)^2``,
            kcal/mol/A^2 (AMBER ``rk2``).
        hb_restraint_k2, hb_restraint_k3 (float): flat-bottom H-bond restraint used during MD
            (``dist_md.rst`` rk2/rk3), kcal/mol/A^2.
        integrator (str): ``'langevin'`` or ``'langevin-middle'``.
        friction (float): Langevin collision rate in 1/ps.
        timestep (float, optional): time step in ps; defaults to 0.004 with HMR, else 0.002.
        deterministic_forces (bool): bitwise reproducible forces (slower).
        threads (int, optional): CPU threads when running on the CPU platform.
        nonbonded_cutoff (float): direct-space cutoff in A.
    """

    def __init__(self, structure, key_interaction, gpu_id=None, hmr=False, precision='mixed',
                 temperature=300.0, chunk_restraint_k=1.0, smd_spring_k=50.0, hb_restraint_k2=1.0,
                 hb_restraint_k3=10.0, integrator='langevin', friction=4.0, timestep=None,
                 deterministic_forces=False, threads=None, nonbonded_cutoff=9.0):
        if len(key_interaction) < 2:
            raise ValueError('The key interaction (protein atom, ligand atom) is missing from the system pickle.')
        self.protein_atom, self.ligand_atom = int(key_interaction[0]), int(key_interaction[1])
        self.temperature = float(temperature)
        self.dt = float(timestep) if timestep else (0.004 if hmr else 0.002)
        self.smd_spring_k = float(smd_spring_k)
        if hmr and not structure_has_hmr(structure):
            print('HMR requested but the system has standard hydrogen masses: repartitioning them now.')
            apply_hmr(structure)
        elif not hmr and structure_has_hmr(structure):
            print('Note: the system has repartitioned hydrogen masses; use HMR to run with a 4 fs time step.')
        self.structure = structure

        system = structure.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=nonbonded_cutoff * u.angstrom,
                                        constraints=app.HBonds, hydrogenMass=None)
        self._restraint = applyHarmonicPositionalRestraints(
            system, chunk_restraint_k, structure.positions, getHeavyAtomsInSystem(structure), parameter_name='chunk_k')
        applyLigandChunkRestraint(system, hb_restraint_k2, hb_restraint_k3, 2 * u.angstrom, 3 * u.angstrom,
                                  4 * u.angstrom, key_interaction, prefix='hb_', switch_parameter='hb_on')
        addSteeringRestraint(system, self.protein_atom, self.ligand_atom, spring_constant=smd_spring_k, prefix='smd_')
        self._smd_k = self._parameter_value(system, 'smd_k')
        self._parameters = {force.getGlobalParameterName(i): force.getGlobalParameterDefaultValue(i)
                            for force in system.getForces() if hasattr(force, 'getNumGlobalParameters')
                            for i in range(force.getNumGlobalParameters())}
        self.system = system

        self.integrator = make_integrator(integrator, self.temperature, friction, self.dt)
        platform, properties = get_platform(gpu_id, precision, deterministic_forces, threads)
        self._device_index = properties.get('DeviceIndex')
        self.simulation = app.Simulation(structure.topology, system, self.integrator, platform, properties)
        self.context = self.simulation.context
        self.timings = []  # one record per run_md / run_smd call (phase, temperature, simulated_ps, wall_s)
        if structure.box_vectors is not None:
            self.context.setPeriodicBoxVectors(*structure.box_vectors)
        self.context.setPositions(structure.positions)

    def device_name(self):
        """Name of the GPU (or the platform name on the CPU)."""
        platform = self.context.getPlatform()
        if 'DeviceName' in platform.getPropertyNames():
            return platform.getPropertyValue(self.context, 'DeviceName')
        return platform.getName()

    @staticmethod
    def _parameter_value(system, name):
        for force in system.getForces():
            if hasattr(force, 'getNumGlobalParameters'):
                for i in range(force.getNumGlobalParameters()):
                    if force.getGlobalParameterName(i) == name:
                        return force.getGlobalParameterDefaultValue(i)
        raise KeyError(name)

    # ------------------------------------------------------------------ state handling
    def load_checkpoint(self, path):
        """Load an OpenMM checkpoint, also accepting ones written with another precision/platform.

        Only the dynamical state is taken from the checkpoint: the global parameters it stores
        (e.g. restraint force constants of the run that wrote it) are reset to this simulation's values.
        """
        with open(path, 'rb') as fh:
            checkpoint = fh.read()
        try:
            self.context.loadCheckpoint(checkpoint)
        except Exception as err:
            state = _state_from_foreign_checkpoint(self.system, checkpoint, self._device_index)
            if state is None:
                raise RuntimeError(f'Could not read the checkpoint {path}: {err}') from err
            print(f'{path} was written with another platform/precision; loaded its coordinates and velocities.')
            self.set_state(state)
        for name, value in self._parameters.items():
            self.context.setParameter(name, value)

    def save_checkpoint(self, path):
        self.simulation.saveCheckpoint(str(path))

    def get_state(self):
        """Positions, velocities and box of the current context (no RNG state, no parameters)."""
        return self.context.getState(getPositions=True, getVelocities=True)

    def set_state(self, state):
        self.context.setState(state)

    def set_restraint_reference(self, positions=None):
        """Use ``positions`` (default: the current coordinates) as reference of the chunk restraints."""
        if positions is None:
            positions = self.context.getState(getPositions=True).getPositions(asNumpy=True)
        positions = positions.value_in_unit(u.nanometer)
        for i in range(self._restraint.getNumParticles()):
            atom, _ = self._restraint.getParticleParameters(i)
            self._restraint.setParticleParameters(i, atom, list(positions[atom]))
        self._restraint.updateParametersInContext(self.context)

    # ------------------------------------------------------------------ helpers
    def _set_phase(self, phase, temperature):
        self.context.setParameter('hb_on', 1.0 if phase == 'md' else 0.0)
        self.context.setParameter('smd_k', self._smd_k if phase == 'smd' else 0.0)
        self.integrator.setTemperature(temperature * u.kelvin)

    def _attach_reporters(self, csv_file, dcd_file, total_steps, log_interval_ps, traj_interval_ps):
        self.simulation.reporters = []
        self.simulation.currentStep = 0
        if csv_file:
            interval = max(1, min(int(round(log_interval_ps / self.dt)), total_steps))  # at least one report
            self.simulation.reporters.append(app.StateDataReporter(
                str(csv_file), interval, step=True, time=True, totalEnergy=True, kineticEnergy=True,
                potentialEnergy=True, temperature=True, density=True, progress=True, totalSteps=total_steps,
                speed=True))
        if dcd_file and traj_interval_ps:
            interval = max(1, int(round(traj_interval_ps / self.dt)))
            self.simulation.reporters.append(app.DCDReporter(str(dcd_file), interval))

    def _finish_phase(self, pdb_file=None, checkpoint_file=None):
        self.simulation.reporters = []  # closes the report files
        if checkpoint_file:
            self.save_checkpoint(checkpoint_file)
        if pdb_file:
            positions = self.context.getState(getPositions=True).getPositions()
            with open(pdb_file, 'w') as fh:
                app.PDBFile.writeFile(self.simulation.topology, positions, fh)

    def _smd_derivative(self):
        """dE/dr0 of the steering restraint (kJ/mol/nm), computed for its force group only."""
        state = self.context.getState(getParameterDerivatives=True, groups={SMD_FORCE_GROUP})
        return state.getEnergyParameterDerivatives()['smd_r0']

    # ------------------------------------------------------------------ phases
    def run_md(self, length_ns, csv_file=None, dcd_file=None, pdb_file=None, checkpoint_file=None,
               temperature=None, log_interval_ps=4.0, traj_interval_ps=4.0):
        """Run restrained MD (H-bond flat-bottom restraint on, steering off).

        Args:
            length_ns (float): simulated time in ns.
            csv_file, dcd_file, pdb_file, checkpoint_file (str, optional): outputs.
            temperature (float, optional): K, defaults to the simulation temperature.
            log_interval_ps, traj_interval_ps (float): reporting intervals; 0 disables the trajectory.
        """
        temperature = self.temperature if temperature is None else float(temperature)
        n_steps = int(round(length_ns * 1000.0 / self.dt))
        self._set_phase('md', temperature)
        self._attach_reporters(csv_file, dcd_file, n_steps, log_interval_ps, traj_interval_ps)
        start = time.time()
        self.simulation.step(n_steps)
        elapsed = time.time() - start
        self.timings.append(dict(phase='md', temperature=temperature, simulated_ps=n_steps * self.dt, wall_s=elapsed))
        self._finish_phase(pdb_file, checkpoint_file)
        print(f'  MD {length_ns:g} ns at {temperature:g} K: {elapsed:.0f} s ({_ns_per_day(n_steps * self.dt, elapsed)})')
        if csv_file:
            check_temperature(csv_file, temperature)

    def run_smd(self, temperature, dat_file, csv_file=None, dcd_file=None, pdb_file=None, start_dist=2.5,
                distance=2.5, speed=0.005, report_interval_ps=0.1, log_interval_ps=0.4, traj_interval_ps=4.0):
        """Steer the H-bond from ``start_dist`` to ``start_dist + distance`` (A) at ``speed`` (A/ps).

        The steering restraint ``E = k (r - r0)^2`` is switched on with ``r0 = start_dist`` and
        ``r0`` is advanced every ``report_interval_ps``. One row per increment is written to
        ``dat_file`` in the AMBER ``jar=1`` format (``r0  r  force  work``; A, kcal/mol/A,
        kcal/mol), where ``r`` and ``force`` are measured at the end of the increment.

        Returns:
            numpy.ndarray: the work profile written to ``dat_file`` (n_increments x 4).
        """
        temperature = float(temperature)
        steps_per_update = max(1, int(round(report_interval_ps / self.dt)))
        n_updates = int(round(distance / (speed * steps_per_update * self.dt)))
        dr = distance / n_updates / 10.0                      # nm per increment
        k = self._smd_k                                      # kJ/mol/nm^2
        r0 = start_dist / 10.0                               # nm
        self._set_phase('smd', temperature)
        self.context.setParameter('smd_r0', r0)
        self._attach_reporters(csv_file, dcd_file, n_updates * steps_per_update, log_interval_ps, traj_interval_ps)

        profile = np.empty((n_updates, 4))
        work = 0.0
        r = r0 - self._smd_derivative() / (2 * k)
        start = time.time()
        for j in range(n_updates):
            r0_new = r0 + dr
            work += k * ((r - r0_new) ** 2 - (r - r0) ** 2)  # work of moving r0 at fixed coordinates
            r0 = r0_new
            self.context.setParameter('smd_r0', r0)
            self.simulation.step(steps_per_update)
            derivative = self._smd_derivative()
            r = r0 - derivative / (2 * k)
            profile[j] = (r0 * 10.0, r * 10.0, derivative / KJ_PER_KCAL / 10.0, work / KJ_PER_KCAL)
        elapsed = time.time() - start
        self.timings.append(dict(phase='smd', temperature=temperature, simulated_ps=n_updates * steps_per_update * self.dt,
                                 wall_s=elapsed))
        np.savetxt(dat_file, profile, fmt='%12.5f', delimiter='  ')
        self._finish_phase(pdb_file)
        print(f'  SMD at {temperature:g} K ({start_dist:g} -> {start_dist + distance:g} A): {elapsed:.0f} s '
              f'({_ns_per_day(n_updates * steps_per_update * self.dt, elapsed)})')
        if csv_file:
            check_temperature(csv_file, temperature)
        return profile


def _ns_per_day(simulated_ps, elapsed_s):
    if elapsed_s <= 0:
        return 'n/a'
    return f'{simulated_ps / 1000.0 / (elapsed_s / 86400.0):.1f} ns/day'


TIMING_COLUMNS = ['cycle', 'phase', 'temperature', 'simulated_ps', 'wall_s', 'ns_per_day', 'device']


def _append_timing(timings_file, cycle, record, device):
    """Append the wall time of one MD/SMD run to ``timings_file`` (CSV, header written on creation)."""
    new = not os.path.exists(timings_file)
    with open(timings_file, 'a', newline='') as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(TIMING_COLUMNS)
        ns_day = record['simulated_ps'] / 1000.0 / (record['wall_s'] / 86400.0) if record['wall_s'] > 0 else ''
        writer.writerow([cycle, record['phase'], record['temperature'], f"{record['simulated_ps']:g}",
                         f"{record['wall_s']:.2f}", f'{ns_day:.1f}' if ns_day != '' else '', device])


def run_duck_cycles(checkpoint, pickle_file='complex_system.pickle', n_cycles=20, md_length=0.5, gpu_id=None,
                    start_dist=2.5, init_velocity=0.00001, save_dir='duck_runs', wqb_threshold=None, hmr=False,
                    temperatures=(300, 325), traj_interval_ps=4.0, **simulation_kwargs):
    """Run the DUck production: ``n_cycles`` x (MD + SMD at each temperature + WQB check).

    Existing outputs are reused, so an interrupted run can simply be relaunched: ``md_{i}.chk``
    marks a finished MD and ``smd_{i}_{T}.pdb`` a finished SMD. The wall time of every MD and SMD
    run is appended to ``save_dir/timings.csv``.

    Args:
        checkpoint (str): equilibrated system checkpoint (e.g. ``equil.chk``).
        pickle_file (str): preparation pickle with the parmed structure and the key interaction.
        n_cycles (int): number of MD/SMD cycles.
        md_length (float): MD length per cycle in ns.
        gpu_id (int, optional): GPU index; ``None`` runs on the CPU.
        start_dist (float): initial r0 of the SMD in A.
        init_velocity (float): steering speed in A per 2 fs (legacy unit; 1e-5 = 5 A/ns). The
            speed in A/ps is kept constant when HMR (4 fs) is used.
        save_dir (str): output directory.
        wqb_threshold (float, optional): stop as soon as one SMD yields a WQB below this value.
        hmr (bool): run with a 4 fs time step on a mass-repartitioned system.
        temperatures (sequence): SMD temperatures in K.
        traj_interval_ps (float): DCD output interval for MD and SMD (0 disables trajectories).
        **simulation_kwargs: extra options for ``DuckSimulation`` (precision, integrator, ...).

    Returns:
        float or None: the WQB that triggered the early stop, else None.
    """
    from duck.utils.analysis_and_report import get_Wqb_value
    from duck.utils.check_system import check_if_equlibrated

    for path in (checkpoint, pickle_file):
        if not os.path.exists(path):
            raise OSError(f'{path} can not be found. Something might have gone wrong during equilibration.')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    structure, key_interaction = load_complex_pickle(pickle_file)
    simulation = DuckSimulation(structure, key_interaction, gpu_id=gpu_id, hmr=hmr, **simulation_kwargs)
    simulation.load_checkpoint(checkpoint)
    simulation.set_restraint_reference()  # restraints relative to the equilibrated system, as AMBER -ref 3_eq.rst
    speed = speed_from_init_velocity(init_velocity)
    platform = simulation.context.getPlatform().getName()
    device = simulation.device_name()
    timings_file = save_dir / 'timings.csv'
    print(f'DUck production on {platform} ({device}): {n_cycles} cycles, dt = {simulation.dt * 1000:g} fs, '
          f'SMD speed = {speed * 1000:g} A/ns')

    for i in range(n_cycles):
        md = save_dir / f'md_{i}'
        if Path(f'{md}.chk').exists():
            print(f'md_{i} already calculated, loading it')
            simulation.load_checkpoint(f'{md}.chk')
        else:
            print(f'Simulating md_{i}')
            simulation.run_md(md_length, csv_file=f'{md}.csv', dcd_file=f'{md}.dcd', pdb_file=f'{md}.pdb',
                              checkpoint_file=f'{md}.chk', traj_interval_ps=traj_interval_ps)
            _append_timing(timings_file, i, simulation.timings[-1], device)
            if not check_if_equlibrated(f'{md}.csv', 3):
                raise EquilibrationError(f'md_{i} is not stable; stopping the DUck production.')
        md_state = simulation.get_state()

        for temperature in temperatures:
            smd = save_dir / f'smd_{i}_{temperature}'
            if Path(f'{smd}.pdb').exists():
                print(f'smd_{i}_{temperature} already calculated, skipping')
            else:
                print(f'Simulating smd_{i}_{temperature}')
                simulation.set_state(md_state)
                simulation.run_smd(temperature, f'{smd}.dat', csv_file=f'{smd}.csv', dcd_file=f'{smd}.dcd',
                                   pdb_file=f'{smd}.pdb', start_dist=start_dist, speed=speed,
                                   traj_interval_ps=traj_interval_ps)
                _append_timing(timings_file, i, simulation.timings[-1], device)
            wqb = get_Wqb_value(f'{smd}.dat')[0]
            if wqb_threshold is not None and wqb < wqb_threshold:
                print(f'DUck replica {i}_{temperature} yielded a WQB of {wqb:.2f} kcal/mol, which is lower than '
                      f'the wqb_threshold ({wqb_threshold}).\nStopping OpenDUck simulations.')
                return wqb
            print(f'  WQB: {wqb:.2f} kcal/mol')
        simulation.set_state(md_state)  # the next MD continues from the end of this one
    return None
