"""Minimisation and equilibration of the prepared protein-ligand system in OpenMM.

Protocol (chunk heavy atoms restrained to the prepared coordinates throughout):

1. Energy minimisation (at most 1000 iterations).
2. NVT heating: 100 ps at 300 K starting from the minimised structure, with the flat-bottom
   restraint that keeps the key H-bond formed (``dist_md.rst``).
3. NPT density equilibration: 100 ps at 300 K and 1 bar (Monte Carlo barostat).

The final state is written to ``equil.chk``, the starting point of the DUck production.
"""
import pickle

import parmed
from parmed.openmm import NetCDFReporter

from duck.utils import cal_ints, duck_stuff
from duck.utils.compat import mm, app, unit as u
from duck.utils.exceptions import EquilibrationError

# intermediate files removed with clean=True (density.csv is kept: it is used to check the equilibration)
EQUILIBRATION_FILES = ['minimisation.pdb', 'heating.csv', 'heating.dcd', 'heating_final.pdb', '3_eq.nc',
                       'density_final.pdb']


def do_equlibrate(force_constant_equilibrate=1.0, gpu_id=0, keyInteraction=None, clean=False, hmr=False,
                  precision='mixed', pickle_file='complex_system.pickle', temperature=300.0, friction=4.0,
                  heating_ps=100.0, density_ps=100.0, report_ps=2.0, deterministic_forces=False, threads=None):
    """Minimise, heat and equilibrate the density of the system stored in ``pickle_file``.

    Args:
        force_constant_equilibrate (float): restraint on chunk heavy atoms and k2 of the H-bond
            restraint, kcal/mol/A^2.
        gpu_id (int, optional): GPU index; ``None`` runs on the CPU.
        keyInteraction (list, optional): ``[protein_atom, ligand_atom, distance]``. If not given it
            is searched with :func:`duck.utils.cal_ints.find_interaction`.
        clean (bool): remove the intermediate files listed in ``EQUILIBRATION_FILES``.
        hmr (bool): use a 4 fs time step (hydrogen masses are repartitioned if the system lacks HMR).
        precision (str): GPU precision, ``mixed`` (default), ``single`` or ``double``.
        pickle_file (str): preparation pickle.
        temperature (float): K.
        friction (float): Langevin collision rate in 1/ps.
        heating_ps, density_ps (float): length of the NVT and NPT stages in ps.
        report_ps (float): reporting interval in ps.
        deterministic_forces (bool): bitwise reproducible forces (slower).
        threads (int, optional): CPU threads for the CPU platform.

    Returns:
        list: ``['equil.chk']``
    """
    if not keyInteraction:
        keyInteraction = cal_ints.find_interaction()
    platform, platformProperties = duck_stuff.get_platform(gpu_id, precision, deterministic_forces, threads)
    combined_pmd = duck_stuff.loadPickle(pickle_file)
    combined_pmd.symmetry = None
    chunk_heavy_atoms = duck_stuff.getHeavyAtomsInSystem(combined_pmd)
    dt = 0.004 if hmr else 0.002
    if hmr and not duck_stuff.structure_has_hmr(combined_pmd):
        print('HMR requested but the system has standard hydrogen masses: repartitioning them now.')
        duck_stuff.apply_hmr(combined_pmd)
    report_steps = max(1, int(round(report_ps / dt)))

    # Minimisation
    print('Minimising...')
    system = combined_pmd.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=9 * u.angstrom)
    duck_stuff.applyHarmonicPositionalRestraints(system, force_constant_equilibrate, combined_pmd.positions,
                                                 chunk_heavy_atoms)
    simulation = app.Simulation(combined_pmd.topology, system, mm.VerletIntegrator(1 * u.femtosecond), platform,
                                platformProperties)
    simulation.context.setPositions(combined_pmd.positions)
    print('Platform:', simulation.context.getPlatform().getName(),
          {key: simulation.context.getPlatform().getPropertyValue(simulation.context, key)
           for key in simulation.context.getPlatform().getPropertyNames()})
    simulation.minimizeEnergy(maxIterations=1000)
    positions = simulation.context.getState(getPositions=True).getPositions()
    with open('minimisation.pdb', 'w') as fh:
        app.PDBFile.writeFile(simulation.topology, positions, fh)

    # NVT heating from the minimised positions (restraint references are the prepared coordinates)
    steps = int(round(heating_ps / dt))
    system = combined_pmd.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=9 * u.angstrom,
                                       constraints=app.HBonds, hydrogenMass=None)
    duck_stuff.applyHarmonicPositionalRestraints(system, force_constant_equilibrate, combined_pmd.positions,
                                                 chunk_heavy_atoms)
    duck_stuff.applyLigandChunkRestraint(system, force_constant_equilibrate, 10.0, 2 * u.angstrom, 3 * u.angstrom,
                                         4 * u.angstrom, keyInteraction)
    integrator = mm.LangevinIntegrator(temperature * u.kelvin, friction / u.picosecond, dt * u.picosecond)
    simulation = app.Simulation(combined_pmd.topology, system, integrator, platform, platformProperties)
    simulation.context.setPositions(positions)
    simulation.reporters.append(app.StateDataReporter(
        'heating.csv', report_steps, time=True, potentialEnergy=True, temperature=True, density=True,
        remainingTime=True, speed=True, totalSteps=steps))
    simulation.reporters.append(app.DCDReporter('heating.dcd', report_steps))
    print('Heating...')
    simulation.step(steps)
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    positions, velocities = state.getPositions(), state.getVelocities()
    with open('heating_final.pdb', 'w') as fh:
        app.PDBFile.writeFile(simulation.topology, positions, fh)
    simulation.reporters = []

    # NPT density equilibration
    steps = int(round(density_ps / dt))
    simulation = duck_stuff.setUpNPTEquilibration(system, combined_pmd, platform, platformProperties, positions,
                                                  velocities, dt=dt, temperature=temperature, friction=friction)
    simulation.reporters.append(app.StateDataReporter(
        'density.csv', report_steps, time=True, potentialEnergy=True, temperature=True, density=True,
        remainingTime=True, speed=True, totalSteps=steps))
    simulation.reporters.append(NetCDFReporter('3_eq.nc', report_steps, vels=True))
    print('Correcting density...')
    simulation.step(steps)
    positions = simulation.context.getState(getPositions=True).getPositions()
    with open('density_final.pdb', 'w') as fh:
        app.PDBFile.writeFile(simulation.topology, positions, fh)
    checkpoint = 'equil.chk'
    simulation.saveCheckpoint(checkpoint)
    simulation.reporters = []

    if clean:
        cal_ints.clean_up_files(files_to_delete=EQUILIBRATION_FILES)
    return [checkpoint]


def equilibrate_from_amber_prep(interaction, prmtop_file, inpcrd_file, chunk, gpu_id, force_constant_eq=1.0,
                                **equilibration_kwargs):
    """Equilibrate a system prepared for AMBER (prmtop + inpcrd) to run the OpenMM DUck protocol.

    The key interaction is located through the chunk file, the system and the interaction are
    written to ``complex_system.pickle`` and :func:`do_equlibrate` is run.

    Args:
        interaction (str): receptor atom as ``chain_resname_resid_atomname``.
        prmtop_file, inpcrd_file (str): AMBER topology and coordinates.
        chunk (str): receptor chunk (PDB or mol2) with the original residue numbering.
        gpu_id (int, optional): GPU index; ``None`` runs on the CPU.
        force_constant_eq (float): equilibration restraint in kcal/mol/A^2.
        **equilibration_kwargs: extra options for :func:`do_equlibrate`.

    Returns:
        list: ``['equil.chk']``

    Raises:
        EquilibrationError: if the potential energy of the NPT stage looks unstable.
    """
    from duck.utils.check_system import check_if_equlibrated
    c_pm = parmed.load_file(prmtop_file, inpcrd_file)
    keyInteraction = cal_ints.find_interaction_amber_input(combined_pmd=c_pm, chunk_file=chunk, res_atom=interaction)
    with open('complex_system.pickle', 'wb') as fh:
        pickle.dump([c_pm] + keyInteraction, fh, protocol=pickle.HIGHEST_PROTOCOL)
    checkpoint = do_equlibrate(keyInteraction=keyInteraction, force_constant_equilibrate=force_constant_eq,
                               gpu_id=gpu_id, **equilibration_kwargs)
    if not check_if_equlibrated('density.csv', 1):
        raise EquilibrationError('System is not equilibrated.')
    return checkpoint


if __name__ == "__main__":
    do_equlibrate()
