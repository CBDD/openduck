"""Tests of the OpenMM restraints and of the MD/SMD engine on a small water box (CPU platform)."""
import pickle

import numpy as np
import parmed
import pytest

from duck.steps.production import DuckSimulation, run_duck_cycles, speed_from_init_velocity
from duck.utils import duck_stuff
from duck.utils.compat import mm, app, unit as u

KJ_PER_KCAL = 4.184


@pytest.fixture(scope='module')
def water_box():
    """~1000-atom TIP3P box where two water oxygens play the role of the H-bond pair and one
    water is renamed to a 'receptor' residue (so it is restrained as chunk heavy atoms)."""
    forcefield = app.ForceField('amber99sb.xml', 'tip3p.xml')
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(forcefield, boxSize=mm.Vec3(2.2, 2.2, 2.2) * u.nanometer)
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * u.nanometer,
                                     rigidWater=False, constraints=None)
    structure = parmed.openmm.load_topology(modeller.topology, system, modeller.positions)
    coords = structure.coordinates
    oxygens = [a.idx for a in structure.atoms if a.element == 8]
    # pick the oxygen pair closest to 2.8 A
    first = oxygens[0]
    dist = np.linalg.norm(coords[oxygens] - coords[first], axis=1)
    second = oxygens[int(np.argmin(np.abs(dist - 2.8) + (dist < 0.1) * 100))]
    structure.atoms[first].residue.name = 'REC'     # receptor-like residue -> chunk restraint
    structure.atoms[second].residue.name = 'UNL'    # ligand-like residue
    return structure, [first, second, float(np.linalg.norm(coords[first] - coords[second]))]


def _energy(system, positions, box, group=-1):
    context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName('Reference'))
    context.setPeriodicBoxVectors(*box)
    context.setPositions(positions)
    return context.getState(getEnergy=True, groups=group).getPotentialEnergy().value_in_unit(u.kilocalorie_per_mole)


def _two_particles(r_angstrom):
    system = mm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    box = [mm.Vec3(3, 0, 0), mm.Vec3(0, 3, 0), mm.Vec3(0, 0, 3)]
    system.setDefaultPeriodicBoxVectors(*box)
    positions = [mm.Vec3(0, 0, 0), mm.Vec3(r_angstrom / 10.0, 0, 0)]
    return system, positions, box


def test_steering_restraint_uses_amber_convention():
    system, positions, box = _two_particles(3.0)
    duck_stuff.addSteeringRestraint(system, 0, 1, spring_constant=50.0, r0=0.25)
    # E = k (r - r0)^2 with k = 50 kcal/mol/A^2 (no factor 1/2, as AMBER rk2)
    assert _energy(system, positions, box) == pytest.approx(50.0 * 0.5 ** 2, rel=1e-6)
    assert system.getForce(0).getForceGroup() == duck_stuff.SMD_FORCE_GROUP


@pytest.mark.parametrize('r, expected', [(1.5, 1.0 * 0.5 ** 2), (2.5, 0.0), (3.5, 10.0 * 0.5 ** 2),
                                         (5.0, 10.0 * 1.0 * (2 * 5.0 - 4.0 - 3.0))])
def test_flat_bottom_restraint_matches_dist_md(r, expected):
    system, positions, box = _two_particles(r)
    duck_stuff.applyLigandChunkRestraint(system, 1.0, 10.0, 2 * u.angstrom, 3 * u.angstrom, 4 * u.angstrom, [0, 1],
                                         prefix='hb_', switch_parameter='hb_on')
    assert _energy(system, positions, box) == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_positional_restraint_energy():
    system, positions, box = _two_particles(3.0)
    reference = [mm.Vec3(0.1, 0, 0), mm.Vec3(0.3, 0, 0)]
    duck_stuff.applyHarmonicPositionalRestraints(system, 2.0, reference, [0])
    assert _energy(system, positions, box) == pytest.approx(2.0 * 1.0 ** 2, rel=1e-6)


def test_heavy_atoms_exclude_solvent_ions_and_ligand(water_box):
    structure, key = water_box
    heavy = duck_stuff.getHeavyAtomsInSystem(structure)
    assert heavy == [key[0]]   # only the oxygen of the receptor-like residue
    ions = parmed.Structure()
    for name in ('NA', 'CL', 'Na+', 'ZN'):
        ions.add_atom(parmed.Atom(name=name, atomic_number=11 if name != 'ZN' else 30), name, 1)
    names = [a.residue.name for a in ions.topology.atoms()]
    assert [names[i] for i in duck_stuff.getHeavyAtomsInSystem(ions)] == ['ZN']  # structural metals stay


def test_speed_conversion():
    assert speed_from_init_velocity(0.00001) == pytest.approx(0.005)  # 5 A/ns


def test_smd_work_is_exact_for_the_stepwise_protocol(water_box, tmp_path):
    structure, key = water_box
    sim = DuckSimulation(structure, key, gpu_id=None, timestep=0.001, chunk_restraint_k=1.0)
    sim.context.setVelocitiesToTemperature(300)
    start = key[2]
    rows = []
    # record the distance from the coordinates after every increment to check the reported profile
    original_step = sim.simulation.step

    def step_and_record(n):
        original_step(n)
        pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.angstrom)
        rows.append(np.linalg.norm(pos[key[0]] - pos[key[1]]))
    sim.simulation.step = step_and_record
    profile = sim.run_smd(300, str(tmp_path / 'smd.dat'), start_dist=start, distance=0.2, speed=0.05,
                          report_interval_ps=0.02)
    assert profile.shape == (200, 4)
    np.testing.assert_allclose(profile[:, 0], start + 0.001 * np.arange(1, 201), atol=1e-9)   # r0
    np.testing.assert_allclose(profile[:, 1], rows, atol=1e-4)                                 # r
    np.testing.assert_allclose(profile[:, 2], -2 * 50.0 * (profile[:, 1] - profile[:, 0]), atol=1e-3)  # force
    # work of each increment at fixed coordinates: k [(r - r0_new)^2 - (r - r0_old)^2]
    r_before = np.concatenate([[start], profile[:-1, 1]])  # r just before each increment (initial r = start)
    r0_old = np.concatenate([[start], profile[:-1, 0]])
    increments = 50.0 * ((r_before - profile[:, 0]) ** 2 - (r_before - r0_old) ** 2)
    increments[0] = profile[0, 3]  # the initial distance is not exactly `start`
    np.testing.assert_allclose(np.diff(profile[:, 3]), increments[1:], atol=1e-6)
    # and it agrees with the AMBER-style integral of the force over r0
    trapezoid = np.sum(0.5 * (profile[1:, 2] + profile[:-1, 2]) * np.diff(profile[:, 0]))
    assert profile[-1, 3] - profile[0, 3] == pytest.approx(trapezoid, abs=0.05 * max(1.0, abs(trapezoid)))
    written = np.loadtxt(tmp_path / 'smd.dat')
    np.testing.assert_allclose(written, profile, atol=1e-5)


def test_phases_switch_restraints(water_box):
    structure, key = water_box
    sim = DuckSimulation(structure, key, gpu_id=None, timestep=0.001)
    sim._set_phase('md', 300)
    assert sim.context.getParameter('hb_on') == 1.0 and sim.context.getParameter('smd_k') == 0.0
    sim._set_phase('smd', 325)
    assert sim.context.getParameter('hb_on') == 0.0
    assert sim.context.getParameter('smd_k') == pytest.approx(50.0 * KJ_PER_KCAL * 100)
    assert sim.integrator.getTemperature().value_in_unit(u.kelvin) == pytest.approx(325)


def test_run_duck_cycles_and_resume(water_box, tmp_path, monkeypatch):
    structure, key = water_box
    monkeypatch.chdir(tmp_path)
    with open('complex_system.pickle', 'wb') as fh:
        pickle.dump([structure] + key, fh)
    sim = DuckSimulation(structure, key, gpu_id=None, timestep=0.001)
    sim.context.setVelocitiesToTemperature(300)
    sim.save_checkpoint('equil.chk')
    kwargs = dict(n_cycles=1, md_length=0.001, gpu_id=None, start_dist=key[2], init_velocity=0.002,
                  timestep=0.001, traj_interval_ps=0)
    run_duck_cycles('equil.chk', 'complex_system.pickle', **kwargs)
    for name in ('md_0.chk', 'md_0.csv', 'md_0.pdb', 'smd_0_300.dat', 'smd_0_325.dat', 'smd_0_325.pdb'):
        assert (tmp_path / 'duck_runs' / name).exists()
    profile = np.loadtxt(tmp_path / 'duck_runs' / 'smd_0_300.dat')
    assert profile.shape == (25, 4)   # 2.5 A at 1 A/ps, one row per 0.1 ps
    mtime = (tmp_path / 'duck_runs' / 'smd_0_300.dat').stat().st_mtime
    run_duck_cycles('equil.chk', 'complex_system.pickle', **kwargs)   # everything is reused
    assert (tmp_path / 'duck_runs' / 'smd_0_300.dat').stat().st_mtime == mtime


def test_load_checkpoint_keeps_this_simulation_parameters(water_box, tmp_path):
    structure, key = water_box
    writer = DuckSimulation(structure, key, gpu_id=None, timestep=0.001, chunk_restraint_k=0.1)
    writer.save_checkpoint(tmp_path / 'other.chk')
    reader = DuckSimulation(structure, key, gpu_id=None, timestep=0.001, chunk_restraint_k=1.0)
    reader.load_checkpoint(tmp_path / 'other.chk')
    assert reader.context.getParameter('chunk_k') == pytest.approx(1.0 * KJ_PER_KCAL * 100)


def test_hmr_is_applied_to_amber_topologies(water_box, tmp_path):
    structure, key = water_box
    structure.save(str(tmp_path / 'box.prmtop'))
    structure.save(str(tmp_path / 'box.inpcrd'))
    amber = parmed.load_file(str(tmp_path / 'box.prmtop'), str(tmp_path / 'box.inpcrd'))
    assert type(amber).__name__ == 'AmberParm' and not duck_stuff.structure_has_hmr(amber)
    sim = DuckSimulation(amber, key, gpu_id=None, hmr=True)
    assert sim.dt == 0.004 and duck_stuff.structure_has_hmr(sim.structure)
