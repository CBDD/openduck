"""Shared OpenMM building blocks for the DUck simulations.

Restraints follow AMBER's energy conventions so that the OpenMM protocol mirrors the
AMBER DUck inputs (``duck/templates/amber_inputs``):

* positional restraints, ``restraint_wt`` in AMBER: ``E = k * |x - x0|^2``
* flat-bottom H-bond restraint used during MD (``dist_md.rst``)
* steering restraint used during SMD (``dist_duck.rst``, ``jar=1``): ``E = k * (r - r0)^2``

Note that AMBER restraint force constants carry no factor 1/2.
"""
import pickle
import warnings

from duck.utils.compat import mm, app, unit as u

KCAL_PER_MOL_PER_A2 = u.kilocalorie_per_mole / u.angstrom**2

# Residues that are never part of the restrained "chunk": solvent, ligand and bulk ions.
# Mirrors the AMBER restraint mask (``Amber_templates.extract_residuenumbers``).
NON_CHUNK_RESIDUES = frozenset([
    'HOH', 'WAT', 'IP3', 'LIG', 'UNL',
    'NA', 'CL', 'Na+', 'Cl-', 'K+', 'Rb+', 'Cs+', 'Li+', 'IP', 'IM', 'IB',
])
WATER_RESIDUES = frozenset(['HOH', 'WAT', 'IP3', 'TIP3', 'TIP4', 'SOL'])

# Force group that only contains the steering restraint, so its energy derivative can be
# evaluated without recomputing the rest of the system (parmed uses groups 0-2 and 11).
SMD_FORCE_GROUP = 30


def _to_kj_per_nm2(k):
    """Convert a force constant from kcal/mol/A^2 to OpenMM units (kJ/mol/nm^2)."""
    return (k * KCAL_PER_MOL_PER_A2).value_in_unit(u.kilojoule_per_mole / u.nanometer**2)


def _to_nm(value):
    """Strip the unit of a length (plain floats are assumed to be in nm already)."""
    return value.value_in_unit(u.nanometer) if u.is_quantity(value) else value


def _is_hydrogen(atom):
    """Return True for hydrogens of an OpenMM topology atom (element based, name as fallback)."""
    if atom.element is not None:
        return atom.element.atomic_number == 1
    return atom.name.startswith('H')


def getHeavyAtomsInSystem(combined_pmd):
    """Return the indices of the heavy atoms of the receptor chunk.

    Solvent, ligand and bulk ions (``NON_CHUNK_RESIDUES``) are excluded; cofactors and
    structural metals that are part of the receptor are kept, as in the AMBER mask
    ``(:chunk_residues & !@H=)``.

    Args:
        combined_pmd (parmed.Structure): the prepared protein-ligand system.

    Returns:
        list of int: atom indices to restrain.
    """
    return [atom.index for atom in combined_pmd.topology.atoms()
            if atom.residue.name not in NON_CHUNK_RESIDUES and not _is_hydrogen(atom)]


def getRestrainedAtoms(combined_pmd, waters_to_restrain=0):
    """Atoms restrained to their reference positions during equilibration, MD and SMD.

    The chunk heavy atoms (:func:`getHeavyAtomsInSystem`) plus all atoms of the first
    ``waters_to_restrain`` water molecules, which are the structural waters retained during the
    preparation (e.g. a water bridge steered with water steering). Mirrors the AMBER mask
    ``(:chunk & !@H=) | :waters``.

    Args:
        combined_pmd (parmed.Structure): the prepared protein-ligand system.
        waters_to_restrain (int): number of waters, counted from the first one, to restrain.

    Returns:
        list of int: sorted atom indices.
    """
    atoms = set(getHeavyAtomsInSystem(combined_pmd))
    if waters_to_restrain:
        waters = [residue for residue in combined_pmd.topology.residues() if residue.name in WATER_RESIDUES]
        if len(waters) < waters_to_restrain:
            raise ValueError(f'{waters_to_restrain} waters to restrain requested but the system has {len(waters)}.')
        atoms.update(atom.index for residue in waters[:waters_to_restrain] for atom in residue.atoms())
    return sorted(atoms)


def applyHarmonicPositionalRestraints(system, forceConstantInKcalPerMolePerAngSquared, positions,
                                      indexOfAtomsToBeModified, parameter_name='k'):
    """Restrain atoms to reference positions, mimicking AMBER's ``restraint_wt``.

    ``E = k * periodicdistance(x, x0)^2`` with ``k`` stored as the global context parameter
    ``parameter_name`` (kJ/mol/nm^2), so it can be changed later with
    ``context.setParameter``.

    Args:
        system (openmm.System): system to which the force is added.
        forceConstantInKcalPerMolePerAngSquared (float): AMBER-style force constant.
        positions (list): reference positions for all particles (Quantity or nm).
        indexOfAtomsToBeModified (list of int): atoms to restrain.
        parameter_name (str): name of the global force-constant parameter.

    Returns:
        openmm.CustomExternalForce: the force added to ``system``.
    """
    force = mm.CustomExternalForce(f'{parameter_name}*periodicdistance(x, y, z, x0, y0, z0)^2')
    force.addGlobalParameter(parameter_name, _to_kj_per_nm2(forceConstantInKcalPerMolePerAngSquared))
    force.addPerParticleParameter('x0')
    force.addPerParticleParameter('y0')
    force.addPerParticleParameter('z0')
    if u.is_quantity(positions):
        positions = positions.value_in_unit(u.nanometer)
    for i in indexOfAtomsToBeModified:
        position = positions[i]
        if u.is_quantity(position):
            position = position.value_in_unit(u.nanometer)
        force.addParticle(i, list(position))
    system.addForce(force)
    return force


def applyLigandChunkRestraint(system, k2, k3, R2, R3, R4, indexOfAffectedAtoms, prefix='',
                              switch_parameter=None):
    """Add the flat-bottom restraint that keeps the key H-bond formed during MD (``dist_md.rst``).

    Harmonic with ``k2`` below ``R2``, flat between ``R2`` and ``R3``, harmonic with ``k3``
    between ``R3`` and ``R4`` and linear beyond ``R4`` (AMBER convention, no factor 1/2).

    Args:
        system (openmm.System): system to which the force is added.
        k2, k3 (float): force constants in kcal/mol/A^2.
        R2, R3, R4 (Quantity): distances of the flat-bottom well.
        indexOfAffectedAtoms (sequence): (protein atom index, ligand atom index, ...).
        prefix (str): prefix for the global parameter names (avoids clashes between forces).
        switch_parameter (str, optional): name of an extra global parameter (default 1) that
            multiplies the energy, so the restraint can be switched off with
            ``context.setParameter(switch_parameter, 0)``.

    Returns:
        openmm.CustomBondForce: the force added to ``system``.
    """
    p = prefix
    energy = f'step({p}R2 - r) * f1 + step(r - {p}R3) * select(step(r - {p}R4), f3, f2)'
    if switch_parameter:
        energy = f'{switch_parameter} * ({energy})'
    restraint_force = mm.CustomBondForce(';'.join([
        energy,
        f'f1 = {p}k2 * (r - {p}R2)^2',
        f'f2 = {p}k3 * (r - {p}R3)^2',
        f'f3 = {p}k3 * ({p}R4 - {p}R3) * (2 * r - {p}R4 - {p}R3)',
    ]))
    restraint_force.addGlobalParameter(f'{p}k2', _to_kj_per_nm2(k2))
    restraint_force.addGlobalParameter(f'{p}k3', _to_kj_per_nm2(k3))
    restraint_force.addGlobalParameter(f'{p}R2', _to_nm(R2))
    restraint_force.addGlobalParameter(f'{p}R3', _to_nm(R3))
    restraint_force.addGlobalParameter(f'{p}R4', _to_nm(R4))
    if switch_parameter:
        restraint_force.addGlobalParameter(switch_parameter, 1.0)
    restraint_force.addBond(int(indexOfAffectedAtoms[0]), int(indexOfAffectedAtoms[1]))
    restraint_force.setUsesPeriodicBoundaryConditions(True)
    system.addForce(restraint_force)
    return restraint_force


def addSteeringRestraint(system, protein_atom, ligand_atom, spring_constant=50.0, r0=0.25,
                         prefix='smd_'):
    """Add the AMBER ``jar=1`` steering restraint ``E = k * (r - r0)^2`` between two atoms.

    ``r0`` is a global parameter (nm) moved during the SMD, and the energy derivative with
    respect to it is enabled so the work can be accumulated exactly without downloading
    coordinates: ``dE/dr0 = -2k (r - r0)``, i.e. AMBER's reported force. The force lives in
    its own force group (``SMD_FORCE_GROUP``).

    Args:
        system (openmm.System): system to which the force is added.
        protein_atom, ligand_atom (int): atom indices of the H-bond.
        spring_constant (float): k in kcal/mol/A^2 (AMBER ``rk2``).
        r0 (float): initial target distance in nm.
        prefix (str): prefix for the global parameters (``{prefix}k`` and ``{prefix}r0``).

    Returns:
        openmm.CustomBondForce: the force added to ``system``.
    """
    force = mm.CustomBondForce(f'{prefix}k*(r - {prefix}r0)^2')
    force.addGlobalParameter(f'{prefix}k', _to_kj_per_nm2(spring_constant))
    force.addGlobalParameter(f'{prefix}r0', r0)
    force.addEnergyParameterDerivative(f'{prefix}r0')
    force.addBond(int(protein_atom), int(ligand_atom), [])
    force.setUsesPeriodicBoundaryConditions(True)
    force.setForceGroup(SMD_FORCE_GROUP)
    system.addForce(force)
    return force


def get_platform(gpu_id=None, precision='mixed', deterministic_forces=False, threads=None):
    """Select the OpenMM platform and its properties.

    Args:
        gpu_id (int or str, optional): GPU device index. ``None`` runs on the CPU.
        precision (str): ``'mixed'`` (default, the analogue of AMBER's SPFP), ``'single'``
            or ``'double'``. Double precision is very slow on consumer GPUs.
        deterministic_forces (bool): request bitwise reproducible forces (~20% slower).
        threads (int, optional): number of CPU threads for the CPU platform.

    Returns:
        tuple: (openmm.Platform, dict of platform properties).
    """
    properties = {}
    if gpu_id is not None and str(gpu_id).lower() not in ('', 'none', 'cpu'):
        for name in ('CUDA', 'OpenCL'):
            try:
                platform = mm.Platform.getPlatformByName(name)
                break
            except Exception:
                continue
        else:
            raise RuntimeError('A GPU was requested but neither the CUDA nor the OpenCL OpenMM platform is available.')
        properties['DeviceIndex'] = str(gpu_id)
        properties['Precision'] = precision
    else:
        platform = mm.Platform.getPlatformByName('CPU')
        if threads:
            properties['Threads'] = str(threads)
    if deterministic_forces:
        properties['DeterministicForces'] = 'true'
    return platform, properties


def setUpNPTEquilibration(system, combined_pmd, platform, platformProperties, positions, velocities,
                          dt=0.002, temperature=300.0, pressure=1.0, friction=4.0):
    """Add a Monte Carlo barostat to ``system`` and return a ready-to-run NPT simulation.

    Args:
        system (openmm.System): system without barostat (it is modified in place).
        combined_pmd (parmed.Structure): structure providing the topology.
        platform, platformProperties: as returned by ``get_platform``.
        positions, velocities: initial coordinates and velocities.
        dt (float): time step in ps.
        temperature (float): temperature in K.
        pressure (float): pressure in bar.
        friction (float): Langevin collision rate in 1/ps.

    Returns:
        openmm.app.Simulation
    """
    system.addForce(mm.MonteCarloBarostat(pressure * u.bar, temperature * u.kelvin))
    integrator = mm.LangevinIntegrator(temperature * u.kelvin, friction / u.picosecond, dt * u.picosecond)
    simulation = app.Simulation(combined_pmd.topology, system, integrator, platform, platformProperties)
    simulation.context.setPositions(positions)
    simulation.context.setVelocities(velocities)
    return simulation


def load_complex_pickle(fname='complex_system.pickle'):
    """Load the prepared system written by the preparation step.

    The pickle stores ``[parmed.Structure, protein_atom_idx, ligand_atom_idx, distance_A]``
    (the interaction is appended after the interaction search).

    Returns:
        tuple: (parmed.Structure, list with the key interaction, possibly empty).
    """
    with open(fname, 'rb') as fh:
        content = pickle.load(fh)
    return content[0], list(content[1:])


def loadPickle(fname='complex_system.pickle'):
    """Return only the parmed structure stored in the preparation pickle."""
    return load_complex_pickle(fname)[0]


def structure_has_hmr(structure, threshold=1.5):
    """True if the (non-water) hydrogens of ``structure`` carry repartitioned masses."""
    for atom in structure.atoms:
        if atom.element == 1 and atom.residue.name not in WATER_RESIDUES:
            return atom.mass > threshold
    return False


def apply_hmr(structure):
    """Repartition hydrogen masses (3.024 Da, waters untouched) in place using parmed."""
    import parmed
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        parmed.tools.actions.HMassRepartition(structure).execute()
    return structure
