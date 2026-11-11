#!/usr/bin/env python
"""openduck-crooks: bidirectional steered MD (break and reform an interaction) analysed with the Crooks theorem.

An example of a new protocol built on the OpenDUck library. Each cycle runs DUck-style MD, an
optional approach of the spring to the bound distance, a forward pull that breaks the interaction
(2.5 -> 5 A by default) and a reverse pull that reforms it with new velocities. The free energy
comes from the crossing of the forward and reverse work distributions (Crooks fluctuation theorem),
with BAR, the Gaussian intersection and Jarzynski for comparison.

Subcommands:
    openmm-full-protocol      prepare (chunk, parametrize, solvate), equilibrate and run the pulls in OpenMM
    openmm-from-equilibrated  run the pulls from an equilibrated checkpoint (e.g. ``openduck openmm-prepare --do-equilibrate``)
    openmm-from-amber         run the pulls from an AMBER topology/coordinates (e.g. MCPB.py metal sites or an
                              ``openduck amber-prepare`` folder)
    amber-prepare             prepare the system and write the pmemd inputs and job script of the same protocol
    report                    Crooks / BAR / Jarzynski free energies, WQB and plots of finished runs (both engines)

Like ``openduck``, every subcommand accepts its options on the command line or in a YAML file (``-y``).
The receptor side of the interaction can be a retained water (``--water-steering``, water bridges) or a
metal (e.g. ``_ZN1_330_ZN`` with ``--custom-forcefield`` for its bonded parameters). With ``--equilibrations N``
the cycles are run after N independent equilibrations (``EQ_<k>/``, ``--cycles`` per equilibration), which
``report`` pools.
"""
import argparse
import os
import pickle
import sys

if __package__ in (None, ''):  # run as a file from a checkout: use that checkout's library
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from duck.utils.exceptions import EquilibrationError  # noqa: E402
from scripts.openduck import (_add_chunk_arguments, _add_engine_arguments, _add_equilibrations_argument,  # noqa: E402
                              _add_main_arguments, _add_parametrization_arguments, _add_water_steering_arguments,
                              _append_interaction_to_pickle, _check_equilibrations, _chunk_receptor, _equilibrate,
                              _prepared_with_interaction, _run_with_equilibrations, apply_yaml_defaults)

REQUIRED = {
    'openmm-full-protocol': ('ligand', 'receptor', 'interaction'),
    'openmm-from-equilibrated': ('equilibrated_system', 'pickle'),
    'openmm-from-amber': ('topology', 'coordinates'),
    'amber-prepare': ('ligand', 'receptor', 'interaction'),
}


# ---------------------------------------------------------------------------------------------- arguments
def _add_pull_arguments(group, engine='openmm'):
    group.add_argument('-n', '--cycles', type=int, default=20,
                       help='Number of cycles (per equilibration with --equilibrations); each gives one forward and one reverse '
                            'work. Default = 20.')
    group.add_argument('-m', '--md-length', type=float, default=0.5, help='MD between cycles in ns. Default = 0.5.')
    group.add_argument('--start-distance', type=float, default=2.5, help='Bound distance (A) where the pulls start/end. Default = 2.5.')
    group.add_argument('--end-distance', type=float, default=5.0, help='Quasi-bound distance (A) of the pulls. Default = 5.0.')
    group.add_argument('--speed', type=float, default=5.0, help='Steering speed in A/ns, both directions. Default = 5 (DUck).')
    group.add_argument('--spring-constant', type=float, default=50.0,
                       help='Steering spring k of E = k (r - r0)^2 in kcal/mol/A^2 (AMBER rk2). Default = 50 (DUck; the thesis '
                            'Crooks protocol used 500).')
    group.add_argument('-T', '--temperature', type=float, default=300.0, help='Temperature of MD and pulls (K). Default = 300.')
    group.add_argument('--approach-length', type=float, default=0.1,
                       help='ns to move the spring from the current distance to the start distance before each forward pull, '
                            'so it starts close to equilibrium with the spring. 0 starts the pull from the MD like a DUck SMD. '
                            'Default = 0.1 (the thesis protocol used 1).')
    group.add_argument('--approach-spring-constant', type=float, default=None,
                       help='Spring constant of the approach in kcal/mol/A^2. Default: the steering spring constant.')
    group.add_argument('--md-restraint', type=float, nargs=5, default=[2.0, 3.0, 4.0, 1.0, 10.0], metavar=('R2', 'R3', 'R4', 'K2', 'K3'),
                       help='Flat-bottom restraint keeping the interaction formed during MD (dist_md.rst): harmonic (K2) below R2, '
                            'flat to R3, harmonic (K3) to R4 and linear beyond. Default = 2 3 4 1 10 (DUck).')
    group.add_argument('-W', '--wqb-threshold', type=float, default=None,
                       help='Stop after a cycle whose forward pull gives a WQB below this value. Default: run all cycles.')
    group.add_argument('--trajectory-interval', type=float, default=0.0, help='ps between saved frames (0: no trajectories). Default = 0.')
    if engine == 'openmm':
        group.add_argument('--reverse-hold', type=float, default=0.0,
                           help='ns with the spring fixed at the end distance before each reverse pull. Default = 0 (thesis protocol).')
        group.add_argument('--keep-velocities', action='store_true',
                           help='Continue the reverse pull with the velocities of the forward one instead of drawing new ones.')
        group.add_argument('--force-constant-chunk', type=float, default=1.0,
                           help='Restraint on the chunk heavy atoms in kcal/mol/A^2 (AMBER restraint_wt). Default = 1.0.')


def _add_openmm_options(parser):
    group = parser.add_argument_group('Engine arguments')
    group.add_argument('-H', '--HMR', action='store_true', help='Run at dt = 4 fs (hydrogen masses are repartitioned if needed).')
    _add_engine_arguments(group)
    group.add_argument('-t', '--threads', type=int, default=None, help='CPU threads when running on the CPU.')


def parse_input(argv=None):
    parser = argparse.ArgumentParser(description='Bidirectional steered MD and Crooks fluctuation theorem free energies with OpenDUck.')
    parser.set_defaults(command=None)
    modes = parser.add_subparsers(title='Choose one of the following actions', metavar='', dest='command')

    full = modes.add_parser('openmm-full-protocol', help='Preparation, equilibration and bidirectional pulls in OpenMM.',
                            description='Prepare the complex (optional chunking, parametrization, solvation), equilibrate it and '
                                        'run the cycles of MD, forward and reverse pulls in OpenMM.')
    group = full.add_argument_group('Main arguments')
    _add_main_arguments(group, 'the bidirectional protocol')
    group.add_argument('--keep-all-files', action='store_true', help='Keep the intermediate files of the preparation.')
    group.add_argument('--resume', action='store_true', help='Reuse the preparation and equilibration found in the folder.')
    _add_chunk_arguments(full)
    group = full.add_argument_group('Parameterization arguments')
    _add_parametrization_arguments(group)
    _add_water_steering_arguments(group)
    group.add_argument('-F', '--force-constant-eq', type=float, default=1, help='Restraint during equilibration (kcal/mol/A^2).')
    _add_equilibrations_argument(group)
    _add_pull_arguments(full.add_argument_group('Pull arguments'))
    _add_openmm_options(full)
    full.set_defaults(func=do_full_protocol)

    equil = modes.add_parser('openmm-from-equilibrated', help='Bidirectional pulls in OpenMM from an equilibrated system.',
                             description='Run the cycles from an equilibrated checkpoint and its preparation pickle (with the key '
                                         'interaction), e.g. from openduck openmm-prepare --do-equilibrate.')
    equil.add_argument('-y', '--yaml-input', type=str, default=None, help='Input yaml file with the arguments.')
    equil.add_argument('-s', '--equilibrated-system', default=None, help='Equilibrated system (*.chk).')
    equil.add_argument('-p', '--pickle', default=None, help='Pickle output from the preparation.')
    equil.add_argument('--waters-to-restrain', default=None, type=int,
                       help='Number of (retained) waters, counted from the first one, restrained with the chunk (water steering).')
    _add_pull_arguments(equil.add_argument_group('Pull arguments'))
    _add_openmm_options(equil)
    equil.set_defaults(func=do_from_equilibrated)

    amber_sys = modes.add_parser('openmm-from-amber', help='Bidirectional pulls in OpenMM from an AMBER topology and coordinates.',
                                 description='Run the cycles from an AMBER prmtop and coordinates (e.g. a tleap/MCPB.py system with a '
                                             'metal site, or an openduck amber-prepare folder). The interaction is given by its atom '
                                             'indices or located through the receptor chunk. With --skip-equilibration the coordinates '
                                             '(and velocities, if present) are used as the equilibrated state, e.g. an AMBER 3_eq.rst.')
    amber_sys.add_argument('-y', '--yaml-input', type=str, default=None, help='Input yaml file with the arguments.')
    amber_sys.add_argument('--topology', default=None, help='AMBER topology (prmtop).')
    amber_sys.add_argument('-c', '--coordinates', default=None, help='AMBER coordinates or restart (inpcrd, rst7).')
    amber_sys.add_argument('--atom-indices', type=int, nargs=2, default=None, metavar=('RECEPTOR', 'LIGAND'),
                           help='1-based atom numbers of the interaction, as in AMBER (iat of dist_duck.rst).')
    amber_sys.add_argument('-i', '--interaction', default=None, help='Receptor atom as chain_resname_resid_atomname (with --receptor).')
    amber_sys.add_argument('-r', '--receptor', default=None, help='Receptor chunk (PDB or mol2) with the original residue numbering.')
    amber_sys.add_argument('--waters-to-restrain', default=None, type=int,
                           help='Number of (retained) waters, counted from the first one, restrained with the chunk (water steering).')
    amber_sys.add_argument('--skip-equilibration', action='store_true', help='The coordinates are already equilibrated.')
    amber_sys.add_argument('-F', '--force-constant-eq', type=float, default=1, help='Restraint during equilibration (kcal/mol/A^2).')
    amber_sys.add_argument('--keep-all-files', action='store_true', help='Keep the intermediate equilibration files.')
    amber_sys.add_argument('--resume', action='store_true', help='Reuse the equilibrated system and the pulls found in the folder.')
    _add_equilibrations_argument(amber_sys)
    _add_pull_arguments(amber_sys.add_argument_group('Pull arguments'))
    _add_openmm_options(amber_sys)
    amber_sys.set_defaults(func=do_from_amber)

    amber = modes.add_parser('amber-prepare', help='Preparation, pmemd inputs and job script of the bidirectional protocol.',
                             description='Prepare the complex and write the pmemd inputs (approach.in, forward.in, reverse.in, md.in, '
                                         '...) and the job script; the pulls are written to CYCLE_<i>/forward.dat and reverse.dat.')
    _add_main_arguments(amber.add_argument_group('Main arguments'), 'the bidirectional protocol in AMBER')
    _add_chunk_arguments(amber)
    group = amber.add_argument_group('Parameterization arguments')
    _add_parametrization_arguments(group, water_models=('tip3p', 'spce', 'tip4pew'))
    _add_water_steering_arguments(group)
    group.add_argument('--ligand-hb-elements', default=[7, 8], type=int, nargs='+',
                       help='Atomic numbers accepted for the ligand atom of the interaction. Default = 7 8.')
    group.add_argument('-H', '--HMR', action='store_true', help='Hydrogen mass repartitioning (4 fs time step).')
    group.add_argument('-q', '--queue-template', type=str, default='local', help='Queue template (local, Slurm, SGE). Default = local.')
    group.add_argument('--seed', default='-1', type=str, help='Seed for the AMBER inputs (-1: random; required with --equilibrations).')
    _add_equilibrations_argument(group)
    group.add_argument('--keep-all-files', action='store_true', help='Keep the intermediate files of preparation and simulations.')
    _add_pull_arguments(amber.add_argument_group('Pull arguments'), engine='amber')
    amber.set_defaults(func=do_amber_prepare)

    report = modes.add_parser('report', help='Free energies and plots of finished bidirectional runs.',
                              description='Crooks (KDE crossing), Gaussian intersection, BAR and Jarzynski free energies with bootstrap '
                                          'errors, plus the DUck WQB and dG_QB of the forward pulls, for every folder matching the pattern.')
    report.add_argument('-p', '--pattern', type=str, default='.', help='Wildcard pattern of the run folders. Default = current folder.')
    report.add_argument('-o', '--output', default='stdout', help='Output file. Default: print to stdout.')
    report.add_argument('-of', '--output-format', default='tbl', choices=('csv', 'tbl'), help='Output format. Default = tbl.')
    report.add_argument('--plot', action='store_true', help='Write crooks_report.png (works, distributions, free energy) in each folder.')
    report.add_argument('-i', '--iterations', type=int, default=100, help='Bootstrap iterations. Default = 100.')
    report.add_argument('-s', '--sample-size', type=int, default=None, help='Pulls drawn per direction and iteration. Default: all of them.')
    report.add_argument('--bandwidth', default='scott',
                        help='KDE bandwidth: scott, silverman or a width in kcal/mol. Default = scott.')
    report.add_argument('-T', '--temperature', type=float, default=None, help='Temperature (K). Default: from protocol.json (or 300).')
    report.add_argument('--seed', type=int, default=None, help='Seed of the bootstrap (reproducible reports).')
    report.set_defaults(func=do_report)

    args = parser.parse_args(argv)
    if args.command is None:
        return args, parser
    subparser = modes.choices[args.command]
    if getattr(args, 'yaml_input', None):
        apply_yaml_defaults(subparser, args.yaml_input)
        args = parser.parse_args(argv)
    missing = [name for name in REQUIRED.get(args.command, ()) if getattr(args, name, None) in (None, '')]
    if missing:
        subparser.error('missing ' + ', '.join('--' + name.replace('_', '-') for name in missing) + ' (command line or yaml).')
    if args.command == 'openmm-from-amber' and not args.atom_indices and not (args.interaction and args.receptor):
        subparser.error('give the interaction with --atom-indices, or with --interaction and --receptor.')
    if getattr(args, 'water_steering', False) and not args.waters_to_restrain:
        args.waters_to_restrain = 1
    _check_equilibrations(args, subparser)
    if getattr(args, 'skip_equilibration', False) and args.equilibrations > 1:
        subparser.error('--skip-equilibration uses the given coordinates as the only equilibrated state; it cannot be '
                        'combined with --equilibrations.')
    if getattr(args, 'start_distance', None) == getattr(args, 'end_distance', None) and args.command != 'report':
        subparser.error('--start-distance and --end-distance must differ.')
    if args.command == 'report':
        try:
            args.bandwidth = float(args.bandwidth)
        except ValueError:
            if args.bandwidth not in ('scott', 'silverman'):
                subparser.error('--bandwidth must be scott, silverman or a number (kcal/mol).')
    return args, parser


# ---------------------------------------------------------------------------------------------- OpenMM
def _refuse_to_mix_runs(args):
    '''A new preparation or equilibration must not be mixed with the pulls of a previous run.'''
    from duck.utils.crooks import OPENMM_DIR
    from duck.utils.equilibrations import find_equilibrations
    if not args.resume and (os.path.isdir(OPENMM_DIR) or find_equilibrations('.')):
        sys.exit(f'{OPENMM_DIR}/ or EQ_*/ already exist: use --resume to continue that run, or remove them to start a new '
                 'one (a new preparation or equilibration would be mixed with the existing pulls).')
    if os.path.isdir(OPENMM_DIR) if args.equilibrations > 1 else find_equilibrations('.'):
        sys.exit(f'This folder holds a run with {f"one equilibration ({OPENMM_DIR}/)" if args.equilibrations > 1 else "independent equilibrations (EQ_*/)"}: '
                 'resume it with its original --equilibrations, or use another folder.')


def _run_pulls(args, checkpoint='equil.chk', pickle_file='complex_system.pickle'):
    '''Run the bidirectional cycles in the current directory (outputs in crooks_runs/).'''
    from duck.steps.bidirectional import run_bidirectional_cycles
    r2, r3, r4, k2, k3 = args.md_restraint
    return run_bidirectional_cycles(
        checkpoint, pickle_file, n_cycles=args.cycles, md_length=args.md_length, start_dist=args.start_distance,
        end_dist=args.end_distance, speed=args.speed / 1000.0, temperature=args.temperature,
        approach_length=args.approach_length, approach_spring_k=args.approach_spring_constant,
        reverse_hold=args.reverse_hold, reassign_velocities=not args.keep_velocities, gpu_id=args.gpu_id, hmr=args.HMR,
        wqb_threshold=args.wqb_threshold, traj_interval_ps=args.trajectory_interval,
        waters_to_restrain=args.waters_to_restrain or 0, smd_spring_k=args.spring_constant,
        chunk_restraint_k=args.force_constant_chunk, hb_restraint_distances=(r2, r3, r4), hb_restraint_k2=k2,
        hb_restraint_k3=k3, precision=args.precision, integrator=args.integrator,
        deterministic_forces=args.deterministic_forces, threads=args.threads)


def _pulls_with_equilibrations(args):
    '''Bidirectional cycles after each of the args.equilibrations independent equilibrations (EQ_<k>/).'''
    # a cycle below the WQB threshold ends the run early (fewer pulls than cycles) and stops the other equilibrations
    return _run_with_equilibrations(args, lambda checkpoint, pickle_file: len(_run_pulls(args, checkpoint, pickle_file)) < args.cycles)


def do_full_protocol(args):
    '''Prepare and equilibrate the complex in the current folder, then run the bidirectional cycles.'''
    from duck.steps.parametrize import prepare_system
    from duck.utils.cal_ints import find_interaction
    _refuse_to_mix_runs(args)
    if args.threads:
        os.environ['OPENMM_CPU_THREADS'] = str(args.threads)
    if args.resume and _prepared_with_interaction('complex_system.pickle'):
        print('complex_system.pickle already exists, skipping the preparation (resume mode).')
    else:
        chunked_file = _chunk_receptor(args)
        os.environ['OMP_NUM_THREADS'] = '1'  # otherwise sqm uses all threads and is slower
        prepare_system(args.ligand, chunked_file, forcefield_str=f'{args.protein_forcefield}.xml', water_ff_str=args.water_model,
                       small_molecule_ff=args.small_molecule_forcefield, waters_to_retain=args.waters_to_retain,
                       custom_forcefield=args.custom_forcefield, box_buffer_distance=args.solvent_buffer_distance,
                       ionicStrength=args.ionic_strength, fix_ligand_file=args.fix_ligand, clean_up=not args.keep_all_files,
                       hmr=args.HMR)
        results = find_interaction(args.interaction, args.waters_to_retain if args.water_steering else args.receptor)
        print('Key interaction (receptor atom, ligand atom, distance):', results)
        _append_interaction_to_pickle(results)
    if args.equilibrations > 1:
        return _pulls_with_equilibrations(args)
    _equilibrate(args)
    _run_pulls(args)


def do_from_equilibrated(args):
    _run_pulls(args, checkpoint=args.equilibrated_system, pickle_file=args.pickle)


def _equilibrated_checkpoint_from_amber(structure, key_interaction, args, checkpoint='equil.chk'):
    '''Write an OpenMM checkpoint with the AMBER coordinates/velocities (already equilibrated system).'''
    import parmed
    from duck.steps.production import DuckSimulation
    from duck.utils.compat import unit as u
    simulation = DuckSimulation(structure, key_interaction, gpu_id=args.gpu_id, hmr=args.HMR, precision=args.precision,
                                threads=args.threads, temperature=args.temperature)
    velocities = getattr(structure, 'velocities', None)
    if velocities is None:
        try:  # parmed.load_file does not keep the velocities of an AMBER restart
            restart = parmed.amber.Rst7.open(args.coordinates)
            velocities = restart.velocities.reshape(-1, 3) if restart.hasvels else None
        except Exception:
            velocities = None
    if velocities is not None:
        simulation.context.setVelocities(u.Quantity(velocities, u.angstrom / u.picosecond))  # parmed: A/ps
        print(f'Using the coordinates and velocities of {args.coordinates} as the equilibrated state.')
    else:
        simulation.reassign_velocities(args.temperature)
        print(f'{args.coordinates} has no velocities: drawing them at {args.temperature:g} K.')
    simulation.save_checkpoint(checkpoint)
    return checkpoint


def do_from_amber(args):
    '''Bidirectional cycles from an AMBER topology and coordinates.'''
    import parmed
    from duck.utils.cal_ints import find_interaction_amber_input
    _refuse_to_mix_runs(args)
    if args.resume and os.path.isfile('complex_system.pickle') and args.equilibrations > 1:
        print('complex_system.pickle already exists, continuing the equilibrations and pulls (resume mode).')
        return _pulls_with_equilibrations(args)
    if args.resume and os.path.isfile('equil.chk') and os.path.isfile('complex_system.pickle'):
        print('equil.chk and complex_system.pickle already exist, continuing the pulls (resume mode).')
        return _run_pulls(args)
    structure = parmed.load_file(args.topology, args.coordinates)
    if args.atom_indices:
        receptor_atom, ligand_atom = (i - 1 for i in args.atom_indices)
        distance = float(sum((a - b) ** 2 for a, b in zip(structure.coordinates[receptor_atom],
                                                         structure.coordinates[ligand_atom])) ** 0.5)
        key_interaction = [receptor_atom, ligand_atom, distance]
    else:
        key_interaction = find_interaction_amber_input(combined_pmd=structure, chunk_file=args.receptor, res_atom=args.interaction)
    print('Key interaction (receptor atom, ligand atom, distance; 0-based):', key_interaction)
    with open('complex_system.pickle', 'wb') as fh:
        pickle.dump([structure] + list(key_interaction), fh, protocol=pickle.HIGHEST_PROTOCOL)
    if args.skip_equilibration:
        checkpoint = _equilibrated_checkpoint_from_amber(structure, key_interaction, args)
    elif args.equilibrations > 1:
        return _pulls_with_equilibrations(args)
    else:
        _equilibrate(args, resume=False)
        checkpoint = 'equil.chk'
    _run_pulls(args, checkpoint=checkpoint)


# ---------------------------------------------------------------------------------------------- AMBER
def do_amber_prepare(args):
    '''Prepare the complex and write the pmemd inputs and job script of the bidirectional protocol.'''
    from duck.steps.bidirectional import write_protocol
    from duck.utils.amber_inputs import Crooks_amber_templates, Crooks_queue_templates
    from scripts.openduck import prepare_sys_for_amber
    chunk_file = _chunk_receptor(args)
    options = dict(start_dist=args.start_distance, end_dist=args.end_distance, speed=args.speed, spring_k=args.spring_constant,
                   approach_length=args.approach_length, approach_spring_k=args.approach_spring_constant,
                   trajectory_interval=args.trajectory_interval, md_length=args.md_length, temperature=args.temperature,
                   md_restraint=args.md_restraint)
    prepared = prepare_sys_for_amber(
        args.ligand, args.receptor, chunk_file, args.interaction, args.HMR, small_molecule_forcefield=args.small_molecule_forcefield,
        water_ff_str=args.water_model, forcefield_str=f'{args.protein_forcefield}.xml', ionic_strength=args.ionic_strength,
        box_buffer_distance=args.solvent_buffer_distance, waters_to_retain=args.waters_to_retain,
        custom_forcefield=args.custom_forcefield, seed=args.seed, fix_ligand_file=args.fix_ligand,
        clean_up=not args.keep_all_files, water_steering=args.water_steering, waters_to_restrain=args.waters_to_restrain,
        lig_HB_elements=args.ligand_hb_elements, inputs_class=Crooks_amber_templates, inputs_options=options)
    Crooks_queue_templates(prepared[1:], approach=args.approach_length > 0, wqb_threshold=args.wqb_threshold or 0,
                           replicas=args.cycles, hmr=args.HMR, keep_intermediate_files=args.keep_all_files,
                           equilibrations=args.equilibrations).write_queue_file(kind=args.queue_template)
    write_protocol('.', engine='amber', start_dist=args.start_distance, end_dist=args.end_distance, speed_A_per_ns=args.speed,
                   temperature=args.temperature, spring_k=args.spring_constant, md_length_ns=args.md_length,
                   approach_length_ns=args.approach_length, approach_spring_k=args.approach_spring_constant or args.spring_constant,
                   reverse_hold_ns=0.0, reassign_velocities=True, n_cycles=args.cycles, equilibrations=args.equilibrations,
                   waters_to_restrain=args.waters_to_restrain or 0, key_interaction=[int(x) for x in prepared[1:3]])
    where = 'EQ_<k>/CYCLE_<i>/' if args.equilibrations > 1 else 'CYCLE_<i>/'
    print(f'AMBER inputs written; launch duck_queue.q (pulls in {where}forward.dat and reverse.dat).')


# ---------------------------------------------------------------------------------------------- report
def do_report(args):
    import glob
    import pandas as pd
    from duck.utils.crooks import REPORT_COLUMNS, analyse_folder
    folders = sorted(glob.glob(args.pattern))
    if not folders:
        sys.exit(f'No folders match the pattern {args.pattern}')
    rows = []
    for folder in folders:
        try:
            rows.append(analyse_folder(folder, temperature=args.temperature, bootstrap_samples=args.iterations,
                                       sample_size=args.sample_size, bandwidth=args.bandwidth, seed=args.seed, plot=args.plot))
        except (OSError, ValueError) as error:
            sys.stderr.write(f'Warning: {folder} skipped: {error}\n')
    if not rows:
        sys.exit('No bidirectional runs could be analysed.')
    df = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    output = sys.stdout if args.output == 'stdout' else args.output
    df.to_csv(output, index=False, sep='\t' if args.output_format == 'tbl' else ',', float_format='%.3f')


def main(argv=None):
    args, parser = parse_input(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        args.func(args)
    except EquilibrationError as error:
        sys.stderr.write(f'ERROR: {error}\n')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
