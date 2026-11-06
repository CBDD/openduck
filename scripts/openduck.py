#!/usr/bin/env python
"""OpenDUck command line interface.

Subcommands:
    openmm-prepare            chunk (optional), parametrize, solvate and equilibrate a system for OpenMM
    openmm-full-protocol      openmm-prepare + DUck production (MD/SMD cycles) in OpenMM
    openmm-from-equilibrated  DUck production from an equilibrated checkpoint and its preparation pickle
    openmm-from-amber         equilibration + DUck production starting from an AMBER topology/coordinates
    amber-prepare             chunk (optional), parametrize and solvate a system and write AMBER inputs/queue files
    report                    WQB / Jarzynski report of finished DUck runs (AMBER or OpenMM)
    chunk                     reduce a receptor to a pocket around the key interaction
    prepare-cofactor          convert frcmod + mol2 parameters to an OpenMM XML force field

Every subcommand accepts its options on the command line or in a YAML file (``-y``) whose keys
are the option names with underscores (e.g. ``smd_cycles: 20``; ``ligand_mol`` and
``receptor_pdb`` are accepted for ``ligand`` and ``receptor``). Options given on the command
line take precedence over the YAML file.

With ``--equilibrations N`` (openmm-full-protocol, openmm-from-amber, amber-prepare) the prepared system is
equilibrated N times independently and each equilibration runs ``--smd-cycles`` cycles in ``EQ_<k>/``;
``report`` pools them (see :mod:`duck.utils.equilibrations`).
"""
import argparse
import glob
import multiprocessing as mp
import os
import pickle
import shutil
import sys
import traceback
from contextlib import redirect_stdout

import yaml

from duck.utils.exceptions import EquilibrationError

YAML_ALIASES = {'ligand_mol': 'ligand', 'receptor_pdb': 'receptor'}
REQUIRED = {
    'openmm-prepare': ('ligand', 'receptor', 'interaction'),
    'openmm-full-protocol': ('ligand', 'receptor', 'interaction'),
    'openmm-from-equilibrated': ('equilibrated_system', 'pickle'),
    'openmm-from-amber': ('topology', 'coordinates', 'interaction', 'receptor'),
    'amber-prepare': ('ligand', 'receptor', 'interaction'),
    'chunk': ('receptor', 'interaction'),
    'prepare-cofactor': ('mol2', 'frcmod', 'output'),
}
SDF_EXTENSIONS = ('.sdf', '.sd')


# ---------------------------------------------------------------------------------------------- arguments
def str2bool(value):
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in ('true', 'yes', 'y', '1', 'on'):
        return True
    if str(value).strip().lower() in ('false', 'no', 'n', '0', 'off'):
        return False
    raise argparse.ArgumentTypeError(f'Expected a boolean value, got "{value}".')


def _yaml_value(action, value):
    """Convert a YAML value with the type of the corresponding argparse option."""
    if value is None or (isinstance(value, str) and value.strip().lower() in ('none', 'null', '')):
        return None
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return str2bool(value)
    convert = action.type or (lambda x: x)
    if action.nargs in ('+', '*') or (isinstance(action.nargs, int) and action.nargs > 1):
        values = value.replace(',', ' ').split() if isinstance(value, str) else list(
            value if isinstance(value, (list, tuple)) else [value])
        converted = [convert(str(v)) for v in values]
        if isinstance(action.nargs, int) and len(converted) != action.nargs:
            raise ValueError(f'expected {action.nargs} values, got {len(converted)}')
        bad = [v for v in converted if action.choices and v not in action.choices]
    else:
        converted = convert(str(value)) if not isinstance(value, (list, dict)) else value
        bad = [converted] if action.choices and converted not in action.choices else []
    if bad:
        raise argparse.ArgumentTypeError(f'invalid choice {bad[0]!r} for {action.dest} (choose from {action.choices})')
    return converted


def apply_yaml_defaults(subparser, yaml_file):
    """Use the values of ``yaml_file`` as defaults of ``subparser`` (command-line options still win)."""
    with open(yaml_file) as fh:
        config = yaml.safe_load(fh) or {}
    actions = {action.dest: action for action in subparser._actions}
    defaults = {}
    for key, value in config.items():
        dest = YAML_ALIASES.get(key, key).replace('-', '_')
        if dest not in actions or dest in ('help', 'yaml_input'):
            sys.stderr.write(f'Warning: unknown option "{key}" in {yaml_file} will be ignored.\n')
            continue
        try:
            defaults[dest] = _yaml_value(actions[dest], value)
        except (argparse.ArgumentTypeError, ValueError) as error:
            subparser.error(f'{yaml_file}: {key}: {error}')
    subparser.set_defaults(**defaults)


def _add_main_arguments(group, what='the protocol'):
    group.add_argument('-y', '--yaml-input', type=str, default=None, help=f'Input yaml file with the arguments for {what}.')
    group.add_argument('-l', '--ligand', type=str, default=None, help='Ligand MOL or SDF file (multi-ligand SDF with --batch).')
    group.add_argument('-i', '--interaction', type=str, default=None,
                       help='Receptor atom of the key interaction as chain_resname_resid_atomname (e.g. A_ASN_140_ND2).')
    group.add_argument('-r', '--receptor', type=str, default=None, help='Receptor (or already chunked receptor) PDB file.')


def _add_chunk_arguments(parser):
    group = parser.add_argument_group('Chunking arguments')
    group.add_argument('--do-chunk', action='store_true', help='Chunk the receptor around the interaction atom and add caps.')
    group.add_argument('-c', '--cutoff', type=float, default=9, help='Cutoff distance to define the chunk (in Angstroms). Default = 9 A.')
    group.add_argument('-b', '--ignore-buffers', action='store_true', help='Do not remove buffers (solvent, ions etc.)')


def _add_parametrization_arguments(group, water_models=('tip3p', 'spce')):
    group.add_argument('-f', '--small-molecule-forcefield', type=str.upper, default='SMIRNOFF', choices=('SMIRNOFF', 'GAFF2'),
                       help='Small molecule forcefield to use for parameterization.')
    group.add_argument('-w', '--water-model', default='tip3p', type=str.lower, choices=water_models, help='Water model to parameterize the solvent.')
    group.add_argument('-ff', '--protein-forcefield', default='amber99sb', type=str.lower, choices=('amber99sb', 'amber14-all'),
                       help='Protein forcefield to parameterize the chunked protein.')
    group.add_argument('-ion', '--ionic-strength', default=0.1, type=float,
                       help='Ionic strength (concentration) of the counter ion salts (Na+/Cl-). Default = 0.1 M')
    group.add_argument('-s', '--solvent-buffer-distance', default=10, type=float,
                       help='Buffer distance between the periodic box and the protein (in Angstroms). Default = 10 A')
    group.add_argument('-water', '--waters-to-retain', default='waters_to_retain.pdb', type=str,
                       help='PDB file containing structural water molecules to retain during simulations. Default is waters_to_retain.pdb.')
    group.add_argument('-cf', '--custom-forcefield', default=None, type=str,
                       help='Custom forcefield (in OpenMM XML format) to parameterize e.g. a cofactor or unnatural amino acid present in '
                            'the receptor. Used in addition to the small molecule and protein forcefields.')
    group.add_argument('-fl', '--fix-ligand', action='store_true',
                       help='Some simple fixes for the ligand: ensure tetravalent nitrogens have the right charge assigned and add missing hydrogen atoms.')


def _add_water_steering_arguments(group):
    group.add_argument('--water-steering', default=False, action='store_true',
                       help='Enable water steering, which will use the waters-to-retain as interaction vector to steer the ligand.')
    group.add_argument('--waters-to-restrain', default=None, type=int,
                       help='Number (and order) of waters to restraint. Default is None when executing in the normal mode and 1 when using water-steering.')


def _add_equilibrations_argument(group):
    group.add_argument('--equilibrations', type=int, default=1,
                       help='Independent equilibrations of the prepared system, each followed by its own MD/SMD cycles in '
                            'EQ_<k>/ (the number of cycles is per equilibration). Cycles of one equilibration share its '
                            'structure and restraint reference; spreading them over several equilibrations makes repeated runs '
                            'agree better. Default = 1 (one equilibration in the run folder).')


def _add_batch_arguments(parser, keep_files=False):
    group = parser.add_argument_group('Batch arguments')
    group.add_argument('-B', '--batch', default=False, action='store_true', help='Enable batch processing for multi-ligand SDF.')
    group.add_argument('-t', '--threads', default=1, type=int, help='Number of CPUs for the (batch) preparation.')
    if keep_files:
        group.add_argument('--keep-all-files', default=False, action='store_true',
                           help='Disable cleaning up intermediate files during preparation and simulations.')
    group.add_argument('--resume', default=False, action='store_true',
                       help='Resume mode: ligand folders already prepared are kept and finished steps are skipped instead of overwritten.')
    group.add_argument('-i0', '--index0', default=1, type=int, help='Starting index for naming batch ligands. Default: 1.')
    group.add_argument('-p', '--prefix', default='LIG_target', type=str, help='Prefix to name ligand folder during batch preparation. Default: LIG_target.')


def _add_production_arguments(group):
    group.add_argument('-n', '--smd-cycles', type=int, default=20,
                       help='Number of MD/SMD cycles to perform (per equilibration with --equilibrations).')
    group.add_argument('-m', '--md-length', type=float, default=0.5, help='Length of MD sampling between SMD runs in ns.')
    group.add_argument('-W', '--wqb-threshold', type=float, default=None,
                       help='Minimum WQB threshold; if not reached after each SMD cycle, further simulations will be terminated. '
                            'If not set (default), all SMD cycles will be run.')
    group.add_argument('-v', '--init-velocities', type=float, default=0.00001,
                       help='Steering speed in Angstrom per 2 fs (default 1e-5, i.e. 5 A/ns as in AMBER DUck). '
                            'The speed in A/ns is kept when HMR (4 fs) is used.')
    group.add_argument('-d', '--init-distance', type=float, default=2.5, help='Initial H-bond distance of the SMD in Angstroms. Default = 2.5 A.')
    group.add_argument('--force-constant-chunk', type=float, default=1.0,
                       help='Restraint on the chunk heavy atoms during MD/SMD in kcal/mol/A^2 (AMBER restraint_wt). Default = 1.0.')
    group.add_argument('--smd-spring-constant', type=float, default=50.0,
                       help='Steering spring constant k of E = k (r - r0)^2 in kcal/mol/A^2 (AMBER rk2). Default = 50.')
    group.add_argument('--trajectory-interval', type=float, default=4.0,
                       help='Interval (ps) to save MD/SMD trajectories (DCD); 0 disables them. Default = 4 ps.')


def _add_engine_arguments(group, gpu=True):
    if gpu:
        group.add_argument('-g', '--gpu-id', type=int, default=None, help='GPU ID; if not specified, runs on CPU only.')
    group.add_argument('--precision', type=str.lower, default='mixed', choices=('mixed', 'single', 'double'),
                       help='GPU floating point precision. Default = mixed (double is very slow on consumer GPUs).')
    group.add_argument('--integrator', type=str.lower, default='langevin', choices=('langevin', 'langevin-middle'),
                       help='Integrator for MD/SMD: langevin (leapfrog, as AMBER ntt=3; default) or langevin-middle.')
    group.add_argument('--deterministic-forces', action='store_true', help='Bitwise reproducible forces on the GPU (~20%% slower).')


def parse_input(argv=None):
    ''' Main openduck parser, subparsers define action modes
    '''
    parser = argparse.ArgumentParser(description='Open-source toolkit for dynamic undocking.')
    parser.set_defaults(command=None)
    modes = parser.add_subparsers(title='Open-source dynamic undocking toolkit. Choose one of the following actions', help='',
                                  metavar='', dest='command')

    # openmm-prepare
    openmm_prep = modes.add_parser('openmm-prepare', help='Preparation of systems for OpenMM simulations.',
                                   description='Preparation of systems for dynamic undocking simulations in OpenMM. The ligand, receptor and '
                                               'solvation box are parameterized with the specified forcefields. If specified, the receptor is '
                                               'reduced to a chunked pocket and the system is equilibrated.')
    group = openmm_prep.add_argument_group('Main arguments')
    _add_main_arguments(group, 'OpenMM preparation')
    group.add_argument('--keep-all-files', default=False, action='store_true', help='Disable cleaning up intermediate files.')
    _add_chunk_arguments(openmm_prep)
    group = openmm_prep.add_argument_group('Parameterization arguments')
    _add_parametrization_arguments(group)
    _add_water_steering_arguments(group)
    group = openmm_prep.add_argument_group('Equilibration arguments')
    group.add_argument('--do-equilibrate', action='store_true', help='Perform equilibration after preparing system.')
    group.add_argument('-F', '--force-constant-eq', type=float, default=1, help='Force constant for equilibration (kcal/mol/A^2).')
    group.add_argument('-H', '--HMR', action='store_true', help='Perform Hydrogen Mass Repartition on the topology and equilibrate with dt = 4 fs.')
    _add_engine_arguments(group)
    _add_batch_arguments(openmm_prep)
    openmm_prep.set_defaults(func=do_OpenMM_preparation)

    # openmm-full-protocol
    full = modes.add_parser('openmm-full-protocol', help='OpenDUck full OpenMM protocol.',
                            description='Full dynamic undocking protocol in OpenMM. The ligand, receptor and solvation box are parameterized '
                                        'with the specified parameters. If specified, the receptor is reduced to a chunked pocket. After '
                                        'equilibration, serial iterations of MD and SMD are performed until the WQB or max_cycles threshold is reached.')
    group = full.add_argument_group('Main arguments')
    _add_main_arguments(group, 'the full OpenDUck protocol')
    group.add_argument('-g', '--gpu-id', type=int, default=None, help='GPU ID; if not specified, runs on CPU only.')
    group.add_argument('--keep-all-files', default=False, action='store_true', help='Disable cleaning up intermediate files during simulations.')
    _add_chunk_arguments(full)
    group = full.add_argument_group('Parameterization arguments')
    _add_parametrization_arguments(group)
    _add_water_steering_arguments(group)
    group.add_argument('-H', '--HMR', action='store_true', help='Perform Hydrogen Mass Repartition on the topology and run at dt = 4 fs.')
    group = full.add_argument_group('MD/SMD production arguments')
    group.add_argument('-F', '--force-constant-eq', type=float, default=1, help='Force constant for equilibration (kcal/mol/A^2).')
    _add_equilibrations_argument(group)
    _add_production_arguments(group)
    _add_engine_arguments(group, gpu=False)
    _add_batch_arguments(full)
    full.set_defaults(func=do_full_openMM_protocol)

    # openmm-from-equilibrated
    equil = modes.add_parser('openmm-from-equilibrated', help='OpenDUck OpenMM protocol starting from a pre-equilibrated system.',
                             description='Dynamic undocking starting from a pre-equilibrated system (checkpoint and preparation pickle '
                                         'with the key interaction). Serial iterations of MD and SMD are performed until the WQB or '
                                         'max_cycles threshold is reached.')
    equil.add_argument('-y', '--yaml-input', type=str, default=None,
                       help='Input yaml file with all the arguments for the OpenMM simulations from the equilibrated system.')
    equil.add_argument('-s', '--equilibrated-system', default=None, help='Equilibrated system as input (*.chk).')
    equil.add_argument('-p', '--pickle', default=None, help='Pickle output from preparation.')
    equil.add_argument('-H', '--HMR', action='store_true', help='Run at dt = 4 fs (the system should have been prepared with HMR).')
    equil.add_argument('--keep-all-files', default=False, action='store_true',
                       help='Accepted for compatibility; the production files are always kept.')
    equil.add_argument('--waters-to-restrain', default=None, type=int,
                       help='Number of (retained) waters, counted from the first one, restrained with the chunk (water steering).')
    _add_production_arguments(equil)
    _add_engine_arguments(equil)
    equil.set_defaults(func=do_openMM_from_equil)

    # openmm-from-amber
    openmm_prmtop = modes.add_parser('openmm-from-amber', help='OpenDUck OpenMM protocol starting from an AMBER topology and coordinates.',
                                     description='OpenDUck OpenMM protocol starting from an AMBER topology and coordinates. Using the AMBER '
                                                 'topology (prmtop) and coordinates (inpcrd), identifies the main interaction and serial '
                                                 'iterations of MD and SMD are performed until the WQB or max_cycles threshold is reached.')
    openmm_prmtop.add_argument('-y', '--yaml-input', type=str, default=None,
                               help='Input yaml file with all the arguments for the OpenMM simulations from the AMBER system.')
    openmm_prmtop.add_argument('-c', '--coordinates', default=None, type=str, help='AMBER input coordinates.')
    openmm_prmtop.add_argument('-t', '--topology', default=None, type=str, help='AMBER input topology.')
    openmm_prmtop.add_argument('-i', '--interaction', default=None, type=str, help='Protein atom to use for ligand interaction.')
    openmm_prmtop.add_argument('-r', '--receptor', type=str, default=None, help='Receptor chunk (PDB or mol2) with the original residue numbering.')
    openmm_prmtop.add_argument('-H', '--HMR', action='store_true', help='Run at dt = 4 fs (hydrogen masses are repartitioned if needed).')
    openmm_prmtop.add_argument('--keep-all-files', default=False, action='store_true', help='Disable cleaning up intermediate files during simulations.')
    _add_equilibrations_argument(openmm_prmtop)
    _add_production_arguments(openmm_prmtop)
    _add_engine_arguments(openmm_prmtop)
    openmm_prmtop.set_defaults(func=do_openMM_from_amber)

    # amber-prepare
    amber = modes.add_parser('amber-prepare', help='Preparation of systems, inputs and queue files for AMBER simulations.',
                             description='Preparation of systems, inputs and queue files for AMBER simulations. The ligand, receptor and '
                                         'solvation box are parameterized with the specified parameters. If specified, the receptor is '
                                         'reduced to a chunked pocket for a faster production. The input and queue files are prepared '
                                         'from templates found in the duck/templates directory.')
    _add_main_arguments(amber.add_argument_group('Main arguments'), 'the system preparation and inputs/queueing for AMBER')
    _add_chunk_arguments(amber)
    group = amber.add_argument_group('Parameterization arguments')
    _add_parametrization_arguments(group, water_models=('tip3p', 'spce', 'tip4pew'))
    group.add_argument('-q', '--queue-template', type=str, default='local',
                       help='Queue template to write the job script (local, Slurm, SGE or a custom {name}.q template). Default = local.')
    group.add_argument('-H', '--HMR', action='store_true', help='Perform Hydrogen Mass Repartition on the topology and use it for the input files.')
    group.add_argument('-n', '--smd-cycles', type=int, default=20,
                       help='Number of SMD replicas to perform (per equilibration with --equilibrations).')
    _add_equilibrations_argument(group)
    group.add_argument('-W', '--wqb-threshold', type=float, default=0,
                       help='Minimum WQB threshold; if not reached after each SMD cycle, further simulations will be terminated. '
                            'Default = 0 (all SMD cycles are run).')
    group.add_argument('--seed', default='-1', type=str, help='Specify seed for AMBER inputs (-1: random; required with --equilibrations).')
    _add_water_steering_arguments(group)
    group.add_argument('-e', '--ligand-hb-elements', default=[7, 8], type=int, nargs='+',
                       help='Control which elements are accepted in the ligand to define the steering interaction. Specify using the '
                            'atomic number separated with a space. Default is 7 and 8 (nitrogen and oxygen)')
    _add_batch_arguments(amber, keep_files=True)
    amber.set_defaults(func=do_AMBER_preparation)

    # report
    report = modes.add_parser('report', help='Generate a report for OpenDUck results.',
                              description='Generate a table report for dynamic undocking output. For a multi-ligand report, use the pattern '
                                          'flag with wildcards to the directories.')
    report.add_argument('-p', '--pattern', type=str, default='.', help='Wildcard pattern to find folders with DUck data.')
    report.add_argument('-d', '--data', type=str, default='min', choices=('min', 'single', 'avg', 'jarzynski', 'all'),
                        help='Mode to compile the report [min | single | avg | jarzynski | all]')
    report.add_argument('-o', '--output', default='stdout', help='Output file; default is printing report to stdout.')
    report.add_argument('-of', '--output-format', default='tbl', choices=('csv', 'sdf', 'tbl'), type=str, help='Output format, [csv | sdf | tbl].')
    report.add_argument('--plot', default=False, action='store_true', help='Plot work or energy values to file.')
    report.add_argument('-s', '--subsample-size', default=20, type=int, help='Subsample size for Jarzynski bootstrapping.')
    report.add_argument('-i', '--iterations', default=20, type=int, help='Number of bootstrapping iterations for Jarzynski analysis.')
    report.add_argument('-t', '--step-threshold', default=None, type=int,
                        help='Number of initial work points where the minimum is searched (Jarzynski analysis). Default: first half of the profile.')
    report.add_argument('-f', '--format', type=str.lower, default='auto', choices=('auto', 'amber', 'openmm'),
                        help='Engine used to generate results (auto detects duck_runs/ for OpenMM). Default = auto')
    report.add_argument('--seed', type=int, default=None, help='Random seed for the bootstrapping (reproducible reports).')
    report.set_defaults(func=do_report)

    # chunk
    chunk = modes.add_parser('chunk', help='Chunk a protein for dynamic undocking.',
                             description='Reduce the receptor protein to a pocket by chunking it around the specified interaction. The amino '
                                         'acids within a certain cutoff are considered part of the chunk and the protein segments are capped.')
    _add_main_arguments(chunk, 'chunking')
    chunk.add_argument('-c', '--cutoff', type=float, default=9, help='Cutoff distance to define chunking (in Angstroms). Default = 9 A.')
    chunk.add_argument('-b', '--ignore-buffers', action='store_true', help='Do not remove buffers (solvent, ions etc.)')
    chunk.add_argument('-o', '--output', type=str, default='protein_out.pdb', help='Output name for the chunked protein receptor.')
    chunk.add_argument('--keep-all-files', default=False, action='store_true', help='Disable cleaning up intermediate files during preparation.')
    chunk.set_defaults(func=do_chunk)

    # prepare-cofactor
    cofactor = modes.add_parser('prepare-cofactor', help='Create the parameters file for a cofactor to input as custom forcefield.',
                                description='To perform simulations with cofactors new non-standard parameters need to be provided. OpenMM '
                                            'accepts parameters in xml format, where charges and forcefield parameters are stored, however Amber '
                                            'stores charges and parameters separatedly. This is a helper program to create an xml parameters file.')
    cofactor.add_argument('-y', '--yaml-input', type=str, default=None, help='Input yaml file with all the arguments for the cofactor parameter file preparation.')
    cofactor.add_argument('-m', '--mol2', type=str, default=None, help='Mol2 with the partial charges of the cofactor')
    cofactor.add_argument('-f', '--frcmod', type=str, default=None,
                          help='frcmod of the complete set of parameters for your molecule. Reminder: use the -a Y flag in parmchk2.')
    cofactor.add_argument('-o', '--output', type=str, default=None, help='Output parameters file in xml format.')
    cofactor.set_defaults(func=do_prepare_cofactor)

    args = parser.parse_args(argv)
    if args.command is None:
        return args, parser
    subparser = modes.choices[args.command]
    if getattr(args, 'yaml_input', None):
        apply_yaml_defaults(subparser, args.yaml_input)
        args = parser.parse_args(argv)
    args_sanitation(args, subparser)
    return args, parser


def args_sanitation(args, subparser):
    '''Check the required arguments (from the command line or the yaml file) and their consistency.'''
    missing = [name for name in REQUIRED.get(args.command, ()) if getattr(args, name, None) in (None, '')]
    if missing:
        subparser.error('The input needs to be either the input yaml or specified on the command line: missing '
                        + ', '.join('--' + name.replace('_', '-') for name in missing) + '.')
    if getattr(args, 'batch', False) and not str(args.ligand).lower().endswith(SDF_EXTENSIONS):
        subparser.error('Batch processing requires the ligand to be in SD or SDF format.')
    if getattr(args, 'water_steering', False) and not args.waters_to_restrain:
        args.waters_to_restrain = 1
    _check_equilibrations(args, subparser)
    if args.command == 'report':
        if args.output == 'stdout':
            args.output = sys.stdout
        if (args.iterations != 20 or args.subsample_size != 20) and args.data not in ('all', 'jarzynski'):
            subparser.error('Iterations and subsample size affect bootstrapping which is only performed when doing Jarzynski analysis.')
    return args


def _check_equilibrations(args, subparser):
    '''--equilibrations must be positive, and independent AMBER equilibrations need random seeds.'''
    if getattr(args, 'equilibrations', 1) < 1:
        subparser.error('--equilibrations must be at least 1.')
    if getattr(args, 'equilibrations', 1) > 1 and str(getattr(args, 'seed', '-1')).strip() != '-1':
        subparser.error('--equilibrations needs random seeds (--seed -1): with a fixed seed every equilibration would be the same.')


def _simulation_options(args):
    '''DuckSimulation keyword arguments from the parsed options.'''
    return dict(precision=args.precision, integrator=args.integrator, deterministic_forces=args.deterministic_forces,
                chunk_restraint_k=args.force_constant_chunk, smd_spring_k=args.smd_spring_constant,
                threads=getattr(args, 'threads', None))


def _run_production(args, checkpoint='equil.chk', pickle_file='complex_system.pickle'):
    '''Run the DUck MD/SMD cycles in the current directory.'''
    from duck.steps.production import run_duck_cycles
    # runs started with OpenDUck <= 0.2 renamed the equilibrated files before the production
    if not os.path.exists(checkpoint) and checkpoint == 'equil.chk' and os.path.exists('eql.chk'):
        checkpoint = 'eql.chk'
    if not os.path.exists(pickle_file) and pickle_file == 'complex_system.pickle' and os.path.exists('cs.pickle'):
        pickle_file = 'cs.pickle'
    return run_duck_cycles(checkpoint, pickle_file, n_cycles=args.smd_cycles, md_length=args.md_length, gpu_id=args.gpu_id,
                           start_dist=args.init_distance, init_velocity=args.init_velocities, save_dir='duck_runs',
                           wqb_threshold=args.wqb_threshold, hmr=args.HMR, traj_interval_ps=args.trajectory_interval,
                           waters_to_restrain=getattr(args, 'waters_to_restrain', None) or 0, **_simulation_options(args))


# ---------------------------------------------------------------------------------------------- preparation
def _append_interaction_to_pickle(results, pickle_file='complex_system.pickle'):
    '''Store [structure, protein_idx, ligand_idx, distance] in the preparation pickle; return the structure.'''
    with open(pickle_file, 'rb') as f:
        p = pickle.load(f)[:1] + list(results)
    with open(pickle_file, 'wb') as f:
        pickle.dump(p, f, protocol=pickle.HIGHEST_PROTOCOL)
    p[0].save('system_complex.inpcrd', overwrite=True)
    return p


def prepare_sys_for_amber(ligand_file, protein_file, chunk_file, interaction, HMR,
                          small_molecule_forcefield='SMIRNOFF', water_ff_str='tip3p', forcefield_str='amber99sb.xml',
                          ionic_strength=0.1, box_buffer_distance=10, waters_to_retain="waters_to_retain.pdb", custom_forcefield=None,
                          seed='-1', fix_ligand_file=False, clean_up=False, water_steering=False, waters_to_restrain=None,
                          lig_HB_elements=(7, 8), inputs_class=None, inputs_options=None):
    '''
    Prepares the system for AMBER simulation by parameterizing the ligand and finding the specified interaction
    between the ligand and protein. The resulting complex is saved to a pickle file, 'complex_system.pickle', and
    the AMBER inputs are written.

    Args:
        ligand_file (str): Path to the ligand file (MOL or SDF).
        protein_file (str): Receptor PDB with the numbering used in `interaction`.
        chunk_file (str): Receptor (chunk) to parametrize.
        interaction (str): Receptor atom that forms the main interaction with the ligand (chain_resname_resid_atomname).
        HMR (bool): Whether or not to use hydrogen mass repartitioning (HMR).
        small_molecule_forcefield (str, optional): Force field to use for ligand parameterization. Default is 'SMIRNOFF'.
        water_ff_str (str, optional): Water model (tip3p, spce or tip4pew). Default is 'tip3p'.
        forcefield_str (str, optional): Force field to use for protein parameterization. Default is 'amber99sb.xml'.
        ionic_strength (float, optional): Ionic strength of the system in M. Default is 0.1 M.
        box_buffer_distance (float, optional): Distance in Angstroms between the solute and the edges of the box. Default is 10.
        waters_to_retain (str, optional): Path to the file containing the water molecules to retain. Default is "waters_to_retain.pdb".
        custom_forcefield (str, optional): Extra OpenMM XML force field.
        seed (str, optional): Seed for pmemd ('-1' for random). Default is '-1'.
        fix_ligand_file (bool, optional): Whether or not to fix the ligand file during parameterization. Default is False.
        clean_up (bool, optional): Remove intermediate files.
        water_steering (bool, optional): Use the retained water as the receptor side of the interaction.
        waters_to_restrain (int, optional): Number of retained waters to restrain.
        lig_HB_elements (sequence of int, optional): Accepted elements of the ligand interaction atom.
        inputs_class (class, optional): writer of the pmemd inputs, ``Amber_templates`` (DUck) by default; e.g.
            ``Crooks_amber_templates`` for the bidirectional protocol.
        inputs_options (dict, optional): extra keyword arguments for ``inputs_class``.

    Returns:
        list: [structure, receptor atom index, ligand atom index, distance] as stored in the pickle.
    '''
    from duck.steps.parametrize import prepare_system
    from duck.utils.cal_ints import find_interaction
    from duck.utils.amber_inputs import Amber_templates

    prepare_system(ligand_file, chunk_file, forcefield_str=forcefield_str,
                   hmr=HMR, small_molecule_ff=small_molecule_forcefield, water_ff_str=water_ff_str,
                   box_buffer_distance=box_buffer_distance, ionicStrength=ionic_strength, waters_to_retain=waters_to_retain,
                   custom_forcefield=custom_forcefield, fix_ligand_file=fix_ligand_file, clean_up=clean_up)
    results = find_interaction(interaction, waters_to_retain if water_steering else protein_file, lig_HB_elements)
    p = _append_interaction_to_pickle(results)
    amber_inputs = (inputs_class or Amber_templates)(structure=p[0], interaction=p[1:], hmr=HMR, seed=seed,
                                                     waters_masked=waters_to_restrain, **(inputs_options or {}))
    amber_inputs.write_all_inputs()
    return p


def _basename(path):
    return os.path.basename(path) if path else path


def _has_results(folder):
    '''True if ``folder`` holds a prepared system or simulation results (AMBER or OpenMM, any OpenDUck version).'''
    markers = ('complex_system.pickle', 'cs.pickle', 'duck_runs', 'equil.chk', 'eql.chk', '3_eq.rst')
    return any(os.path.exists(os.path.join(folder, m)) for m in markers) or bool(
        glob.glob(os.path.join(folder, 'DUCK_*')) or glob.glob(os.path.join(folder, 'EQ_*')))


def _prepare_ligand_folder(ligand_string, lig_indx, base_dir, files, prefix, resume):
    '''Create ``{prefix}_{lig_indx}`` with the ligand and copies of the shared input files.

    In resume mode, folders with a prepared system or results are kept untouched; only folders where the
    preparation did not produce anything are prepared again.

    Returns:
        str or None: the folder name, or None if it is kept because of the resume mode.
    '''
    from duck.utils.amber_inputs import write_string_to_file
    lig_dir = os.path.join(base_dir, f'{prefix}_{lig_indx}')
    if os.path.isdir(lig_dir):
        if resume and _has_results(lig_dir):
            print(f'{prefix}_{lig_indx} already exists and will be kept (resume mode).')
            return None
        print(f'WARNING: {prefix}_{lig_indx} already exists and it will be overwritten.')
        shutil.rmtree(lig_dir, ignore_errors=True)
    os.makedirs(lig_dir, exist_ok=True)
    write_string_to_file(string=ligand_string, file=os.path.join(lig_dir, f'lig_{lig_indx}.mol'))
    for file in files:
        if file and os.path.isfile(os.path.join(base_dir, file)):
            shutil.copyfile(os.path.join(base_dir, file), os.path.join(lig_dir, os.path.basename(file)), follow_symlinks=True)
    return lig_dir


def AMBER_prepare_ligand_in_folder(ligand_string, lig_indx, protein, chunk, interaction, HMR, base_dir, small_molecule_forcefield='SMIRNOFF',
                                   water_model='tip3p', forcefield='amber99sb', ion_strength=0.1, box_buffer_distance=10,
                                   waters_to_retain='waters_to_retain.pdb', custom_forcefield=None, seed='-1', fix_ligand=False,
                                   clean_up=False, resume=False, prefix='LIG_target', water_steering=False, waters_to_restrain=None,
                                   ligands_HB_elements=(7, 8)):
    '''
    Generate the folder for one ligand of a batch and prepare it for AMBER (log in preparation.out).
    '''
    from duck.utils.amber_inputs import Queue_templates
    lig_dir = _prepare_ligand_folder(ligand_string, lig_indx, base_dir, (protein, chunk, waters_to_retain), prefix, resume)
    if lig_dir is None:
        return f'{prefix}_{lig_indx} skipped.'
    os.chdir(lig_dir)
    print(f'Working on {prefix}_{lig_indx}')
    os.environ['OMP_NUM_THREADS'] = '1'  # one thread per sqm (AM1-BCC) process
    if custom_forcefield:
        custom_forcefield = os.path.join(base_dir, custom_forcefield)
    try:
        with open('preparation.out', 'w') as o, redirect_stdout(o):
            prepare_sys_for_amber(f'lig_{lig_indx}.mol', os.path.basename(protein), os.path.basename(chunk), interaction, HMR,
                                  small_molecule_forcefield=small_molecule_forcefield, water_ff_str=water_model,
                                  forcefield_str=f'{forcefield}.xml', ionic_strength=ion_strength, box_buffer_distance=box_buffer_distance,
                                  waters_to_retain=_basename(waters_to_retain), custom_forcefield=custom_forcefield, seed=seed,
                                  fix_ligand_file=fix_ligand, clean_up=clean_up, water_steering=water_steering,
                                  waters_to_restrain=waters_to_restrain, lig_HB_elements=ligands_HB_elements)
        Queue_templates().copy_getWqbValues_script()
    except Exception:
        with open('preparation.out', 'a') as o:
            traceback.print_exc(file=o)
        return f'{prefix}_{lig_indx} failed (see {prefix}_{lig_indx}/preparation.out).'
    finally:
        os.chdir(base_dir)
    return f'{prefix}_{lig_indx} prepared correctly'


def openmm_prepare_ligand_in_folder(ligand_string, lig_indx, protein, chunk, interaction, HMR, base_dir, small_molecule_forcefield='SMIRNOFF',
                                    water_model='tip3p', forcefield='amber99sb', ion_strength=0.1, box_buffer_distance=10,
                                    waters_to_retain='waters_to_retain.pdb', custom_forcefield=None, fix_ligand=False, clean_up=False,
                                    resume=False, prefix='LIG_target', water_steering=False):
    '''
    Generate the folder for one ligand of a batch and prepare it for OpenMM (log in preparation.out).

    Returns:
        tuple: (folder, status) with status 'prepared', 'kept' (resume mode) or 'failed'.
    '''
    from duck.steps.parametrize import prepare_system
    from duck.utils.cal_ints import find_interaction
    folder = f'{prefix}_{lig_indx}'
    lig_dir = _prepare_ligand_folder(ligand_string, lig_indx, base_dir, (protein, chunk, waters_to_retain), prefix, resume)
    if lig_dir is None:
        return folder, 'kept'
    os.chdir(lig_dir)
    print(f'Working on {folder}')
    os.environ['OMP_NUM_THREADS'] = '1'  # one thread per sqm (AM1-BCC) process
    if custom_forcefield:
        custom_forcefield = os.path.join(base_dir, custom_forcefield)
    try:
        with open('preparation.out', 'w') as o, redirect_stdout(o):
            prepare_system(f'lig_{lig_indx}.mol', os.path.basename(chunk), forcefield_str=f'{forcefield}.xml', water_ff_str=water_model,
                           small_molecule_ff=small_molecule_forcefield, waters_to_retain=_basename(waters_to_retain),
                           custom_forcefield=custom_forcefield, box_buffer_distance=box_buffer_distance, ionicStrength=ion_strength,
                           fix_ligand_file=fix_ligand, clean_up=clean_up, hmr=HMR)
            # the interaction is defined with the numbering of the original receptor (tleap renumbers chunks)
            reference = _basename(waters_to_retain) if water_steering else os.path.basename(protein)
            _append_interaction_to_pickle(find_interaction(interaction, reference))
    except Exception:
        with open('preparation.out', 'a') as o:
            traceback.print_exc(file=o)
        return folder, 'failed'
    finally:
        os.chdir(base_dir)
    return folder, 'prepared'


def _chunk_receptor(args):
    if not args.do_chunk:
        return args.receptor
    from duck.steps.chunk import duck_chunk
    print('Chunking protein')
    return duck_chunk(args.receptor, args.ligand, args.interaction, args.cutoff, ignore_buffers=args.ignore_buffers,
                      keep_all_files=args.keep_all_files)


def _equilibrate(args, pickle_file='complex_system.pickle', resume=None):
    '''Equilibrate the system of ``pickle_file`` in the current directory (skipped in resume mode if equil.chk exists).'''
    from duck.steps.equlibrate import do_equlibrate
    from duck.utils.check_system import check_if_equlibrated
    from duck.utils.duck_stuff import getRestrainedAtoms, load_complex_pickle
    resume = getattr(args, 'resume', False) if resume is None else resume
    if resume and (os.path.isfile('equil.chk') or os.path.isfile('eql.chk')):
        print('The system is already equilibrated, skipping the equilibration (resume mode).')
        return
    structure, key_interaction = load_complex_pickle(pickle_file)
    do_equlibrate(force_constant_equilibrate=getattr(args, 'force_constant_eq', 1.0), gpu_id=args.gpu_id,
                  keyInteraction=key_interaction, clean=not getattr(args, 'keep_all_files', False), hmr=args.HMR,
                  precision=args.precision, pickle_file=pickle_file, deterministic_forces=args.deterministic_forces,
                  threads=getattr(args, 'threads', None),
                  restrained_atoms=getRestrainedAtoms(structure, getattr(args, 'waters_to_restrain', None) or 0),
                  hb_restraint=getattr(args, 'md_restraint', None))
    if not check_if_equlibrated('density.csv', 1):
        raise EquilibrationError(f'The system in {os.getcwd()} is not equilibrated.')


def _run_with_equilibrations(args, production, pickle_file='complex_system.pickle'):
    '''Run ``production(checkpoint, pickle_file)`` after each of the ``args.equilibrations`` independent equilibrations
    of the prepared system, one ``EQ_<k>/`` folder each; a true return value stops the remaining ones.'''
    from duck.utils.equilibrations import run_independent_equilibrations
    pickle_file = os.path.abspath(pickle_file)
    return run_independent_equilibrations(args.equilibrations, lambda: _equilibrate(args, pickle_file, resume=False),
                                          lambda checkpoint: production(checkpoint, pickle_file),
                                          resume=getattr(args, 'resume', False))


def do_OpenMM_preparation(args):
    '''
    Prepare the protein-ligand complex(es) and, if requested, equilibrate them with OpenMM.

    Returns:
        list of str: absolute paths of the successfully prepared (and equilibrated) systems.
    '''
    os.environ["OPENMM_CPU_THREADS"] = str(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    from duck.steps.parametrize import prepare_system
    from duck.utils.cal_ints import find_interaction

    base_dir = os.getcwd()
    chunked_file = _chunk_receptor(args)
    if args.batch:
        from duck.utils.amber_inputs import ligand_string_generator
        ligand_jobs = [(ligand_string, j + args.index0, args.receptor, chunked_file, args.interaction, args.HMR, base_dir,
                        args.small_molecule_forcefield, args.water_model, args.protein_forcefield, args.ionic_strength,
                        args.solvent_buffer_distance, args.waters_to_retain, args.custom_forcefield, args.fix_ligand,
                        not args.keep_all_files, args.resume, args.prefix, args.water_steering)
                       for j, ligand_string in enumerate(ligand_string_generator(args.ligand))]
        with mp.Pool(args.threads) as pool:
            results = pool.starmap(openmm_prepare_ligand_in_folder, ligand_jobs)
        failed = [folder for folder, status in results if status == 'failed']
        for folder in failed:
            sys.stderr.write(f'Preparation of {folder} failed, see {folder}/preparation.out\n')
        folders = [os.path.join(base_dir, folder) for folder, status in results if status != 'failed']
    elif args.resume and _prepared_with_interaction('complex_system.pickle'):
        print('complex_system.pickle already exists, skipping the preparation (resume mode).')
        folders = [base_dir]
    else:
        os.environ['OMP_NUM_THREADS'] = '1'  # otherwise sqm uses all threads and is slower
        prepare_system(args.ligand, chunked_file, forcefield_str=f'{args.protein_forcefield}.xml', water_ff_str=args.water_model,
                       small_molecule_ff=args.small_molecule_forcefield, waters_to_retain=args.waters_to_retain,
                       custom_forcefield=args.custom_forcefield, box_buffer_distance=args.solvent_buffer_distance,
                       ionicStrength=args.ionic_strength, fix_ligand_file=args.fix_ligand, clean_up=not args.keep_all_files,
                       hmr=args.HMR)
        os.environ['OMP_NUM_THREADS'] = str(args.threads)
        results = find_interaction(args.interaction, args.waters_to_retain if args.water_steering else args.receptor)
        print('Key interaction (protein atom, ligand atom, distance):', results)
        _append_interaction_to_pickle(results)
        folders = [base_dir]

    if args.do_equilibrate:
        equilibrated = []
        for folder in folders:
            os.chdir(folder)
            print(f'Equilibrating {folder}')
            try:
                _equilibrate(args)
                equilibrated.append(folder)
            except Exception as error:
                if not args.batch:
                    raise
                sys.stderr.write(f'Equilibration of {folder} failed: {error}\n')
            finally:
                os.chdir(base_dir)
        folders = equilibrated
    return folders


# ---------------------------------------------------------------------------------------------- workflows
def _prepared_with_interaction(pickle_file):
    if not os.path.isfile(pickle_file):
        return False
    from duck.utils.duck_stuff import load_complex_pickle
    return len(load_complex_pickle(pickle_file)[1]) >= 2


def do_full_openMM_protocol(args):
    '''
    Full OpenDUck protocol in OpenMM: preparation, equilibration and MD/SMD production for each ligand.
    '''
    from duck.utils.equilibrations import find_equilibrations
    if not args.batch and not args.resume and (os.path.isdir('duck_runs') or find_equilibrations('.')):
        sys.exit('duck_runs/ or EQ_*/ already exist: use --resume to continue that run, or remove them to start a new one '
                 '(a new preparation and equilibration would be mixed with the existing MD/SMD results).')
    if not args.batch and (os.path.isdir('duck_runs') if args.equilibrations > 1 else find_equilibrations('.')):
        sys.exit(f'This folder holds a run with {"one equilibration (duck_runs/)" if args.equilibrations > 1 else "independent equilibrations (EQ_*/)"}: '
                 'resume it with its original --equilibrations, or use another folder.')
    args.do_equilibrate = args.equilibrations == 1  # independent equilibrations are run next to their production
    base_dir = os.getcwd()
    folders = do_OpenMM_preparation(args)
    print('Completed the preparation of all ligands')
    failed = []
    for folder in folders:
        print(f'Starting calculations for {folder}')
        os.chdir(folder)
        try:
            if args.equilibrations > 1:
                _run_with_equilibrations(args, lambda checkpoint, pickle_file: _run_production(args, checkpoint, pickle_file) is not None)
            else:
                _run_production(args)
        except Exception as error:
            if not args.batch:
                raise
            failed.append(folder)
            sys.stderr.write(f'DUck production of {folder} failed: {error}\n')
        finally:
            os.chdir(base_dir)
    if failed:
        raise RuntimeError(f'{len(failed)} system(s) failed: {", ".join(failed)}')


def do_openMM_from_equil(args):
    '''
    DUck production from an equilibrated checkpoint and its preparation pickle.
    '''
    _run_production(args, checkpoint=args.equilibrated_system, pickle_file=args.pickle)


def do_openMM_from_amber(args):
    '''
    Equilibrate an AMBER-prepared system (prmtop/inpcrd) and run the DUck production in OpenMM.
    '''
    from duck.steps.equlibrate import equilibrate_from_amber_prep, pickle_from_amber_prep
    if args.equilibrations > 1:
        pickle_from_amber_prep(args.interaction, args.topology, args.coordinates, args.receptor)
        _run_with_equilibrations(args, lambda checkpoint, pickle_file: _run_production(args, checkpoint, pickle_file) is not None)
        return
    equilibrate_from_amber_prep(interaction=args.interaction, prmtop_file=args.topology, inpcrd_file=args.coordinates,
                                chunk=args.receptor, gpu_id=args.gpu_id, force_constant_eq=1.0, hmr=args.HMR,
                                precision=args.precision, deterministic_forces=args.deterministic_forces,
                                clean=not args.keep_all_files)
    _run_production(args)


def do_AMBER_preparation(args):
    '''
    Prepare the protein-ligand complex(es) for the AMBER DUck protocol and write the AMBER inputs and queue files.
    '''
    from duck.utils.amber_inputs import Queue_templates, handle_error, ligand_string_generator
    chunk_file = _chunk_receptor(args)
    queue_options = dict(wqb_threshold=args.wqb_threshold, replicas=args.smd_cycles, hmr=args.HMR,
                         keep_intermediate_files=args.keep_all_files, equilibrations=args.equilibrations)
    if args.batch:
        base_dir = os.getcwd()
        with mp.Pool(args.threads) as pool:
            jobs = [pool.apply_async(AMBER_prepare_ligand_in_folder,
                                     args=(ligand_string, j + args.index0, args.receptor, chunk_file, args.interaction, args.HMR, base_dir,
                                           args.small_molecule_forcefield, args.water_model, args.protein_forcefield, args.ionic_strength,
                                           args.solvent_buffer_distance, args.waters_to_retain, args.custom_forcefield, args.seed,
                                           args.fix_ligand, not args.keep_all_files, args.resume, args.prefix, args.water_steering,
                                           args.waters_to_restrain, args.ligand_hb_elements),
                                     error_callback=handle_error)
                    for j, ligand_string in enumerate(ligand_string_generator(args.ligand))]
            messages = [job.get() for job in jobs if job.wait() is None and job.successful()]
        for message in messages:
            print(message)
        if args.queue_template == 'local':
            # there is no array version of the local script: write one script per ligand folder
            for j in range(len(jobs)):
                folder = os.path.join(base_dir, f'{args.prefix}_{j + args.index0}')
                if os.path.isfile(os.path.join(folder, 'complex_system.pickle')):
                    os.chdir(folder)
                    Queue_templates(**queue_options).write_queue_file(kind='local')
                    os.chdir(base_dir)
        else:
            Queue_templates(array_limit=len(jobs), **queue_options).write_queue_file(kind=args.queue_template)
    else:
        if args.threads != 1:
            print('WARNING: The number of threads does not have an impact if the batch mode is not enabled.')
        prepare_sys_for_amber(args.ligand, args.receptor, chunk_file, args.interaction, args.HMR,
                              small_molecule_forcefield=args.small_molecule_forcefield, water_ff_str=args.water_model,
                              forcefield_str=f'{args.protein_forcefield}.xml', ionic_strength=args.ionic_strength,
                              box_buffer_distance=args.solvent_buffer_distance, waters_to_retain=args.waters_to_retain,
                              custom_forcefield=args.custom_forcefield, seed=args.seed, fix_ligand_file=args.fix_ligand,
                              clean_up=not args.keep_all_files, water_steering=args.water_steering,
                              waters_to_restrain=args.waters_to_restrain, lig_HB_elements=args.ligand_hb_elements)
        Queue_templates(**queue_options).write_queue_file(kind=args.queue_template)


def do_report(args):
    '''
    Generate a report from the dynamic undocking output. The DUck folders are looked upon the args.pattern.
    The input layout depends on args.format ('amber', 'openmm' or 'auto'), and the output on args.data
    (min, single, avg, jarzynski or all). The replicas of independent equilibrations (EQ_<k>/) are pooled.
    '''
    from duck.utils.analysis_and_report import build_report_df, do_jarzynski_analysis, get_mols_and_format, get_Wqb_value_all
    from duck.utils.equilibrations import find_equilibrations
    folders = sorted(glob.glob(args.pattern))
    if not folders:
        sys.exit(f'No folders match the pattern {args.pattern}')
    wqb_info = {}
    currdir = os.getcwd()
    for folder in folders:
        wqb_info[folder] = []
        os.chdir(folder)
        layout = args.format
        if layout == 'auto':  # runs with independent equilibrations have their outputs in EQ_<k>/
            layout = 'openmm' if os.path.isdir(os.path.join((find_equilibrations('.') or ['.'])[0], 'duck_runs')) else 'amber'
        try:
            if args.data in ('min', 'single', 'avg', 'all'):
                wqb_info[folder].extend(get_Wqb_value_all(mode=layout, plot=args.plot))
            if args.data in ('jarzynski', 'all'):
                try:
                    expavg, sd, sem_v = do_jarzynski_analysis(index_threshold=args.step_threshold, sample_size=args.subsample_size,
                                                              samples=args.iterations, plot=args.plot, mode=layout, seed=args.seed)
                except ValueError as error:
                    sys.stderr.write(f'Warning: Jarzynski analysis of {folder} failed: {error}\n')
                    expavg, sd, sem_v = 'Nan', 'Nan', 'Nan'
                wqb_info[folder].extend([expavg, sd, sem_v])
        finally:
            os.chdir(currdir)

    df = build_report_df(wqb_info, mode=args.data)
    if args.output_format == 'csv':
        df.to_csv(args.output, index=False)
    elif args.output_format == 'tbl':
        df.to_csv(args.output, index=False, sep='\t')
    elif args.output_format == 'sdf':
        from rdkit import Chem
        with Chem.SDWriter(args.output) as w:
            for mol in get_mols_and_format(df, mode=args.data):
                w.write(mol)


def do_chunk(args):
    from duck.steps.chunk import duck_chunk
    duck_chunk(args.receptor, args.ligand, args.interaction, args.cutoff, output_name=args.output,
               ignore_buffers=args.ignore_buffers, keep_all_files=args.keep_all_files)


def do_prepare_cofactor(args):
    from duck.utils.frcmod_mol2_to_xml import frcmod_mol2_to_xml
    frcmod_mol2_to_xml(frcmod=args.frcmod, mol2=args.mol2, out_xml=args.output)


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
