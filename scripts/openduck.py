import argparse
import pickle
import shutil
import sys
from pathlib import Path
import os
import yaml

def args_sanitation(parser, modes):
    '''Sanitize the parser to allow yaml or command line inputs with a proper formating for the rest of the script to work
    '''
    args = parser.parse_args()
    # check if everything is ok
    if args.mode == 'full-protocol':
        if (args.yaml_input is None) and (args.ligand is None or args.interaction is None or args.receptor is None):
            modes.choices['openmm-full-protocol'].error('The input needs to be either the input yaml or specified on the command line (ligand, receptor interaction).')
        elif args.yaml_input:
            input_arguments = yaml.load(open(args.yaml_input), Loader=yaml.FullLoader)
            if all(item in list(input_arguments.keys()) for item in ['receptor_pdb', 'interaction', 'ligand_mol']):
                #transfer all required items
                args.ligand = str(input_arguments['ligand_mol'])
                args.receptor = str(input_arguments['receptor_pdb'])
                args.interaction = str(input_arguments['interaction'])

                # overwrite the defaults from command line args
                if 'gpu_id' in input_arguments: args.gpu_id =  input_arguments['gpu_id']
                if 'do_chunk' in input_arguments: args.do_chunk =  bool(input_arguments['do_chunk'])
                if 'cutoff' in input_arguments: args.cutoff =  float(input_arguments['cutoff'])
                if 'ignore_buffers' in input_arguments: args.ignore_buffers =  bool(input_arguments['ignore_buffers'])
                if 'small_molecule_forcefield' in input_arguments: args.small_molecule_forcefield =  str(input_arguments['small_molecule_forcefield']).upper()
                if 'water_model' in input_arguments: args.water_model =  str(input_arguments['water_model']).lower()
                if 'protein_forcefield' in input_arguments: args.protein_forcefield =  str(input_arguments['protein_forcefield']).lower()
                if 'ionic_strength' in input_arguments: args.ionic_strength =  float(input_arguments['ionic_strength'])
                if 'solvent_buffer_distance' in input_arguments: args.solvent_buffer_distance =  float(input_arguments['solvent_buffer_distance'])
                if 'waters_to_retain' in input_arguments: args.waters_to_retain =  str(input_arguments['waters_to_retain'])
                if 'smd_cycles' in input_arguments: args.smd_cycles =  int(input_arguments['smd_cycles'])
                if 'md_length' in input_arguments: args.md_length =  float(input_arguments['md_length'])
                if 'init_velocities' in input_arguments: args.init_velocities =  float(input_arguments['init_velocities'])
                if 'init_distance' in input_arguments: args.init_distance =  float(input_arguments['init_distance'])
                if 'force_constant_eq' in input_arguments: args.force_constant_eq =  float(input_arguments['force_constant_eq'])
                if 'wqb_threshold' in input_arguments: args.wqb_threshold = float(input_arguments['wqb_threshold'])
            else:
                modes.choices['openmm-full-protocol'].error('You need to specify at least "ligand_mol", "receptor_pdb" and "interaction" in the yaml file.')
        elif (args.ligand is None or args.interaction is None or args.receptor is None):
            modes.choices['openmm-full-protocol'].error('The parameters --ligand, --interaction and --receptor are required.')
        else:
            # all good
            pass
    elif args.mode == 'from-equilibration':
        if (args.yaml_input is None) and (args.equilibrated_system is None or args.pickle is None):
            modes.choices['openmm-from-equilibrated'].error('The input needs to be either the input yaml or specified on the command line system "pickle" and "equilibrated_system" from parameterization.')
        elif args.yaml_input:
            input_arguments = yaml.load(open(args.yaml_input), Loader=yaml.FullLoader)
            if all(item in list(input_arguments.keys()) for item in ['pickle', 'equilibrated_system']):
                #transfer required items
                args.pickle = input_arguments['pickle']
                args.equilibrated_system = input_arguments['equilibrated_system']

                # overwrite the defaults from command line args
                if 'gpu_id' in input_arguments: args.gpu_id =  input_arguments['gpu_id']
                if 'smd_cycles' in input_arguments: args.smd_cycles =  int(input_arguments['smd_cycles'])
                if 'md_length' in input_arguments: args.md_length =  float(input_arguments['md_length'])
                if 'init_velocities' in input_arguments: args.init_velocities =  float(input_arguments['init_velocities'])
                if 'init_distance' in input_arguments: args.init_distance =  float(input_arguments['init_distance'])
                if 'wqb_threshold' in input_arguments: args.wqb_threshold = float(input_arguments['wqb_threshold'])

            else:
                modes.choices['openmm-from-equilibrated'].error('You need to specify at least "pickle" and "equilibrated_system" in the yaml file.')
        elif (args.pickle is None or args.equilibrated_system is None):
            modes.choices['openmm-from-equilibrated'].error('The parameters --pickle and --equilibrated_system are required.')
        else:
            #all good
            pass
    elif args.mode == 'openmm-preparation':
        if (args.yaml_input is None) and (args.ligand is None or args.interaction is None or args.receptor is None):
            modes.choices['openmm-prepare'].error('The input needs to be either the input yaml or specified on the command line (ligand, receptor interaction).')
        elif args.yaml_input:
            input_arguments = yaml.load(open(args.yaml_input), Loader=yaml.FullLoader)
            if all(item in list(input_arguments.keys()) for item in ['receptor_pdb', 'interaction', 'ligand_mol']):
                #transfer all required items
                args.ligand = str(input_arguments['ligand_mol'])
                args.receptor = str(input_arguments['receptor_pdb'])
                args.interaction = str(input_arguments['interaction'])

                # overwrite the defaults from command line args
                if 'do_chunk' in input_arguments: args.do_chunk =  bool(input_arguments['do_chunk'])
                if 'cutoff' in input_arguments: args.cutoff =  float(input_arguments['cutoff'])
                if 'ignore_buffers' in input_arguments: args.ignore_buffers =  bool(input_arguments['ignore_buffers'])
                if 'small_molecule_forcefield' in input_arguments: args.small_molecule_forcefield =  str(input_arguments['small_molecule_forcefield']).upper()
                if 'water_model' in input_arguments: args.water_model =  str(input_arguments['water_model']).lower()
                if 'protein_forcefield' in input_arguments: args.protein_forcefield =  str(input_arguments['protein_forcefield']).lower()
                if 'ionic_strength' in input_arguments: args.ionic_strength =  float(input_arguments['ionic_strength'])
                if 'solvent_buffer_distance' in input_arguments: args.solvent_buffer_distance =  float(input_arguments['solvent_buffer_distance'])
                if 'waters_to_retain' in input_arguments: args.waters_to_retain =  str(input_arguments['waters_to_retain'])
                if 'do_equilibrate' in input_arguments: args.do_equilibrate =  str(input_arguments['do_equilibrate'])
                if 'gpu_id' in input_arguments: args.gpu_id =  str(input_arguments['gpu_id'])
                if 'force_constant_eq' in input_arguments: args.force_constant_eq =  float(input_arguments['force_constant_eq'])
            else:
                modes.choices['openmm-prepare'].error('You need to specify at least "ligand_mol", "receptor_pdb" and "interaction" in the yaml file.')
        elif (args.ligand is None or args.interaction is None or args.receptor is None):
            modes.choices['openmm-prepare'].error('The parameters --ligand, --interaction and --receptor are required.')
        else:
            # all good
            pass
    elif args.mode == 'from-amber':
        if (args.yaml_input is None) and (args.interaction is None or args.coordinates is None and args.topology is None):
            modes.choices['openmm-from-amber'].error('The input needs to be either the input yaml or specified on the command line (topology, coordinates and interaction).')
        elif args.yaml_input:
            input_arguments = yaml.load(open(args.yaml_input), Loader=yaml.FullLoader)
            if all(item in list(input_arguments.keys()) for item in ['topology', 'interaction', 'coordinates', 'receptor']):
                #transfer all required items
                args.topology = str(input_arguments['topology'])
                args.coordinates = str(input_arguments['coordinates'])
                args.interaction = str(input_arguments['interaction'])
                args.receptor = str(input_arguments['receptor'])

                # overwrite the defaults from command line args
                if 'smd_cycles' in input_arguments: args.smd_cycles =  int(input_arguments['smd_cycles'])
                if 'md_length' in input_arguments: args.md_length =  float(input_arguments['md_length'])
                if 'init_velocities' in input_arguments: args.init_velocities =  float(input_arguments['init_velocities'])
                if 'init_distance' in input_arguments: args.init_distance =  float(input_arguments['init_distance'])
                if 'gpu_id' in input_arguments: args.gpu_id =  int(input_arguments['gpu_id'])
                if 'wqb_threshold' in input_arguments: args.wqb_threshold = float(input_arguments['wqb_threshold'])

            else:
                modes.choices['openmm-from-amber'].error('You need to specify at least the AMBER topology and coordinates, a receptor and the interaction and a receptor file.')
        elif (args.interaction is None or args.topology is None or args.coordinates is None):
            modes.choices['openmm-from-amber'].error('The parameters --topology, --coordinates, --interaction are required.')
        else:
            #all good
            pass
    elif args.mode == 'Amber-preparation':
        if (args.yaml_input is None) and (args.ligand is None or args.interaction is None or args.receptor is None):
            modes.choices['amber-prepare'].error('The input needs to be either the input yaml or specified on the command line (ligand, receptor interaction).')
        elif args.yaml_input:
            input_arguments = yaml.load(open(args.yaml_input), Loader=yaml.FullLoader)
            if all(item in list(input_arguments.keys()) for item in ['receptor_pdb', 'interaction', 'ligand_mol']):
                #transfer all required items
                args.ligand = str(input_arguments['ligand_mol'])
                args.receptor = str(input_arguments['receptor_pdb'])
                args.interaction = str(input_arguments['interaction'])

                # overwrite the defaults from command line args
                if 'do_chunk' in input_arguments: args.do_chunk =  bool(input_arguments['do_chunk'])
                if 'cutoff' in input_arguments: args.cutoff =  float(input_arguments['cutoff'])
                if 'ignore_buffers' in input_arguments: args.ignore_buffers =  bool(input_arguments['ignore_buffers'])
                if 'small_molecule_forcefield' in input_arguments: args.small_molecule_forcefield =  str(input_arguments['small_molecule_forcefield']).upper()
                if 'water_model' in input_arguments: args.water_model =  str(input_arguments['water_model']).lower()
                if 'protein_forcefield' in input_arguments: args.protein_forcefield =  str(input_arguments['protein_forcefield']).lower()
                if 'ionic_strength' in input_arguments: args.ionic_strength =  float(input_arguments['ionic_strength'])
                if 'solvent_buffer_distance' in input_arguments: args.solvent_buffer_distance =  float(input_arguments['solvent_buffer_distance'])
                if 'waters_to_retain' in input_arguments: args.waters_to_retain =  str(input_arguments['waters_to_retain'])
                if 'seed' in input_arguments: args.seed =  str(input_arguments['seed'])
                if 'queue_template' in input_arguments: args.queue_template = str(input_arguments['queue_template'])
                if 'HMR' in input_arguments: args.HMR = bool(input_arguments['HMR'])
                if 'wqb_threshold' in input_arguments: args.wqb_threshold = int(input_arguments['wqb_threshold'])
                if 'smd_cycles' in input_arguments: args.smd_cycles = int(input_arguments['smd_cycles'])
                if 'batch' in input_arguments: args.batch = int(input_arguments['batch'])
                if 'threads' in input_arguments: args.threads = int(input_arguments['threads'])
                if args.queue_template == 'local' and args.batch: args.queue_template = None # no local array script
                if (not args.ligand.endswith('.sdf') and not args.ligand.endswith('.sd')) and args.batch:
                    modes.choices['amber-prepare'].error('Batch processing requires the ligand to be in SD or SDF format.')
            else:
                modes.choices['amber-prepare'].error('You need to specify at least "ligand_mol", "receptor_pdb" and "interaction" in the yaml file.')
        elif (args.ligand is None or args.interaction is None or args.receptor is None):
            modes.choices['amber-prepare'].error('The parameters --ligand, --interaction and --receptor are required.')
        else:
            # all good
            pass
    elif args.mode == 'Report':
        if args.output == 'stdout': args.output = sys.stdout # little trick to print
        if (args.iterations != 20 or args.subsample_size != 20) and args.data not in ('all', 'jarzynski'):
            modes.choices['report'].error('Iterations and subsample size affect bootstrapping which is only performed when doing Jarzynski analysis.')
        if  args.format == 'openmm' and args.data in ('all', 'jarzynski') and args.step_threshold > 1250:
            print('OpenMM duck output has 1250 report steps. The index_threshold to find the minima needs to be >1250. It will be changed to 600.')
            args.index_threshold == 600
    elif args.mode == 'Chunk':
        if (args.yaml_input is None) and (args.ligand is None or args.interaction is None or args.receptor is None):
            modes.choices['chunk'].error('The input needs to be either the input yaml or specified on the command line (ligand, receptor interaction).')
        elif args.yaml_input:
            input_arguments = yaml.load(open(args.yaml_input), Loader=yaml.FullLoader)
            if all(item in list(input_arguments.keys()) for item in ['receptor_pdb', 'interaction', 'ligand_mol']):
                #transfer all required items
                args.ligand = str(input_arguments['ligand_mol'])
                args.receptor = str(input_arguments['receptor_pdb'])
                args.interaction = str(input_arguments['interaction'])

                # overwrite the defaults from command line args
                if 'cutoff' in input_arguments: args.cutoff =  float(input_arguments['cutoff'])
                if 'ignore_buffers' in input_arguments: args.ignore_buffers =  bool(input_arguments['ignore_buffers'])
                if 'output' in input_arguments: args.output =  str(input_arguments['output'])


            else:
                modes.choices['chunk'].error('You need to specify at least "ligand_mol", "receptor_pdb" and "interaction" in the yaml file.')
        elif (args.ligand is None or args.interaction is None or args.receptor is None):
            modes.choices['chunk'].error('The parameters --ligand, --interaction and --receptor are required.')
        else:
            # all good
            pass
        pass
    return args

def parse_input():
    ''' Main openduck parser, subparsers define action modes
    '''
    parser = argparse.ArgumentParser(description='Open-source toolkit for dynamic undocking.')
    parser.set_defaults(mode=None)
    modes = parser.add_subparsers(title='Open-source dynamic undocking toolkit. Choose one of the following actions:', help='', metavar='')

    #Arguments for OPENMM_PREPARATION
    openmm_prep = modes.add_parser('openmm-prepare', help='Preparation of systems for OpenMM simulations.', description='Preparation of systems for dynamic undocking simulations in OpenMM. The ligand, receptor and solvation box are parameterized with the specified forcefields. If specified, the receptor is reduced to a chunked pocket and the system is equilibrated.')
    openmm_prep.set_defaults(mode='openmm-preparation')
    openmm_prep_main = openmm_prep.add_argument_group('Main arguments')
    openmm_prep_main.add_argument('-y', '--yaml-input', type=str, default = None, help='Input yaml file with all the arguments for OpenMM preparation.')
    openmm_prep_main.add_argument('-l', '--ligand', type=str, default = None, help='Ligand MOL file to use as reference for interaction.')
    openmm_prep_main.add_argument('-i', '--interaction', type=str, default = None, help='Protein atom to use for ligand interaction.')
    openmm_prep_main.add_argument('-r', '--receptor', type=str, default = None, help='Protein or chunked protein in PDB format used as receptor.')
    openmm_chunk = openmm_prep.add_argument_group('Chunking arguments')
    openmm_chunk.add_argument('--do-chunk', action='store_true', help='Chunk initial receptor based on the interaction with ligand and add cappings.')
    openmm_chunk.add_argument('-c', '--cutoff', type=float, default = 9, help='Cutoff distance to define chunking (in Angstroms). Default = 9 A.')
    openmm_chunk.add_argument('-b', '--ignore-buffers', action='store_true', help='Do not remove buffers (solvent, ions etc.)')
    openmm_preprep = openmm_prep.add_argument_group('Parameterization arguments')
    openmm_preprep.add_argument('-f', '--small-molecule-forcefield', type=str,  default = 'SMIRNOFF', choices=('SMIRNOFF', 'GAFF2'), help='Small molecule forcefield to use for parameterization.')
    openmm_preprep.add_argument('-w', '--water-model', default='tip3p', type=str.lower, choices = ('tip3p', 'spce'), help='Water model to parameterize the solvent.')
    openmm_preprep.add_argument('-ff','--protein-forcefield', default='amber99sb', type=str.lower, choices=('amber99sb', 'amber14-all'), help='Protein forcefield to parameterize the chunked protein.')
    openmm_preprep.add_argument('-ion','--ionic-strength', default=0.1, type=float, help='Ionic strength (concentration) of the counter ion salts (Na+/Cl-). Default = 0.1 M')
    openmm_preprep.add_argument('-s','--solvent-buffer-distance', default=10, type=float, help='Buffer distance between the periodic box and the protein (in Angstroms). Default = 10 A')
    openmm_preprep.add_argument('-water','--waters-to-retain', default='waters_to_retain.pdb', type=str, help='PDB file containing structural water molecules to retain during simulations. Default is waters_to_retain.pdb.')
    openmm_preprep.add_argument('-fl','--fix-ligand', action='store_true', help='Some simple fixes for the ligand: ensure tetravalent nitrogens have the right charge assigned and add missing hydrogen atoms.')
    openmm_prepeq = openmm_prep.add_argument_group('Equilibration arguments')
    openmm_prepeq.add_argument('--do-equilibrate', action='store_true', help='Perform equilibration after preparing system.')
    openmm_prepeq.add_argument('-F', '--force-constant-eq', type=float, default = 1, help='Force constant for equilibration')
    openmm_prepeq.add_argument('-g', '--gpu-id', type=int, default=None, help='GPU ID; if not specified, runs on CPU only.')

    #Arguments for OpenMM full-protocol
    full = modes.add_parser('openmm-full-protocol', help='OpenDUck full OpenMM protocol.', description='Full dynamic undocking protocol in OpenMM. The ligand, receptor and solvation box are parameterized with the specified parameters. If specified, the receptor is reduced to a chunked pocket. After equilibration, serial iterations of MD and SMD are performed until the WQB or max_cycles threshold is reached.')
    full.set_defaults(mode='full-protocol')
    full_main = full.add_argument_group('Main arguments')
    full_main.add_argument('-y', '--yaml-input', type=str, default = None, help='Input yaml file with all the arguments for the full OpenDUck protocol.')
    full_main.add_argument('-l', '--ligand', type=str, default = None, help='Ligand MOL file to use as reference for interaction.')
    full_main.add_argument('-i', '--interaction', type=str, default = None, help='Protein atom to use for ligand interaction.')
    full_main.add_argument('-r', '--receptor', type=str, default = None, help='Protein or chunked protein in PDB format used as receptor.')
    full_main.add_argument('-g', '--gpu-id', type=int, default=None, help='GPU ID; if not specified, runs on CPU only.')
    openmm_chunk = full.add_argument_group('Chunking arguments')
    openmm_chunk.add_argument('--do-chunk', action='store_true', help='Chunk initial receptor based on the interaction with ligand and add cappings.')
    openmm_chunk.add_argument('-c', '--cutoff', type=float, default = 9, help='Cutoff distance to define chunking (in Angstroms). Default = 9 A.')
    openmm_chunk.add_argument('-b', '--ignore-buffers', action='store_true', help='Do not remove buffers (solvent, ions etc.)')
    prep = full.add_argument_group('Parameterization arguments')
    prep.add_argument('-f', '--small-molecule-forcefield', type=str,  default = 'SMIRNOFF', choices=('SMIRNOFF', 'GAFF2'), help='Small molecule forcefield to use for parameterization.')
    prep.add_argument('-w', '--water-model', default='tip3p', type=str.lower, choices = ('tip3p', 'spce'), help='Water model to parameterize the solvent.')
    prep.add_argument('-ff','--protein-forcefield', default='amber99sb', type=str.lower, choices=('amber99sb', 'amber14-all'), help='Protein forcefield to parameterize the chunked protein.')
    prep.add_argument('-ion','--ionic-strength', default=0.1, type=float, help='Ionic strength (concentration) of the counter ion salts (Na+/Cl-). Default = 0.1 M')
    prep.add_argument('-s','--solvent-buffer-distance', default=10, type=float, help='Buffer distance between the periodic box and the protein (in Angstroms). Default = 10 A')
    prep.add_argument('-water','--waters-to-retain', default='waters_to_retain.pdb', type=str, help='PDB file containing structural water molecules to retain during simulations. Default is waters_to_retain.pdb.')
    prep.add_argument('-fl','--fix-ligand', action='store_true', help='Some simple fixes for the ligand: ensure tetravalent nitrogens have the right charge assigned and add missing hydrogen atoms.')
    prod = full.add_argument_group('MD/SMD production arguments')
    prod.add_argument('-F', '--force-constant-eq', type=float, default = 1, help='Force constant for equilibration.')
    prod.add_argument('-n', '--smd-cycles', type=int, default = 20, help='Number of MD/SMD cycles to perform.')
    prod.add_argument('-m', '--md-length', type=float, default=0.5, help='Length of MD sampling between SMD runs in ns.')
    prod.add_argument('-W', '--wqb-threshold', type=float, default=None, help='Minimum WQB threshold; if not reached after each SMD cycle, further simulations will be terminated. If not set (default), all SMD cycles will be run.')
    prod.add_argument('-v', '--init-velocities', type=float, default=0.00001, help='Set initial velocities when heating.')
    prod.add_argument('-d', '--init-distance', type=float, default=2.5, help='Set initial hydrogen bond distance for SMD in Angstroms. Default = 2.5 A.')

    #Arguments for OpenMM form equilibrated system
    equil = modes.add_parser('openmm-from-equilibrated', help='OpenDUck OpenMM protocol starting from a pre-equilibrated system.', description='Dynamic undocking starting from a pre-equilibrated system. A chunk file from an equilibrated protein-ligand complex will be taken as input. After identifing the main interaction, serial iterations of MD and SMD are performed until the WQB or max_cycles threshold is reached.')
    equil.add_argument('-y', '--yaml-input', type=str, default = None, help='Input yaml file with all the arguments for the OpenMM simulations from the equilibrated system.')
    equil.add_argument('-s', '--equilibrated-system', default=None, help='Equilibrated system as input (*.chk).')
    equil.add_argument('-p', '--pickle', default=None, help='Pickle output from preparation.')
    equil.add_argument('-n', '--smd-cycles', type=int, default = 20, help='Number of MD/SMD cycles to perform.')
    equil.add_argument('-m', '--md-length', type=float, default=0.5, help='Length of MD sampling between SMD runs in ns.')
    equil.add_argument('-W', '--wqb-threshold', type=float, default=None, help='Minimum WQB threshold; if not reached after each SMD cycle, further simulations will be terminated. If not set (default), all SMD cycles will be run.')
    equil.add_argument('-v', '--init-velocities', type=float, default=0.00001, help='Set initial velocities when heating.')
    equil.add_argument('-d', '--init-distance', type=float, default=2.5, help='Set initial hydrogen bond distance for SMD in Angstroms. Default = 2.5 A.')
    equil.add_argument('-g', '--gpu-id', type=int, default=None, help='GPU ID; if not specified, runs on CPU only.')
    equil.set_defaults(mode='from-equilibration')

    openmm_prmtop = modes.add_parser('openmm-from-amber', help='OpenDUck OpenMM protocol starting from an AMBER topology and coordinates.', description='OpenDUck OpenMM protocol starting from an AMBER topology and coordinates. Using the AMBER topology (prmtop) and coordinates (inpcrd), identifies the main interaction and serial iterations of MD and SMD are performed until the WQB or max_cycles threshold is reached.')
    openmm_prmtop.set_defaults(mode='from-amber')
    openmm_prmtop.add_argument('-y', '--yaml-input', type=str, default = None, help='Input yaml file with all the arguments for the OpenMM simulations from the equilibrated system.')
    openmm_prmtop.add_argument('-c', '--coordinates', default=None, type=str, help='AMBER input coordinates.')
    openmm_prmtop.add_argument('-t', '--topology', default=None, type=str, help='AMBER input topology.')
    openmm_prmtop.add_argument('-i', '--interaction', default=None, type=str, help='Protein atom to use for ligand interaction.')
    openmm_prmtop.add_argument('-r', '--receptor', type=str, default = None, help='Receptor MOL2 file.')
    openmm_prmtop.add_argument('-n', '--smd-cycles', type=int, default = 20, help='Number of MD/SMD cycles to perform.')
    openmm_prmtop.add_argument('-m', '--md-length', type=float, default=0.5, help='Length of MD sampling between SMD runs in ns.')
    openmm_prmtop.add_argument('-W', '--wqb-threshold', type=float, default=None, help='Minimum WQB threshold; if not reached after each SMD cycle, further simulations will be terminated. If not set (default), all SMD cycles will be run.')
    openmm_prmtop.add_argument('-v', '--init-velocities', type=float, default=0.00001, help='Set initial velocities when heating.')
    openmm_prmtop.add_argument('-d', '--init-distance', type=float, default=2.5, help='Set initial hydrogen bond distance for SMD in Angstroms. Default = 2.5 A.')
    openmm_prmtop.add_argument('-g', '--gpu-id', type=int, default=None, help='GPU ID; if not specified, runs on CPU only.')

    #Arguments for OPENMM_PREPARATION
    amber = modes.add_parser('amber-prepare', help='Preparation of systems, inputs and queue files for AMBER simulations.', description='Preparation of systems, inputs and queue files for AMBER simulations. The ligand, receptor and solvation box are parameterized with the specified parameters. If specified, the receptor is reduced to a chunked pocket for a faster production. The input and queue files are prepared from templates found in the duck/templates directory.')
    amber_main = amber.add_argument_group('Main arguments')
    amber.set_defaults(mode='Amber-preparation')
    amber_main.add_argument('-y', '--yaml-input', type=str, default = None, help='Input yaml file with all the arguments for the system preparation and inputs/queueing for AMBER.')
    amber_main.add_argument('-l', '--ligand', type=str, default = None, help='Ligand MOL file to use as reference for interaction.')
    amber_main.add_argument('-i', '--interaction', type=str, default = None, help='Protein atom to use for ligand interaction.')
    amber_main.add_argument('-r', '--receptor', type=str, default = None, help='Protein or chunked protein in PDB format used as receptor.')
    amber_chunk = amber.add_argument_group('Chunking arguments')
    amber_chunk.add_argument('--do-chunk', action='store_true', help='Chunk initial receptor based on the interaction with ligand and add cappings.')
    amber_chunk.add_argument('-c', '--cutoff', type=float, default = 9, help='Cutoff distance to define chunking (in Angstroms). Default = 9 A.')
    amber_chunk.add_argument('-b', '--ignore-buffers', action='store_true', help='Do not remove buffers (solvent, ions etc.)')
    amber_prep = amber.add_argument_group('Parameterization arguments')
    amber_prep.add_argument('-f', '--small-molecule-forcefield', type=str,  default = 'SMIRNOFF', choices=('SMIRNOFF', 'GAFF2'), help='Small molecule forcefield to use for parameterization.')
    amber_prep.add_argument('-w', '--water-model', default='tip3p', type=str.lower, choices = ('tip3p', 'spce', 'tip4pew'), help='Water model to parameterize the solvent.')
    amber_prep.add_argument('-q', '--queue-template', type=str, default = 'local', help='Write out a queue file from templates.')
    amber_prep.add_argument('-H','--HMR', action='store_true', help ='Perform Hydrogen Mass Repartition on the topology and use it for the input files.')
    amber_prep.add_argument('-n', '--smd-cycles', type=int, default=20, help='Number of SMD replicas to perform.')
    amber_prep.add_argument('-W', '--wqb-threshold', type=float, default=0, help='Minimum WQB threshold; if not reached after each SMD cycle, further simulations will be terminated. If not set (default), all SMD cycles will be run.')
    amber_prep.add_argument('-ff','--protein-forcefield', default='amber99sb', type=str.lower, choices=('amber99sb', 'amber14-all'), help='Protein forcefield to parameterize the chunked protein.')
    amber_prep.add_argument('-ion','--ionic-strength', default=0.1, type=float, help='Ionic strength (concentration) of the counter ion salts (Na+/Cl-). Default = 0.1 M')
    amber_prep.add_argument('-s','--solvent-buffer-distance', default=10, type=float, help='Buffer distance between the periodic box and the protein (in Angstroms). Default = 10 A')
    amber_prep.add_argument('-water','--waters-to-retain', default='waters_to_retain.pdb', type=str, help='PDB file containing structural water molecules to retain during simulations. Default is waters_to_retain.pdb.')
    amber_prep.add_argument('--seed', default='-1', type=str, help='Specify seed for AMBER inputs.')
    amber_prep.add_argument('-B', '--batch', default=False, action='store_true', help='Enable batch processing for multi-ligand SDF.')
    amber_prep.add_argument('-t', '--threads', default=1, type=int, help='Define number of CPUs for batch processing.')
    amber_prep.add_argument('-fl','--fix-ligand', action='store_true', help='Some simple fixes for the ligand: ensure tetravalent nitrogens have the right charge assigned and add missing hydrogen atoms.')

    #Arguments for report
    report = modes.add_parser('report', help='Generate a report for OpenDUck results.', description='Generate a table report for dynamic undocking output. For a multi-ligand report, use the pattern flag with wildcards to the directories.')
    report.set_defaults(mode='Report')
    report.add_argument('-p', '--pattern', type=str, default='.', help='Wildcard pattern to find folders with DUck data.')
    report.add_argument('-d', '--data', type=str, default='min', choices=('min', 'single', 'avg', 'jarzynski', 'all'), help='Mode to compile the report [min | single | avg | jarzynski | all]')
    report.add_argument('-o', '--output', default='stdout', help = 'Output file; default is printing report to stdout.')
    report.add_argument('-of', '--output-format', default='tbl', choices=('csv', 'sdf', 'tbl') , type=str, help='Output format, [csv | sdf | tbl].')
    report.add_argument('--plot', default=False, action='store_true', help='Plot work or energy values to file.')
    report.add_argument('-s', '--subsample-size', default=20, type=int, help='Subsample size for Jarzynski bootstrapping.')
    report.add_argument('-i', '--iterations', default=20, type=int, help='Number of bootstrapping iterations for Jarzynski analysis.')
    report.add_argument('-t', '--step-threshold', default=2500, type=int, help='Step threshold to find the minima. Only needed with custom executions of DUck. Default = 2500 steps')
    report.add_argument('-f', '--format', type=str.lower, default='amber', choices=('amber', 'openmm'), help='Engine used to generate results. Default = amber')

    #Arguments for chunk
    chunk = modes.add_parser('chunk', help='Chunk a protein for dynamic undocking.', description='Reduce the receptor protein to a pocket by chunking it around the specified interaction. The amino acids within a certain cutoff are considered part of the chunk and the protein segments are capped.')
    chunk.set_defaults(mode='Chunk')
    chunk.add_argument('-y', '--yaml-input', type=str, default=None, help='Input yaml file with all the arguments for chunking.')
    chunk.add_argument('-l', '--ligand', type=str, default = None, help='Ligand MOL file to use as reference for interaction.')
    chunk.add_argument('-i', '--interaction', type=str, default = None, help='Protein atom to use for ligand interaction.')
    chunk.add_argument('-r', '--receptor', type=str, default = None, help='Protein or chunked protein in PDB format used as receptor.')
    chunk.add_argument('-c', '--cutoff', type=float, default = 9, help='Cutoff distance to define chunking (in Angstroms). Default = 9 A.')
    chunk.add_argument('-b', '--ignore-buffers', action='store_true', help='Do not remove buffers (solvent, ions etc.)')
    chunk.add_argument('-o', '--output', type=str, default='protein_out.pdb',help='Output format for the chunked protein receptor.')

    args = args_sanitation(parser, modes)

    return args, parser

def duck_smd_runs(input_checkpoint, pickle, num_runs, md_len, gpu_id, start_dist, init_velocity, save_dir, wqb_threshold=None):
    """
    Run molecular dynamics and steered molecular dynamics simulations on a system specified by an input checkpoint and a pickle file.

    Args:
        input_checkpoint (str): The path to the input checkpoint file.
        pickle (str): The path to the pickle file.
        num_runs (int): The number of runs to perform.
        md_len (float): The length of each MD run in picoseconds.
        gpu_id (int): The ID of the GPU to use.
        start_dist (float): The starting distance for the steered MD simulations.
        init_velocity (float): The initial velocity for the steered MD simulations.
        save_dir (str): The directory to save the output files.
        wqb_threshold (float): An optional threshold for the work required to break a bond. If specified and the work obtained is lower than the threshold, the function will terminate and return the wqb. Default is None.

    Returns:
       If the `wqb_threshold` is reached, the function returns the `wqb` value. Otherwise, the function returns None.
    """
    from duck.steps.normal_md import perform_md
    from duck.steps.steered_md import run_steered_md
    from duck.utils.check_system import check_if_equlibrated
    from duck.utils.analysis_and_report import get_Wqb_value
    import filecmp

    if not os.path.isfile(input_checkpoint):
        raise OSError(f'{input_checkpoint} can not be found. Something might have gone wrong during equilibration.')
    if not os.path.isfile(pickle):
        raise OSError(f'{pickle} can not be found. Something might have gone wrong during equilibration.')

    # Why is this shutil here? are the md/smd functions hardcoded to read this specific files?
    # check if files are the same, as shutil complains if it is the case.
    if not os.path.isfile('equil.chk'):
        shutil.copyfile(input_checkpoint, "equil.chk")
    if not os.path.isfile('copmlex_system.pickle'):
        shutil.copyfile(pickle, "complex_system.pickle")

    '''if not filecmp.cmp(input_checkpoint, 'equil.chk'):
        shutil.copyfile(input_checkpoint, "equil.chk")
    if not filecmp.cmp(pickle, "complex_system.pickle"):
        shutil.copyfile(pickle, "complex_system.pickle")
    '''

    # Now do the MD
    # remember start_dist
    if not Path(save_dir).exists(): save_dir.mkdir()
    for i in range(num_runs):
        if i == 0:
            md_start = str(input_checkpoint)
        else:
            md_start = str(Path(save_dir, "md_" + str(i - 1) + ".chk"))
        log_file = str(Path(save_dir, "md_" + str(i) + ".csv"))
        print(f'Simulating md_{str(i)}')
        perform_md(
            checkpoint_in_file=md_start,
            checkpoint_out_file=str(Path(save_dir, "md_" + str(i) + ".chk")),
            csv_out_file=log_file,
            pdb_out_file=str(Path(save_dir, "md_" + str(i) + ".pdb")),
            dcd_out_file=str(Path(save_dir, "md_" + str(i) + ".dcd")),
            md_len=md_len,
            gpu_id=gpu_id,
        )
        # Open the file and check that the potential is stable and negative
        if not check_if_equlibrated(log_file, 3):
            print("SYSTEM NOT EQUILIBRATED")
            sys.exit()
        # Now find the interaction and save to a file

        print(f'Simulating smd_{str(i)}_300')
        run_steered_md(
            300,
            str(Path(save_dir, "md_" + str(i) + ".chk")),
            str(Path(save_dir, "smd_" + str(i) + "_300.csv")),
            str(Path(save_dir, "smd_" + str(i) + "_300.dat")),
            str(Path(save_dir, "smd_" + str(i) + "_300.pdb")),
            str(Path(save_dir, "smd_" + str(i) + "_300.dcd")),
            start_dist,
            init_velocity=init_velocity,
            gpu_id=gpu_id,
        )
        #check if wqb is higher or lower than threshold to continue
        wqb, data, min_abs_work = get_Wqb_value(str(Path(save_dir, "smd_" + str(i) + "_300.dat")), mode='openmm')
        if wqb_threshold is not None and wqb < wqb_threshold:
            print(f'DUck replica {i}_300 yielded a wqb of {wqb}, which is lower that the wqb_threshold ({wqb_threshold}).\nStopping OpenDUck simulations.')
            return wqb
        else:
            print(f'Wqb: {wqb}')
        print(f'Simulating smd_{str(i)}_325')
        run_steered_md(
            325,
            str(Path(save_dir, "md_" + str(i) + ".chk")),
            str(Path(save_dir, "smd_" + str(i) + "_325.csv")),
            str(Path(save_dir, "smd_" + str(i) + "_325.dat")),
            str(Path(save_dir, "smd_" + str(i) + "_325.pdb")),
            str(Path(save_dir, "smd_" + str(i) + "_325.dcd")),
            start_dist,
            init_velocity=init_velocity,
            gpu_id=gpu_id,
        )
        #check if wqb is higher or lower than threshold to continue
        wqb, data, min_abs_work = get_Wqb_value(str(Path(save_dir, "smd_" + str(i) + "_325.dat")), mode='openmm')
        if wqb_threshold is not None and wqb < wqb_threshold:
            print(f'DUck replica {i}_325 yielded a wqb of {wqb}, which is lower that the wqb_threshold ({wqb_threshold}).\nStopping OpenDUck simulations.')
            return wqb
        else:
            print(f'Wqb: {wqb}')

def prepare_sys_for_amber(ligand_file, protein_file, chunk_file, interaction, HMR,  small_molecule_forcefield='SMIRNOFF', water_ff_str = 'tip3p.xml', forcefield_str='amber99sb.xml', ionic_strength = 0.1, box_buffer_distance = 10, waters_to_retain="waters_to_retain.pdb", seed='-1', fix_ligand_file=False):
    '''
    Prepares the system for AMBER simulation by parameterizing the ligand and finding the specified interaction
    between the ligand and protein. The resulting complex is saved to a pickle file, 'complex_system.pickle', and
    the AMBER inputs are written for the specified queue system.

    Args:
        ligand_file (str): Path to the ligand file in .mol2 or .PDB format.
        protein_file (str): Path to the protein-receptor file in .PDB format.
        chunk_file (str): Path to the chunk file in .PDB format.
        interaction (str): String specifying the receptor atom that forms the main interaction with the ligand.
        HMR (bool): Whether or not to use hydrogen mass repartitioning (HMR) during parameterization.
        small_molecule_forcefield (str, optional): Force field to use for ligand parameterization. Default is 'SMIRNOFF'.
        water_ff_str (str, optional): Force field to use for water molecules. Default is 'tip3p.xml'.
        forcefield_str (str, optional): Force field to use for protein parameterization. Default is 'amber99sb.xml'.
        ionic_strength (float, optional): Ionic strength of the system in M. Default is 0.1 M.
        box_buffer_distance (float, optional): Distance in Angstroms to buffer the protein-ligand complex from the edges of the simulation box. Default is 10.
        waters_to_retain (str, optional): Path to the file containing the water molecules to retain during simulation. Default is "waters_to_retain.pdb".
        seed (str, optional): Seed value for random number generation during parameterization. Default is '-1'.
        fix_ligand_file (bool, optional): Whether or not to fix the ligand file during parameterization. Default is False.
    '''
    from duck.steps.parametrize import prepare_system
    from duck.utils.cal_ints import find_interaction
    from duck.utils.amber_inputs import Amber_templates
    # Parameterize the ligand
    prepare_system(ligand_file, chunk_file, forcefield_str=forcefield_str,
                   hmr=HMR, small_molecule_ff=small_molecule_forcefield, water_ff_str = water_ff_str,
                   box_buffer_distance = box_buffer_distance, ionicStrength = ionic_strength, waters_to_retain=waters_to_retain, fix_ligand_file=fix_ligand_file)

    # Now find the interaction and save to a file
    results = find_interaction(interaction, protein_file)
    with open('complex_system.pickle', 'rb') as f:
        p = pickle.load(f) + results
    with open('complex_system.pickle', 'wb') as f:
        pickle.dump(p, f, protocol=pickle.HIGHEST_PROTOCOL)
    #p is a tupple with the following objects inside (parmed_structure, prot_index, ligand_index, pairmeandistance)
    p[0].save('system_complex.inpcrd', overwrite=True)

    AMBER = Amber_templates(structure=p[0], interaction=p[1:],hmr=HMR, seed=seed)
    amber.write_all_inputs()

def AMBER_prepare_ligand_in_folder(ligand_string, lig_indx, protein, chunk, interaction, HMR, base_dir, small_molecule_forcefield = 'SMIRNOFF', water_model = 'tip3p', forcefield = 'amber99sb', ion_strength = 0.1, box_buffer_distance = 10, waters_to_retain='waters_to_retain.pdb', seed='-1', fix_ligand=False):
    '''
    Generate the folder for a ligand preparation and prepare such ligand.
    '''
    from duck.utils.amber_inputs import write_string_to_file
    from contextlib import redirect_stdout,redirect_stderr

    os.chdir(base_dir)

    #Create the ligand folder
    if os.path.isdir(f'LIG_target_{lig_indx}'):
        print(f'WARNING: LIG_target_{lig_indx} already exist and it will be overwritten.')
        shutil.rmtree(f'./LIG_target_{lig_indx}', ignore_errors=True)
    os.mkdir(f'LIG_target_{lig_indx}')
    os.chdir(f'LIG_target_{lig_indx}')
    print(f'Working on LIG_target_{lig_indx}')
    with open('preparation.out', 'w') as o:
        with redirect_stdout(o):
            # Copying files to ligand foldef; ligand and prot
            write_string_to_file(string=ligand_string, file=f'lig_{lig_indx}.mol')
            shutil.copyfile(f'../{protein}', f'./{protein}', follow_symlinks=True)
            if os.path.isfile(f'../{waters_to_retain}'):
                shutil.copyfile(f'../{waters_to_retain}', f'./{waters_to_retain}', follow_symlinks=True)

            prepare_sys_for_amber(f'lig_{lig_indx}.mol', protein, chunk, interaction, HMR,
                                  small_molecule_forcefield=small_molecule_forcefield, water_ff_str=f'{water_model}',
                                  forcefield_str=f'{forcefield}.xml', ionic_strength = ion_strength,
                                  box_buffer_distance = box_buffer_distance, waters_to_retain=f"{waters_to_retain}", seed=seed, fix_ligand_file=fix_ligand)

    return(f'Lig_target_{lig_indx} prepared correctly')

#### main functions
def do_full_openMM_protocol(args):
    '''
    Perform full openduck in OpenMM protocol following old run_full_duck_pipeline.py script
    '''
    args.do_equilibrate = True
    do_OpenMM_preparation(args)

    # set up phase, I don't know why are the names changed here. Might be better to ommit it
    pickle_path = Path('complex_system.pickle')
    new_pickle_path = Path('cs.pickle')
    pickle_path.rename(new_pickle_path)
    equil_path = Path('equil.chk')
    new_equil_path = Path('eql.chk')
    equil_path.rename(new_equil_path)
    print('checkpoint_path', equil_path)
    save_dir = Path('duck_runs')
    if not save_dir.exists(): save_dir.mkdir()

    # Now production
    duck_smd_runs(input_checkpoint=new_equil_path,
                pickle=new_pickle_path,
                num_runs=args.smd_cycles,
                md_len=args.md_length,
                gpu_id=args.gpu_id,
                start_dist=args.init_distance,
                init_velocity=args.init_velocities,
                save_dir=save_dir,
                wqb_threshold=args.wqb_threshold)

def do_openMM_from_equil(args):
    '''
    Perform openduck from equilibrated chk in OpenMM following old run_from_eq.py script
    '''
    save_dir = Path('duck_runs')
    if not save_dir.exists(): save_dir.mkdir()
    #only need to do production
    duck_smd_runs(input_checkpoint=args.equilibrated_system,
            pickle=args.pickle,
            num_runs=args.smd_cycles,
            md_len=args.md_length,
            gpu_id=args.gpu_id,
            start_dist=args.init_distance,
            init_velocity=args.init_velocities,
            save_dir=save_dir,
            wqb_threshold=args.wqb_threshold)

def do_openMM_from_amber(args):
    '''
    Perform openduck from an AMBER topology. Extracted from the old 'from_amber_input.py'
    '''
    from duck.steps.equlibrate import equilibrate_from_amber_prep
    save_dir = Path('duck_runs')
    if not save_dir.exists(): save_dir.mkdir()

    equilibrate_from_amber_prep(interaction=args.interaction,
                                prmtop_file=args.topology,
                                inpcrd_file=args.coordinates,
                                chunk=args.receptor,
                                gpu_id=args.gpu_id,
                                force_constant_eq=1.0)

    pickle_path = Path('complex_system.pickle')
    new_pickle_path = Path('cs.pickle')
    pickle_path.rename(new_pickle_path)

    equil_path = Path('equil.chk')
    new_equil_path = Path('eql.chk')
    equil_path.rename(new_equil_path)
    print('checkpoint_path', equil_path)

    duck_smd_runs(input_checkpoint=new_equil_path,
                  pickle=new_pickle_path,
                  num_runs=args.smd_cycles,
                  md_len=args.md_length,
                  gpu_id=args.gpu_id,
                  start_dist=args.init_distance,
                  init_velocity=args.init_velocities,
                  save_dir=save_dir,
                  wqb_threshold=args.wqb_threshold)

def do_AMBER_preparation(args):
    '''
    Prepare the protein-ligand complex for OpenDUck protocol with the specified arguments and generate the AMBER input files and queque files.
    '''
    # adapted from duck_prepare_sys_for_amber.py and batch_duck_prepare_for_amber.py
    from duck.utils.amber_inputs import Queue_templates

    # create chunk if needed
    if args.do_chunk:
        from duck.steps.chunk import duck_chunk
        print('Chunking protein')
        chunk_file = duck_chunk(args.receptor,args.ligand,args.interaction,args.cutoff, ignore_buffers=args.ignore_buffers)
    else: chunk_file = args.receptor

    if args.batch:
        import multiprocessing as mp
        from duck.utils.amber_inputs import log_result, handle_error, ligand_string_generator
        pool = mp.Pool(args.threads)
        base_dir = os.getcwd()

        r = [pool.apply_async(AMBER_prepare_ligand_in_folder,
                          args=(ligand_string, j+1, args.receptor, chunk_file,
                                args.interaction, args.HMR, base_dir,
                                args.small_molecule_forcefield, args.water_model, args.protein_forcefield,
                                args.ionic_strength, args.solvent_buffer_distance, args.waters_to_retain, args.seed),
                          callback=log_result,
                          error_callback=handle_error) for j, ligand_string in enumerate(ligand_string_generator(args.ligand))]
        pool.close()
        pool.join()

        queue = Queue_templates(wqb_threshold=args.wqb_threshold, replicas=args.smd_cycles, array_limit=len(r), hmr=args.HMR)
    else:
        if args.threads != 1:
            print('WARNING: The number of threads does not have an impact if the batch mode is not enabled.')
        prepare_sys_for_amber(args.ligand, args.receptor, chunk_file, args.interaction, args.HMR,
        small_molecule_forcefield=args.small_molecule_forcefield, water_ff_str = args.water_model,
        forcefield_str=f'{args.protein_forcefield}.xml', ionic_strength = args.ionic_strength,
        box_buffer_distance = args.solvent_buffer_distance, waters_to_retain=args.waters_to_retain, seed=args.seed, fix_ligand_file=args.fix_ligand)
        queue = Queue_templates(wqb_threshold=args.wqb_threshold, replicas=args.smd_cycles, hmr=args.HMR)
    queue.write_queue_file(kind=args.queue_template)

def do_report(args):
    '''
    Generate a report from the dynamic undocking output. The DUck folders are looked upon the args.pattern.
    The formating input depends on the args.format (either 'openmm' or 'amber'), and the output generated depends on the args.data argument (min, single, avg, jarzynski or all).
    '''
    from duck.utils.analysis_and_report import get_Wqb_value_AMBER_all, get_Wqb_value_Openmm_all, do_jarzynski_analysis, build_report_df, get_mols_and_format
    import glob
    #iterate_folders
    folders = glob.glob(args.pattern)
    wqb_info = {}
    for folder in folders:
        wqb_info.setdefault(folder, [])
        #calculate_wqb and/or jaryznski
        currdir = os.getcwd()
        os.chdir(folder)
        if args.data in ('min', 'single', 'avg', 'all'):
            if args.format == 'amber':
                wqb = get_Wqb_value_AMBER_all(prefix='DUCK', file='duck.dat', plot=args.plot)
            elif args.format == 'openmm':
                wqb = get_Wqb_value_Openmm_all(folder='duck_runs', pattern='smd_*.dat', plot=args.plot)
            wqb_info[folder].extend(wqb)
        if args.data == 'jarzynski' or args.data == 'all':
            expavg, sd, sem_v = do_jarzynski_analysis(index_threshold = args.step_threshold, sample_size=args.subsample_size, samples=args.iterations, plot=args.plot, mode =args.format)
            wqb_info[folder].extend([expavg, sd, sem_v])
        os.chdir(currdir)

    df = build_report_df(wqb_info, mode=args.data)
    if args.output_format == 'csv':
        df.to_csv(args.output, index=False)
    elif args.output_format == 'tbl':
        df.to_csv(args.output, index=False, sep='\t')
    elif args.output_format == 'sdf':
        from rdkit import Chem
        mols = get_mols_and_format(df, mode=args.data)
        with Chem.SDWriter(args.output) as w:
            [w.write(mol) for mol in mols]

def do_OpenMM_preparation(args):
    '''
    Prepare the protein-ligand complex from the parsed arguments and perform equilibration using OpenMM
    '''
    from duck.utils.check_system import check_if_equlibrated
    from duck.steps.equlibrate import do_equlibrate
    from duck.steps.parametrize import prepare_system
    from duck.utils.cal_ints import find_interaction
    # create chunk
    if args.do_chunk:
        print('Chunking protein')
        from duck.steps.chunk import duck_chunk
        chunked_file = duck_chunk(args.receptor,args.ligand,args.interaction,args.cutoff, ignore_buffers=args.ignore_buffers)
    else: chunked_file = args.receptor
    # prepare system
    prepare_system(args.ligand, chunked_file, forcefield_str=f'{args.protein_forcefield}.xml', water_ff_str = f'{args.water_model}',
            small_molecule_ff=args.small_molecule_forcefield, waters_to_retain=args.waters_to_retain,
            box_buffer_distance = args.solvent_buffer_distance, ionicStrength = args.ionic_strength, fix_ligand_file=args.fix_ligand)
    results = find_interaction(args.interaction, args.receptor)
    with open('complex_system.pickle', 'rb') as f:
        p = pickle.load(f) + results
    with open('complex_system.pickle', 'wb') as f:
        pickle.dump(p, f, protocol=pickle.HIGHEST_PROTOCOL)
    p[0].save('system_complex.inpcrd', overwrite=True)

    # Equlibration
    print(results)
    if args.do_equilibrate:
        do_equlibrate(force_constant_equilibrate=args.force_constant_eq, gpu_id=args.gpu_id, keyInteraction=results)
        if not check_if_equlibrated("density.csv", 1):
            raise EquilibrationError("System is not equilibrated.") # Does this exist?

def main():
    # Parse and sanitize the inputs
    args, parser = parse_input()

    # Chose and perform the specified action
    if args.mode == 'full-protocol':
        do_full_openMM_protocol(args)
    elif args.mode == 'openmm-preparation':
        do_OpenMM_preparation(args)
    elif args.mode == 'from-equilibration':
        do_openMM_from_equil(args)
    elif args.mode == 'from-amber':
        #WIP
        do_openMM_from_amber(args)
        print('Still in progress')
    elif args.mode == 'Amber-preparation':
        do_AMBER_preparation(args)
    elif args.mode == 'Report':
        do_report(args)
    elif args.mode == 'Chunk':
        from duck.steps.chunk import duck_chunk
        print(args)
        duck_chunk(args.receptor,args.ligand,args.interaction,args.cutoff,output_name=args.output, ignore_buffers=args.ignore_buffers)
    else:
        parser.print_help()
if __name__ == '__main__':
    main()
