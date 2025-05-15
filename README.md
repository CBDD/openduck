[![stable](http://badges.github.io/stability-badges/dist/stable.svg)](http://github.com/badges/stability-badges)
[![Version](http://img.shields.io/badge/version-0.2.0-blue.svg?style=flat)](https://github.com/CBDD/openduck)
[![License](http://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat)](https://github.com/CBDD/openduck/LICENSE)

# Welcome to OpenDUck

OpenDUck is an open-source implementation of dynamic undocking (DUck), an application of steered molecular dynamics developed as a fast algorithm to calculate the work necessary to reach a quasi-bound state, at which a certain native contact is broken between the ligand and receptor. Due to its cheap and fast nature, DUck is mainly employed as a post-docking filter during virtual screening campaigns.

More details can be found in the original dynamic undocking publication:

> Ruiz-Carmona, S., Schmidtke, P., Luque, F. et al. Dynamic undocking and the quasi-bound state as tools for drug discovery. Nature Chem 9, 201–206 (2017). https://doi.org/10.1038/nchem.2660

## Installation

### Conda (update)

We recommend you install OpenDUck into a fresh Conda environment.
```{bash}
$ git clone git@github.com:CBDD/openduck.git
$ cd openduck
$ conda env create -f environment.yaml
$ conda activate openduck
$ python setup.py install
```
### Singularity
A Singularity definition file (`openDUck_singularity.def`) is included in the repository.  
This can be used to build a self-contained Singularity image, allowing you to conveniently run OpenDUck calculations without worrying about installing dependencies manually.

To build the Singularity image, run:

```{bash}
singularity build openDUck.sif openDUck_singularity.def 
```

Once built, you can execute OpenDUck commands directly inside the container. For example:

```{bash}
singularity exec openDUck.sif openduck -h
```

## OpenDUck usage

OpenDUck can be used either as a Python library or as an executable. The executable provided is `openduck`, which is automatically added to your path if you install with pip or conda. Alternatively you can use openduck.py in the scripts subdirectory.

OpenDUck comes with several submodules to launch the different steps of dynamic undocking:

```{bash}
$ conda activate openduck

$ openduck -h

usage: openduck [-h]  ...

Open-source toolkit for dynamic undocking.

optional arguments:
  -h, --help            show this help message and exit

Open-source dynamic undocking toolkit. Choose one of the following actions::

    openmm-prepare      Preparation of systems for OpenMM simulations.
    openmm-full-protocol
                        OpenDUck full OpenMM protocol.
    openmm-from-equilibrated
                        OpenDUck OpenMM protocol starting from a pre-equilibrated system.
    openmm-from-amber   OpenDUck OpenMM protocol starting from an AMBER topology and coordinates.
    amber-prepare       Preparation of systems, inputs and queue files for AMBER simulations.
    report              Generate a report for OpenDUck results.
    chunk               Chunk a protein for dynamic undocking.
```

Each of the OpenDUck modules accepts the input either as command line arguments or as a yaml file.
The arguments in the input.yaml are expected to follow the command-line nomenclature.

### Running OpenDUck with OpenMM

The OpenMM implementation of OpenDUck is divided in two steps: the preparation using `openmm_forcefields` and the production (MD/SMD) following the dynamic undocking protocol.
During the preparation step, the protein is reduced to the pocket region during the _chunking_ step, then the ligand and receptor are parametrized and the solvation box is generated. After equilibration, the sequential MD/SMD simulations, sampling and steering the main interaction from the specified distances, are run in OpenMM.

The openMM submodules are set up to allow independent usage of the previously mentioned steps:

    * `openmm-full-protocol`: Executes the full OpenDuck protocol, from the protein-ligand files to the steered simulations
    * `openmm-prepare`: Executes only the preparation and equilibration of the protein-ligand complex
    * `openmm-from-equilibrated`: Executes only the production of MD/SMD runs from an equilibrated input.
    * `openmm-from-amber`: Starts the openDuck simulations from an AMBER topology.

#### openmm-full-protocol

The complete OpenDUck protocol through OpenMM can be executed using the `openmm-full-protocol` submodule.
The full protocol executes the following steps:

    * Chunking (optional)
    * Preparation
      * Ligand parametrization
      * Receptor parametrization
      * Box generation
    * Equilibration
    * (MD + SMD@300K + SMD@325K + WQB check ) x smd-cycles

```{bash}
$ openduck openmm-full-protocol -h

usage: openduck openmm-full-protocol [-h] [-y YAML_INPUT] [-l LIGAND] [-i INTERACTION] [-r RECEPTOR] [-g GPU_ID] [--keep-all-files] [--do-chunk] [-c CUTOFF] [-b] [-f {SMIRNOFF,GAFF2}]
                                     [-w {tip3p,spce}] [-ff {amber99sb,amber14-all}] [-ion IONIC_STRENGTH] [-s SOLVENT_BUFFER_DISTANCE] [-water WATERS_TO_RETAIN] [-cf CUSTOM_FORCEFIELD]
                                     [-fl] [-H] [-F FORCE_CONSTANT_EQ] [-n SMD_CYCLES] [-m MD_LENGTH] [-W WQB_THRESHOLD] [-v INIT_VELOCITIES] [-d INIT_DISTANCE] [-B] [-t THREADS] [--resume]
                                     [-i0 INDEX0] [-p PREFIX]

Full dynamic undocking protocol in OpenMM. The ligand, receptor and solvation box are parameterized with the specified parameters. If specified, the receptor is reduced to a chunked pocket.
After equilibration, serial iterations of MD and SMD are performed until the WQB or max_cycles threshold is reached.

optional arguments:
  -h, --help            show this help message and exit

Main arguments:
  -y YAML_INPUT, --yaml-input YAML_INPUT
                        Input yaml file with all the arguments for the full OpenDUck protocol.
  -l LIGAND, --ligand LIGAND
                        Ligand MOL file to use as reference for interaction.
  -i INTERACTION, --interaction INTERACTION
                        Protein atom to use for ligand interaction.
  -r RECEPTOR, --receptor RECEPTOR
                        Protein or chunked protein in PDB format used as receptor.
  -g GPU_ID, --gpu-id GPU_ID
                        GPU ID; if not specified, runs on CPU only.
  --keep-all-files      Disable cleaning up intermediate files during simulations.

Chunking arguments:
  --do-chunk            Chunk initial receptor based on the interaction with ligand and add cappings.
  -c CUTOFF, --cutoff CUTOFF
                        Cutoff distance to define chunking (in Angstroms). Default = 9 A.
  -b, --ignore-buffers  Do not remove buffers (solvent, ions etc.)

Parameterization arguments:
  -f {SMIRNOFF,GAFF2}, --small-molecule-forcefield {SMIRNOFF,GAFF2}
                        Small molecule forcefield to use for parameterization.
  -w {tip3p,spce}, --water-model {tip3p,spce}
                        Water model to parameterize the solvent.
  -ff {amber99sb,amber14-all}, --protein-forcefield {amber99sb,amber14-all}
                        Protein forcefield to parameterize the chunked protein.
  -ion IONIC_STRENGTH, --ionic-strength IONIC_STRENGTH
                        Ionic strength (concentration) of the counter ion salts (Na+/Cl-). Default = 0.1 M
  -s SOLVENT_BUFFER_DISTANCE, --solvent-buffer-distance SOLVENT_BUFFER_DISTANCE
                        Buffer distance between the periodic box and the protein (in Angstroms). Default = 10 A
  -water WATERS_TO_RETAIN, --waters-to-retain WATERS_TO_RETAIN
                        PDB file containing structural water molecules to retain during simulations. Default is waters_to_retain.pdb.
  -cf CUSTOM_FORCEFIELD, --custom-forcefield CUSTOM_FORCEFIELD
                        Custom forcefield (in Open Force Field XML format) to parameterize e.g. a cofactor or unnatural amino acid present in the PDB file included under --receptor. Will be
                        used in addition to the forcefields specified by --small-molecule-forcefield and --protein-forcefield.
  -fl, --fix-ligand     Some simple fixes for the ligand: ensure tetravalent nitrogens have the right charge assigned and add missing hydrogen atoms.
  -H, --HMR             Perform Hydrogen Mass Repartition on the topology and run at dt=0.04 ps.

MD/SMD production arguments:
  -F FORCE_CONSTANT_EQ, --force-constant-eq FORCE_CONSTANT_EQ
                        Force constant for equilibration.
  -n SMD_CYCLES, --smd-cycles SMD_CYCLES
                        Number of MD/SMD cycles to perform.
  -m MD_LENGTH, --md-length MD_LENGTH
                        Length of MD sampling between SMD runs in ns.
  -W WQB_THRESHOLD, --wqb-threshold WQB_THRESHOLD
                        Minimum WQB threshold; if not reached after each SMD cycle, further simulations will be terminated. If not set (default), all SMD cycles will be run.
  -v INIT_VELOCITIES, --init-velocities INIT_VELOCITIES
                        Set initial velocities when heating.
  -d INIT_DISTANCE, --init-distance INIT_DISTANCE
                        Set initial hydrogen bond distance for SMD in Angstroms. Default = 2.5 A.

Batch arguments:
  -B, --batch           Enable batch processing for multi-ligand SDF.
  -t THREADS, --threads THREADS
                        Define number of CPUs for batch processing.
  --resume              Enable the resume mode. Protecting LIG_target folders already prepared and starting from the not done, avoiding overwritting.
  -i0 INDEX0, --index0 INDEX0
                        Starting index for naming batch ligands. Default: 1.
  -p PREFIX, --prefix PREFIX
                        Prefix to name ligand folder during batch preparation. Default: LIG_target.
```

A valid example is provided in the test subfolder, and can be executed using the command-line arguments like the following:

```{bash}
$ openduck openmm-full-protocol -l brd4_lig.mol -r 4LR6_aligned_chunk_nowat.pdb -i A_ASN_140_ND2 \
                                -g 0 -f GAFF2 -ff amber14-all -w tip3p -w 6 -n 20
```

Similarly, an execution with the input parameters in a yaml is also possible (the example has slightly different parameters as the command line, just for ilustration purposes).
```{bash}
openduck openmm-full-protocol -y input.yaml
```
where the `input.yaml` file has the following
```{yaml}
# Main Arguments
interaction : A_ASN_140_ND2
receptor_pdb : 4LR6_aligned_chunk_nowat.pdb
ligand_mol : brd4_lig.mol
gpu_id : 0

# Chunking Arguments
do_chunk : True
cutoff : 10
ignore_buffers : False

# Preparation Arguments
small_molecule_forcefield : smirnoff
protein_forcefield : amber14-all
water_model : tip3p
ionic_strength : 0.05
waters_to_retain : waters_to_retain.pdb
solvent_buffer_distance : 15
force_constant_eq : 1

# Production Arguments
smd_cycles : 20
md_length : 0.5
wqb_threshold : 6
init_velocities : 0.00001
init_distance : 2.5
fix_ligand : False
```
#### Batch launching OpenMM SLURM jobs with `openmm_job_launcher`

For larger sets of ligands, OpenDUck now provides a dedicated launcher tool called `openmm_job_launcher` to facilitate batch execution across SLURM clusters.

This script automates:
- Splitting a multi-ligand SDF into batches.
- Preparing corresponding input SDF and YAML configuration files.
- Creating SLURM submission scripts for each job.
- Optionally running the jobs inside a Singularity container.
- Supporting both CPU and GPU executions.

The launcher is particularly useful for virtual screening campaigns where a large number of ligands need to be processed efficiently.

You can access the help menu with:

```bash
openmm-job-launcher --help
```
A basic usage example:
```bash
openmm-job-launcher --ligand my_ligands.sdf --protein receptor.pdb --interaction A_ASN_140_ND2 --output jobs_output --jobs 4 --gpu --smd-cycles 5
```

This will:

- Split the input ligands into 10 jobs.
- Prepare the required input files for each job.
- Create SLURM scripts ready for submission.
- Launch the jobs automatically if sbatch is available.

There is also support for running jobs inside a Singularity container, which can be especially useful on HPC systems.
If a native `openduck` installation is not detected, the launcher will automatically attempt to use a default `openDUck.sif` image, if available.
Alternatively, you can explicitly specify a Singularity image using the `--singularity [singularity_path]` argument:

```bash
openmm-job-launcher --ligand my_ligands.sdf --protein receptor.pdb --interaction A_ASN_140_ND2 --output jobs_output --jobs 10 --gpu --singularity openDUck.sif
```

#### openmm-prepare & openmm-from-equilibrated

The OpenDUck protocol can also be executed independently in two steps: the preparation & equilibration with the `openmm-prepare` subcommand and the production with the `openmm-from-equilibrated` subcommand. Following the schema presented above, each substep is performed by running the following subcommands.

    openmm-prepare
      * Chunking (optional)
      * Preparation
        * Ligand parametrization
        * Receptor parametrization
        * Box generation
      * Equilibration
    openmm-from-equilibrated
      * (MD + SMD@300K + SMD@325K + WQB check ) x smd-cycles

Each of the two subcommands share arguments with the openmm-full-protocol, based on the execution steps they have in common. As with the previous command, the execution can come from flagged arguments or with the same arguments in a yaml input file.

Sample yaml input files for `openmm-prepare` and `openmm-from-equilibrated` are provided below:

```
# openmm-prepare input.yaml
# Main arguments
ligand_mol : '1a28_lig.mol'
receptor_pdb : '1a28_prot.pdb'
interaction : 'A_ARG_766_NH2'

# Chunking arguments
do_chunk : True
cutoff : 10
ignore_buffers : False

#Preparation arguments
small_molecule_forcefield : 'GAFF2'
protein_forcefield : 'amber14-all'
water_model : 'spce'
ionic_strength : 0.1
solvent_buffer_distance : 12

#Equilibration arguments
do_equilibrate : True
gpu_id : 0
force_constant_eq : 1
```

```
 # openmm-from-equilibrated
equilibrated_system : equil.chk
pickle : complex_system.pickle
smd_cycles : 20
md_length : 0.5
wqb_threshold : 6
gpu_id : 0
```

#### openmm-from-amber

Alternatively, the OpenDUck protocol can be launched from an AMBER topology and coordinates. This can be done using the `openmm-from-amber` subcommand.

``` {bash}
$ openduck openmm-from-amber -h

usage: openduck openmm-from-amber [-h] [-y YAML_INPUT] [-c COORDINATES] [-t TOPOLOGY] [-i INTERACTION]
                                  [-r RECEPTOR] [-n SMD_CYCLES] [-m MD_LENGTH] [-W WQB_THRESHOLD]
                                  [-v INIT_VELOCITIES] [-d INIT_DISTANCE] [-g GPU_ID]

OpenDUck OpenMM protocol starting from an AMBER topology and coordinates. Using the AMBER topology (prmtop) and
coordinates (inpcrd), identifies the main interaction and serial iterations of MD and SMD are performed until the
WQB or max_cycles threshold is reached.

optional arguments:
  -h, --help            show this help message and exit
  -y YAML_INPUT, --yaml-input YAML_INPUT
                        Input yaml file with all the arguments for the OpenMM simulations from the
                        equilibrated system.
  -c COORDINATES, --coordinates COORDINATES
                        AMBER input coordinates.
  -t TOPOLOGY, --topology TOPOLOGY
                        AMBER input topology.
  -i INTERACTION, --interaction INTERACTION
                        Protein atom to use for ligand interaction.
  -r RECEPTOR, --receptor RECEPTOR
                        Receptor MOL2 file.
  -n SMD_CYCLES, --smd-cycles SMD_CYCLES
                        Number of MD/SMD cycles to perform.
  -m MD_LENGTH, --md-length MD_LENGTH
                        Length of MD sampling between SMD runs in ns.
  -W WQB_THRESHOLD, --wqb-threshold WQB_THRESHOLD
                        Minimum WQB threshold; if not reached after each SMD cycle, further
                        simulations will be terminated. If not set (default), all SMD cycles will
                        be run.
  -v INIT_VELOCITIES, --init-velocities INIT_VELOCITIES
                        Set initial velocities when heating.
  -d INIT_DISTANCE, --init-distance INIT_DISTANCE
                        Set initial hydrogen bond distance for SMD in Angstroms. Default = 2.5 A.
  -g GPU_ID, --gpu-id GPU_ID
                        GPU ID; if not specified, runs on CPU only.
  --keep-all-files      Disable cleaning up intermediate files during simulations.

```

If preferred, the script can also be executed from a yaml file with the commands. A sample input.yaml file could be the following:
```{yaml}
interaction: A_ASN_140_ND2
coordinates: system_complex.inpcrd
topology: system_complex.prmtop
receptor: 4LR6_aligned_chunk_nowat.pdb
smd_cycles: 20
wqb_threshold: 6
gpu_id: 0
```

### Running OpenDUck in AMBER

Originally dynamic undocking was designed to run in AMBER; in OpenDUck we have maintained the same protocol for AMBER, replacing the parametrization and file-generation previously done by MOE with an open-source implementation. As with the original protocol, the user is free to choose their preferred queueing headers and preparation via either the commandline arguments or the input yaml file.
Preparation flags are very similar to those for OpenMM, as most of the implementation is shared.

```{bash}
$ openduck amber-prepare -h

usage: openduck amber-prepare [-h] [-y YAML_INPUT] [-l LIGAND] [-i INTERACTION] [-r RECEPTOR] [--do-chunk]
                              [-c CUTOFF] [-b] [-f {SMIRNOFF,GAFF2}] [-w {tip3p,spce,tip4pew}]
                              [-q QUEUE_TEMPLATE] [-H] [-n SMD_CYCLES] [-W WQB_THRESHOLD]
                              [-ff {amber99sb,amber14-all}] [-ion IONIC_STRENGTH] [-s SOLVENT_BUFFER_DISTANCE]
                              [-water WATERS_TO_RETAIN] [--seed SEED] [-B] [-t THREADS] [-fl]

Preparation of systems, inputs and queue files for AMBER simulations. The ligand, receptor and solvation box are
parameterized with the specified parameters. If specified, the receptor is reduced to a chunked pocket for a
faster production. The input and queue files are prepared from templates found in the duck/templates directory.

optional arguments:
  -h, --help            show this help message and exit

Main arguments:
  -y YAML_INPUT, --yaml-input YAML_INPUT
                        Input yaml file with all the arguments for the system preparation and
                        inputs/queueing for AMBER.
  -l LIGAND, --ligand LIGAND
                        Ligand MOL file to use as reference for interaction.
  -i INTERACTION, --interaction INTERACTION
                        Protein atom to use for ligand interaction.
  -r RECEPTOR, --receptor RECEPTOR
                        Protein or chunked protein in PDB format used as receptor.

Chunking arguments:
  --do-chunk            Chunk initial receptor based on the interaction with ligand and add
                        cappings.
  -c CUTOFF, --cutoff CUTOFF
                        Cutoff distance to define chunking (in Angstroms). Default = 9 A.
  -b, --ignore-buffers  Do not remove buffers (solvent, ions etc.)

Parameterization arguments:
  -f {SMIRNOFF,GAFF2}, --small-molecule-forcefield {SMIRNOFF,GAFF2}
                        Small molecule forcefield to use for parameterization.
  -w {tip3p,spce,tip4pew}, --water-model {tip3p,spce,tip4pew}
                        Water model to parameterize the solvent.
  -q QUEUE_TEMPLATE, --queue-template QUEUE_TEMPLATE
                        Write out a queue file from templates.
  -H, --HMR             Perform Hydrogen Mass Repartition on the topology and use it for the input
                        files.
  -n SMD_CYCLES, --smd-cycles SMD_CYCLES
                        Number of SMD replicas to perform.
  -W WQB_THRESHOLD, --wqb-threshold WQB_THRESHOLD
                        Minimum WQB threshold; if not reached after each SMD cycle, further
                        simulations will be terminated. If not set (default), all SMD cycles will
                        be run.
  -ff {amber99sb,amber14-all}, --protein-forcefield {amber99sb,amber14-all}
                        Protein forcefield to parameterize the chunked protein.
  -ion IONIC_STRENGTH, --ionic-strength IONIC_STRENGTH
                        Ionic strength (concentration) of the counter ion salts (Na+/Cl-). Default
                        = 0.1 M
  -s SOLVENT_BUFFER_DISTANCE, --solvent-buffer-distance SOLVENT_BUFFER_DISTANCE
                        Buffer distance between the periodic box and the protein (in Angstroms).
                        Default = 10 A
  -water WATERS_TO_RETAIN, --waters-to-retain WATERS_TO_RETAIN
                        PDB file containing structural water molecules to retain during
                        simulations. Default is waters_to_retain.pdb.
  -cf CUSTOM_FORCEFIELD, --custom-forcefield CUSTOM_FORCEFIELD
                        Custom forcefield (in Open Force Field XML format) to parameterize e.g. a
                        cofactor or unnatural amino acid present in the PDB file included under
                        --receptor. Will be used in addition to the forcefields specified by --small-
                        molecule-forcefield and --protein-forcefield.
  --seed SEED           Specify seed for AMBER inputs.
  -fl, --fix-ligand     Some simple fixes for the ligand: ensure tetravalent nitrogens have the
                        right charge assigned and add missing hydrogen atoms.
  --water-steering      Enable water steering, which will use the waters-to-retain as interaction
                        vector to steer the ligand.
  --waters-to-restrain WATERS_TO_RESTRAIN
                        Number (and order) of waters to restraint. Default is None when executing in
                        the normal mode and 1 when using water-steering.
  -e LIGAND_HB_ELEMENTS [LIGAND_HB_ELEMENTS ...], --ligand-hb-elements LIGAND_HB_ELEMENTS [LIGAND_HB_ELEMENTS ...]
                        Control which elements are accepted in the ligand to define the steering
                        interaction. Specify using the atomic number separated with a space. Default is
                        7 and 8 (nitrogen and oxygen)

Batch argments:
  -B, --batch           Enable batch processing for multi-ligand SDF.
  -t THREADS, --threads THREADS
                        Define number of CPUs for batch processing.
  --keep-all-files      Disable cleaning up intermediate files during preparation and simulations.
  --resume              Enable the resume mode. Protecting LIG_target folders already prepared and
                        starting from the not done, avoiding overwritting.
  -i0 INDEX0, --index0 INDEX0
                        Starting index for naming batch ligands. Default: 1.
  -p PREFIX, --prefix PREFIX
                        Prefix to name ligand folder during batch preparation. Default:
                        LIG_target.
```

```{bash}
$ openduck amber-prepare -l ../1a28_lig.mol -r ../1a28_prot.pdb -i A_ARG_766_NH2 \
                         --do-chunk -c 10 -b False \
                         -f GAFF2 -ff amber99sb -w SPCE -ion 0.05 -s 11 \
                         --HMR -n 4 -W 4 -q Slurm
```

Typically, dynamic undocking is employed as a post-docking filter in a virtual screening campaign. As such, multiple ligands might be generated simultaneously. A batch execution of the DUck preparation can be executed specifying the `--batch` argument and providing an SD-file with multiple ligands as input. Each ligand and the associated file structure will be generated in the `LIG_target_{n}` subfolder, where `n` is the ligand's position in the input SDF.

```{bash}
$ openduck amber-prepare -l ../brd4_ligands.sdf -r ../4LR6_aligned_chunk_nowat.pdb -i A_ASN_140_ND2 \
                         --waters-to-retain ../waters_to_retain.pdb -f gaff2 -ff amber14-all -w spce -ion 1 -s 30
                         --HMR --smd-cycles 10 -wqb-threshold 8 -q SGE --seed -1 --batch --threads 8
```

The queueing template is stored in `duck/templates/queueing_templates`. To customize a template, a new file can be added in the directory with the following format: `{queue_name}_array.q` or `{queue_name}.q` either if the expected execution is in an array or not. For a local execution, the `local` argument can be given for a plain list of commands.

The templates have the following format:
```
#!/bin/bash
#HEADER

{functions}

{commands}
```

### Analysis

After successfully running OpenDUck using either OpenMM or AMBER, the results can be compiled using the report submodule.
By default, the lowest $W_{QB}$ is used as a reporter for the robustness of the studied interaction, however, a better (and more formally appropiate) descriptor has been shown to be the $\Delta_{QB}$ obtained using the Jarzynski equality. When specified, a bootstrapping is performed to assess the convergence of the reported value.

**Important**

Due to the different formating of AMBER and OpenMM steered molecular dynamics output, the `--format` argument is needed when analyzing either output.
When analyzing a single result, the default pattern is the current directory (`.`). If you wish to analyze multiple directories, specify the wildcard pattern.

```{bash}
$ openduck report -h

usage: openduck report [-h] [-p PATTERN] [-d {min,single,avg,jarzynski,all}] [-o OUTPUT] [-of {csv,sdf,tbl}]
                       [--plot] [-s SUBSAMPLE_SIZE] [-i ITERATIONS] [-t STEP_THRESHOLD] [-f {amber,openmm}]

Generate a table report for dynamic undocking output. For a multi-ligand report, use the pattern flag with
wildcards to the directories.

optional arguments:
  -h, --help            show this help message and exit
  -p PATTERN, --pattern PATTERN
                        Wildcard pattern to find folders with DUck data.
  -d {min,single,avg,jarzynski,all}, --data {min,single,avg,jarzynski,all}
                        Mode to compile the report [min | single | avg | jarzynski | all]
  -o OUTPUT, --output OUTPUT
                        Output file; default is printing report to stdout.
  -of {csv,sdf,tbl}, --output-format {csv,sdf,tbl}
                        Output format, [csv | sdf | tbl].
  --plot                Plot work or energy values to file.
  -s SUBSAMPLE_SIZE, --subsample-size SUBSAMPLE_SIZE
                        Subsample size for Jarzynski bootstrapping.
  -i ITERATIONS, --iterations ITERATIONS
                        Number of bootstrapping iterations for Jarzynski analysis.
  -t STEP_THRESHOLD, --step-threshold STEP_THRESHOLD
                        Step threshold to find the minima. Only needed with custom executions of DUck. Default =
                        2500 steps
  -f {amber,openmm}, --format {amber,openmm}
                        Engine used to generate results. Default = amber
```

#### Report example

Parsing $W_{QB}$ in a tabulated output format from an example from the benchmark dataset.
```{bash}
$ cd Iridium/1a28/ARG766_HN2
$ openduck report -p '*/'

System  WQB
SMIRNOFF/       5.6297999999999995
GAFF2/  5.74429
SMIRNOFF_nohmr/ 6.922040000000001
GAFF2_nohmr/    6.264200000000001
GAFF2_bigbox/   5.180540000000001
GAFF2_tip4pew/  7.60925
GAFF2_spce/     6.17347
```

Calculating and reporting the $\Delta_{QB}$ of the same simulations as before:
```{bash}
$ openduck report -p '*/' -d jarzynski

System  Jarzynski       Jarzynski_SD    Jarzynski_SEM
SMIRNOFF/       6.391165243930965       0.12701922010484387     0.02033935321315065
GAFF2/  6.990007044361647       0.23871054814271678     0.038224279367893604
SMIRNOFF_nohmr/ 7.690209327251983       0.10126872662893496     0.01621597423328342
GAFF2_nohmr/    7.075924262571702   0.10628038471267455 0.01701848178973498
GAFF2_bigbox/   6.670905906961162   0.3167545344338863  0.05072131880828811
GAFF2_tip4pew/  9.007291835327141   0.33968295997031067 0.05439280525909316
GAFF2_spce/ 7.31123937991228    0.24521160219077848 0.03926528115039682

```
