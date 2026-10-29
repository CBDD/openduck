[![Version](http://img.shields.io/badge/version-0.3.0-blue.svg?style=flat)](https://github.com/CBDD/openduck)
[![License](http://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat)](https://github.com/CBDD/openduck/LICENSE)

# Welcome to OpenDUck

OpenDUck is an open-source implementation of dynamic undocking (DUck), an application of steered molecular dynamics developed as a fast algorithm to calculate the work necessary to reach a quasi-bound state, at which a certain native contact is broken between the ligand and receptor. Due to its cheap and fast nature, DUck is mainly employed as a post-docking filter during virtual screening campaigns.

More details can be found in the original dynamic undocking publication:

> Ruiz-Carmona, S., Schmidtke, P., Luque, F. et al. Dynamic undocking and the quasi-bound state as tools for drug discovery. Nature Chem 9, 201–206 (2017). https://doi.org/10.1038/nchem.2660

OpenDUck can be used as a command line tool (`openduck`) or as a Python library (`duck`). It prepares the system (receptor chunking, ligand/receptor parameterization and solvation), runs the DUck protocol in OpenMM, or writes the inputs and job scripts to run it with AMBER, and analyses the results (WQB and Jarzynski ΔG<sub>QB</sub>).

## Installation

We recommend you install OpenDUck into a fresh Conda environment.
```bash
git clone git@github.com:CBDD/openduck.git
cd openduck
conda env create -f environment.yaml
conda activate openduck
pip install .            # or `pip install -e .` for development
```
Run the unit tests with `python -m pytest tests` and an end-to-end smoke test with `./test.sh DATA_DIR [GPU_ID]` (see the header of `test.sh` for the input files it needs).

### Singularity
A Singularity definition file (`openDUck_singularity.def`) is included in the repository to build a self-contained image:

```bash
singularity build openDUck.sif openDUck_singularity.def
singularity exec openDUck.sif openduck -h
```

## OpenDUck usage

```
$ openduck -h
usage: openduck [-h]  ...

Open-source toolkit for dynamic undocking.

Open-source dynamic undocking toolkit. Choose one of the following actions:
    openmm-prepare      Preparation of systems for OpenMM simulations.
    openmm-full-protocol
                        OpenDUck full OpenMM protocol.
    openmm-from-equilibrated
                        OpenDUck OpenMM protocol starting from a pre-equilibrated system.
    openmm-from-amber   OpenDUck OpenMM protocol starting from an AMBER topology and coordinates.
    amber-prepare       Preparation of systems, inputs and queue files for AMBER simulations.
    report              Generate a report for OpenDUck results.
    chunk               Chunk a protein for dynamic undocking.
    prepare-cofactor    Create the parameters file for a cofactor to input as custom forcefield.
```

Every subcommand prints its options with `openduck <subcommand> -h`.

### Input yaml files

Every subcommand accepts its options either on the command line or in a yaml file given with `-y`. The keys are the long option names with underscores (`--smd-cycles` → `smd_cycles`); `ligand_mol` and `receptor_pdb` are also accepted for `ligand` and `receptor`. Values are converted and validated like command-line values, unknown keys are reported, and **options given on the command line take precedence over the yaml file**, so a yaml file can be reused while changing a single option:

```bash
openduck openmm-full-protocol -y input.yaml --smd-cycles 5
```

Commented examples for each subcommand are in `duck/templates/yaml_templates`.

### Running OpenDUck with OpenMM

The OpenMM route is divided in two steps, the preparation with `openmmforcefields` and the production (MD/SMD) following the dynamic undocking protocol:

* `openmm-full-protocol`: executes the full OpenDUck protocol, from the protein-ligand files to the steered simulations.
* `openmm-prepare`: chunking (optional), preparation and (with `--do-equilibrate`) equilibration.
* `openmm-from-equilibrated`: production of MD/SMD runs from an equilibrated checkpoint (`equil.chk`) and its preparation pickle (`complex_system.pickle`).
* `openmm-from-amber`: equilibration and production starting from an AMBER topology and coordinates.

The full protocol executes the following steps:

    * Chunking (optional): residues within --cutoff of the interaction atom, capped and protonated with tleap
    * Preparation
      * Ligand parametrization (SMIRNOFF or GAFF2, AM1-BCC charges)
      * Receptor parametrization (amber99sb or amber14-all, plus an optional custom XML force field)
      * Solvation box (cubic, --solvent-buffer-distance) with structural waters, Na+/Cl- at --ionic-strength
    * Equilibration: minimisation, 100 ps NVT heating, 100 ps NPT
    * (MD + SMD@300K + SMD@325K + WQB check) x smd-cycles

#### The OpenMM DUck protocol

The OpenMM production mirrors the AMBER DUck inputs in `duck/templates/amber_inputs`:

| Phase | Settings |
| --- | --- |
| MD (`md_length`, 0.5 ns) | 300 K, Langevin (γ = 4 ps<sup>-1</sup>), PME with a 9 Å cutoff, H-bonds constrained; the key H-bond is kept formed by the flat-bottom restraint of `dist_md.rst` (k = 1/10 kcal mol<sup>-1</sup> Å<sup>-2</sup> below 2 Å / above 3 Å) |
| SMD at 300 K and 325 K | both from the end of the MD; the H-bond distance is steered from `init_distance` (2.5 Å) to `init_distance` + 2.5 Å at 5 Å/ns with E = k (r − r<sub>0</sub>)<sup>2</sup>, k = 50 kcal mol<sup>-1</sup> Å<sup>-2</sup>, as `dist_duck.rst` with `jar=1` |
| Restraints | chunk heavy atoms restrained to the equilibrated coordinates with 1 kcal mol<sup>-1</sup> Å<sup>-2</sup> (`restraint_wt`, `-ref 3_eq.rst`); solvent, ions and the ligand are free |
| Time step | 2 fs, or 4 fs with `--HMR` (same simulated times: the SMD always lasts 500 ps) |

Like AMBER's `jar=1`, the steering restraint acts between the two H-bond atoms, so there is no steering vector to update. Its reference distance r<sub>0</sub> advances every 0.1 ps (5000 increments of 0.0005 Å, the resolution of AMBER's `DUMPAVE`), and the work of each increment is computed exactly from the analytic derivative of the restraint energy, without copying coordinates from the GPU. The whole production (all MD and SMD runs of all cycles) runs in a single OpenMM Context.

Performance notes:

* `--precision mixed` (default) is the equivalent of AMBER's SPFP. `double` is 10–50 times slower on consumer GPUs.
* `--HMR` doubles the speed (4 fs time step); the system is prepared with repartitioned hydrogen masses.
* DCD trajectories are written every `--trajectory-interval` ps (4 ps by default, ~20 MB per SMD for a 15k-atom system); use `--trajectory-interval 0` in virtual screening.
* On an RTX 3070, 10–18k-atom systems run at 430–480 ns/day (2 fs) and 820–880 ns/day (4 fs, HMR) in both MD and SMD: a 500 ps SMD of an 18.5k-atom system takes 96 s (52 s with HMR), against ~32 min with OpenDUck ≤ 0.2.

Runs can be resumed: relaunching `openmm-from-equilibrated`, or `openmm-full-protocol`/`openmm-prepare` with `--resume`, reuses the finished preparation, equilibration, MD (`duck_runs/md_{i}.chk`) and SMD (`duck_runs/smd_{i}_{T}.pdb`) runs. Old checkpoints written in double precision are read transparently.

#### openmm-full-protocol

```bash
$ openduck openmm-full-protocol -l brd4_lig.mol -r 4LR6_aligned_chunk_nowat.pdb -i A_ASN_140_ND2 \
                                -g 0 -f GAFF2 -ff amber14-all -w tip3p -W 6 -n 20 -H
```

or, with the parameters in a yaml file, `openduck openmm-full-protocol -y input.yaml` with:
```yaml
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
HMR : True

# Production Arguments
smd_cycles : 20
md_length : 0.5
wqb_threshold : 6
init_distance : 2.5
trajectory_interval : 0
```

Main options (see `openduck openmm-full-protocol -h` for all of them):

| Option | Default | Description |
| --- | --- | --- |
| `-l/--ligand`, `-r/--receptor`, `-i/--interaction` | | ligand (MOL/SDF), receptor PDB and receptor atom of the H-bond as `chain_resname_resid_atomname` |
| `--do-chunk`, `-c/--cutoff` | off, 9 Å | reduce the receptor to the residues within the cutoff of the interaction atom |
| `-f`, `-ff`, `-w` | SMIRNOFF, amber99sb, tip3p | small molecule force field (SMIRNOFF, GAFF2), protein force field (amber99sb, amber14-all) and water model (tip3p, spce) |
| `-ion`, `-s`, `-water`, `-cf`, `-fl` | 0.1 M, 10 Å, `waters_to_retain.pdb` | ionic strength, box buffer, structural waters (used if the file exists), custom XML force field, simple ligand fixes |
| `-H/--HMR` | off | hydrogen mass repartitioning and 4 fs time step |
| `-n/--smd-cycles`, `-m/--md-length`, `-W/--wqb-threshold` | 20, 0.5 ns, none | MD/SMD cycles, MD length per cycle and early-stop WQB threshold |
| `-d/--init-distance`, `-v/--init-velocities` | 2.5 Å, 1e-5 Å/2 fs | initial SMD distance and steering speed (1e-5 Å per 2 fs = 5 Å/ns) |
| `--force-constant-chunk`, `--smd-spring-constant` | 1.0, 50 | chunk restraint and steering spring constant (kcal mol<sup>-1</sup> Å<sup>-2</sup>, AMBER convention) |
| `-g/--gpu-id`, `--precision`, `--integrator`, `--trajectory-interval` | CPU, mixed, langevin, 4 ps | execution options |
| `-B/--batch`, `-t/--threads`, `--resume`, `-i0`, `-p` | | multi-ligand SDF: each ligand is prepared in parallel in `{prefix}_{n}` and then simulated sequentially on the GPU; a failing ligand is reported and skipped |

#### Batch launching OpenMM SLURM jobs with `openmm_job_launcher`

For larger sets of ligands, `openmm_job_launcher` splits a multi-ligand SDF into batches, writes the input SDF and yaml files of each job and submits them to SLURM (optionally inside a Singularity container, on CPU or GPU):

```bash
openmm_job_launcher --ligand my_ligands.sdf --protein receptor.pdb --interaction A_ASN_140_ND2 --output jobs_output --jobs 4 --gpu --smd-cycles 5
openmm_job_launcher --ligand my_ligands.sdf --protein receptor.pdb --interaction A_ASN_140_ND2 --output jobs_output --jobs 10 --gpu --singularity openDUck.sif
```
If a native `openduck` installation is not found, the launcher uses `openDUck.sif` from the launch directory. Run `openmm_job_launcher --help` for all options.

#### openmm-prepare & openmm-from-equilibrated

The protocol can also be executed in two steps, the preparation and equilibration (`openmm-prepare --do-equilibrate`) and the production (`openmm-from-equilibrated`):

```yaml
# openmm-prepare input.yaml
ligand_mol : '1a28_lig.mol'
receptor_pdb : '1a28_prot.pdb'
interaction : 'A_ARG_766_NH2'
do_chunk : True
cutoff : 10
small_molecule_forcefield : 'GAFF2'
protein_forcefield : 'amber14-all'
water_model : 'spce'
ionic_strength : 0.1
solvent_buffer_distance : 12
do_equilibrate : True
gpu_id : 0
```

```yaml
# openmm-from-equilibrated input.yaml
equilibrated_system : equil.chk
pickle : complex_system.pickle
smd_cycles : 20
md_length : 0.5
wqb_threshold : 6
gpu_id : 0
```

#### openmm-from-amber

The protocol can be launched from an AMBER topology and coordinates (e.g. from `amber-prepare`). The receptor chunk (PDB or MOE mol2) is used to locate the interaction atom with the original residue numbering:

```bash
openduck openmm-from-amber -t system_complex.prmtop -c system_complex.inpcrd -r receptor_chunk.pdb -i A_ASN_140_ND2 -n 20 -W 6 -g 0
```

### Running OpenDUck in AMBER

Dynamic undocking was originally designed to run in AMBER; OpenDUck keeps the same protocol, replacing the parametrization and file generation previously done with MOE. `amber-prepare` shares the preparation options of the OpenMM route and writes the pmemd inputs (`1_min.in`, `2_heating*.in`, `3_eq.in`, `md.in`, `duck.in`, `duck_325K.in`, `dist_md.rst`, `dist_duck.rst`), the topology (`system_complex.prmtop`, `HMR_system_complex.prmtop` with `-H`) and a job script built from a queue template (`-q local|Slurm|SGE`).

```bash
openduck amber-prepare -l 1a28_lig.mol -r 1a28_prot.pdb -i A_ARG_766_NH2 --do-chunk -c 10 \
                       -f GAFF2 -ff amber99sb -w spce -ion 0.05 -s 11 -H -n 4 -W 4 -q Slurm
```

Typically, dynamic undocking is employed as a post-docking filter in a virtual screening campaign. A batch preparation is executed with `--batch` and a multi-ligand SD-file; each ligand is prepared in the `LIG_target_{n}` subfolder, where `n` is the ligand's position in the input SDF, and an array job script is written (with the `local` template, one script is written in every ligand folder).

```bash
openduck amber-prepare -l brd4_ligands.sdf -r 4LR6_aligned_chunk_nowat.pdb -i A_ASN_140_ND2 \
                       --waters-to-retain waters_to_retain.pdb -f gaff2 -ff amber14-all -w spce -ion 0.1 -s 11 \
                       -H --smd-cycles 10 --wqb-threshold 6 -q SGE --seed -1 --batch --threads 8
```

The queueing templates are stored in `duck/templates/queueing_templates`. To customize a template, add a file named `{queue_name}.q` (single job) or `{queue_name}_array.q` (array job) with the following format:
```
#!/bin/bash
#HEADER

{functions}

{commands}
```

### Analysis

The `report` subcommand compiles the results of AMBER (`DUCK_*/duck.dat`) and OpenMM (`duck_runs/smd_*.dat`) runs; the layout is detected automatically (`-f auto`). By default the lowest W<sub>QB</sub> over the replicas is reported. A more formally appropriate descriptor is the ΔG<sub>QB</sub> obtained with the Jarzynski equality (`-d jarzynski`), whose uncertainty is estimated by bootstrapping the replicas (`-s` replicas per sample, `-i` samples, `--seed` for reproducible numbers).

```
usage: openduck report [-h] [-p PATTERN] [-d {min,single,avg,jarzynski,all}] [-o OUTPUT]
                          [-of {csv,sdf,tbl}] [--plot] [-s SUBSAMPLE_SIZE] [-i ITERATIONS]
                          [-t STEP_THRESHOLD] [-f {auto,amber,openmm}] [--seed SEED]
```

When analyzing a single result, the default pattern is the current directory (`.`). To analyze multiple directories, give a wildcard pattern:

For example, the same progesterone receptor H-bond (1a28, ARG766 NH2) simulated from the same equilibrated system with AMBER (`amber/`, 21 cycles) and with the OpenMM engine (`openmm/`, 20 cycles), both with HMR:
```
$ openduck report -p '*/' -d all --seed 1
System   WQB    Average  SD     Jarzynski  Jarzynski_SD  Jarzynski_SEM  ...
amber/   6.381  7.505    0.596  7.207      0.169         0.027
openmm/  6.059  7.635    0.575  7.253      0.308         0.049
```

SMD reports use the AMBER `jar=1` layout in both engines: one row per 0.1 ps with the target distance r<sub>0</sub> (Å), the distance r (Å), the force −2k(r − r<sub>0</sub>) (kcal mol<sup>-1</sup> Å<sup>-1</sup>) and the accumulated work (kcal/mol). Reports written by OpenDUck ≤ 0.2 (10 columns with units, 1250 rows) are still read.

### Benchmarking OpenMM against AMBER

Two scripts in `benchmarks/` re-run AMBER DUck validation sets (folders `<dataset>/<pdb>/<interaction>/<condition>` listed in `<dataset>/systems.txt`) with the OpenMM engine on exactly the same prepared systems (same topology, coordinates and steered atom pair) and compare both engines. Every OpenMM production also writes `duck_runs/timings.csv` with the wall time, speed and GPU of each MD and SMD run.

```bash
# run (resumable); results in OUT/<dataset>/<pdb>/<interaction>/GAFF2_openMM/ with DUCK_<n>/duck.dat as in AMBER
python benchmarks/run_openmm_validation.py run --data-root openduck_validation --out-root OUT --gpu-id 0
python benchmarks/run_openmm_validation.py jobs --data-root openduck_validation --out-root OUT   # SGE (A40) / Slurm arrays
# compare execution times and the WQB / Jarzynski dG_QB regression
python benchmarks/compare_openmm_amber.py --openmm-root OUT --data-root openduck_validation --jobs 8
```

### Using OpenDUck as a Python library

```python
from duck.steps.production import DuckSimulation, run_duck_cycles
from duck.utils.duck_stuff import load_complex_pickle

# full production from an equilibrated system
run_duck_cycles('equil.chk', 'complex_system.pickle', n_cycles=20, md_length=0.5, gpu_id=0, hmr=True, wqb_threshold=6)

# or drive single phases
structure, key_interaction = load_complex_pickle('complex_system.pickle')
sim = DuckSimulation(structure, key_interaction, gpu_id=0, hmr=True)
sim.load_checkpoint('equil.chk')
sim.run_md(0.5, csv_file='md.csv')
work_profile = sim.run_smd(300, 'smd_300.dat', start_dist=2.5, speed=0.005)  # numpy array (r0, r, force, work)
```

The analysis functions (`duck.utils.analysis_and_report`), the restraint builders (`duck.utils.duck_stuff`), the preparation (`duck.steps.parametrize`, `duck.steps.chunk`) and the AMBER templates (`duck.utils.amber_inputs`) can be used in the same way to build new protocols.

### Changes in 0.3

See [CHANGELOG.md](CHANGELOG.md). The OpenMM production was rewritten to reproduce the AMBER protocol and to run ~20 times faster; **OpenMM results are therefore not directly comparable with those of OpenDUck ≤ 0.2**, which used a spring half as stiff as AMBER's, a steering anchor updated every 400 fs that pulled only the ligand atom, restrained Na<sup>+</sup>/Cl<sup>-</sup> ions, chunk restraints referenced to the pre-minimisation coordinates and, with HMR, half the steering speed at 300 K.
