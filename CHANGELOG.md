# Changelog

## 0.3.0 (unreleased)

### OpenMM production engine (rewritten)

* New `duck.steps.production` module with `DuckSimulation` (one OpenMM Context for all MD and SMD runs of a
  DUck production) and `run_duck_cycles` (the MD/SMD/WQB loop, previously `duck_smd_runs` in the CLI).
  `perform_md` and `run_steered_md` keep their signatures as thin wrappers.
* **Steering protocol identical to AMBER's `jar=1`** (`dist_duck.rst`): the restraint `E = k (r - r0)^2`
  (k = 50 kcal/mol/Å², no factor ½) acts between the two H-bond atoms, and r0 advances every 0.1 ps
  (5000 increments over 2.5 Å, the resolution of AMBER's DUMPAVE). The previous implementation pulled only the
  ligand atom towards an anchor fixed at the protein atom position, updated every 400 fs, with `0.5·k`, i.e. a
  spring half as stiff as AMBER's.
* The work is exact for the stepwise protocol (`W += E(r; r0_new) - E(r; r0_old)`) and is computed from the
  analytic energy derivative of the steering force group, so no coordinates are copied from the GPU during the SMD.
* **Restraints as in AMBER**: chunk heavy atoms restrained with 1.0 kcal/mol/Å² during MD and SMD to the
  equilibrated coordinates (AMBER `-ref 3_eq.rst`; was the pre-minimisation coordinates). Configurable with
  `--force-constant-chunk` and `--smd-spring-constant`. The previous MD/SMD code nominally used 0.1, but its
  restraint parameter had the same name (`k`) as the equilibration one, so loading `equil.chk` replaced it with the
  equilibration force constant (1.0 by default): in practice the old runs already used 1.0.
* Checkpoints only provide the dynamical state: the global parameters they store are reset to the values of the
  current simulation (previously they silently overrode the requested force constants).
* **Speed**: mixed precision by default (was `CudaPrecision=double`, 1/64-rate FP64 on consumer GPUs), no
  per-cycle System/Context re-creation, no second Context per SMD. Measured on an RTX 3070 for an 18.5k-atom
  system: 22 ns/day → 430–450 ns/day (2 fs) / 740–820 ns/day (4 fs, HMR) for both MD and SMD; a 500 ps SMD
  takes 96 s (52 s with HMR) instead of ~32 min. Most of the gain comes from the precision: the original steering
  loop in mixed precision reaches ~85% of the new SMD speed.
* Validation (progesterone receptor ARG766 H-bond, 9.8k atoms, all runs from the same AMBER-equilibrated state):
  ΔG_QB = 7.21 ± 0.17 kcal/mol with AMBER pmemd.cuda (HMR, 42 SMDs), 7.25 ± 0.31 with the new OpenMM engine (HMR,
  40 SMDs) and 7.22 ± 0.14 without HMR (20 SMDs); mean replica WQB within 0.15 kcal/mol of AMBER (Welch p ≈ 0.5).
  The previous OpenMM protocol gave 8.32 ± 0.14 kcal/mol (WQB +1.2 kcal/mol at 300 K, p < 0.001).
* SMD speed defined in Å/ps: with HMR the SMD now lasts 500 ps as in AMBER (it used to last 1 ns at 300 K,
  i.e. half the steering speed, because the speed was defined per step).
* SMD output (`smd_{i}_{T}.dat`) in the AMBER DUMPAVE layout (`r0 r force work`, Å, kcal/mol/Å, kcal/mol).
* Branching from the MD end state uses `Context.setState`, so each SMD and the next MD get independent random
  numbers (checkpoint reloading restored the same RNG state for all branches).
* Robust restarts: finished MD/SMD runs are reused (`openmm-from-equilibrated`, or `--resume` for
  `openmm-prepare`/`openmm-full-protocol`, which then also skip the finished preparation and equilibration);
  checkpoints written with another precision/platform (e.g. old double-precision ones) are read transparently.
  Relaunching `openmm-full-protocol` without `--resume` over an existing `duck_runs/` is refused instead of mixing a
  new equilibration with the old MD/SMD results. In batch mode `--resume` keeps every folder with a prepared system
  or results (AMBER `DUCK_*`, `duck_runs/`, OpenDUck ≤ 0.2 `cs.pickle`/`eql.chk`).
* New options: `--precision`, `--integrator` (`langevin`, the default, or `langevin-middle`),
  `--deterministic-forces`, `--trajectory-interval` (0 disables DCD trajectories), `--HMR` for
  `openmm-from-equilibrated` and `openmm-from-amber`.
* The mean temperature of every run is checked against the thermostat and a warning is printed if it deviates.

### Bug fixes

* Na⁺/Cl⁻ ions were position-restrained as "chunk heavy atoms" in all OpenMM phases (the exclusion list used
  AMBER ion names `Na+`/`Cl-`, but the residues are named `NA`/`CL`).
* With HMR the 325 K SMD was run without HMR settings and the 300 K one at half speed.
* NPT equilibration used a 2 fs time step with HMR (so it lasted 50 ps); `totalSteps` of the equilibration
  reporters was hard-coded.
* The H-bond restraint of MD/equilibration did not use periodic boundary conditions.
* `openmm-full-protocol` without `--batch` crashed (it changed to `<cwd>//<cwd>`).
* Batch OpenMM preparation with `--do-chunk` looked for the interaction in the tleap-renumbered chunk; in
  `--resume` mode skipped folders were used as `"<folder> skipped."`; one failing ligand aborted the batch.
* YAML inputs: `do_equilibrate: False` was read as the string `'False'` (truthy); `wqb_threshold` was truncated
  to an integer in `amber-prepare`; yaml values overrode explicit command-line options; `gpu_id` types differed
  between subcommands. All yaml values are now converted and validated with the argparse types and the command
  line takes precedence.
* `report`: the step threshold correction for OpenMM reports was a no-op (`==`); the WQB of a profile whose
  minimum is its first point was reported as 0; temperatures were classified with `'325' in path`; the library
  called `exit()` on malformed files (see `duck/utils/analysis_and_report.py` for the full list).
* `amber-prepare --batch -q local` crashed at the end (no local array template); one local script is now written
  per ligand folder.
* AMBER restraint masks: the residue range compaction overshot ranges after a gap (`[1,2,3,5,6]` → `1-4,5,6`).
* Lower-case force field names (`-f gaff2`, as in the examples) were rejected on the command line.
* Chunking: phosphate buffers (`PO4`) were not removed (typo `P04`); file names with dots or a `.mol` substring
  in the path were mangled.
* `parametrize.prepare_system` defaulted to a non-existent `tip3p.xml.prmtop` for unknown water models.
* `redefine_HB_pair.py` could not run (undefined attribute and globals); its `--exclude` and
  `--ligand-hb-elements` options were ignored.
* `getWqbValues.py` (AMBER jobs) printed text instead of a number when a minimum was not found, breaking the
  early-stop check. Such replicas now count as WQB = 0 (as in the OpenMM route), so they stop the job when a
  threshold is set.
* `report -of sdf` failed on cleaned-up runs because the default clean-up deleted `ligand.pdb`; it is now kept.
* `-d single` reports took the replica columns from the first system only.

### Analysis

* Jarzynski/fluctuation-dissipation analysis vectorised (identical results, ~12x faster without
  multiprocessing), log-sum-exp for numerical stability, `--seed` for reproducible bootstrapping.
* SMD reports in the AMBER and both OpenMM layouts are detected automatically, also in `report -f auto` (default).

### Code clean-up

* `duck.utils.compat` supports both `openmm` (≥ 7.6) and `simtk.openmm` (7.5) imports; `pkg_resources` replaced.
* Ligand parametrization functions merged into `generateFFStructureRDK_from_template`.
* The AMBER chunk restraint mask also excludes `HOH` residues, as the OpenMM restraints do (only relevant if the
  receptor file itself carries waters parametrized through a custom force field).
* Removed dead code: `duck/steps/prepare.py`, `duck/steps/prepare_mmgbsa.py` (could not run),
  `generateSMIRNOFFStructureRDK_old`, `getCAAtomsInSystem`, `getAtomSerialFromAmberMaskHbond`, unused chunking
  helpers.
* `setup.py` metadata (Python 3 only, dependencies, single-sourced version), conda recipe entry points,
  `test.sh` rewritten as a working smoke test, pytest suite in `tests/`.
