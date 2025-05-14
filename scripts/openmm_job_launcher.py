#!/usr/bin/env python3
"""
OpenDUck OpenMM SLURM Job Launcher
----------------------------------
Automates the setup and submission of OpenDUck docking simulations using the OpenMM backend on SLURM clusters.

Created on Thu Apr 10 11:35:19 2025
Author: Jochem Nelen (jnelen@ucam.edu)

This script prepares SLURM jobs for OpenDUck docking runs with OpenMM.
It batches ligand molecules, generates input SDF and YAML configuration files,
and constructs job submission scripts with customizable SLURM parameters.
"""

import argparse
import glob
import logging
import shutil
import sys
import datetime
import yaml
import os
import subprocess

from pathlib import Path
from typing import Tuple, List, Dict, Optional

from rdkit import Chem

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the SLURM job generator.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="An automatic SLURM launcher for OpenDUck calculations using the OpenMM backend.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "-l",
        "--ligand",
        type=Path,
        required=True,
        help="Path to ligand (SDF file or directory).",
    )
    parser.add_argument(
        "-p",
        "--protein",
        type=Path,
        required=True,
        help="Path to the receptor protein .pdb file.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Output base directory."
    )

    # Job settings
    job_group = parser.add_argument_group("Job Settings")
    job_group.add_argument(
        "--time",
        "-t",
        "-tj",
        default="",
        help="Max runtime (e.g., 01:00:00)",
    )
    job_group.add_argument(
        "--queue",
        "-qu",
        type=str,
        default="",
        help="Queue/partition to submit to.",
    )
    job_group.add_argument(
        "--mem",
        "-m",
        default="4G",
        help="Memory per job (default: 4G).",
    )
    job_group.add_argument(
        "--gpu",
        "-gpu",
        "--GPU",
        action="store_true",
        default=False,
        help="Use GPU.",
    )
    job_group.add_argument(
        "--cores",
        "-c",
        type=int,
        default=None,
        help="Cores per job (default: 1 w/ GPU, else 8).",
    )
    job_group.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="Number of jobs.",
    )
    job_group.add_argument(
        "--singularity",
        nargs="?",
        const="AUTO",
        default=None,
        metavar="IMG_PATH",
        help=(
            "Optional Singularity support:\n"
            "  * omit the flag            : auto-detect runtime\n"
            "  * --singularity            : use default openDUck.sif if present\n"
            "  * --singularity IMG        : use the specified Singularity image"
        ),
    )

    # OpenDUck protocol settings
    duck_group = parser.add_argument_group("OpenDUck Protocol Settings")
    duck_group.add_argument(
        "--interaction",
        type=str,
        required=True,
        help="Interaction atom (e.g., _LEU_31_N).",
    )
    duck_group.add_argument(
        "--do-chunk",
        action="store_true",
        help="Enable receptor chunking.",
    )
    duck_group.add_argument(
        "--cutoff",
        type=float,
        default=9.0,
        help="Chunking cutoff distance (Angstroms). Default: 9.0A",
    )
    duck_group.add_argument(
        "--ionic-strength",
        type=float,
        default=0.10,
        help="Ionic strength (concentration) of the counter ion salts (Na+/Cl-). Default = 0.1 M",
    )
    duck_group.add_argument(
        "--ignore-buffers",
        action="store_true",
        help="Do not remove buffers (solvent, ions etc.)",
    )
    duck_group.add_argument(
        "--small-molecule-forcefield",
        type=str,
        choices=["gaff2", "smirnoff"],
        default="gaff2",
        help="Small molecule forcefield (default: gaff2)",
    )
    duck_group.add_argument(
        "--protein-forcefield",
        type=str,
        choices=["amber99sb", "amber14-all"],
        default="amber14-all",
        help="Protein forcefield (default: amber14-all)",
    )
    duck_group.add_argument(
        "--water-model",
        type=str,
        choices=["tip3p", "spce"],
        default="tip3p",
        help="Water model (default: tip3p)",
    )
    duck_group.add_argument(
        "--hmr",
        "--HMR",
        dest="HMR",
        action="store_true",
        help="Enable hydrogen mass repartitioning (HMR).",
    )
    duck_group.add_argument(
        "--smd-cycles",
        type=int,
        default=10,
        help="Number of SMD cycles (default: 10)",
    )
    duck_group.add_argument(
        "--wqb-threshold",
        type=float,
        default=None,
        help="WQB threshold (default: None, no threshold)",
    )
    duck_group.add_argument(
        "--fix-ligand",
        action="store_true",
        help="Apply simple fixes for the ligand: assign correct charges on tetravalent nitrogens and add missing hydrogen atoms.",
    )

    return parser.parse_args()


def check_singularity_available() -> None:
    """Exit if singularity command is not available."""
    if not shutil.which("singularity"):
        logging.error(
            "Singularity executable not found in PATH.\n"
            "Please install Singularity or install openDUck and run without Singularity."
        )
        sys.exit(1)


def resolve_singularity_image(user_arg: Optional[str]) -> Optional[Path]:
    """
    Decide which Singularity image (if any) to use.

    Args:
        user_arg: The --singularity argument from the CLI, or None.

    Returns:
        - Path to the Singularity image if using Singularity.
        - None if running without Singularity.

    Exits early if the environment is invalid.
    """

    # Default expected singularity image
    default_img = Path("openDUck.sif").resolve()

    # --singularity flag was used
    if user_arg:
        check_singularity_available()

        if user_arg == "AUTO":
            if default_img.exists():
                logging.info("Using default Singularity image: %s", default_img)
                return default_img
            logging.error(
                "Requested Singularity, but default image not found at: %s", default_img
            )
            sys.exit(1)
        else:
            img = Path(user_arg).expanduser().resolve()
            if img.exists():
                logging.info("Using user-specified Singularity image: %s", img)
                return img
            logging.error(
                "Singularity image %s not found, please provide a correct path.", img
            )
            sys.exit(1)

    if shutil.which("openduck"):
        singularity_container = os.environ.get("SINGULARITY_CONTAINER")
        if singularity_container:
            logging.info(
                "Detected running inside Singularity container: %s",
                singularity_container,
            )
            return Path(singularity_container)
        logging.info("Native OpenDUck detected. Running without Singularity.")
        return None

    if default_img.exists():
        # Check if singularity is installed
        check_singularity_available()

        logging.info(
            "Native OpenDUck not found. Falling back to Singularity image: %s",
            default_img,
        )
        return default_img

    logging.error(
        "Cannot find native OpenDUck command or Singularity image.\n"
        "Please install OpenDUck or place openDUck.sif in the launch directory."
    )
    sys.exit(1)


def process_sdf(ligand_path: Path) -> Tuple[Dict[str, Chem.Mol], int]:
    """
    Parse molecules from a single SDF file.

    Args:
        ligand_path (Path): Path to the SDF file.

    Returns:
        Tuple[Dict[str, Chem.Mol], int]: A dictionary mapping molecule names to RDKit molecule objects,
            and a count of invalid molecules.
    """
    mol_dict: Dict[str, Chem.Mol] = {}
    invalid_mols: int = 0
    suppl = Chem.SDMolSupplier(str(ligand_path), removeHs=False)

    for mol in suppl:
        if mol is None:
            invalid_mols += 1
            continue

        if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
            mol_name = mol.GetProp("_Name")
        else:
            smiles = Chem.MolToSmiles(mol)
            mol_name = f"mol_{smiles}"
            mol.SetProp("_Name", mol_name)

        mol_dict[mol_name] = mol

    return mol_dict, invalid_mols


def build_slurm_flag(flag_name: str, value: str, prefix: str = "--") -> str:
    """
    Return a formatted SLURM flag string if a value is provided, else returns an empty string.

    Args:
        flag_name (str): The flag's name.
        value (str): The flag's value.
        prefix (str): Prefix for the flag (default is '--').

    Returns:
        str: Formatted flag string or empty if value is empty.
    """
    return f"{prefix}{flag_name} {value}" if value else ""


def prepare_output_directory(base_output_path: Path) -> Path:
    """
    Prepare the output directory by appending a timestamp. If the directory already exists,
    prompt the user to remove it.

    Args:
        base_output_path (Path): The base output directory provided by the user.

    Returns:
        Path: The prepared output directory path.
    """
    date_str = datetime.datetime.now().strftime("%Y_%m_%d")
    output_dir = base_output_path.parent / f"{base_output_path.name}_{date_str}"

    if output_dir.exists():
        logging.info("The directory '%s' already exists.", output_dir)
        answer = input("Do you want to remove it? (y/n): ").strip().lower()
        while answer not in {"y", "n", "yes", "no"}:
            answer = input("Please enter y(es) or n(o): ").strip().lower()
        if answer in {"y", "yes"}:
            shutil.rmtree(output_dir)
            logging.info("Removed existing directory: %s", output_dir)
        else:
            logging.error("Operation cancelled by user.")
            sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Created output directory: %s", output_dir)
    return output_dir


def batch_mols(molecules: List[str], num_jobs: int) -> List[List[str]]:
    """
    Split molecule names into `num_jobs` balanced batches.

    Args:
        molecules (List[str]): List of molecule identifiers.
        num_jobs (int): Number of batches/jobs desired.

    Returns:
        List[List[str]]: List of molecule batches.
    """
    total_mols = len(molecules)
    if num_jobs > total_mols:
        logging.info("More jobs than molecules. Launching one job per molecule.")
        return [[mol] for mol in molecules]

    k, m = divmod(total_mols, num_jobs)
    return [
        molecules[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
        for i in range(num_jobs)
    ]


def write_yaml(
    yaml_path: Path,
    args: argparse.Namespace,
    protein_target_path: str,
    ligand_path: str,
) -> None:
    """
    Write a YAML configuration file for OpenDUck.

    Args:
        yaml_path (Path): Full path where the YAML file is to be written.
        args (argparse.Namespace): Parsed command-line arguments.
        protein_target_path (str): Relative path to the protein file from the job directory.
        ligand_path (str): Relative path to the ligand file from the job directory.
    """
    config: Dict[str, object] = {
        "interaction": args.interaction,
        "receptor_pdb": protein_target_path,
        "ligand_mol": ligand_path,
        "threads": args.cores,
        "small_molecule_forcefield": args.small_molecule_forcefield,
        "protein_forcefield": args.protein_forcefield,
        "water_model": args.water_model,
        "HMR": args.HMR,
        "smd_cycles": args.smd_cycles,
        "batch": args.batch,
    }

    if args.wqb_threshold is not None:
        config["wqb_threshold"] = args.wqb_threshold
    if args.ionic_strength:
        config["ionic_strength"] = args.ionic_strength
    if args.gpu:
        config["gpu_id"] = 0
    if args.fix_ligand:
        config["fix_ligand"] = True
    if args.do_chunk:
        config["do_chunk"] = True
        config["cutoff"] = args.cutoff

    with open(yaml_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)


def main() -> None:
    """
    Main function that manages the SLURM job generation process.
    """
    args = parse_arguments()

    # Identify (potential) Singularity Path
    singularity_path: Optional[Path] = resolve_singularity_image(args.singularity)

    # Set default cores if not explicitly provided.
    if args.cores is None:
        args.cores = 1 if args.gpu else 8

    queue_arg = build_slurm_flag("partition", args.queue)
    time_arg = build_slurm_flag("time", args.time)

    # Validate input file/directory
    if not args.ligand.exists():
        raise FileNotFoundError(f"Ligand path does not exist: {args.ligand}")
    if not args.protein.exists():
        raise FileNotFoundError(f"Protein file does not exist: {args.protein}")
    if not args.protein.is_file() or args.protein.suffix.lower() != ".pdb":
        raise ValueError(f"Protein file must be a .pdb file: {args.protein}")

    # Process ligand molecules from file or directory
    mol_dict: Dict[str, Chem.Mol] = {}
    total_invalid_mols: int = 0
    if args.ligand.is_file():
        mol_dict, total_invalid_mols = process_sdf(args.ligand)
    elif args.ligand.is_dir():
        for ligand_file in glob.glob(str(args.ligand / "*.sdf")):
            single_dict, invalid_count = process_sdf(Path(ligand_file))
            mol_dict.update(single_dict)
            total_invalid_mols += invalid_count
    else:
        raise ValueError("Ligand path must be a file or directory.")

    logging.info(
        "Identified %d valid compounds, %d invalid", len(mol_dict), total_invalid_mols
    )

    # Prepare the output directory and copy the protein file there
    output_dir: Path = prepare_output_directory(args.output)
    protein_target_path: str = shutil.copy(str(args.protein), str(output_dir))

    # Ask for the number of jobs if not specified before
    if args.jobs is None:
        answer = input("Enter the number of jobs to run: ").strip()
        if not answer:
            logging.error("No input provided. Operation cancelled by user.")
            sys.exit(1)
        try:
            args.jobs = int(answer)
        except ValueError as e:
            logging.error("Invalid input: %s. Operation cancelled.", e)
            sys.exit(1)

    # Decide on job batching
    if args.jobs <= 0 or args.jobs >= len(mol_dict):
        args.jobs = len(mol_dict)
        logging.info("More jobs than Compounds, launching one job per compound...")
        args.batch = False
    else:
        args.batch = True

    batches: List[List[str]] = batch_mols(list(mol_dict.keys()), args.jobs)

    # Generate job directories and scripts
    for idx, mol_batch in enumerate(batches, start=1):
        if not args.batch:
            safe_mol_name = "".join(
                c if c.isalnum() or c in ("-", "_") else "_" for c in mol_batch[0]
            )
            folder_name = f"{idx}_{safe_mol_name}"
        else:
            folder_name = f"{idx}_batch_mols"

        job_dir: Path = output_dir / folder_name
        job_dir.mkdir(parents=True, exist_ok=True)

        relative_protein_path: str = os.path.relpath(protein_target_path, job_dir)

        # Write SDF file for ligand molecules in this batch
        sdf_name: str = f"input_mol_job_{idx}.sdf"
        sdf_path: Path = job_dir / sdf_name
        with Chem.SDWriter(str(sdf_path)) as writer:
            for mol_id in mol_batch:
                writer.write(mol_dict[mol_id])

        # Write YAML configuration file
        yaml_filename: str = f"input_protocol_job_{idx}.yaml"
        yaml_path: Path = job_dir / yaml_filename
        write_yaml(
            yaml_path=yaml_path,
            args=args,
            protein_target_path=str(relative_protein_path),
            ligand_path=sdf_name,
        )

        # Construct common SLURM arguments and job command
        common_slurm_args = (
            f"--mem {args.mem} "
            f"--output=job_{idx}_%j.out "
            f"--job-name=openDUck_{idx} "
            f"-c {args.cores} "
            f"{time_arg} {queue_arg} "
            f"{'--gres=gpu:1' if args.gpu else ''}"
        ).strip()

        if singularity_path:
            gpu_flag = "--nv " if args.gpu else ""
            run_command = (
                f"OPENMM_CPU_THREADS={args.cores} "
                f"singularity exec {gpu_flag}--bind $PWD {singularity_path} "
                f"openduck openmm-full-protocol -y {yaml_filename}"
            )
        else:
            run_command = (
                f"OPENMM_CPU_THREADS={args.cores} "
                f"openduck openmm-full-protocol -y {yaml_filename}"
            )

        job_cmd: str = (
            f'sbatch --chdir "{job_dir}" --wrap="{run_command}" {common_slurm_args}'
        )

        # Write the job script to file
        job_script_path: Path = job_dir / f"job_{idx}.sh"
        with open(job_script_path, "w") as jobfile:
            jobfile.write("#!/usr/bin/env bash\n")
            jobfile.write(job_cmd + "\n")

    launch_cmd = f'for f in {str(output_dir)}/*/job_*.sh; do sh "$f"; done'

    # Write final launcher of all jobs
    with open(f"{output_dir}/launch_jobs.sh", "w") as job_launch_file:
        job_launch_file.write("#!/usr/bin/env bash\n")
        job_launch_file.write(f"{launch_cmd}\n")

    # Check if sbatch command is available
    if shutil.which("sbatch") is not None:
        subprocess.run(launch_cmd, shell=True)
    else:
        logging.warning(
            f"The sbatch command doesn't seem to be available. Try to launch the jobs manually (without Singularity) by running:\nsh {output_dir}/launch_jobs.sh"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("An error occurred during job generation: %s", e)
        sys.exit(1)
