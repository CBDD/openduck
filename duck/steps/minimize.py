import subprocess
from shutil import which

def run_minimization():
    # default to pmemd; if it doesn't exist fall back to sander, which should be installed as a dependency of AmberTools
    engine = which('pmemd') or which('sander')
    if not engine:
        raise Exception("Neither pmemd nor sander are installed. AmberTools should be installed as an OpenDuCK dependency; check this is the case.")
    print(f'running minimization with {engine}...')
    cmd_line = [engine, "-O", "-i", "1_min.in", "-o", "min.out", "-p", "system_complex.prmtop", "-c", "system_complex.inpcrd", "-r", "min.rst", "-ref", "system_complex.inpcrd"]
    process = subprocess.run(cmd_line, capture_output=True)
    if process.returncode != 0:
        raise Exception(f"{engine} run failed with the following command: {' '.join(cmd_line)}, the following stdout: {process.stdout}, the following stderr: {process.stderr}, and the following exit code {process.returncode}",)

