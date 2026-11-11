import re
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(path.join(here, "duck", "__init__.py"), encoding="utf-8") as f:
    version = re.search(r'__version__\s*=\s*"([^"]+)"', f.read()).group(1)

setup(
    name="openduck",
    version=version,
    description="Open-source implementation of dynamic undocking (DUck)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CBDD/openduck.git",
    author="Alvaro Serrano, Mihaela Sminova, Simon Bray, Anthony Bradley",
    author_email="alvaro.serrano.morras@gmail.com",
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    keywords="dynamic undocking, steered molecular dynamics, OpenMM, AMBER, drug discovery",
    python_requires=">=3.9",
    packages=find_packages(exclude=["contrib", "docs", "tests", "test"]),
    # OpenMM, AmberTools, parmed, RDKit, PDBFixer, openff-toolkit and openmmforcefields are installed with conda
    # (see environment.yaml and environment-py39.yaml).
    install_requires=["numpy", "pandas", "scipy", "matplotlib", "pyyaml"],
    extras_require={"test": ["pytest"]},
    package_data={
        "duck": ["parameters/*.offxml", "parameters/waters/*", "parameters/tleap/*",
                 "templates/amber_inputs/*", "templates/queueing_templates/*", "templates/yaml_templates/*"]
    },
    entry_points={
        "console_scripts": [
            "openduck=scripts.openduck:main",
            "openduck-crooks=scripts.openduck_crooks:main",
            "openmm_job_launcher=scripts.openmm_job_launcher:main",
        ]
    },
)
