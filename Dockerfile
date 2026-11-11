# OpenDUck with the conda environment of environment.yaml. Build from the repository root:
#     docker build -t openduck .
#     docker run --rm --gpus all -v "$PWD":/work -w /work openduck openduck openmm-full-protocol -y input.yaml
# Build arguments (--build-arg NAME=VALUE):
#     ENV_FILE  environment.yaml (newest tested versions, CUDA 12.9) or environment-py39.yaml (published
#               validation, CUDA 11.7). The GPU driver must support that CUDA version ("CUDA Version" in nvidia-smi).
FROM ubuntu:24.04
LABEL maintainer="Anthony Bradley <anthony.richard.bradley@gmail.com>"
ARG ENV_FILE=environment.yaml

RUN apt-get update -qq \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends wget ca-certificates bzip2 \
 && rm -rf /var/lib/apt/lists/*

# Miniforge: conda-forge only
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
 && bash /tmp/miniforge.sh -b -p /opt/conda \
 && rm /tmp/miniforge.sh

COPY environment.yaml environment-py39.yaml /opt/openduck/
RUN /opt/conda/bin/conda env create -q -n openduck -f /opt/openduck/$ENV_FILE \
 && /opt/conda/bin/conda clean -a -y -q

COPY setup.py README.md LICENSE /opt/openduck/
COPY duck /opt/openduck/duck
COPY scripts /opt/openduck/scripts
COPY tests /opt/openduck/tests
RUN /opt/conda/envs/openduck/bin/python -m pip install -q --no-deps /opt/openduck

ENV PATH=/opt/conda/envs/openduck/bin:/opt/conda/bin:$PATH \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8
CMD ["openduck", "-h"]
