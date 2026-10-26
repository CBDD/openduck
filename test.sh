#!/bin/bash
# OpenDUck integration smoke test.
#
# Usage: ./test.sh DATA_DIR [GPU_ID]
#
# DATA_DIR must contain 1a28_prot.pdb, 1a28_lig.mol, 4LR6_aligned_chunk_nowat.pdb, brd4_lig.mol,
# brd4_ligands.sdf and waters_to_retain.pdb. Without GPU_ID the OpenMM steps run on the CPU (slow).
# Every step runs in its own folder under ./openduck_smoke_test and logs to run.log.
# Fast unit tests: python -m pytest tests
set -u
DATA=$(realpath "${1:?usage: ./test.sh DATA_DIR [GPU_ID]}")
GPU=${2:-}
GPU_FLAG=${GPU:+-g $GPU}
OUT=$(pwd)/openduck_smoke_test
mkdir -p "$OUT"
status=0

step() {
    local name=$1; shift
    mkdir -p "$OUT/$name"
    (cd "$OUT/$name" && "$@" > run.log 2>&1)
    local code=$?
    if [ $code -eq 0 ]; then echo "PASS $name"; else echo "FAIL $name (see $OUT/$name/run.log)"; status=1; fi
}

step chunk openduck chunk -r "$DATA/1a28_prot.pdb" -i A_ARG_766_NH2 -c 9
step amber-prepare openduck amber-prepare -l "$DATA/1a28_lig.mol" -r "$DATA/1a28_prot.pdb" -i A_ARG_766_NH2 \
    --do-chunk -c 9 -f gaff2 -w tip3p -H -n 2 -q local
step openmm-full-protocol openduck openmm-full-protocol -l "$DATA/brd4_lig.mol" -r "$DATA/4LR6_aligned_chunk_nowat.pdb" \
    -i A_ASN_140_ND2 --waters-to-retain "$DATA/waters_to_retain.pdb" -f gaff2 -ff amber14-all -H -n 1 -m 0.02 \
    --trajectory-interval 0 $GPU_FLAG
step openmm-from-equilibrated openduck openmm-from-equilibrated -s ../openmm-full-protocol/equil.chk \
    -p ../openmm-full-protocol/complex_system.pickle -H -n 1 -m 0.02 --trajectory-interval 0 $GPU_FLAG
step report openduck report -p "$OUT/openmm-f*" -d all --seed 1

exit $status
