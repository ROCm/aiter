#!/bin/bash
set -euo pipefail
mkdir -p /work
cp /opt/dsv4-mk-tp/reproduction/dsv4-monokernel/serve.sh /work/serve.sh
cd /work
export ATOM_DSV4_HYBRID=${ATOM_DSV4_HYBRID:-1}
export ATOM_DSV4_HYBRID_GEMM=${ATOM_DSV4_HYBRID_GEMM:-1}
export ATOM_DSV4_HYBRID_ATTN=${ATOM_DSV4_HYBRID_ATTN:-1}
export ATOM_DSV4_HYBRID_CHECK=${ATOM_DSV4_HYBRID_CHECK:-1}
export ATOM_FORCE_ATTN_TRITON=1 FAKE_EPLB=0
exec bash /work/serve.sh "$@"
