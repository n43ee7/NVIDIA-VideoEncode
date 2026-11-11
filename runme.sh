#!/bin/sh
# Configures on first use, rebuilds what changed, then runs. ARCH=86 ./runme.sh targets another GPU.
D=$(cd "$(dirname "$0")" && pwd)
PATH="$PATH:/usr/local/cuda/bin"
[ -f "$D/build/Makefile" ] || cmake -S "$D" -B "$D/build" ${ARCH:+-DCMAKE_CUDA_ARCHITECTURES=$ARCH} >/dev/null || exit 1
cmake --build "$D/build" -j"$(nproc)" >/dev/null || { echo "[ERROR]   build failed, see: cmake --build $D/build"; exit 1; }
exec "$D/build/nvve" "$@"
