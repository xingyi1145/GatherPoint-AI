#!/bin/bash
set -e

echo "Compiling HIP kernel for Radeon GPU..."

# -O3 maximizes execution speed
# -fPIC and -shared tell it to build a Python-compatible .so library
hipcc -O3 -fPIC -shared intersect.hip -o libintersect.so

# Fail loudly instead of reporting success when hipcc silently produced nothing.
if [ ! -f libintersect.so ]; then
    echo "ERROR: libintersect.so was not created." >&2
    exit 1
fi

echo "Compilation complete! Created libintersect.so"
nm -D libintersect.so | grep -q " calculate_center$" \
    && echo "Verified: calculate_center symbol is present." \
    || { echo "ERROR: calculate_center symbol missing from libintersect.so" >&2; exit 1; }