#!/bin/bash
echo "Compiling HIP kernel for Radeon GPU..."

# -O3 maximizes execution speed
# -fPIC and -shared tell it to build a Python-compatible .so library
hipcc -O3 -fPIC -shared intersect.hip -o libintersect.so

echo "Compilation complete! Created libintersect.so"