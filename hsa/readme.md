# ASM Kernel Config Code Generator

This script (codegen.py) merges per-architecture CSV kernel metadata into a single generated C++ header containing:

# HOW
Assembly CSVs must include `knl_name` and `co_name`. The concise aliases
`kernel` and `object` are also accepted and normalized to those canonical C++
fields during code generation. Leading/trailing whitespace is ignored, which
allows manifests to align columns for readability.
