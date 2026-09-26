# Add MiniMax-M3 Model in ATOM/AITER

## Scope

ATOM/AITER + MiniMax-M3

## Repository

ROCm/aiter

## Purpose

Add a model PR note for MiniMax-M3 related AITER support and validation.

## YAML Changes

- Add `MiniMax-M3-MXFP4` to the default ATOM downstream model set in
  `.github/workflows/atom-test.yaml`.

## Notes

This does not change AITER runtime code. It does expand the ATOM downstream
workflow model matrix.

This work was moved from the mistaken ATOM repository target to AITER.
