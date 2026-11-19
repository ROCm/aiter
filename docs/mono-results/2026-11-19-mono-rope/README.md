# MONO rope Git migration validation

See [operator documentation](../../mono-rope.md) for the API, constraints and reproduction commands.

- `validation.json`: branch, counts, scope and resolved setup issues.
- `source-manifest.json`: SHA256 of all migrated implementation/test files.
- `migration-identity.json`: exact persistent body, descriptor and metadata identity; 460 unchanged A07 fallbacks.
- `static-contract.json`: DEV/HT, EXEC, SGPR and resource audit.
- `call-probes.json`: 36 passing normal probes and 3 passing negative controls.
- `import-validation.json`: GPU import and execution checks with the original shared JIT files, plus machine-code identity after narrowing the integration to GPU environments.

The external raw evidence directory is recorded in `validation.json`. No new performance measurements are claimed for this migration.
