# gfx1250-strict (MI450 A0) configs: NOT TUNED

Every DEFAULT.json here holds a single generic "any" entry, copied from the
gfx1250 table's "any" entry. There are no per-M buckets and no N/K-specialized
files, so loaders report these as untuned (is_tuned=False) and use the same
config for every shape. They exist so config-driven kernels can run on A0;
they say nothing about A0 performance. Replace a family's DEFAULT.json with
real A0 tables once it is tuned.

Families whose DEFAULT.json is keyed by kernel variant or shape rather than
by M (attention, KDA, grouped GEMM, block_cat, MoE a8w4/a4w4) have no generic
entry and are not here; resolve_config_dir() reads the gfx1250 table for them
with a one-time "untuned" warning.
