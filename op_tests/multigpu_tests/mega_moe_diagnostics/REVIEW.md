# Fine S1F probes: three pre-launch reviews

1. Address/ownership: one record per M64 producer wave; one per (M,N,wave)
   consumer task. Trace arrays sized from compact capacity. No shared log head,
   no new atomics, no LDS allocation. Only lane0 writes its disjoint records.
   Unused pull inp_scale argument carries trace pointer; runtime args checked.
2. Synchronization/semantics: no new wait or barrier. Clock is wave-local
   s_get_shader_cycles_u64 with compiler memory clobber. Never subtract clocks
   from different waves/TGs. Producer sample is the first M16 of each M64;
   its timestamps are written only after actual readiness publication. Earlier
   producer trace stores can affect later store waits; report perturbation.
   Consumer probes add timestamp stores inside hooks; overhead measured versus
   original with identical E2E harness. Fine intervals are issue spans unless
   bounded by an existing wait. No claim that scale/payload stores have retired.
3. Launch: reuse EP4 changing-input correctness first, then one full balanced
   tpr16384 run, all same CSV/roles. External-ready, TILE_READY0, P96/C160,
   no experimental scheduler/clusters. rocm-smi idle gate before every launch,
   1s polling, 240s process timeout; use shared gpu-lock if present. No foreign
   process termination. Source snapshots and checksums retained per run.
