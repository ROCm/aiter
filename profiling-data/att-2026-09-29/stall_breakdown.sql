WITH f AS (
 SELECT 'Fly B128 p32' tag,* FROM read_csv_auto('/tmp/ua_repro/att/fly_b128_p32/stats_ui_output_agent_6574_dispatch_393.csv')
 UNION ALL SELECT 'Triton B128 p32',* FROM read_csv_auto('/tmp/ua_repro/att/tri_b128_p32/stats_ui_output_agent_1528_dispatch_393.csv')
 UNION ALL SELECT 'Fly B64 p32',* FROM read_csv_auto('/tmp/ua_repro/att/fly_b64_p32/stats_ui_output_agent_31771_dispatch_201.csv')
 UNION ALL SELECT 'Fly B128 p64',* FROM read_csv_auto('/tmp/ua_repro/att/fly_b128_p64/stats_ui_output_agent_62838_dispatch_393.csv')
 UNION ALL SELECT 'Triton B128 p64',* FROM read_csv_auto('/tmp/ua_repro/att/tri_b128_p64/stats_ui_output_agent_26968_dispatch_393.csv')
),
w AS (
 SELECT 'Fly B128 p32' tag,count(*) nw,sum(duration) ticks FROM read_json_auto('/tmp/ua_repro/att/fly_b128_p32/ui_output_agent_6574_dispatch_393/se*_wv*.json')
 UNION ALL SELECT 'Triton B128 p32',count(*),sum(duration) FROM read_json_auto('/tmp/ua_repro/att/tri_b128_p32/ui_output_agent_1528_dispatch_393/se*_wv*.json')
 UNION ALL SELECT 'Fly B64 p32',count(*),sum(duration) FROM read_json_auto('/tmp/ua_repro/att/fly_b64_p32/ui_output_agent_31771_dispatch_201/se*_wv*.json')
 UNION ALL SELECT 'Fly B128 p64',count(*),sum(duration) FROM read_json_auto('/tmp/ua_repro/att/fly_b128_p64/ui_output_agent_62838_dispatch_393/se*_wv*.json')
 UNION ALL SELECT 'Triton B128 p64',count(*),sum(duration) FROM read_json_auto('/tmp/ua_repro/att/tri_b128_p64/ui_output_agent_26968_dispatch_393/se*_wv*.json')
),
b AS (
SELECT tag,
 CASE WHEN Instruction LIKE 's_waitcnt vmcnt%' THEN 'VMEM wait'
 WHEN Instruction LIKE 's_waitcnt lgkmcnt%' THEN 'LGKM wait (LDS/SMEM)'
 WHEN Instruction LIKE 's_barrier%' THEN 'WG barrier'
 WHEN Instruction LIKE 's_waitcnt%' THEN 'other wait'
 WHEN Instruction LIKE 'v_mfma%' THEN 'MFMA issue/dependency'
 WHEN Instruction LIKE 'global_load%' OR Instruction LIKE 'buffer_load%' THEN 'VMEM load issue/stall'
 WHEN Instruction LIKE 'ds_%' THEN 'LDS issue/dependency'
 ELSE 'other issue/dependency' END AS category,
 sum(Hitcount) hits,sum(Stall) stalled,sum(Latency-Stall-Idle) issued,sum(Idle) idle
 FROM f GROUP BY 1,2)
SELECT b.tag,b.category,w.nw,round(b.stalled::DOUBLE/w.nw) stall_per_wave,round(100*b.stalled::DOUBLE/w.ticks,2) stall_pct,
 round(b.issued::DOUBLE/w.nw) issue_per_wave, round(b.idle::DOUBLE/w.nw) idle_per_wave,
 round(w.ticks::DOUBLE/w.nw) wave_ticks
 FROM b JOIN w USING (tag) ORDER BY tag,stall_per_wave DESC;
