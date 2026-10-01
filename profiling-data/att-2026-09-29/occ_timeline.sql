WITH waves AS (
 SELECT 'Fly B128 p32' tag, name, wave.begin AS b,wave."end" AS e FROM read_json_auto('/tmp/ua_repro/att/fly_b128_p32/ui_output_agent_6574_dispatch_393/se*_wv*.json')
 UNION ALL SELECT 'Fly B64 p32',name,wave.begin,wave."end" FROM read_json_auto('/tmp/ua_repro/att/fly_b64_p32/ui_output_agent_31771_dispatch_201/se*_wv*.json')
 UNION ALL SELECT 'Triton B128 p32',name,wave.begin,wave."end" FROM read_json_auto('/tmp/ua_repro/att/tri_b128_p32/ui_output_agent_1528_dispatch_393/se*_wv*.json')
), edges AS (SELECT tag,min(b) t0,max(e) t1 FROM waves GROUP BY tag),
 ticks AS (SELECT tag, round(p*100,0)::INTEGER AS pct, t0 + (t1-t0)*p AS tick FROM edges,unnest([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]) AS x(p))
SELECT t.tag,t.pct,count(*) FILTER (WHERE w.b<=t.tick AND w.e>t.tick) AS traced_waves,
 round(count(*) FILTER (WHERE w.b<=t.tick AND w.e>t.tick)/8.0,2) AS waves_per_traced_CU,
 round(count(*) FILTER (WHERE w.b<=t.tick AND w.e>t.tick)/32.0,2) AS waves_per_traced_SIMD
FROM ticks t JOIN waves w USING(tag) GROUP BY t.tag,t.pct,t.tick ORDER BY t.tag,t.pct;
