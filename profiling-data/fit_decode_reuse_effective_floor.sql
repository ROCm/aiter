.read /home/jograner/projects/aiter/unified-attention-gemma4/profiling-data/fit_decode_reuse.sql
DROP TABLE picks;
DROP TABLE losses;
DROP TABLE params;
CREATE TABLE params AS SELECT row_number() OVER() id,1.0 a,b/5.0 b,c/2.0 c,d/2.0 d,ramp/2.0 ramp
FROM range(0,16) bb(b),range(-4,21,2) cc(c),range(-12,13,2) dd(d),range(1,9) rr(ramp);
CREATE TABLE picks AS SELECT id,page,batch,ctx,arg_min(s,struct_pack(value:=model_cost,s:=s)) s
FROM (SELECT p.id,f.*, (ceil(batch*16*s/(304.0*least(8.0,greatest(3.0,tiles/ramp))))*(tiles*a+c)
+tiles*b*(full_waves+rem/2432.0)+d*(s>1)::INT) model_cost
FROM features f CROSS JOIN params p) GROUP BY ALL;
CREATE TABLE losses AS SELECT p.id,avg(100*(t.B_us/o.best-1)) mean_loss,
max(100*(t.B_us/o.best-1)) max_loss,count(t.S) measured
FROM picks p LEFT JOIN timings t USING(page,batch,ctx,s) JOIN oracle o USING(page,batch,ctx) GROUP BY p.id;
SELECT params.*,losses.* EXCLUDE(id) FROM losses JOIN params USING(id)
WHERE measured=(SELECT count(*) FROM cells) ORDER BY mean_loss,max_loss LIMIT 15;
COPY (SELECT params.*,losses.* EXCLUDE(id) FROM losses JOIN params USING(id)
WHERE measured=(SELECT count(*) FROM cells) ORDER BY mean_loss,max_loss)
TO '/tmp/ua_repro/reuse_effective_floor_model_fits.csv' (HEADER);
COPY (SELECT p.*,t.B_us,o.best,100*(t.B_us/o.best-1) loss FROM picks p LEFT JOIN timings t USING(page,batch,ctx,s)
JOIN oracle o USING(page,batch,ctx) WHERE p.id=(SELECT id FROM losses WHERE measured=32 ORDER BY mean_loss,max_loss LIMIT 1)
ORDER BY page,batch) TO '/tmp/ua_repro/reuse_effective_floor_model_picks.csv' (HEADER);
