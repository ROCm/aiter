.read /home/jograner/projects/aiter/unified-attention-gemma4/profiling-data/fit_decode_reuse.sql
DROP TABLE picks;
DROP TABLE losses;
DROP TABLE params;
CREATE TABLE params AS SELECT row_number() OVER() id,1.0 a,b/5.0 b,c::DOUBLE c,d::DOUBLE d,e/2.0 e, issue_waves
FROM range(0,11,2) bb(b),range(-2,11,2) cc(c),range(-6,7,2) dd(d),range(1,11) ee(e),range(1,8) ww(issue_waves);
CREATE TABLE picks AS SELECT id,page,batch,ctx,arg_min(s,struct_pack(value:=model_cost,s:=s)) s
FROM (SELECT p.id,f.*, ((full_waves+(rem>0)::INT)*(tiles*a+c)+tiles*b*(full_waves+rem/2432.0)+d*(s>1)::INT
+e*tiles*ceil(batch*16*s/(304.0*issue_waves))) model_cost
FROM features f CROSS JOIN params p) GROUP BY ALL;
CREATE TABLE losses AS SELECT p.id,avg(100*(t.B_us/o.best-1)) mean_loss,
max(100*(t.B_us/o.best-1)) max_loss,count(t.S) measured
FROM picks p LEFT JOIN timings t USING(page,batch,ctx,s) JOIN oracle o USING(page,batch,ctx) GROUP BY p.id;
SELECT params.*,losses.* EXCLUDE(id) FROM losses JOIN params USING(id)
WHERE measured=(SELECT count(*) FROM cells) ORDER BY mean_loss,max_loss LIMIT 15;
COPY (SELECT params.*,losses.* EXCLUDE(id) FROM losses JOIN params USING(id)
WHERE measured=(SELECT count(*) FROM cells) ORDER BY mean_loss,max_loss)
TO '/tmp/ua_repro/reuse_rounds_model_fits.csv' (HEADER);
COPY (SELECT p.*,t.B_us,o.best,100*(t.B_us/o.best-1) loss FROM picks p LEFT JOIN timings t USING(page,batch,ctx,s)
JOIN oracle o USING(page,batch,ctx) WHERE p.id=(SELECT id FROM losses WHERE measured=32 ORDER BY mean_loss,max_loss LIMIT 1)
ORDER BY page,batch) TO '/tmp/ua_repro/reuse_rounds_model_picks.csv' (HEADER);
