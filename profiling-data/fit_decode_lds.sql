.read /home/jograner/projects/aiter/unified-attention-gemma4/profiling-data/eval_decode_lds.sql
DELETE FROM params;
INSERT INTO params SELECT row_number() OVER() id,1.0 a,b/10.0,c/2.0,d/2.0
FROM range(0,31) bb(b),range(-4,21) cc(c),range(-16,9) dd(d);
CREATE TABLE costs AS SELECT p.id,f.page,f.batch,f.ctx,f.s,
 ((full_waves+(rem>0)::INT)*(tiles*a+c)+tiles*b*(full_waves+rem/1216.0)+d*(s>1)::INT) AS model_cost
FROM features f CROSS JOIN params p;
CREATE TABLE fitted_picks AS SELECT id,page,batch,ctx,arg_min(s,struct_pack(value:=model_cost,s:=s)) s
FROM costs GROUP BY ALL;
CREATE TABLE oracle AS SELECT page,batch,ctx,min(B_us) best FROM timings GROUP BY ALL;
CREATE TABLE losses AS SELECT p.id,avg(100*(t.B_us/o.best-1)) mean_loss,
 max(100*(t.B_us/o.best-1)) max_loss,count(t.S) measured
FROM fitted_picks p LEFT JOIN timings t USING(page,batch,ctx,s)
JOIN oracle o USING(page,batch,ctx) GROUP BY p.id;
SELECT params.*,losses.* EXCLUDE(id) FROM losses JOIN params USING(id)
WHERE measured=(SELECT count(*) FROM cells)
ORDER BY mean_loss,max_loss LIMIT 15;
COPY (SELECT params.*,losses.* EXCLUDE(id) FROM losses JOIN params USING(id)
WHERE measured=(SELECT count(*) FROM cells) ORDER BY mean_loss,max_loss LIMIT 15)
TO '/tmp/ua_repro/lds_model_fits.csv' (HEADER);
