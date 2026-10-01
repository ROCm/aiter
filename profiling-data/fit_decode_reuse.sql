SET threads=4;
CREATE TABLE timings AS SELECT * FROM read_csv_auto(['/tmp/ua_repro/reuse_grid_p*.csv','/tmp/ua_repro/reuse_small_p*.csv']);
CREATE TABLE cells AS SELECT DISTINCT page,batch,ctx FROM timings;
CREATE TABLE features AS SELECT *, (batch*16*s)//2432 full_waves,(batch*16*s)%2432 rem,
ceil(ceil(least(ctx,1024)::DOUBLE/page)/s)*(page//32) tiles FROM cells CROSS JOIN range(1,17) r(s)
WHERE s <= least(16,least(ctx,1024)//64,ceil(least(ctx,1024)::DOUBLE/page));
CREATE TABLE params AS SELECT 0 id,1.0::DOUBLE a,0.89473684::DOUBLE b,2.1::DOUBLE c,-2.5::DOUBLE d;
INSERT INTO params SELECT row_number() OVER() id,1.0 a,b/10.0,c/2.0,d/2.0
FROM range(0,31) bb(b),range(-4,21) cc(c),range(-16,9) dd(d);
CREATE TABLE picks AS SELECT id,page,batch,ctx,arg_min(s,struct_pack(value:=model_cost,s:=s)) s
FROM (SELECT p.id,f.*, ((full_waves+(rem>0)::INT)*(tiles*a+c)+tiles*b*(full_waves+rem/2432.0)+d*(s>1)::INT) model_cost
FROM features f CROSS JOIN params p) GROUP BY ALL;
CREATE TABLE oracle AS SELECT page,batch,ctx,min(B_us) best FROM timings GROUP BY ALL;
CREATE TABLE losses AS SELECT p.id,avg(100*(t.B_us/o.best-1)) mean_loss,
max(100*(t.B_us/o.best-1)) max_loss,count(t.S) measured
FROM picks p LEFT JOIN timings t USING(page,batch,ctx,s) JOIN oracle o USING(page,batch,ctx) GROUP BY p.id;
SELECT params.*,losses.* EXCLUDE(id) FROM losses JOIN params USING(id)
WHERE measured=(SELECT count(*) FROM cells) OR id=0 ORDER BY mean_loss,max_loss LIMIT 15;
COPY (SELECT params.*,losses.* EXCLUDE(id) FROM losses JOIN params USING(id)
ORDER BY mean_loss,max_loss) TO '/tmp/ua_repro/reuse_model_fits.csv' (HEADER);
SELECT p.*,t.B_us,o.best,100*(t.B_us/o.best-1) loss FROM picks p LEFT JOIN timings t USING(page,batch,ctx,s)
JOIN oracle o USING(page,batch,ctx) WHERE p.id=0 ORDER BY page,batch;
