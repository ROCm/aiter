SET threads=4;
CREATE TABLE timings AS SELECT * FROM read_csv_auto(['/tmp/ua_repro/reuse_grid_p*.csv','/tmp/ua_repro/reuse_small_p*.csv']);
CREATE TABLE features AS SELECT *, (batch*16*s)//2432 full_waves,(batch*16*s)%2432 rem,
ceil(ceil(least(ctx,1024)::DOUBLE/page)/s)*(page//32) tiles FROM (SELECT DISTINCT page,batch,ctx FROM timings) CROSS JOIN range(1,17) r(s)
WHERE s <= least(16,least(ctx,1024)//64,ceil(least(ctx,1024)::DOUBLE/page));
CREATE TABLE params AS SELECT * FROM (VALUES ('original',0.89473684,2.1,-2.5),('simple',0.6,4.0,-4.0),('flat',0.0,2.0,0.0)) t(name,b,c,d);
CREATE TABLE picks AS SELECT name,page,batch,ctx,arg_min(s,struct_pack(value:=model_cost,s:=s)) s
FROM (SELECT p.name,f.*, (ceil(batch*16*s/(304.0*least(8.0,greatest(3.0,tiles/2.0))))*(tiles+c)
+tiles*b*(full_waves+rem/2432.0)+d*(s>1)::INT) model_cost
FROM features f CROSS JOIN params p) GROUP BY ALL;
CREATE TABLE results AS SELECT p.*,t.B_us,o.best,100*(t.B_us/o.best-1) loss FROM picks p LEFT JOIN timings t USING(page,batch,ctx,s)
JOIN (SELECT page,batch,ctx,min(B_us) best FROM timings GROUP BY ALL) o USING(page,batch,ctx);
SELECT name,avg(loss) mean_loss,max(loss) max_loss,count(loss) measured FROM results GROUP BY name;
COPY (SELECT * FROM results ORDER BY name,page,batch) TO '/tmp/ua_repro/reuse_final_model_picks.csv' (HEADER);
