CREATE TABLE timings AS SELECT * FROM read_csv_auto('/tmp/ua_repro/lds_grid_*.csv');
CREATE TABLE params AS SELECT 0 id, 1.0::DOUBLE a, 0.89473684::DOUBLE b, 2.1::DOUBLE c, -2.5::DOUBLE d;
CREATE TABLE cells AS SELECT DISTINCT page, batch, ctx FROM timings;
CREATE TABLE features AS
SELECT *, (batch*16*s)//1216 full_waves, (batch*16*s)%1216 rem,
       ceil(ceil(least(ctx,1024)::DOUBLE/page)/s)*(page//32) tiles
FROM cells CROSS JOIN range(1,17) r(s)
WHERE s <= least(16,least(ctx,1024)//64,ceil(least(ctx,1024)::DOUBLE/page));
CREATE TABLE picks AS
SELECT id,page,batch,ctx,s FROM
(SELECT p.id,f.*, row_number() OVER(PARTITION BY id,page,batch,ctx ORDER BY
 ((full_waves+(rem>0)::INT)*(tiles*a+c)+tiles*b*(full_waves+rem/1216.0)+d*(s>1)::INT),s) rank
FROM features f CROSS JOIN params p) WHERE rank=1;
SELECT p.*,arg_min(t.S,t.B_us) best_s,min(t.B_us) best_us,
       max(t.B_us) FILTER(WHERE t.S=p.s) chosen_us,
       100*(chosen_us/best_us-1) loss FROM picks p JOIN timings t USING(page,batch,ctx)
GROUP BY ALL ORDER BY page,batch;
SELECT avg(loss) mean_loss,max(loss) max_loss,count(*) cells,count(loss) measured FROM
(SELECT p.*, 100*(max(t.B_us) FILTER(WHERE t.S=p.s)/min(t.B_us)-1) loss
FROM picks p JOIN timings t USING(page,batch,ctx) GROUP BY ALL);
