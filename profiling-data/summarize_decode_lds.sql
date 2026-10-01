SELECT page,ctx,batch,S_A,S_B,round(A_us,2) A_us,round(B_us,2) B_us,
       round(T_us,2) T_us,round(BA,3) BA,round(B_over_T,3) BT
FROM read_csv_auto('/tmp/ua_repro/lds_auto_*.csv') ORDER BY ctx DESC,page,batch;
SELECT page,ctx,count(*) cells,exp(avg(ln(BA))) geomean_BA,
       exp(avg(ln(B_over_T))) geomean_BT,min(BA) min_BA,max(BA) max_BA
FROM read_csv_auto('/tmp/ua_repro/lds_auto_*.csv') GROUP BY ALL ORDER BY ctx DESC,page;
SELECT page,batch,arg_min(S,A_us) A_best,arg_min(S,B_us) B_best,
       round(min(B_us)/min(A_us),3) best_BA,round(min(B_us)/min(T_us),3) best_BT
FROM read_csv_auto('/tmp/ua_repro/lds_grid_*.csv') GROUP BY ALL ORDER BY page,batch;
