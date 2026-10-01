create view d as select * from read_csv_auto(['dc_p32_c32768.csv','dc_p64_c32768.csv','dc_p32_c4096.csv','dc_p64_c4096.csv','dc_p32_c600.csv','dc_p64_c600.csv'], union_by_name=true);
select ctx,page,batch,round(F_us,1) F,round(A_us,1) A,round(B_us,1) B,round(F_A,3) F_A,round(F_B,3) F_B,round(F_us/least(A_us,B_us),3) F_best,round("F-A",3) dFA,round("F-B",3) dFB,round(maxref,2) mr from d order by ctx desc,page,batch;
select ctx,page, case when batch<=4 then 'B1-4' when batch<=56 then 'B5-56' else 'B64+' end reg, count(*) n, round(exp(avg(ln(F_A))),3) gA, round(exp(avg(ln(F_B))),3) gB, round(exp(avg(ln(F_us/least(A_us,B_us)))),3) gBest from d group by all order by ctx desc,page,reg;
select ctx,page,'B5-256' reg, round(exp(avg(ln(F_B))),3) gB from d where batch>=5 group by all order by ctx desc,page;
select ctx,page,batch,round(F_us/least(A_us,B_us),3) F_best, round(F_B,3) F_B, round(F_A,3) F_A, case when B_us<A_us then 'B' else 'A' end best from d where F_us>least(A_us,B_us) order by ctx desc,page,batch;
select max(spread_F) from d; select max(abs("F-A")/maxref), max(abs("F-B")/maxref) from d;
