from pathlib import Path
r=Path(__file__).parent
s=(r/'moe_stage_scoped.py').read_text().replace('Gamma,Count,Health,','Gamma,Count,Health,Z,Scatter,Ready,')
s+='''
            gl.amd.gfx1250.tdm.async_wait(0)
            gl.inline_asm_elementwise("s_wait_storecnt 0\\n s_mov_b32 $0, 0",constraints="=s,s,~{memory}",args=[expert_slot],dtype=gl.int32,is_pure=False,pack=1)
            gl.barrier()
            tile=gl.program_id(0)%32
            old=gl.atomic_or(Ready+tile,1<<expert_slot,sem='relaxed',scope='gpu')
            if (old|(1<<expert_slot))==63:
                cols=tile*128+gl.arange(0,128,layout=gl.BlockedLayout([1],[32],[4],[0]))
                acc=gl.full((128,),0,gl.float32,layout=gl.BlockedLayout([1],[32],[4],[0]))
                for j in gl.static_range(6):
                    row=gl.load(Scatter+j)
                    vals=gl.amd.gfx1250.buffer_load(Y,row*4096+cols,cache='.cg')
                    acc+=vals.to(gl.float32)
                gl.store(Z+cols,acc)
                gl.atomic_xchg(Ready+tile,0,sem='relaxed',scope='gpu')
'''
(r/'moe_stage_reduce.py').write_text(s)
s=(r/'test_moe_scoped.py').read_text().replace('from moe_stage_scoped import','from moe_stage_reduce import').replace("'moe_stage_scoped_results.json'","'moe_stage_reduce_results.json'").replace("f'moe-stage-scoped-b","f'moe-stage-reduce-b")
s=s.replace('args=(xq,','z=torch.empty((1,4096),device="cuda",dtype=torch.bfloat16);scatter=torch.randperm(6,device="cuda",dtype=torch.int32);ready=torch.zeros(32,device="cuda",dtype=torch.int32)\nrz=moe_gemm_a8w4(ri,w2,rs,s2,routing_data=rd,gammas=gamma,scatter_indx=scatter,swizzle_mx_scale="GFX1250_SCALE",preshuffled=True)\nargs=(xq,').replace('gamma,counter,health,a.stride()','gamma,counter,health,z,scatter,ready,a.stride()')
s=s.replace('torch.testing.assert_close(y,ry,rtol=0,atol=0)','torch.testing.assert_close(y,ry,rtol=0,atol=0);torch.testing.assert_close(z,rz,rtol=0,atol=0);assert not ready.any().item()')
s=s.replace("return moe_gemm_a8w4(qi,w2,qs,s2,routing_data=rd,gammas=gamma,", "return moe_gemm_a8w4(qi,w2,qs,s2,routing_data=rd,gammas=gamma,scatter_indx=scatter,")
(r/'test_moe_reduce.py').write_text(s)
