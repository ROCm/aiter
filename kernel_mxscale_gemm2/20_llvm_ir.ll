; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

@mxscale_a8w4_64x256x256_1x4_2buf_arena = external addrspace(3) global [106496 x i8], align 1024

define amdgpu_kernel void @kernel_mxscale_gemm2(ptr addrspace(1) %0, <{ <{ i32, i32, i32 }>, <{ i64, i64 }> }> %1, ptr addrspace(1) %2, <{ <{ i32, i32, i32 }>, <{ i64, i64 }> }> %3, ptr addrspace(1) %4, <{ <{ i32, i32, i32 }>, <{ i64, i64 }> }> %5, ptr addrspace(1) %6, <{ <{ i32, i32, i32 }>, <{ i64, i64 }> }> %7, ptr addrspace(1) %8, <{ <{ i32, i32, i32 }>, <{ i64, i64 }> }> %9, ptr addrspace(1) %10, <{ <{ i32, i32, i32 }>, <{ i64, i64 }> }> %11, ptr addrspace(1) %12, <{ <{ i32 }> }> %13, ptr addrspace(1) %14, <{ <{ i32 }> }> %15, ptr addrspace(1) %16, <{ <{ i32 }> }> %17, i32 %18, i32 %19, i32 %20) #0 !reqd_work_group_size !1 {
  call void @llvm.amdgcn.s.setreg(i32 282, i32 1)
  %22 = call range(i32 0, 128) i32 @llvm.amdgcn.workitem.id.x()
  %23 = sext i32 %22 to i64
  %24 = call i32 @llvm.amdgcn.workgroup.id.y()
  %25 = sext i32 %24 to i64
  %26 = call i32 @llvm.amdgcn.workgroup.id.x()
  %27 = sext i32 %26 to i64
  %28 = addrspacecast ptr addrspace(1) %12 to ptr
  %29 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %28, i16 0, i64 4294967295, i32 159744)
  %30 = trunc i64 %27 to i32
  %31 = mul i32 %30, 4
  %32 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %29, i32 %31, i32 0, i32 0)
  %33 = icmp sgt i32 %32, 0
  br i1 %33, label %34, label %1181

34:                                               ; preds = %21
  %35 = mul i64 %25, 256
  %36 = mul i64 %27, 64
  %37 = mul i64 %27, 192
  %38 = mul i64 %27, 16
  %39 = mul i64 %27, 768
  %40 = trunc i64 %23 to i32
  %41 = sdiv i32 %40, 32
  %42 = srem i32 %41, 4
  %43 = sdiv i32 %40, 16
  %44 = srem i32 %43, 2
  %45 = srem i32 %40, 16
  %46 = sext i32 %42 to i64
  %47 = sext i32 %44 to i64
  %48 = sext i32 %45 to i64
  %49 = mul i64 %46, 64
  %50 = mul i64 %46, 4096
  %51 = mul i64 %48, 64
  %52 = add i64 %50, %51
  %53 = mul i64 %47, 8
  %54 = add i64 %52, %53
  %55 = mul i64 %46, 8192
  %56 = add i64 %35, %49
  %57 = ptrtoint ptr addrspace(1) %0 to i64
  %58 = mul i64 %27, 196608
  %59 = add i64 %58, %56
  %60 = mul i64 %59, 2
  %61 = add i64 %57, %60
  %62 = add i64 ptrtoint (ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena to i64), %55
  %63 = trunc i64 %62 to i32
  %64 = trunc i64 %61 to i32
  %65 = lshr i64 %61, 32
  %66 = trunc i64 %65 to i32
  %67 = or i32 %66, -2147483648
  %68 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %63, i64 1
  %69 = insertelement <4 x i32> %68, i32 %64, i64 2
  %70 = insertelement <4 x i32> %69, i32 %67, i64 3
  %71 = call i32 @llvm.amdgcn.wave.id()
  %72 = sext i32 %71 to i64
  %73 = urem i64 %72, 4
  %74 = udiv i64 %72, 4
  %75 = mul i64 %73, 16
  %76 = mul i64 %74, 256
  %77 = mul i64 %73, 4352
  %78 = add i64 %77, %76
  %79 = add i64 ptrtoint (ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena to i64), %78
  %80 = trunc i64 %79 to i32
  %81 = mul i64 %73, 4
  %82 = mul i64 %74, 2048
  %83 = mul i64 %73, 8192
  %84 = add i64 %83, %82
  %85 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i32 17408) to i64), %84
  %86 = trunc i64 %85 to i32
  %87 = mul i64 %74, 32
  %88 = mul i64 %73, 128
  %89 = add i64 %88, %87
  %90 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i32 50176) to i64), %89
  %91 = trunc i64 %90 to i32
  %92 = mul i64 %73, 512
  %93 = add i64 %92, %87
  %94 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i32 50704) to i64), %93
  %95 = trunc i64 %94 to i32
  %96 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i32 53248) to i64), %78
  %97 = trunc i64 %96 to i32
  %98 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i32 70656) to i64), %84
  %99 = trunc i64 %98 to i32
  %100 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i32 103424) to i64), %89
  %101 = trunc i64 %100 to i32
  %102 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i32 103952) to i64), %93
  %103 = trunc i64 %102 to i32
  %104 = ptrtoint ptr addrspace(1) %2 to i64
  %105 = add i64 %36, %75
  %106 = mul i64 %105, 3072
  %107 = add i64 %106, %76
  %108 = add i64 %104, %107
  %109 = trunc i64 %108 to i32
  %110 = lshr i64 %108, 32
  %111 = trunc i64 %110 to i32
  %112 = or i32 %111, -2147483648
  %113 = udiv i64 %35, 16
  %114 = add i64 %37, %113
  %115 = ptrtoint ptr addrspace(1) %4 to i64
  %116 = add i64 %114, %81
  %117 = mul i64 %116, 24576
  %118 = add i64 %117, %82
  %119 = add i64 %115, %118
  %120 = trunc i64 %119 to i32
  %121 = lshr i64 %119, 32
  %122 = trunc i64 %121 to i32
  %123 = or i32 %122, -2147483648
  %124 = ptrtoint ptr addrspace(1) %6 to i64
  %125 = add i64 %38, %81
  %126 = mul i64 %125, 384
  %127 = add i64 %126, %87
  %128 = add i64 %124, %127
  %129 = trunc i64 %128 to i32
  %130 = lshr i64 %128, 32
  %131 = trunc i64 %130 to i32
  %132 = or i32 %131, -2147483648
  %133 = udiv i64 %35, 4
  %134 = add i64 %39, %133
  %135 = ptrtoint ptr addrspace(1) %8 to i64
  %136 = add i64 %134, %75
  %137 = mul i64 %136, 384
  %138 = add i64 %137, %87
  %139 = add i64 %135, %138
  %140 = trunc i64 %139 to i32
  %141 = lshr i64 %139, 32
  %142 = trunc i64 %141 to i32
  %143 = or i32 %142, -2147483648
  %144 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %80, i64 1
  %145 = insertelement <4 x i32> %144, i32 %109, i64 2
  %146 = insertelement <4 x i32> %145, i32 %112, i64 3
  %147 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %86, i64 1
  %148 = insertelement <4 x i32> %147, i32 %120, i64 2
  %149 = insertelement <4 x i32> %148, i32 %123, i64 3
  %150 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %91, i64 1
  %151 = insertelement <4 x i32> %150, i32 %129, i64 2
  %152 = insertelement <4 x i32> %151, i32 %132, i64 3
  %153 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %95, i64 1
  %154 = insertelement <4 x i32> %153, i32 %140, i64 2
  %155 = insertelement <4 x i32> %154, i32 %143, i64 3
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %146, <8 x i32> <i32 122683392, i32 16777216, i32 1048576, i32 16777216, i32 16, i32 3072, i32 0, i32 0>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %149, <8 x i32> <i32 0, i32 134217728, i32 262144, i32 134217728, i32 4, i32 24576, i32 0, i32 0>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %152, <8 x i32> <i32 0, i32 2097152, i32 262144, i32 2097152, i32 4, i32 384, i32 0, i32 0>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %155, <8 x i32> <i32 0, i32 2097152, i32 1048576, i32 2097152, i32 16, i32 384, i32 0, i32 0>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  %156 = add i32 %109, 256
  %157 = icmp ult i32 %156, %109
  %158 = add i32 %112, 1
  %159 = select i1 %157, i32 %158, i32 %112
  %160 = add i32 %120, 2048
  %161 = icmp ult i32 %160, %120
  %162 = add i32 %123, 1
  %163 = select i1 %161, i32 %162, i32 %123
  %164 = add i32 %129, 32
  %165 = icmp ult i32 %164, %129
  %166 = add i32 %132, 1
  %167 = select i1 %165, i32 %166, i32 %132
  %168 = add i32 %140, 32
  %169 = icmp ult i32 %168, %140
  %170 = add i32 %143, 1
  %171 = select i1 %169, i32 %170, i32 %143
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  br label %172

172:                                              ; preds = %199, %34
  %173 = phi i64 [ %679, %199 ], [ 0, %34 ]
  %174 = phi <8 x float> [ %641, %199 ], [ zeroinitializer, %34 ]
  %175 = phi <8 x float> [ %642, %199 ], [ zeroinitializer, %34 ]
  %176 = phi <8 x float> [ %643, %199 ], [ zeroinitializer, %34 ]
  %177 = phi <8 x float> [ %644, %199 ], [ zeroinitializer, %34 ]
  %178 = phi <8 x float> [ %648, %199 ], [ zeroinitializer, %34 ]
  %179 = phi <8 x float> [ %647, %199 ], [ zeroinitializer, %34 ]
  %180 = phi <8 x float> [ %646, %199 ], [ zeroinitializer, %34 ]
  %181 = phi <8 x float> [ %645, %199 ], [ zeroinitializer, %34 ]
  %182 = phi <8 x float> [ %671, %199 ], [ zeroinitializer, %34 ]
  %183 = phi <8 x float> [ %672, %199 ], [ zeroinitializer, %34 ]
  %184 = phi <8 x float> [ %673, %199 ], [ zeroinitializer, %34 ]
  %185 = phi <8 x float> [ %674, %199 ], [ zeroinitializer, %34 ]
  %186 = phi <8 x float> [ %678, %199 ], [ zeroinitializer, %34 ]
  %187 = phi <8 x float> [ %677, %199 ], [ zeroinitializer, %34 ]
  %188 = phi <8 x float> [ %676, %199 ], [ zeroinitializer, %34 ]
  %189 = phi <8 x float> [ %675, %199 ], [ zeroinitializer, %34 ]
  %190 = phi i32 [ %564, %199 ], [ %156, %34 ]
  %191 = phi i32 [ %567, %199 ], [ %159, %34 ]
  %192 = phi i32 [ %568, %199 ], [ %160, %34 ]
  %193 = phi i32 [ %571, %199 ], [ %163, %34 ]
  %194 = phi i32 [ %572, %199 ], [ %164, %34 ]
  %195 = phi i32 [ %575, %199 ], [ %167, %34 ]
  %196 = phi i32 [ %576, %199 ], [ %168, %34 ]
  %197 = phi i32 [ %579, %199 ], [ %171, %34 ]
  %198 = icmp slt i64 %173, 5
  br i1 %198, label %199, label %680

199:                                              ; preds = %172
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  %200 = mul i64 %173, 512
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %201 = mul i64 %48, 272
  %202 = mul i64 %47, 16
  %203 = add i64 %201, %202
  %204 = mul i64 %48, 16
  %205 = mul i64 %47, 256
  %206 = add i64 %55, %204
  %207 = add i64 %206, %205
  %208 = mul i64 %48, 32
  %209 = mul i64 %47, 4
  %210 = add i64 %208, %209
  %211 = udiv i64 %49, 4
  %212 = add i64 %211, %48
  %213 = mul i64 %212, 32
  %214 = add i64 %213, %209
  %215 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i32 17408) to i64), %207
  %216 = trunc i64 %215 to i32
  %217 = inttoptr i32 %216 to ptr addrspace(3)
  %218 = load <4 x i32>, ptr addrspace(3) %217, align 16
  %219 = getelementptr i8, ptr addrspace(3) %217, i32 512
  %220 = load <4 x i32>, ptr addrspace(3) %219, align 16
  %221 = shufflevector <4 x i32> %218, <4 x i32> %220, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %222 = getelementptr i8, ptr addrspace(3) %217, i32 2048
  %223 = load <4 x i32>, ptr addrspace(3) %222, align 16
  %224 = getelementptr i8, ptr addrspace(3) %217, i32 2560
  %225 = load <4 x i32>, ptr addrspace(3) %224, align 16
  %226 = shufflevector <4 x i32> %223, <4 x i32> %225, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %227 = getelementptr i8, ptr addrspace(3) %217, i32 4096
  %228 = load <4 x i32>, ptr addrspace(3) %227, align 16
  %229 = getelementptr i8, ptr addrspace(3) %217, i32 4608
  %230 = load <4 x i32>, ptr addrspace(3) %229, align 16
  %231 = shufflevector <4 x i32> %228, <4 x i32> %230, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %232 = getelementptr i8, ptr addrspace(3) %217, i32 6144
  %233 = load <4 x i32>, ptr addrspace(3) %232, align 16
  %234 = getelementptr i8, ptr addrspace(3) %217, i32 6656
  %235 = load <4 x i32>, ptr addrspace(3) %234, align 16
  %236 = shufflevector <4 x i32> %233, <4 x i32> %235, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %237 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i32 50704) to i64), %214
  %238 = trunc i64 %237 to i32
  %239 = inttoptr i32 %238 to ptr addrspace(3)
  %240 = load <4 x i32>, ptr addrspace(3) %239, align 16
  %241 = extractelement <4 x i32> %240, i64 0
  %242 = extractelement <4 x i32> %240, i64 1
  %243 = extractelement <4 x i32> %240, i64 2
  %244 = extractelement <4 x i32> %240, i64 3
  %245 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i32 50176) to i64), %210
  %246 = trunc i64 %245 to i32
  %247 = inttoptr i32 %246 to ptr addrspace(3)
  %248 = load <4 x i32>, ptr addrspace(3) %247, align 16
  %249 = extractelement <4 x i32> %248, i64 0
  %250 = extractelement <4 x i32> %248, i64 1
  %251 = extractelement <4 x i32> %248, i64 2
  %252 = extractelement <4 x i32> %248, i64 3
  %253 = add i64 ptrtoint (ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena to i64), %203
  %254 = trunc i64 %253 to i32
  %255 = inttoptr i32 %254 to ptr addrspace(3)
  %256 = load <4 x i32>, ptr addrspace(3) %255, align 16
  %257 = getelementptr i8, ptr addrspace(3) %255, i32 32
  %258 = load <4 x i32>, ptr addrspace(3) %257, align 16
  %259 = getelementptr i8, ptr addrspace(3) %255, i32 64
  %260 = load <4 x i32>, ptr addrspace(3) %259, align 16
  %261 = getelementptr i8, ptr addrspace(3) %255, i32 96
  %262 = load <4 x i32>, ptr addrspace(3) %261, align 16
  %263 = shufflevector <4 x i32> %256, <4 x i32> %258, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %264 = shufflevector <4 x i32> %260, <4 x i32> %262, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %265 = shufflevector <8 x i32> %263, <8 x i32> %264, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %266 = getelementptr i8, ptr addrspace(3) %255, i32 4352
  %267 = load <4 x i32>, ptr addrspace(3) %266, align 16
  %268 = getelementptr i8, ptr addrspace(3) %255, i32 4384
  %269 = load <4 x i32>, ptr addrspace(3) %268, align 16
  %270 = getelementptr i8, ptr addrspace(3) %255, i32 4416
  %271 = load <4 x i32>, ptr addrspace(3) %270, align 16
  %272 = getelementptr i8, ptr addrspace(3) %255, i32 4448
  %273 = load <4 x i32>, ptr addrspace(3) %272, align 16
  %274 = shufflevector <4 x i32> %267, <4 x i32> %269, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %275 = shufflevector <4 x i32> %271, <4 x i32> %273, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %276 = shufflevector <8 x i32> %274, <8 x i32> %275, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %277 = getelementptr i8, ptr addrspace(3) %217, i32 1024
  %278 = load <4 x i32>, ptr addrspace(3) %277, align 16
  %279 = getelementptr i8, ptr addrspace(3) %217, i32 1536
  %280 = load <4 x i32>, ptr addrspace(3) %279, align 16
  %281 = shufflevector <4 x i32> %278, <4 x i32> %280, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %282 = getelementptr i8, ptr addrspace(3) %217, i32 3072
  %283 = load <4 x i32>, ptr addrspace(3) %282, align 16
  %284 = getelementptr i8, ptr addrspace(3) %217, i32 3584
  %285 = load <4 x i32>, ptr addrspace(3) %284, align 16
  %286 = shufflevector <4 x i32> %283, <4 x i32> %285, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %287 = getelementptr i8, ptr addrspace(3) %217, i32 5120
  %288 = load <4 x i32>, ptr addrspace(3) %287, align 16
  %289 = getelementptr i8, ptr addrspace(3) %217, i32 5632
  %290 = load <4 x i32>, ptr addrspace(3) %289, align 16
  %291 = shufflevector <4 x i32> %288, <4 x i32> %290, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %292 = getelementptr i8, ptr addrspace(3) %217, i32 7168
  %293 = load <4 x i32>, ptr addrspace(3) %292, align 16
  %294 = getelementptr i8, ptr addrspace(3) %217, i32 7680
  %295 = load <4 x i32>, ptr addrspace(3) %294, align 16
  %296 = shufflevector <4 x i32> %293, <4 x i32> %295, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %297 = getelementptr i8, ptr addrspace(3) %239, i32 16
  %298 = load <4 x i32>, ptr addrspace(3) %297, align 16
  %299 = extractelement <4 x i32> %298, i64 0
  %300 = extractelement <4 x i32> %298, i64 1
  %301 = extractelement <4 x i32> %298, i64 2
  %302 = extractelement <4 x i32> %298, i64 3
  %303 = getelementptr i8, ptr addrspace(3) %247, i32 16
  %304 = load <4 x i32>, ptr addrspace(3) %303, align 16
  %305 = extractelement <4 x i32> %304, i64 0
  %306 = extractelement <4 x i32> %304, i64 1
  %307 = extractelement <4 x i32> %304, i64 2
  %308 = extractelement <4 x i32> %304, i64 3
  call void @llvm.amdgcn.s.wait.dscnt(i16 10)
  %309 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %221, i32 0, <16 x i32> %265, i16 0, <8 x float> %174, i32 0, i32 0, i32 %241, i32 0, i32 0, i32 %249, i1 false, i1 false)
  %310 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %226, i32 0, <16 x i32> %265, i16 0, <8 x float> %175, i32 0, i32 0, i32 %242, i32 0, i32 0, i32 %249, i1 false, i1 false)
  %311 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %231, i32 0, <16 x i32> %265, i16 0, <8 x float> %176, i32 0, i32 0, i32 %243, i32 0, i32 0, i32 %249, i1 false, i1 false)
  %312 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %236, i32 0, <16 x i32> %265, i16 0, <8 x float> %177, i32 0, i32 0, i32 %244, i32 0, i32 0, i32 %249, i1 false, i1 false)
  %313 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %236, i32 0, <16 x i32> %276, i16 0, <8 x float> %181, i32 0, i32 0, i32 %244, i32 0, i32 0, i32 %250, i1 false, i1 false)
  %314 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %231, i32 0, <16 x i32> %276, i16 0, <8 x float> %180, i32 0, i32 0, i32 %243, i32 0, i32 0, i32 %250, i1 false, i1 false)
  %315 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %226, i32 0, <16 x i32> %276, i16 0, <8 x float> %179, i32 0, i32 0, i32 %242, i32 0, i32 0, i32 %250, i1 false, i1 false)
  %316 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %221, i32 0, <16 x i32> %276, i16 0, <8 x float> %178, i32 0, i32 0, i32 %241, i32 0, i32 0, i32 %250, i1 false, i1 false)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %317 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %97, i64 1
  %318 = insertelement <4 x i32> %317, i32 %190, i64 2
  %319 = insertelement <4 x i32> %318, i32 %191, i64 3
  %320 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %99, i64 1
  %321 = insertelement <4 x i32> %320, i32 %192, i64 2
  %322 = insertelement <4 x i32> %321, i32 %193, i64 3
  %323 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %101, i64 1
  %324 = insertelement <4 x i32> %323, i32 %194, i64 2
  %325 = insertelement <4 x i32> %324, i32 %195, i64 3
  %326 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %103, i64 1
  %327 = insertelement <4 x i32> %326, i32 %196, i64 2
  %328 = insertelement <4 x i32> %327, i32 %197, i64 3
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %319, <8 x i32> <i32 122683392, i32 16777216, i32 1048576, i32 16777216, i32 16, i32 3072, i32 0, i32 0>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %322, <8 x i32> <i32 0, i32 134217728, i32 262144, i32 134217728, i32 4, i32 24576, i32 0, i32 0>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %325, <8 x i32> <i32 0, i32 2097152, i32 262144, i32 2097152, i32 4, i32 384, i32 0, i32 0>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %328, <8 x i32> <i32 0, i32 2097152, i32 1048576, i32 2097152, i32 16, i32 384, i32 0, i32 0>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  %329 = add i32 %190, 256
  %330 = icmp ult i32 %329, %190
  %331 = add i32 %191, 1
  %332 = select i1 %330, i32 %331, i32 %191
  %333 = add i32 %192, 2048
  %334 = icmp ult i32 %333, %192
  %335 = add i32 %193, 1
  %336 = select i1 %334, i32 %335, i32 %193
  %337 = add i32 %194, 32
  %338 = icmp ult i32 %337, %194
  %339 = add i32 %195, 1
  %340 = select i1 %338, i32 %339, i32 %195
  %341 = add i32 %196, 32
  %342 = icmp ult i32 %341, %196
  %343 = add i32 %197, 1
  %344 = select i1 %342, i32 %343, i32 %197
  %345 = add i64 %200, 768
  %346 = udiv i64 %345, 2
  %347 = urem i64 %23, 64
  %348 = add i64 %36, %347
  %349 = mul i64 %348, 3072
  %350 = add i64 %349, %345
  %351 = add i64 %104, %350
  %352 = inttoptr i64 %351 to ptr addrspace(1)
  call void @llvm.amdgcn.global.prefetch(ptr addrspace(1) %352, i32 8)
  %353 = mul i64 %346, 16
  %354 = urem i64 %23, 16
  %355 = add i64 %114, %354
  %356 = mul i64 %355, 24576
  %357 = add i64 %356, %353
  %358 = add i64 %115, %357
  %359 = inttoptr i64 %358 to ptr addrspace(1)
  call void @llvm.amdgcn.global.prefetch(ptr addrspace(1) %359, i32 8)
  %360 = getelementptr i8, ptr addrspace(3) %255, i32 8704
  %361 = load <4 x i32>, ptr addrspace(3) %360, align 16
  %362 = getelementptr i8, ptr addrspace(3) %255, i32 8736
  %363 = load <4 x i32>, ptr addrspace(3) %362, align 16
  %364 = getelementptr i8, ptr addrspace(3) %255, i32 8768
  %365 = load <4 x i32>, ptr addrspace(3) %364, align 16
  %366 = getelementptr i8, ptr addrspace(3) %255, i32 8800
  %367 = load <4 x i32>, ptr addrspace(3) %366, align 16
  %368 = shufflevector <4 x i32> %361, <4 x i32> %363, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %369 = shufflevector <4 x i32> %365, <4 x i32> %367, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %370 = shufflevector <8 x i32> %368, <8 x i32> %369, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %371 = getelementptr i8, ptr addrspace(3) %255, i32 13056
  %372 = load <4 x i32>, ptr addrspace(3) %371, align 16
  %373 = getelementptr i8, ptr addrspace(3) %255, i32 13088
  %374 = load <4 x i32>, ptr addrspace(3) %373, align 16
  %375 = getelementptr i8, ptr addrspace(3) %255, i32 13120
  %376 = load <4 x i32>, ptr addrspace(3) %375, align 16
  %377 = getelementptr i8, ptr addrspace(3) %255, i32 13152
  %378 = load <4 x i32>, ptr addrspace(3) %377, align 16
  %379 = shufflevector <4 x i32> %372, <4 x i32> %374, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %380 = shufflevector <4 x i32> %376, <4 x i32> %378, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %381 = shufflevector <8 x i32> %379, <8 x i32> %380, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 10)
  %382 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %221, i32 0, <16 x i32> %370, i16 0, <8 x float> %182, i32 0, i32 0, i32 %241, i32 0, i32 0, i32 %251, i1 false, i1 false)
  %383 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %226, i32 0, <16 x i32> %370, i16 0, <8 x float> %183, i32 0, i32 0, i32 %242, i32 0, i32 0, i32 %251, i1 false, i1 false)
  %384 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %231, i32 0, <16 x i32> %370, i16 0, <8 x float> %184, i32 0, i32 0, i32 %243, i32 0, i32 0, i32 %251, i1 false, i1 false)
  %385 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %236, i32 0, <16 x i32> %370, i16 0, <8 x float> %185, i32 0, i32 0, i32 %244, i32 0, i32 0, i32 %251, i1 false, i1 false)
  %386 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %236, i32 0, <16 x i32> %381, i16 0, <8 x float> %189, i32 0, i32 0, i32 %244, i32 0, i32 0, i32 %252, i1 false, i1 false)
  %387 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %231, i32 0, <16 x i32> %381, i16 0, <8 x float> %188, i32 0, i32 0, i32 %243, i32 0, i32 0, i32 %252, i1 false, i1 false)
  %388 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %226, i32 0, <16 x i32> %381, i16 0, <8 x float> %187, i32 0, i32 0, i32 %242, i32 0, i32 0, i32 %252, i1 false, i1 false)
  %389 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %221, i32 0, <16 x i32> %381, i16 0, <8 x float> %186, i32 0, i32 0, i32 %241, i32 0, i32 0, i32 %252, i1 false, i1 false)
  %390 = getelementptr i8, ptr addrspace(3) %255, i32 128
  %391 = load <4 x i32>, ptr addrspace(3) %390, align 16
  %392 = getelementptr i8, ptr addrspace(3) %255, i32 160
  %393 = load <4 x i32>, ptr addrspace(3) %392, align 16
  %394 = getelementptr i8, ptr addrspace(3) %255, i32 192
  %395 = load <4 x i32>, ptr addrspace(3) %394, align 16
  %396 = getelementptr i8, ptr addrspace(3) %255, i32 224
  %397 = load <4 x i32>, ptr addrspace(3) %396, align 16
  %398 = shufflevector <4 x i32> %391, <4 x i32> %393, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %399 = shufflevector <4 x i32> %395, <4 x i32> %397, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %400 = shufflevector <8 x i32> %398, <8 x i32> %399, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %401 = getelementptr i8, ptr addrspace(3) %255, i32 4480
  %402 = load <4 x i32>, ptr addrspace(3) %401, align 16
  %403 = getelementptr i8, ptr addrspace(3) %255, i32 4512
  %404 = load <4 x i32>, ptr addrspace(3) %403, align 16
  %405 = getelementptr i8, ptr addrspace(3) %255, i32 4544
  %406 = load <4 x i32>, ptr addrspace(3) %405, align 16
  %407 = getelementptr i8, ptr addrspace(3) %255, i32 4576
  %408 = load <4 x i32>, ptr addrspace(3) %407, align 16
  %409 = shufflevector <4 x i32> %402, <4 x i32> %404, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %410 = shufflevector <4 x i32> %406, <4 x i32> %408, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %411 = shufflevector <8 x i32> %409, <8 x i32> %410, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %412 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %281, i32 0, <16 x i32> %400, i16 0, <8 x float> %309, i32 0, i32 0, i32 %299, i32 0, i32 0, i32 %305, i1 false, i1 false)
  %413 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %286, i32 0, <16 x i32> %400, i16 0, <8 x float> %310, i32 0, i32 0, i32 %300, i32 0, i32 0, i32 %305, i1 false, i1 false)
  %414 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %291, i32 0, <16 x i32> %400, i16 0, <8 x float> %311, i32 0, i32 0, i32 %301, i32 0, i32 0, i32 %305, i1 false, i1 false)
  %415 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %296, i32 0, <16 x i32> %400, i16 0, <8 x float> %312, i32 0, i32 0, i32 %302, i32 0, i32 0, i32 %305, i1 false, i1 false)
  %416 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %296, i32 0, <16 x i32> %411, i16 0, <8 x float> %313, i32 0, i32 0, i32 %302, i32 0, i32 0, i32 %306, i1 false, i1 false)
  %417 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %291, i32 0, <16 x i32> %411, i16 0, <8 x float> %314, i32 0, i32 0, i32 %301, i32 0, i32 0, i32 %306, i1 false, i1 false)
  %418 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %286, i32 0, <16 x i32> %411, i16 0, <8 x float> %315, i32 0, i32 0, i32 %300, i32 0, i32 0, i32 %306, i1 false, i1 false)
  %419 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %281, i32 0, <16 x i32> %411, i16 0, <8 x float> %316, i32 0, i32 0, i32 %299, i32 0, i32 0, i32 %306, i1 false, i1 false)
  %420 = getelementptr i8, ptr addrspace(3) %255, i32 8832
  %421 = load <4 x i32>, ptr addrspace(3) %420, align 16
  %422 = getelementptr i8, ptr addrspace(3) %255, i32 8864
  %423 = load <4 x i32>, ptr addrspace(3) %422, align 16
  %424 = getelementptr i8, ptr addrspace(3) %255, i32 8896
  %425 = load <4 x i32>, ptr addrspace(3) %424, align 16
  %426 = getelementptr i8, ptr addrspace(3) %255, i32 8928
  %427 = load <4 x i32>, ptr addrspace(3) %426, align 16
  %428 = shufflevector <4 x i32> %421, <4 x i32> %423, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %429 = shufflevector <4 x i32> %425, <4 x i32> %427, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %430 = shufflevector <8 x i32> %428, <8 x i32> %429, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %431 = getelementptr i8, ptr addrspace(3) %255, i32 13184
  %432 = load <4 x i32>, ptr addrspace(3) %431, align 16
  %433 = getelementptr i8, ptr addrspace(3) %255, i32 13216
  %434 = load <4 x i32>, ptr addrspace(3) %433, align 16
  %435 = getelementptr i8, ptr addrspace(3) %255, i32 13248
  %436 = load <4 x i32>, ptr addrspace(3) %435, align 16
  %437 = getelementptr i8, ptr addrspace(3) %255, i32 13280
  %438 = load <4 x i32>, ptr addrspace(3) %437, align 16
  %439 = shufflevector <4 x i32> %432, <4 x i32> %434, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %440 = shufflevector <4 x i32> %436, <4 x i32> %438, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %441 = shufflevector <8 x i32> %439, <8 x i32> %440, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %442 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %281, i32 0, <16 x i32> %430, i16 0, <8 x float> %382, i32 0, i32 0, i32 %299, i32 0, i32 0, i32 %307, i1 false, i1 false)
  %443 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %286, i32 0, <16 x i32> %430, i16 0, <8 x float> %383, i32 0, i32 0, i32 %300, i32 0, i32 0, i32 %307, i1 false, i1 false)
  %444 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %291, i32 0, <16 x i32> %430, i16 0, <8 x float> %384, i32 0, i32 0, i32 %301, i32 0, i32 0, i32 %307, i1 false, i1 false)
  %445 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %296, i32 0, <16 x i32> %430, i16 0, <8 x float> %385, i32 0, i32 0, i32 %302, i32 0, i32 0, i32 %307, i1 false, i1 false)
  %446 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %296, i32 0, <16 x i32> %441, i16 0, <8 x float> %386, i32 0, i32 0, i32 %302, i32 0, i32 0, i32 %308, i1 false, i1 false)
  %447 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %291, i32 0, <16 x i32> %441, i16 0, <8 x float> %387, i32 0, i32 0, i32 %301, i32 0, i32 0, i32 %308, i1 false, i1 false)
  %448 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %286, i32 0, <16 x i32> %441, i16 0, <8 x float> %388, i32 0, i32 0, i32 %300, i32 0, i32 0, i32 %308, i1 false, i1 false)
  %449 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %281, i32 0, <16 x i32> %441, i16 0, <8 x float> %389, i32 0, i32 0, i32 %299, i32 0, i32 0, i32 %308, i1 false, i1 false)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 18, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 10, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 8, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %450 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i32 70656) to i64), %207
  %451 = trunc i64 %450 to i32
  %452 = inttoptr i32 %451 to ptr addrspace(3)
  %453 = load <4 x i32>, ptr addrspace(3) %452, align 16
  %454 = getelementptr i8, ptr addrspace(3) %452, i32 512
  %455 = load <4 x i32>, ptr addrspace(3) %454, align 16
  %456 = shufflevector <4 x i32> %453, <4 x i32> %455, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %457 = getelementptr i8, ptr addrspace(3) %452, i32 2048
  %458 = load <4 x i32>, ptr addrspace(3) %457, align 16
  %459 = getelementptr i8, ptr addrspace(3) %452, i32 2560
  %460 = load <4 x i32>, ptr addrspace(3) %459, align 16
  %461 = shufflevector <4 x i32> %458, <4 x i32> %460, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %462 = getelementptr i8, ptr addrspace(3) %452, i32 4096
  %463 = load <4 x i32>, ptr addrspace(3) %462, align 16
  %464 = getelementptr i8, ptr addrspace(3) %452, i32 4608
  %465 = load <4 x i32>, ptr addrspace(3) %464, align 16
  %466 = shufflevector <4 x i32> %463, <4 x i32> %465, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %467 = getelementptr i8, ptr addrspace(3) %452, i32 6144
  %468 = load <4 x i32>, ptr addrspace(3) %467, align 16
  %469 = getelementptr i8, ptr addrspace(3) %452, i32 6656
  %470 = load <4 x i32>, ptr addrspace(3) %469, align 16
  %471 = shufflevector <4 x i32> %468, <4 x i32> %470, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %472 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i32 103952) to i64), %214
  %473 = trunc i64 %472 to i32
  %474 = inttoptr i32 %473 to ptr addrspace(3)
  %475 = load <4 x i32>, ptr addrspace(3) %474, align 16
  %476 = extractelement <4 x i32> %475, i64 0
  %477 = extractelement <4 x i32> %475, i64 1
  %478 = extractelement <4 x i32> %475, i64 2
  %479 = extractelement <4 x i32> %475, i64 3
  %480 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i32 103424) to i64), %210
  %481 = trunc i64 %480 to i32
  %482 = inttoptr i32 %481 to ptr addrspace(3)
  %483 = load <4 x i32>, ptr addrspace(3) %482, align 16
  %484 = extractelement <4 x i32> %483, i64 0
  %485 = extractelement <4 x i32> %483, i64 1
  %486 = extractelement <4 x i32> %483, i64 2
  %487 = extractelement <4 x i32> %483, i64 3
  %488 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i32 53248) to i64), %203
  %489 = trunc i64 %488 to i32
  %490 = inttoptr i32 %489 to ptr addrspace(3)
  %491 = load <4 x i32>, ptr addrspace(3) %490, align 16
  %492 = getelementptr i8, ptr addrspace(3) %490, i32 32
  %493 = load <4 x i32>, ptr addrspace(3) %492, align 16
  %494 = getelementptr i8, ptr addrspace(3) %490, i32 64
  %495 = load <4 x i32>, ptr addrspace(3) %494, align 16
  %496 = getelementptr i8, ptr addrspace(3) %490, i32 96
  %497 = load <4 x i32>, ptr addrspace(3) %496, align 16
  %498 = shufflevector <4 x i32> %491, <4 x i32> %493, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %499 = shufflevector <4 x i32> %495, <4 x i32> %497, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %500 = shufflevector <8 x i32> %498, <8 x i32> %499, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %501 = getelementptr i8, ptr addrspace(3) %490, i32 4352
  %502 = load <4 x i32>, ptr addrspace(3) %501, align 16
  %503 = getelementptr i8, ptr addrspace(3) %490, i32 4384
  %504 = load <4 x i32>, ptr addrspace(3) %503, align 16
  %505 = getelementptr i8, ptr addrspace(3) %490, i32 4416
  %506 = load <4 x i32>, ptr addrspace(3) %505, align 16
  %507 = getelementptr i8, ptr addrspace(3) %490, i32 4448
  %508 = load <4 x i32>, ptr addrspace(3) %507, align 16
  %509 = shufflevector <4 x i32> %502, <4 x i32> %504, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %510 = shufflevector <4 x i32> %506, <4 x i32> %508, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %511 = shufflevector <8 x i32> %509, <8 x i32> %510, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %512 = getelementptr i8, ptr addrspace(3) %452, i32 1024
  %513 = load <4 x i32>, ptr addrspace(3) %512, align 16
  %514 = getelementptr i8, ptr addrspace(3) %452, i32 1536
  %515 = load <4 x i32>, ptr addrspace(3) %514, align 16
  %516 = shufflevector <4 x i32> %513, <4 x i32> %515, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %517 = getelementptr i8, ptr addrspace(3) %452, i32 3072
  %518 = load <4 x i32>, ptr addrspace(3) %517, align 16
  %519 = getelementptr i8, ptr addrspace(3) %452, i32 3584
  %520 = load <4 x i32>, ptr addrspace(3) %519, align 16
  %521 = shufflevector <4 x i32> %518, <4 x i32> %520, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %522 = getelementptr i8, ptr addrspace(3) %452, i32 5120
  %523 = load <4 x i32>, ptr addrspace(3) %522, align 16
  %524 = getelementptr i8, ptr addrspace(3) %452, i32 5632
  %525 = load <4 x i32>, ptr addrspace(3) %524, align 16
  %526 = shufflevector <4 x i32> %523, <4 x i32> %525, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %527 = getelementptr i8, ptr addrspace(3) %452, i32 7168
  %528 = load <4 x i32>, ptr addrspace(3) %527, align 16
  %529 = getelementptr i8, ptr addrspace(3) %452, i32 7680
  %530 = load <4 x i32>, ptr addrspace(3) %529, align 16
  %531 = shufflevector <4 x i32> %528, <4 x i32> %530, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %532 = getelementptr i8, ptr addrspace(3) %474, i32 16
  %533 = load <4 x i32>, ptr addrspace(3) %532, align 16
  %534 = extractelement <4 x i32> %533, i64 0
  %535 = extractelement <4 x i32> %533, i64 1
  %536 = extractelement <4 x i32> %533, i64 2
  %537 = extractelement <4 x i32> %533, i64 3
  %538 = getelementptr i8, ptr addrspace(3) %482, i32 16
  %539 = load <4 x i32>, ptr addrspace(3) %538, align 16
  %540 = extractelement <4 x i32> %539, i64 0
  %541 = extractelement <4 x i32> %539, i64 1
  %542 = extractelement <4 x i32> %539, i64 2
  %543 = extractelement <4 x i32> %539, i64 3
  call void @llvm.amdgcn.s.wait.dscnt(i16 10)
  %544 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %456, i32 0, <16 x i32> %500, i16 0, <8 x float> %412, i32 0, i32 0, i32 %476, i32 0, i32 0, i32 %484, i1 false, i1 false)
  %545 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %461, i32 0, <16 x i32> %500, i16 0, <8 x float> %413, i32 0, i32 0, i32 %477, i32 0, i32 0, i32 %484, i1 false, i1 false)
  %546 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %466, i32 0, <16 x i32> %500, i16 0, <8 x float> %414, i32 0, i32 0, i32 %478, i32 0, i32 0, i32 %484, i1 false, i1 false)
  %547 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %471, i32 0, <16 x i32> %500, i16 0, <8 x float> %415, i32 0, i32 0, i32 %479, i32 0, i32 0, i32 %484, i1 false, i1 false)
  %548 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %471, i32 0, <16 x i32> %511, i16 0, <8 x float> %416, i32 0, i32 0, i32 %479, i32 0, i32 0, i32 %485, i1 false, i1 false)
  %549 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %466, i32 0, <16 x i32> %511, i16 0, <8 x float> %417, i32 0, i32 0, i32 %478, i32 0, i32 0, i32 %485, i1 false, i1 false)
  %550 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %461, i32 0, <16 x i32> %511, i16 0, <8 x float> %418, i32 0, i32 0, i32 %477, i32 0, i32 0, i32 %485, i1 false, i1 false)
  %551 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %456, i32 0, <16 x i32> %511, i16 0, <8 x float> %419, i32 0, i32 0, i32 %476, i32 0, i32 0, i32 %485, i1 false, i1 false)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %552 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %80, i64 1
  %553 = insertelement <4 x i32> %552, i32 %329, i64 2
  %554 = insertelement <4 x i32> %553, i32 %332, i64 3
  %555 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %86, i64 1
  %556 = insertelement <4 x i32> %555, i32 %333, i64 2
  %557 = insertelement <4 x i32> %556, i32 %336, i64 3
  %558 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %91, i64 1
  %559 = insertelement <4 x i32> %558, i32 %337, i64 2
  %560 = insertelement <4 x i32> %559, i32 %340, i64 3
  %561 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %95, i64 1
  %562 = insertelement <4 x i32> %561, i32 %341, i64 2
  %563 = insertelement <4 x i32> %562, i32 %344, i64 3
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %554, <8 x i32> <i32 122683392, i32 16777216, i32 1048576, i32 16777216, i32 16, i32 3072, i32 0, i32 0>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %557, <8 x i32> <i32 0, i32 134217728, i32 262144, i32 134217728, i32 4, i32 24576, i32 0, i32 0>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %560, <8 x i32> <i32 0, i32 2097152, i32 262144, i32 2097152, i32 4, i32 384, i32 0, i32 0>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %563, <8 x i32> <i32 0, i32 2097152, i32 1048576, i32 2097152, i32 16, i32 384, i32 0, i32 0>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  %564 = add i32 %190, 512
  %565 = icmp ult i32 %564, %329
  %566 = add i32 %332, 1
  %567 = select i1 %565, i32 %566, i32 %332
  %568 = add i32 %192, 4096
  %569 = icmp ult i32 %568, %333
  %570 = add i32 %336, 1
  %571 = select i1 %569, i32 %570, i32 %336
  %572 = add i32 %194, 64
  %573 = icmp ult i32 %572, %337
  %574 = add i32 %340, 1
  %575 = select i1 %573, i32 %574, i32 %340
  %576 = add i32 %196, 64
  %577 = icmp ult i32 %576, %341
  %578 = add i32 %344, 1
  %579 = select i1 %577, i32 %578, i32 %344
  %580 = add i64 %200, 1024
  %581 = udiv i64 %580, 2
  %582 = add i64 %349, %580
  %583 = add i64 %104, %582
  %584 = inttoptr i64 %583 to ptr addrspace(1)
  call void @llvm.amdgcn.global.prefetch(ptr addrspace(1) %584, i32 8)
  %585 = mul i64 %581, 16
  %586 = add i64 %356, %585
  %587 = add i64 %115, %586
  %588 = inttoptr i64 %587 to ptr addrspace(1)
  call void @llvm.amdgcn.global.prefetch(ptr addrspace(1) %588, i32 8)
  %589 = getelementptr i8, ptr addrspace(3) %490, i32 8704
  %590 = load <4 x i32>, ptr addrspace(3) %589, align 16
  %591 = getelementptr i8, ptr addrspace(3) %490, i32 8736
  %592 = load <4 x i32>, ptr addrspace(3) %591, align 16
  %593 = getelementptr i8, ptr addrspace(3) %490, i32 8768
  %594 = load <4 x i32>, ptr addrspace(3) %593, align 16
  %595 = getelementptr i8, ptr addrspace(3) %490, i32 8800
  %596 = load <4 x i32>, ptr addrspace(3) %595, align 16
  %597 = shufflevector <4 x i32> %590, <4 x i32> %592, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %598 = shufflevector <4 x i32> %594, <4 x i32> %596, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %599 = shufflevector <8 x i32> %597, <8 x i32> %598, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %600 = getelementptr i8, ptr addrspace(3) %490, i32 13056
  %601 = load <4 x i32>, ptr addrspace(3) %600, align 16
  %602 = getelementptr i8, ptr addrspace(3) %490, i32 13088
  %603 = load <4 x i32>, ptr addrspace(3) %602, align 16
  %604 = getelementptr i8, ptr addrspace(3) %490, i32 13120
  %605 = load <4 x i32>, ptr addrspace(3) %604, align 16
  %606 = getelementptr i8, ptr addrspace(3) %490, i32 13152
  %607 = load <4 x i32>, ptr addrspace(3) %606, align 16
  %608 = shufflevector <4 x i32> %601, <4 x i32> %603, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %609 = shufflevector <4 x i32> %605, <4 x i32> %607, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %610 = shufflevector <8 x i32> %608, <8 x i32> %609, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 10)
  %611 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %456, i32 0, <16 x i32> %599, i16 0, <8 x float> %442, i32 0, i32 0, i32 %476, i32 0, i32 0, i32 %486, i1 false, i1 false)
  %612 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %461, i32 0, <16 x i32> %599, i16 0, <8 x float> %443, i32 0, i32 0, i32 %477, i32 0, i32 0, i32 %486, i1 false, i1 false)
  %613 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %466, i32 0, <16 x i32> %599, i16 0, <8 x float> %444, i32 0, i32 0, i32 %478, i32 0, i32 0, i32 %486, i1 false, i1 false)
  %614 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %471, i32 0, <16 x i32> %599, i16 0, <8 x float> %445, i32 0, i32 0, i32 %479, i32 0, i32 0, i32 %486, i1 false, i1 false)
  %615 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %471, i32 0, <16 x i32> %610, i16 0, <8 x float> %446, i32 0, i32 0, i32 %479, i32 0, i32 0, i32 %487, i1 false, i1 false)
  %616 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %466, i32 0, <16 x i32> %610, i16 0, <8 x float> %447, i32 0, i32 0, i32 %478, i32 0, i32 0, i32 %487, i1 false, i1 false)
  %617 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %461, i32 0, <16 x i32> %610, i16 0, <8 x float> %448, i32 0, i32 0, i32 %477, i32 0, i32 0, i32 %487, i1 false, i1 false)
  %618 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %456, i32 0, <16 x i32> %610, i16 0, <8 x float> %449, i32 0, i32 0, i32 %476, i32 0, i32 0, i32 %487, i1 false, i1 false)
  %619 = getelementptr i8, ptr addrspace(3) %490, i32 128
  %620 = load <4 x i32>, ptr addrspace(3) %619, align 16
  %621 = getelementptr i8, ptr addrspace(3) %490, i32 160
  %622 = load <4 x i32>, ptr addrspace(3) %621, align 16
  %623 = getelementptr i8, ptr addrspace(3) %490, i32 192
  %624 = load <4 x i32>, ptr addrspace(3) %623, align 16
  %625 = getelementptr i8, ptr addrspace(3) %490, i32 224
  %626 = load <4 x i32>, ptr addrspace(3) %625, align 16
  %627 = shufflevector <4 x i32> %620, <4 x i32> %622, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %628 = shufflevector <4 x i32> %624, <4 x i32> %626, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %629 = shufflevector <8 x i32> %627, <8 x i32> %628, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %630 = getelementptr i8, ptr addrspace(3) %490, i32 4480
  %631 = load <4 x i32>, ptr addrspace(3) %630, align 16
  %632 = getelementptr i8, ptr addrspace(3) %490, i32 4512
  %633 = load <4 x i32>, ptr addrspace(3) %632, align 16
  %634 = getelementptr i8, ptr addrspace(3) %490, i32 4544
  %635 = load <4 x i32>, ptr addrspace(3) %634, align 16
  %636 = getelementptr i8, ptr addrspace(3) %490, i32 4576
  %637 = load <4 x i32>, ptr addrspace(3) %636, align 16
  %638 = shufflevector <4 x i32> %631, <4 x i32> %633, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %639 = shufflevector <4 x i32> %635, <4 x i32> %637, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %640 = shufflevector <8 x i32> %638, <8 x i32> %639, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %641 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %516, i32 0, <16 x i32> %629, i16 0, <8 x float> %544, i32 0, i32 0, i32 %534, i32 0, i32 0, i32 %540, i1 false, i1 false)
  %642 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %521, i32 0, <16 x i32> %629, i16 0, <8 x float> %545, i32 0, i32 0, i32 %535, i32 0, i32 0, i32 %540, i1 false, i1 false)
  %643 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %526, i32 0, <16 x i32> %629, i16 0, <8 x float> %546, i32 0, i32 0, i32 %536, i32 0, i32 0, i32 %540, i1 false, i1 false)
  %644 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %531, i32 0, <16 x i32> %629, i16 0, <8 x float> %547, i32 0, i32 0, i32 %537, i32 0, i32 0, i32 %540, i1 false, i1 false)
  %645 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %531, i32 0, <16 x i32> %640, i16 0, <8 x float> %548, i32 0, i32 0, i32 %537, i32 0, i32 0, i32 %541, i1 false, i1 false)
  %646 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %526, i32 0, <16 x i32> %640, i16 0, <8 x float> %549, i32 0, i32 0, i32 %536, i32 0, i32 0, i32 %541, i1 false, i1 false)
  %647 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %521, i32 0, <16 x i32> %640, i16 0, <8 x float> %550, i32 0, i32 0, i32 %535, i32 0, i32 0, i32 %541, i1 false, i1 false)
  %648 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %516, i32 0, <16 x i32> %640, i16 0, <8 x float> %551, i32 0, i32 0, i32 %534, i32 0, i32 0, i32 %541, i1 false, i1 false)
  %649 = getelementptr i8, ptr addrspace(3) %490, i32 8832
  %650 = load <4 x i32>, ptr addrspace(3) %649, align 16
  %651 = getelementptr i8, ptr addrspace(3) %490, i32 8864
  %652 = load <4 x i32>, ptr addrspace(3) %651, align 16
  %653 = getelementptr i8, ptr addrspace(3) %490, i32 8896
  %654 = load <4 x i32>, ptr addrspace(3) %653, align 16
  %655 = getelementptr i8, ptr addrspace(3) %490, i32 8928
  %656 = load <4 x i32>, ptr addrspace(3) %655, align 16
  %657 = shufflevector <4 x i32> %650, <4 x i32> %652, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %658 = shufflevector <4 x i32> %654, <4 x i32> %656, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %659 = shufflevector <8 x i32> %657, <8 x i32> %658, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %660 = getelementptr i8, ptr addrspace(3) %490, i32 13184
  %661 = load <4 x i32>, ptr addrspace(3) %660, align 16
  %662 = getelementptr i8, ptr addrspace(3) %490, i32 13216
  %663 = load <4 x i32>, ptr addrspace(3) %662, align 16
  %664 = getelementptr i8, ptr addrspace(3) %490, i32 13248
  %665 = load <4 x i32>, ptr addrspace(3) %664, align 16
  %666 = getelementptr i8, ptr addrspace(3) %490, i32 13280
  %667 = load <4 x i32>, ptr addrspace(3) %666, align 16
  %668 = shufflevector <4 x i32> %661, <4 x i32> %663, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %669 = shufflevector <4 x i32> %665, <4 x i32> %667, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %670 = shufflevector <8 x i32> %668, <8 x i32> %669, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %671 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %516, i32 0, <16 x i32> %659, i16 0, <8 x float> %611, i32 0, i32 0, i32 %534, i32 0, i32 0, i32 %542, i1 false, i1 false)
  %672 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %521, i32 0, <16 x i32> %659, i16 0, <8 x float> %612, i32 0, i32 0, i32 %535, i32 0, i32 0, i32 %542, i1 false, i1 false)
  %673 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %526, i32 0, <16 x i32> %659, i16 0, <8 x float> %613, i32 0, i32 0, i32 %536, i32 0, i32 0, i32 %542, i1 false, i1 false)
  %674 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %531, i32 0, <16 x i32> %659, i16 0, <8 x float> %614, i32 0, i32 0, i32 %537, i32 0, i32 0, i32 %542, i1 false, i1 false)
  %675 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %531, i32 0, <16 x i32> %670, i16 0, <8 x float> %615, i32 0, i32 0, i32 %537, i32 0, i32 0, i32 %543, i1 false, i1 false)
  %676 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %526, i32 0, <16 x i32> %670, i16 0, <8 x float> %616, i32 0, i32 0, i32 %536, i32 0, i32 0, i32 %543, i1 false, i1 false)
  %677 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %521, i32 0, <16 x i32> %670, i16 0, <8 x float> %617, i32 0, i32 0, i32 %535, i32 0, i32 0, i32 %543, i1 false, i1 false)
  %678 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %516, i32 0, <16 x i32> %670, i16 0, <8 x float> %618, i32 0, i32 0, i32 %534, i32 0, i32 0, i32 %543, i1 false, i1 false)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 18, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 10, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 8, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %679 = add i64 %173, 1
  br label %172

680:                                              ; preds = %172
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %681 = mul i64 %48, 272
  %682 = mul i64 %47, 16
  %683 = add i64 %681, %682
  %684 = mul i64 %48, 16
  %685 = mul i64 %47, 256
  %686 = add i64 %55, %684
  %687 = add i64 %686, %685
  %688 = mul i64 %48, 32
  %689 = mul i64 %47, 4
  %690 = add i64 %688, %689
  %691 = udiv i64 %49, 4
  %692 = add i64 %691, %48
  %693 = mul i64 %692, 32
  %694 = add i64 %693, %689
  %695 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i32 17408) to i64), %687
  %696 = trunc i64 %695 to i32
  %697 = inttoptr i32 %696 to ptr addrspace(3)
  %698 = load <4 x i32>, ptr addrspace(3) %697, align 16
  %699 = getelementptr i8, ptr addrspace(3) %697, i32 512
  %700 = load <4 x i32>, ptr addrspace(3) %699, align 16
  %701 = shufflevector <4 x i32> %698, <4 x i32> %700, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %702 = getelementptr i8, ptr addrspace(3) %697, i32 2048
  %703 = load <4 x i32>, ptr addrspace(3) %702, align 16
  %704 = getelementptr i8, ptr addrspace(3) %697, i32 2560
  %705 = load <4 x i32>, ptr addrspace(3) %704, align 16
  %706 = shufflevector <4 x i32> %703, <4 x i32> %705, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %707 = getelementptr i8, ptr addrspace(3) %697, i32 4096
  %708 = load <4 x i32>, ptr addrspace(3) %707, align 16
  %709 = getelementptr i8, ptr addrspace(3) %697, i32 4608
  %710 = load <4 x i32>, ptr addrspace(3) %709, align 16
  %711 = shufflevector <4 x i32> %708, <4 x i32> %710, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %712 = getelementptr i8, ptr addrspace(3) %697, i32 6144
  %713 = load <4 x i32>, ptr addrspace(3) %712, align 16
  %714 = getelementptr i8, ptr addrspace(3) %697, i32 6656
  %715 = load <4 x i32>, ptr addrspace(3) %714, align 16
  %716 = shufflevector <4 x i32> %713, <4 x i32> %715, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %717 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i32 50704) to i64), %694
  %718 = trunc i64 %717 to i32
  %719 = inttoptr i32 %718 to ptr addrspace(3)
  %720 = load <4 x i32>, ptr addrspace(3) %719, align 16
  %721 = extractelement <4 x i32> %720, i64 0
  %722 = extractelement <4 x i32> %720, i64 1
  %723 = extractelement <4 x i32> %720, i64 2
  %724 = extractelement <4 x i32> %720, i64 3
  %725 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i32 50176) to i64), %690
  %726 = trunc i64 %725 to i32
  %727 = inttoptr i32 %726 to ptr addrspace(3)
  %728 = load <4 x i32>, ptr addrspace(3) %727, align 16
  %729 = extractelement <4 x i32> %728, i64 0
  %730 = extractelement <4 x i32> %728, i64 1
  %731 = extractelement <4 x i32> %728, i64 2
  %732 = extractelement <4 x i32> %728, i64 3
  %733 = add i64 ptrtoint (ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena to i64), %683
  %734 = trunc i64 %733 to i32
  %735 = inttoptr i32 %734 to ptr addrspace(3)
  %736 = load <4 x i32>, ptr addrspace(3) %735, align 16
  %737 = getelementptr i8, ptr addrspace(3) %735, i32 32
  %738 = load <4 x i32>, ptr addrspace(3) %737, align 16
  %739 = getelementptr i8, ptr addrspace(3) %735, i32 64
  %740 = load <4 x i32>, ptr addrspace(3) %739, align 16
  %741 = getelementptr i8, ptr addrspace(3) %735, i32 96
  %742 = load <4 x i32>, ptr addrspace(3) %741, align 16
  %743 = shufflevector <4 x i32> %736, <4 x i32> %738, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %744 = shufflevector <4 x i32> %740, <4 x i32> %742, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %745 = shufflevector <8 x i32> %743, <8 x i32> %744, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %746 = getelementptr i8, ptr addrspace(3) %735, i32 4352
  %747 = load <4 x i32>, ptr addrspace(3) %746, align 16
  %748 = getelementptr i8, ptr addrspace(3) %735, i32 4384
  %749 = load <4 x i32>, ptr addrspace(3) %748, align 16
  %750 = getelementptr i8, ptr addrspace(3) %735, i32 4416
  %751 = load <4 x i32>, ptr addrspace(3) %750, align 16
  %752 = getelementptr i8, ptr addrspace(3) %735, i32 4448
  %753 = load <4 x i32>, ptr addrspace(3) %752, align 16
  %754 = shufflevector <4 x i32> %747, <4 x i32> %749, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %755 = shufflevector <4 x i32> %751, <4 x i32> %753, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %756 = shufflevector <8 x i32> %754, <8 x i32> %755, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %757 = getelementptr i8, ptr addrspace(3) %697, i32 1024
  %758 = load <4 x i32>, ptr addrspace(3) %757, align 16
  %759 = getelementptr i8, ptr addrspace(3) %697, i32 1536
  %760 = load <4 x i32>, ptr addrspace(3) %759, align 16
  %761 = shufflevector <4 x i32> %758, <4 x i32> %760, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %762 = getelementptr i8, ptr addrspace(3) %697, i32 3072
  %763 = load <4 x i32>, ptr addrspace(3) %762, align 16
  %764 = getelementptr i8, ptr addrspace(3) %697, i32 3584
  %765 = load <4 x i32>, ptr addrspace(3) %764, align 16
  %766 = shufflevector <4 x i32> %763, <4 x i32> %765, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %767 = getelementptr i8, ptr addrspace(3) %697, i32 5120
  %768 = load <4 x i32>, ptr addrspace(3) %767, align 16
  %769 = getelementptr i8, ptr addrspace(3) %697, i32 5632
  %770 = load <4 x i32>, ptr addrspace(3) %769, align 16
  %771 = shufflevector <4 x i32> %768, <4 x i32> %770, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %772 = getelementptr i8, ptr addrspace(3) %697, i32 7168
  %773 = load <4 x i32>, ptr addrspace(3) %772, align 16
  %774 = getelementptr i8, ptr addrspace(3) %697, i32 7680
  %775 = load <4 x i32>, ptr addrspace(3) %774, align 16
  %776 = shufflevector <4 x i32> %773, <4 x i32> %775, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %777 = getelementptr i8, ptr addrspace(3) %719, i32 16
  %778 = load <4 x i32>, ptr addrspace(3) %777, align 16
  %779 = extractelement <4 x i32> %778, i64 0
  %780 = extractelement <4 x i32> %778, i64 1
  %781 = extractelement <4 x i32> %778, i64 2
  %782 = extractelement <4 x i32> %778, i64 3
  %783 = getelementptr i8, ptr addrspace(3) %727, i32 16
  %784 = load <4 x i32>, ptr addrspace(3) %783, align 16
  %785 = extractelement <4 x i32> %784, i64 0
  %786 = extractelement <4 x i32> %784, i64 1
  %787 = extractelement <4 x i32> %784, i64 2
  %788 = extractelement <4 x i32> %784, i64 3
  call void @llvm.amdgcn.s.wait.dscnt(i16 10)
  %789 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %701, i32 0, <16 x i32> %745, i16 0, <8 x float> %174, i32 0, i32 0, i32 %721, i32 0, i32 0, i32 %729, i1 false, i1 false)
  %790 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %706, i32 0, <16 x i32> %745, i16 0, <8 x float> %175, i32 0, i32 0, i32 %722, i32 0, i32 0, i32 %729, i1 false, i1 false)
  %791 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %711, i32 0, <16 x i32> %745, i16 0, <8 x float> %176, i32 0, i32 0, i32 %723, i32 0, i32 0, i32 %729, i1 false, i1 false)
  %792 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %716, i32 0, <16 x i32> %745, i16 0, <8 x float> %177, i32 0, i32 0, i32 %724, i32 0, i32 0, i32 %729, i1 false, i1 false)
  %793 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %716, i32 0, <16 x i32> %756, i16 0, <8 x float> %181, i32 0, i32 0, i32 %724, i32 0, i32 0, i32 %730, i1 false, i1 false)
  %794 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %711, i32 0, <16 x i32> %756, i16 0, <8 x float> %180, i32 0, i32 0, i32 %723, i32 0, i32 0, i32 %730, i1 false, i1 false)
  %795 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %706, i32 0, <16 x i32> %756, i16 0, <8 x float> %179, i32 0, i32 0, i32 %722, i32 0, i32 0, i32 %730, i1 false, i1 false)
  %796 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %701, i32 0, <16 x i32> %756, i16 0, <8 x float> %178, i32 0, i32 0, i32 %721, i32 0, i32 0, i32 %730, i1 false, i1 false)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %797 = add i64 %76, 2816
  %798 = add i64 %106, %797
  %799 = add i64 %104, %798
  %800 = trunc i64 %799 to i32
  %801 = lshr i64 %799, 32
  %802 = trunc i64 %801 to i32
  %803 = or i32 %802, -2147483648
  %804 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %97, i64 1
  %805 = insertelement <4 x i32> %804, i32 %800, i64 2
  %806 = insertelement <4 x i32> %805, i32 %803, i64 3
  %807 = add i64 %82, 22528
  %808 = add i64 %117, %807
  %809 = add i64 %115, %808
  %810 = trunc i64 %809 to i32
  %811 = lshr i64 %809, 32
  %812 = trunc i64 %811 to i32
  %813 = or i32 %812, -2147483648
  %814 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %99, i64 1
  %815 = insertelement <4 x i32> %814, i32 %810, i64 2
  %816 = insertelement <4 x i32> %815, i32 %813, i64 3
  %817 = add i64 %87, 352
  %818 = add i64 %126, %817
  %819 = add i64 %124, %818
  %820 = trunc i64 %819 to i32
  %821 = lshr i64 %819, 32
  %822 = trunc i64 %821 to i32
  %823 = or i32 %822, -2147483648
  %824 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %101, i64 1
  %825 = insertelement <4 x i32> %824, i32 %820, i64 2
  %826 = insertelement <4 x i32> %825, i32 %823, i64 3
  %827 = add i64 %137, %817
  %828 = add i64 %135, %827
  %829 = trunc i64 %828 to i32
  %830 = lshr i64 %828, 32
  %831 = trunc i64 %830 to i32
  %832 = or i32 %831, -2147483648
  %833 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %103, i64 1
  %834 = insertelement <4 x i32> %833, i32 %829, i64 2
  %835 = insertelement <4 x i32> %834, i32 %832, i64 3
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %806, <8 x i32> <i32 122683392, i32 16777216, i32 1048576, i32 16777216, i32 16, i32 3072, i32 0, i32 0>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %816, <8 x i32> <i32 0, i32 134217728, i32 262144, i32 134217728, i32 4, i32 24576, i32 0, i32 0>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %826, <8 x i32> <i32 0, i32 2097152, i32 262144, i32 2097152, i32 4, i32 384, i32 0, i32 0>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %835, <8 x i32> <i32 0, i32 2097152, i32 1048576, i32 2097152, i32 16, i32 384, i32 0, i32 0>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  %836 = getelementptr i8, ptr addrspace(3) %735, i32 8704
  %837 = load <4 x i32>, ptr addrspace(3) %836, align 16
  %838 = getelementptr i8, ptr addrspace(3) %735, i32 8736
  %839 = load <4 x i32>, ptr addrspace(3) %838, align 16
  %840 = getelementptr i8, ptr addrspace(3) %735, i32 8768
  %841 = load <4 x i32>, ptr addrspace(3) %840, align 16
  %842 = getelementptr i8, ptr addrspace(3) %735, i32 8800
  %843 = load <4 x i32>, ptr addrspace(3) %842, align 16
  %844 = shufflevector <4 x i32> %837, <4 x i32> %839, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %845 = shufflevector <4 x i32> %841, <4 x i32> %843, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %846 = shufflevector <8 x i32> %844, <8 x i32> %845, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %847 = getelementptr i8, ptr addrspace(3) %735, i32 13056
  %848 = load <4 x i32>, ptr addrspace(3) %847, align 16
  %849 = getelementptr i8, ptr addrspace(3) %735, i32 13088
  %850 = load <4 x i32>, ptr addrspace(3) %849, align 16
  %851 = getelementptr i8, ptr addrspace(3) %735, i32 13120
  %852 = load <4 x i32>, ptr addrspace(3) %851, align 16
  %853 = getelementptr i8, ptr addrspace(3) %735, i32 13152
  %854 = load <4 x i32>, ptr addrspace(3) %853, align 16
  %855 = shufflevector <4 x i32> %848, <4 x i32> %850, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %856 = shufflevector <4 x i32> %852, <4 x i32> %854, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %857 = shufflevector <8 x i32> %855, <8 x i32> %856, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 10)
  %858 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %701, i32 0, <16 x i32> %846, i16 0, <8 x float> %182, i32 0, i32 0, i32 %721, i32 0, i32 0, i32 %731, i1 false, i1 false)
  %859 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %706, i32 0, <16 x i32> %846, i16 0, <8 x float> %183, i32 0, i32 0, i32 %722, i32 0, i32 0, i32 %731, i1 false, i1 false)
  %860 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %711, i32 0, <16 x i32> %846, i16 0, <8 x float> %184, i32 0, i32 0, i32 %723, i32 0, i32 0, i32 %731, i1 false, i1 false)
  %861 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %716, i32 0, <16 x i32> %846, i16 0, <8 x float> %185, i32 0, i32 0, i32 %724, i32 0, i32 0, i32 %731, i1 false, i1 false)
  %862 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %716, i32 0, <16 x i32> %857, i16 0, <8 x float> %189, i32 0, i32 0, i32 %724, i32 0, i32 0, i32 %732, i1 false, i1 false)
  %863 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %711, i32 0, <16 x i32> %857, i16 0, <8 x float> %188, i32 0, i32 0, i32 %723, i32 0, i32 0, i32 %732, i1 false, i1 false)
  %864 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %706, i32 0, <16 x i32> %857, i16 0, <8 x float> %187, i32 0, i32 0, i32 %722, i32 0, i32 0, i32 %732, i1 false, i1 false)
  %865 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %701, i32 0, <16 x i32> %857, i16 0, <8 x float> %186, i32 0, i32 0, i32 %721, i32 0, i32 0, i32 %732, i1 false, i1 false)
  %866 = getelementptr i8, ptr addrspace(3) %735, i32 128
  %867 = load <4 x i32>, ptr addrspace(3) %866, align 16
  %868 = getelementptr i8, ptr addrspace(3) %735, i32 160
  %869 = load <4 x i32>, ptr addrspace(3) %868, align 16
  %870 = getelementptr i8, ptr addrspace(3) %735, i32 192
  %871 = load <4 x i32>, ptr addrspace(3) %870, align 16
  %872 = getelementptr i8, ptr addrspace(3) %735, i32 224
  %873 = load <4 x i32>, ptr addrspace(3) %872, align 16
  %874 = shufflevector <4 x i32> %867, <4 x i32> %869, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %875 = shufflevector <4 x i32> %871, <4 x i32> %873, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %876 = shufflevector <8 x i32> %874, <8 x i32> %875, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %877 = getelementptr i8, ptr addrspace(3) %735, i32 4480
  %878 = load <4 x i32>, ptr addrspace(3) %877, align 16
  %879 = getelementptr i8, ptr addrspace(3) %735, i32 4512
  %880 = load <4 x i32>, ptr addrspace(3) %879, align 16
  %881 = getelementptr i8, ptr addrspace(3) %735, i32 4544
  %882 = load <4 x i32>, ptr addrspace(3) %881, align 16
  %883 = getelementptr i8, ptr addrspace(3) %735, i32 4576
  %884 = load <4 x i32>, ptr addrspace(3) %883, align 16
  %885 = shufflevector <4 x i32> %878, <4 x i32> %880, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %886 = shufflevector <4 x i32> %882, <4 x i32> %884, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %887 = shufflevector <8 x i32> %885, <8 x i32> %886, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %888 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %761, i32 0, <16 x i32> %876, i16 0, <8 x float> %789, i32 0, i32 0, i32 %779, i32 0, i32 0, i32 %785, i1 false, i1 false)
  %889 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %766, i32 0, <16 x i32> %876, i16 0, <8 x float> %790, i32 0, i32 0, i32 %780, i32 0, i32 0, i32 %785, i1 false, i1 false)
  %890 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %771, i32 0, <16 x i32> %876, i16 0, <8 x float> %791, i32 0, i32 0, i32 %781, i32 0, i32 0, i32 %785, i1 false, i1 false)
  %891 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %776, i32 0, <16 x i32> %876, i16 0, <8 x float> %792, i32 0, i32 0, i32 %782, i32 0, i32 0, i32 %785, i1 false, i1 false)
  %892 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %776, i32 0, <16 x i32> %887, i16 0, <8 x float> %793, i32 0, i32 0, i32 %782, i32 0, i32 0, i32 %786, i1 false, i1 false)
  %893 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %771, i32 0, <16 x i32> %887, i16 0, <8 x float> %794, i32 0, i32 0, i32 %781, i32 0, i32 0, i32 %786, i1 false, i1 false)
  %894 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %766, i32 0, <16 x i32> %887, i16 0, <8 x float> %795, i32 0, i32 0, i32 %780, i32 0, i32 0, i32 %786, i1 false, i1 false)
  %895 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %761, i32 0, <16 x i32> %887, i16 0, <8 x float> %796, i32 0, i32 0, i32 %779, i32 0, i32 0, i32 %786, i1 false, i1 false)
  %896 = getelementptr i8, ptr addrspace(3) %735, i32 8832
  %897 = load <4 x i32>, ptr addrspace(3) %896, align 16
  %898 = getelementptr i8, ptr addrspace(3) %735, i32 8864
  %899 = load <4 x i32>, ptr addrspace(3) %898, align 16
  %900 = getelementptr i8, ptr addrspace(3) %735, i32 8896
  %901 = load <4 x i32>, ptr addrspace(3) %900, align 16
  %902 = getelementptr i8, ptr addrspace(3) %735, i32 8928
  %903 = load <4 x i32>, ptr addrspace(3) %902, align 16
  %904 = shufflevector <4 x i32> %897, <4 x i32> %899, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %905 = shufflevector <4 x i32> %901, <4 x i32> %903, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %906 = shufflevector <8 x i32> %904, <8 x i32> %905, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %907 = getelementptr i8, ptr addrspace(3) %735, i32 13184
  %908 = load <4 x i32>, ptr addrspace(3) %907, align 16
  %909 = getelementptr i8, ptr addrspace(3) %735, i32 13216
  %910 = load <4 x i32>, ptr addrspace(3) %909, align 16
  %911 = getelementptr i8, ptr addrspace(3) %735, i32 13248
  %912 = load <4 x i32>, ptr addrspace(3) %911, align 16
  %913 = getelementptr i8, ptr addrspace(3) %735, i32 13280
  %914 = load <4 x i32>, ptr addrspace(3) %913, align 16
  %915 = shufflevector <4 x i32> %908, <4 x i32> %910, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %916 = shufflevector <4 x i32> %912, <4 x i32> %914, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %917 = shufflevector <8 x i32> %915, <8 x i32> %916, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %918 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %761, i32 0, <16 x i32> %906, i16 0, <8 x float> %858, i32 0, i32 0, i32 %779, i32 0, i32 0, i32 %787, i1 false, i1 false)
  %919 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %766, i32 0, <16 x i32> %906, i16 0, <8 x float> %859, i32 0, i32 0, i32 %780, i32 0, i32 0, i32 %787, i1 false, i1 false)
  %920 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %771, i32 0, <16 x i32> %906, i16 0, <8 x float> %860, i32 0, i32 0, i32 %781, i32 0, i32 0, i32 %787, i1 false, i1 false)
  %921 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %776, i32 0, <16 x i32> %906, i16 0, <8 x float> %861, i32 0, i32 0, i32 %782, i32 0, i32 0, i32 %787, i1 false, i1 false)
  %922 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %776, i32 0, <16 x i32> %917, i16 0, <8 x float> %862, i32 0, i32 0, i32 %782, i32 0, i32 0, i32 %788, i1 false, i1 false)
  %923 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %771, i32 0, <16 x i32> %917, i16 0, <8 x float> %863, i32 0, i32 0, i32 %781, i32 0, i32 0, i32 %788, i1 false, i1 false)
  %924 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %766, i32 0, <16 x i32> %917, i16 0, <8 x float> %864, i32 0, i32 0, i32 %780, i32 0, i32 0, i32 %788, i1 false, i1 false)
  %925 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %761, i32 0, <16 x i32> %917, i16 0, <8 x float> %865, i32 0, i32 0, i32 %779, i32 0, i32 0, i32 %788, i1 false, i1 false)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 18, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 10, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 8, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 8, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %926 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i32 70656) to i64), %687
  %927 = trunc i64 %926 to i32
  %928 = inttoptr i32 %927 to ptr addrspace(3)
  %929 = load <4 x i32>, ptr addrspace(3) %928, align 16
  %930 = getelementptr i8, ptr addrspace(3) %928, i32 512
  %931 = load <4 x i32>, ptr addrspace(3) %930, align 16
  %932 = shufflevector <4 x i32> %929, <4 x i32> %931, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %933 = getelementptr i8, ptr addrspace(3) %928, i32 2048
  %934 = load <4 x i32>, ptr addrspace(3) %933, align 16
  %935 = getelementptr i8, ptr addrspace(3) %928, i32 2560
  %936 = load <4 x i32>, ptr addrspace(3) %935, align 16
  %937 = shufflevector <4 x i32> %934, <4 x i32> %936, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %938 = getelementptr i8, ptr addrspace(3) %928, i32 4096
  %939 = load <4 x i32>, ptr addrspace(3) %938, align 16
  %940 = getelementptr i8, ptr addrspace(3) %928, i32 4608
  %941 = load <4 x i32>, ptr addrspace(3) %940, align 16
  %942 = shufflevector <4 x i32> %939, <4 x i32> %941, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %943 = getelementptr i8, ptr addrspace(3) %928, i32 6144
  %944 = load <4 x i32>, ptr addrspace(3) %943, align 16
  %945 = getelementptr i8, ptr addrspace(3) %928, i32 6656
  %946 = load <4 x i32>, ptr addrspace(3) %945, align 16
  %947 = shufflevector <4 x i32> %944, <4 x i32> %946, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %948 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i32 103952) to i64), %694
  %949 = trunc i64 %948 to i32
  %950 = inttoptr i32 %949 to ptr addrspace(3)
  %951 = load <4 x i32>, ptr addrspace(3) %950, align 16
  %952 = extractelement <4 x i32> %951, i64 0
  %953 = extractelement <4 x i32> %951, i64 1
  %954 = extractelement <4 x i32> %951, i64 2
  %955 = extractelement <4 x i32> %951, i64 3
  %956 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i32 103424) to i64), %690
  %957 = trunc i64 %956 to i32
  %958 = inttoptr i32 %957 to ptr addrspace(3)
  %959 = load <4 x i32>, ptr addrspace(3) %958, align 16
  %960 = extractelement <4 x i32> %959, i64 0
  %961 = extractelement <4 x i32> %959, i64 1
  %962 = extractelement <4 x i32> %959, i64 2
  %963 = extractelement <4 x i32> %959, i64 3
  %964 = add i64 ptrtoint (ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i32 53248) to i64), %683
  %965 = trunc i64 %964 to i32
  %966 = inttoptr i32 %965 to ptr addrspace(3)
  %967 = load <4 x i32>, ptr addrspace(3) %966, align 16
  %968 = getelementptr i8, ptr addrspace(3) %966, i32 32
  %969 = load <4 x i32>, ptr addrspace(3) %968, align 16
  %970 = getelementptr i8, ptr addrspace(3) %966, i32 64
  %971 = load <4 x i32>, ptr addrspace(3) %970, align 16
  %972 = getelementptr i8, ptr addrspace(3) %966, i32 96
  %973 = load <4 x i32>, ptr addrspace(3) %972, align 16
  %974 = shufflevector <4 x i32> %967, <4 x i32> %969, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %975 = shufflevector <4 x i32> %971, <4 x i32> %973, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %976 = shufflevector <8 x i32> %974, <8 x i32> %975, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %977 = getelementptr i8, ptr addrspace(3) %966, i32 4352
  %978 = load <4 x i32>, ptr addrspace(3) %977, align 16
  %979 = getelementptr i8, ptr addrspace(3) %966, i32 4384
  %980 = load <4 x i32>, ptr addrspace(3) %979, align 16
  %981 = getelementptr i8, ptr addrspace(3) %966, i32 4416
  %982 = load <4 x i32>, ptr addrspace(3) %981, align 16
  %983 = getelementptr i8, ptr addrspace(3) %966, i32 4448
  %984 = load <4 x i32>, ptr addrspace(3) %983, align 16
  %985 = shufflevector <4 x i32> %978, <4 x i32> %980, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %986 = shufflevector <4 x i32> %982, <4 x i32> %984, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %987 = shufflevector <8 x i32> %985, <8 x i32> %986, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %988 = getelementptr i8, ptr addrspace(3) %928, i32 1024
  %989 = load <4 x i32>, ptr addrspace(3) %988, align 16
  %990 = getelementptr i8, ptr addrspace(3) %928, i32 1536
  %991 = load <4 x i32>, ptr addrspace(3) %990, align 16
  %992 = shufflevector <4 x i32> %989, <4 x i32> %991, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %993 = getelementptr i8, ptr addrspace(3) %928, i32 3072
  %994 = load <4 x i32>, ptr addrspace(3) %993, align 16
  %995 = getelementptr i8, ptr addrspace(3) %928, i32 3584
  %996 = load <4 x i32>, ptr addrspace(3) %995, align 16
  %997 = shufflevector <4 x i32> %994, <4 x i32> %996, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %998 = getelementptr i8, ptr addrspace(3) %928, i32 5120
  %999 = load <4 x i32>, ptr addrspace(3) %998, align 16
  %1000 = getelementptr i8, ptr addrspace(3) %928, i32 5632
  %1001 = load <4 x i32>, ptr addrspace(3) %1000, align 16
  %1002 = shufflevector <4 x i32> %999, <4 x i32> %1001, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1003 = getelementptr i8, ptr addrspace(3) %928, i32 7168
  %1004 = load <4 x i32>, ptr addrspace(3) %1003, align 16
  %1005 = getelementptr i8, ptr addrspace(3) %928, i32 7680
  %1006 = load <4 x i32>, ptr addrspace(3) %1005, align 16
  %1007 = shufflevector <4 x i32> %1004, <4 x i32> %1006, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1008 = getelementptr i8, ptr addrspace(3) %950, i32 16
  %1009 = load <4 x i32>, ptr addrspace(3) %1008, align 16
  %1010 = extractelement <4 x i32> %1009, i64 0
  %1011 = extractelement <4 x i32> %1009, i64 1
  %1012 = extractelement <4 x i32> %1009, i64 2
  %1013 = extractelement <4 x i32> %1009, i64 3
  %1014 = getelementptr i8, ptr addrspace(3) %958, i32 16
  %1015 = load <4 x i32>, ptr addrspace(3) %1014, align 16
  %1016 = extractelement <4 x i32> %1015, i64 0
  %1017 = extractelement <4 x i32> %1015, i64 1
  %1018 = extractelement <4 x i32> %1015, i64 2
  %1019 = extractelement <4 x i32> %1015, i64 3
  call void @llvm.amdgcn.s.wait.dscnt(i16 10)
  %1020 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %932, i32 0, <16 x i32> %976, i16 0, <8 x float> %888, i32 0, i32 0, i32 %952, i32 0, i32 0, i32 %960, i1 false, i1 false)
  %1021 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %937, i32 0, <16 x i32> %976, i16 0, <8 x float> %889, i32 0, i32 0, i32 %953, i32 0, i32 0, i32 %960, i1 false, i1 false)
  %1022 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %942, i32 0, <16 x i32> %976, i16 0, <8 x float> %890, i32 0, i32 0, i32 %954, i32 0, i32 0, i32 %960, i1 false, i1 false)
  %1023 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %947, i32 0, <16 x i32> %976, i16 0, <8 x float> %891, i32 0, i32 0, i32 %955, i32 0, i32 0, i32 %960, i1 false, i1 false)
  %1024 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %947, i32 0, <16 x i32> %987, i16 0, <8 x float> %892, i32 0, i32 0, i32 %955, i32 0, i32 0, i32 %961, i1 false, i1 false)
  %1025 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %942, i32 0, <16 x i32> %987, i16 0, <8 x float> %893, i32 0, i32 0, i32 %954, i32 0, i32 0, i32 %961, i1 false, i1 false)
  %1026 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %937, i32 0, <16 x i32> %987, i16 0, <8 x float> %894, i32 0, i32 0, i32 %953, i32 0, i32 0, i32 %961, i1 false, i1 false)
  %1027 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %932, i32 0, <16 x i32> %987, i16 0, <8 x float> %895, i32 0, i32 0, i32 %952, i32 0, i32 0, i32 %961, i1 false, i1 false)
  %1028 = getelementptr i8, ptr addrspace(3) %966, i32 8704
  %1029 = load <4 x i32>, ptr addrspace(3) %1028, align 16
  %1030 = getelementptr i8, ptr addrspace(3) %966, i32 8736
  %1031 = load <4 x i32>, ptr addrspace(3) %1030, align 16
  %1032 = getelementptr i8, ptr addrspace(3) %966, i32 8768
  %1033 = load <4 x i32>, ptr addrspace(3) %1032, align 16
  %1034 = getelementptr i8, ptr addrspace(3) %966, i32 8800
  %1035 = load <4 x i32>, ptr addrspace(3) %1034, align 16
  %1036 = shufflevector <4 x i32> %1029, <4 x i32> %1031, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1037 = shufflevector <4 x i32> %1033, <4 x i32> %1035, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1038 = shufflevector <8 x i32> %1036, <8 x i32> %1037, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1039 = getelementptr i8, ptr addrspace(3) %966, i32 13056
  %1040 = load <4 x i32>, ptr addrspace(3) %1039, align 16
  %1041 = getelementptr i8, ptr addrspace(3) %966, i32 13088
  %1042 = load <4 x i32>, ptr addrspace(3) %1041, align 16
  %1043 = getelementptr i8, ptr addrspace(3) %966, i32 13120
  %1044 = load <4 x i32>, ptr addrspace(3) %1043, align 16
  %1045 = getelementptr i8, ptr addrspace(3) %966, i32 13152
  %1046 = load <4 x i32>, ptr addrspace(3) %1045, align 16
  %1047 = shufflevector <4 x i32> %1040, <4 x i32> %1042, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1048 = shufflevector <4 x i32> %1044, <4 x i32> %1046, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1049 = shufflevector <8 x i32> %1047, <8 x i32> %1048, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 10)
  %1050 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %932, i32 0, <16 x i32> %1038, i16 0, <8 x float> %918, i32 0, i32 0, i32 %952, i32 0, i32 0, i32 %962, i1 false, i1 false)
  %1051 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %937, i32 0, <16 x i32> %1038, i16 0, <8 x float> %919, i32 0, i32 0, i32 %953, i32 0, i32 0, i32 %962, i1 false, i1 false)
  %1052 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %942, i32 0, <16 x i32> %1038, i16 0, <8 x float> %920, i32 0, i32 0, i32 %954, i32 0, i32 0, i32 %962, i1 false, i1 false)
  %1053 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %947, i32 0, <16 x i32> %1038, i16 0, <8 x float> %921, i32 0, i32 0, i32 %955, i32 0, i32 0, i32 %962, i1 false, i1 false)
  %1054 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %947, i32 0, <16 x i32> %1049, i16 0, <8 x float> %922, i32 0, i32 0, i32 %955, i32 0, i32 0, i32 %963, i1 false, i1 false)
  %1055 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %942, i32 0, <16 x i32> %1049, i16 0, <8 x float> %923, i32 0, i32 0, i32 %954, i32 0, i32 0, i32 %963, i1 false, i1 false)
  %1056 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %937, i32 0, <16 x i32> %1049, i16 0, <8 x float> %924, i32 0, i32 0, i32 %953, i32 0, i32 0, i32 %963, i1 false, i1 false)
  %1057 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %932, i32 0, <16 x i32> %1049, i16 0, <8 x float> %925, i32 0, i32 0, i32 %952, i32 0, i32 0, i32 %963, i1 false, i1 false)
  %1058 = getelementptr i8, ptr addrspace(3) %966, i32 128
  %1059 = load <4 x i32>, ptr addrspace(3) %1058, align 16
  %1060 = getelementptr i8, ptr addrspace(3) %966, i32 160
  %1061 = load <4 x i32>, ptr addrspace(3) %1060, align 16
  %1062 = getelementptr i8, ptr addrspace(3) %966, i32 192
  %1063 = load <4 x i32>, ptr addrspace(3) %1062, align 16
  %1064 = getelementptr i8, ptr addrspace(3) %966, i32 224
  %1065 = load <4 x i32>, ptr addrspace(3) %1064, align 16
  %1066 = shufflevector <4 x i32> %1059, <4 x i32> %1061, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1067 = shufflevector <4 x i32> %1063, <4 x i32> %1065, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1068 = shufflevector <8 x i32> %1066, <8 x i32> %1067, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1069 = getelementptr i8, ptr addrspace(3) %966, i32 4480
  %1070 = load <4 x i32>, ptr addrspace(3) %1069, align 16
  %1071 = getelementptr i8, ptr addrspace(3) %966, i32 4512
  %1072 = load <4 x i32>, ptr addrspace(3) %1071, align 16
  %1073 = getelementptr i8, ptr addrspace(3) %966, i32 4544
  %1074 = load <4 x i32>, ptr addrspace(3) %1073, align 16
  %1075 = getelementptr i8, ptr addrspace(3) %966, i32 4576
  %1076 = load <4 x i32>, ptr addrspace(3) %1075, align 16
  %1077 = shufflevector <4 x i32> %1070, <4 x i32> %1072, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1078 = shufflevector <4 x i32> %1074, <4 x i32> %1076, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1079 = shufflevector <8 x i32> %1077, <8 x i32> %1078, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1080 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %992, i32 0, <16 x i32> %1068, i16 0, <8 x float> %1020, i32 0, i32 0, i32 %1010, i32 0, i32 0, i32 %1016, i1 false, i1 false)
  %1081 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %997, i32 0, <16 x i32> %1068, i16 0, <8 x float> %1021, i32 0, i32 0, i32 %1011, i32 0, i32 0, i32 %1016, i1 false, i1 false)
  %1082 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %1002, i32 0, <16 x i32> %1068, i16 0, <8 x float> %1022, i32 0, i32 0, i32 %1012, i32 0, i32 0, i32 %1016, i1 false, i1 false)
  %1083 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %1007, i32 0, <16 x i32> %1068, i16 0, <8 x float> %1023, i32 0, i32 0, i32 %1013, i32 0, i32 0, i32 %1016, i1 false, i1 false)
  %1084 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %1007, i32 0, <16 x i32> %1079, i16 0, <8 x float> %1024, i32 0, i32 0, i32 %1013, i32 0, i32 0, i32 %1017, i1 false, i1 false)
  %1085 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %1002, i32 0, <16 x i32> %1079, i16 0, <8 x float> %1025, i32 0, i32 0, i32 %1012, i32 0, i32 0, i32 %1017, i1 false, i1 false)
  %1086 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %997, i32 0, <16 x i32> %1079, i16 0, <8 x float> %1026, i32 0, i32 0, i32 %1011, i32 0, i32 0, i32 %1017, i1 false, i1 false)
  %1087 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %992, i32 0, <16 x i32> %1079, i16 0, <8 x float> %1027, i32 0, i32 0, i32 %1010, i32 0, i32 0, i32 %1017, i1 false, i1 false)
  %1088 = getelementptr i8, ptr addrspace(3) %966, i32 8832
  %1089 = load <4 x i32>, ptr addrspace(3) %1088, align 16
  %1090 = getelementptr i8, ptr addrspace(3) %966, i32 8864
  %1091 = load <4 x i32>, ptr addrspace(3) %1090, align 16
  %1092 = getelementptr i8, ptr addrspace(3) %966, i32 8896
  %1093 = load <4 x i32>, ptr addrspace(3) %1092, align 16
  %1094 = getelementptr i8, ptr addrspace(3) %966, i32 8928
  %1095 = load <4 x i32>, ptr addrspace(3) %1094, align 16
  %1096 = shufflevector <4 x i32> %1089, <4 x i32> %1091, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1097 = shufflevector <4 x i32> %1093, <4 x i32> %1095, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1098 = shufflevector <8 x i32> %1096, <8 x i32> %1097, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1099 = getelementptr i8, ptr addrspace(3) %966, i32 13184
  %1100 = load <4 x i32>, ptr addrspace(3) %1099, align 16
  %1101 = getelementptr i8, ptr addrspace(3) %966, i32 13216
  %1102 = load <4 x i32>, ptr addrspace(3) %1101, align 16
  %1103 = getelementptr i8, ptr addrspace(3) %966, i32 13248
  %1104 = load <4 x i32>, ptr addrspace(3) %1103, align 16
  %1105 = getelementptr i8, ptr addrspace(3) %966, i32 13280
  %1106 = load <4 x i32>, ptr addrspace(3) %1105, align 16
  %1107 = shufflevector <4 x i32> %1100, <4 x i32> %1102, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1108 = shufflevector <4 x i32> %1104, <4 x i32> %1106, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1109 = shufflevector <8 x i32> %1107, <8 x i32> %1108, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1110 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %992, i32 0, <16 x i32> %1098, i16 0, <8 x float> %1050, i32 0, i32 0, i32 %1010, i32 0, i32 0, i32 %1018, i1 false, i1 false)
  %1111 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %997, i32 0, <16 x i32> %1098, i16 0, <8 x float> %1051, i32 0, i32 0, i32 %1011, i32 0, i32 0, i32 %1018, i1 false, i1 false)
  %1112 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %1002, i32 0, <16 x i32> %1098, i16 0, <8 x float> %1052, i32 0, i32 0, i32 %1012, i32 0, i32 0, i32 %1018, i1 false, i1 false)
  %1113 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %1007, i32 0, <16 x i32> %1098, i16 0, <8 x float> %1053, i32 0, i32 0, i32 %1013, i32 0, i32 0, i32 %1018, i1 false, i1 false)
  %1114 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %1007, i32 0, <16 x i32> %1109, i16 0, <8 x float> %1054, i32 0, i32 0, i32 %1013, i32 0, i32 0, i32 %1019, i1 false, i1 false)
  %1115 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %1002, i32 0, <16 x i32> %1109, i16 0, <8 x float> %1055, i32 0, i32 0, i32 %1012, i32 0, i32 0, i32 %1019, i1 false, i1 false)
  %1116 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %997, i32 0, <16 x i32> %1109, i16 0, <8 x float> %1056, i32 0, i32 0, i32 %1011, i32 0, i32 0, i32 %1019, i1 false, i1 false)
  %1117 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 4, <8 x i32> %992, i32 0, <16 x i32> %1109, i16 0, <8 x float> %1057, i32 0, i32 0, i32 %1010, i32 0, i32 0, i32 %1019, i1 false, i1 false)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1118 = fptrunc <8 x float> %1080 to <8 x bfloat>
  %1119 = bitcast <8 x bfloat> %1118 to <8 x half>
  %1120 = getelementptr half, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i64 %54
  store <8 x half> %1119, ptr addrspace(3) %1120, align 2
  %1121 = add i64 %54, 16
  %1122 = fptrunc <8 x float> %1081 to <8 x bfloat>
  %1123 = bitcast <8 x bfloat> %1122 to <8 x half>
  %1124 = getelementptr half, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i64 %1121
  store <8 x half> %1123, ptr addrspace(3) %1124, align 2
  %1125 = add i64 %54, 32
  %1126 = fptrunc <8 x float> %1082 to <8 x bfloat>
  %1127 = bitcast <8 x bfloat> %1126 to <8 x half>
  %1128 = getelementptr half, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i64 %1125
  store <8 x half> %1127, ptr addrspace(3) %1128, align 2
  %1129 = add i64 %54, 48
  %1130 = fptrunc <8 x float> %1083 to <8 x bfloat>
  %1131 = bitcast <8 x bfloat> %1130 to <8 x half>
  %1132 = getelementptr half, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i64 %1129
  store <8 x half> %1131, ptr addrspace(3) %1132, align 2
  %1133 = add i64 %54, 1024
  %1134 = fptrunc <8 x float> %1087 to <8 x bfloat>
  %1135 = bitcast <8 x bfloat> %1134 to <8 x half>
  %1136 = getelementptr half, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i64 %1133
  store <8 x half> %1135, ptr addrspace(3) %1136, align 2
  %1137 = add i64 %54, 1040
  %1138 = fptrunc <8 x float> %1086 to <8 x bfloat>
  %1139 = bitcast <8 x bfloat> %1138 to <8 x half>
  %1140 = getelementptr half, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i64 %1137
  store <8 x half> %1139, ptr addrspace(3) %1140, align 2
  %1141 = add i64 %54, 1056
  %1142 = fptrunc <8 x float> %1085 to <8 x bfloat>
  %1143 = bitcast <8 x bfloat> %1142 to <8 x half>
  %1144 = getelementptr half, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i64 %1141
  store <8 x half> %1143, ptr addrspace(3) %1144, align 2
  %1145 = add i64 %54, 1072
  %1146 = fptrunc <8 x float> %1084 to <8 x bfloat>
  %1147 = bitcast <8 x bfloat> %1146 to <8 x half>
  %1148 = getelementptr half, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i64 %1145
  store <8 x half> %1147, ptr addrspace(3) %1148, align 2
  %1149 = add i64 %54, 2048
  %1150 = fptrunc <8 x float> %1110 to <8 x bfloat>
  %1151 = bitcast <8 x bfloat> %1150 to <8 x half>
  %1152 = getelementptr half, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i64 %1149
  store <8 x half> %1151, ptr addrspace(3) %1152, align 2
  %1153 = add i64 %54, 2064
  %1154 = fptrunc <8 x float> %1111 to <8 x bfloat>
  %1155 = bitcast <8 x bfloat> %1154 to <8 x half>
  %1156 = getelementptr half, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i64 %1153
  store <8 x half> %1155, ptr addrspace(3) %1156, align 2
  %1157 = add i64 %54, 2080
  %1158 = fptrunc <8 x float> %1112 to <8 x bfloat>
  %1159 = bitcast <8 x bfloat> %1158 to <8 x half>
  %1160 = getelementptr half, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i64 %1157
  store <8 x half> %1159, ptr addrspace(3) %1160, align 2
  %1161 = add i64 %54, 2096
  %1162 = fptrunc <8 x float> %1113 to <8 x bfloat>
  %1163 = bitcast <8 x bfloat> %1162 to <8 x half>
  %1164 = getelementptr half, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i64 %1161
  store <8 x half> %1163, ptr addrspace(3) %1164, align 2
  %1165 = add i64 %54, 3072
  %1166 = fptrunc <8 x float> %1117 to <8 x bfloat>
  %1167 = bitcast <8 x bfloat> %1166 to <8 x half>
  %1168 = getelementptr half, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i64 %1165
  store <8 x half> %1167, ptr addrspace(3) %1168, align 2
  %1169 = add i64 %54, 3088
  %1170 = fptrunc <8 x float> %1116 to <8 x bfloat>
  %1171 = bitcast <8 x bfloat> %1170 to <8 x half>
  %1172 = getelementptr half, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i64 %1169
  store <8 x half> %1171, ptr addrspace(3) %1172, align 2
  %1173 = add i64 %54, 3104
  %1174 = fptrunc <8 x float> %1115 to <8 x bfloat>
  %1175 = bitcast <8 x bfloat> %1174 to <8 x half>
  %1176 = getelementptr half, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i64 %1173
  store <8 x half> %1175, ptr addrspace(3) %1176, align 2
  %1177 = add i64 %54, 3120
  %1178 = fptrunc <8 x float> %1114 to <8 x bfloat>
  %1179 = bitcast <8 x bfloat> %1178 to <8 x half>
  %1180 = getelementptr half, ptr addrspace(3) @mxscale_a8w4_64x256x256_1x4_2buf_arena, i64 %1177
  store <8 x half> %1179, ptr addrspace(3) %1180, align 2
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  call void @llvm.amdgcn.tensor.store.from.lds(<4 x i32> %70, <8 x i32> <i32 65536, i32 4194304, i32 4194304, i32 4194304, i32 64, i32 3072, i32 0, i32 0>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  br label %1181

1181:                                             ; preds = %680, %21
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.amdgcn.s.setreg(i32 immarg, i32) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr readnone, i16, i64, i32) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.wave.id() #2

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @llvm.amdgcn.tensor.load.to.lds(<4 x i32>, <8 x i32>, <4 x i32>, <4 x i32>, <8 x i32>, i32 immarg) #5

; Function Attrs: nocallback nofree nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.amdgcn.s.wait.tensorcnt(i16 immarg) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier.signal(i32 immarg) #7

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier.wait(i16 immarg) #7

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #7

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.wait.dscnt(i16 immarg) #8

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v8i32.v16i32(i32 immarg, <8 x i32>, i32 immarg, <16 x i32>, i16 immarg, <8 x float>, i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg, i32, i1 immarg, i1 immarg) #9

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.group.barrier(i32 immarg, i32 immarg, i32 immarg) #7

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @llvm.amdgcn.tensor.store.from.lds(<4 x i32>, <8 x i32>, <4 x i32>, <4 x i32>, <8 x i32>, i32 immarg) #5

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @llvm.amdgcn.global.prefetch(ptr addrspace(1) captures(none), i32 immarg) #10

attributes #0 = { "amdgpu-flat-work-group-size"="128,128" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind willreturn }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #5 = { convergent nocallback nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) }
attributes #6 = { nocallback nofree nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #7 = { convergent nocallback nofree nounwind willreturn }
attributes #8 = { nocallback nofree nounwind willreturn }
attributes #9 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #10 = { nocallback nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 128, i32 1, i32 1}
