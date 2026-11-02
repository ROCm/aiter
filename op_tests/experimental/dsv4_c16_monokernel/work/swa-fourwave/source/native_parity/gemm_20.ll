; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

@__dynamic_shared_0 = external dso_local addrspace(3) global [0 x i8], align 1024

define amdgpu_kernel void @gemm_a8w8_mx128_t16x32x1024_mw1_nw2_nb2_sk1_cm1_cn1(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10) #0 !reqd_work_group_size !1 {
  %12 = sdiv i32 %7, 1024
  %13 = mul i32 %12, 1024
  %14 = icmp ne i32 %7, %13
  %15 = icmp slt i32 %7, 0
  %16 = icmp ne i1 %15, false
  %17 = and i1 %14, %16
  %18 = add i32 %12, -1
  %19 = select i1 %17, i32 %18, i32 %12
  %20 = sext i32 %7 to i64
  %21 = sext i32 %9 to i64
  %22 = sext i32 %10 to i64
  %23 = call range(i32 0, 64) i32 @llvm.amdgcn.workitem.id.x()
  %24 = sext i32 %23 to i64
  %25 = trunc i64 %24 to i32
  %26 = call i32 @llvm.amdgcn.workgroup.id.x()
  %27 = sext i32 %26 to i64
  %28 = trunc i64 %27 to i32
  %29 = call i32 @llvm.amdgcn.workgroup.id.y()
  %30 = sext i32 %29 to i64
  %31 = trunc i64 %30 to i32
  %32 = sdiv i32 %25, 32
  %33 = mul i32 %32, 32
  %34 = icmp ne i32 %25, %33
  %35 = icmp slt i32 %25, 0
  %36 = icmp ne i1 %35, false
  %37 = and i1 %34, %36
  %38 = add i32 %32, -1
  %39 = select i1 %37, i32 %38, i32 %32
  %40 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %39)
  %41 = srem i32 %25, 32
  %42 = srem i32 %41, 16
  %43 = sdiv i32 %41, 16
  %44 = mul i32 %43, 16
  %45 = icmp ne i32 %41, %44
  %46 = icmp slt i32 %41, 0
  %47 = icmp ne i1 %46, false
  %48 = and i1 %45, %47
  %49 = add i32 %43, -1
  %50 = select i1 %48, i32 %49, i32 %43
  %51 = sdiv i32 %40, 2
  %52 = mul i32 %51, 2
  %53 = icmp ne i32 %40, %52
  %54 = icmp slt i32 %40, 0
  %55 = icmp ne i1 %54, false
  %56 = and i1 %53, %55
  %57 = add i32 %51, -1
  %58 = select i1 %56, i32 %57, i32 %51
  %59 = srem i32 %40, 2
  %60 = mul i32 %28, 16
  %61 = mul i32 %31, 32
  %62 = sext i32 %60 to i64
  %63 = sext i32 %61 to i64
  %64 = sub i32 %5, %60
  %65 = sext i32 %8 to i64
  %66 = mul i64 %62, %21
  %67 = sdiv i64 %63, 16
  %68 = mul i64 %67, 16
  %69 = icmp ne i64 %63, %68
  %70 = icmp slt i64 %63, 0
  %71 = icmp ne i1 %70, false
  %72 = and i1 %69, %71
  %73 = add i64 %67, -1
  %74 = select i1 %72, i64 %73, i64 %67
  %75 = mul i64 %20, 16
  %76 = mul i64 %74, %75
  %77 = getelementptr i8, ptr addrspace(1) %1, i64 %66
  %78 = getelementptr i8, ptr addrspace(1) %2, i64 %76
  %79 = sdiv i64 %63, 128
  %80 = mul i64 %79, 128
  %81 = icmp ne i64 %63, %80
  %82 = icmp slt i64 %63, 0
  %83 = icmp ne i1 %82, false
  %84 = and i1 %81, %83
  %85 = add i64 %79, -1
  %86 = select i1 %84, i64 %85, i64 %79
  %87 = sdiv i64 %20, 128
  %88 = mul i64 %87, 128
  %89 = icmp ne i64 %20, %88
  %90 = icmp slt i64 %20, 0
  %91 = icmp ne i1 %90, false
  %92 = and i1 %89, %91
  %93 = add i64 %87, -1
  %94 = select i1 %92, i64 %93, i64 %87
  %95 = mul i64 %86, %94
  %96 = getelementptr i8, ptr addrspace(1) %3, i64 %62
  %97 = getelementptr i8, ptr addrspace(1) %4, i64 %95
  %98 = mul i32 %58, 16
  %99 = mul i32 %59, 16
  %100 = icmp eq i32 %40, 0
  br i1 %100, label %101, label %128

101:                                              ; preds = %11
  %102 = icmp eq i64 %21, -2147483648
  %103 = select i1 %102, i64 1024, i64 %21
  %104 = ptrtoint ptr addrspace(1) %77 to i64
  %105 = trunc i64 %104 to i32
  %106 = lshr i64 %104, 32
  %107 = trunc i64 %106 to i32
  %108 = or i32 %107, -2147483648
  %109 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) @__dynamic_shared_0 to i32), i32 poison, i32 poison>, i32 %105, i64 2
  %110 = insertelement <4 x i32> %109, i32 %108, i64 3
  %111 = call i32 @llvm.smax.i32(i32 %64, i32 0)
  %112 = and i32 %111, 65535
  %113 = shl i32 %112, 16
  %114 = or i32 %113, 32767
  %115 = lshr i32 %111, 16
  %116 = and i32 %115, 65535
  %117 = or i32 %116, 67108864
  %118 = trunc i64 %103 to i32
  %119 = lshr i64 %103, 32
  %120 = trunc i64 %119 to i32
  %121 = and i32 %120, 65535
  %122 = insertelement <8 x i32> <i32 133169152, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %114, i64 2
  %123 = insertelement <8 x i32> %122, i32 %117, i64 3
  %124 = insertelement <8 x i32> %123, i32 16, i64 4
  %125 = insertelement <8 x i32> %124, i32 %118, i64 5
  %126 = insertelement <8 x i32> %125, i32 %121, i64 6
  %127 = insertelement <8 x i32> %126, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %110, <8 x i32> %127, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %128

128:                                              ; preds = %101, %11
  %129 = icmp eq i32 %40, 1
  br i1 %129, label %130, label %147

130:                                              ; preds = %128
  %131 = icmp eq i64 %75, -2147483648
  %132 = select i1 %131, i64 16384, i64 %75
  %133 = ptrtoint ptr addrspace(1) %78 to i64
  %134 = trunc i64 %133 to i32
  %135 = lshr i64 %133, 32
  %136 = trunc i64 %135 to i32
  %137 = or i32 %136, -2147483648
  %138 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 16640) to i32), i32 poison, i32 poison>, i32 %134, i64 2
  %139 = insertelement <4 x i32> %138, i32 %137, i64 3
  %140 = trunc i64 %132 to i32
  %141 = lshr i64 %132, 32
  %142 = trunc i64 %141 to i32
  %143 = and i32 %142, 65535
  %144 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 1073774591, i32 2, i32 poison, i32 poison, i32 poison>, i32 %140, i64 5
  %145 = insertelement <8 x i32> %144, i32 %143, i64 6
  %146 = insertelement <8 x i32> %145, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %139, <8 x i32> %146, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %147

147:                                              ; preds = %130, %128
  br i1 %100, label %148, label %175

148:                                              ; preds = %147
  %149 = icmp eq i64 %65, -2147483648
  %150 = select i1 %149, i64 16, i64 %65
  %151 = ptrtoint ptr addrspace(1) %96 to i64
  %152 = trunc i64 %151 to i32
  %153 = lshr i64 %151, 32
  %154 = trunc i64 %153 to i32
  %155 = or i32 %154, -2147483648
  %156 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 49408) to i32), i32 poison, i32 poison>, i32 %152, i64 2
  %157 = insertelement <4 x i32> %156, i32 %155, i64 3
  %158 = call i32 @llvm.smax.i32(i32 %64, i32 0)
  %159 = and i32 %158, 65535
  %160 = shl i32 %159, 16
  %161 = lshr i32 %158, 16
  %162 = and i32 %161, 65535
  %163 = or i32 %162, -65536
  %164 = trunc i64 %150 to i32
  %165 = lshr i64 %150, 32
  %166 = trunc i64 %165 to i32
  %167 = and i32 %166, 65535
  %168 = insertelement <8 x i32> <i32 2097152, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %160, i64 1
  %169 = insertelement <8 x i32> %168, i32 %163, i64 2
  %170 = insertelement <8 x i32> %169, i32 1081343, i64 3
  %171 = insertelement <8 x i32> %170, i32 8, i64 4
  %172 = insertelement <8 x i32> %171, i32 %164, i64 5
  %173 = insertelement <8 x i32> %172, i32 %167, i64 6
  %174 = insertelement <8 x i32> %173, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %157, <8 x i32> %174, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %175

175:                                              ; preds = %148, %147
  br i1 %129, label %176, label %193

176:                                              ; preds = %175
  %177 = icmp eq i64 %94, -2147483648
  %178 = select i1 %177, i64 8, i64 %94
  %179 = ptrtoint ptr addrspace(1) %97 to i64
  %180 = trunc i64 %179 to i32
  %181 = lshr i64 %179, 32
  %182 = trunc i64 %181 to i32
  %183 = or i32 %182, -2147483648
  %184 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 49536) to i32), i32 poison, i32 poison>, i32 %180, i64 2
  %185 = insertelement <4 x i32> %184, i32 %183, i64 3
  %186 = trunc i64 %178 to i32
  %187 = lshr i64 %178, 32
  %188 = trunc i64 %187 to i32
  %189 = and i32 %188, 65535
  %190 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 557055, i32 1, i32 poison, i32 poison, i32 poison>, i32 %186, i64 5
  %191 = insertelement <8 x i32> %190, i32 %189, i64 6
  %192 = insertelement <8 x i32> %191, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %185, <8 x i32> %192, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %193

193:                                              ; preds = %176, %175
  %194 = sub i32 %19, 1
  br label %195

195:                                              ; preds = %403, %193
  %196 = phi i32 [ %204, %403 ], [ 0, %193 ]
  %197 = phi <8 x float> [ %837, %403 ], [ zeroinitializer, %193 ]
  %198 = icmp slt i32 %196, %194
  br i1 %198, label %199, label %838

199:                                              ; preds = %195
  %200 = srem i32 %196, 2
  %201 = mul i32 %200, 50176
  %202 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %201
  %203 = ptrtoint ptr addrspace(3) %202 to i32
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %204 = add i32 %196, 1
  %205 = sdiv i32 %99, 16
  %206 = mul i32 %205, 16
  %207 = icmp ne i32 %99, %206
  %208 = icmp slt i32 %99, 0
  %209 = icmp ne i1 %208, false
  %210 = and i1 %207, %209
  %211 = add i32 %205, -1
  %212 = select i1 %210, i32 %211, i32 %205
  %213 = mul i32 %212, 16384
  %214 = add i32 %213, 16640
  %215 = mul i32 %50, 256
  %216 = add i32 %214, %215
  %217 = mul i32 %42, 16
  %218 = add i32 %216, %217
  %219 = sext i32 %218 to i64
  %220 = add i32 %203, %218
  %221 = inttoptr i32 %220 to ptr addrspace(3)
  %222 = load <4 x i32>, ptr addrspace(3) %221, align 16
  %223 = add i64 %219, 512
  %224 = trunc i64 %223 to i32
  %225 = add i32 %203, %224
  %226 = inttoptr i32 %225 to ptr addrspace(3)
  %227 = load <4 x i32>, ptr addrspace(3) %226, align 16
  %228 = add i64 %219, 1024
  %229 = trunc i64 %228 to i32
  %230 = add i32 %203, %229
  %231 = inttoptr i32 %230 to ptr addrspace(3)
  %232 = load <4 x i32>, ptr addrspace(3) %231, align 16
  %233 = add i64 %219, 1536
  %234 = trunc i64 %233 to i32
  %235 = add i32 %203, %234
  %236 = inttoptr i32 %235 to ptr addrspace(3)
  %237 = load <4 x i32>, ptr addrspace(3) %236, align 16
  %238 = shufflevector <4 x i32> %222, <4 x i32> %227, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %239 = shufflevector <4 x i32> %232, <4 x i32> %237, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %240 = shufflevector <8 x i32> %238, <8 x i32> %239, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %241 = sdiv i32 %99, 128
  %242 = mul i32 %241, 128
  %243 = icmp ne i32 %99, %242
  %244 = icmp slt i32 %99, 0
  %245 = icmp ne i1 %244, false
  %246 = and i1 %243, %245
  %247 = add i32 %241, -1
  %248 = select i1 %246, i32 %247, i32 %241
  %249 = mul i32 %248, 8
  %250 = add i32 %249, 49536
  %251 = add i32 %201, %250
  %252 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %251
  %253 = load i8, ptr addrspace(3) %252, align 1
  %254 = zext i8 %253 to i32
  %255 = call i32 @llvm.amdgcn.perm(i32 %254, i32 %254, i32 0)
  %256 = add i32 %98, %42
  %257 = add i32 %256, 49408
  %258 = add i32 %201, %257
  %259 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %258
  %260 = load i8, ptr addrspace(3) %259, align 1
  %261 = zext i8 %260 to i32
  %262 = call i32 @llvm.amdgcn.perm(i32 %261, i32 %261, i32 0)
  %263 = mul i32 %256, 1040
  %264 = mul i32 %50, 16
  %265 = add i32 %263, %264
  %266 = sext i32 %265 to i64
  %267 = add i32 %203, %265
  %268 = inttoptr i32 %267 to ptr addrspace(3)
  %269 = load <4 x i32>, ptr addrspace(3) %268, align 16
  %270 = add i64 %266, 32
  %271 = trunc i64 %270 to i32
  %272 = add i32 %203, %271
  %273 = inttoptr i32 %272 to ptr addrspace(3)
  %274 = load <4 x i32>, ptr addrspace(3) %273, align 16
  %275 = add i64 %266, 64
  %276 = trunc i64 %275 to i32
  %277 = add i32 %203, %276
  %278 = inttoptr i32 %277 to ptr addrspace(3)
  %279 = load <4 x i32>, ptr addrspace(3) %278, align 16
  %280 = add i64 %266, 96
  %281 = trunc i64 %280 to i32
  %282 = add i32 %203, %281
  %283 = inttoptr i32 %282 to ptr addrspace(3)
  %284 = load <4 x i32>, ptr addrspace(3) %283, align 16
  %285 = shufflevector <4 x i32> %269, <4 x i32> %274, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %286 = shufflevector <4 x i32> %279, <4 x i32> %284, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %287 = shufflevector <8 x i32> %285, <8 x i32> %286, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %288 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %240, i32 0, <16 x i32> %287, i16 0, <8 x float> %197, i32 0, i32 0, i32 %255, i32 0, i32 0, i32 %262, i1 false, i1 false)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %289 = srem i32 %204, 2
  %290 = mul i32 %289, 50176
  %291 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %290
  %292 = sext i32 %204 to i64
  %293 = mul i64 %292, 1024
  br i1 %100, label %294, label %324

294:                                              ; preds = %199
  %295 = icmp eq i64 %21, -2147483648
  %296 = select i1 %295, i64 1024, i64 %21
  %297 = ptrtoint ptr addrspace(1) %77 to i64
  %298 = add i64 %297, %293
  %299 = ptrtoint ptr addrspace(3) %291 to i32
  %300 = trunc i64 %298 to i32
  %301 = lshr i64 %298, 32
  %302 = trunc i64 %301 to i32
  %303 = or i32 %302, -2147483648
  %304 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %299, i64 1
  %305 = insertelement <4 x i32> %304, i32 %300, i64 2
  %306 = insertelement <4 x i32> %305, i32 %303, i64 3
  %307 = call i32 @llvm.smax.i32(i32 %64, i32 0)
  %308 = and i32 %307, 65535
  %309 = shl i32 %308, 16
  %310 = or i32 %309, 32767
  %311 = lshr i32 %307, 16
  %312 = and i32 %311, 65535
  %313 = or i32 %312, 67108864
  %314 = trunc i64 %296 to i32
  %315 = lshr i64 %296, 32
  %316 = trunc i64 %315 to i32
  %317 = and i32 %316, 65535
  %318 = insertelement <8 x i32> <i32 133169152, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %310, i64 2
  %319 = insertelement <8 x i32> %318, i32 %313, i64 3
  %320 = insertelement <8 x i32> %319, i32 16, i64 4
  %321 = insertelement <8 x i32> %320, i32 %314, i64 5
  %322 = insertelement <8 x i32> %321, i32 %317, i64 6
  %323 = insertelement <8 x i32> %322, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %306, <8 x i32> %323, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %324

324:                                              ; preds = %294, %199
  %325 = getelementptr i8, ptr addrspace(3) %291, i32 16640
  %326 = mul i64 %292, 16384
  br i1 %129, label %327, label %347

327:                                              ; preds = %324
  %328 = icmp eq i64 %75, -2147483648
  %329 = select i1 %328, i64 16384, i64 %75
  %330 = ptrtoint ptr addrspace(1) %78 to i64
  %331 = add i64 %330, %326
  %332 = ptrtoint ptr addrspace(3) %325 to i32
  %333 = trunc i64 %331 to i32
  %334 = lshr i64 %331, 32
  %335 = trunc i64 %334 to i32
  %336 = or i32 %335, -2147483648
  %337 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %332, i64 1
  %338 = insertelement <4 x i32> %337, i32 %333, i64 2
  %339 = insertelement <4 x i32> %338, i32 %336, i64 3
  %340 = trunc i64 %329 to i32
  %341 = lshr i64 %329, 32
  %342 = trunc i64 %341 to i32
  %343 = and i32 %342, 65535
  %344 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 1073774591, i32 2, i32 poison, i32 poison, i32 poison>, i32 %340, i64 5
  %345 = insertelement <8 x i32> %344, i32 %343, i64 6
  %346 = insertelement <8 x i32> %345, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %339, <8 x i32> %346, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %347

347:                                              ; preds = %327, %324
  %348 = getelementptr i8, ptr addrspace(3) %291, i32 49408
  %349 = mul i64 %292, 8
  %350 = mul i64 %349, %65
  br i1 %100, label %351, label %381

351:                                              ; preds = %347
  %352 = icmp eq i64 %65, -2147483648
  %353 = select i1 %352, i64 16, i64 %65
  %354 = ptrtoint ptr addrspace(1) %96 to i64
  %355 = add i64 %354, %350
  %356 = ptrtoint ptr addrspace(3) %348 to i32
  %357 = trunc i64 %355 to i32
  %358 = lshr i64 %355, 32
  %359 = trunc i64 %358 to i32
  %360 = or i32 %359, -2147483648
  %361 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %356, i64 1
  %362 = insertelement <4 x i32> %361, i32 %357, i64 2
  %363 = insertelement <4 x i32> %362, i32 %360, i64 3
  %364 = call i32 @llvm.smax.i32(i32 %64, i32 0)
  %365 = and i32 %364, 65535
  %366 = shl i32 %365, 16
  %367 = lshr i32 %364, 16
  %368 = and i32 %367, 65535
  %369 = or i32 %368, -65536
  %370 = trunc i64 %353 to i32
  %371 = lshr i64 %353, 32
  %372 = trunc i64 %371 to i32
  %373 = and i32 %372, 65535
  %374 = insertelement <8 x i32> <i32 2097152, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %366, i64 1
  %375 = insertelement <8 x i32> %374, i32 %369, i64 2
  %376 = insertelement <8 x i32> %375, i32 1081343, i64 3
  %377 = insertelement <8 x i32> %376, i32 8, i64 4
  %378 = insertelement <8 x i32> %377, i32 %370, i64 5
  %379 = insertelement <8 x i32> %378, i32 %373, i64 6
  %380 = insertelement <8 x i32> %379, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %363, <8 x i32> %380, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %381

381:                                              ; preds = %351, %347
  %382 = getelementptr i8, ptr addrspace(3) %291, i32 49536
  br i1 %129, label %383, label %403

383:                                              ; preds = %381
  %384 = icmp eq i64 %94, -2147483648
  %385 = select i1 %384, i64 8, i64 %94
  %386 = ptrtoint ptr addrspace(1) %97 to i64
  %387 = add i64 %386, %349
  %388 = ptrtoint ptr addrspace(3) %382 to i32
  %389 = trunc i64 %387 to i32
  %390 = lshr i64 %387, 32
  %391 = trunc i64 %390 to i32
  %392 = or i32 %391, -2147483648
  %393 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %388, i64 1
  %394 = insertelement <4 x i32> %393, i32 %389, i64 2
  %395 = insertelement <4 x i32> %394, i32 %392, i64 3
  %396 = trunc i64 %385 to i32
  %397 = lshr i64 %385, 32
  %398 = trunc i64 %397 to i32
  %399 = and i32 %398, 65535
  %400 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 557055, i32 1, i32 poison, i32 poison, i32 poison>, i32 %396, i64 5
  %401 = insertelement <8 x i32> %400, i32 %399, i64 6
  %402 = insertelement <8 x i32> %401, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %395, <8 x i32> %402, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %403

403:                                              ; preds = %383, %381
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %404 = add i32 %213, 18688
  %405 = add i32 %404, %215
  %406 = add i32 %405, %217
  %407 = sext i32 %406 to i64
  %408 = add i32 %203, %406
  %409 = inttoptr i32 %408 to ptr addrspace(3)
  %410 = load <4 x i32>, ptr addrspace(3) %409, align 16
  %411 = add i64 %407, 512
  %412 = trunc i64 %411 to i32
  %413 = add i32 %203, %412
  %414 = inttoptr i32 %413 to ptr addrspace(3)
  %415 = load <4 x i32>, ptr addrspace(3) %414, align 16
  %416 = add i64 %407, 1024
  %417 = trunc i64 %416 to i32
  %418 = add i32 %203, %417
  %419 = inttoptr i32 %418 to ptr addrspace(3)
  %420 = load <4 x i32>, ptr addrspace(3) %419, align 16
  %421 = add i64 %407, 1536
  %422 = trunc i64 %421 to i32
  %423 = add i32 %203, %422
  %424 = inttoptr i32 %423 to ptr addrspace(3)
  %425 = load <4 x i32>, ptr addrspace(3) %424, align 16
  %426 = shufflevector <4 x i32> %410, <4 x i32> %415, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %427 = shufflevector <4 x i32> %420, <4 x i32> %425, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %428 = shufflevector <8 x i32> %426, <8 x i32> %427, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %429 = add i32 %249, 49537
  %430 = add i32 %201, %429
  %431 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %430
  %432 = load i8, ptr addrspace(3) %431, align 1
  %433 = zext i8 %432 to i32
  %434 = call i32 @llvm.amdgcn.perm(i32 %433, i32 %433, i32 0)
  %435 = add i32 %256, 49424
  %436 = add i32 %201, %435
  %437 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %436
  %438 = load i8, ptr addrspace(3) %437, align 1
  %439 = zext i8 %438 to i32
  %440 = call i32 @llvm.amdgcn.perm(i32 %439, i32 %439, i32 0)
  %441 = add i32 %263, 128
  %442 = add i32 %441, %264
  %443 = sext i32 %442 to i64
  %444 = add i32 %203, %442
  %445 = inttoptr i32 %444 to ptr addrspace(3)
  %446 = load <4 x i32>, ptr addrspace(3) %445, align 16
  %447 = add i64 %443, 32
  %448 = trunc i64 %447 to i32
  %449 = add i32 %203, %448
  %450 = inttoptr i32 %449 to ptr addrspace(3)
  %451 = load <4 x i32>, ptr addrspace(3) %450, align 16
  %452 = add i64 %443, 64
  %453 = trunc i64 %452 to i32
  %454 = add i32 %203, %453
  %455 = inttoptr i32 %454 to ptr addrspace(3)
  %456 = load <4 x i32>, ptr addrspace(3) %455, align 16
  %457 = add i64 %443, 96
  %458 = trunc i64 %457 to i32
  %459 = add i32 %203, %458
  %460 = inttoptr i32 %459 to ptr addrspace(3)
  %461 = load <4 x i32>, ptr addrspace(3) %460, align 16
  %462 = shufflevector <4 x i32> %446, <4 x i32> %451, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %463 = shufflevector <4 x i32> %456, <4 x i32> %461, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %464 = shufflevector <8 x i32> %462, <8 x i32> %463, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %465 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %428, i32 0, <16 x i32> %464, i16 0, <8 x float> %288, i32 0, i32 0, i32 %434, i32 0, i32 0, i32 %440, i1 false, i1 false)
  %466 = add i32 %213, 20736
  %467 = add i32 %466, %215
  %468 = add i32 %467, %217
  %469 = sext i32 %468 to i64
  %470 = add i32 %203, %468
  %471 = inttoptr i32 %470 to ptr addrspace(3)
  %472 = load <4 x i32>, ptr addrspace(3) %471, align 16
  %473 = add i64 %469, 512
  %474 = trunc i64 %473 to i32
  %475 = add i32 %203, %474
  %476 = inttoptr i32 %475 to ptr addrspace(3)
  %477 = load <4 x i32>, ptr addrspace(3) %476, align 16
  %478 = add i64 %469, 1024
  %479 = trunc i64 %478 to i32
  %480 = add i32 %203, %479
  %481 = inttoptr i32 %480 to ptr addrspace(3)
  %482 = load <4 x i32>, ptr addrspace(3) %481, align 16
  %483 = add i64 %469, 1536
  %484 = trunc i64 %483 to i32
  %485 = add i32 %203, %484
  %486 = inttoptr i32 %485 to ptr addrspace(3)
  %487 = load <4 x i32>, ptr addrspace(3) %486, align 16
  %488 = shufflevector <4 x i32> %472, <4 x i32> %477, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %489 = shufflevector <4 x i32> %482, <4 x i32> %487, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %490 = shufflevector <8 x i32> %488, <8 x i32> %489, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %491 = add i32 %249, 49538
  %492 = add i32 %201, %491
  %493 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %492
  %494 = load i8, ptr addrspace(3) %493, align 1
  %495 = zext i8 %494 to i32
  %496 = call i32 @llvm.amdgcn.perm(i32 %495, i32 %495, i32 0)
  %497 = add i32 %256, 49440
  %498 = add i32 %201, %497
  %499 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %498
  %500 = load i8, ptr addrspace(3) %499, align 1
  %501 = zext i8 %500 to i32
  %502 = call i32 @llvm.amdgcn.perm(i32 %501, i32 %501, i32 0)
  %503 = add i32 %263, 256
  %504 = add i32 %503, %264
  %505 = sext i32 %504 to i64
  %506 = add i32 %203, %504
  %507 = inttoptr i32 %506 to ptr addrspace(3)
  %508 = load <4 x i32>, ptr addrspace(3) %507, align 16
  %509 = add i64 %505, 32
  %510 = trunc i64 %509 to i32
  %511 = add i32 %203, %510
  %512 = inttoptr i32 %511 to ptr addrspace(3)
  %513 = load <4 x i32>, ptr addrspace(3) %512, align 16
  %514 = add i64 %505, 64
  %515 = trunc i64 %514 to i32
  %516 = add i32 %203, %515
  %517 = inttoptr i32 %516 to ptr addrspace(3)
  %518 = load <4 x i32>, ptr addrspace(3) %517, align 16
  %519 = add i64 %505, 96
  %520 = trunc i64 %519 to i32
  %521 = add i32 %203, %520
  %522 = inttoptr i32 %521 to ptr addrspace(3)
  %523 = load <4 x i32>, ptr addrspace(3) %522, align 16
  %524 = shufflevector <4 x i32> %508, <4 x i32> %513, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %525 = shufflevector <4 x i32> %518, <4 x i32> %523, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %526 = shufflevector <8 x i32> %524, <8 x i32> %525, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %527 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %490, i32 0, <16 x i32> %526, i16 0, <8 x float> %465, i32 0, i32 0, i32 %496, i32 0, i32 0, i32 %502, i1 false, i1 false)
  %528 = add i32 %213, 22784
  %529 = add i32 %528, %215
  %530 = add i32 %529, %217
  %531 = sext i32 %530 to i64
  %532 = add i32 %203, %530
  %533 = inttoptr i32 %532 to ptr addrspace(3)
  %534 = load <4 x i32>, ptr addrspace(3) %533, align 16
  %535 = add i64 %531, 512
  %536 = trunc i64 %535 to i32
  %537 = add i32 %203, %536
  %538 = inttoptr i32 %537 to ptr addrspace(3)
  %539 = load <4 x i32>, ptr addrspace(3) %538, align 16
  %540 = add i64 %531, 1024
  %541 = trunc i64 %540 to i32
  %542 = add i32 %203, %541
  %543 = inttoptr i32 %542 to ptr addrspace(3)
  %544 = load <4 x i32>, ptr addrspace(3) %543, align 16
  %545 = add i64 %531, 1536
  %546 = trunc i64 %545 to i32
  %547 = add i32 %203, %546
  %548 = inttoptr i32 %547 to ptr addrspace(3)
  %549 = load <4 x i32>, ptr addrspace(3) %548, align 16
  %550 = shufflevector <4 x i32> %534, <4 x i32> %539, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %551 = shufflevector <4 x i32> %544, <4 x i32> %549, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %552 = shufflevector <8 x i32> %550, <8 x i32> %551, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %553 = add i32 %249, 49539
  %554 = add i32 %201, %553
  %555 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %554
  %556 = load i8, ptr addrspace(3) %555, align 1
  %557 = zext i8 %556 to i32
  %558 = call i32 @llvm.amdgcn.perm(i32 %557, i32 %557, i32 0)
  %559 = add i32 %256, 49456
  %560 = add i32 %201, %559
  %561 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %560
  %562 = load i8, ptr addrspace(3) %561, align 1
  %563 = zext i8 %562 to i32
  %564 = call i32 @llvm.amdgcn.perm(i32 %563, i32 %563, i32 0)
  %565 = add i32 %263, 384
  %566 = add i32 %565, %264
  %567 = sext i32 %566 to i64
  %568 = add i32 %203, %566
  %569 = inttoptr i32 %568 to ptr addrspace(3)
  %570 = load <4 x i32>, ptr addrspace(3) %569, align 16
  %571 = add i64 %567, 32
  %572 = trunc i64 %571 to i32
  %573 = add i32 %203, %572
  %574 = inttoptr i32 %573 to ptr addrspace(3)
  %575 = load <4 x i32>, ptr addrspace(3) %574, align 16
  %576 = add i64 %567, 64
  %577 = trunc i64 %576 to i32
  %578 = add i32 %203, %577
  %579 = inttoptr i32 %578 to ptr addrspace(3)
  %580 = load <4 x i32>, ptr addrspace(3) %579, align 16
  %581 = add i64 %567, 96
  %582 = trunc i64 %581 to i32
  %583 = add i32 %203, %582
  %584 = inttoptr i32 %583 to ptr addrspace(3)
  %585 = load <4 x i32>, ptr addrspace(3) %584, align 16
  %586 = shufflevector <4 x i32> %570, <4 x i32> %575, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %587 = shufflevector <4 x i32> %580, <4 x i32> %585, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %588 = shufflevector <8 x i32> %586, <8 x i32> %587, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %589 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %552, i32 0, <16 x i32> %588, i16 0, <8 x float> %527, i32 0, i32 0, i32 %558, i32 0, i32 0, i32 %564, i1 false, i1 false)
  %590 = add i32 %213, 24832
  %591 = add i32 %590, %215
  %592 = add i32 %591, %217
  %593 = sext i32 %592 to i64
  %594 = add i32 %203, %592
  %595 = inttoptr i32 %594 to ptr addrspace(3)
  %596 = load <4 x i32>, ptr addrspace(3) %595, align 16
  %597 = add i64 %593, 512
  %598 = trunc i64 %597 to i32
  %599 = add i32 %203, %598
  %600 = inttoptr i32 %599 to ptr addrspace(3)
  %601 = load <4 x i32>, ptr addrspace(3) %600, align 16
  %602 = add i64 %593, 1024
  %603 = trunc i64 %602 to i32
  %604 = add i32 %203, %603
  %605 = inttoptr i32 %604 to ptr addrspace(3)
  %606 = load <4 x i32>, ptr addrspace(3) %605, align 16
  %607 = add i64 %593, 1536
  %608 = trunc i64 %607 to i32
  %609 = add i32 %203, %608
  %610 = inttoptr i32 %609 to ptr addrspace(3)
  %611 = load <4 x i32>, ptr addrspace(3) %610, align 16
  %612 = shufflevector <4 x i32> %596, <4 x i32> %601, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %613 = shufflevector <4 x i32> %606, <4 x i32> %611, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %614 = shufflevector <8 x i32> %612, <8 x i32> %613, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %615 = add i32 %249, 49540
  %616 = add i32 %201, %615
  %617 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %616
  %618 = load i8, ptr addrspace(3) %617, align 1
  %619 = zext i8 %618 to i32
  %620 = call i32 @llvm.amdgcn.perm(i32 %619, i32 %619, i32 0)
  %621 = add i32 %256, 49472
  %622 = add i32 %201, %621
  %623 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %622
  %624 = load i8, ptr addrspace(3) %623, align 1
  %625 = zext i8 %624 to i32
  %626 = call i32 @llvm.amdgcn.perm(i32 %625, i32 %625, i32 0)
  %627 = add i32 %263, 512
  %628 = add i32 %627, %264
  %629 = sext i32 %628 to i64
  %630 = add i32 %203, %628
  %631 = inttoptr i32 %630 to ptr addrspace(3)
  %632 = load <4 x i32>, ptr addrspace(3) %631, align 16
  %633 = add i64 %629, 32
  %634 = trunc i64 %633 to i32
  %635 = add i32 %203, %634
  %636 = inttoptr i32 %635 to ptr addrspace(3)
  %637 = load <4 x i32>, ptr addrspace(3) %636, align 16
  %638 = add i64 %629, 64
  %639 = trunc i64 %638 to i32
  %640 = add i32 %203, %639
  %641 = inttoptr i32 %640 to ptr addrspace(3)
  %642 = load <4 x i32>, ptr addrspace(3) %641, align 16
  %643 = add i64 %629, 96
  %644 = trunc i64 %643 to i32
  %645 = add i32 %203, %644
  %646 = inttoptr i32 %645 to ptr addrspace(3)
  %647 = load <4 x i32>, ptr addrspace(3) %646, align 16
  %648 = shufflevector <4 x i32> %632, <4 x i32> %637, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %649 = shufflevector <4 x i32> %642, <4 x i32> %647, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %650 = shufflevector <8 x i32> %648, <8 x i32> %649, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %651 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %614, i32 0, <16 x i32> %650, i16 0, <8 x float> %589, i32 0, i32 0, i32 %620, i32 0, i32 0, i32 %626, i1 false, i1 false)
  %652 = add i32 %213, 26880
  %653 = add i32 %652, %215
  %654 = add i32 %653, %217
  %655 = sext i32 %654 to i64
  %656 = add i32 %203, %654
  %657 = inttoptr i32 %656 to ptr addrspace(3)
  %658 = load <4 x i32>, ptr addrspace(3) %657, align 16
  %659 = add i64 %655, 512
  %660 = trunc i64 %659 to i32
  %661 = add i32 %203, %660
  %662 = inttoptr i32 %661 to ptr addrspace(3)
  %663 = load <4 x i32>, ptr addrspace(3) %662, align 16
  %664 = add i64 %655, 1024
  %665 = trunc i64 %664 to i32
  %666 = add i32 %203, %665
  %667 = inttoptr i32 %666 to ptr addrspace(3)
  %668 = load <4 x i32>, ptr addrspace(3) %667, align 16
  %669 = add i64 %655, 1536
  %670 = trunc i64 %669 to i32
  %671 = add i32 %203, %670
  %672 = inttoptr i32 %671 to ptr addrspace(3)
  %673 = load <4 x i32>, ptr addrspace(3) %672, align 16
  %674 = shufflevector <4 x i32> %658, <4 x i32> %663, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %675 = shufflevector <4 x i32> %668, <4 x i32> %673, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %676 = shufflevector <8 x i32> %674, <8 x i32> %675, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %677 = add i32 %249, 49541
  %678 = add i32 %201, %677
  %679 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %678
  %680 = load i8, ptr addrspace(3) %679, align 1
  %681 = zext i8 %680 to i32
  %682 = call i32 @llvm.amdgcn.perm(i32 %681, i32 %681, i32 0)
  %683 = add i32 %256, 49488
  %684 = add i32 %201, %683
  %685 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %684
  %686 = load i8, ptr addrspace(3) %685, align 1
  %687 = zext i8 %686 to i32
  %688 = call i32 @llvm.amdgcn.perm(i32 %687, i32 %687, i32 0)
  %689 = add i32 %263, 640
  %690 = add i32 %689, %264
  %691 = sext i32 %690 to i64
  %692 = add i32 %203, %690
  %693 = inttoptr i32 %692 to ptr addrspace(3)
  %694 = load <4 x i32>, ptr addrspace(3) %693, align 16
  %695 = add i64 %691, 32
  %696 = trunc i64 %695 to i32
  %697 = add i32 %203, %696
  %698 = inttoptr i32 %697 to ptr addrspace(3)
  %699 = load <4 x i32>, ptr addrspace(3) %698, align 16
  %700 = add i64 %691, 64
  %701 = trunc i64 %700 to i32
  %702 = add i32 %203, %701
  %703 = inttoptr i32 %702 to ptr addrspace(3)
  %704 = load <4 x i32>, ptr addrspace(3) %703, align 16
  %705 = add i64 %691, 96
  %706 = trunc i64 %705 to i32
  %707 = add i32 %203, %706
  %708 = inttoptr i32 %707 to ptr addrspace(3)
  %709 = load <4 x i32>, ptr addrspace(3) %708, align 16
  %710 = shufflevector <4 x i32> %694, <4 x i32> %699, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %711 = shufflevector <4 x i32> %704, <4 x i32> %709, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %712 = shufflevector <8 x i32> %710, <8 x i32> %711, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %713 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %676, i32 0, <16 x i32> %712, i16 0, <8 x float> %651, i32 0, i32 0, i32 %682, i32 0, i32 0, i32 %688, i1 false, i1 false)
  %714 = add i32 %213, 28928
  %715 = add i32 %714, %215
  %716 = add i32 %715, %217
  %717 = sext i32 %716 to i64
  %718 = add i32 %203, %716
  %719 = inttoptr i32 %718 to ptr addrspace(3)
  %720 = load <4 x i32>, ptr addrspace(3) %719, align 16
  %721 = add i64 %717, 512
  %722 = trunc i64 %721 to i32
  %723 = add i32 %203, %722
  %724 = inttoptr i32 %723 to ptr addrspace(3)
  %725 = load <4 x i32>, ptr addrspace(3) %724, align 16
  %726 = add i64 %717, 1024
  %727 = trunc i64 %726 to i32
  %728 = add i32 %203, %727
  %729 = inttoptr i32 %728 to ptr addrspace(3)
  %730 = load <4 x i32>, ptr addrspace(3) %729, align 16
  %731 = add i64 %717, 1536
  %732 = trunc i64 %731 to i32
  %733 = add i32 %203, %732
  %734 = inttoptr i32 %733 to ptr addrspace(3)
  %735 = load <4 x i32>, ptr addrspace(3) %734, align 16
  %736 = shufflevector <4 x i32> %720, <4 x i32> %725, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %737 = shufflevector <4 x i32> %730, <4 x i32> %735, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %738 = shufflevector <8 x i32> %736, <8 x i32> %737, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %739 = add i32 %249, 49542
  %740 = add i32 %201, %739
  %741 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %740
  %742 = load i8, ptr addrspace(3) %741, align 1
  %743 = zext i8 %742 to i32
  %744 = call i32 @llvm.amdgcn.perm(i32 %743, i32 %743, i32 0)
  %745 = add i32 %256, 49504
  %746 = add i32 %201, %745
  %747 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %746
  %748 = load i8, ptr addrspace(3) %747, align 1
  %749 = zext i8 %748 to i32
  %750 = call i32 @llvm.amdgcn.perm(i32 %749, i32 %749, i32 0)
  %751 = add i32 %263, 768
  %752 = add i32 %751, %264
  %753 = sext i32 %752 to i64
  %754 = add i32 %203, %752
  %755 = inttoptr i32 %754 to ptr addrspace(3)
  %756 = load <4 x i32>, ptr addrspace(3) %755, align 16
  %757 = add i64 %753, 32
  %758 = trunc i64 %757 to i32
  %759 = add i32 %203, %758
  %760 = inttoptr i32 %759 to ptr addrspace(3)
  %761 = load <4 x i32>, ptr addrspace(3) %760, align 16
  %762 = add i64 %753, 64
  %763 = trunc i64 %762 to i32
  %764 = add i32 %203, %763
  %765 = inttoptr i32 %764 to ptr addrspace(3)
  %766 = load <4 x i32>, ptr addrspace(3) %765, align 16
  %767 = add i64 %753, 96
  %768 = trunc i64 %767 to i32
  %769 = add i32 %203, %768
  %770 = inttoptr i32 %769 to ptr addrspace(3)
  %771 = load <4 x i32>, ptr addrspace(3) %770, align 16
  %772 = shufflevector <4 x i32> %756, <4 x i32> %761, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %773 = shufflevector <4 x i32> %766, <4 x i32> %771, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %774 = shufflevector <8 x i32> %772, <8 x i32> %773, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %775 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %738, i32 0, <16 x i32> %774, i16 0, <8 x float> %713, i32 0, i32 0, i32 %744, i32 0, i32 0, i32 %750, i1 false, i1 false)
  %776 = add i32 %213, 30976
  %777 = add i32 %776, %215
  %778 = add i32 %777, %217
  %779 = sext i32 %778 to i64
  %780 = add i32 %203, %778
  %781 = inttoptr i32 %780 to ptr addrspace(3)
  %782 = load <4 x i32>, ptr addrspace(3) %781, align 16
  %783 = add i64 %779, 512
  %784 = trunc i64 %783 to i32
  %785 = add i32 %203, %784
  %786 = inttoptr i32 %785 to ptr addrspace(3)
  %787 = load <4 x i32>, ptr addrspace(3) %786, align 16
  %788 = add i64 %779, 1024
  %789 = trunc i64 %788 to i32
  %790 = add i32 %203, %789
  %791 = inttoptr i32 %790 to ptr addrspace(3)
  %792 = load <4 x i32>, ptr addrspace(3) %791, align 16
  %793 = add i64 %779, 1536
  %794 = trunc i64 %793 to i32
  %795 = add i32 %203, %794
  %796 = inttoptr i32 %795 to ptr addrspace(3)
  %797 = load <4 x i32>, ptr addrspace(3) %796, align 16
  %798 = shufflevector <4 x i32> %782, <4 x i32> %787, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %799 = shufflevector <4 x i32> %792, <4 x i32> %797, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %800 = shufflevector <8 x i32> %798, <8 x i32> %799, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %801 = add i32 %249, 49543
  %802 = add i32 %201, %801
  %803 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %802
  %804 = load i8, ptr addrspace(3) %803, align 1
  %805 = zext i8 %804 to i32
  %806 = call i32 @llvm.amdgcn.perm(i32 %805, i32 %805, i32 0)
  %807 = add i32 %256, 49520
  %808 = add i32 %201, %807
  %809 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %808
  %810 = load i8, ptr addrspace(3) %809, align 1
  %811 = zext i8 %810 to i32
  %812 = call i32 @llvm.amdgcn.perm(i32 %811, i32 %811, i32 0)
  %813 = add i32 %263, 896
  %814 = add i32 %813, %264
  %815 = sext i32 %814 to i64
  %816 = add i32 %203, %814
  %817 = inttoptr i32 %816 to ptr addrspace(3)
  %818 = load <4 x i32>, ptr addrspace(3) %817, align 16
  %819 = add i64 %815, 32
  %820 = trunc i64 %819 to i32
  %821 = add i32 %203, %820
  %822 = inttoptr i32 %821 to ptr addrspace(3)
  %823 = load <4 x i32>, ptr addrspace(3) %822, align 16
  %824 = add i64 %815, 64
  %825 = trunc i64 %824 to i32
  %826 = add i32 %203, %825
  %827 = inttoptr i32 %826 to ptr addrspace(3)
  %828 = load <4 x i32>, ptr addrspace(3) %827, align 16
  %829 = add i64 %815, 96
  %830 = trunc i64 %829 to i32
  %831 = add i32 %203, %830
  %832 = inttoptr i32 %831 to ptr addrspace(3)
  %833 = load <4 x i32>, ptr addrspace(3) %832, align 16
  %834 = shufflevector <4 x i32> %818, <4 x i32> %823, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %835 = shufflevector <4 x i32> %828, <4 x i32> %833, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %836 = shufflevector <8 x i32> %834, <8 x i32> %835, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %837 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %800, i32 0, <16 x i32> %836, i16 0, <8 x float> %775, i32 0, i32 0, i32 %806, i32 0, i32 0, i32 %812, i1 false, i1 false)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 10, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 6, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 4, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 6, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 4, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 6, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 4, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 6, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 4, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 6, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 4, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 6, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 4, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 6, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 4, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  br label %195

838:                                              ; preds = %195
  %839 = srem i32 %194, 2
  %840 = mul i32 %839, 50176
  %841 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %840
  %842 = ptrtoint ptr addrspace(3) %841 to i32
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %843 = sdiv i32 %99, 16
  %844 = mul i32 %843, 16
  %845 = icmp ne i32 %99, %844
  %846 = icmp slt i32 %99, 0
  %847 = icmp ne i1 %846, false
  %848 = and i1 %845, %847
  %849 = add i32 %843, -1
  %850 = select i1 %848, i32 %849, i32 %843
  %851 = mul i32 %850, 16384
  %852 = add i32 %851, 16640
  %853 = mul i32 %50, 256
  %854 = add i32 %852, %853
  %855 = mul i32 %42, 16
  %856 = add i32 %854, %855
  %857 = sext i32 %856 to i64
  %858 = add i32 %842, %856
  %859 = inttoptr i32 %858 to ptr addrspace(3)
  %860 = load <4 x i32>, ptr addrspace(3) %859, align 16
  %861 = add i64 %857, 512
  %862 = trunc i64 %861 to i32
  %863 = add i32 %842, %862
  %864 = inttoptr i32 %863 to ptr addrspace(3)
  %865 = load <4 x i32>, ptr addrspace(3) %864, align 16
  %866 = add i64 %857, 1024
  %867 = trunc i64 %866 to i32
  %868 = add i32 %842, %867
  %869 = inttoptr i32 %868 to ptr addrspace(3)
  %870 = load <4 x i32>, ptr addrspace(3) %869, align 16
  %871 = add i64 %857, 1536
  %872 = trunc i64 %871 to i32
  %873 = add i32 %842, %872
  %874 = inttoptr i32 %873 to ptr addrspace(3)
  %875 = load <4 x i32>, ptr addrspace(3) %874, align 16
  %876 = shufflevector <4 x i32> %860, <4 x i32> %865, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %877 = shufflevector <4 x i32> %870, <4 x i32> %875, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %878 = shufflevector <8 x i32> %876, <8 x i32> %877, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %879 = sdiv i32 %99, 128
  %880 = mul i32 %879, 128
  %881 = icmp ne i32 %99, %880
  %882 = icmp slt i32 %99, 0
  %883 = icmp ne i1 %882, false
  %884 = and i1 %881, %883
  %885 = add i32 %879, -1
  %886 = select i1 %884, i32 %885, i32 %879
  %887 = mul i32 %886, 8
  %888 = add i32 %887, 49536
  %889 = add i32 %840, %888
  %890 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %889
  %891 = load i8, ptr addrspace(3) %890, align 1
  %892 = zext i8 %891 to i32
  %893 = call i32 @llvm.amdgcn.perm(i32 %892, i32 %892, i32 0)
  %894 = add i32 %98, %42
  %895 = add i32 %894, 49408
  %896 = add i32 %840, %895
  %897 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %896
  %898 = load i8, ptr addrspace(3) %897, align 1
  %899 = zext i8 %898 to i32
  %900 = call i32 @llvm.amdgcn.perm(i32 %899, i32 %899, i32 0)
  %901 = mul i32 %894, 1040
  %902 = mul i32 %50, 16
  %903 = add i32 %901, %902
  %904 = sext i32 %903 to i64
  %905 = add i32 %842, %903
  %906 = inttoptr i32 %905 to ptr addrspace(3)
  %907 = load <4 x i32>, ptr addrspace(3) %906, align 16
  %908 = add i64 %904, 32
  %909 = trunc i64 %908 to i32
  %910 = add i32 %842, %909
  %911 = inttoptr i32 %910 to ptr addrspace(3)
  %912 = load <4 x i32>, ptr addrspace(3) %911, align 16
  %913 = add i64 %904, 64
  %914 = trunc i64 %913 to i32
  %915 = add i32 %842, %914
  %916 = inttoptr i32 %915 to ptr addrspace(3)
  %917 = load <4 x i32>, ptr addrspace(3) %916, align 16
  %918 = add i64 %904, 96
  %919 = trunc i64 %918 to i32
  %920 = add i32 %842, %919
  %921 = inttoptr i32 %920 to ptr addrspace(3)
  %922 = load <4 x i32>, ptr addrspace(3) %921, align 16
  %923 = shufflevector <4 x i32> %907, <4 x i32> %912, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %924 = shufflevector <4 x i32> %917, <4 x i32> %922, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %925 = shufflevector <8 x i32> %923, <8 x i32> %924, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %926 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %878, i32 0, <16 x i32> %925, i16 0, <8 x float> %197, i32 0, i32 0, i32 %893, i32 0, i32 0, i32 %900, i1 false, i1 false)
  %927 = add i32 %851, 18688
  %928 = add i32 %927, %853
  %929 = add i32 %928, %855
  %930 = sext i32 %929 to i64
  %931 = add i32 %842, %929
  %932 = inttoptr i32 %931 to ptr addrspace(3)
  %933 = load <4 x i32>, ptr addrspace(3) %932, align 16
  %934 = add i64 %930, 512
  %935 = trunc i64 %934 to i32
  %936 = add i32 %842, %935
  %937 = inttoptr i32 %936 to ptr addrspace(3)
  %938 = load <4 x i32>, ptr addrspace(3) %937, align 16
  %939 = add i64 %930, 1024
  %940 = trunc i64 %939 to i32
  %941 = add i32 %842, %940
  %942 = inttoptr i32 %941 to ptr addrspace(3)
  %943 = load <4 x i32>, ptr addrspace(3) %942, align 16
  %944 = add i64 %930, 1536
  %945 = trunc i64 %944 to i32
  %946 = add i32 %842, %945
  %947 = inttoptr i32 %946 to ptr addrspace(3)
  %948 = load <4 x i32>, ptr addrspace(3) %947, align 16
  %949 = shufflevector <4 x i32> %933, <4 x i32> %938, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %950 = shufflevector <4 x i32> %943, <4 x i32> %948, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %951 = shufflevector <8 x i32> %949, <8 x i32> %950, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %952 = add i32 %887, 49537
  %953 = add i32 %840, %952
  %954 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %953
  %955 = load i8, ptr addrspace(3) %954, align 1
  %956 = zext i8 %955 to i32
  %957 = call i32 @llvm.amdgcn.perm(i32 %956, i32 %956, i32 0)
  %958 = add i32 %894, 49424
  %959 = add i32 %840, %958
  %960 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %959
  %961 = load i8, ptr addrspace(3) %960, align 1
  %962 = zext i8 %961 to i32
  %963 = call i32 @llvm.amdgcn.perm(i32 %962, i32 %962, i32 0)
  %964 = add i32 %901, 128
  %965 = add i32 %964, %902
  %966 = sext i32 %965 to i64
  %967 = add i32 %842, %965
  %968 = inttoptr i32 %967 to ptr addrspace(3)
  %969 = load <4 x i32>, ptr addrspace(3) %968, align 16
  %970 = add i64 %966, 32
  %971 = trunc i64 %970 to i32
  %972 = add i32 %842, %971
  %973 = inttoptr i32 %972 to ptr addrspace(3)
  %974 = load <4 x i32>, ptr addrspace(3) %973, align 16
  %975 = add i64 %966, 64
  %976 = trunc i64 %975 to i32
  %977 = add i32 %842, %976
  %978 = inttoptr i32 %977 to ptr addrspace(3)
  %979 = load <4 x i32>, ptr addrspace(3) %978, align 16
  %980 = add i64 %966, 96
  %981 = trunc i64 %980 to i32
  %982 = add i32 %842, %981
  %983 = inttoptr i32 %982 to ptr addrspace(3)
  %984 = load <4 x i32>, ptr addrspace(3) %983, align 16
  %985 = shufflevector <4 x i32> %969, <4 x i32> %974, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %986 = shufflevector <4 x i32> %979, <4 x i32> %984, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %987 = shufflevector <8 x i32> %985, <8 x i32> %986, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %988 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %951, i32 0, <16 x i32> %987, i16 0, <8 x float> %926, i32 0, i32 0, i32 %957, i32 0, i32 0, i32 %963, i1 false, i1 false)
  %989 = add i32 %851, 20736
  %990 = add i32 %989, %853
  %991 = add i32 %990, %855
  %992 = sext i32 %991 to i64
  %993 = add i32 %842, %991
  %994 = inttoptr i32 %993 to ptr addrspace(3)
  %995 = load <4 x i32>, ptr addrspace(3) %994, align 16
  %996 = add i64 %992, 512
  %997 = trunc i64 %996 to i32
  %998 = add i32 %842, %997
  %999 = inttoptr i32 %998 to ptr addrspace(3)
  %1000 = load <4 x i32>, ptr addrspace(3) %999, align 16
  %1001 = add i64 %992, 1024
  %1002 = trunc i64 %1001 to i32
  %1003 = add i32 %842, %1002
  %1004 = inttoptr i32 %1003 to ptr addrspace(3)
  %1005 = load <4 x i32>, ptr addrspace(3) %1004, align 16
  %1006 = add i64 %992, 1536
  %1007 = trunc i64 %1006 to i32
  %1008 = add i32 %842, %1007
  %1009 = inttoptr i32 %1008 to ptr addrspace(3)
  %1010 = load <4 x i32>, ptr addrspace(3) %1009, align 16
  %1011 = shufflevector <4 x i32> %995, <4 x i32> %1000, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1012 = shufflevector <4 x i32> %1005, <4 x i32> %1010, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1013 = shufflevector <8 x i32> %1011, <8 x i32> %1012, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1014 = add i32 %887, 49538
  %1015 = add i32 %840, %1014
  %1016 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1015
  %1017 = load i8, ptr addrspace(3) %1016, align 1
  %1018 = zext i8 %1017 to i32
  %1019 = call i32 @llvm.amdgcn.perm(i32 %1018, i32 %1018, i32 0)
  %1020 = add i32 %894, 49440
  %1021 = add i32 %840, %1020
  %1022 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1021
  %1023 = load i8, ptr addrspace(3) %1022, align 1
  %1024 = zext i8 %1023 to i32
  %1025 = call i32 @llvm.amdgcn.perm(i32 %1024, i32 %1024, i32 0)
  %1026 = add i32 %901, 256
  %1027 = add i32 %1026, %902
  %1028 = sext i32 %1027 to i64
  %1029 = add i32 %842, %1027
  %1030 = inttoptr i32 %1029 to ptr addrspace(3)
  %1031 = load <4 x i32>, ptr addrspace(3) %1030, align 16
  %1032 = add i64 %1028, 32
  %1033 = trunc i64 %1032 to i32
  %1034 = add i32 %842, %1033
  %1035 = inttoptr i32 %1034 to ptr addrspace(3)
  %1036 = load <4 x i32>, ptr addrspace(3) %1035, align 16
  %1037 = add i64 %1028, 64
  %1038 = trunc i64 %1037 to i32
  %1039 = add i32 %842, %1038
  %1040 = inttoptr i32 %1039 to ptr addrspace(3)
  %1041 = load <4 x i32>, ptr addrspace(3) %1040, align 16
  %1042 = add i64 %1028, 96
  %1043 = trunc i64 %1042 to i32
  %1044 = add i32 %842, %1043
  %1045 = inttoptr i32 %1044 to ptr addrspace(3)
  %1046 = load <4 x i32>, ptr addrspace(3) %1045, align 16
  %1047 = shufflevector <4 x i32> %1031, <4 x i32> %1036, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1048 = shufflevector <4 x i32> %1041, <4 x i32> %1046, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1049 = shufflevector <8 x i32> %1047, <8 x i32> %1048, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1050 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1013, i32 0, <16 x i32> %1049, i16 0, <8 x float> %988, i32 0, i32 0, i32 %1019, i32 0, i32 0, i32 %1025, i1 false, i1 false)
  %1051 = add i32 %851, 22784
  %1052 = add i32 %1051, %853
  %1053 = add i32 %1052, %855
  %1054 = sext i32 %1053 to i64
  %1055 = add i32 %842, %1053
  %1056 = inttoptr i32 %1055 to ptr addrspace(3)
  %1057 = load <4 x i32>, ptr addrspace(3) %1056, align 16
  %1058 = add i64 %1054, 512
  %1059 = trunc i64 %1058 to i32
  %1060 = add i32 %842, %1059
  %1061 = inttoptr i32 %1060 to ptr addrspace(3)
  %1062 = load <4 x i32>, ptr addrspace(3) %1061, align 16
  %1063 = add i64 %1054, 1024
  %1064 = trunc i64 %1063 to i32
  %1065 = add i32 %842, %1064
  %1066 = inttoptr i32 %1065 to ptr addrspace(3)
  %1067 = load <4 x i32>, ptr addrspace(3) %1066, align 16
  %1068 = add i64 %1054, 1536
  %1069 = trunc i64 %1068 to i32
  %1070 = add i32 %842, %1069
  %1071 = inttoptr i32 %1070 to ptr addrspace(3)
  %1072 = load <4 x i32>, ptr addrspace(3) %1071, align 16
  %1073 = shufflevector <4 x i32> %1057, <4 x i32> %1062, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1074 = shufflevector <4 x i32> %1067, <4 x i32> %1072, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1075 = shufflevector <8 x i32> %1073, <8 x i32> %1074, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1076 = add i32 %887, 49539
  %1077 = add i32 %840, %1076
  %1078 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1077
  %1079 = load i8, ptr addrspace(3) %1078, align 1
  %1080 = zext i8 %1079 to i32
  %1081 = call i32 @llvm.amdgcn.perm(i32 %1080, i32 %1080, i32 0)
  %1082 = add i32 %894, 49456
  %1083 = add i32 %840, %1082
  %1084 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1083
  %1085 = load i8, ptr addrspace(3) %1084, align 1
  %1086 = zext i8 %1085 to i32
  %1087 = call i32 @llvm.amdgcn.perm(i32 %1086, i32 %1086, i32 0)
  %1088 = add i32 %901, 384
  %1089 = add i32 %1088, %902
  %1090 = sext i32 %1089 to i64
  %1091 = add i32 %842, %1089
  %1092 = inttoptr i32 %1091 to ptr addrspace(3)
  %1093 = load <4 x i32>, ptr addrspace(3) %1092, align 16
  %1094 = add i64 %1090, 32
  %1095 = trunc i64 %1094 to i32
  %1096 = add i32 %842, %1095
  %1097 = inttoptr i32 %1096 to ptr addrspace(3)
  %1098 = load <4 x i32>, ptr addrspace(3) %1097, align 16
  %1099 = add i64 %1090, 64
  %1100 = trunc i64 %1099 to i32
  %1101 = add i32 %842, %1100
  %1102 = inttoptr i32 %1101 to ptr addrspace(3)
  %1103 = load <4 x i32>, ptr addrspace(3) %1102, align 16
  %1104 = add i64 %1090, 96
  %1105 = trunc i64 %1104 to i32
  %1106 = add i32 %842, %1105
  %1107 = inttoptr i32 %1106 to ptr addrspace(3)
  %1108 = load <4 x i32>, ptr addrspace(3) %1107, align 16
  %1109 = shufflevector <4 x i32> %1093, <4 x i32> %1098, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1110 = shufflevector <4 x i32> %1103, <4 x i32> %1108, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1111 = shufflevector <8 x i32> %1109, <8 x i32> %1110, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1112 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1075, i32 0, <16 x i32> %1111, i16 0, <8 x float> %1050, i32 0, i32 0, i32 %1081, i32 0, i32 0, i32 %1087, i1 false, i1 false)
  %1113 = add i32 %851, 24832
  %1114 = add i32 %1113, %853
  %1115 = add i32 %1114, %855
  %1116 = sext i32 %1115 to i64
  %1117 = add i32 %842, %1115
  %1118 = inttoptr i32 %1117 to ptr addrspace(3)
  %1119 = load <4 x i32>, ptr addrspace(3) %1118, align 16
  %1120 = add i64 %1116, 512
  %1121 = trunc i64 %1120 to i32
  %1122 = add i32 %842, %1121
  %1123 = inttoptr i32 %1122 to ptr addrspace(3)
  %1124 = load <4 x i32>, ptr addrspace(3) %1123, align 16
  %1125 = add i64 %1116, 1024
  %1126 = trunc i64 %1125 to i32
  %1127 = add i32 %842, %1126
  %1128 = inttoptr i32 %1127 to ptr addrspace(3)
  %1129 = load <4 x i32>, ptr addrspace(3) %1128, align 16
  %1130 = add i64 %1116, 1536
  %1131 = trunc i64 %1130 to i32
  %1132 = add i32 %842, %1131
  %1133 = inttoptr i32 %1132 to ptr addrspace(3)
  %1134 = load <4 x i32>, ptr addrspace(3) %1133, align 16
  %1135 = shufflevector <4 x i32> %1119, <4 x i32> %1124, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1136 = shufflevector <4 x i32> %1129, <4 x i32> %1134, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1137 = shufflevector <8 x i32> %1135, <8 x i32> %1136, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1138 = add i32 %887, 49540
  %1139 = add i32 %840, %1138
  %1140 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1139
  %1141 = load i8, ptr addrspace(3) %1140, align 1
  %1142 = zext i8 %1141 to i32
  %1143 = call i32 @llvm.amdgcn.perm(i32 %1142, i32 %1142, i32 0)
  %1144 = add i32 %894, 49472
  %1145 = add i32 %840, %1144
  %1146 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1145
  %1147 = load i8, ptr addrspace(3) %1146, align 1
  %1148 = zext i8 %1147 to i32
  %1149 = call i32 @llvm.amdgcn.perm(i32 %1148, i32 %1148, i32 0)
  %1150 = add i32 %901, 512
  %1151 = add i32 %1150, %902
  %1152 = sext i32 %1151 to i64
  %1153 = add i32 %842, %1151
  %1154 = inttoptr i32 %1153 to ptr addrspace(3)
  %1155 = load <4 x i32>, ptr addrspace(3) %1154, align 16
  %1156 = add i64 %1152, 32
  %1157 = trunc i64 %1156 to i32
  %1158 = add i32 %842, %1157
  %1159 = inttoptr i32 %1158 to ptr addrspace(3)
  %1160 = load <4 x i32>, ptr addrspace(3) %1159, align 16
  %1161 = add i64 %1152, 64
  %1162 = trunc i64 %1161 to i32
  %1163 = add i32 %842, %1162
  %1164 = inttoptr i32 %1163 to ptr addrspace(3)
  %1165 = load <4 x i32>, ptr addrspace(3) %1164, align 16
  %1166 = add i64 %1152, 96
  %1167 = trunc i64 %1166 to i32
  %1168 = add i32 %842, %1167
  %1169 = inttoptr i32 %1168 to ptr addrspace(3)
  %1170 = load <4 x i32>, ptr addrspace(3) %1169, align 16
  %1171 = shufflevector <4 x i32> %1155, <4 x i32> %1160, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1172 = shufflevector <4 x i32> %1165, <4 x i32> %1170, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1173 = shufflevector <8 x i32> %1171, <8 x i32> %1172, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1174 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1137, i32 0, <16 x i32> %1173, i16 0, <8 x float> %1112, i32 0, i32 0, i32 %1143, i32 0, i32 0, i32 %1149, i1 false, i1 false)
  %1175 = add i32 %851, 26880
  %1176 = add i32 %1175, %853
  %1177 = add i32 %1176, %855
  %1178 = sext i32 %1177 to i64
  %1179 = add i32 %842, %1177
  %1180 = inttoptr i32 %1179 to ptr addrspace(3)
  %1181 = load <4 x i32>, ptr addrspace(3) %1180, align 16
  %1182 = add i64 %1178, 512
  %1183 = trunc i64 %1182 to i32
  %1184 = add i32 %842, %1183
  %1185 = inttoptr i32 %1184 to ptr addrspace(3)
  %1186 = load <4 x i32>, ptr addrspace(3) %1185, align 16
  %1187 = add i64 %1178, 1024
  %1188 = trunc i64 %1187 to i32
  %1189 = add i32 %842, %1188
  %1190 = inttoptr i32 %1189 to ptr addrspace(3)
  %1191 = load <4 x i32>, ptr addrspace(3) %1190, align 16
  %1192 = add i64 %1178, 1536
  %1193 = trunc i64 %1192 to i32
  %1194 = add i32 %842, %1193
  %1195 = inttoptr i32 %1194 to ptr addrspace(3)
  %1196 = load <4 x i32>, ptr addrspace(3) %1195, align 16
  %1197 = shufflevector <4 x i32> %1181, <4 x i32> %1186, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1198 = shufflevector <4 x i32> %1191, <4 x i32> %1196, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1199 = shufflevector <8 x i32> %1197, <8 x i32> %1198, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1200 = add i32 %887, 49541
  %1201 = add i32 %840, %1200
  %1202 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1201
  %1203 = load i8, ptr addrspace(3) %1202, align 1
  %1204 = zext i8 %1203 to i32
  %1205 = call i32 @llvm.amdgcn.perm(i32 %1204, i32 %1204, i32 0)
  %1206 = add i32 %894, 49488
  %1207 = add i32 %840, %1206
  %1208 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1207
  %1209 = load i8, ptr addrspace(3) %1208, align 1
  %1210 = zext i8 %1209 to i32
  %1211 = call i32 @llvm.amdgcn.perm(i32 %1210, i32 %1210, i32 0)
  %1212 = add i32 %901, 640
  %1213 = add i32 %1212, %902
  %1214 = sext i32 %1213 to i64
  %1215 = add i32 %842, %1213
  %1216 = inttoptr i32 %1215 to ptr addrspace(3)
  %1217 = load <4 x i32>, ptr addrspace(3) %1216, align 16
  %1218 = add i64 %1214, 32
  %1219 = trunc i64 %1218 to i32
  %1220 = add i32 %842, %1219
  %1221 = inttoptr i32 %1220 to ptr addrspace(3)
  %1222 = load <4 x i32>, ptr addrspace(3) %1221, align 16
  %1223 = add i64 %1214, 64
  %1224 = trunc i64 %1223 to i32
  %1225 = add i32 %842, %1224
  %1226 = inttoptr i32 %1225 to ptr addrspace(3)
  %1227 = load <4 x i32>, ptr addrspace(3) %1226, align 16
  %1228 = add i64 %1214, 96
  %1229 = trunc i64 %1228 to i32
  %1230 = add i32 %842, %1229
  %1231 = inttoptr i32 %1230 to ptr addrspace(3)
  %1232 = load <4 x i32>, ptr addrspace(3) %1231, align 16
  %1233 = shufflevector <4 x i32> %1217, <4 x i32> %1222, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1234 = shufflevector <4 x i32> %1227, <4 x i32> %1232, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1235 = shufflevector <8 x i32> %1233, <8 x i32> %1234, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1236 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1199, i32 0, <16 x i32> %1235, i16 0, <8 x float> %1174, i32 0, i32 0, i32 %1205, i32 0, i32 0, i32 %1211, i1 false, i1 false)
  %1237 = add i32 %851, 28928
  %1238 = add i32 %1237, %853
  %1239 = add i32 %1238, %855
  %1240 = sext i32 %1239 to i64
  %1241 = add i32 %842, %1239
  %1242 = inttoptr i32 %1241 to ptr addrspace(3)
  %1243 = load <4 x i32>, ptr addrspace(3) %1242, align 16
  %1244 = add i64 %1240, 512
  %1245 = trunc i64 %1244 to i32
  %1246 = add i32 %842, %1245
  %1247 = inttoptr i32 %1246 to ptr addrspace(3)
  %1248 = load <4 x i32>, ptr addrspace(3) %1247, align 16
  %1249 = add i64 %1240, 1024
  %1250 = trunc i64 %1249 to i32
  %1251 = add i32 %842, %1250
  %1252 = inttoptr i32 %1251 to ptr addrspace(3)
  %1253 = load <4 x i32>, ptr addrspace(3) %1252, align 16
  %1254 = add i64 %1240, 1536
  %1255 = trunc i64 %1254 to i32
  %1256 = add i32 %842, %1255
  %1257 = inttoptr i32 %1256 to ptr addrspace(3)
  %1258 = load <4 x i32>, ptr addrspace(3) %1257, align 16
  %1259 = shufflevector <4 x i32> %1243, <4 x i32> %1248, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1260 = shufflevector <4 x i32> %1253, <4 x i32> %1258, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1261 = shufflevector <8 x i32> %1259, <8 x i32> %1260, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1262 = add i32 %887, 49542
  %1263 = add i32 %840, %1262
  %1264 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1263
  %1265 = load i8, ptr addrspace(3) %1264, align 1
  %1266 = zext i8 %1265 to i32
  %1267 = call i32 @llvm.amdgcn.perm(i32 %1266, i32 %1266, i32 0)
  %1268 = add i32 %894, 49504
  %1269 = add i32 %840, %1268
  %1270 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1269
  %1271 = load i8, ptr addrspace(3) %1270, align 1
  %1272 = zext i8 %1271 to i32
  %1273 = call i32 @llvm.amdgcn.perm(i32 %1272, i32 %1272, i32 0)
  %1274 = add i32 %901, 768
  %1275 = add i32 %1274, %902
  %1276 = sext i32 %1275 to i64
  %1277 = add i32 %842, %1275
  %1278 = inttoptr i32 %1277 to ptr addrspace(3)
  %1279 = load <4 x i32>, ptr addrspace(3) %1278, align 16
  %1280 = add i64 %1276, 32
  %1281 = trunc i64 %1280 to i32
  %1282 = add i32 %842, %1281
  %1283 = inttoptr i32 %1282 to ptr addrspace(3)
  %1284 = load <4 x i32>, ptr addrspace(3) %1283, align 16
  %1285 = add i64 %1276, 64
  %1286 = trunc i64 %1285 to i32
  %1287 = add i32 %842, %1286
  %1288 = inttoptr i32 %1287 to ptr addrspace(3)
  %1289 = load <4 x i32>, ptr addrspace(3) %1288, align 16
  %1290 = add i64 %1276, 96
  %1291 = trunc i64 %1290 to i32
  %1292 = add i32 %842, %1291
  %1293 = inttoptr i32 %1292 to ptr addrspace(3)
  %1294 = load <4 x i32>, ptr addrspace(3) %1293, align 16
  %1295 = shufflevector <4 x i32> %1279, <4 x i32> %1284, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1296 = shufflevector <4 x i32> %1289, <4 x i32> %1294, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1297 = shufflevector <8 x i32> %1295, <8 x i32> %1296, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1298 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1261, i32 0, <16 x i32> %1297, i16 0, <8 x float> %1236, i32 0, i32 0, i32 %1267, i32 0, i32 0, i32 %1273, i1 false, i1 false)
  %1299 = add i32 %851, 30976
  %1300 = add i32 %1299, %853
  %1301 = add i32 %1300, %855
  %1302 = sext i32 %1301 to i64
  %1303 = add i32 %842, %1301
  %1304 = inttoptr i32 %1303 to ptr addrspace(3)
  %1305 = load <4 x i32>, ptr addrspace(3) %1304, align 16
  %1306 = add i64 %1302, 512
  %1307 = trunc i64 %1306 to i32
  %1308 = add i32 %842, %1307
  %1309 = inttoptr i32 %1308 to ptr addrspace(3)
  %1310 = load <4 x i32>, ptr addrspace(3) %1309, align 16
  %1311 = add i64 %1302, 1024
  %1312 = trunc i64 %1311 to i32
  %1313 = add i32 %842, %1312
  %1314 = inttoptr i32 %1313 to ptr addrspace(3)
  %1315 = load <4 x i32>, ptr addrspace(3) %1314, align 16
  %1316 = add i64 %1302, 1536
  %1317 = trunc i64 %1316 to i32
  %1318 = add i32 %842, %1317
  %1319 = inttoptr i32 %1318 to ptr addrspace(3)
  %1320 = load <4 x i32>, ptr addrspace(3) %1319, align 16
  %1321 = shufflevector <4 x i32> %1305, <4 x i32> %1310, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1322 = shufflevector <4 x i32> %1315, <4 x i32> %1320, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1323 = shufflevector <8 x i32> %1321, <8 x i32> %1322, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1324 = add i32 %887, 49543
  %1325 = add i32 %840, %1324
  %1326 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1325
  %1327 = load i8, ptr addrspace(3) %1326, align 1
  %1328 = zext i8 %1327 to i32
  %1329 = call i32 @llvm.amdgcn.perm(i32 %1328, i32 %1328, i32 0)
  %1330 = add i32 %894, 49520
  %1331 = add i32 %840, %1330
  %1332 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1331
  %1333 = load i8, ptr addrspace(3) %1332, align 1
  %1334 = zext i8 %1333 to i32
  %1335 = call i32 @llvm.amdgcn.perm(i32 %1334, i32 %1334, i32 0)
  %1336 = add i32 %901, 896
  %1337 = add i32 %1336, %902
  %1338 = sext i32 %1337 to i64
  %1339 = add i32 %842, %1337
  %1340 = inttoptr i32 %1339 to ptr addrspace(3)
  %1341 = load <4 x i32>, ptr addrspace(3) %1340, align 16
  %1342 = add i64 %1338, 32
  %1343 = trunc i64 %1342 to i32
  %1344 = add i32 %842, %1343
  %1345 = inttoptr i32 %1344 to ptr addrspace(3)
  %1346 = load <4 x i32>, ptr addrspace(3) %1345, align 16
  %1347 = add i64 %1338, 64
  %1348 = trunc i64 %1347 to i32
  %1349 = add i32 %842, %1348
  %1350 = inttoptr i32 %1349 to ptr addrspace(3)
  %1351 = load <4 x i32>, ptr addrspace(3) %1350, align 16
  %1352 = add i64 %1338, 96
  %1353 = trunc i64 %1352 to i32
  %1354 = add i32 %842, %1353
  %1355 = inttoptr i32 %1354 to ptr addrspace(3)
  %1356 = load <4 x i32>, ptr addrspace(3) %1355, align 16
  %1357 = shufflevector <4 x i32> %1341, <4 x i32> %1346, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1358 = shufflevector <4 x i32> %1351, <4 x i32> %1356, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1359 = shufflevector <8 x i32> %1357, <8 x i32> %1358, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1360 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1323, i32 0, <16 x i32> %1359, i16 0, <8 x float> %1298, i32 0, i32 0, i32 %1329, i32 0, i32 0, i32 %1335, i1 false, i1 false)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 10, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 6, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 4, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 6, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 4, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 6, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 4, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 6, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 4, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 6, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 4, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 6, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 4, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 6, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 4, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %1361 = mul i32 %50, 8
  %1362 = add i32 %99, %1361
  %1363 = fptrunc <8 x float> %1360 to <8 x bfloat>
  %1364 = bitcast <8 x bfloat> %1363 to <16 x i8>
  %1365 = mul i32 %894, 32
  %1366 = add i32 %1365, %1362
  %1367 = mul i32 %1366, 2
  %1368 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1367
  store <16 x i8> %1364, ptr addrspace(3) %1368, align 1
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %1369 = mul i64 %62, %22
  %1370 = add i64 %1369, %63
  %1371 = getelementptr bfloat, ptr addrspace(1) %0, i64 %1370
  %1372 = icmp eq i64 %22, -2147483648
  %1373 = select i1 %1372, i64 32, i64 %22
  %1374 = ptrtoint ptr addrspace(1) %1371 to i64
  %1375 = call i32 @llvm.amdgcn.wave.id()
  %1376 = urem i32 %1375, 2
  %1377 = mul i32 %1376, 8
  %1378 = zext i32 %1377 to i64
  %1379 = mul i64 %1378, %1373
  %1380 = mul i64 %1379, 2
  %1381 = add i64 %1374, %1380
  %1382 = mul i32 %1376, 512
  %1383 = add i32 ptrtoint (ptr addrspace(3) @__dynamic_shared_0 to i32), %1382
  %1384 = trunc i64 %1381 to i32
  %1385 = lshr i64 %1381, 32
  %1386 = trunc i64 %1385 to i32
  %1387 = or i32 %1386, -2147483648
  %1388 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %1383, i64 1
  %1389 = insertelement <4 x i32> %1388, i32 %1384, i64 2
  %1390 = insertelement <4 x i32> %1389, i32 %1387, i64 3
  %1391 = sub i32 %64, %1377
  %1392 = call i32 @llvm.smax.i32(i32 %1391, i32 0)
  %1393 = and i32 %1392, 65535
  %1394 = shl i32 %1393, 16
  %1395 = or i32 %1394, 32767
  %1396 = lshr i32 %1392, 16
  %1397 = and i32 %1396, 65535
  %1398 = or i32 %1397, 2097152
  %1399 = trunc i64 %1373 to i32
  %1400 = lshr i64 %1373, 32
  %1401 = trunc i64 %1400 to i32
  %1402 = and i32 %1401, 65535
  %1403 = insertelement <8 x i32> <i32 65536, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %1395, i64 2
  %1404 = insertelement <8 x i32> %1403, i32 %1398, i64 3
  %1405 = insertelement <8 x i32> %1404, i32 8, i64 4
  %1406 = insertelement <8 x i32> %1405, i32 %1399, i64 5
  %1407 = insertelement <8 x i32> %1406, i32 %1402, i64 6
  %1408 = insertelement <8 x i32> %1407, i32 0, i64 7
  call void @llvm.amdgcn.tensor.store.from.lds(<4 x i32> %1390, <8 x i32> %1408, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #1

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #3

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @llvm.amdgcn.tensor.load.to.lds(<4 x i32>, <8 x i32>, <4 x i32>, <4 x i32>, <8 x i32>, i32 immarg) #4

; Function Attrs: nocallback nofree nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.amdgcn.s.wait.tensorcnt(i16 immarg) #5

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier.signal(i32 immarg) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier.wait(i16 immarg) #6

; Function Attrs: nocallback nofree nounwind speculatable willreturn memory(none)
declare i32 @llvm.amdgcn.perm(i32, i32, i32) #7

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.wait.dscnt(i16 immarg) #8

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 immarg, <16 x i32>, i32 immarg, <16 x i32>, i16 immarg, <8 x float>, i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg, i32, i1 immarg, i1 immarg) #2

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.group.barrier(i32 immarg, i32 immarg, i32 immarg) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #6

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.wave.id() #1

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @llvm.amdgcn.tensor.store.from.lds(<4 x i32>, <8 x i32>, <4 x i32>, <4 x i32>, <8 x i32>, i32 immarg) #4

attributes #0 = { "amdgpu-flat-work-group-size"="64,64" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { convergent nocallback nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) }
attributes #5 = { nocallback nofree nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #6 = { convergent nocallback nofree nounwind willreturn }
attributes #7 = { nocallback nofree nounwind speculatable willreturn memory(none) }
attributes #8 = { nocallback nofree nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 64, i32 1, i32 1}
