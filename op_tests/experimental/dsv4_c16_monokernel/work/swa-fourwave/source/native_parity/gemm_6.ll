; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

@__dynamic_shared_0 = external dso_local addrspace(3) global [0 x i8], align 1024

define amdgpu_kernel void @gemm_a8w8_mx128_t16x128x512_mw1_nw4_nb2_sk1_cm1_cn1(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10) #0 !reqd_work_group_size !1 {
  %12 = sdiv i32 %7, 512
  %13 = mul i32 %12, 512
  %14 = icmp ne i32 %7, %13
  %15 = icmp slt i32 %7, 0
  %16 = icmp ne i1 %15, false
  %17 = and i1 %14, %16
  %18 = add i32 %12, -1
  %19 = select i1 %17, i32 %18, i32 %12
  %20 = sext i32 %7 to i64
  %21 = sext i32 %9 to i64
  %22 = sext i32 %10 to i64
  %23 = call range(i32 0, 128) i32 @llvm.amdgcn.workitem.id.x()
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
  %51 = sdiv i32 %40, 4
  %52 = mul i32 %51, 4
  %53 = icmp ne i32 %40, %52
  %54 = icmp slt i32 %40, 0
  %55 = icmp ne i1 %54, false
  %56 = and i1 %53, %55
  %57 = add i32 %51, -1
  %58 = select i1 %56, i32 %57, i32 %51
  %59 = srem i32 %40, 4
  %60 = mul i32 %28, 16
  %61 = mul i32 %31, 128
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
  %99 = mul i32 %59, 32
  %100 = icmp eq i32 %40, 0
  br i1 %100, label %101, label %128

101:                                              ; preds = %11
  %102 = icmp eq i64 %21, -2147483648
  %103 = select i1 %102, i64 512, i64 %21
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
  %117 = or i32 %116, 33554432
  %118 = trunc i64 %103 to i32
  %119 = lshr i64 %103, 32
  %120 = trunc i64 %119 to i32
  %121 = and i32 %120, 65535
  %122 = insertelement <8 x i32> <i32 128974848, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %114, i64 2
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
  %132 = select i1 %131, i64 8192, i64 %75
  %133 = ptrtoint ptr addrspace(1) %78 to i64
  %134 = trunc i64 %133 to i32
  %135 = lshr i64 %133, 32
  %136 = trunc i64 %135 to i32
  %137 = or i32 %136, -2147483648
  %138 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 8448) to i32), i32 poison, i32 poison>, i32 %134, i64 2
  %139 = insertelement <4 x i32> %138, i32 %137, i64 3
  %140 = trunc i64 %132 to i32
  %141 = lshr i64 %132, 32
  %142 = trunc i64 %141 to i32
  %143 = and i32 %142, 65535
  %144 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 536903679, i32 8, i32 poison, i32 poison, i32 poison>, i32 %140, i64 5
  %145 = insertelement <8 x i32> %144, i32 %143, i64 6
  %146 = insertelement <8 x i32> %145, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %139, <8 x i32> %146, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %147

147:                                              ; preds = %130, %128
  %148 = icmp eq i32 %40, 2
  br i1 %148, label %149, label %176

149:                                              ; preds = %147
  %150 = icmp eq i64 %65, -2147483648
  %151 = select i1 %150, i64 16, i64 %65
  %152 = ptrtoint ptr addrspace(1) %96 to i64
  %153 = trunc i64 %152 to i32
  %154 = lshr i64 %152, 32
  %155 = trunc i64 %154 to i32
  %156 = or i32 %155, -2147483648
  %157 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 73984) to i32), i32 poison, i32 poison>, i32 %153, i64 2
  %158 = insertelement <4 x i32> %157, i32 %156, i64 3
  %159 = call i32 @llvm.smax.i32(i32 %64, i32 0)
  %160 = and i32 %159, 65535
  %161 = shl i32 %160, 16
  %162 = lshr i32 %159, 16
  %163 = and i32 %162, 65535
  %164 = or i32 %163, -65536
  %165 = trunc i64 %151 to i32
  %166 = lshr i64 %151, 32
  %167 = trunc i64 %166 to i32
  %168 = and i32 %167, 65535
  %169 = insertelement <8 x i32> <i32 2097152, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %161, i64 1
  %170 = insertelement <8 x i32> %169, i32 %164, i64 2
  %171 = insertelement <8 x i32> %170, i32 1081343, i64 3
  %172 = insertelement <8 x i32> %171, i32 4, i64 4
  %173 = insertelement <8 x i32> %172, i32 %165, i64 5
  %174 = insertelement <8 x i32> %173, i32 %168, i64 6
  %175 = insertelement <8 x i32> %174, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %158, <8 x i32> %175, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %176

176:                                              ; preds = %149, %147
  %177 = icmp eq i32 %40, 3
  br i1 %177, label %178, label %195

178:                                              ; preds = %176
  %179 = icmp eq i64 %94, -2147483648
  %180 = select i1 %179, i64 4, i64 %94
  %181 = ptrtoint ptr addrspace(1) %97 to i64
  %182 = trunc i64 %181 to i32
  %183 = lshr i64 %181, 32
  %184 = trunc i64 %183 to i32
  %185 = or i32 %184, -2147483648
  %186 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 74048) to i32), i32 poison, i32 poison>, i32 %182, i64 2
  %187 = insertelement <4 x i32> %186, i32 %185, i64 3
  %188 = trunc i64 %180 to i32
  %189 = lshr i64 %180, 32
  %190 = trunc i64 %189 to i32
  %191 = and i32 %190, 65535
  %192 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 294911, i32 1, i32 poison, i32 poison, i32 poison>, i32 %188, i64 5
  %193 = insertelement <8 x i32> %192, i32 %191, i64 6
  %194 = insertelement <8 x i32> %193, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %187, <8 x i32> %194, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %195

195:                                              ; preds = %178, %176
  %196 = sub i32 %19, 1
  br label %197

197:                                              ; preds = %450, %195
  %198 = phi i32 [ %207, %450 ], [ 0, %195 ]
  %199 = phi <8 x float> [ %731, %450 ], [ zeroinitializer, %195 ]
  %200 = phi <8 x float> [ %732, %450 ], [ zeroinitializer, %195 ]
  %201 = icmp slt i32 %198, %196
  br i1 %201, label %202, label %733

202:                                              ; preds = %197
  %203 = srem i32 %198, 2
  %204 = mul i32 %203, 74752
  %205 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %204
  %206 = ptrtoint ptr addrspace(3) %205 to i32
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %207 = add i32 %198, 1
  %208 = sdiv i32 %99, 16
  %209 = mul i32 %208, 16
  %210 = icmp ne i32 %99, %209
  %211 = icmp slt i32 %99, 0
  %212 = icmp ne i1 %211, false
  %213 = and i1 %210, %212
  %214 = add i32 %208, -1
  %215 = select i1 %213, i32 %214, i32 %208
  %216 = mul i32 %215, 8192
  %217 = add i32 %216, 8448
  %218 = mul i32 %50, 256
  %219 = add i32 %217, %218
  %220 = mul i32 %42, 16
  %221 = add i32 %219, %220
  %222 = sext i32 %221 to i64
  %223 = add i32 %206, %221
  %224 = inttoptr i32 %223 to ptr addrspace(3)
  %225 = load <4 x i32>, ptr addrspace(3) %224, align 16
  %226 = add i64 %222, 512
  %227 = trunc i64 %226 to i32
  %228 = add i32 %206, %227
  %229 = inttoptr i32 %228 to ptr addrspace(3)
  %230 = load <4 x i32>, ptr addrspace(3) %229, align 16
  %231 = add i64 %222, 1024
  %232 = trunc i64 %231 to i32
  %233 = add i32 %206, %232
  %234 = inttoptr i32 %233 to ptr addrspace(3)
  %235 = load <4 x i32>, ptr addrspace(3) %234, align 16
  %236 = add i64 %222, 1536
  %237 = trunc i64 %236 to i32
  %238 = add i32 %206, %237
  %239 = inttoptr i32 %238 to ptr addrspace(3)
  %240 = load <4 x i32>, ptr addrspace(3) %239, align 16
  %241 = shufflevector <4 x i32> %225, <4 x i32> %230, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %242 = shufflevector <4 x i32> %235, <4 x i32> %240, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %243 = shufflevector <8 x i32> %241, <8 x i32> %242, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %244 = add i32 %215, 1
  %245 = mul i32 %244, 8192
  %246 = add i32 %245, 8448
  %247 = add i32 %246, %218
  %248 = add i32 %247, %220
  %249 = sext i32 %248 to i64
  %250 = add i32 %206, %248
  %251 = inttoptr i32 %250 to ptr addrspace(3)
  %252 = load <4 x i32>, ptr addrspace(3) %251, align 16
  %253 = add i64 %249, 512
  %254 = trunc i64 %253 to i32
  %255 = add i32 %206, %254
  %256 = inttoptr i32 %255 to ptr addrspace(3)
  %257 = load <4 x i32>, ptr addrspace(3) %256, align 16
  %258 = add i64 %249, 1024
  %259 = trunc i64 %258 to i32
  %260 = add i32 %206, %259
  %261 = inttoptr i32 %260 to ptr addrspace(3)
  %262 = load <4 x i32>, ptr addrspace(3) %261, align 16
  %263 = add i64 %249, 1536
  %264 = trunc i64 %263 to i32
  %265 = add i32 %206, %264
  %266 = inttoptr i32 %265 to ptr addrspace(3)
  %267 = load <4 x i32>, ptr addrspace(3) %266, align 16
  %268 = shufflevector <4 x i32> %252, <4 x i32> %257, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %269 = shufflevector <4 x i32> %262, <4 x i32> %267, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %270 = shufflevector <8 x i32> %268, <8 x i32> %269, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %271 = sdiv i32 %99, 128
  %272 = mul i32 %271, 128
  %273 = icmp ne i32 %99, %272
  %274 = icmp slt i32 %99, 0
  %275 = icmp ne i1 %274, false
  %276 = and i1 %273, %275
  %277 = add i32 %271, -1
  %278 = select i1 %276, i32 %277, i32 %271
  %279 = mul i32 %278, 4
  %280 = add i32 %279, 74048
  %281 = add i32 %204, %280
  %282 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %281
  %283 = load i8, ptr addrspace(3) %282, align 1
  %284 = zext i8 %283 to i32
  %285 = call i32 @llvm.amdgcn.perm(i32 %284, i32 %284, i32 0)
  %286 = add i32 %99, 16
  %287 = sdiv i32 %286, 128
  %288 = mul i32 %287, 128
  %289 = icmp ne i32 %286, %288
  %290 = icmp slt i32 %286, 0
  %291 = icmp ne i1 %290, false
  %292 = and i1 %289, %291
  %293 = add i32 %287, -1
  %294 = select i1 %292, i32 %293, i32 %287
  %295 = mul i32 %294, 4
  %296 = add i32 %295, 74048
  %297 = add i32 %204, %296
  %298 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %297
  %299 = load i8, ptr addrspace(3) %298, align 1
  %300 = zext i8 %299 to i32
  %301 = call i32 @llvm.amdgcn.perm(i32 %300, i32 %300, i32 0)
  %302 = add i32 %98, %42
  %303 = add i32 %302, 73984
  %304 = add i32 %204, %303
  %305 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %304
  %306 = load i8, ptr addrspace(3) %305, align 1
  %307 = zext i8 %306 to i32
  %308 = call i32 @llvm.amdgcn.perm(i32 %307, i32 %307, i32 0)
  %309 = mul i32 %302, 528
  %310 = mul i32 %50, 16
  %311 = add i32 %309, %310
  %312 = sext i32 %311 to i64
  %313 = add i32 %206, %311
  %314 = inttoptr i32 %313 to ptr addrspace(3)
  %315 = load <4 x i32>, ptr addrspace(3) %314, align 16
  %316 = add i64 %312, 32
  %317 = trunc i64 %316 to i32
  %318 = add i32 %206, %317
  %319 = inttoptr i32 %318 to ptr addrspace(3)
  %320 = load <4 x i32>, ptr addrspace(3) %319, align 16
  %321 = add i64 %312, 64
  %322 = trunc i64 %321 to i32
  %323 = add i32 %206, %322
  %324 = inttoptr i32 %323 to ptr addrspace(3)
  %325 = load <4 x i32>, ptr addrspace(3) %324, align 16
  %326 = add i64 %312, 96
  %327 = trunc i64 %326 to i32
  %328 = add i32 %206, %327
  %329 = inttoptr i32 %328 to ptr addrspace(3)
  %330 = load <4 x i32>, ptr addrspace(3) %329, align 16
  %331 = shufflevector <4 x i32> %315, <4 x i32> %320, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %332 = shufflevector <4 x i32> %325, <4 x i32> %330, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %333 = shufflevector <8 x i32> %331, <8 x i32> %332, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %334 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %243, i32 0, <16 x i32> %333, i16 0, <8 x float> %199, i32 0, i32 0, i32 %285, i32 0, i32 0, i32 %308, i1 false, i1 false)
  %335 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %270, i32 0, <16 x i32> %333, i16 0, <8 x float> %200, i32 0, i32 0, i32 %301, i32 0, i32 0, i32 %308, i1 false, i1 false)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %336 = srem i32 %207, 2
  %337 = mul i32 %336, 74752
  %338 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %337
  %339 = sext i32 %207 to i64
  %340 = mul i64 %339, 512
  br i1 %100, label %341, label %371

341:                                              ; preds = %202
  %342 = icmp eq i64 %21, -2147483648
  %343 = select i1 %342, i64 512, i64 %21
  %344 = ptrtoint ptr addrspace(1) %77 to i64
  %345 = add i64 %344, %340
  %346 = ptrtoint ptr addrspace(3) %338 to i32
  %347 = trunc i64 %345 to i32
  %348 = lshr i64 %345, 32
  %349 = trunc i64 %348 to i32
  %350 = or i32 %349, -2147483648
  %351 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %346, i64 1
  %352 = insertelement <4 x i32> %351, i32 %347, i64 2
  %353 = insertelement <4 x i32> %352, i32 %350, i64 3
  %354 = call i32 @llvm.smax.i32(i32 %64, i32 0)
  %355 = and i32 %354, 65535
  %356 = shl i32 %355, 16
  %357 = or i32 %356, 32767
  %358 = lshr i32 %354, 16
  %359 = and i32 %358, 65535
  %360 = or i32 %359, 33554432
  %361 = trunc i64 %343 to i32
  %362 = lshr i64 %343, 32
  %363 = trunc i64 %362 to i32
  %364 = and i32 %363, 65535
  %365 = insertelement <8 x i32> <i32 128974848, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %357, i64 2
  %366 = insertelement <8 x i32> %365, i32 %360, i64 3
  %367 = insertelement <8 x i32> %366, i32 16, i64 4
  %368 = insertelement <8 x i32> %367, i32 %361, i64 5
  %369 = insertelement <8 x i32> %368, i32 %364, i64 6
  %370 = insertelement <8 x i32> %369, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %353, <8 x i32> %370, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %371

371:                                              ; preds = %341, %202
  %372 = getelementptr i8, ptr addrspace(3) %338, i32 8448
  %373 = mul i64 %339, 8192
  br i1 %129, label %374, label %394

374:                                              ; preds = %371
  %375 = icmp eq i64 %75, -2147483648
  %376 = select i1 %375, i64 8192, i64 %75
  %377 = ptrtoint ptr addrspace(1) %78 to i64
  %378 = add i64 %377, %373
  %379 = ptrtoint ptr addrspace(3) %372 to i32
  %380 = trunc i64 %378 to i32
  %381 = lshr i64 %378, 32
  %382 = trunc i64 %381 to i32
  %383 = or i32 %382, -2147483648
  %384 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %379, i64 1
  %385 = insertelement <4 x i32> %384, i32 %380, i64 2
  %386 = insertelement <4 x i32> %385, i32 %383, i64 3
  %387 = trunc i64 %376 to i32
  %388 = lshr i64 %376, 32
  %389 = trunc i64 %388 to i32
  %390 = and i32 %389, 65535
  %391 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 536903679, i32 8, i32 poison, i32 poison, i32 poison>, i32 %387, i64 5
  %392 = insertelement <8 x i32> %391, i32 %390, i64 6
  %393 = insertelement <8 x i32> %392, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %386, <8 x i32> %393, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %394

394:                                              ; preds = %374, %371
  %395 = getelementptr i8, ptr addrspace(3) %338, i32 73984
  %396 = mul i64 %339, 4
  %397 = mul i64 %396, %65
  br i1 %148, label %398, label %428

398:                                              ; preds = %394
  %399 = icmp eq i64 %65, -2147483648
  %400 = select i1 %399, i64 16, i64 %65
  %401 = ptrtoint ptr addrspace(1) %96 to i64
  %402 = add i64 %401, %397
  %403 = ptrtoint ptr addrspace(3) %395 to i32
  %404 = trunc i64 %402 to i32
  %405 = lshr i64 %402, 32
  %406 = trunc i64 %405 to i32
  %407 = or i32 %406, -2147483648
  %408 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %403, i64 1
  %409 = insertelement <4 x i32> %408, i32 %404, i64 2
  %410 = insertelement <4 x i32> %409, i32 %407, i64 3
  %411 = call i32 @llvm.smax.i32(i32 %64, i32 0)
  %412 = and i32 %411, 65535
  %413 = shl i32 %412, 16
  %414 = lshr i32 %411, 16
  %415 = and i32 %414, 65535
  %416 = or i32 %415, -65536
  %417 = trunc i64 %400 to i32
  %418 = lshr i64 %400, 32
  %419 = trunc i64 %418 to i32
  %420 = and i32 %419, 65535
  %421 = insertelement <8 x i32> <i32 2097152, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %413, i64 1
  %422 = insertelement <8 x i32> %421, i32 %416, i64 2
  %423 = insertelement <8 x i32> %422, i32 1081343, i64 3
  %424 = insertelement <8 x i32> %423, i32 4, i64 4
  %425 = insertelement <8 x i32> %424, i32 %417, i64 5
  %426 = insertelement <8 x i32> %425, i32 %420, i64 6
  %427 = insertelement <8 x i32> %426, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %410, <8 x i32> %427, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %428

428:                                              ; preds = %398, %394
  %429 = getelementptr i8, ptr addrspace(3) %338, i32 74048
  br i1 %177, label %430, label %450

430:                                              ; preds = %428
  %431 = icmp eq i64 %94, -2147483648
  %432 = select i1 %431, i64 4, i64 %94
  %433 = ptrtoint ptr addrspace(1) %97 to i64
  %434 = add i64 %433, %396
  %435 = ptrtoint ptr addrspace(3) %429 to i32
  %436 = trunc i64 %434 to i32
  %437 = lshr i64 %434, 32
  %438 = trunc i64 %437 to i32
  %439 = or i32 %438, -2147483648
  %440 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %435, i64 1
  %441 = insertelement <4 x i32> %440, i32 %436, i64 2
  %442 = insertelement <4 x i32> %441, i32 %439, i64 3
  %443 = trunc i64 %432 to i32
  %444 = lshr i64 %432, 32
  %445 = trunc i64 %444 to i32
  %446 = and i32 %445, 65535
  %447 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 294911, i32 1, i32 poison, i32 poison, i32 poison>, i32 %443, i64 5
  %448 = insertelement <8 x i32> %447, i32 %446, i64 6
  %449 = insertelement <8 x i32> %448, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %442, <8 x i32> %449, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %450

450:                                              ; preds = %430, %428
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %451 = add i32 %216, 10496
  %452 = add i32 %451, %218
  %453 = add i32 %452, %220
  %454 = sext i32 %453 to i64
  %455 = add i32 %206, %453
  %456 = inttoptr i32 %455 to ptr addrspace(3)
  %457 = load <4 x i32>, ptr addrspace(3) %456, align 16
  %458 = add i64 %454, 512
  %459 = trunc i64 %458 to i32
  %460 = add i32 %206, %459
  %461 = inttoptr i32 %460 to ptr addrspace(3)
  %462 = load <4 x i32>, ptr addrspace(3) %461, align 16
  %463 = add i64 %454, 1024
  %464 = trunc i64 %463 to i32
  %465 = add i32 %206, %464
  %466 = inttoptr i32 %465 to ptr addrspace(3)
  %467 = load <4 x i32>, ptr addrspace(3) %466, align 16
  %468 = add i64 %454, 1536
  %469 = trunc i64 %468 to i32
  %470 = add i32 %206, %469
  %471 = inttoptr i32 %470 to ptr addrspace(3)
  %472 = load <4 x i32>, ptr addrspace(3) %471, align 16
  %473 = shufflevector <4 x i32> %457, <4 x i32> %462, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %474 = shufflevector <4 x i32> %467, <4 x i32> %472, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %475 = shufflevector <8 x i32> %473, <8 x i32> %474, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %476 = add i32 %245, 10496
  %477 = add i32 %476, %218
  %478 = add i32 %477, %220
  %479 = sext i32 %478 to i64
  %480 = add i32 %206, %478
  %481 = inttoptr i32 %480 to ptr addrspace(3)
  %482 = load <4 x i32>, ptr addrspace(3) %481, align 16
  %483 = add i64 %479, 512
  %484 = trunc i64 %483 to i32
  %485 = add i32 %206, %484
  %486 = inttoptr i32 %485 to ptr addrspace(3)
  %487 = load <4 x i32>, ptr addrspace(3) %486, align 16
  %488 = add i64 %479, 1024
  %489 = trunc i64 %488 to i32
  %490 = add i32 %206, %489
  %491 = inttoptr i32 %490 to ptr addrspace(3)
  %492 = load <4 x i32>, ptr addrspace(3) %491, align 16
  %493 = add i64 %479, 1536
  %494 = trunc i64 %493 to i32
  %495 = add i32 %206, %494
  %496 = inttoptr i32 %495 to ptr addrspace(3)
  %497 = load <4 x i32>, ptr addrspace(3) %496, align 16
  %498 = shufflevector <4 x i32> %482, <4 x i32> %487, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %499 = shufflevector <4 x i32> %492, <4 x i32> %497, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %500 = shufflevector <8 x i32> %498, <8 x i32> %499, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %501 = add i32 %279, 74049
  %502 = add i32 %204, %501
  %503 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %502
  %504 = load i8, ptr addrspace(3) %503, align 1
  %505 = zext i8 %504 to i32
  %506 = call i32 @llvm.amdgcn.perm(i32 %505, i32 %505, i32 0)
  %507 = add i32 %295, 74049
  %508 = add i32 %204, %507
  %509 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %508
  %510 = load i8, ptr addrspace(3) %509, align 1
  %511 = zext i8 %510 to i32
  %512 = call i32 @llvm.amdgcn.perm(i32 %511, i32 %511, i32 0)
  %513 = add i32 %302, 74000
  %514 = add i32 %204, %513
  %515 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %514
  %516 = load i8, ptr addrspace(3) %515, align 1
  %517 = zext i8 %516 to i32
  %518 = call i32 @llvm.amdgcn.perm(i32 %517, i32 %517, i32 0)
  %519 = add i32 %309, 128
  %520 = add i32 %519, %310
  %521 = sext i32 %520 to i64
  %522 = add i32 %206, %520
  %523 = inttoptr i32 %522 to ptr addrspace(3)
  %524 = load <4 x i32>, ptr addrspace(3) %523, align 16
  %525 = add i64 %521, 32
  %526 = trunc i64 %525 to i32
  %527 = add i32 %206, %526
  %528 = inttoptr i32 %527 to ptr addrspace(3)
  %529 = load <4 x i32>, ptr addrspace(3) %528, align 16
  %530 = add i64 %521, 64
  %531 = trunc i64 %530 to i32
  %532 = add i32 %206, %531
  %533 = inttoptr i32 %532 to ptr addrspace(3)
  %534 = load <4 x i32>, ptr addrspace(3) %533, align 16
  %535 = add i64 %521, 96
  %536 = trunc i64 %535 to i32
  %537 = add i32 %206, %536
  %538 = inttoptr i32 %537 to ptr addrspace(3)
  %539 = load <4 x i32>, ptr addrspace(3) %538, align 16
  %540 = shufflevector <4 x i32> %524, <4 x i32> %529, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %541 = shufflevector <4 x i32> %534, <4 x i32> %539, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %542 = shufflevector <8 x i32> %540, <8 x i32> %541, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %543 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %475, i32 0, <16 x i32> %542, i16 0, <8 x float> %334, i32 0, i32 0, i32 %506, i32 0, i32 0, i32 %518, i1 false, i1 false)
  %544 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %500, i32 0, <16 x i32> %542, i16 0, <8 x float> %335, i32 0, i32 0, i32 %512, i32 0, i32 0, i32 %518, i1 false, i1 false)
  %545 = add i32 %216, 12544
  %546 = add i32 %545, %218
  %547 = add i32 %546, %220
  %548 = sext i32 %547 to i64
  %549 = add i32 %206, %547
  %550 = inttoptr i32 %549 to ptr addrspace(3)
  %551 = load <4 x i32>, ptr addrspace(3) %550, align 16
  %552 = add i64 %548, 512
  %553 = trunc i64 %552 to i32
  %554 = add i32 %206, %553
  %555 = inttoptr i32 %554 to ptr addrspace(3)
  %556 = load <4 x i32>, ptr addrspace(3) %555, align 16
  %557 = add i64 %548, 1024
  %558 = trunc i64 %557 to i32
  %559 = add i32 %206, %558
  %560 = inttoptr i32 %559 to ptr addrspace(3)
  %561 = load <4 x i32>, ptr addrspace(3) %560, align 16
  %562 = add i64 %548, 1536
  %563 = trunc i64 %562 to i32
  %564 = add i32 %206, %563
  %565 = inttoptr i32 %564 to ptr addrspace(3)
  %566 = load <4 x i32>, ptr addrspace(3) %565, align 16
  %567 = shufflevector <4 x i32> %551, <4 x i32> %556, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %568 = shufflevector <4 x i32> %561, <4 x i32> %566, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %569 = shufflevector <8 x i32> %567, <8 x i32> %568, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %570 = add i32 %245, 12544
  %571 = add i32 %570, %218
  %572 = add i32 %571, %220
  %573 = sext i32 %572 to i64
  %574 = add i32 %206, %572
  %575 = inttoptr i32 %574 to ptr addrspace(3)
  %576 = load <4 x i32>, ptr addrspace(3) %575, align 16
  %577 = add i64 %573, 512
  %578 = trunc i64 %577 to i32
  %579 = add i32 %206, %578
  %580 = inttoptr i32 %579 to ptr addrspace(3)
  %581 = load <4 x i32>, ptr addrspace(3) %580, align 16
  %582 = add i64 %573, 1024
  %583 = trunc i64 %582 to i32
  %584 = add i32 %206, %583
  %585 = inttoptr i32 %584 to ptr addrspace(3)
  %586 = load <4 x i32>, ptr addrspace(3) %585, align 16
  %587 = add i64 %573, 1536
  %588 = trunc i64 %587 to i32
  %589 = add i32 %206, %588
  %590 = inttoptr i32 %589 to ptr addrspace(3)
  %591 = load <4 x i32>, ptr addrspace(3) %590, align 16
  %592 = shufflevector <4 x i32> %576, <4 x i32> %581, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %593 = shufflevector <4 x i32> %586, <4 x i32> %591, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %594 = shufflevector <8 x i32> %592, <8 x i32> %593, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %595 = add i32 %279, 74050
  %596 = add i32 %204, %595
  %597 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %596
  %598 = load i8, ptr addrspace(3) %597, align 1
  %599 = zext i8 %598 to i32
  %600 = call i32 @llvm.amdgcn.perm(i32 %599, i32 %599, i32 0)
  %601 = add i32 %295, 74050
  %602 = add i32 %204, %601
  %603 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %602
  %604 = load i8, ptr addrspace(3) %603, align 1
  %605 = zext i8 %604 to i32
  %606 = call i32 @llvm.amdgcn.perm(i32 %605, i32 %605, i32 0)
  %607 = add i32 %302, 74016
  %608 = add i32 %204, %607
  %609 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %608
  %610 = load i8, ptr addrspace(3) %609, align 1
  %611 = zext i8 %610 to i32
  %612 = call i32 @llvm.amdgcn.perm(i32 %611, i32 %611, i32 0)
  %613 = add i32 %309, 256
  %614 = add i32 %613, %310
  %615 = sext i32 %614 to i64
  %616 = add i32 %206, %614
  %617 = inttoptr i32 %616 to ptr addrspace(3)
  %618 = load <4 x i32>, ptr addrspace(3) %617, align 16
  %619 = add i64 %615, 32
  %620 = trunc i64 %619 to i32
  %621 = add i32 %206, %620
  %622 = inttoptr i32 %621 to ptr addrspace(3)
  %623 = load <4 x i32>, ptr addrspace(3) %622, align 16
  %624 = add i64 %615, 64
  %625 = trunc i64 %624 to i32
  %626 = add i32 %206, %625
  %627 = inttoptr i32 %626 to ptr addrspace(3)
  %628 = load <4 x i32>, ptr addrspace(3) %627, align 16
  %629 = add i64 %615, 96
  %630 = trunc i64 %629 to i32
  %631 = add i32 %206, %630
  %632 = inttoptr i32 %631 to ptr addrspace(3)
  %633 = load <4 x i32>, ptr addrspace(3) %632, align 16
  %634 = shufflevector <4 x i32> %618, <4 x i32> %623, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %635 = shufflevector <4 x i32> %628, <4 x i32> %633, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %636 = shufflevector <8 x i32> %634, <8 x i32> %635, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %637 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %569, i32 0, <16 x i32> %636, i16 0, <8 x float> %543, i32 0, i32 0, i32 %600, i32 0, i32 0, i32 %612, i1 false, i1 false)
  %638 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %594, i32 0, <16 x i32> %636, i16 0, <8 x float> %544, i32 0, i32 0, i32 %606, i32 0, i32 0, i32 %612, i1 false, i1 false)
  %639 = add i32 %216, 14592
  %640 = add i32 %639, %218
  %641 = add i32 %640, %220
  %642 = sext i32 %641 to i64
  %643 = add i32 %206, %641
  %644 = inttoptr i32 %643 to ptr addrspace(3)
  %645 = load <4 x i32>, ptr addrspace(3) %644, align 16
  %646 = add i64 %642, 512
  %647 = trunc i64 %646 to i32
  %648 = add i32 %206, %647
  %649 = inttoptr i32 %648 to ptr addrspace(3)
  %650 = load <4 x i32>, ptr addrspace(3) %649, align 16
  %651 = add i64 %642, 1024
  %652 = trunc i64 %651 to i32
  %653 = add i32 %206, %652
  %654 = inttoptr i32 %653 to ptr addrspace(3)
  %655 = load <4 x i32>, ptr addrspace(3) %654, align 16
  %656 = add i64 %642, 1536
  %657 = trunc i64 %656 to i32
  %658 = add i32 %206, %657
  %659 = inttoptr i32 %658 to ptr addrspace(3)
  %660 = load <4 x i32>, ptr addrspace(3) %659, align 16
  %661 = shufflevector <4 x i32> %645, <4 x i32> %650, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %662 = shufflevector <4 x i32> %655, <4 x i32> %660, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %663 = shufflevector <8 x i32> %661, <8 x i32> %662, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %664 = add i32 %245, 14592
  %665 = add i32 %664, %218
  %666 = add i32 %665, %220
  %667 = sext i32 %666 to i64
  %668 = add i32 %206, %666
  %669 = inttoptr i32 %668 to ptr addrspace(3)
  %670 = load <4 x i32>, ptr addrspace(3) %669, align 16
  %671 = add i64 %667, 512
  %672 = trunc i64 %671 to i32
  %673 = add i32 %206, %672
  %674 = inttoptr i32 %673 to ptr addrspace(3)
  %675 = load <4 x i32>, ptr addrspace(3) %674, align 16
  %676 = add i64 %667, 1024
  %677 = trunc i64 %676 to i32
  %678 = add i32 %206, %677
  %679 = inttoptr i32 %678 to ptr addrspace(3)
  %680 = load <4 x i32>, ptr addrspace(3) %679, align 16
  %681 = add i64 %667, 1536
  %682 = trunc i64 %681 to i32
  %683 = add i32 %206, %682
  %684 = inttoptr i32 %683 to ptr addrspace(3)
  %685 = load <4 x i32>, ptr addrspace(3) %684, align 16
  %686 = shufflevector <4 x i32> %670, <4 x i32> %675, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %687 = shufflevector <4 x i32> %680, <4 x i32> %685, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %688 = shufflevector <8 x i32> %686, <8 x i32> %687, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %689 = add i32 %279, 74051
  %690 = add i32 %204, %689
  %691 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %690
  %692 = load i8, ptr addrspace(3) %691, align 1
  %693 = zext i8 %692 to i32
  %694 = call i32 @llvm.amdgcn.perm(i32 %693, i32 %693, i32 0)
  %695 = add i32 %295, 74051
  %696 = add i32 %204, %695
  %697 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %696
  %698 = load i8, ptr addrspace(3) %697, align 1
  %699 = zext i8 %698 to i32
  %700 = call i32 @llvm.amdgcn.perm(i32 %699, i32 %699, i32 0)
  %701 = add i32 %302, 74032
  %702 = add i32 %204, %701
  %703 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %702
  %704 = load i8, ptr addrspace(3) %703, align 1
  %705 = zext i8 %704 to i32
  %706 = call i32 @llvm.amdgcn.perm(i32 %705, i32 %705, i32 0)
  %707 = add i32 %309, 384
  %708 = add i32 %707, %310
  %709 = sext i32 %708 to i64
  %710 = add i32 %206, %708
  %711 = inttoptr i32 %710 to ptr addrspace(3)
  %712 = load <4 x i32>, ptr addrspace(3) %711, align 16
  %713 = add i64 %709, 32
  %714 = trunc i64 %713 to i32
  %715 = add i32 %206, %714
  %716 = inttoptr i32 %715 to ptr addrspace(3)
  %717 = load <4 x i32>, ptr addrspace(3) %716, align 16
  %718 = add i64 %709, 64
  %719 = trunc i64 %718 to i32
  %720 = add i32 %206, %719
  %721 = inttoptr i32 %720 to ptr addrspace(3)
  %722 = load <4 x i32>, ptr addrspace(3) %721, align 16
  %723 = add i64 %709, 96
  %724 = trunc i64 %723 to i32
  %725 = add i32 %206, %724
  %726 = inttoptr i32 %725 to ptr addrspace(3)
  %727 = load <4 x i32>, ptr addrspace(3) %726, align 16
  %728 = shufflevector <4 x i32> %712, <4 x i32> %717, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %729 = shufflevector <4 x i32> %722, <4 x i32> %727, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %730 = shufflevector <8 x i32> %728, <8 x i32> %729, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %731 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %663, i32 0, <16 x i32> %730, i16 0, <8 x float> %637, i32 0, i32 0, i32 %694, i32 0, i32 0, i32 %706, i1 false, i1 false)
  %732 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %688, i32 0, <16 x i32> %730, i16 0, <8 x float> %638, i32 0, i32 0, i32 %700, i32 0, i32 0, i32 %706, i1 false, i1 false)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 15, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 11, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 4, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 11, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 4, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 11, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 4, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  br label %197

733:                                              ; preds = %197
  %734 = srem i32 %196, 2
  %735 = mul i32 %734, 74752
  %736 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %735
  %737 = ptrtoint ptr addrspace(3) %736 to i32
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %738 = sdiv i32 %99, 16
  %739 = mul i32 %738, 16
  %740 = icmp ne i32 %99, %739
  %741 = icmp slt i32 %99, 0
  %742 = icmp ne i1 %741, false
  %743 = and i1 %740, %742
  %744 = add i32 %738, -1
  %745 = select i1 %743, i32 %744, i32 %738
  %746 = mul i32 %745, 8192
  %747 = add i32 %746, 8448
  %748 = mul i32 %50, 256
  %749 = add i32 %747, %748
  %750 = mul i32 %42, 16
  %751 = add i32 %749, %750
  %752 = sext i32 %751 to i64
  %753 = add i32 %737, %751
  %754 = inttoptr i32 %753 to ptr addrspace(3)
  %755 = load <4 x i32>, ptr addrspace(3) %754, align 16
  %756 = add i64 %752, 512
  %757 = trunc i64 %756 to i32
  %758 = add i32 %737, %757
  %759 = inttoptr i32 %758 to ptr addrspace(3)
  %760 = load <4 x i32>, ptr addrspace(3) %759, align 16
  %761 = add i64 %752, 1024
  %762 = trunc i64 %761 to i32
  %763 = add i32 %737, %762
  %764 = inttoptr i32 %763 to ptr addrspace(3)
  %765 = load <4 x i32>, ptr addrspace(3) %764, align 16
  %766 = add i64 %752, 1536
  %767 = trunc i64 %766 to i32
  %768 = add i32 %737, %767
  %769 = inttoptr i32 %768 to ptr addrspace(3)
  %770 = load <4 x i32>, ptr addrspace(3) %769, align 16
  %771 = shufflevector <4 x i32> %755, <4 x i32> %760, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %772 = shufflevector <4 x i32> %765, <4 x i32> %770, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %773 = shufflevector <8 x i32> %771, <8 x i32> %772, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %774 = add i32 %745, 1
  %775 = mul i32 %774, 8192
  %776 = add i32 %775, 8448
  %777 = add i32 %776, %748
  %778 = add i32 %777, %750
  %779 = sext i32 %778 to i64
  %780 = add i32 %737, %778
  %781 = inttoptr i32 %780 to ptr addrspace(3)
  %782 = load <4 x i32>, ptr addrspace(3) %781, align 16
  %783 = add i64 %779, 512
  %784 = trunc i64 %783 to i32
  %785 = add i32 %737, %784
  %786 = inttoptr i32 %785 to ptr addrspace(3)
  %787 = load <4 x i32>, ptr addrspace(3) %786, align 16
  %788 = add i64 %779, 1024
  %789 = trunc i64 %788 to i32
  %790 = add i32 %737, %789
  %791 = inttoptr i32 %790 to ptr addrspace(3)
  %792 = load <4 x i32>, ptr addrspace(3) %791, align 16
  %793 = add i64 %779, 1536
  %794 = trunc i64 %793 to i32
  %795 = add i32 %737, %794
  %796 = inttoptr i32 %795 to ptr addrspace(3)
  %797 = load <4 x i32>, ptr addrspace(3) %796, align 16
  %798 = shufflevector <4 x i32> %782, <4 x i32> %787, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %799 = shufflevector <4 x i32> %792, <4 x i32> %797, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %800 = shufflevector <8 x i32> %798, <8 x i32> %799, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %801 = sdiv i32 %99, 128
  %802 = mul i32 %801, 128
  %803 = icmp ne i32 %99, %802
  %804 = icmp slt i32 %99, 0
  %805 = icmp ne i1 %804, false
  %806 = and i1 %803, %805
  %807 = add i32 %801, -1
  %808 = select i1 %806, i32 %807, i32 %801
  %809 = mul i32 %808, 4
  %810 = add i32 %809, 74048
  %811 = add i32 %735, %810
  %812 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %811
  %813 = load i8, ptr addrspace(3) %812, align 1
  %814 = zext i8 %813 to i32
  %815 = call i32 @llvm.amdgcn.perm(i32 %814, i32 %814, i32 0)
  %816 = add i32 %99, 16
  %817 = sdiv i32 %816, 128
  %818 = mul i32 %817, 128
  %819 = icmp ne i32 %816, %818
  %820 = icmp slt i32 %816, 0
  %821 = icmp ne i1 %820, false
  %822 = and i1 %819, %821
  %823 = add i32 %817, -1
  %824 = select i1 %822, i32 %823, i32 %817
  %825 = mul i32 %824, 4
  %826 = add i32 %825, 74048
  %827 = add i32 %735, %826
  %828 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %827
  %829 = load i8, ptr addrspace(3) %828, align 1
  %830 = zext i8 %829 to i32
  %831 = call i32 @llvm.amdgcn.perm(i32 %830, i32 %830, i32 0)
  %832 = add i32 %98, %42
  %833 = add i32 %832, 73984
  %834 = add i32 %735, %833
  %835 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %834
  %836 = load i8, ptr addrspace(3) %835, align 1
  %837 = zext i8 %836 to i32
  %838 = call i32 @llvm.amdgcn.perm(i32 %837, i32 %837, i32 0)
  %839 = mul i32 %832, 528
  %840 = mul i32 %50, 16
  %841 = add i32 %839, %840
  %842 = sext i32 %841 to i64
  %843 = add i32 %737, %841
  %844 = inttoptr i32 %843 to ptr addrspace(3)
  %845 = load <4 x i32>, ptr addrspace(3) %844, align 16
  %846 = add i64 %842, 32
  %847 = trunc i64 %846 to i32
  %848 = add i32 %737, %847
  %849 = inttoptr i32 %848 to ptr addrspace(3)
  %850 = load <4 x i32>, ptr addrspace(3) %849, align 16
  %851 = add i64 %842, 64
  %852 = trunc i64 %851 to i32
  %853 = add i32 %737, %852
  %854 = inttoptr i32 %853 to ptr addrspace(3)
  %855 = load <4 x i32>, ptr addrspace(3) %854, align 16
  %856 = add i64 %842, 96
  %857 = trunc i64 %856 to i32
  %858 = add i32 %737, %857
  %859 = inttoptr i32 %858 to ptr addrspace(3)
  %860 = load <4 x i32>, ptr addrspace(3) %859, align 16
  %861 = shufflevector <4 x i32> %845, <4 x i32> %850, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %862 = shufflevector <4 x i32> %855, <4 x i32> %860, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %863 = shufflevector <8 x i32> %861, <8 x i32> %862, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %864 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %773, i32 0, <16 x i32> %863, i16 0, <8 x float> %199, i32 0, i32 0, i32 %815, i32 0, i32 0, i32 %838, i1 false, i1 false)
  %865 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %800, i32 0, <16 x i32> %863, i16 0, <8 x float> %200, i32 0, i32 0, i32 %831, i32 0, i32 0, i32 %838, i1 false, i1 false)
  %866 = add i32 %746, 10496
  %867 = add i32 %866, %748
  %868 = add i32 %867, %750
  %869 = sext i32 %868 to i64
  %870 = add i32 %737, %868
  %871 = inttoptr i32 %870 to ptr addrspace(3)
  %872 = load <4 x i32>, ptr addrspace(3) %871, align 16
  %873 = add i64 %869, 512
  %874 = trunc i64 %873 to i32
  %875 = add i32 %737, %874
  %876 = inttoptr i32 %875 to ptr addrspace(3)
  %877 = load <4 x i32>, ptr addrspace(3) %876, align 16
  %878 = add i64 %869, 1024
  %879 = trunc i64 %878 to i32
  %880 = add i32 %737, %879
  %881 = inttoptr i32 %880 to ptr addrspace(3)
  %882 = load <4 x i32>, ptr addrspace(3) %881, align 16
  %883 = add i64 %869, 1536
  %884 = trunc i64 %883 to i32
  %885 = add i32 %737, %884
  %886 = inttoptr i32 %885 to ptr addrspace(3)
  %887 = load <4 x i32>, ptr addrspace(3) %886, align 16
  %888 = shufflevector <4 x i32> %872, <4 x i32> %877, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %889 = shufflevector <4 x i32> %882, <4 x i32> %887, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %890 = shufflevector <8 x i32> %888, <8 x i32> %889, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %891 = add i32 %775, 10496
  %892 = add i32 %891, %748
  %893 = add i32 %892, %750
  %894 = sext i32 %893 to i64
  %895 = add i32 %737, %893
  %896 = inttoptr i32 %895 to ptr addrspace(3)
  %897 = load <4 x i32>, ptr addrspace(3) %896, align 16
  %898 = add i64 %894, 512
  %899 = trunc i64 %898 to i32
  %900 = add i32 %737, %899
  %901 = inttoptr i32 %900 to ptr addrspace(3)
  %902 = load <4 x i32>, ptr addrspace(3) %901, align 16
  %903 = add i64 %894, 1024
  %904 = trunc i64 %903 to i32
  %905 = add i32 %737, %904
  %906 = inttoptr i32 %905 to ptr addrspace(3)
  %907 = load <4 x i32>, ptr addrspace(3) %906, align 16
  %908 = add i64 %894, 1536
  %909 = trunc i64 %908 to i32
  %910 = add i32 %737, %909
  %911 = inttoptr i32 %910 to ptr addrspace(3)
  %912 = load <4 x i32>, ptr addrspace(3) %911, align 16
  %913 = shufflevector <4 x i32> %897, <4 x i32> %902, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %914 = shufflevector <4 x i32> %907, <4 x i32> %912, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %915 = shufflevector <8 x i32> %913, <8 x i32> %914, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %916 = add i32 %809, 74049
  %917 = add i32 %735, %916
  %918 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %917
  %919 = load i8, ptr addrspace(3) %918, align 1
  %920 = zext i8 %919 to i32
  %921 = call i32 @llvm.amdgcn.perm(i32 %920, i32 %920, i32 0)
  %922 = add i32 %825, 74049
  %923 = add i32 %735, %922
  %924 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %923
  %925 = load i8, ptr addrspace(3) %924, align 1
  %926 = zext i8 %925 to i32
  %927 = call i32 @llvm.amdgcn.perm(i32 %926, i32 %926, i32 0)
  %928 = add i32 %832, 74000
  %929 = add i32 %735, %928
  %930 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %929
  %931 = load i8, ptr addrspace(3) %930, align 1
  %932 = zext i8 %931 to i32
  %933 = call i32 @llvm.amdgcn.perm(i32 %932, i32 %932, i32 0)
  %934 = add i32 %839, 128
  %935 = add i32 %934, %840
  %936 = sext i32 %935 to i64
  %937 = add i32 %737, %935
  %938 = inttoptr i32 %937 to ptr addrspace(3)
  %939 = load <4 x i32>, ptr addrspace(3) %938, align 16
  %940 = add i64 %936, 32
  %941 = trunc i64 %940 to i32
  %942 = add i32 %737, %941
  %943 = inttoptr i32 %942 to ptr addrspace(3)
  %944 = load <4 x i32>, ptr addrspace(3) %943, align 16
  %945 = add i64 %936, 64
  %946 = trunc i64 %945 to i32
  %947 = add i32 %737, %946
  %948 = inttoptr i32 %947 to ptr addrspace(3)
  %949 = load <4 x i32>, ptr addrspace(3) %948, align 16
  %950 = add i64 %936, 96
  %951 = trunc i64 %950 to i32
  %952 = add i32 %737, %951
  %953 = inttoptr i32 %952 to ptr addrspace(3)
  %954 = load <4 x i32>, ptr addrspace(3) %953, align 16
  %955 = shufflevector <4 x i32> %939, <4 x i32> %944, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %956 = shufflevector <4 x i32> %949, <4 x i32> %954, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %957 = shufflevector <8 x i32> %955, <8 x i32> %956, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %958 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %890, i32 0, <16 x i32> %957, i16 0, <8 x float> %864, i32 0, i32 0, i32 %921, i32 0, i32 0, i32 %933, i1 false, i1 false)
  %959 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %915, i32 0, <16 x i32> %957, i16 0, <8 x float> %865, i32 0, i32 0, i32 %927, i32 0, i32 0, i32 %933, i1 false, i1 false)
  %960 = add i32 %746, 12544
  %961 = add i32 %960, %748
  %962 = add i32 %961, %750
  %963 = sext i32 %962 to i64
  %964 = add i32 %737, %962
  %965 = inttoptr i32 %964 to ptr addrspace(3)
  %966 = load <4 x i32>, ptr addrspace(3) %965, align 16
  %967 = add i64 %963, 512
  %968 = trunc i64 %967 to i32
  %969 = add i32 %737, %968
  %970 = inttoptr i32 %969 to ptr addrspace(3)
  %971 = load <4 x i32>, ptr addrspace(3) %970, align 16
  %972 = add i64 %963, 1024
  %973 = trunc i64 %972 to i32
  %974 = add i32 %737, %973
  %975 = inttoptr i32 %974 to ptr addrspace(3)
  %976 = load <4 x i32>, ptr addrspace(3) %975, align 16
  %977 = add i64 %963, 1536
  %978 = trunc i64 %977 to i32
  %979 = add i32 %737, %978
  %980 = inttoptr i32 %979 to ptr addrspace(3)
  %981 = load <4 x i32>, ptr addrspace(3) %980, align 16
  %982 = shufflevector <4 x i32> %966, <4 x i32> %971, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %983 = shufflevector <4 x i32> %976, <4 x i32> %981, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %984 = shufflevector <8 x i32> %982, <8 x i32> %983, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %985 = add i32 %775, 12544
  %986 = add i32 %985, %748
  %987 = add i32 %986, %750
  %988 = sext i32 %987 to i64
  %989 = add i32 %737, %987
  %990 = inttoptr i32 %989 to ptr addrspace(3)
  %991 = load <4 x i32>, ptr addrspace(3) %990, align 16
  %992 = add i64 %988, 512
  %993 = trunc i64 %992 to i32
  %994 = add i32 %737, %993
  %995 = inttoptr i32 %994 to ptr addrspace(3)
  %996 = load <4 x i32>, ptr addrspace(3) %995, align 16
  %997 = add i64 %988, 1024
  %998 = trunc i64 %997 to i32
  %999 = add i32 %737, %998
  %1000 = inttoptr i32 %999 to ptr addrspace(3)
  %1001 = load <4 x i32>, ptr addrspace(3) %1000, align 16
  %1002 = add i64 %988, 1536
  %1003 = trunc i64 %1002 to i32
  %1004 = add i32 %737, %1003
  %1005 = inttoptr i32 %1004 to ptr addrspace(3)
  %1006 = load <4 x i32>, ptr addrspace(3) %1005, align 16
  %1007 = shufflevector <4 x i32> %991, <4 x i32> %996, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1008 = shufflevector <4 x i32> %1001, <4 x i32> %1006, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1009 = shufflevector <8 x i32> %1007, <8 x i32> %1008, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1010 = add i32 %809, 74050
  %1011 = add i32 %735, %1010
  %1012 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1011
  %1013 = load i8, ptr addrspace(3) %1012, align 1
  %1014 = zext i8 %1013 to i32
  %1015 = call i32 @llvm.amdgcn.perm(i32 %1014, i32 %1014, i32 0)
  %1016 = add i32 %825, 74050
  %1017 = add i32 %735, %1016
  %1018 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1017
  %1019 = load i8, ptr addrspace(3) %1018, align 1
  %1020 = zext i8 %1019 to i32
  %1021 = call i32 @llvm.amdgcn.perm(i32 %1020, i32 %1020, i32 0)
  %1022 = add i32 %832, 74016
  %1023 = add i32 %735, %1022
  %1024 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1023
  %1025 = load i8, ptr addrspace(3) %1024, align 1
  %1026 = zext i8 %1025 to i32
  %1027 = call i32 @llvm.amdgcn.perm(i32 %1026, i32 %1026, i32 0)
  %1028 = add i32 %839, 256
  %1029 = add i32 %1028, %840
  %1030 = sext i32 %1029 to i64
  %1031 = add i32 %737, %1029
  %1032 = inttoptr i32 %1031 to ptr addrspace(3)
  %1033 = load <4 x i32>, ptr addrspace(3) %1032, align 16
  %1034 = add i64 %1030, 32
  %1035 = trunc i64 %1034 to i32
  %1036 = add i32 %737, %1035
  %1037 = inttoptr i32 %1036 to ptr addrspace(3)
  %1038 = load <4 x i32>, ptr addrspace(3) %1037, align 16
  %1039 = add i64 %1030, 64
  %1040 = trunc i64 %1039 to i32
  %1041 = add i32 %737, %1040
  %1042 = inttoptr i32 %1041 to ptr addrspace(3)
  %1043 = load <4 x i32>, ptr addrspace(3) %1042, align 16
  %1044 = add i64 %1030, 96
  %1045 = trunc i64 %1044 to i32
  %1046 = add i32 %737, %1045
  %1047 = inttoptr i32 %1046 to ptr addrspace(3)
  %1048 = load <4 x i32>, ptr addrspace(3) %1047, align 16
  %1049 = shufflevector <4 x i32> %1033, <4 x i32> %1038, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1050 = shufflevector <4 x i32> %1043, <4 x i32> %1048, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1051 = shufflevector <8 x i32> %1049, <8 x i32> %1050, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1052 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %984, i32 0, <16 x i32> %1051, i16 0, <8 x float> %958, i32 0, i32 0, i32 %1015, i32 0, i32 0, i32 %1027, i1 false, i1 false)
  %1053 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1009, i32 0, <16 x i32> %1051, i16 0, <8 x float> %959, i32 0, i32 0, i32 %1021, i32 0, i32 0, i32 %1027, i1 false, i1 false)
  %1054 = add i32 %746, 14592
  %1055 = add i32 %1054, %748
  %1056 = add i32 %1055, %750
  %1057 = sext i32 %1056 to i64
  %1058 = add i32 %737, %1056
  %1059 = inttoptr i32 %1058 to ptr addrspace(3)
  %1060 = load <4 x i32>, ptr addrspace(3) %1059, align 16
  %1061 = add i64 %1057, 512
  %1062 = trunc i64 %1061 to i32
  %1063 = add i32 %737, %1062
  %1064 = inttoptr i32 %1063 to ptr addrspace(3)
  %1065 = load <4 x i32>, ptr addrspace(3) %1064, align 16
  %1066 = add i64 %1057, 1024
  %1067 = trunc i64 %1066 to i32
  %1068 = add i32 %737, %1067
  %1069 = inttoptr i32 %1068 to ptr addrspace(3)
  %1070 = load <4 x i32>, ptr addrspace(3) %1069, align 16
  %1071 = add i64 %1057, 1536
  %1072 = trunc i64 %1071 to i32
  %1073 = add i32 %737, %1072
  %1074 = inttoptr i32 %1073 to ptr addrspace(3)
  %1075 = load <4 x i32>, ptr addrspace(3) %1074, align 16
  %1076 = shufflevector <4 x i32> %1060, <4 x i32> %1065, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1077 = shufflevector <4 x i32> %1070, <4 x i32> %1075, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1078 = shufflevector <8 x i32> %1076, <8 x i32> %1077, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1079 = add i32 %775, 14592
  %1080 = add i32 %1079, %748
  %1081 = add i32 %1080, %750
  %1082 = sext i32 %1081 to i64
  %1083 = add i32 %737, %1081
  %1084 = inttoptr i32 %1083 to ptr addrspace(3)
  %1085 = load <4 x i32>, ptr addrspace(3) %1084, align 16
  %1086 = add i64 %1082, 512
  %1087 = trunc i64 %1086 to i32
  %1088 = add i32 %737, %1087
  %1089 = inttoptr i32 %1088 to ptr addrspace(3)
  %1090 = load <4 x i32>, ptr addrspace(3) %1089, align 16
  %1091 = add i64 %1082, 1024
  %1092 = trunc i64 %1091 to i32
  %1093 = add i32 %737, %1092
  %1094 = inttoptr i32 %1093 to ptr addrspace(3)
  %1095 = load <4 x i32>, ptr addrspace(3) %1094, align 16
  %1096 = add i64 %1082, 1536
  %1097 = trunc i64 %1096 to i32
  %1098 = add i32 %737, %1097
  %1099 = inttoptr i32 %1098 to ptr addrspace(3)
  %1100 = load <4 x i32>, ptr addrspace(3) %1099, align 16
  %1101 = shufflevector <4 x i32> %1085, <4 x i32> %1090, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1102 = shufflevector <4 x i32> %1095, <4 x i32> %1100, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1103 = shufflevector <8 x i32> %1101, <8 x i32> %1102, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1104 = add i32 %809, 74051
  %1105 = add i32 %735, %1104
  %1106 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1105
  %1107 = load i8, ptr addrspace(3) %1106, align 1
  %1108 = zext i8 %1107 to i32
  %1109 = call i32 @llvm.amdgcn.perm(i32 %1108, i32 %1108, i32 0)
  %1110 = add i32 %825, 74051
  %1111 = add i32 %735, %1110
  %1112 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1111
  %1113 = load i8, ptr addrspace(3) %1112, align 1
  %1114 = zext i8 %1113 to i32
  %1115 = call i32 @llvm.amdgcn.perm(i32 %1114, i32 %1114, i32 0)
  %1116 = add i32 %832, 74032
  %1117 = add i32 %735, %1116
  %1118 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1117
  %1119 = load i8, ptr addrspace(3) %1118, align 1
  %1120 = zext i8 %1119 to i32
  %1121 = call i32 @llvm.amdgcn.perm(i32 %1120, i32 %1120, i32 0)
  %1122 = add i32 %839, 384
  %1123 = add i32 %1122, %840
  %1124 = sext i32 %1123 to i64
  %1125 = add i32 %737, %1123
  %1126 = inttoptr i32 %1125 to ptr addrspace(3)
  %1127 = load <4 x i32>, ptr addrspace(3) %1126, align 16
  %1128 = add i64 %1124, 32
  %1129 = trunc i64 %1128 to i32
  %1130 = add i32 %737, %1129
  %1131 = inttoptr i32 %1130 to ptr addrspace(3)
  %1132 = load <4 x i32>, ptr addrspace(3) %1131, align 16
  %1133 = add i64 %1124, 64
  %1134 = trunc i64 %1133 to i32
  %1135 = add i32 %737, %1134
  %1136 = inttoptr i32 %1135 to ptr addrspace(3)
  %1137 = load <4 x i32>, ptr addrspace(3) %1136, align 16
  %1138 = add i64 %1124, 96
  %1139 = trunc i64 %1138 to i32
  %1140 = add i32 %737, %1139
  %1141 = inttoptr i32 %1140 to ptr addrspace(3)
  %1142 = load <4 x i32>, ptr addrspace(3) %1141, align 16
  %1143 = shufflevector <4 x i32> %1127, <4 x i32> %1132, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1144 = shufflevector <4 x i32> %1137, <4 x i32> %1142, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1145 = shufflevector <8 x i32> %1143, <8 x i32> %1144, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1146 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1078, i32 0, <16 x i32> %1145, i16 0, <8 x float> %1052, i32 0, i32 0, i32 %1109, i32 0, i32 0, i32 %1121, i1 false, i1 false)
  %1147 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1103, i32 0, <16 x i32> %1145, i16 0, <8 x float> %1053, i32 0, i32 0, i32 %1115, i32 0, i32 0, i32 %1121, i1 false, i1 false)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 15, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 11, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 4, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 11, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 4, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 11, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 4, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 256, i32 0, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 0, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %1148 = mul i32 %50, 8
  %1149 = add i32 %99, %1148
  %1150 = fptrunc <8 x float> %1146 to <8 x bfloat>
  %1151 = bitcast <8 x bfloat> %1150 to <16 x i8>
  %1152 = mul i32 %832, 136
  %1153 = add i32 %1152, %1149
  %1154 = mul i32 %1153, 2
  %1155 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1154
  store <16 x i8> %1151, ptr addrspace(3) %1155, align 1
  %1156 = add i32 %816, %1148
  %1157 = fptrunc <8 x float> %1147 to <8 x bfloat>
  %1158 = bitcast <8 x bfloat> %1157 to <16 x i8>
  %1159 = add i32 %1152, %1156
  %1160 = mul i32 %1159, 2
  %1161 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1160
  store <16 x i8> %1158, ptr addrspace(3) %1161, align 1
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %1162 = mul i64 %62, %22
  %1163 = add i64 %1162, %63
  %1164 = getelementptr bfloat, ptr addrspace(1) %0, i64 %1163
  %1165 = icmp eq i64 %22, -2147483648
  %1166 = select i1 %1165, i64 136, i64 %22
  %1167 = ptrtoint ptr addrspace(1) %1164 to i64
  %1168 = call i32 @llvm.amdgcn.wave.id()
  %1169 = urem i32 %1168, 4
  %1170 = mul i32 %1169, 4
  %1171 = zext i32 %1170 to i64
  %1172 = mul i64 %1171, %1166
  %1173 = mul i64 %1172, 2
  %1174 = add i64 %1167, %1173
  %1175 = mul i32 %1169, 1088
  %1176 = add i32 ptrtoint (ptr addrspace(3) @__dynamic_shared_0 to i32), %1175
  %1177 = trunc i64 %1174 to i32
  %1178 = lshr i64 %1174, 32
  %1179 = trunc i64 %1178 to i32
  %1180 = or i32 %1179, -2147483648
  %1181 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %1176, i64 1
  %1182 = insertelement <4 x i32> %1181, i32 %1177, i64 2
  %1183 = insertelement <4 x i32> %1182, i32 %1180, i64 3
  %1184 = sub i32 %64, %1170
  %1185 = call i32 @llvm.smax.i32(i32 %1184, i32 0)
  %1186 = and i32 %1185, 65535
  %1187 = shl i32 %1186, 16
  %1188 = lshr i32 %1185, 16
  %1189 = and i32 %1188, 65535
  %1190 = or i32 %1189, 8912896
  %1191 = trunc i64 %1166 to i32
  %1192 = lshr i64 %1166, 32
  %1193 = trunc i64 %1192 to i32
  %1194 = and i32 %1193, 65535
  %1195 = insertelement <8 x i32> <i32 65536, i32 8388608, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %1187, i64 2
  %1196 = insertelement <8 x i32> %1195, i32 %1190, i64 3
  %1197 = insertelement <8 x i32> %1196, i32 4, i64 4
  %1198 = insertelement <8 x i32> %1197, i32 %1191, i64 5
  %1199 = insertelement <8 x i32> %1198, i32 %1194, i64 6
  %1200 = insertelement <8 x i32> %1199, i32 0, i64 7
  call void @llvm.amdgcn.tensor.store.from.lds(<4 x i32> %1183, <8 x i32> %1200, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
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

attributes #0 = { "amdgpu-flat-work-group-size"="128,128" "uniform-work-group-size" }
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
!1 = !{i32 128, i32 1, i32 1}
