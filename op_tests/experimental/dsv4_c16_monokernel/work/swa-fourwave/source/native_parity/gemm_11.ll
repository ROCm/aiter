; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

@__dynamic_shared_0 = external dso_local addrspace(3) global [0 x i8], align 1024

define amdgpu_kernel void @batched_gemm_a8w8_mx128_t32x32x1024_mw2_nw2_nb4_sk1_cm1_cn1_mbn(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10) #0 !reqd_work_group_size !1 {
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
  %23 = call range(i32 0, 128) i32 @llvm.amdgcn.workitem.id.x()
  %24 = sext i32 %23 to i64
  %25 = trunc i64 %24 to i32
  %26 = call i32 @llvm.amdgcn.workgroup.id.x()
  %27 = sext i32 %26 to i64
  %28 = trunc i64 %27 to i32
  %29 = call i32 @llvm.amdgcn.workgroup.id.y()
  %30 = sext i32 %29 to i64
  %31 = trunc i64 %30 to i32
  %32 = call i32 @llvm.amdgcn.workgroup.id.z()
  %33 = sext i32 %32 to i64
  %34 = trunc i64 %33 to i32
  %35 = sext i32 %34 to i64
  %36 = sdiv i32 %25, 32
  %37 = mul i32 %36, 32
  %38 = icmp ne i32 %25, %37
  %39 = icmp slt i32 %25, 0
  %40 = icmp ne i1 %39, false
  %41 = and i1 %38, %40
  %42 = add i32 %36, -1
  %43 = select i1 %41, i32 %42, i32 %36
  %44 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %43)
  %45 = srem i32 %25, 32
  %46 = srem i32 %45, 16
  %47 = sdiv i32 %45, 16
  %48 = mul i32 %47, 16
  %49 = icmp ne i32 %45, %48
  %50 = icmp slt i32 %45, 0
  %51 = icmp ne i1 %50, false
  %52 = and i1 %49, %51
  %53 = add i32 %47, -1
  %54 = select i1 %52, i32 %53, i32 %47
  %55 = sdiv i32 %44, 2
  %56 = mul i32 %55, 2
  %57 = icmp ne i32 %44, %56
  %58 = icmp slt i32 %44, 0
  %59 = icmp ne i1 %58, false
  %60 = and i1 %57, %59
  %61 = add i32 %55, -1
  %62 = select i1 %60, i32 %61, i32 %55
  %63 = srem i32 %44, 2
  %64 = mul i32 %28, 32
  %65 = mul i32 %31, 32
  %66 = sext i32 %64 to i64
  %67 = sext i32 %65 to i64
  %68 = sub i32 %5, %64
  %69 = sext i32 %8 to i64
  %70 = mul i64 %66, %21
  %71 = sdiv i64 %67, 16
  %72 = mul i64 %71, 16
  %73 = icmp ne i64 %67, %72
  %74 = icmp slt i64 %67, 0
  %75 = icmp ne i1 %74, false
  %76 = and i1 %73, %75
  %77 = add i64 %71, -1
  %78 = select i1 %76, i64 %77, i64 %71
  %79 = mul i64 %20, 16
  %80 = mul i64 %78, %79
  %81 = mul i64 %35, %20
  %82 = add i64 %70, %81
  %83 = sext i32 %6 to i64
  %84 = mul i64 %35, %83
  %85 = mul i64 %84, %20
  %86 = add i64 %80, %85
  %87 = getelementptr i8, ptr addrspace(1) %1, i64 %82
  %88 = getelementptr i8, ptr addrspace(1) %2, i64 %86
  %89 = sdiv i64 %67, 128
  %90 = mul i64 %89, 128
  %91 = icmp ne i64 %67, %90
  %92 = icmp slt i64 %67, 0
  %93 = icmp ne i1 %92, false
  %94 = and i1 %91, %93
  %95 = add i64 %89, -1
  %96 = select i1 %94, i64 %95, i64 %89
  %97 = sdiv i64 %20, 128
  %98 = mul i64 %97, 128
  %99 = icmp ne i64 %20, %98
  %100 = icmp slt i64 %20, 0
  %101 = icmp ne i1 %100, false
  %102 = and i1 %99, %101
  %103 = add i64 %97, -1
  %104 = select i1 %102, i64 %103, i64 %97
  %105 = mul i64 %96, %104
  %106 = sdiv i64 %83, 128
  %107 = mul i64 %106, 128
  %108 = icmp ne i64 %83, %107
  %109 = icmp slt i64 %83, 0
  %110 = icmp ne i1 %109, false
  %111 = and i1 %108, %110
  %112 = add i64 %106, -1
  %113 = select i1 %111, i64 %112, i64 %106
  %114 = mul i64 %35, %113
  %115 = mul i64 %114, %104
  %116 = add i64 %105, %115
  %117 = mul i64 %66, %69
  %118 = mul i64 %35, %104
  %119 = add i64 %117, %118
  %120 = getelementptr i8, ptr addrspace(1) %3, i64 %119
  %121 = getelementptr i8, ptr addrspace(1) %4, i64 %116
  %122 = mul i32 %62, 16
  %123 = mul i32 %63, 16
  %124 = icmp eq i32 %44, 0
  br i1 %124, label %125, label %152

125:                                              ; preds = %11
  %126 = icmp eq i64 %21, -2147483648
  %127 = select i1 %126, i64 1024, i64 %21
  %128 = ptrtoint ptr addrspace(1) %87 to i64
  %129 = trunc i64 %128 to i32
  %130 = lshr i64 %128, 32
  %131 = trunc i64 %130 to i32
  %132 = or i32 %131, -2147483648
  %133 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) @__dynamic_shared_0 to i32), i32 poison, i32 poison>, i32 %129, i64 2
  %134 = insertelement <4 x i32> %133, i32 %132, i64 3
  %135 = call i32 @llvm.smax.i32(i32 %68, i32 0)
  %136 = and i32 %135, 65535
  %137 = shl i32 %136, 16
  %138 = or i32 %137, 32767
  %139 = lshr i32 %135, 16
  %140 = and i32 %139, 65535
  %141 = or i32 %140, 67108864
  %142 = trunc i64 %127 to i32
  %143 = lshr i64 %127, 32
  %144 = trunc i64 %143 to i32
  %145 = and i32 %144, 65535
  %146 = insertelement <8 x i32> <i32 133169152, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %138, i64 2
  %147 = insertelement <8 x i32> %146, i32 %141, i64 3
  %148 = insertelement <8 x i32> %147, i32 32, i64 4
  %149 = insertelement <8 x i32> %148, i32 %142, i64 5
  %150 = insertelement <8 x i32> %149, i32 %145, i64 6
  %151 = insertelement <8 x i32> %150, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %134, <8 x i32> %151, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %152

152:                                              ; preds = %125, %11
  %153 = icmp eq i32 %44, 1
  br i1 %153, label %154, label %171

154:                                              ; preds = %152
  %155 = icmp eq i64 %79, -2147483648
  %156 = select i1 %155, i64 16384, i64 %79
  %157 = ptrtoint ptr addrspace(1) %88 to i64
  %158 = trunc i64 %157 to i32
  %159 = lshr i64 %157, 32
  %160 = trunc i64 %159 to i32
  %161 = or i32 %160, -2147483648
  %162 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 33280) to i32), i32 poison, i32 poison>, i32 %158, i64 2
  %163 = insertelement <4 x i32> %162, i32 %161, i64 3
  %164 = trunc i64 %156 to i32
  %165 = lshr i64 %156, 32
  %166 = trunc i64 %165 to i32
  %167 = and i32 %166, 65535
  %168 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 1073774591, i32 2, i32 poison, i32 poison, i32 poison>, i32 %164, i64 5
  %169 = insertelement <8 x i32> %168, i32 %167, i64 6
  %170 = insertelement <8 x i32> %169, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %163, <8 x i32> %170, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %171

171:                                              ; preds = %154, %152
  %172 = icmp eq i32 %44, 2
  br i1 %172, label %173, label %200

173:                                              ; preds = %171
  %174 = icmp eq i64 %69, -2147483648
  %175 = select i1 %174, i64 8, i64 %69
  %176 = ptrtoint ptr addrspace(1) %120 to i64
  %177 = trunc i64 %176 to i32
  %178 = lshr i64 %176, 32
  %179 = trunc i64 %178 to i32
  %180 = or i32 %179, -2147483648
  %181 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 66048) to i32), i32 poison, i32 poison>, i32 %177, i64 2
  %182 = insertelement <4 x i32> %181, i32 %180, i64 3
  %183 = call i32 @llvm.smax.i32(i32 %68, i32 0)
  %184 = and i32 %183, 65535
  %185 = shl i32 %184, 16
  %186 = or i32 %185, 32767
  %187 = lshr i32 %183, 16
  %188 = and i32 %187, 65535
  %189 = or i32 %188, 524288
  %190 = trunc i64 %175 to i32
  %191 = lshr i64 %175, 32
  %192 = trunc i64 %191 to i32
  %193 = and i32 %192, 65535
  %194 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %186, i64 2
  %195 = insertelement <8 x i32> %194, i32 %189, i64 3
  %196 = insertelement <8 x i32> %195, i32 32, i64 4
  %197 = insertelement <8 x i32> %196, i32 %190, i64 5
  %198 = insertelement <8 x i32> %197, i32 %193, i64 6
  %199 = insertelement <8 x i32> %198, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %182, <8 x i32> %199, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %200

200:                                              ; preds = %173, %171
  %201 = icmp eq i32 %44, 3
  br i1 %201, label %202, label %219

202:                                              ; preds = %200
  %203 = icmp eq i64 %104, -2147483648
  %204 = select i1 %203, i64 8, i64 %104
  %205 = ptrtoint ptr addrspace(1) %121 to i64
  %206 = trunc i64 %205 to i32
  %207 = lshr i64 %205, 32
  %208 = trunc i64 %207 to i32
  %209 = or i32 %208, -2147483648
  %210 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 66304) to i32), i32 poison, i32 poison>, i32 %206, i64 2
  %211 = insertelement <4 x i32> %210, i32 %209, i64 3
  %212 = trunc i64 %204 to i32
  %213 = lshr i64 %204, 32
  %214 = trunc i64 %213 to i32
  %215 = and i32 %214, 65535
  %216 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 557055, i32 1, i32 poison, i32 poison, i32 poison>, i32 %212, i64 5
  %217 = insertelement <8 x i32> %216, i32 %215, i64 6
  %218 = insertelement <8 x i32> %217, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %211, <8 x i32> %218, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %219

219:                                              ; preds = %202, %200
  br i1 %124, label %220, label %248

220:                                              ; preds = %219
  %221 = icmp eq i64 %21, -2147483648
  %222 = select i1 %221, i64 1024, i64 %21
  %223 = ptrtoint ptr addrspace(1) %87 to i64
  %224 = add i64 %223, 1024
  %225 = trunc i64 %224 to i32
  %226 = lshr i64 %224, 32
  %227 = trunc i64 %226 to i32
  %228 = or i32 %227, -2147483648
  %229 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 66560) to i32), i32 poison, i32 poison>, i32 %225, i64 2
  %230 = insertelement <4 x i32> %229, i32 %228, i64 3
  %231 = call i32 @llvm.smax.i32(i32 %68, i32 0)
  %232 = and i32 %231, 65535
  %233 = shl i32 %232, 16
  %234 = or i32 %233, 32767
  %235 = lshr i32 %231, 16
  %236 = and i32 %235, 65535
  %237 = or i32 %236, 67108864
  %238 = trunc i64 %222 to i32
  %239 = lshr i64 %222, 32
  %240 = trunc i64 %239 to i32
  %241 = and i32 %240, 65535
  %242 = insertelement <8 x i32> <i32 133169152, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %234, i64 2
  %243 = insertelement <8 x i32> %242, i32 %237, i64 3
  %244 = insertelement <8 x i32> %243, i32 32, i64 4
  %245 = insertelement <8 x i32> %244, i32 %238, i64 5
  %246 = insertelement <8 x i32> %245, i32 %241, i64 6
  %247 = insertelement <8 x i32> %246, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %230, <8 x i32> %247, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %248

248:                                              ; preds = %220, %219
  br i1 %153, label %249, label %267

249:                                              ; preds = %248
  %250 = icmp eq i64 %79, -2147483648
  %251 = select i1 %250, i64 16384, i64 %79
  %252 = ptrtoint ptr addrspace(1) %88 to i64
  %253 = add i64 %252, 16384
  %254 = trunc i64 %253 to i32
  %255 = lshr i64 %253, 32
  %256 = trunc i64 %255 to i32
  %257 = or i32 %256, -2147483648
  %258 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 99840) to i32), i32 poison, i32 poison>, i32 %254, i64 2
  %259 = insertelement <4 x i32> %258, i32 %257, i64 3
  %260 = trunc i64 %251 to i32
  %261 = lshr i64 %251, 32
  %262 = trunc i64 %261 to i32
  %263 = and i32 %262, 65535
  %264 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 1073774591, i32 2, i32 poison, i32 poison, i32 poison>, i32 %260, i64 5
  %265 = insertelement <8 x i32> %264, i32 %263, i64 6
  %266 = insertelement <8 x i32> %265, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %259, <8 x i32> %266, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %267

267:                                              ; preds = %249, %248
  br i1 %172, label %268, label %296

268:                                              ; preds = %267
  %269 = icmp eq i64 %69, -2147483648
  %270 = select i1 %269, i64 8, i64 %69
  %271 = ptrtoint ptr addrspace(1) %120 to i64
  %272 = add i64 %271, 8
  %273 = trunc i64 %272 to i32
  %274 = lshr i64 %272, 32
  %275 = trunc i64 %274 to i32
  %276 = or i32 %275, -2147483648
  %277 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 132608) to i32), i32 poison, i32 poison>, i32 %273, i64 2
  %278 = insertelement <4 x i32> %277, i32 %276, i64 3
  %279 = call i32 @llvm.smax.i32(i32 %68, i32 0)
  %280 = and i32 %279, 65535
  %281 = shl i32 %280, 16
  %282 = or i32 %281, 32767
  %283 = lshr i32 %279, 16
  %284 = and i32 %283, 65535
  %285 = or i32 %284, 524288
  %286 = trunc i64 %270 to i32
  %287 = lshr i64 %270, 32
  %288 = trunc i64 %287 to i32
  %289 = and i32 %288, 65535
  %290 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %282, i64 2
  %291 = insertelement <8 x i32> %290, i32 %285, i64 3
  %292 = insertelement <8 x i32> %291, i32 32, i64 4
  %293 = insertelement <8 x i32> %292, i32 %286, i64 5
  %294 = insertelement <8 x i32> %293, i32 %289, i64 6
  %295 = insertelement <8 x i32> %294, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %278, <8 x i32> %295, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %296

296:                                              ; preds = %268, %267
  br i1 %201, label %297, label %315

297:                                              ; preds = %296
  %298 = icmp eq i64 %104, -2147483648
  %299 = select i1 %298, i64 8, i64 %104
  %300 = ptrtoint ptr addrspace(1) %121 to i64
  %301 = add i64 %300, 8
  %302 = trunc i64 %301 to i32
  %303 = lshr i64 %301, 32
  %304 = trunc i64 %303 to i32
  %305 = or i32 %304, -2147483648
  %306 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 132864) to i32), i32 poison, i32 poison>, i32 %302, i64 2
  %307 = insertelement <4 x i32> %306, i32 %305, i64 3
  %308 = trunc i64 %299 to i32
  %309 = lshr i64 %299, 32
  %310 = trunc i64 %309 to i32
  %311 = and i32 %310, 65535
  %312 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 557055, i32 1, i32 poison, i32 poison, i32 poison>, i32 %308, i64 5
  %313 = insertelement <8 x i32> %312, i32 %311, i64 6
  %314 = insertelement <8 x i32> %313, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %307, <8 x i32> %314, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %315

315:                                              ; preds = %297, %296
  br i1 %124, label %316, label %344

316:                                              ; preds = %315
  %317 = icmp eq i64 %21, -2147483648
  %318 = select i1 %317, i64 1024, i64 %21
  %319 = ptrtoint ptr addrspace(1) %87 to i64
  %320 = add i64 %319, 2048
  %321 = trunc i64 %320 to i32
  %322 = lshr i64 %320, 32
  %323 = trunc i64 %322 to i32
  %324 = or i32 %323, -2147483648
  %325 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 133120) to i32), i32 poison, i32 poison>, i32 %321, i64 2
  %326 = insertelement <4 x i32> %325, i32 %324, i64 3
  %327 = call i32 @llvm.smax.i32(i32 %68, i32 0)
  %328 = and i32 %327, 65535
  %329 = shl i32 %328, 16
  %330 = or i32 %329, 32767
  %331 = lshr i32 %327, 16
  %332 = and i32 %331, 65535
  %333 = or i32 %332, 67108864
  %334 = trunc i64 %318 to i32
  %335 = lshr i64 %318, 32
  %336 = trunc i64 %335 to i32
  %337 = and i32 %336, 65535
  %338 = insertelement <8 x i32> <i32 133169152, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %330, i64 2
  %339 = insertelement <8 x i32> %338, i32 %333, i64 3
  %340 = insertelement <8 x i32> %339, i32 32, i64 4
  %341 = insertelement <8 x i32> %340, i32 %334, i64 5
  %342 = insertelement <8 x i32> %341, i32 %337, i64 6
  %343 = insertelement <8 x i32> %342, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %326, <8 x i32> %343, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %344

344:                                              ; preds = %316, %315
  br i1 %153, label %345, label %363

345:                                              ; preds = %344
  %346 = icmp eq i64 %79, -2147483648
  %347 = select i1 %346, i64 16384, i64 %79
  %348 = ptrtoint ptr addrspace(1) %88 to i64
  %349 = add i64 %348, 32768
  %350 = trunc i64 %349 to i32
  %351 = lshr i64 %349, 32
  %352 = trunc i64 %351 to i32
  %353 = or i32 %352, -2147483648
  %354 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 166400) to i32), i32 poison, i32 poison>, i32 %350, i64 2
  %355 = insertelement <4 x i32> %354, i32 %353, i64 3
  %356 = trunc i64 %347 to i32
  %357 = lshr i64 %347, 32
  %358 = trunc i64 %357 to i32
  %359 = and i32 %358, 65535
  %360 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 1073774591, i32 2, i32 poison, i32 poison, i32 poison>, i32 %356, i64 5
  %361 = insertelement <8 x i32> %360, i32 %359, i64 6
  %362 = insertelement <8 x i32> %361, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %355, <8 x i32> %362, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %363

363:                                              ; preds = %345, %344
  br i1 %172, label %364, label %392

364:                                              ; preds = %363
  %365 = icmp eq i64 %69, -2147483648
  %366 = select i1 %365, i64 8, i64 %69
  %367 = ptrtoint ptr addrspace(1) %120 to i64
  %368 = add i64 %367, 16
  %369 = trunc i64 %368 to i32
  %370 = lshr i64 %368, 32
  %371 = trunc i64 %370 to i32
  %372 = or i32 %371, -2147483648
  %373 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 199168) to i32), i32 poison, i32 poison>, i32 %369, i64 2
  %374 = insertelement <4 x i32> %373, i32 %372, i64 3
  %375 = call i32 @llvm.smax.i32(i32 %68, i32 0)
  %376 = and i32 %375, 65535
  %377 = shl i32 %376, 16
  %378 = or i32 %377, 32767
  %379 = lshr i32 %375, 16
  %380 = and i32 %379, 65535
  %381 = or i32 %380, 524288
  %382 = trunc i64 %366 to i32
  %383 = lshr i64 %366, 32
  %384 = trunc i64 %383 to i32
  %385 = and i32 %384, 65535
  %386 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %378, i64 2
  %387 = insertelement <8 x i32> %386, i32 %381, i64 3
  %388 = insertelement <8 x i32> %387, i32 32, i64 4
  %389 = insertelement <8 x i32> %388, i32 %382, i64 5
  %390 = insertelement <8 x i32> %389, i32 %385, i64 6
  %391 = insertelement <8 x i32> %390, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %374, <8 x i32> %391, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %392

392:                                              ; preds = %364, %363
  br i1 %201, label %393, label %411

393:                                              ; preds = %392
  %394 = icmp eq i64 %104, -2147483648
  %395 = select i1 %394, i64 8, i64 %104
  %396 = ptrtoint ptr addrspace(1) %121 to i64
  %397 = add i64 %396, 16
  %398 = trunc i64 %397 to i32
  %399 = lshr i64 %397, 32
  %400 = trunc i64 %399 to i32
  %401 = or i32 %400, -2147483648
  %402 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 199424) to i32), i32 poison, i32 poison>, i32 %398, i64 2
  %403 = insertelement <4 x i32> %402, i32 %401, i64 3
  %404 = trunc i64 %395 to i32
  %405 = lshr i64 %395, 32
  %406 = trunc i64 %405 to i32
  %407 = and i32 %406, 65535
  %408 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 557055, i32 1, i32 poison, i32 poison, i32 poison>, i32 %404, i64 5
  %409 = insertelement <8 x i32> %408, i32 %407, i64 6
  %410 = insertelement <8 x i32> %409, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %403, <8 x i32> %410, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %411

411:                                              ; preds = %393, %392
  %412 = sub i32 %19, 3
  br label %413

413:                                              ; preds = %621, %411
  %414 = phi i32 [ %1056, %621 ], [ 0, %411 ]
  %415 = phi <8 x float> [ %1055, %621 ], [ zeroinitializer, %411 ]
  %416 = icmp slt i32 %414, %412
  br i1 %416, label %417, label %1057

417:                                              ; preds = %413
  %418 = srem i32 %414, 4
  %419 = mul i32 %418, 66560
  %420 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %419
  %421 = ptrtoint ptr addrspace(3) %420 to i32
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 2)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %422 = add i32 %414, 3
  %423 = sdiv i32 %123, 16
  %424 = mul i32 %423, 16
  %425 = icmp ne i32 %123, %424
  %426 = icmp slt i32 %123, 0
  %427 = icmp ne i1 %426, false
  %428 = and i1 %425, %427
  %429 = add i32 %423, -1
  %430 = select i1 %428, i32 %429, i32 %423
  %431 = mul i32 %430, 16384
  %432 = add i32 %431, 33280
  %433 = mul i32 %54, 256
  %434 = add i32 %432, %433
  %435 = mul i32 %46, 16
  %436 = add i32 %434, %435
  %437 = sext i32 %436 to i64
  %438 = add i32 %421, %436
  %439 = inttoptr i32 %438 to ptr addrspace(3)
  %440 = load <4 x i32>, ptr addrspace(3) %439, align 16
  %441 = add i64 %437, 512
  %442 = trunc i64 %441 to i32
  %443 = add i32 %421, %442
  %444 = inttoptr i32 %443 to ptr addrspace(3)
  %445 = load <4 x i32>, ptr addrspace(3) %444, align 16
  %446 = add i64 %437, 1024
  %447 = trunc i64 %446 to i32
  %448 = add i32 %421, %447
  %449 = inttoptr i32 %448 to ptr addrspace(3)
  %450 = load <4 x i32>, ptr addrspace(3) %449, align 16
  %451 = add i64 %437, 1536
  %452 = trunc i64 %451 to i32
  %453 = add i32 %421, %452
  %454 = inttoptr i32 %453 to ptr addrspace(3)
  %455 = load <4 x i32>, ptr addrspace(3) %454, align 16
  %456 = shufflevector <4 x i32> %440, <4 x i32> %445, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %457 = shufflevector <4 x i32> %450, <4 x i32> %455, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %458 = shufflevector <8 x i32> %456, <8 x i32> %457, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %459 = sdiv i32 %123, 128
  %460 = mul i32 %459, 128
  %461 = icmp ne i32 %123, %460
  %462 = icmp slt i32 %123, 0
  %463 = icmp ne i1 %462, false
  %464 = and i1 %461, %463
  %465 = add i32 %459, -1
  %466 = select i1 %464, i32 %465, i32 %459
  %467 = mul i32 %466, 8
  %468 = add i32 %467, 66304
  %469 = add i32 %419, %468
  %470 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %469
  %471 = load i8, ptr addrspace(3) %470, align 1
  %472 = zext i8 %471 to i32
  %473 = call i32 @llvm.amdgcn.perm(i32 %472, i32 %472, i32 0)
  %474 = add i32 %122, %46
  %475 = mul i32 %474, 8
  %476 = add i32 %475, 66048
  %477 = add i32 %419, %476
  %478 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %477
  %479 = load i8, ptr addrspace(3) %478, align 1
  %480 = zext i8 %479 to i32
  %481 = call i32 @llvm.amdgcn.perm(i32 %480, i32 %480, i32 0)
  %482 = mul i32 %474, 1040
  %483 = mul i32 %54, 16
  %484 = add i32 %482, %483
  %485 = sext i32 %484 to i64
  %486 = add i32 %421, %484
  %487 = inttoptr i32 %486 to ptr addrspace(3)
  %488 = load <4 x i32>, ptr addrspace(3) %487, align 16
  %489 = add i64 %485, 32
  %490 = trunc i64 %489 to i32
  %491 = add i32 %421, %490
  %492 = inttoptr i32 %491 to ptr addrspace(3)
  %493 = load <4 x i32>, ptr addrspace(3) %492, align 16
  %494 = add i64 %485, 64
  %495 = trunc i64 %494 to i32
  %496 = add i32 %421, %495
  %497 = inttoptr i32 %496 to ptr addrspace(3)
  %498 = load <4 x i32>, ptr addrspace(3) %497, align 16
  %499 = add i64 %485, 96
  %500 = trunc i64 %499 to i32
  %501 = add i32 %421, %500
  %502 = inttoptr i32 %501 to ptr addrspace(3)
  %503 = load <4 x i32>, ptr addrspace(3) %502, align 16
  %504 = shufflevector <4 x i32> %488, <4 x i32> %493, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %505 = shufflevector <4 x i32> %498, <4 x i32> %503, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %506 = shufflevector <8 x i32> %504, <8 x i32> %505, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %507 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %458, i32 0, <16 x i32> %506, i16 0, <8 x float> %415, i32 0, i32 0, i32 %473, i32 0, i32 0, i32 %481, i1 false, i1 false)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %508 = srem i32 %422, 4
  %509 = mul i32 %508, 66560
  %510 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %509
  %511 = sext i32 %422 to i64
  %512 = mul i64 %511, 1024
  br i1 %124, label %513, label %543

513:                                              ; preds = %417
  %514 = icmp eq i64 %21, -2147483648
  %515 = select i1 %514, i64 1024, i64 %21
  %516 = ptrtoint ptr addrspace(1) %87 to i64
  %517 = add i64 %516, %512
  %518 = ptrtoint ptr addrspace(3) %510 to i32
  %519 = trunc i64 %517 to i32
  %520 = lshr i64 %517, 32
  %521 = trunc i64 %520 to i32
  %522 = or i32 %521, -2147483648
  %523 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %518, i64 1
  %524 = insertelement <4 x i32> %523, i32 %519, i64 2
  %525 = insertelement <4 x i32> %524, i32 %522, i64 3
  %526 = call i32 @llvm.smax.i32(i32 %68, i32 0)
  %527 = and i32 %526, 65535
  %528 = shl i32 %527, 16
  %529 = or i32 %528, 32767
  %530 = lshr i32 %526, 16
  %531 = and i32 %530, 65535
  %532 = or i32 %531, 67108864
  %533 = trunc i64 %515 to i32
  %534 = lshr i64 %515, 32
  %535 = trunc i64 %534 to i32
  %536 = and i32 %535, 65535
  %537 = insertelement <8 x i32> <i32 133169152, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %529, i64 2
  %538 = insertelement <8 x i32> %537, i32 %532, i64 3
  %539 = insertelement <8 x i32> %538, i32 32, i64 4
  %540 = insertelement <8 x i32> %539, i32 %533, i64 5
  %541 = insertelement <8 x i32> %540, i32 %536, i64 6
  %542 = insertelement <8 x i32> %541, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %525, <8 x i32> %542, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %543

543:                                              ; preds = %513, %417
  %544 = getelementptr i8, ptr addrspace(3) %510, i32 33280
  %545 = mul i64 %511, 16384
  br i1 %153, label %546, label %566

546:                                              ; preds = %543
  %547 = icmp eq i64 %79, -2147483648
  %548 = select i1 %547, i64 16384, i64 %79
  %549 = ptrtoint ptr addrspace(1) %88 to i64
  %550 = add i64 %549, %545
  %551 = ptrtoint ptr addrspace(3) %544 to i32
  %552 = trunc i64 %550 to i32
  %553 = lshr i64 %550, 32
  %554 = trunc i64 %553 to i32
  %555 = or i32 %554, -2147483648
  %556 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %551, i64 1
  %557 = insertelement <4 x i32> %556, i32 %552, i64 2
  %558 = insertelement <4 x i32> %557, i32 %555, i64 3
  %559 = trunc i64 %548 to i32
  %560 = lshr i64 %548, 32
  %561 = trunc i64 %560 to i32
  %562 = and i32 %561, 65535
  %563 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 1073774591, i32 2, i32 poison, i32 poison, i32 poison>, i32 %559, i64 5
  %564 = insertelement <8 x i32> %563, i32 %562, i64 6
  %565 = insertelement <8 x i32> %564, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %558, <8 x i32> %565, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %566

566:                                              ; preds = %546, %543
  %567 = getelementptr i8, ptr addrspace(3) %510, i32 66048
  %568 = mul i64 %511, 8
  br i1 %172, label %569, label %599

569:                                              ; preds = %566
  %570 = icmp eq i64 %69, -2147483648
  %571 = select i1 %570, i64 8, i64 %69
  %572 = ptrtoint ptr addrspace(1) %120 to i64
  %573 = add i64 %572, %568
  %574 = ptrtoint ptr addrspace(3) %567 to i32
  %575 = trunc i64 %573 to i32
  %576 = lshr i64 %573, 32
  %577 = trunc i64 %576 to i32
  %578 = or i32 %577, -2147483648
  %579 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %574, i64 1
  %580 = insertelement <4 x i32> %579, i32 %575, i64 2
  %581 = insertelement <4 x i32> %580, i32 %578, i64 3
  %582 = call i32 @llvm.smax.i32(i32 %68, i32 0)
  %583 = and i32 %582, 65535
  %584 = shl i32 %583, 16
  %585 = or i32 %584, 32767
  %586 = lshr i32 %582, 16
  %587 = and i32 %586, 65535
  %588 = or i32 %587, 524288
  %589 = trunc i64 %571 to i32
  %590 = lshr i64 %571, 32
  %591 = trunc i64 %590 to i32
  %592 = and i32 %591, 65535
  %593 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %585, i64 2
  %594 = insertelement <8 x i32> %593, i32 %588, i64 3
  %595 = insertelement <8 x i32> %594, i32 32, i64 4
  %596 = insertelement <8 x i32> %595, i32 %589, i64 5
  %597 = insertelement <8 x i32> %596, i32 %592, i64 6
  %598 = insertelement <8 x i32> %597, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %581, <8 x i32> %598, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %599

599:                                              ; preds = %569, %566
  %600 = getelementptr i8, ptr addrspace(3) %510, i32 66304
  br i1 %201, label %601, label %621

601:                                              ; preds = %599
  %602 = icmp eq i64 %104, -2147483648
  %603 = select i1 %602, i64 8, i64 %104
  %604 = ptrtoint ptr addrspace(1) %121 to i64
  %605 = add i64 %604, %568
  %606 = ptrtoint ptr addrspace(3) %600 to i32
  %607 = trunc i64 %605 to i32
  %608 = lshr i64 %605, 32
  %609 = trunc i64 %608 to i32
  %610 = or i32 %609, -2147483648
  %611 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %606, i64 1
  %612 = insertelement <4 x i32> %611, i32 %607, i64 2
  %613 = insertelement <4 x i32> %612, i32 %610, i64 3
  %614 = trunc i64 %603 to i32
  %615 = lshr i64 %603, 32
  %616 = trunc i64 %615 to i32
  %617 = and i32 %616, 65535
  %618 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 557055, i32 1, i32 poison, i32 poison, i32 poison>, i32 %614, i64 5
  %619 = insertelement <8 x i32> %618, i32 %617, i64 6
  %620 = insertelement <8 x i32> %619, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %613, <8 x i32> %620, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %621

621:                                              ; preds = %601, %599
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %622 = add i32 %431, 35328
  %623 = add i32 %622, %433
  %624 = add i32 %623, %435
  %625 = sext i32 %624 to i64
  %626 = add i32 %421, %624
  %627 = inttoptr i32 %626 to ptr addrspace(3)
  %628 = load <4 x i32>, ptr addrspace(3) %627, align 16
  %629 = add i64 %625, 512
  %630 = trunc i64 %629 to i32
  %631 = add i32 %421, %630
  %632 = inttoptr i32 %631 to ptr addrspace(3)
  %633 = load <4 x i32>, ptr addrspace(3) %632, align 16
  %634 = add i64 %625, 1024
  %635 = trunc i64 %634 to i32
  %636 = add i32 %421, %635
  %637 = inttoptr i32 %636 to ptr addrspace(3)
  %638 = load <4 x i32>, ptr addrspace(3) %637, align 16
  %639 = add i64 %625, 1536
  %640 = trunc i64 %639 to i32
  %641 = add i32 %421, %640
  %642 = inttoptr i32 %641 to ptr addrspace(3)
  %643 = load <4 x i32>, ptr addrspace(3) %642, align 16
  %644 = shufflevector <4 x i32> %628, <4 x i32> %633, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %645 = shufflevector <4 x i32> %638, <4 x i32> %643, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %646 = shufflevector <8 x i32> %644, <8 x i32> %645, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %647 = add i32 %467, 66305
  %648 = add i32 %419, %647
  %649 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %648
  %650 = load i8, ptr addrspace(3) %649, align 1
  %651 = zext i8 %650 to i32
  %652 = call i32 @llvm.amdgcn.perm(i32 %651, i32 %651, i32 0)
  %653 = add i32 %475, 66049
  %654 = add i32 %419, %653
  %655 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %654
  %656 = load i8, ptr addrspace(3) %655, align 1
  %657 = zext i8 %656 to i32
  %658 = call i32 @llvm.amdgcn.perm(i32 %657, i32 %657, i32 0)
  %659 = add i32 %482, 128
  %660 = add i32 %659, %483
  %661 = sext i32 %660 to i64
  %662 = add i32 %421, %660
  %663 = inttoptr i32 %662 to ptr addrspace(3)
  %664 = load <4 x i32>, ptr addrspace(3) %663, align 16
  %665 = add i64 %661, 32
  %666 = trunc i64 %665 to i32
  %667 = add i32 %421, %666
  %668 = inttoptr i32 %667 to ptr addrspace(3)
  %669 = load <4 x i32>, ptr addrspace(3) %668, align 16
  %670 = add i64 %661, 64
  %671 = trunc i64 %670 to i32
  %672 = add i32 %421, %671
  %673 = inttoptr i32 %672 to ptr addrspace(3)
  %674 = load <4 x i32>, ptr addrspace(3) %673, align 16
  %675 = add i64 %661, 96
  %676 = trunc i64 %675 to i32
  %677 = add i32 %421, %676
  %678 = inttoptr i32 %677 to ptr addrspace(3)
  %679 = load <4 x i32>, ptr addrspace(3) %678, align 16
  %680 = shufflevector <4 x i32> %664, <4 x i32> %669, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %681 = shufflevector <4 x i32> %674, <4 x i32> %679, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %682 = shufflevector <8 x i32> %680, <8 x i32> %681, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %683 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %646, i32 0, <16 x i32> %682, i16 0, <8 x float> %507, i32 0, i32 0, i32 %652, i32 0, i32 0, i32 %658, i1 false, i1 false)
  %684 = add i32 %431, 37376
  %685 = add i32 %684, %433
  %686 = add i32 %685, %435
  %687 = sext i32 %686 to i64
  %688 = add i32 %421, %686
  %689 = inttoptr i32 %688 to ptr addrspace(3)
  %690 = load <4 x i32>, ptr addrspace(3) %689, align 16
  %691 = add i64 %687, 512
  %692 = trunc i64 %691 to i32
  %693 = add i32 %421, %692
  %694 = inttoptr i32 %693 to ptr addrspace(3)
  %695 = load <4 x i32>, ptr addrspace(3) %694, align 16
  %696 = add i64 %687, 1024
  %697 = trunc i64 %696 to i32
  %698 = add i32 %421, %697
  %699 = inttoptr i32 %698 to ptr addrspace(3)
  %700 = load <4 x i32>, ptr addrspace(3) %699, align 16
  %701 = add i64 %687, 1536
  %702 = trunc i64 %701 to i32
  %703 = add i32 %421, %702
  %704 = inttoptr i32 %703 to ptr addrspace(3)
  %705 = load <4 x i32>, ptr addrspace(3) %704, align 16
  %706 = shufflevector <4 x i32> %690, <4 x i32> %695, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %707 = shufflevector <4 x i32> %700, <4 x i32> %705, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %708 = shufflevector <8 x i32> %706, <8 x i32> %707, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %709 = add i32 %467, 66306
  %710 = add i32 %419, %709
  %711 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %710
  %712 = load i8, ptr addrspace(3) %711, align 1
  %713 = zext i8 %712 to i32
  %714 = call i32 @llvm.amdgcn.perm(i32 %713, i32 %713, i32 0)
  %715 = add i32 %475, 66050
  %716 = add i32 %419, %715
  %717 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %716
  %718 = load i8, ptr addrspace(3) %717, align 1
  %719 = zext i8 %718 to i32
  %720 = call i32 @llvm.amdgcn.perm(i32 %719, i32 %719, i32 0)
  %721 = add i32 %482, 256
  %722 = add i32 %721, %483
  %723 = sext i32 %722 to i64
  %724 = add i32 %421, %722
  %725 = inttoptr i32 %724 to ptr addrspace(3)
  %726 = load <4 x i32>, ptr addrspace(3) %725, align 16
  %727 = add i64 %723, 32
  %728 = trunc i64 %727 to i32
  %729 = add i32 %421, %728
  %730 = inttoptr i32 %729 to ptr addrspace(3)
  %731 = load <4 x i32>, ptr addrspace(3) %730, align 16
  %732 = add i64 %723, 64
  %733 = trunc i64 %732 to i32
  %734 = add i32 %421, %733
  %735 = inttoptr i32 %734 to ptr addrspace(3)
  %736 = load <4 x i32>, ptr addrspace(3) %735, align 16
  %737 = add i64 %723, 96
  %738 = trunc i64 %737 to i32
  %739 = add i32 %421, %738
  %740 = inttoptr i32 %739 to ptr addrspace(3)
  %741 = load <4 x i32>, ptr addrspace(3) %740, align 16
  %742 = shufflevector <4 x i32> %726, <4 x i32> %731, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %743 = shufflevector <4 x i32> %736, <4 x i32> %741, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %744 = shufflevector <8 x i32> %742, <8 x i32> %743, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %745 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %708, i32 0, <16 x i32> %744, i16 0, <8 x float> %683, i32 0, i32 0, i32 %714, i32 0, i32 0, i32 %720, i1 false, i1 false)
  %746 = add i32 %431, 39424
  %747 = add i32 %746, %433
  %748 = add i32 %747, %435
  %749 = sext i32 %748 to i64
  %750 = add i32 %421, %748
  %751 = inttoptr i32 %750 to ptr addrspace(3)
  %752 = load <4 x i32>, ptr addrspace(3) %751, align 16
  %753 = add i64 %749, 512
  %754 = trunc i64 %753 to i32
  %755 = add i32 %421, %754
  %756 = inttoptr i32 %755 to ptr addrspace(3)
  %757 = load <4 x i32>, ptr addrspace(3) %756, align 16
  %758 = add i64 %749, 1024
  %759 = trunc i64 %758 to i32
  %760 = add i32 %421, %759
  %761 = inttoptr i32 %760 to ptr addrspace(3)
  %762 = load <4 x i32>, ptr addrspace(3) %761, align 16
  %763 = add i64 %749, 1536
  %764 = trunc i64 %763 to i32
  %765 = add i32 %421, %764
  %766 = inttoptr i32 %765 to ptr addrspace(3)
  %767 = load <4 x i32>, ptr addrspace(3) %766, align 16
  %768 = shufflevector <4 x i32> %752, <4 x i32> %757, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %769 = shufflevector <4 x i32> %762, <4 x i32> %767, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %770 = shufflevector <8 x i32> %768, <8 x i32> %769, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %771 = add i32 %467, 66307
  %772 = add i32 %419, %771
  %773 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %772
  %774 = load i8, ptr addrspace(3) %773, align 1
  %775 = zext i8 %774 to i32
  %776 = call i32 @llvm.amdgcn.perm(i32 %775, i32 %775, i32 0)
  %777 = add i32 %475, 66051
  %778 = add i32 %419, %777
  %779 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %778
  %780 = load i8, ptr addrspace(3) %779, align 1
  %781 = zext i8 %780 to i32
  %782 = call i32 @llvm.amdgcn.perm(i32 %781, i32 %781, i32 0)
  %783 = add i32 %482, 384
  %784 = add i32 %783, %483
  %785 = sext i32 %784 to i64
  %786 = add i32 %421, %784
  %787 = inttoptr i32 %786 to ptr addrspace(3)
  %788 = load <4 x i32>, ptr addrspace(3) %787, align 16
  %789 = add i64 %785, 32
  %790 = trunc i64 %789 to i32
  %791 = add i32 %421, %790
  %792 = inttoptr i32 %791 to ptr addrspace(3)
  %793 = load <4 x i32>, ptr addrspace(3) %792, align 16
  %794 = add i64 %785, 64
  %795 = trunc i64 %794 to i32
  %796 = add i32 %421, %795
  %797 = inttoptr i32 %796 to ptr addrspace(3)
  %798 = load <4 x i32>, ptr addrspace(3) %797, align 16
  %799 = add i64 %785, 96
  %800 = trunc i64 %799 to i32
  %801 = add i32 %421, %800
  %802 = inttoptr i32 %801 to ptr addrspace(3)
  %803 = load <4 x i32>, ptr addrspace(3) %802, align 16
  %804 = shufflevector <4 x i32> %788, <4 x i32> %793, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %805 = shufflevector <4 x i32> %798, <4 x i32> %803, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %806 = shufflevector <8 x i32> %804, <8 x i32> %805, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %807 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %770, i32 0, <16 x i32> %806, i16 0, <8 x float> %745, i32 0, i32 0, i32 %776, i32 0, i32 0, i32 %782, i1 false, i1 false)
  %808 = add i32 %431, 41472
  %809 = add i32 %808, %433
  %810 = add i32 %809, %435
  %811 = sext i32 %810 to i64
  %812 = add i32 %421, %810
  %813 = inttoptr i32 %812 to ptr addrspace(3)
  %814 = load <4 x i32>, ptr addrspace(3) %813, align 16
  %815 = add i64 %811, 512
  %816 = trunc i64 %815 to i32
  %817 = add i32 %421, %816
  %818 = inttoptr i32 %817 to ptr addrspace(3)
  %819 = load <4 x i32>, ptr addrspace(3) %818, align 16
  %820 = add i64 %811, 1024
  %821 = trunc i64 %820 to i32
  %822 = add i32 %421, %821
  %823 = inttoptr i32 %822 to ptr addrspace(3)
  %824 = load <4 x i32>, ptr addrspace(3) %823, align 16
  %825 = add i64 %811, 1536
  %826 = trunc i64 %825 to i32
  %827 = add i32 %421, %826
  %828 = inttoptr i32 %827 to ptr addrspace(3)
  %829 = load <4 x i32>, ptr addrspace(3) %828, align 16
  %830 = shufflevector <4 x i32> %814, <4 x i32> %819, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %831 = shufflevector <4 x i32> %824, <4 x i32> %829, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %832 = shufflevector <8 x i32> %830, <8 x i32> %831, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %833 = add i32 %467, 66308
  %834 = add i32 %419, %833
  %835 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %834
  %836 = load i8, ptr addrspace(3) %835, align 1
  %837 = zext i8 %836 to i32
  %838 = call i32 @llvm.amdgcn.perm(i32 %837, i32 %837, i32 0)
  %839 = add i32 %475, 66052
  %840 = add i32 %419, %839
  %841 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %840
  %842 = load i8, ptr addrspace(3) %841, align 1
  %843 = zext i8 %842 to i32
  %844 = call i32 @llvm.amdgcn.perm(i32 %843, i32 %843, i32 0)
  %845 = add i32 %482, 512
  %846 = add i32 %845, %483
  %847 = sext i32 %846 to i64
  %848 = add i32 %421, %846
  %849 = inttoptr i32 %848 to ptr addrspace(3)
  %850 = load <4 x i32>, ptr addrspace(3) %849, align 16
  %851 = add i64 %847, 32
  %852 = trunc i64 %851 to i32
  %853 = add i32 %421, %852
  %854 = inttoptr i32 %853 to ptr addrspace(3)
  %855 = load <4 x i32>, ptr addrspace(3) %854, align 16
  %856 = add i64 %847, 64
  %857 = trunc i64 %856 to i32
  %858 = add i32 %421, %857
  %859 = inttoptr i32 %858 to ptr addrspace(3)
  %860 = load <4 x i32>, ptr addrspace(3) %859, align 16
  %861 = add i64 %847, 96
  %862 = trunc i64 %861 to i32
  %863 = add i32 %421, %862
  %864 = inttoptr i32 %863 to ptr addrspace(3)
  %865 = load <4 x i32>, ptr addrspace(3) %864, align 16
  %866 = shufflevector <4 x i32> %850, <4 x i32> %855, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %867 = shufflevector <4 x i32> %860, <4 x i32> %865, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %868 = shufflevector <8 x i32> %866, <8 x i32> %867, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %869 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %832, i32 0, <16 x i32> %868, i16 0, <8 x float> %807, i32 0, i32 0, i32 %838, i32 0, i32 0, i32 %844, i1 false, i1 false)
  %870 = add i32 %431, 43520
  %871 = add i32 %870, %433
  %872 = add i32 %871, %435
  %873 = sext i32 %872 to i64
  %874 = add i32 %421, %872
  %875 = inttoptr i32 %874 to ptr addrspace(3)
  %876 = load <4 x i32>, ptr addrspace(3) %875, align 16
  %877 = add i64 %873, 512
  %878 = trunc i64 %877 to i32
  %879 = add i32 %421, %878
  %880 = inttoptr i32 %879 to ptr addrspace(3)
  %881 = load <4 x i32>, ptr addrspace(3) %880, align 16
  %882 = add i64 %873, 1024
  %883 = trunc i64 %882 to i32
  %884 = add i32 %421, %883
  %885 = inttoptr i32 %884 to ptr addrspace(3)
  %886 = load <4 x i32>, ptr addrspace(3) %885, align 16
  %887 = add i64 %873, 1536
  %888 = trunc i64 %887 to i32
  %889 = add i32 %421, %888
  %890 = inttoptr i32 %889 to ptr addrspace(3)
  %891 = load <4 x i32>, ptr addrspace(3) %890, align 16
  %892 = shufflevector <4 x i32> %876, <4 x i32> %881, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %893 = shufflevector <4 x i32> %886, <4 x i32> %891, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %894 = shufflevector <8 x i32> %892, <8 x i32> %893, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %895 = add i32 %467, 66309
  %896 = add i32 %419, %895
  %897 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %896
  %898 = load i8, ptr addrspace(3) %897, align 1
  %899 = zext i8 %898 to i32
  %900 = call i32 @llvm.amdgcn.perm(i32 %899, i32 %899, i32 0)
  %901 = add i32 %475, 66053
  %902 = add i32 %419, %901
  %903 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %902
  %904 = load i8, ptr addrspace(3) %903, align 1
  %905 = zext i8 %904 to i32
  %906 = call i32 @llvm.amdgcn.perm(i32 %905, i32 %905, i32 0)
  %907 = add i32 %482, 640
  %908 = add i32 %907, %483
  %909 = sext i32 %908 to i64
  %910 = add i32 %421, %908
  %911 = inttoptr i32 %910 to ptr addrspace(3)
  %912 = load <4 x i32>, ptr addrspace(3) %911, align 16
  %913 = add i64 %909, 32
  %914 = trunc i64 %913 to i32
  %915 = add i32 %421, %914
  %916 = inttoptr i32 %915 to ptr addrspace(3)
  %917 = load <4 x i32>, ptr addrspace(3) %916, align 16
  %918 = add i64 %909, 64
  %919 = trunc i64 %918 to i32
  %920 = add i32 %421, %919
  %921 = inttoptr i32 %920 to ptr addrspace(3)
  %922 = load <4 x i32>, ptr addrspace(3) %921, align 16
  %923 = add i64 %909, 96
  %924 = trunc i64 %923 to i32
  %925 = add i32 %421, %924
  %926 = inttoptr i32 %925 to ptr addrspace(3)
  %927 = load <4 x i32>, ptr addrspace(3) %926, align 16
  %928 = shufflevector <4 x i32> %912, <4 x i32> %917, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %929 = shufflevector <4 x i32> %922, <4 x i32> %927, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %930 = shufflevector <8 x i32> %928, <8 x i32> %929, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %931 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %894, i32 0, <16 x i32> %930, i16 0, <8 x float> %869, i32 0, i32 0, i32 %900, i32 0, i32 0, i32 %906, i1 false, i1 false)
  %932 = add i32 %431, 45568
  %933 = add i32 %932, %433
  %934 = add i32 %933, %435
  %935 = sext i32 %934 to i64
  %936 = add i32 %421, %934
  %937 = inttoptr i32 %936 to ptr addrspace(3)
  %938 = load <4 x i32>, ptr addrspace(3) %937, align 16
  %939 = add i64 %935, 512
  %940 = trunc i64 %939 to i32
  %941 = add i32 %421, %940
  %942 = inttoptr i32 %941 to ptr addrspace(3)
  %943 = load <4 x i32>, ptr addrspace(3) %942, align 16
  %944 = add i64 %935, 1024
  %945 = trunc i64 %944 to i32
  %946 = add i32 %421, %945
  %947 = inttoptr i32 %946 to ptr addrspace(3)
  %948 = load <4 x i32>, ptr addrspace(3) %947, align 16
  %949 = add i64 %935, 1536
  %950 = trunc i64 %949 to i32
  %951 = add i32 %421, %950
  %952 = inttoptr i32 %951 to ptr addrspace(3)
  %953 = load <4 x i32>, ptr addrspace(3) %952, align 16
  %954 = shufflevector <4 x i32> %938, <4 x i32> %943, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %955 = shufflevector <4 x i32> %948, <4 x i32> %953, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %956 = shufflevector <8 x i32> %954, <8 x i32> %955, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %957 = add i32 %467, 66310
  %958 = add i32 %419, %957
  %959 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %958
  %960 = load i8, ptr addrspace(3) %959, align 1
  %961 = zext i8 %960 to i32
  %962 = call i32 @llvm.amdgcn.perm(i32 %961, i32 %961, i32 0)
  %963 = add i32 %475, 66054
  %964 = add i32 %419, %963
  %965 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %964
  %966 = load i8, ptr addrspace(3) %965, align 1
  %967 = zext i8 %966 to i32
  %968 = call i32 @llvm.amdgcn.perm(i32 %967, i32 %967, i32 0)
  %969 = add i32 %482, 768
  %970 = add i32 %969, %483
  %971 = sext i32 %970 to i64
  %972 = add i32 %421, %970
  %973 = inttoptr i32 %972 to ptr addrspace(3)
  %974 = load <4 x i32>, ptr addrspace(3) %973, align 16
  %975 = add i64 %971, 32
  %976 = trunc i64 %975 to i32
  %977 = add i32 %421, %976
  %978 = inttoptr i32 %977 to ptr addrspace(3)
  %979 = load <4 x i32>, ptr addrspace(3) %978, align 16
  %980 = add i64 %971, 64
  %981 = trunc i64 %980 to i32
  %982 = add i32 %421, %981
  %983 = inttoptr i32 %982 to ptr addrspace(3)
  %984 = load <4 x i32>, ptr addrspace(3) %983, align 16
  %985 = add i64 %971, 96
  %986 = trunc i64 %985 to i32
  %987 = add i32 %421, %986
  %988 = inttoptr i32 %987 to ptr addrspace(3)
  %989 = load <4 x i32>, ptr addrspace(3) %988, align 16
  %990 = shufflevector <4 x i32> %974, <4 x i32> %979, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %991 = shufflevector <4 x i32> %984, <4 x i32> %989, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %992 = shufflevector <8 x i32> %990, <8 x i32> %991, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %993 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %956, i32 0, <16 x i32> %992, i16 0, <8 x float> %931, i32 0, i32 0, i32 %962, i32 0, i32 0, i32 %968, i1 false, i1 false)
  %994 = add i32 %431, 47616
  %995 = add i32 %994, %433
  %996 = add i32 %995, %435
  %997 = sext i32 %996 to i64
  %998 = add i32 %421, %996
  %999 = inttoptr i32 %998 to ptr addrspace(3)
  %1000 = load <4 x i32>, ptr addrspace(3) %999, align 16
  %1001 = add i64 %997, 512
  %1002 = trunc i64 %1001 to i32
  %1003 = add i32 %421, %1002
  %1004 = inttoptr i32 %1003 to ptr addrspace(3)
  %1005 = load <4 x i32>, ptr addrspace(3) %1004, align 16
  %1006 = add i64 %997, 1024
  %1007 = trunc i64 %1006 to i32
  %1008 = add i32 %421, %1007
  %1009 = inttoptr i32 %1008 to ptr addrspace(3)
  %1010 = load <4 x i32>, ptr addrspace(3) %1009, align 16
  %1011 = add i64 %997, 1536
  %1012 = trunc i64 %1011 to i32
  %1013 = add i32 %421, %1012
  %1014 = inttoptr i32 %1013 to ptr addrspace(3)
  %1015 = load <4 x i32>, ptr addrspace(3) %1014, align 16
  %1016 = shufflevector <4 x i32> %1000, <4 x i32> %1005, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1017 = shufflevector <4 x i32> %1010, <4 x i32> %1015, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1018 = shufflevector <8 x i32> %1016, <8 x i32> %1017, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1019 = add i32 %467, 66311
  %1020 = add i32 %419, %1019
  %1021 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1020
  %1022 = load i8, ptr addrspace(3) %1021, align 1
  %1023 = zext i8 %1022 to i32
  %1024 = call i32 @llvm.amdgcn.perm(i32 %1023, i32 %1023, i32 0)
  %1025 = add i32 %475, 66055
  %1026 = add i32 %419, %1025
  %1027 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1026
  %1028 = load i8, ptr addrspace(3) %1027, align 1
  %1029 = zext i8 %1028 to i32
  %1030 = call i32 @llvm.amdgcn.perm(i32 %1029, i32 %1029, i32 0)
  %1031 = add i32 %482, 896
  %1032 = add i32 %1031, %483
  %1033 = sext i32 %1032 to i64
  %1034 = add i32 %421, %1032
  %1035 = inttoptr i32 %1034 to ptr addrspace(3)
  %1036 = load <4 x i32>, ptr addrspace(3) %1035, align 16
  %1037 = add i64 %1033, 32
  %1038 = trunc i64 %1037 to i32
  %1039 = add i32 %421, %1038
  %1040 = inttoptr i32 %1039 to ptr addrspace(3)
  %1041 = load <4 x i32>, ptr addrspace(3) %1040, align 16
  %1042 = add i64 %1033, 64
  %1043 = trunc i64 %1042 to i32
  %1044 = add i32 %421, %1043
  %1045 = inttoptr i32 %1044 to ptr addrspace(3)
  %1046 = load <4 x i32>, ptr addrspace(3) %1045, align 16
  %1047 = add i64 %1033, 96
  %1048 = trunc i64 %1047 to i32
  %1049 = add i32 %421, %1048
  %1050 = inttoptr i32 %1049 to ptr addrspace(3)
  %1051 = load <4 x i32>, ptr addrspace(3) %1050, align 16
  %1052 = shufflevector <4 x i32> %1036, <4 x i32> %1041, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1053 = shufflevector <4 x i32> %1046, <4 x i32> %1051, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1054 = shufflevector <8 x i32> %1052, <8 x i32> %1053, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1055 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1018, i32 0, <16 x i32> %1054, i16 0, <8 x float> %993, i32 0, i32 0, i32 %1024, i32 0, i32 0, i32 %1030, i1 false, i1 false)
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
  %1056 = add i32 %414, 1
  br label %413

1057:                                             ; preds = %413
  %1058 = srem i32 %412, 4
  %1059 = mul i32 %1058, 66560
  %1060 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1059
  %1061 = ptrtoint ptr addrspace(3) %1060 to i32
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 2)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %1062 = sdiv i32 %123, 16
  %1063 = mul i32 %1062, 16
  %1064 = icmp ne i32 %123, %1063
  %1065 = icmp slt i32 %123, 0
  %1066 = icmp ne i1 %1065, false
  %1067 = and i1 %1064, %1066
  %1068 = add i32 %1062, -1
  %1069 = select i1 %1067, i32 %1068, i32 %1062
  %1070 = mul i32 %1069, 16384
  %1071 = add i32 %1070, 33280
  %1072 = mul i32 %54, 256
  %1073 = add i32 %1071, %1072
  %1074 = mul i32 %46, 16
  %1075 = add i32 %1073, %1074
  %1076 = sext i32 %1075 to i64
  %1077 = add i32 %1061, %1075
  %1078 = inttoptr i32 %1077 to ptr addrspace(3)
  %1079 = load <4 x i32>, ptr addrspace(3) %1078, align 16
  %1080 = add i64 %1076, 512
  %1081 = trunc i64 %1080 to i32
  %1082 = add i32 %1061, %1081
  %1083 = inttoptr i32 %1082 to ptr addrspace(3)
  %1084 = load <4 x i32>, ptr addrspace(3) %1083, align 16
  %1085 = add i64 %1076, 1024
  %1086 = trunc i64 %1085 to i32
  %1087 = add i32 %1061, %1086
  %1088 = inttoptr i32 %1087 to ptr addrspace(3)
  %1089 = load <4 x i32>, ptr addrspace(3) %1088, align 16
  %1090 = add i64 %1076, 1536
  %1091 = trunc i64 %1090 to i32
  %1092 = add i32 %1061, %1091
  %1093 = inttoptr i32 %1092 to ptr addrspace(3)
  %1094 = load <4 x i32>, ptr addrspace(3) %1093, align 16
  %1095 = shufflevector <4 x i32> %1079, <4 x i32> %1084, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1096 = shufflevector <4 x i32> %1089, <4 x i32> %1094, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1097 = shufflevector <8 x i32> %1095, <8 x i32> %1096, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1098 = sdiv i32 %123, 128
  %1099 = mul i32 %1098, 128
  %1100 = icmp ne i32 %123, %1099
  %1101 = icmp slt i32 %123, 0
  %1102 = icmp ne i1 %1101, false
  %1103 = and i1 %1100, %1102
  %1104 = add i32 %1098, -1
  %1105 = select i1 %1103, i32 %1104, i32 %1098
  %1106 = mul i32 %1105, 8
  %1107 = add i32 %1106, 66304
  %1108 = add i32 %1059, %1107
  %1109 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1108
  %1110 = load i8, ptr addrspace(3) %1109, align 1
  %1111 = zext i8 %1110 to i32
  %1112 = call i32 @llvm.amdgcn.perm(i32 %1111, i32 %1111, i32 0)
  %1113 = add i32 %122, %46
  %1114 = mul i32 %1113, 8
  %1115 = add i32 %1114, 66048
  %1116 = add i32 %1059, %1115
  %1117 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1116
  %1118 = load i8, ptr addrspace(3) %1117, align 1
  %1119 = zext i8 %1118 to i32
  %1120 = call i32 @llvm.amdgcn.perm(i32 %1119, i32 %1119, i32 0)
  %1121 = mul i32 %1113, 1040
  %1122 = mul i32 %54, 16
  %1123 = add i32 %1121, %1122
  %1124 = sext i32 %1123 to i64
  %1125 = add i32 %1061, %1123
  %1126 = inttoptr i32 %1125 to ptr addrspace(3)
  %1127 = load <4 x i32>, ptr addrspace(3) %1126, align 16
  %1128 = add i64 %1124, 32
  %1129 = trunc i64 %1128 to i32
  %1130 = add i32 %1061, %1129
  %1131 = inttoptr i32 %1130 to ptr addrspace(3)
  %1132 = load <4 x i32>, ptr addrspace(3) %1131, align 16
  %1133 = add i64 %1124, 64
  %1134 = trunc i64 %1133 to i32
  %1135 = add i32 %1061, %1134
  %1136 = inttoptr i32 %1135 to ptr addrspace(3)
  %1137 = load <4 x i32>, ptr addrspace(3) %1136, align 16
  %1138 = add i64 %1124, 96
  %1139 = trunc i64 %1138 to i32
  %1140 = add i32 %1061, %1139
  %1141 = inttoptr i32 %1140 to ptr addrspace(3)
  %1142 = load <4 x i32>, ptr addrspace(3) %1141, align 16
  %1143 = shufflevector <4 x i32> %1127, <4 x i32> %1132, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1144 = shufflevector <4 x i32> %1137, <4 x i32> %1142, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1145 = shufflevector <8 x i32> %1143, <8 x i32> %1144, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1146 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1097, i32 0, <16 x i32> %1145, i16 0, <8 x float> %415, i32 0, i32 0, i32 %1112, i32 0, i32 0, i32 %1120, i1 false, i1 false)
  %1147 = add i32 %1070, 35328
  %1148 = add i32 %1147, %1072
  %1149 = add i32 %1148, %1074
  %1150 = sext i32 %1149 to i64
  %1151 = add i32 %1061, %1149
  %1152 = inttoptr i32 %1151 to ptr addrspace(3)
  %1153 = load <4 x i32>, ptr addrspace(3) %1152, align 16
  %1154 = add i64 %1150, 512
  %1155 = trunc i64 %1154 to i32
  %1156 = add i32 %1061, %1155
  %1157 = inttoptr i32 %1156 to ptr addrspace(3)
  %1158 = load <4 x i32>, ptr addrspace(3) %1157, align 16
  %1159 = add i64 %1150, 1024
  %1160 = trunc i64 %1159 to i32
  %1161 = add i32 %1061, %1160
  %1162 = inttoptr i32 %1161 to ptr addrspace(3)
  %1163 = load <4 x i32>, ptr addrspace(3) %1162, align 16
  %1164 = add i64 %1150, 1536
  %1165 = trunc i64 %1164 to i32
  %1166 = add i32 %1061, %1165
  %1167 = inttoptr i32 %1166 to ptr addrspace(3)
  %1168 = load <4 x i32>, ptr addrspace(3) %1167, align 16
  %1169 = shufflevector <4 x i32> %1153, <4 x i32> %1158, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1170 = shufflevector <4 x i32> %1163, <4 x i32> %1168, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1171 = shufflevector <8 x i32> %1169, <8 x i32> %1170, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1172 = add i32 %1106, 66305
  %1173 = add i32 %1059, %1172
  %1174 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1173
  %1175 = load i8, ptr addrspace(3) %1174, align 1
  %1176 = zext i8 %1175 to i32
  %1177 = call i32 @llvm.amdgcn.perm(i32 %1176, i32 %1176, i32 0)
  %1178 = add i32 %1114, 66049
  %1179 = add i32 %1059, %1178
  %1180 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1179
  %1181 = load i8, ptr addrspace(3) %1180, align 1
  %1182 = zext i8 %1181 to i32
  %1183 = call i32 @llvm.amdgcn.perm(i32 %1182, i32 %1182, i32 0)
  %1184 = add i32 %1121, 128
  %1185 = add i32 %1184, %1122
  %1186 = sext i32 %1185 to i64
  %1187 = add i32 %1061, %1185
  %1188 = inttoptr i32 %1187 to ptr addrspace(3)
  %1189 = load <4 x i32>, ptr addrspace(3) %1188, align 16
  %1190 = add i64 %1186, 32
  %1191 = trunc i64 %1190 to i32
  %1192 = add i32 %1061, %1191
  %1193 = inttoptr i32 %1192 to ptr addrspace(3)
  %1194 = load <4 x i32>, ptr addrspace(3) %1193, align 16
  %1195 = add i64 %1186, 64
  %1196 = trunc i64 %1195 to i32
  %1197 = add i32 %1061, %1196
  %1198 = inttoptr i32 %1197 to ptr addrspace(3)
  %1199 = load <4 x i32>, ptr addrspace(3) %1198, align 16
  %1200 = add i64 %1186, 96
  %1201 = trunc i64 %1200 to i32
  %1202 = add i32 %1061, %1201
  %1203 = inttoptr i32 %1202 to ptr addrspace(3)
  %1204 = load <4 x i32>, ptr addrspace(3) %1203, align 16
  %1205 = shufflevector <4 x i32> %1189, <4 x i32> %1194, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1206 = shufflevector <4 x i32> %1199, <4 x i32> %1204, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1207 = shufflevector <8 x i32> %1205, <8 x i32> %1206, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1208 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1171, i32 0, <16 x i32> %1207, i16 0, <8 x float> %1146, i32 0, i32 0, i32 %1177, i32 0, i32 0, i32 %1183, i1 false, i1 false)
  %1209 = add i32 %1070, 37376
  %1210 = add i32 %1209, %1072
  %1211 = add i32 %1210, %1074
  %1212 = sext i32 %1211 to i64
  %1213 = add i32 %1061, %1211
  %1214 = inttoptr i32 %1213 to ptr addrspace(3)
  %1215 = load <4 x i32>, ptr addrspace(3) %1214, align 16
  %1216 = add i64 %1212, 512
  %1217 = trunc i64 %1216 to i32
  %1218 = add i32 %1061, %1217
  %1219 = inttoptr i32 %1218 to ptr addrspace(3)
  %1220 = load <4 x i32>, ptr addrspace(3) %1219, align 16
  %1221 = add i64 %1212, 1024
  %1222 = trunc i64 %1221 to i32
  %1223 = add i32 %1061, %1222
  %1224 = inttoptr i32 %1223 to ptr addrspace(3)
  %1225 = load <4 x i32>, ptr addrspace(3) %1224, align 16
  %1226 = add i64 %1212, 1536
  %1227 = trunc i64 %1226 to i32
  %1228 = add i32 %1061, %1227
  %1229 = inttoptr i32 %1228 to ptr addrspace(3)
  %1230 = load <4 x i32>, ptr addrspace(3) %1229, align 16
  %1231 = shufflevector <4 x i32> %1215, <4 x i32> %1220, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1232 = shufflevector <4 x i32> %1225, <4 x i32> %1230, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1233 = shufflevector <8 x i32> %1231, <8 x i32> %1232, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1234 = add i32 %1106, 66306
  %1235 = add i32 %1059, %1234
  %1236 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1235
  %1237 = load i8, ptr addrspace(3) %1236, align 1
  %1238 = zext i8 %1237 to i32
  %1239 = call i32 @llvm.amdgcn.perm(i32 %1238, i32 %1238, i32 0)
  %1240 = add i32 %1114, 66050
  %1241 = add i32 %1059, %1240
  %1242 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1241
  %1243 = load i8, ptr addrspace(3) %1242, align 1
  %1244 = zext i8 %1243 to i32
  %1245 = call i32 @llvm.amdgcn.perm(i32 %1244, i32 %1244, i32 0)
  %1246 = add i32 %1121, 256
  %1247 = add i32 %1246, %1122
  %1248 = sext i32 %1247 to i64
  %1249 = add i32 %1061, %1247
  %1250 = inttoptr i32 %1249 to ptr addrspace(3)
  %1251 = load <4 x i32>, ptr addrspace(3) %1250, align 16
  %1252 = add i64 %1248, 32
  %1253 = trunc i64 %1252 to i32
  %1254 = add i32 %1061, %1253
  %1255 = inttoptr i32 %1254 to ptr addrspace(3)
  %1256 = load <4 x i32>, ptr addrspace(3) %1255, align 16
  %1257 = add i64 %1248, 64
  %1258 = trunc i64 %1257 to i32
  %1259 = add i32 %1061, %1258
  %1260 = inttoptr i32 %1259 to ptr addrspace(3)
  %1261 = load <4 x i32>, ptr addrspace(3) %1260, align 16
  %1262 = add i64 %1248, 96
  %1263 = trunc i64 %1262 to i32
  %1264 = add i32 %1061, %1263
  %1265 = inttoptr i32 %1264 to ptr addrspace(3)
  %1266 = load <4 x i32>, ptr addrspace(3) %1265, align 16
  %1267 = shufflevector <4 x i32> %1251, <4 x i32> %1256, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1268 = shufflevector <4 x i32> %1261, <4 x i32> %1266, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1269 = shufflevector <8 x i32> %1267, <8 x i32> %1268, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1270 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1233, i32 0, <16 x i32> %1269, i16 0, <8 x float> %1208, i32 0, i32 0, i32 %1239, i32 0, i32 0, i32 %1245, i1 false, i1 false)
  %1271 = add i32 %1070, 39424
  %1272 = add i32 %1271, %1072
  %1273 = add i32 %1272, %1074
  %1274 = sext i32 %1273 to i64
  %1275 = add i32 %1061, %1273
  %1276 = inttoptr i32 %1275 to ptr addrspace(3)
  %1277 = load <4 x i32>, ptr addrspace(3) %1276, align 16
  %1278 = add i64 %1274, 512
  %1279 = trunc i64 %1278 to i32
  %1280 = add i32 %1061, %1279
  %1281 = inttoptr i32 %1280 to ptr addrspace(3)
  %1282 = load <4 x i32>, ptr addrspace(3) %1281, align 16
  %1283 = add i64 %1274, 1024
  %1284 = trunc i64 %1283 to i32
  %1285 = add i32 %1061, %1284
  %1286 = inttoptr i32 %1285 to ptr addrspace(3)
  %1287 = load <4 x i32>, ptr addrspace(3) %1286, align 16
  %1288 = add i64 %1274, 1536
  %1289 = trunc i64 %1288 to i32
  %1290 = add i32 %1061, %1289
  %1291 = inttoptr i32 %1290 to ptr addrspace(3)
  %1292 = load <4 x i32>, ptr addrspace(3) %1291, align 16
  %1293 = shufflevector <4 x i32> %1277, <4 x i32> %1282, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1294 = shufflevector <4 x i32> %1287, <4 x i32> %1292, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1295 = shufflevector <8 x i32> %1293, <8 x i32> %1294, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1296 = add i32 %1106, 66307
  %1297 = add i32 %1059, %1296
  %1298 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1297
  %1299 = load i8, ptr addrspace(3) %1298, align 1
  %1300 = zext i8 %1299 to i32
  %1301 = call i32 @llvm.amdgcn.perm(i32 %1300, i32 %1300, i32 0)
  %1302 = add i32 %1114, 66051
  %1303 = add i32 %1059, %1302
  %1304 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1303
  %1305 = load i8, ptr addrspace(3) %1304, align 1
  %1306 = zext i8 %1305 to i32
  %1307 = call i32 @llvm.amdgcn.perm(i32 %1306, i32 %1306, i32 0)
  %1308 = add i32 %1121, 384
  %1309 = add i32 %1308, %1122
  %1310 = sext i32 %1309 to i64
  %1311 = add i32 %1061, %1309
  %1312 = inttoptr i32 %1311 to ptr addrspace(3)
  %1313 = load <4 x i32>, ptr addrspace(3) %1312, align 16
  %1314 = add i64 %1310, 32
  %1315 = trunc i64 %1314 to i32
  %1316 = add i32 %1061, %1315
  %1317 = inttoptr i32 %1316 to ptr addrspace(3)
  %1318 = load <4 x i32>, ptr addrspace(3) %1317, align 16
  %1319 = add i64 %1310, 64
  %1320 = trunc i64 %1319 to i32
  %1321 = add i32 %1061, %1320
  %1322 = inttoptr i32 %1321 to ptr addrspace(3)
  %1323 = load <4 x i32>, ptr addrspace(3) %1322, align 16
  %1324 = add i64 %1310, 96
  %1325 = trunc i64 %1324 to i32
  %1326 = add i32 %1061, %1325
  %1327 = inttoptr i32 %1326 to ptr addrspace(3)
  %1328 = load <4 x i32>, ptr addrspace(3) %1327, align 16
  %1329 = shufflevector <4 x i32> %1313, <4 x i32> %1318, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1330 = shufflevector <4 x i32> %1323, <4 x i32> %1328, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1331 = shufflevector <8 x i32> %1329, <8 x i32> %1330, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1332 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1295, i32 0, <16 x i32> %1331, i16 0, <8 x float> %1270, i32 0, i32 0, i32 %1301, i32 0, i32 0, i32 %1307, i1 false, i1 false)
  %1333 = add i32 %1070, 41472
  %1334 = add i32 %1333, %1072
  %1335 = add i32 %1334, %1074
  %1336 = sext i32 %1335 to i64
  %1337 = add i32 %1061, %1335
  %1338 = inttoptr i32 %1337 to ptr addrspace(3)
  %1339 = load <4 x i32>, ptr addrspace(3) %1338, align 16
  %1340 = add i64 %1336, 512
  %1341 = trunc i64 %1340 to i32
  %1342 = add i32 %1061, %1341
  %1343 = inttoptr i32 %1342 to ptr addrspace(3)
  %1344 = load <4 x i32>, ptr addrspace(3) %1343, align 16
  %1345 = add i64 %1336, 1024
  %1346 = trunc i64 %1345 to i32
  %1347 = add i32 %1061, %1346
  %1348 = inttoptr i32 %1347 to ptr addrspace(3)
  %1349 = load <4 x i32>, ptr addrspace(3) %1348, align 16
  %1350 = add i64 %1336, 1536
  %1351 = trunc i64 %1350 to i32
  %1352 = add i32 %1061, %1351
  %1353 = inttoptr i32 %1352 to ptr addrspace(3)
  %1354 = load <4 x i32>, ptr addrspace(3) %1353, align 16
  %1355 = shufflevector <4 x i32> %1339, <4 x i32> %1344, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1356 = shufflevector <4 x i32> %1349, <4 x i32> %1354, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1357 = shufflevector <8 x i32> %1355, <8 x i32> %1356, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1358 = add i32 %1106, 66308
  %1359 = add i32 %1059, %1358
  %1360 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1359
  %1361 = load i8, ptr addrspace(3) %1360, align 1
  %1362 = zext i8 %1361 to i32
  %1363 = call i32 @llvm.amdgcn.perm(i32 %1362, i32 %1362, i32 0)
  %1364 = add i32 %1114, 66052
  %1365 = add i32 %1059, %1364
  %1366 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1365
  %1367 = load i8, ptr addrspace(3) %1366, align 1
  %1368 = zext i8 %1367 to i32
  %1369 = call i32 @llvm.amdgcn.perm(i32 %1368, i32 %1368, i32 0)
  %1370 = add i32 %1121, 512
  %1371 = add i32 %1370, %1122
  %1372 = sext i32 %1371 to i64
  %1373 = add i32 %1061, %1371
  %1374 = inttoptr i32 %1373 to ptr addrspace(3)
  %1375 = load <4 x i32>, ptr addrspace(3) %1374, align 16
  %1376 = add i64 %1372, 32
  %1377 = trunc i64 %1376 to i32
  %1378 = add i32 %1061, %1377
  %1379 = inttoptr i32 %1378 to ptr addrspace(3)
  %1380 = load <4 x i32>, ptr addrspace(3) %1379, align 16
  %1381 = add i64 %1372, 64
  %1382 = trunc i64 %1381 to i32
  %1383 = add i32 %1061, %1382
  %1384 = inttoptr i32 %1383 to ptr addrspace(3)
  %1385 = load <4 x i32>, ptr addrspace(3) %1384, align 16
  %1386 = add i64 %1372, 96
  %1387 = trunc i64 %1386 to i32
  %1388 = add i32 %1061, %1387
  %1389 = inttoptr i32 %1388 to ptr addrspace(3)
  %1390 = load <4 x i32>, ptr addrspace(3) %1389, align 16
  %1391 = shufflevector <4 x i32> %1375, <4 x i32> %1380, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1392 = shufflevector <4 x i32> %1385, <4 x i32> %1390, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1393 = shufflevector <8 x i32> %1391, <8 x i32> %1392, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1394 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1357, i32 0, <16 x i32> %1393, i16 0, <8 x float> %1332, i32 0, i32 0, i32 %1363, i32 0, i32 0, i32 %1369, i1 false, i1 false)
  %1395 = add i32 %1070, 43520
  %1396 = add i32 %1395, %1072
  %1397 = add i32 %1396, %1074
  %1398 = sext i32 %1397 to i64
  %1399 = add i32 %1061, %1397
  %1400 = inttoptr i32 %1399 to ptr addrspace(3)
  %1401 = load <4 x i32>, ptr addrspace(3) %1400, align 16
  %1402 = add i64 %1398, 512
  %1403 = trunc i64 %1402 to i32
  %1404 = add i32 %1061, %1403
  %1405 = inttoptr i32 %1404 to ptr addrspace(3)
  %1406 = load <4 x i32>, ptr addrspace(3) %1405, align 16
  %1407 = add i64 %1398, 1024
  %1408 = trunc i64 %1407 to i32
  %1409 = add i32 %1061, %1408
  %1410 = inttoptr i32 %1409 to ptr addrspace(3)
  %1411 = load <4 x i32>, ptr addrspace(3) %1410, align 16
  %1412 = add i64 %1398, 1536
  %1413 = trunc i64 %1412 to i32
  %1414 = add i32 %1061, %1413
  %1415 = inttoptr i32 %1414 to ptr addrspace(3)
  %1416 = load <4 x i32>, ptr addrspace(3) %1415, align 16
  %1417 = shufflevector <4 x i32> %1401, <4 x i32> %1406, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1418 = shufflevector <4 x i32> %1411, <4 x i32> %1416, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1419 = shufflevector <8 x i32> %1417, <8 x i32> %1418, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1420 = add i32 %1106, 66309
  %1421 = add i32 %1059, %1420
  %1422 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1421
  %1423 = load i8, ptr addrspace(3) %1422, align 1
  %1424 = zext i8 %1423 to i32
  %1425 = call i32 @llvm.amdgcn.perm(i32 %1424, i32 %1424, i32 0)
  %1426 = add i32 %1114, 66053
  %1427 = add i32 %1059, %1426
  %1428 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1427
  %1429 = load i8, ptr addrspace(3) %1428, align 1
  %1430 = zext i8 %1429 to i32
  %1431 = call i32 @llvm.amdgcn.perm(i32 %1430, i32 %1430, i32 0)
  %1432 = add i32 %1121, 640
  %1433 = add i32 %1432, %1122
  %1434 = sext i32 %1433 to i64
  %1435 = add i32 %1061, %1433
  %1436 = inttoptr i32 %1435 to ptr addrspace(3)
  %1437 = load <4 x i32>, ptr addrspace(3) %1436, align 16
  %1438 = add i64 %1434, 32
  %1439 = trunc i64 %1438 to i32
  %1440 = add i32 %1061, %1439
  %1441 = inttoptr i32 %1440 to ptr addrspace(3)
  %1442 = load <4 x i32>, ptr addrspace(3) %1441, align 16
  %1443 = add i64 %1434, 64
  %1444 = trunc i64 %1443 to i32
  %1445 = add i32 %1061, %1444
  %1446 = inttoptr i32 %1445 to ptr addrspace(3)
  %1447 = load <4 x i32>, ptr addrspace(3) %1446, align 16
  %1448 = add i64 %1434, 96
  %1449 = trunc i64 %1448 to i32
  %1450 = add i32 %1061, %1449
  %1451 = inttoptr i32 %1450 to ptr addrspace(3)
  %1452 = load <4 x i32>, ptr addrspace(3) %1451, align 16
  %1453 = shufflevector <4 x i32> %1437, <4 x i32> %1442, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1454 = shufflevector <4 x i32> %1447, <4 x i32> %1452, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1455 = shufflevector <8 x i32> %1453, <8 x i32> %1454, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1456 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1419, i32 0, <16 x i32> %1455, i16 0, <8 x float> %1394, i32 0, i32 0, i32 %1425, i32 0, i32 0, i32 %1431, i1 false, i1 false)
  %1457 = add i32 %1070, 45568
  %1458 = add i32 %1457, %1072
  %1459 = add i32 %1458, %1074
  %1460 = sext i32 %1459 to i64
  %1461 = add i32 %1061, %1459
  %1462 = inttoptr i32 %1461 to ptr addrspace(3)
  %1463 = load <4 x i32>, ptr addrspace(3) %1462, align 16
  %1464 = add i64 %1460, 512
  %1465 = trunc i64 %1464 to i32
  %1466 = add i32 %1061, %1465
  %1467 = inttoptr i32 %1466 to ptr addrspace(3)
  %1468 = load <4 x i32>, ptr addrspace(3) %1467, align 16
  %1469 = add i64 %1460, 1024
  %1470 = trunc i64 %1469 to i32
  %1471 = add i32 %1061, %1470
  %1472 = inttoptr i32 %1471 to ptr addrspace(3)
  %1473 = load <4 x i32>, ptr addrspace(3) %1472, align 16
  %1474 = add i64 %1460, 1536
  %1475 = trunc i64 %1474 to i32
  %1476 = add i32 %1061, %1475
  %1477 = inttoptr i32 %1476 to ptr addrspace(3)
  %1478 = load <4 x i32>, ptr addrspace(3) %1477, align 16
  %1479 = shufflevector <4 x i32> %1463, <4 x i32> %1468, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1480 = shufflevector <4 x i32> %1473, <4 x i32> %1478, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1481 = shufflevector <8 x i32> %1479, <8 x i32> %1480, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1482 = add i32 %1106, 66310
  %1483 = add i32 %1059, %1482
  %1484 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1483
  %1485 = load i8, ptr addrspace(3) %1484, align 1
  %1486 = zext i8 %1485 to i32
  %1487 = call i32 @llvm.amdgcn.perm(i32 %1486, i32 %1486, i32 0)
  %1488 = add i32 %1114, 66054
  %1489 = add i32 %1059, %1488
  %1490 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1489
  %1491 = load i8, ptr addrspace(3) %1490, align 1
  %1492 = zext i8 %1491 to i32
  %1493 = call i32 @llvm.amdgcn.perm(i32 %1492, i32 %1492, i32 0)
  %1494 = add i32 %1121, 768
  %1495 = add i32 %1494, %1122
  %1496 = sext i32 %1495 to i64
  %1497 = add i32 %1061, %1495
  %1498 = inttoptr i32 %1497 to ptr addrspace(3)
  %1499 = load <4 x i32>, ptr addrspace(3) %1498, align 16
  %1500 = add i64 %1496, 32
  %1501 = trunc i64 %1500 to i32
  %1502 = add i32 %1061, %1501
  %1503 = inttoptr i32 %1502 to ptr addrspace(3)
  %1504 = load <4 x i32>, ptr addrspace(3) %1503, align 16
  %1505 = add i64 %1496, 64
  %1506 = trunc i64 %1505 to i32
  %1507 = add i32 %1061, %1506
  %1508 = inttoptr i32 %1507 to ptr addrspace(3)
  %1509 = load <4 x i32>, ptr addrspace(3) %1508, align 16
  %1510 = add i64 %1496, 96
  %1511 = trunc i64 %1510 to i32
  %1512 = add i32 %1061, %1511
  %1513 = inttoptr i32 %1512 to ptr addrspace(3)
  %1514 = load <4 x i32>, ptr addrspace(3) %1513, align 16
  %1515 = shufflevector <4 x i32> %1499, <4 x i32> %1504, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1516 = shufflevector <4 x i32> %1509, <4 x i32> %1514, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1517 = shufflevector <8 x i32> %1515, <8 x i32> %1516, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1518 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1481, i32 0, <16 x i32> %1517, i16 0, <8 x float> %1456, i32 0, i32 0, i32 %1487, i32 0, i32 0, i32 %1493, i1 false, i1 false)
  %1519 = add i32 %1070, 47616
  %1520 = add i32 %1519, %1072
  %1521 = add i32 %1520, %1074
  %1522 = sext i32 %1521 to i64
  %1523 = add i32 %1061, %1521
  %1524 = inttoptr i32 %1523 to ptr addrspace(3)
  %1525 = load <4 x i32>, ptr addrspace(3) %1524, align 16
  %1526 = add i64 %1522, 512
  %1527 = trunc i64 %1526 to i32
  %1528 = add i32 %1061, %1527
  %1529 = inttoptr i32 %1528 to ptr addrspace(3)
  %1530 = load <4 x i32>, ptr addrspace(3) %1529, align 16
  %1531 = add i64 %1522, 1024
  %1532 = trunc i64 %1531 to i32
  %1533 = add i32 %1061, %1532
  %1534 = inttoptr i32 %1533 to ptr addrspace(3)
  %1535 = load <4 x i32>, ptr addrspace(3) %1534, align 16
  %1536 = add i64 %1522, 1536
  %1537 = trunc i64 %1536 to i32
  %1538 = add i32 %1061, %1537
  %1539 = inttoptr i32 %1538 to ptr addrspace(3)
  %1540 = load <4 x i32>, ptr addrspace(3) %1539, align 16
  %1541 = shufflevector <4 x i32> %1525, <4 x i32> %1530, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1542 = shufflevector <4 x i32> %1535, <4 x i32> %1540, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1543 = shufflevector <8 x i32> %1541, <8 x i32> %1542, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1544 = add i32 %1106, 66311
  %1545 = add i32 %1059, %1544
  %1546 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1545
  %1547 = load i8, ptr addrspace(3) %1546, align 1
  %1548 = zext i8 %1547 to i32
  %1549 = call i32 @llvm.amdgcn.perm(i32 %1548, i32 %1548, i32 0)
  %1550 = add i32 %1114, 66055
  %1551 = add i32 %1059, %1550
  %1552 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1551
  %1553 = load i8, ptr addrspace(3) %1552, align 1
  %1554 = zext i8 %1553 to i32
  %1555 = call i32 @llvm.amdgcn.perm(i32 %1554, i32 %1554, i32 0)
  %1556 = add i32 %1121, 896
  %1557 = add i32 %1556, %1122
  %1558 = sext i32 %1557 to i64
  %1559 = add i32 %1061, %1557
  %1560 = inttoptr i32 %1559 to ptr addrspace(3)
  %1561 = load <4 x i32>, ptr addrspace(3) %1560, align 16
  %1562 = add i64 %1558, 32
  %1563 = trunc i64 %1562 to i32
  %1564 = add i32 %1061, %1563
  %1565 = inttoptr i32 %1564 to ptr addrspace(3)
  %1566 = load <4 x i32>, ptr addrspace(3) %1565, align 16
  %1567 = add i64 %1558, 64
  %1568 = trunc i64 %1567 to i32
  %1569 = add i32 %1061, %1568
  %1570 = inttoptr i32 %1569 to ptr addrspace(3)
  %1571 = load <4 x i32>, ptr addrspace(3) %1570, align 16
  %1572 = add i64 %1558, 96
  %1573 = trunc i64 %1572 to i32
  %1574 = add i32 %1061, %1573
  %1575 = inttoptr i32 %1574 to ptr addrspace(3)
  %1576 = load <4 x i32>, ptr addrspace(3) %1575, align 16
  %1577 = shufflevector <4 x i32> %1561, <4 x i32> %1566, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1578 = shufflevector <4 x i32> %1571, <4 x i32> %1576, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1579 = shufflevector <8 x i32> %1577, <8 x i32> %1578, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1580 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1543, i32 0, <16 x i32> %1579, i16 0, <8 x float> %1518, i32 0, i32 0, i32 %1549, i32 0, i32 0, i32 %1555, i1 false, i1 false)
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
  %1581 = add i32 %19, -2
  %1582 = srem i32 %1581, 4
  %1583 = mul i32 %1582, 66560
  %1584 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1583
  %1585 = ptrtoint ptr addrspace(3) %1584 to i32
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 1)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %1586 = add i32 %1585, %1075
  %1587 = inttoptr i32 %1586 to ptr addrspace(3)
  %1588 = load <4 x i32>, ptr addrspace(3) %1587, align 16
  %1589 = add i32 %1585, %1081
  %1590 = inttoptr i32 %1589 to ptr addrspace(3)
  %1591 = load <4 x i32>, ptr addrspace(3) %1590, align 16
  %1592 = add i32 %1585, %1086
  %1593 = inttoptr i32 %1592 to ptr addrspace(3)
  %1594 = load <4 x i32>, ptr addrspace(3) %1593, align 16
  %1595 = add i32 %1585, %1091
  %1596 = inttoptr i32 %1595 to ptr addrspace(3)
  %1597 = load <4 x i32>, ptr addrspace(3) %1596, align 16
  %1598 = shufflevector <4 x i32> %1588, <4 x i32> %1591, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1599 = shufflevector <4 x i32> %1594, <4 x i32> %1597, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1600 = shufflevector <8 x i32> %1598, <8 x i32> %1599, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1601 = add i32 %1583, %1107
  %1602 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1601
  %1603 = load i8, ptr addrspace(3) %1602, align 1
  %1604 = zext i8 %1603 to i32
  %1605 = call i32 @llvm.amdgcn.perm(i32 %1604, i32 %1604, i32 0)
  %1606 = add i32 %1583, %1115
  %1607 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1606
  %1608 = load i8, ptr addrspace(3) %1607, align 1
  %1609 = zext i8 %1608 to i32
  %1610 = call i32 @llvm.amdgcn.perm(i32 %1609, i32 %1609, i32 0)
  %1611 = add i32 %1585, %1123
  %1612 = inttoptr i32 %1611 to ptr addrspace(3)
  %1613 = load <4 x i32>, ptr addrspace(3) %1612, align 16
  %1614 = add i32 %1585, %1129
  %1615 = inttoptr i32 %1614 to ptr addrspace(3)
  %1616 = load <4 x i32>, ptr addrspace(3) %1615, align 16
  %1617 = add i32 %1585, %1134
  %1618 = inttoptr i32 %1617 to ptr addrspace(3)
  %1619 = load <4 x i32>, ptr addrspace(3) %1618, align 16
  %1620 = add i32 %1585, %1139
  %1621 = inttoptr i32 %1620 to ptr addrspace(3)
  %1622 = load <4 x i32>, ptr addrspace(3) %1621, align 16
  %1623 = shufflevector <4 x i32> %1613, <4 x i32> %1616, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1624 = shufflevector <4 x i32> %1619, <4 x i32> %1622, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1625 = shufflevector <8 x i32> %1623, <8 x i32> %1624, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1626 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1600, i32 0, <16 x i32> %1625, i16 0, <8 x float> %1580, i32 0, i32 0, i32 %1605, i32 0, i32 0, i32 %1610, i1 false, i1 false)
  %1627 = add i32 %1585, %1149
  %1628 = inttoptr i32 %1627 to ptr addrspace(3)
  %1629 = load <4 x i32>, ptr addrspace(3) %1628, align 16
  %1630 = add i32 %1585, %1155
  %1631 = inttoptr i32 %1630 to ptr addrspace(3)
  %1632 = load <4 x i32>, ptr addrspace(3) %1631, align 16
  %1633 = add i32 %1585, %1160
  %1634 = inttoptr i32 %1633 to ptr addrspace(3)
  %1635 = load <4 x i32>, ptr addrspace(3) %1634, align 16
  %1636 = add i32 %1585, %1165
  %1637 = inttoptr i32 %1636 to ptr addrspace(3)
  %1638 = load <4 x i32>, ptr addrspace(3) %1637, align 16
  %1639 = shufflevector <4 x i32> %1629, <4 x i32> %1632, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1640 = shufflevector <4 x i32> %1635, <4 x i32> %1638, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1641 = shufflevector <8 x i32> %1639, <8 x i32> %1640, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1642 = add i32 %1583, %1172
  %1643 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1642
  %1644 = load i8, ptr addrspace(3) %1643, align 1
  %1645 = zext i8 %1644 to i32
  %1646 = call i32 @llvm.amdgcn.perm(i32 %1645, i32 %1645, i32 0)
  %1647 = add i32 %1583, %1178
  %1648 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1647
  %1649 = load i8, ptr addrspace(3) %1648, align 1
  %1650 = zext i8 %1649 to i32
  %1651 = call i32 @llvm.amdgcn.perm(i32 %1650, i32 %1650, i32 0)
  %1652 = add i32 %1585, %1185
  %1653 = inttoptr i32 %1652 to ptr addrspace(3)
  %1654 = load <4 x i32>, ptr addrspace(3) %1653, align 16
  %1655 = add i32 %1585, %1191
  %1656 = inttoptr i32 %1655 to ptr addrspace(3)
  %1657 = load <4 x i32>, ptr addrspace(3) %1656, align 16
  %1658 = add i32 %1585, %1196
  %1659 = inttoptr i32 %1658 to ptr addrspace(3)
  %1660 = load <4 x i32>, ptr addrspace(3) %1659, align 16
  %1661 = add i32 %1585, %1201
  %1662 = inttoptr i32 %1661 to ptr addrspace(3)
  %1663 = load <4 x i32>, ptr addrspace(3) %1662, align 16
  %1664 = shufflevector <4 x i32> %1654, <4 x i32> %1657, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1665 = shufflevector <4 x i32> %1660, <4 x i32> %1663, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1666 = shufflevector <8 x i32> %1664, <8 x i32> %1665, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1667 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1641, i32 0, <16 x i32> %1666, i16 0, <8 x float> %1626, i32 0, i32 0, i32 %1646, i32 0, i32 0, i32 %1651, i1 false, i1 false)
  %1668 = add i32 %1585, %1211
  %1669 = inttoptr i32 %1668 to ptr addrspace(3)
  %1670 = load <4 x i32>, ptr addrspace(3) %1669, align 16
  %1671 = add i32 %1585, %1217
  %1672 = inttoptr i32 %1671 to ptr addrspace(3)
  %1673 = load <4 x i32>, ptr addrspace(3) %1672, align 16
  %1674 = add i32 %1585, %1222
  %1675 = inttoptr i32 %1674 to ptr addrspace(3)
  %1676 = load <4 x i32>, ptr addrspace(3) %1675, align 16
  %1677 = add i32 %1585, %1227
  %1678 = inttoptr i32 %1677 to ptr addrspace(3)
  %1679 = load <4 x i32>, ptr addrspace(3) %1678, align 16
  %1680 = shufflevector <4 x i32> %1670, <4 x i32> %1673, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1681 = shufflevector <4 x i32> %1676, <4 x i32> %1679, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1682 = shufflevector <8 x i32> %1680, <8 x i32> %1681, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1683 = add i32 %1583, %1234
  %1684 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1683
  %1685 = load i8, ptr addrspace(3) %1684, align 1
  %1686 = zext i8 %1685 to i32
  %1687 = call i32 @llvm.amdgcn.perm(i32 %1686, i32 %1686, i32 0)
  %1688 = add i32 %1583, %1240
  %1689 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1688
  %1690 = load i8, ptr addrspace(3) %1689, align 1
  %1691 = zext i8 %1690 to i32
  %1692 = call i32 @llvm.amdgcn.perm(i32 %1691, i32 %1691, i32 0)
  %1693 = add i32 %1585, %1247
  %1694 = inttoptr i32 %1693 to ptr addrspace(3)
  %1695 = load <4 x i32>, ptr addrspace(3) %1694, align 16
  %1696 = add i32 %1585, %1253
  %1697 = inttoptr i32 %1696 to ptr addrspace(3)
  %1698 = load <4 x i32>, ptr addrspace(3) %1697, align 16
  %1699 = add i32 %1585, %1258
  %1700 = inttoptr i32 %1699 to ptr addrspace(3)
  %1701 = load <4 x i32>, ptr addrspace(3) %1700, align 16
  %1702 = add i32 %1585, %1263
  %1703 = inttoptr i32 %1702 to ptr addrspace(3)
  %1704 = load <4 x i32>, ptr addrspace(3) %1703, align 16
  %1705 = shufflevector <4 x i32> %1695, <4 x i32> %1698, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1706 = shufflevector <4 x i32> %1701, <4 x i32> %1704, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1707 = shufflevector <8 x i32> %1705, <8 x i32> %1706, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1708 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1682, i32 0, <16 x i32> %1707, i16 0, <8 x float> %1667, i32 0, i32 0, i32 %1687, i32 0, i32 0, i32 %1692, i1 false, i1 false)
  %1709 = add i32 %1585, %1273
  %1710 = inttoptr i32 %1709 to ptr addrspace(3)
  %1711 = load <4 x i32>, ptr addrspace(3) %1710, align 16
  %1712 = add i32 %1585, %1279
  %1713 = inttoptr i32 %1712 to ptr addrspace(3)
  %1714 = load <4 x i32>, ptr addrspace(3) %1713, align 16
  %1715 = add i32 %1585, %1284
  %1716 = inttoptr i32 %1715 to ptr addrspace(3)
  %1717 = load <4 x i32>, ptr addrspace(3) %1716, align 16
  %1718 = add i32 %1585, %1289
  %1719 = inttoptr i32 %1718 to ptr addrspace(3)
  %1720 = load <4 x i32>, ptr addrspace(3) %1719, align 16
  %1721 = shufflevector <4 x i32> %1711, <4 x i32> %1714, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1722 = shufflevector <4 x i32> %1717, <4 x i32> %1720, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1723 = shufflevector <8 x i32> %1721, <8 x i32> %1722, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1724 = add i32 %1583, %1296
  %1725 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1724
  %1726 = load i8, ptr addrspace(3) %1725, align 1
  %1727 = zext i8 %1726 to i32
  %1728 = call i32 @llvm.amdgcn.perm(i32 %1727, i32 %1727, i32 0)
  %1729 = add i32 %1583, %1302
  %1730 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1729
  %1731 = load i8, ptr addrspace(3) %1730, align 1
  %1732 = zext i8 %1731 to i32
  %1733 = call i32 @llvm.amdgcn.perm(i32 %1732, i32 %1732, i32 0)
  %1734 = add i32 %1585, %1309
  %1735 = inttoptr i32 %1734 to ptr addrspace(3)
  %1736 = load <4 x i32>, ptr addrspace(3) %1735, align 16
  %1737 = add i32 %1585, %1315
  %1738 = inttoptr i32 %1737 to ptr addrspace(3)
  %1739 = load <4 x i32>, ptr addrspace(3) %1738, align 16
  %1740 = add i32 %1585, %1320
  %1741 = inttoptr i32 %1740 to ptr addrspace(3)
  %1742 = load <4 x i32>, ptr addrspace(3) %1741, align 16
  %1743 = add i32 %1585, %1325
  %1744 = inttoptr i32 %1743 to ptr addrspace(3)
  %1745 = load <4 x i32>, ptr addrspace(3) %1744, align 16
  %1746 = shufflevector <4 x i32> %1736, <4 x i32> %1739, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1747 = shufflevector <4 x i32> %1742, <4 x i32> %1745, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1748 = shufflevector <8 x i32> %1746, <8 x i32> %1747, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1749 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1723, i32 0, <16 x i32> %1748, i16 0, <8 x float> %1708, i32 0, i32 0, i32 %1728, i32 0, i32 0, i32 %1733, i1 false, i1 false)
  %1750 = add i32 %1585, %1335
  %1751 = inttoptr i32 %1750 to ptr addrspace(3)
  %1752 = load <4 x i32>, ptr addrspace(3) %1751, align 16
  %1753 = add i32 %1585, %1341
  %1754 = inttoptr i32 %1753 to ptr addrspace(3)
  %1755 = load <4 x i32>, ptr addrspace(3) %1754, align 16
  %1756 = add i32 %1585, %1346
  %1757 = inttoptr i32 %1756 to ptr addrspace(3)
  %1758 = load <4 x i32>, ptr addrspace(3) %1757, align 16
  %1759 = add i32 %1585, %1351
  %1760 = inttoptr i32 %1759 to ptr addrspace(3)
  %1761 = load <4 x i32>, ptr addrspace(3) %1760, align 16
  %1762 = shufflevector <4 x i32> %1752, <4 x i32> %1755, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1763 = shufflevector <4 x i32> %1758, <4 x i32> %1761, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1764 = shufflevector <8 x i32> %1762, <8 x i32> %1763, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1765 = add i32 %1583, %1358
  %1766 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1765
  %1767 = load i8, ptr addrspace(3) %1766, align 1
  %1768 = zext i8 %1767 to i32
  %1769 = call i32 @llvm.amdgcn.perm(i32 %1768, i32 %1768, i32 0)
  %1770 = add i32 %1583, %1364
  %1771 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1770
  %1772 = load i8, ptr addrspace(3) %1771, align 1
  %1773 = zext i8 %1772 to i32
  %1774 = call i32 @llvm.amdgcn.perm(i32 %1773, i32 %1773, i32 0)
  %1775 = add i32 %1585, %1371
  %1776 = inttoptr i32 %1775 to ptr addrspace(3)
  %1777 = load <4 x i32>, ptr addrspace(3) %1776, align 16
  %1778 = add i32 %1585, %1377
  %1779 = inttoptr i32 %1778 to ptr addrspace(3)
  %1780 = load <4 x i32>, ptr addrspace(3) %1779, align 16
  %1781 = add i32 %1585, %1382
  %1782 = inttoptr i32 %1781 to ptr addrspace(3)
  %1783 = load <4 x i32>, ptr addrspace(3) %1782, align 16
  %1784 = add i32 %1585, %1387
  %1785 = inttoptr i32 %1784 to ptr addrspace(3)
  %1786 = load <4 x i32>, ptr addrspace(3) %1785, align 16
  %1787 = shufflevector <4 x i32> %1777, <4 x i32> %1780, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1788 = shufflevector <4 x i32> %1783, <4 x i32> %1786, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1789 = shufflevector <8 x i32> %1787, <8 x i32> %1788, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1790 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1764, i32 0, <16 x i32> %1789, i16 0, <8 x float> %1749, i32 0, i32 0, i32 %1769, i32 0, i32 0, i32 %1774, i1 false, i1 false)
  %1791 = add i32 %1585, %1397
  %1792 = inttoptr i32 %1791 to ptr addrspace(3)
  %1793 = load <4 x i32>, ptr addrspace(3) %1792, align 16
  %1794 = add i32 %1585, %1403
  %1795 = inttoptr i32 %1794 to ptr addrspace(3)
  %1796 = load <4 x i32>, ptr addrspace(3) %1795, align 16
  %1797 = add i32 %1585, %1408
  %1798 = inttoptr i32 %1797 to ptr addrspace(3)
  %1799 = load <4 x i32>, ptr addrspace(3) %1798, align 16
  %1800 = add i32 %1585, %1413
  %1801 = inttoptr i32 %1800 to ptr addrspace(3)
  %1802 = load <4 x i32>, ptr addrspace(3) %1801, align 16
  %1803 = shufflevector <4 x i32> %1793, <4 x i32> %1796, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1804 = shufflevector <4 x i32> %1799, <4 x i32> %1802, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1805 = shufflevector <8 x i32> %1803, <8 x i32> %1804, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1806 = add i32 %1583, %1420
  %1807 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1806
  %1808 = load i8, ptr addrspace(3) %1807, align 1
  %1809 = zext i8 %1808 to i32
  %1810 = call i32 @llvm.amdgcn.perm(i32 %1809, i32 %1809, i32 0)
  %1811 = add i32 %1583, %1426
  %1812 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1811
  %1813 = load i8, ptr addrspace(3) %1812, align 1
  %1814 = zext i8 %1813 to i32
  %1815 = call i32 @llvm.amdgcn.perm(i32 %1814, i32 %1814, i32 0)
  %1816 = add i32 %1585, %1433
  %1817 = inttoptr i32 %1816 to ptr addrspace(3)
  %1818 = load <4 x i32>, ptr addrspace(3) %1817, align 16
  %1819 = add i32 %1585, %1439
  %1820 = inttoptr i32 %1819 to ptr addrspace(3)
  %1821 = load <4 x i32>, ptr addrspace(3) %1820, align 16
  %1822 = add i32 %1585, %1444
  %1823 = inttoptr i32 %1822 to ptr addrspace(3)
  %1824 = load <4 x i32>, ptr addrspace(3) %1823, align 16
  %1825 = add i32 %1585, %1449
  %1826 = inttoptr i32 %1825 to ptr addrspace(3)
  %1827 = load <4 x i32>, ptr addrspace(3) %1826, align 16
  %1828 = shufflevector <4 x i32> %1818, <4 x i32> %1821, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1829 = shufflevector <4 x i32> %1824, <4 x i32> %1827, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1830 = shufflevector <8 x i32> %1828, <8 x i32> %1829, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1831 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1805, i32 0, <16 x i32> %1830, i16 0, <8 x float> %1790, i32 0, i32 0, i32 %1810, i32 0, i32 0, i32 %1815, i1 false, i1 false)
  %1832 = add i32 %1585, %1459
  %1833 = inttoptr i32 %1832 to ptr addrspace(3)
  %1834 = load <4 x i32>, ptr addrspace(3) %1833, align 16
  %1835 = add i32 %1585, %1465
  %1836 = inttoptr i32 %1835 to ptr addrspace(3)
  %1837 = load <4 x i32>, ptr addrspace(3) %1836, align 16
  %1838 = add i32 %1585, %1470
  %1839 = inttoptr i32 %1838 to ptr addrspace(3)
  %1840 = load <4 x i32>, ptr addrspace(3) %1839, align 16
  %1841 = add i32 %1585, %1475
  %1842 = inttoptr i32 %1841 to ptr addrspace(3)
  %1843 = load <4 x i32>, ptr addrspace(3) %1842, align 16
  %1844 = shufflevector <4 x i32> %1834, <4 x i32> %1837, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1845 = shufflevector <4 x i32> %1840, <4 x i32> %1843, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1846 = shufflevector <8 x i32> %1844, <8 x i32> %1845, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1847 = add i32 %1583, %1482
  %1848 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1847
  %1849 = load i8, ptr addrspace(3) %1848, align 1
  %1850 = zext i8 %1849 to i32
  %1851 = call i32 @llvm.amdgcn.perm(i32 %1850, i32 %1850, i32 0)
  %1852 = add i32 %1583, %1488
  %1853 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1852
  %1854 = load i8, ptr addrspace(3) %1853, align 1
  %1855 = zext i8 %1854 to i32
  %1856 = call i32 @llvm.amdgcn.perm(i32 %1855, i32 %1855, i32 0)
  %1857 = add i32 %1585, %1495
  %1858 = inttoptr i32 %1857 to ptr addrspace(3)
  %1859 = load <4 x i32>, ptr addrspace(3) %1858, align 16
  %1860 = add i32 %1585, %1501
  %1861 = inttoptr i32 %1860 to ptr addrspace(3)
  %1862 = load <4 x i32>, ptr addrspace(3) %1861, align 16
  %1863 = add i32 %1585, %1506
  %1864 = inttoptr i32 %1863 to ptr addrspace(3)
  %1865 = load <4 x i32>, ptr addrspace(3) %1864, align 16
  %1866 = add i32 %1585, %1511
  %1867 = inttoptr i32 %1866 to ptr addrspace(3)
  %1868 = load <4 x i32>, ptr addrspace(3) %1867, align 16
  %1869 = shufflevector <4 x i32> %1859, <4 x i32> %1862, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1870 = shufflevector <4 x i32> %1865, <4 x i32> %1868, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1871 = shufflevector <8 x i32> %1869, <8 x i32> %1870, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1872 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1846, i32 0, <16 x i32> %1871, i16 0, <8 x float> %1831, i32 0, i32 0, i32 %1851, i32 0, i32 0, i32 %1856, i1 false, i1 false)
  %1873 = add i32 %1585, %1521
  %1874 = inttoptr i32 %1873 to ptr addrspace(3)
  %1875 = load <4 x i32>, ptr addrspace(3) %1874, align 16
  %1876 = add i32 %1585, %1527
  %1877 = inttoptr i32 %1876 to ptr addrspace(3)
  %1878 = load <4 x i32>, ptr addrspace(3) %1877, align 16
  %1879 = add i32 %1585, %1532
  %1880 = inttoptr i32 %1879 to ptr addrspace(3)
  %1881 = load <4 x i32>, ptr addrspace(3) %1880, align 16
  %1882 = add i32 %1585, %1537
  %1883 = inttoptr i32 %1882 to ptr addrspace(3)
  %1884 = load <4 x i32>, ptr addrspace(3) %1883, align 16
  %1885 = shufflevector <4 x i32> %1875, <4 x i32> %1878, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1886 = shufflevector <4 x i32> %1881, <4 x i32> %1884, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1887 = shufflevector <8 x i32> %1885, <8 x i32> %1886, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1888 = add i32 %1583, %1544
  %1889 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1888
  %1890 = load i8, ptr addrspace(3) %1889, align 1
  %1891 = zext i8 %1890 to i32
  %1892 = call i32 @llvm.amdgcn.perm(i32 %1891, i32 %1891, i32 0)
  %1893 = add i32 %1583, %1550
  %1894 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1893
  %1895 = load i8, ptr addrspace(3) %1894, align 1
  %1896 = zext i8 %1895 to i32
  %1897 = call i32 @llvm.amdgcn.perm(i32 %1896, i32 %1896, i32 0)
  %1898 = add i32 %1585, %1557
  %1899 = inttoptr i32 %1898 to ptr addrspace(3)
  %1900 = load <4 x i32>, ptr addrspace(3) %1899, align 16
  %1901 = add i32 %1585, %1563
  %1902 = inttoptr i32 %1901 to ptr addrspace(3)
  %1903 = load <4 x i32>, ptr addrspace(3) %1902, align 16
  %1904 = add i32 %1585, %1568
  %1905 = inttoptr i32 %1904 to ptr addrspace(3)
  %1906 = load <4 x i32>, ptr addrspace(3) %1905, align 16
  %1907 = add i32 %1585, %1573
  %1908 = inttoptr i32 %1907 to ptr addrspace(3)
  %1909 = load <4 x i32>, ptr addrspace(3) %1908, align 16
  %1910 = shufflevector <4 x i32> %1900, <4 x i32> %1903, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1911 = shufflevector <4 x i32> %1906, <4 x i32> %1909, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1912 = shufflevector <8 x i32> %1910, <8 x i32> %1911, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1913 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1887, i32 0, <16 x i32> %1912, i16 0, <8 x float> %1872, i32 0, i32 0, i32 %1892, i32 0, i32 0, i32 %1897, i1 false, i1 false)
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
  %1914 = add i32 %19, -1
  %1915 = srem i32 %1914, 4
  %1916 = mul i32 %1915, 66560
  %1917 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1916
  %1918 = ptrtoint ptr addrspace(3) %1917 to i32
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %1919 = add i32 %1918, %1075
  %1920 = inttoptr i32 %1919 to ptr addrspace(3)
  %1921 = load <4 x i32>, ptr addrspace(3) %1920, align 16
  %1922 = add i32 %1918, %1081
  %1923 = inttoptr i32 %1922 to ptr addrspace(3)
  %1924 = load <4 x i32>, ptr addrspace(3) %1923, align 16
  %1925 = add i32 %1918, %1086
  %1926 = inttoptr i32 %1925 to ptr addrspace(3)
  %1927 = load <4 x i32>, ptr addrspace(3) %1926, align 16
  %1928 = add i32 %1918, %1091
  %1929 = inttoptr i32 %1928 to ptr addrspace(3)
  %1930 = load <4 x i32>, ptr addrspace(3) %1929, align 16
  %1931 = shufflevector <4 x i32> %1921, <4 x i32> %1924, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1932 = shufflevector <4 x i32> %1927, <4 x i32> %1930, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1933 = shufflevector <8 x i32> %1931, <8 x i32> %1932, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1934 = add i32 %1916, %1107
  %1935 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1934
  %1936 = load i8, ptr addrspace(3) %1935, align 1
  %1937 = zext i8 %1936 to i32
  %1938 = call i32 @llvm.amdgcn.perm(i32 %1937, i32 %1937, i32 0)
  %1939 = add i32 %1916, %1115
  %1940 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1939
  %1941 = load i8, ptr addrspace(3) %1940, align 1
  %1942 = zext i8 %1941 to i32
  %1943 = call i32 @llvm.amdgcn.perm(i32 %1942, i32 %1942, i32 0)
  %1944 = add i32 %1918, %1123
  %1945 = inttoptr i32 %1944 to ptr addrspace(3)
  %1946 = load <4 x i32>, ptr addrspace(3) %1945, align 16
  %1947 = add i32 %1918, %1129
  %1948 = inttoptr i32 %1947 to ptr addrspace(3)
  %1949 = load <4 x i32>, ptr addrspace(3) %1948, align 16
  %1950 = add i32 %1918, %1134
  %1951 = inttoptr i32 %1950 to ptr addrspace(3)
  %1952 = load <4 x i32>, ptr addrspace(3) %1951, align 16
  %1953 = add i32 %1918, %1139
  %1954 = inttoptr i32 %1953 to ptr addrspace(3)
  %1955 = load <4 x i32>, ptr addrspace(3) %1954, align 16
  %1956 = shufflevector <4 x i32> %1946, <4 x i32> %1949, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1957 = shufflevector <4 x i32> %1952, <4 x i32> %1955, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1958 = shufflevector <8 x i32> %1956, <8 x i32> %1957, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1959 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1933, i32 0, <16 x i32> %1958, i16 0, <8 x float> %1913, i32 0, i32 0, i32 %1938, i32 0, i32 0, i32 %1943, i1 false, i1 false)
  %1960 = add i32 %1918, %1149
  %1961 = inttoptr i32 %1960 to ptr addrspace(3)
  %1962 = load <4 x i32>, ptr addrspace(3) %1961, align 16
  %1963 = add i32 %1918, %1155
  %1964 = inttoptr i32 %1963 to ptr addrspace(3)
  %1965 = load <4 x i32>, ptr addrspace(3) %1964, align 16
  %1966 = add i32 %1918, %1160
  %1967 = inttoptr i32 %1966 to ptr addrspace(3)
  %1968 = load <4 x i32>, ptr addrspace(3) %1967, align 16
  %1969 = add i32 %1918, %1165
  %1970 = inttoptr i32 %1969 to ptr addrspace(3)
  %1971 = load <4 x i32>, ptr addrspace(3) %1970, align 16
  %1972 = shufflevector <4 x i32> %1962, <4 x i32> %1965, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1973 = shufflevector <4 x i32> %1968, <4 x i32> %1971, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1974 = shufflevector <8 x i32> %1972, <8 x i32> %1973, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1975 = add i32 %1916, %1172
  %1976 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1975
  %1977 = load i8, ptr addrspace(3) %1976, align 1
  %1978 = zext i8 %1977 to i32
  %1979 = call i32 @llvm.amdgcn.perm(i32 %1978, i32 %1978, i32 0)
  %1980 = add i32 %1916, %1178
  %1981 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1980
  %1982 = load i8, ptr addrspace(3) %1981, align 1
  %1983 = zext i8 %1982 to i32
  %1984 = call i32 @llvm.amdgcn.perm(i32 %1983, i32 %1983, i32 0)
  %1985 = add i32 %1918, %1185
  %1986 = inttoptr i32 %1985 to ptr addrspace(3)
  %1987 = load <4 x i32>, ptr addrspace(3) %1986, align 16
  %1988 = add i32 %1918, %1191
  %1989 = inttoptr i32 %1988 to ptr addrspace(3)
  %1990 = load <4 x i32>, ptr addrspace(3) %1989, align 16
  %1991 = add i32 %1918, %1196
  %1992 = inttoptr i32 %1991 to ptr addrspace(3)
  %1993 = load <4 x i32>, ptr addrspace(3) %1992, align 16
  %1994 = add i32 %1918, %1201
  %1995 = inttoptr i32 %1994 to ptr addrspace(3)
  %1996 = load <4 x i32>, ptr addrspace(3) %1995, align 16
  %1997 = shufflevector <4 x i32> %1987, <4 x i32> %1990, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1998 = shufflevector <4 x i32> %1993, <4 x i32> %1996, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1999 = shufflevector <8 x i32> %1997, <8 x i32> %1998, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %2000 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1974, i32 0, <16 x i32> %1999, i16 0, <8 x float> %1959, i32 0, i32 0, i32 %1979, i32 0, i32 0, i32 %1984, i1 false, i1 false)
  %2001 = add i32 %1918, %1211
  %2002 = inttoptr i32 %2001 to ptr addrspace(3)
  %2003 = load <4 x i32>, ptr addrspace(3) %2002, align 16
  %2004 = add i32 %1918, %1217
  %2005 = inttoptr i32 %2004 to ptr addrspace(3)
  %2006 = load <4 x i32>, ptr addrspace(3) %2005, align 16
  %2007 = add i32 %1918, %1222
  %2008 = inttoptr i32 %2007 to ptr addrspace(3)
  %2009 = load <4 x i32>, ptr addrspace(3) %2008, align 16
  %2010 = add i32 %1918, %1227
  %2011 = inttoptr i32 %2010 to ptr addrspace(3)
  %2012 = load <4 x i32>, ptr addrspace(3) %2011, align 16
  %2013 = shufflevector <4 x i32> %2003, <4 x i32> %2006, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2014 = shufflevector <4 x i32> %2009, <4 x i32> %2012, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2015 = shufflevector <8 x i32> %2013, <8 x i32> %2014, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2016 = add i32 %1916, %1234
  %2017 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %2016
  %2018 = load i8, ptr addrspace(3) %2017, align 1
  %2019 = zext i8 %2018 to i32
  %2020 = call i32 @llvm.amdgcn.perm(i32 %2019, i32 %2019, i32 0)
  %2021 = add i32 %1916, %1240
  %2022 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %2021
  %2023 = load i8, ptr addrspace(3) %2022, align 1
  %2024 = zext i8 %2023 to i32
  %2025 = call i32 @llvm.amdgcn.perm(i32 %2024, i32 %2024, i32 0)
  %2026 = add i32 %1918, %1247
  %2027 = inttoptr i32 %2026 to ptr addrspace(3)
  %2028 = load <4 x i32>, ptr addrspace(3) %2027, align 16
  %2029 = add i32 %1918, %1253
  %2030 = inttoptr i32 %2029 to ptr addrspace(3)
  %2031 = load <4 x i32>, ptr addrspace(3) %2030, align 16
  %2032 = add i32 %1918, %1258
  %2033 = inttoptr i32 %2032 to ptr addrspace(3)
  %2034 = load <4 x i32>, ptr addrspace(3) %2033, align 16
  %2035 = add i32 %1918, %1263
  %2036 = inttoptr i32 %2035 to ptr addrspace(3)
  %2037 = load <4 x i32>, ptr addrspace(3) %2036, align 16
  %2038 = shufflevector <4 x i32> %2028, <4 x i32> %2031, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2039 = shufflevector <4 x i32> %2034, <4 x i32> %2037, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2040 = shufflevector <8 x i32> %2038, <8 x i32> %2039, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %2041 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %2015, i32 0, <16 x i32> %2040, i16 0, <8 x float> %2000, i32 0, i32 0, i32 %2020, i32 0, i32 0, i32 %2025, i1 false, i1 false)
  %2042 = add i32 %1918, %1273
  %2043 = inttoptr i32 %2042 to ptr addrspace(3)
  %2044 = load <4 x i32>, ptr addrspace(3) %2043, align 16
  %2045 = add i32 %1918, %1279
  %2046 = inttoptr i32 %2045 to ptr addrspace(3)
  %2047 = load <4 x i32>, ptr addrspace(3) %2046, align 16
  %2048 = add i32 %1918, %1284
  %2049 = inttoptr i32 %2048 to ptr addrspace(3)
  %2050 = load <4 x i32>, ptr addrspace(3) %2049, align 16
  %2051 = add i32 %1918, %1289
  %2052 = inttoptr i32 %2051 to ptr addrspace(3)
  %2053 = load <4 x i32>, ptr addrspace(3) %2052, align 16
  %2054 = shufflevector <4 x i32> %2044, <4 x i32> %2047, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2055 = shufflevector <4 x i32> %2050, <4 x i32> %2053, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2056 = shufflevector <8 x i32> %2054, <8 x i32> %2055, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2057 = add i32 %1916, %1296
  %2058 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %2057
  %2059 = load i8, ptr addrspace(3) %2058, align 1
  %2060 = zext i8 %2059 to i32
  %2061 = call i32 @llvm.amdgcn.perm(i32 %2060, i32 %2060, i32 0)
  %2062 = add i32 %1916, %1302
  %2063 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %2062
  %2064 = load i8, ptr addrspace(3) %2063, align 1
  %2065 = zext i8 %2064 to i32
  %2066 = call i32 @llvm.amdgcn.perm(i32 %2065, i32 %2065, i32 0)
  %2067 = add i32 %1918, %1309
  %2068 = inttoptr i32 %2067 to ptr addrspace(3)
  %2069 = load <4 x i32>, ptr addrspace(3) %2068, align 16
  %2070 = add i32 %1918, %1315
  %2071 = inttoptr i32 %2070 to ptr addrspace(3)
  %2072 = load <4 x i32>, ptr addrspace(3) %2071, align 16
  %2073 = add i32 %1918, %1320
  %2074 = inttoptr i32 %2073 to ptr addrspace(3)
  %2075 = load <4 x i32>, ptr addrspace(3) %2074, align 16
  %2076 = add i32 %1918, %1325
  %2077 = inttoptr i32 %2076 to ptr addrspace(3)
  %2078 = load <4 x i32>, ptr addrspace(3) %2077, align 16
  %2079 = shufflevector <4 x i32> %2069, <4 x i32> %2072, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2080 = shufflevector <4 x i32> %2075, <4 x i32> %2078, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2081 = shufflevector <8 x i32> %2079, <8 x i32> %2080, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %2082 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %2056, i32 0, <16 x i32> %2081, i16 0, <8 x float> %2041, i32 0, i32 0, i32 %2061, i32 0, i32 0, i32 %2066, i1 false, i1 false)
  %2083 = add i32 %1918, %1335
  %2084 = inttoptr i32 %2083 to ptr addrspace(3)
  %2085 = load <4 x i32>, ptr addrspace(3) %2084, align 16
  %2086 = add i32 %1918, %1341
  %2087 = inttoptr i32 %2086 to ptr addrspace(3)
  %2088 = load <4 x i32>, ptr addrspace(3) %2087, align 16
  %2089 = add i32 %1918, %1346
  %2090 = inttoptr i32 %2089 to ptr addrspace(3)
  %2091 = load <4 x i32>, ptr addrspace(3) %2090, align 16
  %2092 = add i32 %1918, %1351
  %2093 = inttoptr i32 %2092 to ptr addrspace(3)
  %2094 = load <4 x i32>, ptr addrspace(3) %2093, align 16
  %2095 = shufflevector <4 x i32> %2085, <4 x i32> %2088, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2096 = shufflevector <4 x i32> %2091, <4 x i32> %2094, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2097 = shufflevector <8 x i32> %2095, <8 x i32> %2096, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2098 = add i32 %1916, %1358
  %2099 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %2098
  %2100 = load i8, ptr addrspace(3) %2099, align 1
  %2101 = zext i8 %2100 to i32
  %2102 = call i32 @llvm.amdgcn.perm(i32 %2101, i32 %2101, i32 0)
  %2103 = add i32 %1916, %1364
  %2104 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %2103
  %2105 = load i8, ptr addrspace(3) %2104, align 1
  %2106 = zext i8 %2105 to i32
  %2107 = call i32 @llvm.amdgcn.perm(i32 %2106, i32 %2106, i32 0)
  %2108 = add i32 %1918, %1371
  %2109 = inttoptr i32 %2108 to ptr addrspace(3)
  %2110 = load <4 x i32>, ptr addrspace(3) %2109, align 16
  %2111 = add i32 %1918, %1377
  %2112 = inttoptr i32 %2111 to ptr addrspace(3)
  %2113 = load <4 x i32>, ptr addrspace(3) %2112, align 16
  %2114 = add i32 %1918, %1382
  %2115 = inttoptr i32 %2114 to ptr addrspace(3)
  %2116 = load <4 x i32>, ptr addrspace(3) %2115, align 16
  %2117 = add i32 %1918, %1387
  %2118 = inttoptr i32 %2117 to ptr addrspace(3)
  %2119 = load <4 x i32>, ptr addrspace(3) %2118, align 16
  %2120 = shufflevector <4 x i32> %2110, <4 x i32> %2113, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2121 = shufflevector <4 x i32> %2116, <4 x i32> %2119, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2122 = shufflevector <8 x i32> %2120, <8 x i32> %2121, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %2123 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %2097, i32 0, <16 x i32> %2122, i16 0, <8 x float> %2082, i32 0, i32 0, i32 %2102, i32 0, i32 0, i32 %2107, i1 false, i1 false)
  %2124 = add i32 %1918, %1397
  %2125 = inttoptr i32 %2124 to ptr addrspace(3)
  %2126 = load <4 x i32>, ptr addrspace(3) %2125, align 16
  %2127 = add i32 %1918, %1403
  %2128 = inttoptr i32 %2127 to ptr addrspace(3)
  %2129 = load <4 x i32>, ptr addrspace(3) %2128, align 16
  %2130 = add i32 %1918, %1408
  %2131 = inttoptr i32 %2130 to ptr addrspace(3)
  %2132 = load <4 x i32>, ptr addrspace(3) %2131, align 16
  %2133 = add i32 %1918, %1413
  %2134 = inttoptr i32 %2133 to ptr addrspace(3)
  %2135 = load <4 x i32>, ptr addrspace(3) %2134, align 16
  %2136 = shufflevector <4 x i32> %2126, <4 x i32> %2129, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2137 = shufflevector <4 x i32> %2132, <4 x i32> %2135, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2138 = shufflevector <8 x i32> %2136, <8 x i32> %2137, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2139 = add i32 %1916, %1420
  %2140 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %2139
  %2141 = load i8, ptr addrspace(3) %2140, align 1
  %2142 = zext i8 %2141 to i32
  %2143 = call i32 @llvm.amdgcn.perm(i32 %2142, i32 %2142, i32 0)
  %2144 = add i32 %1916, %1426
  %2145 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %2144
  %2146 = load i8, ptr addrspace(3) %2145, align 1
  %2147 = zext i8 %2146 to i32
  %2148 = call i32 @llvm.amdgcn.perm(i32 %2147, i32 %2147, i32 0)
  %2149 = add i32 %1918, %1433
  %2150 = inttoptr i32 %2149 to ptr addrspace(3)
  %2151 = load <4 x i32>, ptr addrspace(3) %2150, align 16
  %2152 = add i32 %1918, %1439
  %2153 = inttoptr i32 %2152 to ptr addrspace(3)
  %2154 = load <4 x i32>, ptr addrspace(3) %2153, align 16
  %2155 = add i32 %1918, %1444
  %2156 = inttoptr i32 %2155 to ptr addrspace(3)
  %2157 = load <4 x i32>, ptr addrspace(3) %2156, align 16
  %2158 = add i32 %1918, %1449
  %2159 = inttoptr i32 %2158 to ptr addrspace(3)
  %2160 = load <4 x i32>, ptr addrspace(3) %2159, align 16
  %2161 = shufflevector <4 x i32> %2151, <4 x i32> %2154, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2162 = shufflevector <4 x i32> %2157, <4 x i32> %2160, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2163 = shufflevector <8 x i32> %2161, <8 x i32> %2162, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %2164 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %2138, i32 0, <16 x i32> %2163, i16 0, <8 x float> %2123, i32 0, i32 0, i32 %2143, i32 0, i32 0, i32 %2148, i1 false, i1 false)
  %2165 = add i32 %1918, %1459
  %2166 = inttoptr i32 %2165 to ptr addrspace(3)
  %2167 = load <4 x i32>, ptr addrspace(3) %2166, align 16
  %2168 = add i32 %1918, %1465
  %2169 = inttoptr i32 %2168 to ptr addrspace(3)
  %2170 = load <4 x i32>, ptr addrspace(3) %2169, align 16
  %2171 = add i32 %1918, %1470
  %2172 = inttoptr i32 %2171 to ptr addrspace(3)
  %2173 = load <4 x i32>, ptr addrspace(3) %2172, align 16
  %2174 = add i32 %1918, %1475
  %2175 = inttoptr i32 %2174 to ptr addrspace(3)
  %2176 = load <4 x i32>, ptr addrspace(3) %2175, align 16
  %2177 = shufflevector <4 x i32> %2167, <4 x i32> %2170, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2178 = shufflevector <4 x i32> %2173, <4 x i32> %2176, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2179 = shufflevector <8 x i32> %2177, <8 x i32> %2178, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2180 = add i32 %1916, %1482
  %2181 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %2180
  %2182 = load i8, ptr addrspace(3) %2181, align 1
  %2183 = zext i8 %2182 to i32
  %2184 = call i32 @llvm.amdgcn.perm(i32 %2183, i32 %2183, i32 0)
  %2185 = add i32 %1916, %1488
  %2186 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %2185
  %2187 = load i8, ptr addrspace(3) %2186, align 1
  %2188 = zext i8 %2187 to i32
  %2189 = call i32 @llvm.amdgcn.perm(i32 %2188, i32 %2188, i32 0)
  %2190 = add i32 %1918, %1495
  %2191 = inttoptr i32 %2190 to ptr addrspace(3)
  %2192 = load <4 x i32>, ptr addrspace(3) %2191, align 16
  %2193 = add i32 %1918, %1501
  %2194 = inttoptr i32 %2193 to ptr addrspace(3)
  %2195 = load <4 x i32>, ptr addrspace(3) %2194, align 16
  %2196 = add i32 %1918, %1506
  %2197 = inttoptr i32 %2196 to ptr addrspace(3)
  %2198 = load <4 x i32>, ptr addrspace(3) %2197, align 16
  %2199 = add i32 %1918, %1511
  %2200 = inttoptr i32 %2199 to ptr addrspace(3)
  %2201 = load <4 x i32>, ptr addrspace(3) %2200, align 16
  %2202 = shufflevector <4 x i32> %2192, <4 x i32> %2195, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2203 = shufflevector <4 x i32> %2198, <4 x i32> %2201, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2204 = shufflevector <8 x i32> %2202, <8 x i32> %2203, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %2205 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %2179, i32 0, <16 x i32> %2204, i16 0, <8 x float> %2164, i32 0, i32 0, i32 %2184, i32 0, i32 0, i32 %2189, i1 false, i1 false)
  %2206 = add i32 %1918, %1521
  %2207 = inttoptr i32 %2206 to ptr addrspace(3)
  %2208 = load <4 x i32>, ptr addrspace(3) %2207, align 16
  %2209 = add i32 %1918, %1527
  %2210 = inttoptr i32 %2209 to ptr addrspace(3)
  %2211 = load <4 x i32>, ptr addrspace(3) %2210, align 16
  %2212 = add i32 %1918, %1532
  %2213 = inttoptr i32 %2212 to ptr addrspace(3)
  %2214 = load <4 x i32>, ptr addrspace(3) %2213, align 16
  %2215 = add i32 %1918, %1537
  %2216 = inttoptr i32 %2215 to ptr addrspace(3)
  %2217 = load <4 x i32>, ptr addrspace(3) %2216, align 16
  %2218 = shufflevector <4 x i32> %2208, <4 x i32> %2211, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2219 = shufflevector <4 x i32> %2214, <4 x i32> %2217, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2220 = shufflevector <8 x i32> %2218, <8 x i32> %2219, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2221 = add i32 %1916, %1544
  %2222 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %2221
  %2223 = load i8, ptr addrspace(3) %2222, align 1
  %2224 = zext i8 %2223 to i32
  %2225 = call i32 @llvm.amdgcn.perm(i32 %2224, i32 %2224, i32 0)
  %2226 = add i32 %1916, %1550
  %2227 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %2226
  %2228 = load i8, ptr addrspace(3) %2227, align 1
  %2229 = zext i8 %2228 to i32
  %2230 = call i32 @llvm.amdgcn.perm(i32 %2229, i32 %2229, i32 0)
  %2231 = add i32 %1918, %1557
  %2232 = inttoptr i32 %2231 to ptr addrspace(3)
  %2233 = load <4 x i32>, ptr addrspace(3) %2232, align 16
  %2234 = add i32 %1918, %1563
  %2235 = inttoptr i32 %2234 to ptr addrspace(3)
  %2236 = load <4 x i32>, ptr addrspace(3) %2235, align 16
  %2237 = add i32 %1918, %1568
  %2238 = inttoptr i32 %2237 to ptr addrspace(3)
  %2239 = load <4 x i32>, ptr addrspace(3) %2238, align 16
  %2240 = add i32 %1918, %1573
  %2241 = inttoptr i32 %2240 to ptr addrspace(3)
  %2242 = load <4 x i32>, ptr addrspace(3) %2241, align 16
  %2243 = shufflevector <4 x i32> %2233, <4 x i32> %2236, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2244 = shufflevector <4 x i32> %2239, <4 x i32> %2242, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2245 = shufflevector <8 x i32> %2243, <8 x i32> %2244, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %2246 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %2220, i32 0, <16 x i32> %2245, i16 0, <8 x float> %2205, i32 0, i32 0, i32 %2225, i32 0, i32 0, i32 %2230, i1 false, i1 false)
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
  %2247 = mul i32 %54, 8
  %2248 = add i32 %123, %2247
  %2249 = fptrunc <8 x float> %2246 to <8 x bfloat>
  %2250 = bitcast <8 x bfloat> %2249 to <16 x i8>
  %2251 = mul i32 %1113, 32
  %2252 = add i32 %2251, %2248
  %2253 = mul i32 %2252, 2
  %2254 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %2253
  store <16 x i8> %2250, ptr addrspace(3) %2254, align 1
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %2255 = mul i64 %66, %22
  %2256 = add i64 %2255, %67
  %2257 = add i64 %2256, %84
  %2258 = getelementptr bfloat, ptr addrspace(1) %0, i64 %2257
  %2259 = icmp eq i64 %22, -2147483648
  %2260 = select i1 %2259, i64 32, i64 %22
  %2261 = ptrtoint ptr addrspace(1) %2258 to i64
  %2262 = call i32 @llvm.amdgcn.wave.id()
  %2263 = urem i32 %2262, 4
  %2264 = mul i32 %2263, 8
  %2265 = zext i32 %2264 to i64
  %2266 = mul i64 %2265, %2260
  %2267 = mul i64 %2266, 2
  %2268 = add i64 %2261, %2267
  %2269 = mul i32 %2263, 512
  %2270 = add i32 ptrtoint (ptr addrspace(3) @__dynamic_shared_0 to i32), %2269
  %2271 = trunc i64 %2268 to i32
  %2272 = lshr i64 %2268, 32
  %2273 = trunc i64 %2272 to i32
  %2274 = or i32 %2273, -2147483648
  %2275 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %2270, i64 1
  %2276 = insertelement <4 x i32> %2275, i32 %2271, i64 2
  %2277 = insertelement <4 x i32> %2276, i32 %2274, i64 3
  %2278 = sub i32 %68, %2264
  %2279 = call i32 @llvm.smax.i32(i32 %2278, i32 0)
  %2280 = and i32 %2279, 65535
  %2281 = shl i32 %2280, 16
  %2282 = or i32 %2281, 32767
  %2283 = lshr i32 %2279, 16
  %2284 = and i32 %2283, 65535
  %2285 = or i32 %2284, 2097152
  %2286 = trunc i64 %2260 to i32
  %2287 = lshr i64 %2260, 32
  %2288 = trunc i64 %2287 to i32
  %2289 = and i32 %2288, 65535
  %2290 = insertelement <8 x i32> <i32 65536, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %2282, i64 2
  %2291 = insertelement <8 x i32> %2290, i32 %2285, i64 3
  %2292 = insertelement <8 x i32> %2291, i32 8, i64 4
  %2293 = insertelement <8 x i32> %2292, i32 %2286, i64 5
  %2294 = insertelement <8 x i32> %2293, i32 %2289, i64 6
  %2295 = insertelement <8 x i32> %2294, i32 0, i64 7
  call void @llvm.amdgcn.tensor.store.from.lds(<4 x i32> %2277, <8 x i32> %2295, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #1

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
