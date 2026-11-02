; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

@__dynamic_shared_0 = external dso_local addrspace(3) global [0 x i8], align 1024

define amdgpu_kernel void @gemm_a8w8_mx128_t16x32x1024_mw1_nw2_nb4_sk1_cm1_cn1(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10) #0 !reqd_work_group_size !1 {
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
  br i1 %100, label %194, label %222

194:                                              ; preds = %193
  %195 = icmp eq i64 %21, -2147483648
  %196 = select i1 %195, i64 1024, i64 %21
  %197 = ptrtoint ptr addrspace(1) %77 to i64
  %198 = add i64 %197, 1024
  %199 = trunc i64 %198 to i32
  %200 = lshr i64 %198, 32
  %201 = trunc i64 %200 to i32
  %202 = or i32 %201, -2147483648
  %203 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 50176) to i32), i32 poison, i32 poison>, i32 %199, i64 2
  %204 = insertelement <4 x i32> %203, i32 %202, i64 3
  %205 = call i32 @llvm.smax.i32(i32 %64, i32 0)
  %206 = and i32 %205, 65535
  %207 = shl i32 %206, 16
  %208 = or i32 %207, 32767
  %209 = lshr i32 %205, 16
  %210 = and i32 %209, 65535
  %211 = or i32 %210, 67108864
  %212 = trunc i64 %196 to i32
  %213 = lshr i64 %196, 32
  %214 = trunc i64 %213 to i32
  %215 = and i32 %214, 65535
  %216 = insertelement <8 x i32> <i32 133169152, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %208, i64 2
  %217 = insertelement <8 x i32> %216, i32 %211, i64 3
  %218 = insertelement <8 x i32> %217, i32 16, i64 4
  %219 = insertelement <8 x i32> %218, i32 %212, i64 5
  %220 = insertelement <8 x i32> %219, i32 %215, i64 6
  %221 = insertelement <8 x i32> %220, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %204, <8 x i32> %221, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %222

222:                                              ; preds = %194, %193
  br i1 %129, label %223, label %241

223:                                              ; preds = %222
  %224 = icmp eq i64 %75, -2147483648
  %225 = select i1 %224, i64 16384, i64 %75
  %226 = ptrtoint ptr addrspace(1) %78 to i64
  %227 = add i64 %226, 16384
  %228 = trunc i64 %227 to i32
  %229 = lshr i64 %227, 32
  %230 = trunc i64 %229 to i32
  %231 = or i32 %230, -2147483648
  %232 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 66816) to i32), i32 poison, i32 poison>, i32 %228, i64 2
  %233 = insertelement <4 x i32> %232, i32 %231, i64 3
  %234 = trunc i64 %225 to i32
  %235 = lshr i64 %225, 32
  %236 = trunc i64 %235 to i32
  %237 = and i32 %236, 65535
  %238 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 1073774591, i32 2, i32 poison, i32 poison, i32 poison>, i32 %234, i64 5
  %239 = insertelement <8 x i32> %238, i32 %237, i64 6
  %240 = insertelement <8 x i32> %239, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %233, <8 x i32> %240, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %241

241:                                              ; preds = %223, %222
  %242 = mul i64 %65, 8
  br i1 %100, label %243, label %271

243:                                              ; preds = %241
  %244 = icmp eq i64 %65, -2147483648
  %245 = select i1 %244, i64 16, i64 %65
  %246 = ptrtoint ptr addrspace(1) %96 to i64
  %247 = add i64 %246, %242
  %248 = trunc i64 %247 to i32
  %249 = lshr i64 %247, 32
  %250 = trunc i64 %249 to i32
  %251 = or i32 %250, -2147483648
  %252 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 99584) to i32), i32 poison, i32 poison>, i32 %248, i64 2
  %253 = insertelement <4 x i32> %252, i32 %251, i64 3
  %254 = call i32 @llvm.smax.i32(i32 %64, i32 0)
  %255 = and i32 %254, 65535
  %256 = shl i32 %255, 16
  %257 = lshr i32 %254, 16
  %258 = and i32 %257, 65535
  %259 = or i32 %258, -65536
  %260 = trunc i64 %245 to i32
  %261 = lshr i64 %245, 32
  %262 = trunc i64 %261 to i32
  %263 = and i32 %262, 65535
  %264 = insertelement <8 x i32> <i32 2097152, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %256, i64 1
  %265 = insertelement <8 x i32> %264, i32 %259, i64 2
  %266 = insertelement <8 x i32> %265, i32 1081343, i64 3
  %267 = insertelement <8 x i32> %266, i32 8, i64 4
  %268 = insertelement <8 x i32> %267, i32 %260, i64 5
  %269 = insertelement <8 x i32> %268, i32 %263, i64 6
  %270 = insertelement <8 x i32> %269, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %253, <8 x i32> %270, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %271

271:                                              ; preds = %243, %241
  br i1 %129, label %272, label %290

272:                                              ; preds = %271
  %273 = icmp eq i64 %94, -2147483648
  %274 = select i1 %273, i64 8, i64 %94
  %275 = ptrtoint ptr addrspace(1) %97 to i64
  %276 = add i64 %275, 8
  %277 = trunc i64 %276 to i32
  %278 = lshr i64 %276, 32
  %279 = trunc i64 %278 to i32
  %280 = or i32 %279, -2147483648
  %281 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 99712) to i32), i32 poison, i32 poison>, i32 %277, i64 2
  %282 = insertelement <4 x i32> %281, i32 %280, i64 3
  %283 = trunc i64 %274 to i32
  %284 = lshr i64 %274, 32
  %285 = trunc i64 %284 to i32
  %286 = and i32 %285, 65535
  %287 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 557055, i32 1, i32 poison, i32 poison, i32 poison>, i32 %283, i64 5
  %288 = insertelement <8 x i32> %287, i32 %286, i64 6
  %289 = insertelement <8 x i32> %288, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %282, <8 x i32> %289, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %290

290:                                              ; preds = %272, %271
  br i1 %100, label %291, label %319

291:                                              ; preds = %290
  %292 = icmp eq i64 %21, -2147483648
  %293 = select i1 %292, i64 1024, i64 %21
  %294 = ptrtoint ptr addrspace(1) %77 to i64
  %295 = add i64 %294, 2048
  %296 = trunc i64 %295 to i32
  %297 = lshr i64 %295, 32
  %298 = trunc i64 %297 to i32
  %299 = or i32 %298, -2147483648
  %300 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 100352) to i32), i32 poison, i32 poison>, i32 %296, i64 2
  %301 = insertelement <4 x i32> %300, i32 %299, i64 3
  %302 = call i32 @llvm.smax.i32(i32 %64, i32 0)
  %303 = and i32 %302, 65535
  %304 = shl i32 %303, 16
  %305 = or i32 %304, 32767
  %306 = lshr i32 %302, 16
  %307 = and i32 %306, 65535
  %308 = or i32 %307, 67108864
  %309 = trunc i64 %293 to i32
  %310 = lshr i64 %293, 32
  %311 = trunc i64 %310 to i32
  %312 = and i32 %311, 65535
  %313 = insertelement <8 x i32> <i32 133169152, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %305, i64 2
  %314 = insertelement <8 x i32> %313, i32 %308, i64 3
  %315 = insertelement <8 x i32> %314, i32 16, i64 4
  %316 = insertelement <8 x i32> %315, i32 %309, i64 5
  %317 = insertelement <8 x i32> %316, i32 %312, i64 6
  %318 = insertelement <8 x i32> %317, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %301, <8 x i32> %318, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %319

319:                                              ; preds = %291, %290
  br i1 %129, label %320, label %338

320:                                              ; preds = %319
  %321 = icmp eq i64 %75, -2147483648
  %322 = select i1 %321, i64 16384, i64 %75
  %323 = ptrtoint ptr addrspace(1) %78 to i64
  %324 = add i64 %323, 32768
  %325 = trunc i64 %324 to i32
  %326 = lshr i64 %324, 32
  %327 = trunc i64 %326 to i32
  %328 = or i32 %327, -2147483648
  %329 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 116992) to i32), i32 poison, i32 poison>, i32 %325, i64 2
  %330 = insertelement <4 x i32> %329, i32 %328, i64 3
  %331 = trunc i64 %322 to i32
  %332 = lshr i64 %322, 32
  %333 = trunc i64 %332 to i32
  %334 = and i32 %333, 65535
  %335 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 1073774591, i32 2, i32 poison, i32 poison, i32 poison>, i32 %331, i64 5
  %336 = insertelement <8 x i32> %335, i32 %334, i64 6
  %337 = insertelement <8 x i32> %336, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %330, <8 x i32> %337, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %338

338:                                              ; preds = %320, %319
  %339 = mul i64 %65, 16
  br i1 %100, label %340, label %368

340:                                              ; preds = %338
  %341 = icmp eq i64 %65, -2147483648
  %342 = select i1 %341, i64 16, i64 %65
  %343 = ptrtoint ptr addrspace(1) %96 to i64
  %344 = add i64 %343, %339
  %345 = trunc i64 %344 to i32
  %346 = lshr i64 %344, 32
  %347 = trunc i64 %346 to i32
  %348 = or i32 %347, -2147483648
  %349 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 149760) to i32), i32 poison, i32 poison>, i32 %345, i64 2
  %350 = insertelement <4 x i32> %349, i32 %348, i64 3
  %351 = call i32 @llvm.smax.i32(i32 %64, i32 0)
  %352 = and i32 %351, 65535
  %353 = shl i32 %352, 16
  %354 = lshr i32 %351, 16
  %355 = and i32 %354, 65535
  %356 = or i32 %355, -65536
  %357 = trunc i64 %342 to i32
  %358 = lshr i64 %342, 32
  %359 = trunc i64 %358 to i32
  %360 = and i32 %359, 65535
  %361 = insertelement <8 x i32> <i32 2097152, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %353, i64 1
  %362 = insertelement <8 x i32> %361, i32 %356, i64 2
  %363 = insertelement <8 x i32> %362, i32 1081343, i64 3
  %364 = insertelement <8 x i32> %363, i32 8, i64 4
  %365 = insertelement <8 x i32> %364, i32 %357, i64 5
  %366 = insertelement <8 x i32> %365, i32 %360, i64 6
  %367 = insertelement <8 x i32> %366, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %350, <8 x i32> %367, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %368

368:                                              ; preds = %340, %338
  br i1 %129, label %369, label %387

369:                                              ; preds = %368
  %370 = icmp eq i64 %94, -2147483648
  %371 = select i1 %370, i64 8, i64 %94
  %372 = ptrtoint ptr addrspace(1) %97 to i64
  %373 = add i64 %372, 16
  %374 = trunc i64 %373 to i32
  %375 = lshr i64 %373, 32
  %376 = trunc i64 %375 to i32
  %377 = or i32 %376, -2147483648
  %378 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 149888) to i32), i32 poison, i32 poison>, i32 %374, i64 2
  %379 = insertelement <4 x i32> %378, i32 %377, i64 3
  %380 = trunc i64 %371 to i32
  %381 = lshr i64 %371, 32
  %382 = trunc i64 %381 to i32
  %383 = and i32 %382, 65535
  %384 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 557055, i32 1, i32 poison, i32 poison, i32 poison>, i32 %380, i64 5
  %385 = insertelement <8 x i32> %384, i32 %383, i64 6
  %386 = insertelement <8 x i32> %385, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %379, <8 x i32> %386, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %387

387:                                              ; preds = %369, %368
  %388 = sub i32 %19, 3
  br label %389

389:                                              ; preds = %597, %387
  %390 = phi i32 [ %1032, %597 ], [ 0, %387 ]
  %391 = phi <8 x float> [ %1031, %597 ], [ zeroinitializer, %387 ]
  %392 = icmp slt i32 %390, %388
  br i1 %392, label %393, label %1033

393:                                              ; preds = %389
  %394 = srem i32 %390, 4
  %395 = mul i32 %394, 50176
  %396 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %395
  %397 = ptrtoint ptr addrspace(3) %396 to i32
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 2)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %398 = add i32 %390, 3
  %399 = sdiv i32 %99, 16
  %400 = mul i32 %399, 16
  %401 = icmp ne i32 %99, %400
  %402 = icmp slt i32 %99, 0
  %403 = icmp ne i1 %402, false
  %404 = and i1 %401, %403
  %405 = add i32 %399, -1
  %406 = select i1 %404, i32 %405, i32 %399
  %407 = mul i32 %406, 16384
  %408 = add i32 %407, 16640
  %409 = mul i32 %50, 256
  %410 = add i32 %408, %409
  %411 = mul i32 %42, 16
  %412 = add i32 %410, %411
  %413 = sext i32 %412 to i64
  %414 = add i32 %397, %412
  %415 = inttoptr i32 %414 to ptr addrspace(3)
  %416 = load <4 x i32>, ptr addrspace(3) %415, align 16
  %417 = add i64 %413, 512
  %418 = trunc i64 %417 to i32
  %419 = add i32 %397, %418
  %420 = inttoptr i32 %419 to ptr addrspace(3)
  %421 = load <4 x i32>, ptr addrspace(3) %420, align 16
  %422 = add i64 %413, 1024
  %423 = trunc i64 %422 to i32
  %424 = add i32 %397, %423
  %425 = inttoptr i32 %424 to ptr addrspace(3)
  %426 = load <4 x i32>, ptr addrspace(3) %425, align 16
  %427 = add i64 %413, 1536
  %428 = trunc i64 %427 to i32
  %429 = add i32 %397, %428
  %430 = inttoptr i32 %429 to ptr addrspace(3)
  %431 = load <4 x i32>, ptr addrspace(3) %430, align 16
  %432 = shufflevector <4 x i32> %416, <4 x i32> %421, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %433 = shufflevector <4 x i32> %426, <4 x i32> %431, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %434 = shufflevector <8 x i32> %432, <8 x i32> %433, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %435 = sdiv i32 %99, 128
  %436 = mul i32 %435, 128
  %437 = icmp ne i32 %99, %436
  %438 = icmp slt i32 %99, 0
  %439 = icmp ne i1 %438, false
  %440 = and i1 %437, %439
  %441 = add i32 %435, -1
  %442 = select i1 %440, i32 %441, i32 %435
  %443 = mul i32 %442, 8
  %444 = add i32 %443, 49536
  %445 = add i32 %395, %444
  %446 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %445
  %447 = load i8, ptr addrspace(3) %446, align 1
  %448 = zext i8 %447 to i32
  %449 = call i32 @llvm.amdgcn.perm(i32 %448, i32 %448, i32 0)
  %450 = add i32 %98, %42
  %451 = add i32 %450, 49408
  %452 = add i32 %395, %451
  %453 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %452
  %454 = load i8, ptr addrspace(3) %453, align 1
  %455 = zext i8 %454 to i32
  %456 = call i32 @llvm.amdgcn.perm(i32 %455, i32 %455, i32 0)
  %457 = mul i32 %450, 1040
  %458 = mul i32 %50, 16
  %459 = add i32 %457, %458
  %460 = sext i32 %459 to i64
  %461 = add i32 %397, %459
  %462 = inttoptr i32 %461 to ptr addrspace(3)
  %463 = load <4 x i32>, ptr addrspace(3) %462, align 16
  %464 = add i64 %460, 32
  %465 = trunc i64 %464 to i32
  %466 = add i32 %397, %465
  %467 = inttoptr i32 %466 to ptr addrspace(3)
  %468 = load <4 x i32>, ptr addrspace(3) %467, align 16
  %469 = add i64 %460, 64
  %470 = trunc i64 %469 to i32
  %471 = add i32 %397, %470
  %472 = inttoptr i32 %471 to ptr addrspace(3)
  %473 = load <4 x i32>, ptr addrspace(3) %472, align 16
  %474 = add i64 %460, 96
  %475 = trunc i64 %474 to i32
  %476 = add i32 %397, %475
  %477 = inttoptr i32 %476 to ptr addrspace(3)
  %478 = load <4 x i32>, ptr addrspace(3) %477, align 16
  %479 = shufflevector <4 x i32> %463, <4 x i32> %468, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %480 = shufflevector <4 x i32> %473, <4 x i32> %478, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %481 = shufflevector <8 x i32> %479, <8 x i32> %480, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %482 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %434, i32 0, <16 x i32> %481, i16 0, <8 x float> %391, i32 0, i32 0, i32 %449, i32 0, i32 0, i32 %456, i1 false, i1 false)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %483 = srem i32 %398, 4
  %484 = mul i32 %483, 50176
  %485 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %484
  %486 = sext i32 %398 to i64
  %487 = mul i64 %486, 1024
  br i1 %100, label %488, label %518

488:                                              ; preds = %393
  %489 = icmp eq i64 %21, -2147483648
  %490 = select i1 %489, i64 1024, i64 %21
  %491 = ptrtoint ptr addrspace(1) %77 to i64
  %492 = add i64 %491, %487
  %493 = ptrtoint ptr addrspace(3) %485 to i32
  %494 = trunc i64 %492 to i32
  %495 = lshr i64 %492, 32
  %496 = trunc i64 %495 to i32
  %497 = or i32 %496, -2147483648
  %498 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %493, i64 1
  %499 = insertelement <4 x i32> %498, i32 %494, i64 2
  %500 = insertelement <4 x i32> %499, i32 %497, i64 3
  %501 = call i32 @llvm.smax.i32(i32 %64, i32 0)
  %502 = and i32 %501, 65535
  %503 = shl i32 %502, 16
  %504 = or i32 %503, 32767
  %505 = lshr i32 %501, 16
  %506 = and i32 %505, 65535
  %507 = or i32 %506, 67108864
  %508 = trunc i64 %490 to i32
  %509 = lshr i64 %490, 32
  %510 = trunc i64 %509 to i32
  %511 = and i32 %510, 65535
  %512 = insertelement <8 x i32> <i32 133169152, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %504, i64 2
  %513 = insertelement <8 x i32> %512, i32 %507, i64 3
  %514 = insertelement <8 x i32> %513, i32 16, i64 4
  %515 = insertelement <8 x i32> %514, i32 %508, i64 5
  %516 = insertelement <8 x i32> %515, i32 %511, i64 6
  %517 = insertelement <8 x i32> %516, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %500, <8 x i32> %517, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %518

518:                                              ; preds = %488, %393
  %519 = getelementptr i8, ptr addrspace(3) %485, i32 16640
  %520 = mul i64 %486, 16384
  br i1 %129, label %521, label %541

521:                                              ; preds = %518
  %522 = icmp eq i64 %75, -2147483648
  %523 = select i1 %522, i64 16384, i64 %75
  %524 = ptrtoint ptr addrspace(1) %78 to i64
  %525 = add i64 %524, %520
  %526 = ptrtoint ptr addrspace(3) %519 to i32
  %527 = trunc i64 %525 to i32
  %528 = lshr i64 %525, 32
  %529 = trunc i64 %528 to i32
  %530 = or i32 %529, -2147483648
  %531 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %526, i64 1
  %532 = insertelement <4 x i32> %531, i32 %527, i64 2
  %533 = insertelement <4 x i32> %532, i32 %530, i64 3
  %534 = trunc i64 %523 to i32
  %535 = lshr i64 %523, 32
  %536 = trunc i64 %535 to i32
  %537 = and i32 %536, 65535
  %538 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 1073774591, i32 2, i32 poison, i32 poison, i32 poison>, i32 %534, i64 5
  %539 = insertelement <8 x i32> %538, i32 %537, i64 6
  %540 = insertelement <8 x i32> %539, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %533, <8 x i32> %540, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %541

541:                                              ; preds = %521, %518
  %542 = getelementptr i8, ptr addrspace(3) %485, i32 49408
  %543 = mul i64 %486, 8
  %544 = mul i64 %543, %65
  br i1 %100, label %545, label %575

545:                                              ; preds = %541
  %546 = icmp eq i64 %65, -2147483648
  %547 = select i1 %546, i64 16, i64 %65
  %548 = ptrtoint ptr addrspace(1) %96 to i64
  %549 = add i64 %548, %544
  %550 = ptrtoint ptr addrspace(3) %542 to i32
  %551 = trunc i64 %549 to i32
  %552 = lshr i64 %549, 32
  %553 = trunc i64 %552 to i32
  %554 = or i32 %553, -2147483648
  %555 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %550, i64 1
  %556 = insertelement <4 x i32> %555, i32 %551, i64 2
  %557 = insertelement <4 x i32> %556, i32 %554, i64 3
  %558 = call i32 @llvm.smax.i32(i32 %64, i32 0)
  %559 = and i32 %558, 65535
  %560 = shl i32 %559, 16
  %561 = lshr i32 %558, 16
  %562 = and i32 %561, 65535
  %563 = or i32 %562, -65536
  %564 = trunc i64 %547 to i32
  %565 = lshr i64 %547, 32
  %566 = trunc i64 %565 to i32
  %567 = and i32 %566, 65535
  %568 = insertelement <8 x i32> <i32 2097152, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %560, i64 1
  %569 = insertelement <8 x i32> %568, i32 %563, i64 2
  %570 = insertelement <8 x i32> %569, i32 1081343, i64 3
  %571 = insertelement <8 x i32> %570, i32 8, i64 4
  %572 = insertelement <8 x i32> %571, i32 %564, i64 5
  %573 = insertelement <8 x i32> %572, i32 %567, i64 6
  %574 = insertelement <8 x i32> %573, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %557, <8 x i32> %574, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %575

575:                                              ; preds = %545, %541
  %576 = getelementptr i8, ptr addrspace(3) %485, i32 49536
  br i1 %129, label %577, label %597

577:                                              ; preds = %575
  %578 = icmp eq i64 %94, -2147483648
  %579 = select i1 %578, i64 8, i64 %94
  %580 = ptrtoint ptr addrspace(1) %97 to i64
  %581 = add i64 %580, %543
  %582 = ptrtoint ptr addrspace(3) %576 to i32
  %583 = trunc i64 %581 to i32
  %584 = lshr i64 %581, 32
  %585 = trunc i64 %584 to i32
  %586 = or i32 %585, -2147483648
  %587 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %582, i64 1
  %588 = insertelement <4 x i32> %587, i32 %583, i64 2
  %589 = insertelement <4 x i32> %588, i32 %586, i64 3
  %590 = trunc i64 %579 to i32
  %591 = lshr i64 %579, 32
  %592 = trunc i64 %591 to i32
  %593 = and i32 %592, 65535
  %594 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 557055, i32 1, i32 poison, i32 poison, i32 poison>, i32 %590, i64 5
  %595 = insertelement <8 x i32> %594, i32 %593, i64 6
  %596 = insertelement <8 x i32> %595, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %589, <8 x i32> %596, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %597

597:                                              ; preds = %577, %575
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %598 = add i32 %407, 18688
  %599 = add i32 %598, %409
  %600 = add i32 %599, %411
  %601 = sext i32 %600 to i64
  %602 = add i32 %397, %600
  %603 = inttoptr i32 %602 to ptr addrspace(3)
  %604 = load <4 x i32>, ptr addrspace(3) %603, align 16
  %605 = add i64 %601, 512
  %606 = trunc i64 %605 to i32
  %607 = add i32 %397, %606
  %608 = inttoptr i32 %607 to ptr addrspace(3)
  %609 = load <4 x i32>, ptr addrspace(3) %608, align 16
  %610 = add i64 %601, 1024
  %611 = trunc i64 %610 to i32
  %612 = add i32 %397, %611
  %613 = inttoptr i32 %612 to ptr addrspace(3)
  %614 = load <4 x i32>, ptr addrspace(3) %613, align 16
  %615 = add i64 %601, 1536
  %616 = trunc i64 %615 to i32
  %617 = add i32 %397, %616
  %618 = inttoptr i32 %617 to ptr addrspace(3)
  %619 = load <4 x i32>, ptr addrspace(3) %618, align 16
  %620 = shufflevector <4 x i32> %604, <4 x i32> %609, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %621 = shufflevector <4 x i32> %614, <4 x i32> %619, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %622 = shufflevector <8 x i32> %620, <8 x i32> %621, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %623 = add i32 %443, 49537
  %624 = add i32 %395, %623
  %625 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %624
  %626 = load i8, ptr addrspace(3) %625, align 1
  %627 = zext i8 %626 to i32
  %628 = call i32 @llvm.amdgcn.perm(i32 %627, i32 %627, i32 0)
  %629 = add i32 %450, 49424
  %630 = add i32 %395, %629
  %631 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %630
  %632 = load i8, ptr addrspace(3) %631, align 1
  %633 = zext i8 %632 to i32
  %634 = call i32 @llvm.amdgcn.perm(i32 %633, i32 %633, i32 0)
  %635 = add i32 %457, 128
  %636 = add i32 %635, %458
  %637 = sext i32 %636 to i64
  %638 = add i32 %397, %636
  %639 = inttoptr i32 %638 to ptr addrspace(3)
  %640 = load <4 x i32>, ptr addrspace(3) %639, align 16
  %641 = add i64 %637, 32
  %642 = trunc i64 %641 to i32
  %643 = add i32 %397, %642
  %644 = inttoptr i32 %643 to ptr addrspace(3)
  %645 = load <4 x i32>, ptr addrspace(3) %644, align 16
  %646 = add i64 %637, 64
  %647 = trunc i64 %646 to i32
  %648 = add i32 %397, %647
  %649 = inttoptr i32 %648 to ptr addrspace(3)
  %650 = load <4 x i32>, ptr addrspace(3) %649, align 16
  %651 = add i64 %637, 96
  %652 = trunc i64 %651 to i32
  %653 = add i32 %397, %652
  %654 = inttoptr i32 %653 to ptr addrspace(3)
  %655 = load <4 x i32>, ptr addrspace(3) %654, align 16
  %656 = shufflevector <4 x i32> %640, <4 x i32> %645, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %657 = shufflevector <4 x i32> %650, <4 x i32> %655, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %658 = shufflevector <8 x i32> %656, <8 x i32> %657, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %659 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %622, i32 0, <16 x i32> %658, i16 0, <8 x float> %482, i32 0, i32 0, i32 %628, i32 0, i32 0, i32 %634, i1 false, i1 false)
  %660 = add i32 %407, 20736
  %661 = add i32 %660, %409
  %662 = add i32 %661, %411
  %663 = sext i32 %662 to i64
  %664 = add i32 %397, %662
  %665 = inttoptr i32 %664 to ptr addrspace(3)
  %666 = load <4 x i32>, ptr addrspace(3) %665, align 16
  %667 = add i64 %663, 512
  %668 = trunc i64 %667 to i32
  %669 = add i32 %397, %668
  %670 = inttoptr i32 %669 to ptr addrspace(3)
  %671 = load <4 x i32>, ptr addrspace(3) %670, align 16
  %672 = add i64 %663, 1024
  %673 = trunc i64 %672 to i32
  %674 = add i32 %397, %673
  %675 = inttoptr i32 %674 to ptr addrspace(3)
  %676 = load <4 x i32>, ptr addrspace(3) %675, align 16
  %677 = add i64 %663, 1536
  %678 = trunc i64 %677 to i32
  %679 = add i32 %397, %678
  %680 = inttoptr i32 %679 to ptr addrspace(3)
  %681 = load <4 x i32>, ptr addrspace(3) %680, align 16
  %682 = shufflevector <4 x i32> %666, <4 x i32> %671, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %683 = shufflevector <4 x i32> %676, <4 x i32> %681, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %684 = shufflevector <8 x i32> %682, <8 x i32> %683, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %685 = add i32 %443, 49538
  %686 = add i32 %395, %685
  %687 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %686
  %688 = load i8, ptr addrspace(3) %687, align 1
  %689 = zext i8 %688 to i32
  %690 = call i32 @llvm.amdgcn.perm(i32 %689, i32 %689, i32 0)
  %691 = add i32 %450, 49440
  %692 = add i32 %395, %691
  %693 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %692
  %694 = load i8, ptr addrspace(3) %693, align 1
  %695 = zext i8 %694 to i32
  %696 = call i32 @llvm.amdgcn.perm(i32 %695, i32 %695, i32 0)
  %697 = add i32 %457, 256
  %698 = add i32 %697, %458
  %699 = sext i32 %698 to i64
  %700 = add i32 %397, %698
  %701 = inttoptr i32 %700 to ptr addrspace(3)
  %702 = load <4 x i32>, ptr addrspace(3) %701, align 16
  %703 = add i64 %699, 32
  %704 = trunc i64 %703 to i32
  %705 = add i32 %397, %704
  %706 = inttoptr i32 %705 to ptr addrspace(3)
  %707 = load <4 x i32>, ptr addrspace(3) %706, align 16
  %708 = add i64 %699, 64
  %709 = trunc i64 %708 to i32
  %710 = add i32 %397, %709
  %711 = inttoptr i32 %710 to ptr addrspace(3)
  %712 = load <4 x i32>, ptr addrspace(3) %711, align 16
  %713 = add i64 %699, 96
  %714 = trunc i64 %713 to i32
  %715 = add i32 %397, %714
  %716 = inttoptr i32 %715 to ptr addrspace(3)
  %717 = load <4 x i32>, ptr addrspace(3) %716, align 16
  %718 = shufflevector <4 x i32> %702, <4 x i32> %707, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %719 = shufflevector <4 x i32> %712, <4 x i32> %717, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %720 = shufflevector <8 x i32> %718, <8 x i32> %719, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %721 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %684, i32 0, <16 x i32> %720, i16 0, <8 x float> %659, i32 0, i32 0, i32 %690, i32 0, i32 0, i32 %696, i1 false, i1 false)
  %722 = add i32 %407, 22784
  %723 = add i32 %722, %409
  %724 = add i32 %723, %411
  %725 = sext i32 %724 to i64
  %726 = add i32 %397, %724
  %727 = inttoptr i32 %726 to ptr addrspace(3)
  %728 = load <4 x i32>, ptr addrspace(3) %727, align 16
  %729 = add i64 %725, 512
  %730 = trunc i64 %729 to i32
  %731 = add i32 %397, %730
  %732 = inttoptr i32 %731 to ptr addrspace(3)
  %733 = load <4 x i32>, ptr addrspace(3) %732, align 16
  %734 = add i64 %725, 1024
  %735 = trunc i64 %734 to i32
  %736 = add i32 %397, %735
  %737 = inttoptr i32 %736 to ptr addrspace(3)
  %738 = load <4 x i32>, ptr addrspace(3) %737, align 16
  %739 = add i64 %725, 1536
  %740 = trunc i64 %739 to i32
  %741 = add i32 %397, %740
  %742 = inttoptr i32 %741 to ptr addrspace(3)
  %743 = load <4 x i32>, ptr addrspace(3) %742, align 16
  %744 = shufflevector <4 x i32> %728, <4 x i32> %733, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %745 = shufflevector <4 x i32> %738, <4 x i32> %743, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %746 = shufflevector <8 x i32> %744, <8 x i32> %745, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %747 = add i32 %443, 49539
  %748 = add i32 %395, %747
  %749 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %748
  %750 = load i8, ptr addrspace(3) %749, align 1
  %751 = zext i8 %750 to i32
  %752 = call i32 @llvm.amdgcn.perm(i32 %751, i32 %751, i32 0)
  %753 = add i32 %450, 49456
  %754 = add i32 %395, %753
  %755 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %754
  %756 = load i8, ptr addrspace(3) %755, align 1
  %757 = zext i8 %756 to i32
  %758 = call i32 @llvm.amdgcn.perm(i32 %757, i32 %757, i32 0)
  %759 = add i32 %457, 384
  %760 = add i32 %759, %458
  %761 = sext i32 %760 to i64
  %762 = add i32 %397, %760
  %763 = inttoptr i32 %762 to ptr addrspace(3)
  %764 = load <4 x i32>, ptr addrspace(3) %763, align 16
  %765 = add i64 %761, 32
  %766 = trunc i64 %765 to i32
  %767 = add i32 %397, %766
  %768 = inttoptr i32 %767 to ptr addrspace(3)
  %769 = load <4 x i32>, ptr addrspace(3) %768, align 16
  %770 = add i64 %761, 64
  %771 = trunc i64 %770 to i32
  %772 = add i32 %397, %771
  %773 = inttoptr i32 %772 to ptr addrspace(3)
  %774 = load <4 x i32>, ptr addrspace(3) %773, align 16
  %775 = add i64 %761, 96
  %776 = trunc i64 %775 to i32
  %777 = add i32 %397, %776
  %778 = inttoptr i32 %777 to ptr addrspace(3)
  %779 = load <4 x i32>, ptr addrspace(3) %778, align 16
  %780 = shufflevector <4 x i32> %764, <4 x i32> %769, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %781 = shufflevector <4 x i32> %774, <4 x i32> %779, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %782 = shufflevector <8 x i32> %780, <8 x i32> %781, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %783 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %746, i32 0, <16 x i32> %782, i16 0, <8 x float> %721, i32 0, i32 0, i32 %752, i32 0, i32 0, i32 %758, i1 false, i1 false)
  %784 = add i32 %407, 24832
  %785 = add i32 %784, %409
  %786 = add i32 %785, %411
  %787 = sext i32 %786 to i64
  %788 = add i32 %397, %786
  %789 = inttoptr i32 %788 to ptr addrspace(3)
  %790 = load <4 x i32>, ptr addrspace(3) %789, align 16
  %791 = add i64 %787, 512
  %792 = trunc i64 %791 to i32
  %793 = add i32 %397, %792
  %794 = inttoptr i32 %793 to ptr addrspace(3)
  %795 = load <4 x i32>, ptr addrspace(3) %794, align 16
  %796 = add i64 %787, 1024
  %797 = trunc i64 %796 to i32
  %798 = add i32 %397, %797
  %799 = inttoptr i32 %798 to ptr addrspace(3)
  %800 = load <4 x i32>, ptr addrspace(3) %799, align 16
  %801 = add i64 %787, 1536
  %802 = trunc i64 %801 to i32
  %803 = add i32 %397, %802
  %804 = inttoptr i32 %803 to ptr addrspace(3)
  %805 = load <4 x i32>, ptr addrspace(3) %804, align 16
  %806 = shufflevector <4 x i32> %790, <4 x i32> %795, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %807 = shufflevector <4 x i32> %800, <4 x i32> %805, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %808 = shufflevector <8 x i32> %806, <8 x i32> %807, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %809 = add i32 %443, 49540
  %810 = add i32 %395, %809
  %811 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %810
  %812 = load i8, ptr addrspace(3) %811, align 1
  %813 = zext i8 %812 to i32
  %814 = call i32 @llvm.amdgcn.perm(i32 %813, i32 %813, i32 0)
  %815 = add i32 %450, 49472
  %816 = add i32 %395, %815
  %817 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %816
  %818 = load i8, ptr addrspace(3) %817, align 1
  %819 = zext i8 %818 to i32
  %820 = call i32 @llvm.amdgcn.perm(i32 %819, i32 %819, i32 0)
  %821 = add i32 %457, 512
  %822 = add i32 %821, %458
  %823 = sext i32 %822 to i64
  %824 = add i32 %397, %822
  %825 = inttoptr i32 %824 to ptr addrspace(3)
  %826 = load <4 x i32>, ptr addrspace(3) %825, align 16
  %827 = add i64 %823, 32
  %828 = trunc i64 %827 to i32
  %829 = add i32 %397, %828
  %830 = inttoptr i32 %829 to ptr addrspace(3)
  %831 = load <4 x i32>, ptr addrspace(3) %830, align 16
  %832 = add i64 %823, 64
  %833 = trunc i64 %832 to i32
  %834 = add i32 %397, %833
  %835 = inttoptr i32 %834 to ptr addrspace(3)
  %836 = load <4 x i32>, ptr addrspace(3) %835, align 16
  %837 = add i64 %823, 96
  %838 = trunc i64 %837 to i32
  %839 = add i32 %397, %838
  %840 = inttoptr i32 %839 to ptr addrspace(3)
  %841 = load <4 x i32>, ptr addrspace(3) %840, align 16
  %842 = shufflevector <4 x i32> %826, <4 x i32> %831, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %843 = shufflevector <4 x i32> %836, <4 x i32> %841, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %844 = shufflevector <8 x i32> %842, <8 x i32> %843, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %845 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %808, i32 0, <16 x i32> %844, i16 0, <8 x float> %783, i32 0, i32 0, i32 %814, i32 0, i32 0, i32 %820, i1 false, i1 false)
  %846 = add i32 %407, 26880
  %847 = add i32 %846, %409
  %848 = add i32 %847, %411
  %849 = sext i32 %848 to i64
  %850 = add i32 %397, %848
  %851 = inttoptr i32 %850 to ptr addrspace(3)
  %852 = load <4 x i32>, ptr addrspace(3) %851, align 16
  %853 = add i64 %849, 512
  %854 = trunc i64 %853 to i32
  %855 = add i32 %397, %854
  %856 = inttoptr i32 %855 to ptr addrspace(3)
  %857 = load <4 x i32>, ptr addrspace(3) %856, align 16
  %858 = add i64 %849, 1024
  %859 = trunc i64 %858 to i32
  %860 = add i32 %397, %859
  %861 = inttoptr i32 %860 to ptr addrspace(3)
  %862 = load <4 x i32>, ptr addrspace(3) %861, align 16
  %863 = add i64 %849, 1536
  %864 = trunc i64 %863 to i32
  %865 = add i32 %397, %864
  %866 = inttoptr i32 %865 to ptr addrspace(3)
  %867 = load <4 x i32>, ptr addrspace(3) %866, align 16
  %868 = shufflevector <4 x i32> %852, <4 x i32> %857, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %869 = shufflevector <4 x i32> %862, <4 x i32> %867, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %870 = shufflevector <8 x i32> %868, <8 x i32> %869, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %871 = add i32 %443, 49541
  %872 = add i32 %395, %871
  %873 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %872
  %874 = load i8, ptr addrspace(3) %873, align 1
  %875 = zext i8 %874 to i32
  %876 = call i32 @llvm.amdgcn.perm(i32 %875, i32 %875, i32 0)
  %877 = add i32 %450, 49488
  %878 = add i32 %395, %877
  %879 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %878
  %880 = load i8, ptr addrspace(3) %879, align 1
  %881 = zext i8 %880 to i32
  %882 = call i32 @llvm.amdgcn.perm(i32 %881, i32 %881, i32 0)
  %883 = add i32 %457, 640
  %884 = add i32 %883, %458
  %885 = sext i32 %884 to i64
  %886 = add i32 %397, %884
  %887 = inttoptr i32 %886 to ptr addrspace(3)
  %888 = load <4 x i32>, ptr addrspace(3) %887, align 16
  %889 = add i64 %885, 32
  %890 = trunc i64 %889 to i32
  %891 = add i32 %397, %890
  %892 = inttoptr i32 %891 to ptr addrspace(3)
  %893 = load <4 x i32>, ptr addrspace(3) %892, align 16
  %894 = add i64 %885, 64
  %895 = trunc i64 %894 to i32
  %896 = add i32 %397, %895
  %897 = inttoptr i32 %896 to ptr addrspace(3)
  %898 = load <4 x i32>, ptr addrspace(3) %897, align 16
  %899 = add i64 %885, 96
  %900 = trunc i64 %899 to i32
  %901 = add i32 %397, %900
  %902 = inttoptr i32 %901 to ptr addrspace(3)
  %903 = load <4 x i32>, ptr addrspace(3) %902, align 16
  %904 = shufflevector <4 x i32> %888, <4 x i32> %893, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %905 = shufflevector <4 x i32> %898, <4 x i32> %903, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %906 = shufflevector <8 x i32> %904, <8 x i32> %905, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %907 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %870, i32 0, <16 x i32> %906, i16 0, <8 x float> %845, i32 0, i32 0, i32 %876, i32 0, i32 0, i32 %882, i1 false, i1 false)
  %908 = add i32 %407, 28928
  %909 = add i32 %908, %409
  %910 = add i32 %909, %411
  %911 = sext i32 %910 to i64
  %912 = add i32 %397, %910
  %913 = inttoptr i32 %912 to ptr addrspace(3)
  %914 = load <4 x i32>, ptr addrspace(3) %913, align 16
  %915 = add i64 %911, 512
  %916 = trunc i64 %915 to i32
  %917 = add i32 %397, %916
  %918 = inttoptr i32 %917 to ptr addrspace(3)
  %919 = load <4 x i32>, ptr addrspace(3) %918, align 16
  %920 = add i64 %911, 1024
  %921 = trunc i64 %920 to i32
  %922 = add i32 %397, %921
  %923 = inttoptr i32 %922 to ptr addrspace(3)
  %924 = load <4 x i32>, ptr addrspace(3) %923, align 16
  %925 = add i64 %911, 1536
  %926 = trunc i64 %925 to i32
  %927 = add i32 %397, %926
  %928 = inttoptr i32 %927 to ptr addrspace(3)
  %929 = load <4 x i32>, ptr addrspace(3) %928, align 16
  %930 = shufflevector <4 x i32> %914, <4 x i32> %919, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %931 = shufflevector <4 x i32> %924, <4 x i32> %929, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %932 = shufflevector <8 x i32> %930, <8 x i32> %931, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %933 = add i32 %443, 49542
  %934 = add i32 %395, %933
  %935 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %934
  %936 = load i8, ptr addrspace(3) %935, align 1
  %937 = zext i8 %936 to i32
  %938 = call i32 @llvm.amdgcn.perm(i32 %937, i32 %937, i32 0)
  %939 = add i32 %450, 49504
  %940 = add i32 %395, %939
  %941 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %940
  %942 = load i8, ptr addrspace(3) %941, align 1
  %943 = zext i8 %942 to i32
  %944 = call i32 @llvm.amdgcn.perm(i32 %943, i32 %943, i32 0)
  %945 = add i32 %457, 768
  %946 = add i32 %945, %458
  %947 = sext i32 %946 to i64
  %948 = add i32 %397, %946
  %949 = inttoptr i32 %948 to ptr addrspace(3)
  %950 = load <4 x i32>, ptr addrspace(3) %949, align 16
  %951 = add i64 %947, 32
  %952 = trunc i64 %951 to i32
  %953 = add i32 %397, %952
  %954 = inttoptr i32 %953 to ptr addrspace(3)
  %955 = load <4 x i32>, ptr addrspace(3) %954, align 16
  %956 = add i64 %947, 64
  %957 = trunc i64 %956 to i32
  %958 = add i32 %397, %957
  %959 = inttoptr i32 %958 to ptr addrspace(3)
  %960 = load <4 x i32>, ptr addrspace(3) %959, align 16
  %961 = add i64 %947, 96
  %962 = trunc i64 %961 to i32
  %963 = add i32 %397, %962
  %964 = inttoptr i32 %963 to ptr addrspace(3)
  %965 = load <4 x i32>, ptr addrspace(3) %964, align 16
  %966 = shufflevector <4 x i32> %950, <4 x i32> %955, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %967 = shufflevector <4 x i32> %960, <4 x i32> %965, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %968 = shufflevector <8 x i32> %966, <8 x i32> %967, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %969 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %932, i32 0, <16 x i32> %968, i16 0, <8 x float> %907, i32 0, i32 0, i32 %938, i32 0, i32 0, i32 %944, i1 false, i1 false)
  %970 = add i32 %407, 30976
  %971 = add i32 %970, %409
  %972 = add i32 %971, %411
  %973 = sext i32 %972 to i64
  %974 = add i32 %397, %972
  %975 = inttoptr i32 %974 to ptr addrspace(3)
  %976 = load <4 x i32>, ptr addrspace(3) %975, align 16
  %977 = add i64 %973, 512
  %978 = trunc i64 %977 to i32
  %979 = add i32 %397, %978
  %980 = inttoptr i32 %979 to ptr addrspace(3)
  %981 = load <4 x i32>, ptr addrspace(3) %980, align 16
  %982 = add i64 %973, 1024
  %983 = trunc i64 %982 to i32
  %984 = add i32 %397, %983
  %985 = inttoptr i32 %984 to ptr addrspace(3)
  %986 = load <4 x i32>, ptr addrspace(3) %985, align 16
  %987 = add i64 %973, 1536
  %988 = trunc i64 %987 to i32
  %989 = add i32 %397, %988
  %990 = inttoptr i32 %989 to ptr addrspace(3)
  %991 = load <4 x i32>, ptr addrspace(3) %990, align 16
  %992 = shufflevector <4 x i32> %976, <4 x i32> %981, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %993 = shufflevector <4 x i32> %986, <4 x i32> %991, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %994 = shufflevector <8 x i32> %992, <8 x i32> %993, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %995 = add i32 %443, 49543
  %996 = add i32 %395, %995
  %997 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %996
  %998 = load i8, ptr addrspace(3) %997, align 1
  %999 = zext i8 %998 to i32
  %1000 = call i32 @llvm.amdgcn.perm(i32 %999, i32 %999, i32 0)
  %1001 = add i32 %450, 49520
  %1002 = add i32 %395, %1001
  %1003 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1002
  %1004 = load i8, ptr addrspace(3) %1003, align 1
  %1005 = zext i8 %1004 to i32
  %1006 = call i32 @llvm.amdgcn.perm(i32 %1005, i32 %1005, i32 0)
  %1007 = add i32 %457, 896
  %1008 = add i32 %1007, %458
  %1009 = sext i32 %1008 to i64
  %1010 = add i32 %397, %1008
  %1011 = inttoptr i32 %1010 to ptr addrspace(3)
  %1012 = load <4 x i32>, ptr addrspace(3) %1011, align 16
  %1013 = add i64 %1009, 32
  %1014 = trunc i64 %1013 to i32
  %1015 = add i32 %397, %1014
  %1016 = inttoptr i32 %1015 to ptr addrspace(3)
  %1017 = load <4 x i32>, ptr addrspace(3) %1016, align 16
  %1018 = add i64 %1009, 64
  %1019 = trunc i64 %1018 to i32
  %1020 = add i32 %397, %1019
  %1021 = inttoptr i32 %1020 to ptr addrspace(3)
  %1022 = load <4 x i32>, ptr addrspace(3) %1021, align 16
  %1023 = add i64 %1009, 96
  %1024 = trunc i64 %1023 to i32
  %1025 = add i32 %397, %1024
  %1026 = inttoptr i32 %1025 to ptr addrspace(3)
  %1027 = load <4 x i32>, ptr addrspace(3) %1026, align 16
  %1028 = shufflevector <4 x i32> %1012, <4 x i32> %1017, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1029 = shufflevector <4 x i32> %1022, <4 x i32> %1027, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1030 = shufflevector <8 x i32> %1028, <8 x i32> %1029, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1031 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %994, i32 0, <16 x i32> %1030, i16 0, <8 x float> %969, i32 0, i32 0, i32 %1000, i32 0, i32 0, i32 %1006, i1 false, i1 false)
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
  %1032 = add i32 %390, 1
  br label %389

1033:                                             ; preds = %389
  %1034 = srem i32 %388, 4
  %1035 = mul i32 %1034, 50176
  %1036 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1035
  %1037 = ptrtoint ptr addrspace(3) %1036 to i32
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 2)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %1038 = sdiv i32 %99, 16
  %1039 = mul i32 %1038, 16
  %1040 = icmp ne i32 %99, %1039
  %1041 = icmp slt i32 %99, 0
  %1042 = icmp ne i1 %1041, false
  %1043 = and i1 %1040, %1042
  %1044 = add i32 %1038, -1
  %1045 = select i1 %1043, i32 %1044, i32 %1038
  %1046 = mul i32 %1045, 16384
  %1047 = add i32 %1046, 16640
  %1048 = mul i32 %50, 256
  %1049 = add i32 %1047, %1048
  %1050 = mul i32 %42, 16
  %1051 = add i32 %1049, %1050
  %1052 = sext i32 %1051 to i64
  %1053 = add i32 %1037, %1051
  %1054 = inttoptr i32 %1053 to ptr addrspace(3)
  %1055 = load <4 x i32>, ptr addrspace(3) %1054, align 16
  %1056 = add i64 %1052, 512
  %1057 = trunc i64 %1056 to i32
  %1058 = add i32 %1037, %1057
  %1059 = inttoptr i32 %1058 to ptr addrspace(3)
  %1060 = load <4 x i32>, ptr addrspace(3) %1059, align 16
  %1061 = add i64 %1052, 1024
  %1062 = trunc i64 %1061 to i32
  %1063 = add i32 %1037, %1062
  %1064 = inttoptr i32 %1063 to ptr addrspace(3)
  %1065 = load <4 x i32>, ptr addrspace(3) %1064, align 16
  %1066 = add i64 %1052, 1536
  %1067 = trunc i64 %1066 to i32
  %1068 = add i32 %1037, %1067
  %1069 = inttoptr i32 %1068 to ptr addrspace(3)
  %1070 = load <4 x i32>, ptr addrspace(3) %1069, align 16
  %1071 = shufflevector <4 x i32> %1055, <4 x i32> %1060, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1072 = shufflevector <4 x i32> %1065, <4 x i32> %1070, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1073 = shufflevector <8 x i32> %1071, <8 x i32> %1072, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1074 = sdiv i32 %99, 128
  %1075 = mul i32 %1074, 128
  %1076 = icmp ne i32 %99, %1075
  %1077 = icmp slt i32 %99, 0
  %1078 = icmp ne i1 %1077, false
  %1079 = and i1 %1076, %1078
  %1080 = add i32 %1074, -1
  %1081 = select i1 %1079, i32 %1080, i32 %1074
  %1082 = mul i32 %1081, 8
  %1083 = add i32 %1082, 49536
  %1084 = add i32 %1035, %1083
  %1085 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1084
  %1086 = load i8, ptr addrspace(3) %1085, align 1
  %1087 = zext i8 %1086 to i32
  %1088 = call i32 @llvm.amdgcn.perm(i32 %1087, i32 %1087, i32 0)
  %1089 = add i32 %98, %42
  %1090 = add i32 %1089, 49408
  %1091 = add i32 %1035, %1090
  %1092 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1091
  %1093 = load i8, ptr addrspace(3) %1092, align 1
  %1094 = zext i8 %1093 to i32
  %1095 = call i32 @llvm.amdgcn.perm(i32 %1094, i32 %1094, i32 0)
  %1096 = mul i32 %1089, 1040
  %1097 = mul i32 %50, 16
  %1098 = add i32 %1096, %1097
  %1099 = sext i32 %1098 to i64
  %1100 = add i32 %1037, %1098
  %1101 = inttoptr i32 %1100 to ptr addrspace(3)
  %1102 = load <4 x i32>, ptr addrspace(3) %1101, align 16
  %1103 = add i64 %1099, 32
  %1104 = trunc i64 %1103 to i32
  %1105 = add i32 %1037, %1104
  %1106 = inttoptr i32 %1105 to ptr addrspace(3)
  %1107 = load <4 x i32>, ptr addrspace(3) %1106, align 16
  %1108 = add i64 %1099, 64
  %1109 = trunc i64 %1108 to i32
  %1110 = add i32 %1037, %1109
  %1111 = inttoptr i32 %1110 to ptr addrspace(3)
  %1112 = load <4 x i32>, ptr addrspace(3) %1111, align 16
  %1113 = add i64 %1099, 96
  %1114 = trunc i64 %1113 to i32
  %1115 = add i32 %1037, %1114
  %1116 = inttoptr i32 %1115 to ptr addrspace(3)
  %1117 = load <4 x i32>, ptr addrspace(3) %1116, align 16
  %1118 = shufflevector <4 x i32> %1102, <4 x i32> %1107, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1119 = shufflevector <4 x i32> %1112, <4 x i32> %1117, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1120 = shufflevector <8 x i32> %1118, <8 x i32> %1119, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1121 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1073, i32 0, <16 x i32> %1120, i16 0, <8 x float> %391, i32 0, i32 0, i32 %1088, i32 0, i32 0, i32 %1095, i1 false, i1 false)
  %1122 = add i32 %1046, 18688
  %1123 = add i32 %1122, %1048
  %1124 = add i32 %1123, %1050
  %1125 = sext i32 %1124 to i64
  %1126 = add i32 %1037, %1124
  %1127 = inttoptr i32 %1126 to ptr addrspace(3)
  %1128 = load <4 x i32>, ptr addrspace(3) %1127, align 16
  %1129 = add i64 %1125, 512
  %1130 = trunc i64 %1129 to i32
  %1131 = add i32 %1037, %1130
  %1132 = inttoptr i32 %1131 to ptr addrspace(3)
  %1133 = load <4 x i32>, ptr addrspace(3) %1132, align 16
  %1134 = add i64 %1125, 1024
  %1135 = trunc i64 %1134 to i32
  %1136 = add i32 %1037, %1135
  %1137 = inttoptr i32 %1136 to ptr addrspace(3)
  %1138 = load <4 x i32>, ptr addrspace(3) %1137, align 16
  %1139 = add i64 %1125, 1536
  %1140 = trunc i64 %1139 to i32
  %1141 = add i32 %1037, %1140
  %1142 = inttoptr i32 %1141 to ptr addrspace(3)
  %1143 = load <4 x i32>, ptr addrspace(3) %1142, align 16
  %1144 = shufflevector <4 x i32> %1128, <4 x i32> %1133, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1145 = shufflevector <4 x i32> %1138, <4 x i32> %1143, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1146 = shufflevector <8 x i32> %1144, <8 x i32> %1145, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1147 = add i32 %1082, 49537
  %1148 = add i32 %1035, %1147
  %1149 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1148
  %1150 = load i8, ptr addrspace(3) %1149, align 1
  %1151 = zext i8 %1150 to i32
  %1152 = call i32 @llvm.amdgcn.perm(i32 %1151, i32 %1151, i32 0)
  %1153 = add i32 %1089, 49424
  %1154 = add i32 %1035, %1153
  %1155 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1154
  %1156 = load i8, ptr addrspace(3) %1155, align 1
  %1157 = zext i8 %1156 to i32
  %1158 = call i32 @llvm.amdgcn.perm(i32 %1157, i32 %1157, i32 0)
  %1159 = add i32 %1096, 128
  %1160 = add i32 %1159, %1097
  %1161 = sext i32 %1160 to i64
  %1162 = add i32 %1037, %1160
  %1163 = inttoptr i32 %1162 to ptr addrspace(3)
  %1164 = load <4 x i32>, ptr addrspace(3) %1163, align 16
  %1165 = add i64 %1161, 32
  %1166 = trunc i64 %1165 to i32
  %1167 = add i32 %1037, %1166
  %1168 = inttoptr i32 %1167 to ptr addrspace(3)
  %1169 = load <4 x i32>, ptr addrspace(3) %1168, align 16
  %1170 = add i64 %1161, 64
  %1171 = trunc i64 %1170 to i32
  %1172 = add i32 %1037, %1171
  %1173 = inttoptr i32 %1172 to ptr addrspace(3)
  %1174 = load <4 x i32>, ptr addrspace(3) %1173, align 16
  %1175 = add i64 %1161, 96
  %1176 = trunc i64 %1175 to i32
  %1177 = add i32 %1037, %1176
  %1178 = inttoptr i32 %1177 to ptr addrspace(3)
  %1179 = load <4 x i32>, ptr addrspace(3) %1178, align 16
  %1180 = shufflevector <4 x i32> %1164, <4 x i32> %1169, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1181 = shufflevector <4 x i32> %1174, <4 x i32> %1179, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1182 = shufflevector <8 x i32> %1180, <8 x i32> %1181, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1183 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1146, i32 0, <16 x i32> %1182, i16 0, <8 x float> %1121, i32 0, i32 0, i32 %1152, i32 0, i32 0, i32 %1158, i1 false, i1 false)
  %1184 = add i32 %1046, 20736
  %1185 = add i32 %1184, %1048
  %1186 = add i32 %1185, %1050
  %1187 = sext i32 %1186 to i64
  %1188 = add i32 %1037, %1186
  %1189 = inttoptr i32 %1188 to ptr addrspace(3)
  %1190 = load <4 x i32>, ptr addrspace(3) %1189, align 16
  %1191 = add i64 %1187, 512
  %1192 = trunc i64 %1191 to i32
  %1193 = add i32 %1037, %1192
  %1194 = inttoptr i32 %1193 to ptr addrspace(3)
  %1195 = load <4 x i32>, ptr addrspace(3) %1194, align 16
  %1196 = add i64 %1187, 1024
  %1197 = trunc i64 %1196 to i32
  %1198 = add i32 %1037, %1197
  %1199 = inttoptr i32 %1198 to ptr addrspace(3)
  %1200 = load <4 x i32>, ptr addrspace(3) %1199, align 16
  %1201 = add i64 %1187, 1536
  %1202 = trunc i64 %1201 to i32
  %1203 = add i32 %1037, %1202
  %1204 = inttoptr i32 %1203 to ptr addrspace(3)
  %1205 = load <4 x i32>, ptr addrspace(3) %1204, align 16
  %1206 = shufflevector <4 x i32> %1190, <4 x i32> %1195, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1207 = shufflevector <4 x i32> %1200, <4 x i32> %1205, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1208 = shufflevector <8 x i32> %1206, <8 x i32> %1207, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1209 = add i32 %1082, 49538
  %1210 = add i32 %1035, %1209
  %1211 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1210
  %1212 = load i8, ptr addrspace(3) %1211, align 1
  %1213 = zext i8 %1212 to i32
  %1214 = call i32 @llvm.amdgcn.perm(i32 %1213, i32 %1213, i32 0)
  %1215 = add i32 %1089, 49440
  %1216 = add i32 %1035, %1215
  %1217 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1216
  %1218 = load i8, ptr addrspace(3) %1217, align 1
  %1219 = zext i8 %1218 to i32
  %1220 = call i32 @llvm.amdgcn.perm(i32 %1219, i32 %1219, i32 0)
  %1221 = add i32 %1096, 256
  %1222 = add i32 %1221, %1097
  %1223 = sext i32 %1222 to i64
  %1224 = add i32 %1037, %1222
  %1225 = inttoptr i32 %1224 to ptr addrspace(3)
  %1226 = load <4 x i32>, ptr addrspace(3) %1225, align 16
  %1227 = add i64 %1223, 32
  %1228 = trunc i64 %1227 to i32
  %1229 = add i32 %1037, %1228
  %1230 = inttoptr i32 %1229 to ptr addrspace(3)
  %1231 = load <4 x i32>, ptr addrspace(3) %1230, align 16
  %1232 = add i64 %1223, 64
  %1233 = trunc i64 %1232 to i32
  %1234 = add i32 %1037, %1233
  %1235 = inttoptr i32 %1234 to ptr addrspace(3)
  %1236 = load <4 x i32>, ptr addrspace(3) %1235, align 16
  %1237 = add i64 %1223, 96
  %1238 = trunc i64 %1237 to i32
  %1239 = add i32 %1037, %1238
  %1240 = inttoptr i32 %1239 to ptr addrspace(3)
  %1241 = load <4 x i32>, ptr addrspace(3) %1240, align 16
  %1242 = shufflevector <4 x i32> %1226, <4 x i32> %1231, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1243 = shufflevector <4 x i32> %1236, <4 x i32> %1241, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1244 = shufflevector <8 x i32> %1242, <8 x i32> %1243, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1245 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1208, i32 0, <16 x i32> %1244, i16 0, <8 x float> %1183, i32 0, i32 0, i32 %1214, i32 0, i32 0, i32 %1220, i1 false, i1 false)
  %1246 = add i32 %1046, 22784
  %1247 = add i32 %1246, %1048
  %1248 = add i32 %1247, %1050
  %1249 = sext i32 %1248 to i64
  %1250 = add i32 %1037, %1248
  %1251 = inttoptr i32 %1250 to ptr addrspace(3)
  %1252 = load <4 x i32>, ptr addrspace(3) %1251, align 16
  %1253 = add i64 %1249, 512
  %1254 = trunc i64 %1253 to i32
  %1255 = add i32 %1037, %1254
  %1256 = inttoptr i32 %1255 to ptr addrspace(3)
  %1257 = load <4 x i32>, ptr addrspace(3) %1256, align 16
  %1258 = add i64 %1249, 1024
  %1259 = trunc i64 %1258 to i32
  %1260 = add i32 %1037, %1259
  %1261 = inttoptr i32 %1260 to ptr addrspace(3)
  %1262 = load <4 x i32>, ptr addrspace(3) %1261, align 16
  %1263 = add i64 %1249, 1536
  %1264 = trunc i64 %1263 to i32
  %1265 = add i32 %1037, %1264
  %1266 = inttoptr i32 %1265 to ptr addrspace(3)
  %1267 = load <4 x i32>, ptr addrspace(3) %1266, align 16
  %1268 = shufflevector <4 x i32> %1252, <4 x i32> %1257, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1269 = shufflevector <4 x i32> %1262, <4 x i32> %1267, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1270 = shufflevector <8 x i32> %1268, <8 x i32> %1269, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1271 = add i32 %1082, 49539
  %1272 = add i32 %1035, %1271
  %1273 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1272
  %1274 = load i8, ptr addrspace(3) %1273, align 1
  %1275 = zext i8 %1274 to i32
  %1276 = call i32 @llvm.amdgcn.perm(i32 %1275, i32 %1275, i32 0)
  %1277 = add i32 %1089, 49456
  %1278 = add i32 %1035, %1277
  %1279 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1278
  %1280 = load i8, ptr addrspace(3) %1279, align 1
  %1281 = zext i8 %1280 to i32
  %1282 = call i32 @llvm.amdgcn.perm(i32 %1281, i32 %1281, i32 0)
  %1283 = add i32 %1096, 384
  %1284 = add i32 %1283, %1097
  %1285 = sext i32 %1284 to i64
  %1286 = add i32 %1037, %1284
  %1287 = inttoptr i32 %1286 to ptr addrspace(3)
  %1288 = load <4 x i32>, ptr addrspace(3) %1287, align 16
  %1289 = add i64 %1285, 32
  %1290 = trunc i64 %1289 to i32
  %1291 = add i32 %1037, %1290
  %1292 = inttoptr i32 %1291 to ptr addrspace(3)
  %1293 = load <4 x i32>, ptr addrspace(3) %1292, align 16
  %1294 = add i64 %1285, 64
  %1295 = trunc i64 %1294 to i32
  %1296 = add i32 %1037, %1295
  %1297 = inttoptr i32 %1296 to ptr addrspace(3)
  %1298 = load <4 x i32>, ptr addrspace(3) %1297, align 16
  %1299 = add i64 %1285, 96
  %1300 = trunc i64 %1299 to i32
  %1301 = add i32 %1037, %1300
  %1302 = inttoptr i32 %1301 to ptr addrspace(3)
  %1303 = load <4 x i32>, ptr addrspace(3) %1302, align 16
  %1304 = shufflevector <4 x i32> %1288, <4 x i32> %1293, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1305 = shufflevector <4 x i32> %1298, <4 x i32> %1303, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1306 = shufflevector <8 x i32> %1304, <8 x i32> %1305, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1307 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1270, i32 0, <16 x i32> %1306, i16 0, <8 x float> %1245, i32 0, i32 0, i32 %1276, i32 0, i32 0, i32 %1282, i1 false, i1 false)
  %1308 = add i32 %1046, 24832
  %1309 = add i32 %1308, %1048
  %1310 = add i32 %1309, %1050
  %1311 = sext i32 %1310 to i64
  %1312 = add i32 %1037, %1310
  %1313 = inttoptr i32 %1312 to ptr addrspace(3)
  %1314 = load <4 x i32>, ptr addrspace(3) %1313, align 16
  %1315 = add i64 %1311, 512
  %1316 = trunc i64 %1315 to i32
  %1317 = add i32 %1037, %1316
  %1318 = inttoptr i32 %1317 to ptr addrspace(3)
  %1319 = load <4 x i32>, ptr addrspace(3) %1318, align 16
  %1320 = add i64 %1311, 1024
  %1321 = trunc i64 %1320 to i32
  %1322 = add i32 %1037, %1321
  %1323 = inttoptr i32 %1322 to ptr addrspace(3)
  %1324 = load <4 x i32>, ptr addrspace(3) %1323, align 16
  %1325 = add i64 %1311, 1536
  %1326 = trunc i64 %1325 to i32
  %1327 = add i32 %1037, %1326
  %1328 = inttoptr i32 %1327 to ptr addrspace(3)
  %1329 = load <4 x i32>, ptr addrspace(3) %1328, align 16
  %1330 = shufflevector <4 x i32> %1314, <4 x i32> %1319, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1331 = shufflevector <4 x i32> %1324, <4 x i32> %1329, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1332 = shufflevector <8 x i32> %1330, <8 x i32> %1331, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1333 = add i32 %1082, 49540
  %1334 = add i32 %1035, %1333
  %1335 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1334
  %1336 = load i8, ptr addrspace(3) %1335, align 1
  %1337 = zext i8 %1336 to i32
  %1338 = call i32 @llvm.amdgcn.perm(i32 %1337, i32 %1337, i32 0)
  %1339 = add i32 %1089, 49472
  %1340 = add i32 %1035, %1339
  %1341 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1340
  %1342 = load i8, ptr addrspace(3) %1341, align 1
  %1343 = zext i8 %1342 to i32
  %1344 = call i32 @llvm.amdgcn.perm(i32 %1343, i32 %1343, i32 0)
  %1345 = add i32 %1096, 512
  %1346 = add i32 %1345, %1097
  %1347 = sext i32 %1346 to i64
  %1348 = add i32 %1037, %1346
  %1349 = inttoptr i32 %1348 to ptr addrspace(3)
  %1350 = load <4 x i32>, ptr addrspace(3) %1349, align 16
  %1351 = add i64 %1347, 32
  %1352 = trunc i64 %1351 to i32
  %1353 = add i32 %1037, %1352
  %1354 = inttoptr i32 %1353 to ptr addrspace(3)
  %1355 = load <4 x i32>, ptr addrspace(3) %1354, align 16
  %1356 = add i64 %1347, 64
  %1357 = trunc i64 %1356 to i32
  %1358 = add i32 %1037, %1357
  %1359 = inttoptr i32 %1358 to ptr addrspace(3)
  %1360 = load <4 x i32>, ptr addrspace(3) %1359, align 16
  %1361 = add i64 %1347, 96
  %1362 = trunc i64 %1361 to i32
  %1363 = add i32 %1037, %1362
  %1364 = inttoptr i32 %1363 to ptr addrspace(3)
  %1365 = load <4 x i32>, ptr addrspace(3) %1364, align 16
  %1366 = shufflevector <4 x i32> %1350, <4 x i32> %1355, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1367 = shufflevector <4 x i32> %1360, <4 x i32> %1365, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1368 = shufflevector <8 x i32> %1366, <8 x i32> %1367, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1369 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1332, i32 0, <16 x i32> %1368, i16 0, <8 x float> %1307, i32 0, i32 0, i32 %1338, i32 0, i32 0, i32 %1344, i1 false, i1 false)
  %1370 = add i32 %1046, 26880
  %1371 = add i32 %1370, %1048
  %1372 = add i32 %1371, %1050
  %1373 = sext i32 %1372 to i64
  %1374 = add i32 %1037, %1372
  %1375 = inttoptr i32 %1374 to ptr addrspace(3)
  %1376 = load <4 x i32>, ptr addrspace(3) %1375, align 16
  %1377 = add i64 %1373, 512
  %1378 = trunc i64 %1377 to i32
  %1379 = add i32 %1037, %1378
  %1380 = inttoptr i32 %1379 to ptr addrspace(3)
  %1381 = load <4 x i32>, ptr addrspace(3) %1380, align 16
  %1382 = add i64 %1373, 1024
  %1383 = trunc i64 %1382 to i32
  %1384 = add i32 %1037, %1383
  %1385 = inttoptr i32 %1384 to ptr addrspace(3)
  %1386 = load <4 x i32>, ptr addrspace(3) %1385, align 16
  %1387 = add i64 %1373, 1536
  %1388 = trunc i64 %1387 to i32
  %1389 = add i32 %1037, %1388
  %1390 = inttoptr i32 %1389 to ptr addrspace(3)
  %1391 = load <4 x i32>, ptr addrspace(3) %1390, align 16
  %1392 = shufflevector <4 x i32> %1376, <4 x i32> %1381, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1393 = shufflevector <4 x i32> %1386, <4 x i32> %1391, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1394 = shufflevector <8 x i32> %1392, <8 x i32> %1393, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1395 = add i32 %1082, 49541
  %1396 = add i32 %1035, %1395
  %1397 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1396
  %1398 = load i8, ptr addrspace(3) %1397, align 1
  %1399 = zext i8 %1398 to i32
  %1400 = call i32 @llvm.amdgcn.perm(i32 %1399, i32 %1399, i32 0)
  %1401 = add i32 %1089, 49488
  %1402 = add i32 %1035, %1401
  %1403 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1402
  %1404 = load i8, ptr addrspace(3) %1403, align 1
  %1405 = zext i8 %1404 to i32
  %1406 = call i32 @llvm.amdgcn.perm(i32 %1405, i32 %1405, i32 0)
  %1407 = add i32 %1096, 640
  %1408 = add i32 %1407, %1097
  %1409 = sext i32 %1408 to i64
  %1410 = add i32 %1037, %1408
  %1411 = inttoptr i32 %1410 to ptr addrspace(3)
  %1412 = load <4 x i32>, ptr addrspace(3) %1411, align 16
  %1413 = add i64 %1409, 32
  %1414 = trunc i64 %1413 to i32
  %1415 = add i32 %1037, %1414
  %1416 = inttoptr i32 %1415 to ptr addrspace(3)
  %1417 = load <4 x i32>, ptr addrspace(3) %1416, align 16
  %1418 = add i64 %1409, 64
  %1419 = trunc i64 %1418 to i32
  %1420 = add i32 %1037, %1419
  %1421 = inttoptr i32 %1420 to ptr addrspace(3)
  %1422 = load <4 x i32>, ptr addrspace(3) %1421, align 16
  %1423 = add i64 %1409, 96
  %1424 = trunc i64 %1423 to i32
  %1425 = add i32 %1037, %1424
  %1426 = inttoptr i32 %1425 to ptr addrspace(3)
  %1427 = load <4 x i32>, ptr addrspace(3) %1426, align 16
  %1428 = shufflevector <4 x i32> %1412, <4 x i32> %1417, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1429 = shufflevector <4 x i32> %1422, <4 x i32> %1427, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1430 = shufflevector <8 x i32> %1428, <8 x i32> %1429, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1431 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1394, i32 0, <16 x i32> %1430, i16 0, <8 x float> %1369, i32 0, i32 0, i32 %1400, i32 0, i32 0, i32 %1406, i1 false, i1 false)
  %1432 = add i32 %1046, 28928
  %1433 = add i32 %1432, %1048
  %1434 = add i32 %1433, %1050
  %1435 = sext i32 %1434 to i64
  %1436 = add i32 %1037, %1434
  %1437 = inttoptr i32 %1436 to ptr addrspace(3)
  %1438 = load <4 x i32>, ptr addrspace(3) %1437, align 16
  %1439 = add i64 %1435, 512
  %1440 = trunc i64 %1439 to i32
  %1441 = add i32 %1037, %1440
  %1442 = inttoptr i32 %1441 to ptr addrspace(3)
  %1443 = load <4 x i32>, ptr addrspace(3) %1442, align 16
  %1444 = add i64 %1435, 1024
  %1445 = trunc i64 %1444 to i32
  %1446 = add i32 %1037, %1445
  %1447 = inttoptr i32 %1446 to ptr addrspace(3)
  %1448 = load <4 x i32>, ptr addrspace(3) %1447, align 16
  %1449 = add i64 %1435, 1536
  %1450 = trunc i64 %1449 to i32
  %1451 = add i32 %1037, %1450
  %1452 = inttoptr i32 %1451 to ptr addrspace(3)
  %1453 = load <4 x i32>, ptr addrspace(3) %1452, align 16
  %1454 = shufflevector <4 x i32> %1438, <4 x i32> %1443, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1455 = shufflevector <4 x i32> %1448, <4 x i32> %1453, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1456 = shufflevector <8 x i32> %1454, <8 x i32> %1455, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1457 = add i32 %1082, 49542
  %1458 = add i32 %1035, %1457
  %1459 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1458
  %1460 = load i8, ptr addrspace(3) %1459, align 1
  %1461 = zext i8 %1460 to i32
  %1462 = call i32 @llvm.amdgcn.perm(i32 %1461, i32 %1461, i32 0)
  %1463 = add i32 %1089, 49504
  %1464 = add i32 %1035, %1463
  %1465 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1464
  %1466 = load i8, ptr addrspace(3) %1465, align 1
  %1467 = zext i8 %1466 to i32
  %1468 = call i32 @llvm.amdgcn.perm(i32 %1467, i32 %1467, i32 0)
  %1469 = add i32 %1096, 768
  %1470 = add i32 %1469, %1097
  %1471 = sext i32 %1470 to i64
  %1472 = add i32 %1037, %1470
  %1473 = inttoptr i32 %1472 to ptr addrspace(3)
  %1474 = load <4 x i32>, ptr addrspace(3) %1473, align 16
  %1475 = add i64 %1471, 32
  %1476 = trunc i64 %1475 to i32
  %1477 = add i32 %1037, %1476
  %1478 = inttoptr i32 %1477 to ptr addrspace(3)
  %1479 = load <4 x i32>, ptr addrspace(3) %1478, align 16
  %1480 = add i64 %1471, 64
  %1481 = trunc i64 %1480 to i32
  %1482 = add i32 %1037, %1481
  %1483 = inttoptr i32 %1482 to ptr addrspace(3)
  %1484 = load <4 x i32>, ptr addrspace(3) %1483, align 16
  %1485 = add i64 %1471, 96
  %1486 = trunc i64 %1485 to i32
  %1487 = add i32 %1037, %1486
  %1488 = inttoptr i32 %1487 to ptr addrspace(3)
  %1489 = load <4 x i32>, ptr addrspace(3) %1488, align 16
  %1490 = shufflevector <4 x i32> %1474, <4 x i32> %1479, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1491 = shufflevector <4 x i32> %1484, <4 x i32> %1489, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1492 = shufflevector <8 x i32> %1490, <8 x i32> %1491, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1493 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1456, i32 0, <16 x i32> %1492, i16 0, <8 x float> %1431, i32 0, i32 0, i32 %1462, i32 0, i32 0, i32 %1468, i1 false, i1 false)
  %1494 = add i32 %1046, 30976
  %1495 = add i32 %1494, %1048
  %1496 = add i32 %1495, %1050
  %1497 = sext i32 %1496 to i64
  %1498 = add i32 %1037, %1496
  %1499 = inttoptr i32 %1498 to ptr addrspace(3)
  %1500 = load <4 x i32>, ptr addrspace(3) %1499, align 16
  %1501 = add i64 %1497, 512
  %1502 = trunc i64 %1501 to i32
  %1503 = add i32 %1037, %1502
  %1504 = inttoptr i32 %1503 to ptr addrspace(3)
  %1505 = load <4 x i32>, ptr addrspace(3) %1504, align 16
  %1506 = add i64 %1497, 1024
  %1507 = trunc i64 %1506 to i32
  %1508 = add i32 %1037, %1507
  %1509 = inttoptr i32 %1508 to ptr addrspace(3)
  %1510 = load <4 x i32>, ptr addrspace(3) %1509, align 16
  %1511 = add i64 %1497, 1536
  %1512 = trunc i64 %1511 to i32
  %1513 = add i32 %1037, %1512
  %1514 = inttoptr i32 %1513 to ptr addrspace(3)
  %1515 = load <4 x i32>, ptr addrspace(3) %1514, align 16
  %1516 = shufflevector <4 x i32> %1500, <4 x i32> %1505, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1517 = shufflevector <4 x i32> %1510, <4 x i32> %1515, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1518 = shufflevector <8 x i32> %1516, <8 x i32> %1517, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1519 = add i32 %1082, 49543
  %1520 = add i32 %1035, %1519
  %1521 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1520
  %1522 = load i8, ptr addrspace(3) %1521, align 1
  %1523 = zext i8 %1522 to i32
  %1524 = call i32 @llvm.amdgcn.perm(i32 %1523, i32 %1523, i32 0)
  %1525 = add i32 %1089, 49520
  %1526 = add i32 %1035, %1525
  %1527 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1526
  %1528 = load i8, ptr addrspace(3) %1527, align 1
  %1529 = zext i8 %1528 to i32
  %1530 = call i32 @llvm.amdgcn.perm(i32 %1529, i32 %1529, i32 0)
  %1531 = add i32 %1096, 896
  %1532 = add i32 %1531, %1097
  %1533 = sext i32 %1532 to i64
  %1534 = add i32 %1037, %1532
  %1535 = inttoptr i32 %1534 to ptr addrspace(3)
  %1536 = load <4 x i32>, ptr addrspace(3) %1535, align 16
  %1537 = add i64 %1533, 32
  %1538 = trunc i64 %1537 to i32
  %1539 = add i32 %1037, %1538
  %1540 = inttoptr i32 %1539 to ptr addrspace(3)
  %1541 = load <4 x i32>, ptr addrspace(3) %1540, align 16
  %1542 = add i64 %1533, 64
  %1543 = trunc i64 %1542 to i32
  %1544 = add i32 %1037, %1543
  %1545 = inttoptr i32 %1544 to ptr addrspace(3)
  %1546 = load <4 x i32>, ptr addrspace(3) %1545, align 16
  %1547 = add i64 %1533, 96
  %1548 = trunc i64 %1547 to i32
  %1549 = add i32 %1037, %1548
  %1550 = inttoptr i32 %1549 to ptr addrspace(3)
  %1551 = load <4 x i32>, ptr addrspace(3) %1550, align 16
  %1552 = shufflevector <4 x i32> %1536, <4 x i32> %1541, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1553 = shufflevector <4 x i32> %1546, <4 x i32> %1551, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1554 = shufflevector <8 x i32> %1552, <8 x i32> %1553, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1555 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1518, i32 0, <16 x i32> %1554, i16 0, <8 x float> %1493, i32 0, i32 0, i32 %1524, i32 0, i32 0, i32 %1530, i1 false, i1 false)
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
  %1556 = add i32 %19, -2
  %1557 = srem i32 %1556, 4
  %1558 = mul i32 %1557, 50176
  %1559 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1558
  %1560 = ptrtoint ptr addrspace(3) %1559 to i32
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 1)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %1561 = add i32 %1560, %1051
  %1562 = inttoptr i32 %1561 to ptr addrspace(3)
  %1563 = load <4 x i32>, ptr addrspace(3) %1562, align 16
  %1564 = add i32 %1560, %1057
  %1565 = inttoptr i32 %1564 to ptr addrspace(3)
  %1566 = load <4 x i32>, ptr addrspace(3) %1565, align 16
  %1567 = add i32 %1560, %1062
  %1568 = inttoptr i32 %1567 to ptr addrspace(3)
  %1569 = load <4 x i32>, ptr addrspace(3) %1568, align 16
  %1570 = add i32 %1560, %1067
  %1571 = inttoptr i32 %1570 to ptr addrspace(3)
  %1572 = load <4 x i32>, ptr addrspace(3) %1571, align 16
  %1573 = shufflevector <4 x i32> %1563, <4 x i32> %1566, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1574 = shufflevector <4 x i32> %1569, <4 x i32> %1572, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1575 = shufflevector <8 x i32> %1573, <8 x i32> %1574, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1576 = add i32 %1558, %1083
  %1577 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1576
  %1578 = load i8, ptr addrspace(3) %1577, align 1
  %1579 = zext i8 %1578 to i32
  %1580 = call i32 @llvm.amdgcn.perm(i32 %1579, i32 %1579, i32 0)
  %1581 = add i32 %1558, %1090
  %1582 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1581
  %1583 = load i8, ptr addrspace(3) %1582, align 1
  %1584 = zext i8 %1583 to i32
  %1585 = call i32 @llvm.amdgcn.perm(i32 %1584, i32 %1584, i32 0)
  %1586 = add i32 %1560, %1098
  %1587 = inttoptr i32 %1586 to ptr addrspace(3)
  %1588 = load <4 x i32>, ptr addrspace(3) %1587, align 16
  %1589 = add i32 %1560, %1104
  %1590 = inttoptr i32 %1589 to ptr addrspace(3)
  %1591 = load <4 x i32>, ptr addrspace(3) %1590, align 16
  %1592 = add i32 %1560, %1109
  %1593 = inttoptr i32 %1592 to ptr addrspace(3)
  %1594 = load <4 x i32>, ptr addrspace(3) %1593, align 16
  %1595 = add i32 %1560, %1114
  %1596 = inttoptr i32 %1595 to ptr addrspace(3)
  %1597 = load <4 x i32>, ptr addrspace(3) %1596, align 16
  %1598 = shufflevector <4 x i32> %1588, <4 x i32> %1591, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1599 = shufflevector <4 x i32> %1594, <4 x i32> %1597, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1600 = shufflevector <8 x i32> %1598, <8 x i32> %1599, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1601 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1575, i32 0, <16 x i32> %1600, i16 0, <8 x float> %1555, i32 0, i32 0, i32 %1580, i32 0, i32 0, i32 %1585, i1 false, i1 false)
  %1602 = add i32 %1560, %1124
  %1603 = inttoptr i32 %1602 to ptr addrspace(3)
  %1604 = load <4 x i32>, ptr addrspace(3) %1603, align 16
  %1605 = add i32 %1560, %1130
  %1606 = inttoptr i32 %1605 to ptr addrspace(3)
  %1607 = load <4 x i32>, ptr addrspace(3) %1606, align 16
  %1608 = add i32 %1560, %1135
  %1609 = inttoptr i32 %1608 to ptr addrspace(3)
  %1610 = load <4 x i32>, ptr addrspace(3) %1609, align 16
  %1611 = add i32 %1560, %1140
  %1612 = inttoptr i32 %1611 to ptr addrspace(3)
  %1613 = load <4 x i32>, ptr addrspace(3) %1612, align 16
  %1614 = shufflevector <4 x i32> %1604, <4 x i32> %1607, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1615 = shufflevector <4 x i32> %1610, <4 x i32> %1613, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1616 = shufflevector <8 x i32> %1614, <8 x i32> %1615, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1617 = add i32 %1558, %1147
  %1618 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1617
  %1619 = load i8, ptr addrspace(3) %1618, align 1
  %1620 = zext i8 %1619 to i32
  %1621 = call i32 @llvm.amdgcn.perm(i32 %1620, i32 %1620, i32 0)
  %1622 = add i32 %1558, %1153
  %1623 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1622
  %1624 = load i8, ptr addrspace(3) %1623, align 1
  %1625 = zext i8 %1624 to i32
  %1626 = call i32 @llvm.amdgcn.perm(i32 %1625, i32 %1625, i32 0)
  %1627 = add i32 %1560, %1160
  %1628 = inttoptr i32 %1627 to ptr addrspace(3)
  %1629 = load <4 x i32>, ptr addrspace(3) %1628, align 16
  %1630 = add i32 %1560, %1166
  %1631 = inttoptr i32 %1630 to ptr addrspace(3)
  %1632 = load <4 x i32>, ptr addrspace(3) %1631, align 16
  %1633 = add i32 %1560, %1171
  %1634 = inttoptr i32 %1633 to ptr addrspace(3)
  %1635 = load <4 x i32>, ptr addrspace(3) %1634, align 16
  %1636 = add i32 %1560, %1176
  %1637 = inttoptr i32 %1636 to ptr addrspace(3)
  %1638 = load <4 x i32>, ptr addrspace(3) %1637, align 16
  %1639 = shufflevector <4 x i32> %1629, <4 x i32> %1632, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1640 = shufflevector <4 x i32> %1635, <4 x i32> %1638, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1641 = shufflevector <8 x i32> %1639, <8 x i32> %1640, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1642 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1616, i32 0, <16 x i32> %1641, i16 0, <8 x float> %1601, i32 0, i32 0, i32 %1621, i32 0, i32 0, i32 %1626, i1 false, i1 false)
  %1643 = add i32 %1560, %1186
  %1644 = inttoptr i32 %1643 to ptr addrspace(3)
  %1645 = load <4 x i32>, ptr addrspace(3) %1644, align 16
  %1646 = add i32 %1560, %1192
  %1647 = inttoptr i32 %1646 to ptr addrspace(3)
  %1648 = load <4 x i32>, ptr addrspace(3) %1647, align 16
  %1649 = add i32 %1560, %1197
  %1650 = inttoptr i32 %1649 to ptr addrspace(3)
  %1651 = load <4 x i32>, ptr addrspace(3) %1650, align 16
  %1652 = add i32 %1560, %1202
  %1653 = inttoptr i32 %1652 to ptr addrspace(3)
  %1654 = load <4 x i32>, ptr addrspace(3) %1653, align 16
  %1655 = shufflevector <4 x i32> %1645, <4 x i32> %1648, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1656 = shufflevector <4 x i32> %1651, <4 x i32> %1654, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1657 = shufflevector <8 x i32> %1655, <8 x i32> %1656, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1658 = add i32 %1558, %1209
  %1659 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1658
  %1660 = load i8, ptr addrspace(3) %1659, align 1
  %1661 = zext i8 %1660 to i32
  %1662 = call i32 @llvm.amdgcn.perm(i32 %1661, i32 %1661, i32 0)
  %1663 = add i32 %1558, %1215
  %1664 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1663
  %1665 = load i8, ptr addrspace(3) %1664, align 1
  %1666 = zext i8 %1665 to i32
  %1667 = call i32 @llvm.amdgcn.perm(i32 %1666, i32 %1666, i32 0)
  %1668 = add i32 %1560, %1222
  %1669 = inttoptr i32 %1668 to ptr addrspace(3)
  %1670 = load <4 x i32>, ptr addrspace(3) %1669, align 16
  %1671 = add i32 %1560, %1228
  %1672 = inttoptr i32 %1671 to ptr addrspace(3)
  %1673 = load <4 x i32>, ptr addrspace(3) %1672, align 16
  %1674 = add i32 %1560, %1233
  %1675 = inttoptr i32 %1674 to ptr addrspace(3)
  %1676 = load <4 x i32>, ptr addrspace(3) %1675, align 16
  %1677 = add i32 %1560, %1238
  %1678 = inttoptr i32 %1677 to ptr addrspace(3)
  %1679 = load <4 x i32>, ptr addrspace(3) %1678, align 16
  %1680 = shufflevector <4 x i32> %1670, <4 x i32> %1673, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1681 = shufflevector <4 x i32> %1676, <4 x i32> %1679, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1682 = shufflevector <8 x i32> %1680, <8 x i32> %1681, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1683 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1657, i32 0, <16 x i32> %1682, i16 0, <8 x float> %1642, i32 0, i32 0, i32 %1662, i32 0, i32 0, i32 %1667, i1 false, i1 false)
  %1684 = add i32 %1560, %1248
  %1685 = inttoptr i32 %1684 to ptr addrspace(3)
  %1686 = load <4 x i32>, ptr addrspace(3) %1685, align 16
  %1687 = add i32 %1560, %1254
  %1688 = inttoptr i32 %1687 to ptr addrspace(3)
  %1689 = load <4 x i32>, ptr addrspace(3) %1688, align 16
  %1690 = add i32 %1560, %1259
  %1691 = inttoptr i32 %1690 to ptr addrspace(3)
  %1692 = load <4 x i32>, ptr addrspace(3) %1691, align 16
  %1693 = add i32 %1560, %1264
  %1694 = inttoptr i32 %1693 to ptr addrspace(3)
  %1695 = load <4 x i32>, ptr addrspace(3) %1694, align 16
  %1696 = shufflevector <4 x i32> %1686, <4 x i32> %1689, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1697 = shufflevector <4 x i32> %1692, <4 x i32> %1695, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1698 = shufflevector <8 x i32> %1696, <8 x i32> %1697, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1699 = add i32 %1558, %1271
  %1700 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1699
  %1701 = load i8, ptr addrspace(3) %1700, align 1
  %1702 = zext i8 %1701 to i32
  %1703 = call i32 @llvm.amdgcn.perm(i32 %1702, i32 %1702, i32 0)
  %1704 = add i32 %1558, %1277
  %1705 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1704
  %1706 = load i8, ptr addrspace(3) %1705, align 1
  %1707 = zext i8 %1706 to i32
  %1708 = call i32 @llvm.amdgcn.perm(i32 %1707, i32 %1707, i32 0)
  %1709 = add i32 %1560, %1284
  %1710 = inttoptr i32 %1709 to ptr addrspace(3)
  %1711 = load <4 x i32>, ptr addrspace(3) %1710, align 16
  %1712 = add i32 %1560, %1290
  %1713 = inttoptr i32 %1712 to ptr addrspace(3)
  %1714 = load <4 x i32>, ptr addrspace(3) %1713, align 16
  %1715 = add i32 %1560, %1295
  %1716 = inttoptr i32 %1715 to ptr addrspace(3)
  %1717 = load <4 x i32>, ptr addrspace(3) %1716, align 16
  %1718 = add i32 %1560, %1300
  %1719 = inttoptr i32 %1718 to ptr addrspace(3)
  %1720 = load <4 x i32>, ptr addrspace(3) %1719, align 16
  %1721 = shufflevector <4 x i32> %1711, <4 x i32> %1714, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1722 = shufflevector <4 x i32> %1717, <4 x i32> %1720, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1723 = shufflevector <8 x i32> %1721, <8 x i32> %1722, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1724 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1698, i32 0, <16 x i32> %1723, i16 0, <8 x float> %1683, i32 0, i32 0, i32 %1703, i32 0, i32 0, i32 %1708, i1 false, i1 false)
  %1725 = add i32 %1560, %1310
  %1726 = inttoptr i32 %1725 to ptr addrspace(3)
  %1727 = load <4 x i32>, ptr addrspace(3) %1726, align 16
  %1728 = add i32 %1560, %1316
  %1729 = inttoptr i32 %1728 to ptr addrspace(3)
  %1730 = load <4 x i32>, ptr addrspace(3) %1729, align 16
  %1731 = add i32 %1560, %1321
  %1732 = inttoptr i32 %1731 to ptr addrspace(3)
  %1733 = load <4 x i32>, ptr addrspace(3) %1732, align 16
  %1734 = add i32 %1560, %1326
  %1735 = inttoptr i32 %1734 to ptr addrspace(3)
  %1736 = load <4 x i32>, ptr addrspace(3) %1735, align 16
  %1737 = shufflevector <4 x i32> %1727, <4 x i32> %1730, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1738 = shufflevector <4 x i32> %1733, <4 x i32> %1736, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1739 = shufflevector <8 x i32> %1737, <8 x i32> %1738, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1740 = add i32 %1558, %1333
  %1741 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1740
  %1742 = load i8, ptr addrspace(3) %1741, align 1
  %1743 = zext i8 %1742 to i32
  %1744 = call i32 @llvm.amdgcn.perm(i32 %1743, i32 %1743, i32 0)
  %1745 = add i32 %1558, %1339
  %1746 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1745
  %1747 = load i8, ptr addrspace(3) %1746, align 1
  %1748 = zext i8 %1747 to i32
  %1749 = call i32 @llvm.amdgcn.perm(i32 %1748, i32 %1748, i32 0)
  %1750 = add i32 %1560, %1346
  %1751 = inttoptr i32 %1750 to ptr addrspace(3)
  %1752 = load <4 x i32>, ptr addrspace(3) %1751, align 16
  %1753 = add i32 %1560, %1352
  %1754 = inttoptr i32 %1753 to ptr addrspace(3)
  %1755 = load <4 x i32>, ptr addrspace(3) %1754, align 16
  %1756 = add i32 %1560, %1357
  %1757 = inttoptr i32 %1756 to ptr addrspace(3)
  %1758 = load <4 x i32>, ptr addrspace(3) %1757, align 16
  %1759 = add i32 %1560, %1362
  %1760 = inttoptr i32 %1759 to ptr addrspace(3)
  %1761 = load <4 x i32>, ptr addrspace(3) %1760, align 16
  %1762 = shufflevector <4 x i32> %1752, <4 x i32> %1755, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1763 = shufflevector <4 x i32> %1758, <4 x i32> %1761, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1764 = shufflevector <8 x i32> %1762, <8 x i32> %1763, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1765 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1739, i32 0, <16 x i32> %1764, i16 0, <8 x float> %1724, i32 0, i32 0, i32 %1744, i32 0, i32 0, i32 %1749, i1 false, i1 false)
  %1766 = add i32 %1560, %1372
  %1767 = inttoptr i32 %1766 to ptr addrspace(3)
  %1768 = load <4 x i32>, ptr addrspace(3) %1767, align 16
  %1769 = add i32 %1560, %1378
  %1770 = inttoptr i32 %1769 to ptr addrspace(3)
  %1771 = load <4 x i32>, ptr addrspace(3) %1770, align 16
  %1772 = add i32 %1560, %1383
  %1773 = inttoptr i32 %1772 to ptr addrspace(3)
  %1774 = load <4 x i32>, ptr addrspace(3) %1773, align 16
  %1775 = add i32 %1560, %1388
  %1776 = inttoptr i32 %1775 to ptr addrspace(3)
  %1777 = load <4 x i32>, ptr addrspace(3) %1776, align 16
  %1778 = shufflevector <4 x i32> %1768, <4 x i32> %1771, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1779 = shufflevector <4 x i32> %1774, <4 x i32> %1777, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1780 = shufflevector <8 x i32> %1778, <8 x i32> %1779, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1781 = add i32 %1558, %1395
  %1782 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1781
  %1783 = load i8, ptr addrspace(3) %1782, align 1
  %1784 = zext i8 %1783 to i32
  %1785 = call i32 @llvm.amdgcn.perm(i32 %1784, i32 %1784, i32 0)
  %1786 = add i32 %1558, %1401
  %1787 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1786
  %1788 = load i8, ptr addrspace(3) %1787, align 1
  %1789 = zext i8 %1788 to i32
  %1790 = call i32 @llvm.amdgcn.perm(i32 %1789, i32 %1789, i32 0)
  %1791 = add i32 %1560, %1408
  %1792 = inttoptr i32 %1791 to ptr addrspace(3)
  %1793 = load <4 x i32>, ptr addrspace(3) %1792, align 16
  %1794 = add i32 %1560, %1414
  %1795 = inttoptr i32 %1794 to ptr addrspace(3)
  %1796 = load <4 x i32>, ptr addrspace(3) %1795, align 16
  %1797 = add i32 %1560, %1419
  %1798 = inttoptr i32 %1797 to ptr addrspace(3)
  %1799 = load <4 x i32>, ptr addrspace(3) %1798, align 16
  %1800 = add i32 %1560, %1424
  %1801 = inttoptr i32 %1800 to ptr addrspace(3)
  %1802 = load <4 x i32>, ptr addrspace(3) %1801, align 16
  %1803 = shufflevector <4 x i32> %1793, <4 x i32> %1796, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1804 = shufflevector <4 x i32> %1799, <4 x i32> %1802, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1805 = shufflevector <8 x i32> %1803, <8 x i32> %1804, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1806 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1780, i32 0, <16 x i32> %1805, i16 0, <8 x float> %1765, i32 0, i32 0, i32 %1785, i32 0, i32 0, i32 %1790, i1 false, i1 false)
  %1807 = add i32 %1560, %1434
  %1808 = inttoptr i32 %1807 to ptr addrspace(3)
  %1809 = load <4 x i32>, ptr addrspace(3) %1808, align 16
  %1810 = add i32 %1560, %1440
  %1811 = inttoptr i32 %1810 to ptr addrspace(3)
  %1812 = load <4 x i32>, ptr addrspace(3) %1811, align 16
  %1813 = add i32 %1560, %1445
  %1814 = inttoptr i32 %1813 to ptr addrspace(3)
  %1815 = load <4 x i32>, ptr addrspace(3) %1814, align 16
  %1816 = add i32 %1560, %1450
  %1817 = inttoptr i32 %1816 to ptr addrspace(3)
  %1818 = load <4 x i32>, ptr addrspace(3) %1817, align 16
  %1819 = shufflevector <4 x i32> %1809, <4 x i32> %1812, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1820 = shufflevector <4 x i32> %1815, <4 x i32> %1818, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1821 = shufflevector <8 x i32> %1819, <8 x i32> %1820, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1822 = add i32 %1558, %1457
  %1823 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1822
  %1824 = load i8, ptr addrspace(3) %1823, align 1
  %1825 = zext i8 %1824 to i32
  %1826 = call i32 @llvm.amdgcn.perm(i32 %1825, i32 %1825, i32 0)
  %1827 = add i32 %1558, %1463
  %1828 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1827
  %1829 = load i8, ptr addrspace(3) %1828, align 1
  %1830 = zext i8 %1829 to i32
  %1831 = call i32 @llvm.amdgcn.perm(i32 %1830, i32 %1830, i32 0)
  %1832 = add i32 %1560, %1470
  %1833 = inttoptr i32 %1832 to ptr addrspace(3)
  %1834 = load <4 x i32>, ptr addrspace(3) %1833, align 16
  %1835 = add i32 %1560, %1476
  %1836 = inttoptr i32 %1835 to ptr addrspace(3)
  %1837 = load <4 x i32>, ptr addrspace(3) %1836, align 16
  %1838 = add i32 %1560, %1481
  %1839 = inttoptr i32 %1838 to ptr addrspace(3)
  %1840 = load <4 x i32>, ptr addrspace(3) %1839, align 16
  %1841 = add i32 %1560, %1486
  %1842 = inttoptr i32 %1841 to ptr addrspace(3)
  %1843 = load <4 x i32>, ptr addrspace(3) %1842, align 16
  %1844 = shufflevector <4 x i32> %1834, <4 x i32> %1837, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1845 = shufflevector <4 x i32> %1840, <4 x i32> %1843, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1846 = shufflevector <8 x i32> %1844, <8 x i32> %1845, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1847 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1821, i32 0, <16 x i32> %1846, i16 0, <8 x float> %1806, i32 0, i32 0, i32 %1826, i32 0, i32 0, i32 %1831, i1 false, i1 false)
  %1848 = add i32 %1560, %1496
  %1849 = inttoptr i32 %1848 to ptr addrspace(3)
  %1850 = load <4 x i32>, ptr addrspace(3) %1849, align 16
  %1851 = add i32 %1560, %1502
  %1852 = inttoptr i32 %1851 to ptr addrspace(3)
  %1853 = load <4 x i32>, ptr addrspace(3) %1852, align 16
  %1854 = add i32 %1560, %1507
  %1855 = inttoptr i32 %1854 to ptr addrspace(3)
  %1856 = load <4 x i32>, ptr addrspace(3) %1855, align 16
  %1857 = add i32 %1560, %1512
  %1858 = inttoptr i32 %1857 to ptr addrspace(3)
  %1859 = load <4 x i32>, ptr addrspace(3) %1858, align 16
  %1860 = shufflevector <4 x i32> %1850, <4 x i32> %1853, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1861 = shufflevector <4 x i32> %1856, <4 x i32> %1859, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1862 = shufflevector <8 x i32> %1860, <8 x i32> %1861, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1863 = add i32 %1558, %1519
  %1864 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1863
  %1865 = load i8, ptr addrspace(3) %1864, align 1
  %1866 = zext i8 %1865 to i32
  %1867 = call i32 @llvm.amdgcn.perm(i32 %1866, i32 %1866, i32 0)
  %1868 = add i32 %1558, %1525
  %1869 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1868
  %1870 = load i8, ptr addrspace(3) %1869, align 1
  %1871 = zext i8 %1870 to i32
  %1872 = call i32 @llvm.amdgcn.perm(i32 %1871, i32 %1871, i32 0)
  %1873 = add i32 %1560, %1532
  %1874 = inttoptr i32 %1873 to ptr addrspace(3)
  %1875 = load <4 x i32>, ptr addrspace(3) %1874, align 16
  %1876 = add i32 %1560, %1538
  %1877 = inttoptr i32 %1876 to ptr addrspace(3)
  %1878 = load <4 x i32>, ptr addrspace(3) %1877, align 16
  %1879 = add i32 %1560, %1543
  %1880 = inttoptr i32 %1879 to ptr addrspace(3)
  %1881 = load <4 x i32>, ptr addrspace(3) %1880, align 16
  %1882 = add i32 %1560, %1548
  %1883 = inttoptr i32 %1882 to ptr addrspace(3)
  %1884 = load <4 x i32>, ptr addrspace(3) %1883, align 16
  %1885 = shufflevector <4 x i32> %1875, <4 x i32> %1878, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1886 = shufflevector <4 x i32> %1881, <4 x i32> %1884, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1887 = shufflevector <8 x i32> %1885, <8 x i32> %1886, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1888 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1862, i32 0, <16 x i32> %1887, i16 0, <8 x float> %1847, i32 0, i32 0, i32 %1867, i32 0, i32 0, i32 %1872, i1 false, i1 false)
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
  %1889 = add i32 %19, -1
  %1890 = srem i32 %1889, 4
  %1891 = mul i32 %1890, 50176
  %1892 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1891
  %1893 = ptrtoint ptr addrspace(3) %1892 to i32
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %1894 = add i32 %1893, %1051
  %1895 = inttoptr i32 %1894 to ptr addrspace(3)
  %1896 = load <4 x i32>, ptr addrspace(3) %1895, align 16
  %1897 = add i32 %1893, %1057
  %1898 = inttoptr i32 %1897 to ptr addrspace(3)
  %1899 = load <4 x i32>, ptr addrspace(3) %1898, align 16
  %1900 = add i32 %1893, %1062
  %1901 = inttoptr i32 %1900 to ptr addrspace(3)
  %1902 = load <4 x i32>, ptr addrspace(3) %1901, align 16
  %1903 = add i32 %1893, %1067
  %1904 = inttoptr i32 %1903 to ptr addrspace(3)
  %1905 = load <4 x i32>, ptr addrspace(3) %1904, align 16
  %1906 = shufflevector <4 x i32> %1896, <4 x i32> %1899, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1907 = shufflevector <4 x i32> %1902, <4 x i32> %1905, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1908 = shufflevector <8 x i32> %1906, <8 x i32> %1907, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1909 = add i32 %1891, %1083
  %1910 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1909
  %1911 = load i8, ptr addrspace(3) %1910, align 1
  %1912 = zext i8 %1911 to i32
  %1913 = call i32 @llvm.amdgcn.perm(i32 %1912, i32 %1912, i32 0)
  %1914 = add i32 %1891, %1090
  %1915 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1914
  %1916 = load i8, ptr addrspace(3) %1915, align 1
  %1917 = zext i8 %1916 to i32
  %1918 = call i32 @llvm.amdgcn.perm(i32 %1917, i32 %1917, i32 0)
  %1919 = add i32 %1893, %1098
  %1920 = inttoptr i32 %1919 to ptr addrspace(3)
  %1921 = load <4 x i32>, ptr addrspace(3) %1920, align 16
  %1922 = add i32 %1893, %1104
  %1923 = inttoptr i32 %1922 to ptr addrspace(3)
  %1924 = load <4 x i32>, ptr addrspace(3) %1923, align 16
  %1925 = add i32 %1893, %1109
  %1926 = inttoptr i32 %1925 to ptr addrspace(3)
  %1927 = load <4 x i32>, ptr addrspace(3) %1926, align 16
  %1928 = add i32 %1893, %1114
  %1929 = inttoptr i32 %1928 to ptr addrspace(3)
  %1930 = load <4 x i32>, ptr addrspace(3) %1929, align 16
  %1931 = shufflevector <4 x i32> %1921, <4 x i32> %1924, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1932 = shufflevector <4 x i32> %1927, <4 x i32> %1930, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1933 = shufflevector <8 x i32> %1931, <8 x i32> %1932, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1934 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1908, i32 0, <16 x i32> %1933, i16 0, <8 x float> %1888, i32 0, i32 0, i32 %1913, i32 0, i32 0, i32 %1918, i1 false, i1 false)
  %1935 = add i32 %1893, %1124
  %1936 = inttoptr i32 %1935 to ptr addrspace(3)
  %1937 = load <4 x i32>, ptr addrspace(3) %1936, align 16
  %1938 = add i32 %1893, %1130
  %1939 = inttoptr i32 %1938 to ptr addrspace(3)
  %1940 = load <4 x i32>, ptr addrspace(3) %1939, align 16
  %1941 = add i32 %1893, %1135
  %1942 = inttoptr i32 %1941 to ptr addrspace(3)
  %1943 = load <4 x i32>, ptr addrspace(3) %1942, align 16
  %1944 = add i32 %1893, %1140
  %1945 = inttoptr i32 %1944 to ptr addrspace(3)
  %1946 = load <4 x i32>, ptr addrspace(3) %1945, align 16
  %1947 = shufflevector <4 x i32> %1937, <4 x i32> %1940, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1948 = shufflevector <4 x i32> %1943, <4 x i32> %1946, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1949 = shufflevector <8 x i32> %1947, <8 x i32> %1948, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1950 = add i32 %1891, %1147
  %1951 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1950
  %1952 = load i8, ptr addrspace(3) %1951, align 1
  %1953 = zext i8 %1952 to i32
  %1954 = call i32 @llvm.amdgcn.perm(i32 %1953, i32 %1953, i32 0)
  %1955 = add i32 %1891, %1153
  %1956 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1955
  %1957 = load i8, ptr addrspace(3) %1956, align 1
  %1958 = zext i8 %1957 to i32
  %1959 = call i32 @llvm.amdgcn.perm(i32 %1958, i32 %1958, i32 0)
  %1960 = add i32 %1893, %1160
  %1961 = inttoptr i32 %1960 to ptr addrspace(3)
  %1962 = load <4 x i32>, ptr addrspace(3) %1961, align 16
  %1963 = add i32 %1893, %1166
  %1964 = inttoptr i32 %1963 to ptr addrspace(3)
  %1965 = load <4 x i32>, ptr addrspace(3) %1964, align 16
  %1966 = add i32 %1893, %1171
  %1967 = inttoptr i32 %1966 to ptr addrspace(3)
  %1968 = load <4 x i32>, ptr addrspace(3) %1967, align 16
  %1969 = add i32 %1893, %1176
  %1970 = inttoptr i32 %1969 to ptr addrspace(3)
  %1971 = load <4 x i32>, ptr addrspace(3) %1970, align 16
  %1972 = shufflevector <4 x i32> %1962, <4 x i32> %1965, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1973 = shufflevector <4 x i32> %1968, <4 x i32> %1971, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1974 = shufflevector <8 x i32> %1972, <8 x i32> %1973, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1975 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1949, i32 0, <16 x i32> %1974, i16 0, <8 x float> %1934, i32 0, i32 0, i32 %1954, i32 0, i32 0, i32 %1959, i1 false, i1 false)
  %1976 = add i32 %1893, %1186
  %1977 = inttoptr i32 %1976 to ptr addrspace(3)
  %1978 = load <4 x i32>, ptr addrspace(3) %1977, align 16
  %1979 = add i32 %1893, %1192
  %1980 = inttoptr i32 %1979 to ptr addrspace(3)
  %1981 = load <4 x i32>, ptr addrspace(3) %1980, align 16
  %1982 = add i32 %1893, %1197
  %1983 = inttoptr i32 %1982 to ptr addrspace(3)
  %1984 = load <4 x i32>, ptr addrspace(3) %1983, align 16
  %1985 = add i32 %1893, %1202
  %1986 = inttoptr i32 %1985 to ptr addrspace(3)
  %1987 = load <4 x i32>, ptr addrspace(3) %1986, align 16
  %1988 = shufflevector <4 x i32> %1978, <4 x i32> %1981, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1989 = shufflevector <4 x i32> %1984, <4 x i32> %1987, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1990 = shufflevector <8 x i32> %1988, <8 x i32> %1989, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1991 = add i32 %1891, %1209
  %1992 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1991
  %1993 = load i8, ptr addrspace(3) %1992, align 1
  %1994 = zext i8 %1993 to i32
  %1995 = call i32 @llvm.amdgcn.perm(i32 %1994, i32 %1994, i32 0)
  %1996 = add i32 %1891, %1215
  %1997 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1996
  %1998 = load i8, ptr addrspace(3) %1997, align 1
  %1999 = zext i8 %1998 to i32
  %2000 = call i32 @llvm.amdgcn.perm(i32 %1999, i32 %1999, i32 0)
  %2001 = add i32 %1893, %1222
  %2002 = inttoptr i32 %2001 to ptr addrspace(3)
  %2003 = load <4 x i32>, ptr addrspace(3) %2002, align 16
  %2004 = add i32 %1893, %1228
  %2005 = inttoptr i32 %2004 to ptr addrspace(3)
  %2006 = load <4 x i32>, ptr addrspace(3) %2005, align 16
  %2007 = add i32 %1893, %1233
  %2008 = inttoptr i32 %2007 to ptr addrspace(3)
  %2009 = load <4 x i32>, ptr addrspace(3) %2008, align 16
  %2010 = add i32 %1893, %1238
  %2011 = inttoptr i32 %2010 to ptr addrspace(3)
  %2012 = load <4 x i32>, ptr addrspace(3) %2011, align 16
  %2013 = shufflevector <4 x i32> %2003, <4 x i32> %2006, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2014 = shufflevector <4 x i32> %2009, <4 x i32> %2012, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2015 = shufflevector <8 x i32> %2013, <8 x i32> %2014, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %2016 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1990, i32 0, <16 x i32> %2015, i16 0, <8 x float> %1975, i32 0, i32 0, i32 %1995, i32 0, i32 0, i32 %2000, i1 false, i1 false)
  %2017 = add i32 %1893, %1248
  %2018 = inttoptr i32 %2017 to ptr addrspace(3)
  %2019 = load <4 x i32>, ptr addrspace(3) %2018, align 16
  %2020 = add i32 %1893, %1254
  %2021 = inttoptr i32 %2020 to ptr addrspace(3)
  %2022 = load <4 x i32>, ptr addrspace(3) %2021, align 16
  %2023 = add i32 %1893, %1259
  %2024 = inttoptr i32 %2023 to ptr addrspace(3)
  %2025 = load <4 x i32>, ptr addrspace(3) %2024, align 16
  %2026 = add i32 %1893, %1264
  %2027 = inttoptr i32 %2026 to ptr addrspace(3)
  %2028 = load <4 x i32>, ptr addrspace(3) %2027, align 16
  %2029 = shufflevector <4 x i32> %2019, <4 x i32> %2022, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2030 = shufflevector <4 x i32> %2025, <4 x i32> %2028, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2031 = shufflevector <8 x i32> %2029, <8 x i32> %2030, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2032 = add i32 %1891, %1271
  %2033 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %2032
  %2034 = load i8, ptr addrspace(3) %2033, align 1
  %2035 = zext i8 %2034 to i32
  %2036 = call i32 @llvm.amdgcn.perm(i32 %2035, i32 %2035, i32 0)
  %2037 = add i32 %1891, %1277
  %2038 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %2037
  %2039 = load i8, ptr addrspace(3) %2038, align 1
  %2040 = zext i8 %2039 to i32
  %2041 = call i32 @llvm.amdgcn.perm(i32 %2040, i32 %2040, i32 0)
  %2042 = add i32 %1893, %1284
  %2043 = inttoptr i32 %2042 to ptr addrspace(3)
  %2044 = load <4 x i32>, ptr addrspace(3) %2043, align 16
  %2045 = add i32 %1893, %1290
  %2046 = inttoptr i32 %2045 to ptr addrspace(3)
  %2047 = load <4 x i32>, ptr addrspace(3) %2046, align 16
  %2048 = add i32 %1893, %1295
  %2049 = inttoptr i32 %2048 to ptr addrspace(3)
  %2050 = load <4 x i32>, ptr addrspace(3) %2049, align 16
  %2051 = add i32 %1893, %1300
  %2052 = inttoptr i32 %2051 to ptr addrspace(3)
  %2053 = load <4 x i32>, ptr addrspace(3) %2052, align 16
  %2054 = shufflevector <4 x i32> %2044, <4 x i32> %2047, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2055 = shufflevector <4 x i32> %2050, <4 x i32> %2053, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2056 = shufflevector <8 x i32> %2054, <8 x i32> %2055, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %2057 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %2031, i32 0, <16 x i32> %2056, i16 0, <8 x float> %2016, i32 0, i32 0, i32 %2036, i32 0, i32 0, i32 %2041, i1 false, i1 false)
  %2058 = add i32 %1893, %1310
  %2059 = inttoptr i32 %2058 to ptr addrspace(3)
  %2060 = load <4 x i32>, ptr addrspace(3) %2059, align 16
  %2061 = add i32 %1893, %1316
  %2062 = inttoptr i32 %2061 to ptr addrspace(3)
  %2063 = load <4 x i32>, ptr addrspace(3) %2062, align 16
  %2064 = add i32 %1893, %1321
  %2065 = inttoptr i32 %2064 to ptr addrspace(3)
  %2066 = load <4 x i32>, ptr addrspace(3) %2065, align 16
  %2067 = add i32 %1893, %1326
  %2068 = inttoptr i32 %2067 to ptr addrspace(3)
  %2069 = load <4 x i32>, ptr addrspace(3) %2068, align 16
  %2070 = shufflevector <4 x i32> %2060, <4 x i32> %2063, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2071 = shufflevector <4 x i32> %2066, <4 x i32> %2069, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2072 = shufflevector <8 x i32> %2070, <8 x i32> %2071, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2073 = add i32 %1891, %1333
  %2074 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %2073
  %2075 = load i8, ptr addrspace(3) %2074, align 1
  %2076 = zext i8 %2075 to i32
  %2077 = call i32 @llvm.amdgcn.perm(i32 %2076, i32 %2076, i32 0)
  %2078 = add i32 %1891, %1339
  %2079 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %2078
  %2080 = load i8, ptr addrspace(3) %2079, align 1
  %2081 = zext i8 %2080 to i32
  %2082 = call i32 @llvm.amdgcn.perm(i32 %2081, i32 %2081, i32 0)
  %2083 = add i32 %1893, %1346
  %2084 = inttoptr i32 %2083 to ptr addrspace(3)
  %2085 = load <4 x i32>, ptr addrspace(3) %2084, align 16
  %2086 = add i32 %1893, %1352
  %2087 = inttoptr i32 %2086 to ptr addrspace(3)
  %2088 = load <4 x i32>, ptr addrspace(3) %2087, align 16
  %2089 = add i32 %1893, %1357
  %2090 = inttoptr i32 %2089 to ptr addrspace(3)
  %2091 = load <4 x i32>, ptr addrspace(3) %2090, align 16
  %2092 = add i32 %1893, %1362
  %2093 = inttoptr i32 %2092 to ptr addrspace(3)
  %2094 = load <4 x i32>, ptr addrspace(3) %2093, align 16
  %2095 = shufflevector <4 x i32> %2085, <4 x i32> %2088, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2096 = shufflevector <4 x i32> %2091, <4 x i32> %2094, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2097 = shufflevector <8 x i32> %2095, <8 x i32> %2096, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %2098 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %2072, i32 0, <16 x i32> %2097, i16 0, <8 x float> %2057, i32 0, i32 0, i32 %2077, i32 0, i32 0, i32 %2082, i1 false, i1 false)
  %2099 = add i32 %1893, %1372
  %2100 = inttoptr i32 %2099 to ptr addrspace(3)
  %2101 = load <4 x i32>, ptr addrspace(3) %2100, align 16
  %2102 = add i32 %1893, %1378
  %2103 = inttoptr i32 %2102 to ptr addrspace(3)
  %2104 = load <4 x i32>, ptr addrspace(3) %2103, align 16
  %2105 = add i32 %1893, %1383
  %2106 = inttoptr i32 %2105 to ptr addrspace(3)
  %2107 = load <4 x i32>, ptr addrspace(3) %2106, align 16
  %2108 = add i32 %1893, %1388
  %2109 = inttoptr i32 %2108 to ptr addrspace(3)
  %2110 = load <4 x i32>, ptr addrspace(3) %2109, align 16
  %2111 = shufflevector <4 x i32> %2101, <4 x i32> %2104, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2112 = shufflevector <4 x i32> %2107, <4 x i32> %2110, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2113 = shufflevector <8 x i32> %2111, <8 x i32> %2112, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2114 = add i32 %1891, %1395
  %2115 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %2114
  %2116 = load i8, ptr addrspace(3) %2115, align 1
  %2117 = zext i8 %2116 to i32
  %2118 = call i32 @llvm.amdgcn.perm(i32 %2117, i32 %2117, i32 0)
  %2119 = add i32 %1891, %1401
  %2120 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %2119
  %2121 = load i8, ptr addrspace(3) %2120, align 1
  %2122 = zext i8 %2121 to i32
  %2123 = call i32 @llvm.amdgcn.perm(i32 %2122, i32 %2122, i32 0)
  %2124 = add i32 %1893, %1408
  %2125 = inttoptr i32 %2124 to ptr addrspace(3)
  %2126 = load <4 x i32>, ptr addrspace(3) %2125, align 16
  %2127 = add i32 %1893, %1414
  %2128 = inttoptr i32 %2127 to ptr addrspace(3)
  %2129 = load <4 x i32>, ptr addrspace(3) %2128, align 16
  %2130 = add i32 %1893, %1419
  %2131 = inttoptr i32 %2130 to ptr addrspace(3)
  %2132 = load <4 x i32>, ptr addrspace(3) %2131, align 16
  %2133 = add i32 %1893, %1424
  %2134 = inttoptr i32 %2133 to ptr addrspace(3)
  %2135 = load <4 x i32>, ptr addrspace(3) %2134, align 16
  %2136 = shufflevector <4 x i32> %2126, <4 x i32> %2129, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2137 = shufflevector <4 x i32> %2132, <4 x i32> %2135, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2138 = shufflevector <8 x i32> %2136, <8 x i32> %2137, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %2139 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %2113, i32 0, <16 x i32> %2138, i16 0, <8 x float> %2098, i32 0, i32 0, i32 %2118, i32 0, i32 0, i32 %2123, i1 false, i1 false)
  %2140 = add i32 %1893, %1434
  %2141 = inttoptr i32 %2140 to ptr addrspace(3)
  %2142 = load <4 x i32>, ptr addrspace(3) %2141, align 16
  %2143 = add i32 %1893, %1440
  %2144 = inttoptr i32 %2143 to ptr addrspace(3)
  %2145 = load <4 x i32>, ptr addrspace(3) %2144, align 16
  %2146 = add i32 %1893, %1445
  %2147 = inttoptr i32 %2146 to ptr addrspace(3)
  %2148 = load <4 x i32>, ptr addrspace(3) %2147, align 16
  %2149 = add i32 %1893, %1450
  %2150 = inttoptr i32 %2149 to ptr addrspace(3)
  %2151 = load <4 x i32>, ptr addrspace(3) %2150, align 16
  %2152 = shufflevector <4 x i32> %2142, <4 x i32> %2145, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2153 = shufflevector <4 x i32> %2148, <4 x i32> %2151, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2154 = shufflevector <8 x i32> %2152, <8 x i32> %2153, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2155 = add i32 %1891, %1457
  %2156 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %2155
  %2157 = load i8, ptr addrspace(3) %2156, align 1
  %2158 = zext i8 %2157 to i32
  %2159 = call i32 @llvm.amdgcn.perm(i32 %2158, i32 %2158, i32 0)
  %2160 = add i32 %1891, %1463
  %2161 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %2160
  %2162 = load i8, ptr addrspace(3) %2161, align 1
  %2163 = zext i8 %2162 to i32
  %2164 = call i32 @llvm.amdgcn.perm(i32 %2163, i32 %2163, i32 0)
  %2165 = add i32 %1893, %1470
  %2166 = inttoptr i32 %2165 to ptr addrspace(3)
  %2167 = load <4 x i32>, ptr addrspace(3) %2166, align 16
  %2168 = add i32 %1893, %1476
  %2169 = inttoptr i32 %2168 to ptr addrspace(3)
  %2170 = load <4 x i32>, ptr addrspace(3) %2169, align 16
  %2171 = add i32 %1893, %1481
  %2172 = inttoptr i32 %2171 to ptr addrspace(3)
  %2173 = load <4 x i32>, ptr addrspace(3) %2172, align 16
  %2174 = add i32 %1893, %1486
  %2175 = inttoptr i32 %2174 to ptr addrspace(3)
  %2176 = load <4 x i32>, ptr addrspace(3) %2175, align 16
  %2177 = shufflevector <4 x i32> %2167, <4 x i32> %2170, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2178 = shufflevector <4 x i32> %2173, <4 x i32> %2176, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2179 = shufflevector <8 x i32> %2177, <8 x i32> %2178, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %2180 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %2154, i32 0, <16 x i32> %2179, i16 0, <8 x float> %2139, i32 0, i32 0, i32 %2159, i32 0, i32 0, i32 %2164, i1 false, i1 false)
  %2181 = add i32 %1893, %1496
  %2182 = inttoptr i32 %2181 to ptr addrspace(3)
  %2183 = load <4 x i32>, ptr addrspace(3) %2182, align 16
  %2184 = add i32 %1893, %1502
  %2185 = inttoptr i32 %2184 to ptr addrspace(3)
  %2186 = load <4 x i32>, ptr addrspace(3) %2185, align 16
  %2187 = add i32 %1893, %1507
  %2188 = inttoptr i32 %2187 to ptr addrspace(3)
  %2189 = load <4 x i32>, ptr addrspace(3) %2188, align 16
  %2190 = add i32 %1893, %1512
  %2191 = inttoptr i32 %2190 to ptr addrspace(3)
  %2192 = load <4 x i32>, ptr addrspace(3) %2191, align 16
  %2193 = shufflevector <4 x i32> %2183, <4 x i32> %2186, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2194 = shufflevector <4 x i32> %2189, <4 x i32> %2192, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2195 = shufflevector <8 x i32> %2193, <8 x i32> %2194, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2196 = add i32 %1891, %1519
  %2197 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %2196
  %2198 = load i8, ptr addrspace(3) %2197, align 1
  %2199 = zext i8 %2198 to i32
  %2200 = call i32 @llvm.amdgcn.perm(i32 %2199, i32 %2199, i32 0)
  %2201 = add i32 %1891, %1525
  %2202 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %2201
  %2203 = load i8, ptr addrspace(3) %2202, align 1
  %2204 = zext i8 %2203 to i32
  %2205 = call i32 @llvm.amdgcn.perm(i32 %2204, i32 %2204, i32 0)
  %2206 = add i32 %1893, %1532
  %2207 = inttoptr i32 %2206 to ptr addrspace(3)
  %2208 = load <4 x i32>, ptr addrspace(3) %2207, align 16
  %2209 = add i32 %1893, %1538
  %2210 = inttoptr i32 %2209 to ptr addrspace(3)
  %2211 = load <4 x i32>, ptr addrspace(3) %2210, align 16
  %2212 = add i32 %1893, %1543
  %2213 = inttoptr i32 %2212 to ptr addrspace(3)
  %2214 = load <4 x i32>, ptr addrspace(3) %2213, align 16
  %2215 = add i32 %1893, %1548
  %2216 = inttoptr i32 %2215 to ptr addrspace(3)
  %2217 = load <4 x i32>, ptr addrspace(3) %2216, align 16
  %2218 = shufflevector <4 x i32> %2208, <4 x i32> %2211, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2219 = shufflevector <4 x i32> %2214, <4 x i32> %2217, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %2220 = shufflevector <8 x i32> %2218, <8 x i32> %2219, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %2221 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %2195, i32 0, <16 x i32> %2220, i16 0, <8 x float> %2180, i32 0, i32 0, i32 %2200, i32 0, i32 0, i32 %2205, i1 false, i1 false)
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
  %2222 = mul i32 %50, 8
  %2223 = add i32 %99, %2222
  %2224 = fptrunc <8 x float> %2221 to <8 x bfloat>
  %2225 = bitcast <8 x bfloat> %2224 to <16 x i8>
  %2226 = mul i32 %1089, 32
  %2227 = add i32 %2226, %2223
  %2228 = mul i32 %2227, 2
  %2229 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %2228
  store <16 x i8> %2225, ptr addrspace(3) %2229, align 1
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %2230 = mul i64 %62, %22
  %2231 = add i64 %2230, %63
  %2232 = getelementptr bfloat, ptr addrspace(1) %0, i64 %2231
  %2233 = icmp eq i64 %22, -2147483648
  %2234 = select i1 %2233, i64 32, i64 %22
  %2235 = ptrtoint ptr addrspace(1) %2232 to i64
  %2236 = call i32 @llvm.amdgcn.wave.id()
  %2237 = urem i32 %2236, 2
  %2238 = mul i32 %2237, 8
  %2239 = zext i32 %2238 to i64
  %2240 = mul i64 %2239, %2234
  %2241 = mul i64 %2240, 2
  %2242 = add i64 %2235, %2241
  %2243 = mul i32 %2237, 512
  %2244 = add i32 ptrtoint (ptr addrspace(3) @__dynamic_shared_0 to i32), %2243
  %2245 = trunc i64 %2242 to i32
  %2246 = lshr i64 %2242, 32
  %2247 = trunc i64 %2246 to i32
  %2248 = or i32 %2247, -2147483648
  %2249 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %2244, i64 1
  %2250 = insertelement <4 x i32> %2249, i32 %2245, i64 2
  %2251 = insertelement <4 x i32> %2250, i32 %2248, i64 3
  %2252 = sub i32 %64, %2238
  %2253 = call i32 @llvm.smax.i32(i32 %2252, i32 0)
  %2254 = and i32 %2253, 65535
  %2255 = shl i32 %2254, 16
  %2256 = or i32 %2255, 32767
  %2257 = lshr i32 %2253, 16
  %2258 = and i32 %2257, 65535
  %2259 = or i32 %2258, 2097152
  %2260 = trunc i64 %2234 to i32
  %2261 = lshr i64 %2234, 32
  %2262 = trunc i64 %2261 to i32
  %2263 = and i32 %2262, 65535
  %2264 = insertelement <8 x i32> <i32 65536, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %2256, i64 2
  %2265 = insertelement <8 x i32> %2264, i32 %2259, i64 3
  %2266 = insertelement <8 x i32> %2265, i32 8, i64 4
  %2267 = insertelement <8 x i32> %2266, i32 %2260, i64 5
  %2268 = insertelement <8 x i32> %2267, i32 %2263, i64 6
  %2269 = insertelement <8 x i32> %2268, i32 0, i64 7
  call void @llvm.amdgcn.tensor.store.from.lds(<4 x i32> %2251, <8 x i32> %2269, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
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
