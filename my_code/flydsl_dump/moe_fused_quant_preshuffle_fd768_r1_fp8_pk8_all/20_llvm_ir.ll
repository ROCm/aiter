; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

define amdgpu_kernel void @moe_fused_quant_preshuffle_fd768_r1_fp8_pk8_all(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, i32 %4, i32 %5) #0 {
  %7 = call i32 @llvm.amdgcn.workitem.id.x()
  %8 = sext i32 %7 to i64
  %9 = trunc i64 %8 to i32
  %10 = call i32 @llvm.amdgcn.workgroup.id.x()
  %11 = sext i32 %10 to i64
  %12 = trunc i64 %11 to i32
  %13 = sdiv i32 %9, 32
  %14 = mul i32 %13, 32
  %15 = icmp ne i32 %9, %14
  %16 = icmp slt i32 %9, 0
  %17 = icmp ne i1 %16, false
  %18 = and i1 %15, %17
  %19 = add i32 %13, -1
  %20 = select i1 %18, i32 %19, i32 %13
  %21 = mul i32 %20, 32
  %22 = sub i32 %9, %21
  %23 = mul i32 %12, 8
  %24 = add i32 %23, %20
  %25 = icmp ult i32 %24, %4
  br i1 %25, label %26, label %305

26:                                               ; preds = %6
  %27 = udiv i32 %24, %5
  %28 = mul i32 %27, %5
  %29 = sub i32 %24, %28
  %30 = udiv i32 %29, 16
  %31 = mul i32 %30, 16
  %32 = sub i32 %29, %31
  %33 = udiv i32 %32, 16
  %34 = mul i32 %33, 16
  %35 = sub i32 %32, %34
  %36 = add i32 %31, %35
  %37 = mul i32 %5, 6
  %38 = mul i32 %27, %37
  %39 = mul i32 %36, 6
  %40 = add i32 %38, %39
  %41 = add i32 %40, %33
  %42 = ptrtoint ptr addrspace(1) %1 to i64
  %43 = ptrtoint ptr addrspace(1) %0 to i64
  %44 = ptrtoint ptr addrspace(1) %2 to i64
  %45 = inttoptr i64 %44 to ptr
  %46 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %45, i16 0, i64 4294967295, i32 159744)
  %47 = udiv i32 %22, 4
  %48 = mul i32 %47, 4
  %49 = sub i32 %22, %48
  %50 = icmp eq i32 %49, 0
  %51 = zext i32 %24 to i64
  %52 = mul i64 %51, 768
  %53 = add i64 %42, %52
  %54 = inttoptr i64 %53 to ptr
  %55 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %54, i16 0, i64 768, i32 159744)
  %56 = mul i64 %51, 1536
  %57 = add i64 %43, %56
  %58 = inttoptr i64 %57 to ptr
  %59 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %58, i16 0, i64 1536, i32 159744)
  %60 = icmp ult i32 %47, 24
  br i1 %60, label %61, label %140

61:                                               ; preds = %26
  %62 = mul i32 %47, 32
  %63 = mul i32 %49, 8
  %64 = add i32 %62, %63
  %65 = lshr i32 %64, 1
  %66 = mul i32 %65, 4
  %67 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %59, i32 %66, i32 0, i32 0)
  %68 = bitcast <4 x i32> %67 to <8 x bfloat>
  %69 = fpext <8 x bfloat> %68 to <8 x float>
  %70 = extractelement <8 x float> %69, i64 0
  %71 = call float @llvm.fabs.f32(float %70)
  %72 = call float @llvm.maximum.f32(float %71, float 0.000000e+00)
  %73 = extractelement <8 x float> %69, i64 1
  %74 = call float @llvm.fabs.f32(float %73)
  %75 = call float @llvm.maximum.f32(float %72, float %74)
  %76 = extractelement <8 x float> %69, i64 2
  %77 = call float @llvm.fabs.f32(float %76)
  %78 = call float @llvm.maximum.f32(float %75, float %77)
  %79 = extractelement <8 x float> %69, i64 3
  %80 = call float @llvm.fabs.f32(float %79)
  %81 = call float @llvm.maximum.f32(float %78, float %80)
  %82 = extractelement <8 x float> %69, i64 4
  %83 = call float @llvm.fabs.f32(float %82)
  %84 = call float @llvm.maximum.f32(float %81, float %83)
  %85 = extractelement <8 x float> %69, i64 5
  %86 = call float @llvm.fabs.f32(float %85)
  %87 = call float @llvm.maximum.f32(float %84, float %86)
  %88 = extractelement <8 x float> %69, i64 6
  %89 = call float @llvm.fabs.f32(float %88)
  %90 = call float @llvm.maximum.f32(float %87, float %89)
  %91 = extractelement <8 x float> %69, i64 7
  %92 = call float @llvm.fabs.f32(float %91)
  %93 = call float @llvm.maximum.f32(float %90, float %92)
  %94 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %95 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %94)
  %96 = add i32 %95, 32
  %97 = and i32 %96, -32
  %98 = xor i32 %95, 1
  %99 = icmp slt i32 %98, %97
  %100 = select i1 %99, i32 %98, i32 %95
  %101 = shl i32 %100, 2
  %102 = bitcast float %93 to i32
  %103 = call i32 @llvm.amdgcn.ds.bpermute(i32 %101, i32 %102)
  %104 = bitcast i32 %103 to float
  %105 = call float @llvm.maximum.f32(float %93, float %104)
  %106 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %107 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %106)
  %108 = add i32 %107, 32
  %109 = and i32 %108, -32
  %110 = xor i32 %107, 2
  %111 = icmp slt i32 %110, %109
  %112 = select i1 %111, i32 %110, i32 %107
  %113 = shl i32 %112, 2
  %114 = bitcast float %105 to i32
  %115 = call i32 @llvm.amdgcn.ds.bpermute(i32 %113, i32 %114)
  %116 = bitcast i32 %115 to float
  %117 = call float @llvm.maximum.f32(float %105, float %116)
  %118 = fmul float %117, 0x3F624924A0000000
  %119 = bitcast float %118 to i32
  %120 = and i32 %119, 8388607
  %121 = lshr i32 %119, 23
  %122 = and i32 %121, 255
  %123 = icmp ne i32 %120, 0
  %124 = add i32 %122, 1
  %125 = select i1 %123, i32 %124, i32 %122
  %126 = call i32 @llvm.smax.i32(i32 %125, i32 0)
  %127 = call i32 @llvm.smin.i32(i32 %126, i32 255)
  %128 = shl i32 %127, 23
  %129 = bitcast i32 %128 to float
  %130 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %68, float %129)
  %131 = udiv i32 %47, 4
  %132 = mul i32 %131, 4
  %133 = sub i32 %47, %132
  %134 = trunc i32 %127 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %130, ptr addrspace(8) %55, i32 %64, i32 0, i32 0)
  br i1 %50, label %135, label %139

135:                                              ; preds = %61
  %136 = add i32 %41, %131
  %137 = mul i32 %136, 4
  %138 = add i32 %137, %133
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %134, ptr addrspace(8) %46, i32 %138, i32 0, i32 0)
  br label %139

139:                                              ; preds = %135, %61
  br label %140

140:                                              ; preds = %139, %26
  %141 = add i32 %47, 8
  %142 = icmp ult i32 %141, 24
  br i1 %142, label %143, label %222

143:                                              ; preds = %140
  %144 = mul i32 %141, 32
  %145 = mul i32 %49, 8
  %146 = add i32 %144, %145
  %147 = lshr i32 %146, 1
  %148 = mul i32 %147, 4
  %149 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %59, i32 %148, i32 0, i32 0)
  %150 = bitcast <4 x i32> %149 to <8 x bfloat>
  %151 = fpext <8 x bfloat> %150 to <8 x float>
  %152 = extractelement <8 x float> %151, i64 0
  %153 = call float @llvm.fabs.f32(float %152)
  %154 = call float @llvm.maximum.f32(float %153, float 0.000000e+00)
  %155 = extractelement <8 x float> %151, i64 1
  %156 = call float @llvm.fabs.f32(float %155)
  %157 = call float @llvm.maximum.f32(float %154, float %156)
  %158 = extractelement <8 x float> %151, i64 2
  %159 = call float @llvm.fabs.f32(float %158)
  %160 = call float @llvm.maximum.f32(float %157, float %159)
  %161 = extractelement <8 x float> %151, i64 3
  %162 = call float @llvm.fabs.f32(float %161)
  %163 = call float @llvm.maximum.f32(float %160, float %162)
  %164 = extractelement <8 x float> %151, i64 4
  %165 = call float @llvm.fabs.f32(float %164)
  %166 = call float @llvm.maximum.f32(float %163, float %165)
  %167 = extractelement <8 x float> %151, i64 5
  %168 = call float @llvm.fabs.f32(float %167)
  %169 = call float @llvm.maximum.f32(float %166, float %168)
  %170 = extractelement <8 x float> %151, i64 6
  %171 = call float @llvm.fabs.f32(float %170)
  %172 = call float @llvm.maximum.f32(float %169, float %171)
  %173 = extractelement <8 x float> %151, i64 7
  %174 = call float @llvm.fabs.f32(float %173)
  %175 = call float @llvm.maximum.f32(float %172, float %174)
  %176 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %177 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %176)
  %178 = add i32 %177, 32
  %179 = and i32 %178, -32
  %180 = xor i32 %177, 1
  %181 = icmp slt i32 %180, %179
  %182 = select i1 %181, i32 %180, i32 %177
  %183 = shl i32 %182, 2
  %184 = bitcast float %175 to i32
  %185 = call i32 @llvm.amdgcn.ds.bpermute(i32 %183, i32 %184)
  %186 = bitcast i32 %185 to float
  %187 = call float @llvm.maximum.f32(float %175, float %186)
  %188 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %189 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %188)
  %190 = add i32 %189, 32
  %191 = and i32 %190, -32
  %192 = xor i32 %189, 2
  %193 = icmp slt i32 %192, %191
  %194 = select i1 %193, i32 %192, i32 %189
  %195 = shl i32 %194, 2
  %196 = bitcast float %187 to i32
  %197 = call i32 @llvm.amdgcn.ds.bpermute(i32 %195, i32 %196)
  %198 = bitcast i32 %197 to float
  %199 = call float @llvm.maximum.f32(float %187, float %198)
  %200 = fmul float %199, 0x3F624924A0000000
  %201 = bitcast float %200 to i32
  %202 = and i32 %201, 8388607
  %203 = lshr i32 %201, 23
  %204 = and i32 %203, 255
  %205 = icmp ne i32 %202, 0
  %206 = add i32 %204, 1
  %207 = select i1 %205, i32 %206, i32 %204
  %208 = call i32 @llvm.smax.i32(i32 %207, i32 0)
  %209 = call i32 @llvm.smin.i32(i32 %208, i32 255)
  %210 = shl i32 %209, 23
  %211 = bitcast i32 %210 to float
  %212 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %150, float %211)
  %213 = udiv i32 %141, 4
  %214 = mul i32 %213, 4
  %215 = sub i32 %141, %214
  %216 = trunc i32 %209 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %212, ptr addrspace(8) %55, i32 %146, i32 0, i32 0)
  br i1 %50, label %217, label %221

217:                                              ; preds = %143
  %218 = add i32 %41, %213
  %219 = mul i32 %218, 4
  %220 = add i32 %219, %215
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %216, ptr addrspace(8) %46, i32 %220, i32 0, i32 0)
  br label %221

221:                                              ; preds = %217, %143
  br label %222

222:                                              ; preds = %221, %140
  %223 = add i32 %47, 16
  %224 = icmp ult i32 %223, 24
  br i1 %224, label %225, label %304

225:                                              ; preds = %222
  %226 = mul i32 %223, 32
  %227 = mul i32 %49, 8
  %228 = add i32 %226, %227
  %229 = lshr i32 %228, 1
  %230 = mul i32 %229, 4
  %231 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %59, i32 %230, i32 0, i32 0)
  %232 = bitcast <4 x i32> %231 to <8 x bfloat>
  %233 = fpext <8 x bfloat> %232 to <8 x float>
  %234 = extractelement <8 x float> %233, i64 0
  %235 = call float @llvm.fabs.f32(float %234)
  %236 = call float @llvm.maximum.f32(float %235, float 0.000000e+00)
  %237 = extractelement <8 x float> %233, i64 1
  %238 = call float @llvm.fabs.f32(float %237)
  %239 = call float @llvm.maximum.f32(float %236, float %238)
  %240 = extractelement <8 x float> %233, i64 2
  %241 = call float @llvm.fabs.f32(float %240)
  %242 = call float @llvm.maximum.f32(float %239, float %241)
  %243 = extractelement <8 x float> %233, i64 3
  %244 = call float @llvm.fabs.f32(float %243)
  %245 = call float @llvm.maximum.f32(float %242, float %244)
  %246 = extractelement <8 x float> %233, i64 4
  %247 = call float @llvm.fabs.f32(float %246)
  %248 = call float @llvm.maximum.f32(float %245, float %247)
  %249 = extractelement <8 x float> %233, i64 5
  %250 = call float @llvm.fabs.f32(float %249)
  %251 = call float @llvm.maximum.f32(float %248, float %250)
  %252 = extractelement <8 x float> %233, i64 6
  %253 = call float @llvm.fabs.f32(float %252)
  %254 = call float @llvm.maximum.f32(float %251, float %253)
  %255 = extractelement <8 x float> %233, i64 7
  %256 = call float @llvm.fabs.f32(float %255)
  %257 = call float @llvm.maximum.f32(float %254, float %256)
  %258 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %259 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %258)
  %260 = add i32 %259, 32
  %261 = and i32 %260, -32
  %262 = xor i32 %259, 1
  %263 = icmp slt i32 %262, %261
  %264 = select i1 %263, i32 %262, i32 %259
  %265 = shl i32 %264, 2
  %266 = bitcast float %257 to i32
  %267 = call i32 @llvm.amdgcn.ds.bpermute(i32 %265, i32 %266)
  %268 = bitcast i32 %267 to float
  %269 = call float @llvm.maximum.f32(float %257, float %268)
  %270 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %271 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %270)
  %272 = add i32 %271, 32
  %273 = and i32 %272, -32
  %274 = xor i32 %271, 2
  %275 = icmp slt i32 %274, %273
  %276 = select i1 %275, i32 %274, i32 %271
  %277 = shl i32 %276, 2
  %278 = bitcast float %269 to i32
  %279 = call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %278)
  %280 = bitcast i32 %279 to float
  %281 = call float @llvm.maximum.f32(float %269, float %280)
  %282 = fmul float %281, 0x3F624924A0000000
  %283 = bitcast float %282 to i32
  %284 = and i32 %283, 8388607
  %285 = lshr i32 %283, 23
  %286 = and i32 %285, 255
  %287 = icmp ne i32 %284, 0
  %288 = add i32 %286, 1
  %289 = select i1 %287, i32 %288, i32 %286
  %290 = call i32 @llvm.smax.i32(i32 %289, i32 0)
  %291 = call i32 @llvm.smin.i32(i32 %290, i32 255)
  %292 = shl i32 %291, 23
  %293 = bitcast i32 %292 to float
  %294 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %232, float %293)
  %295 = udiv i32 %223, 4
  %296 = mul i32 %295, 4
  %297 = sub i32 %223, %296
  %298 = trunc i32 %291 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %294, ptr addrspace(8) %55, i32 %228, i32 0, i32 0)
  br i1 %50, label %299, label %303

299:                                              ; preds = %225
  %300 = add i32 %41, %295
  %301 = mul i32 %300, 4
  %302 = add i32 %301, %297
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %298, ptr addrspace(8) %46, i32 %302, i32 0, i32 0)
  br label %303

303:                                              ; preds = %299, %225
  br label %304

304:                                              ; preds = %303, %222
  br label %305

305:                                              ; preds = %304, %6
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr readnone, i16, i64, i32) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #4

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32>, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #6

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #6

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #6 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
