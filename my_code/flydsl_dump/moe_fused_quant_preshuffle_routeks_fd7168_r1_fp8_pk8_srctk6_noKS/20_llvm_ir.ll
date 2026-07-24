; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

define amdgpu_kernel void @moe_fused_quant_preshuffle_routeks_fd7168_r1_fp8_pk8_srctk6_noKS(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, i32 %5, i32 %6, ptr addrspace(1) %7) #0 {
  %9 = call i32 @llvm.amdgcn.workitem.id.x()
  %10 = sext i32 %9 to i64
  %11 = trunc i64 %10 to i32
  %12 = call i32 @llvm.amdgcn.workgroup.id.x()
  %13 = sext i32 %12 to i64
  %14 = trunc i64 %13 to i32
  %15 = sdiv i32 %11, 32
  %16 = mul i32 %15, 32
  %17 = icmp ne i32 %11, %16
  %18 = icmp slt i32 %11, 0
  %19 = icmp ne i1 %18, false
  %20 = and i1 %17, %19
  %21 = add i32 %15, -1
  %22 = select i1 %20, i32 %21, i32 %15
  %23 = mul i32 %22, 32
  %24 = sub i32 %11, %23
  %25 = mul i32 %14, 8
  %26 = add i32 %25, %22
  %27 = ptrtoint ptr addrspace(1) %7 to i64
  %28 = inttoptr i64 %27 to ptr
  %29 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %28, i16 0, i64 4294967295, i32 159744)
  %30 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %29, i32 0, i32 0, i32 0)
  %31 = icmp ult i32 %26, %30
  br i1 %31, label %32, label %2362

32:                                               ; preds = %8
  %33 = ptrtoint ptr addrspace(1) %3 to i64
  %34 = inttoptr i64 %33 to ptr
  %35 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %34, i16 0, i64 4294967295, i32 159744)
  %36 = mul i32 %26, 4
  %37 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %35, i32 %36, i32 0, i32 0)
  %38 = udiv i32 %37, 16
  %39 = mul i32 %38, 16
  %40 = sub i32 %37, %39
  %41 = udiv i32 %40, 16
  %42 = mul i32 %41, 16
  %43 = sub i32 %40, %42
  %44 = add i32 %39, %43
  %45 = mul i32 %44, 56
  %46 = add i32 %45, %41
  %47 = udiv i32 %26, 6
  %48 = ptrtoint ptr addrspace(1) %2 to i64
  %49 = inttoptr i64 %48 to ptr
  %50 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %49, i16 0, i64 4294967295, i32 159744)
  %51 = ptrtoint ptr addrspace(1) %1 to i64
  %52 = ptrtoint ptr addrspace(1) %0 to i64
  %53 = udiv i32 %24, 4
  %54 = mul i32 %53, 4
  %55 = sub i32 %24, %54
  %56 = icmp eq i32 %55, 0
  %57 = zext i32 %37 to i64
  %58 = mul i64 %57, 7168
  %59 = add i64 %51, %58
  %60 = inttoptr i64 %59 to ptr
  %61 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %60, i16 0, i64 7168, i32 159744)
  %62 = zext i32 %47 to i64
  %63 = mul i64 %62, 14336
  %64 = add i64 %52, %63
  %65 = inttoptr i64 %64 to ptr
  %66 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %65, i16 0, i64 14336, i32 159744)
  %67 = icmp ult i32 %53, 224
  br i1 %67, label %68, label %147

68:                                               ; preds = %32
  %69 = mul i32 %53, 32
  %70 = mul i32 %55, 8
  %71 = add i32 %69, %70
  %72 = lshr i32 %71, 1
  %73 = mul i32 %72, 4
  %74 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %73, i32 0, i32 0)
  %75 = bitcast <4 x i32> %74 to <8 x bfloat>
  %76 = fpext <8 x bfloat> %75 to <8 x float>
  %77 = extractelement <8 x float> %76, i64 0
  %78 = call float @llvm.fabs.f32(float %77)
  %79 = call float @llvm.maximum.f32(float %78, float 0.000000e+00)
  %80 = extractelement <8 x float> %76, i64 1
  %81 = call float @llvm.fabs.f32(float %80)
  %82 = call float @llvm.maximum.f32(float %79, float %81)
  %83 = extractelement <8 x float> %76, i64 2
  %84 = call float @llvm.fabs.f32(float %83)
  %85 = call float @llvm.maximum.f32(float %82, float %84)
  %86 = extractelement <8 x float> %76, i64 3
  %87 = call float @llvm.fabs.f32(float %86)
  %88 = call float @llvm.maximum.f32(float %85, float %87)
  %89 = extractelement <8 x float> %76, i64 4
  %90 = call float @llvm.fabs.f32(float %89)
  %91 = call float @llvm.maximum.f32(float %88, float %90)
  %92 = extractelement <8 x float> %76, i64 5
  %93 = call float @llvm.fabs.f32(float %92)
  %94 = call float @llvm.maximum.f32(float %91, float %93)
  %95 = extractelement <8 x float> %76, i64 6
  %96 = call float @llvm.fabs.f32(float %95)
  %97 = call float @llvm.maximum.f32(float %94, float %96)
  %98 = extractelement <8 x float> %76, i64 7
  %99 = call float @llvm.fabs.f32(float %98)
  %100 = call float @llvm.maximum.f32(float %97, float %99)
  %101 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %102 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %101)
  %103 = add i32 %102, 32
  %104 = and i32 %103, -32
  %105 = xor i32 %102, 1
  %106 = icmp slt i32 %105, %104
  %107 = select i1 %106, i32 %105, i32 %102
  %108 = shl i32 %107, 2
  %109 = bitcast float %100 to i32
  %110 = call i32 @llvm.amdgcn.ds.bpermute(i32 %108, i32 %109)
  %111 = bitcast i32 %110 to float
  %112 = call float @llvm.maximum.f32(float %100, float %111)
  %113 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %114 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %113)
  %115 = add i32 %114, 32
  %116 = and i32 %115, -32
  %117 = xor i32 %114, 2
  %118 = icmp slt i32 %117, %116
  %119 = select i1 %118, i32 %117, i32 %114
  %120 = shl i32 %119, 2
  %121 = bitcast float %112 to i32
  %122 = call i32 @llvm.amdgcn.ds.bpermute(i32 %120, i32 %121)
  %123 = bitcast i32 %122 to float
  %124 = call float @llvm.maximum.f32(float %112, float %123)
  %125 = fmul float %124, 0x3F624924A0000000
  %126 = bitcast float %125 to i32
  %127 = and i32 %126, 8388607
  %128 = lshr i32 %126, 23
  %129 = and i32 %128, 255
  %130 = icmp ne i32 %127, 0
  %131 = add i32 %129, 1
  %132 = select i1 %130, i32 %131, i32 %129
  %133 = call i32 @llvm.smax.i32(i32 %132, i32 0)
  %134 = call i32 @llvm.smin.i32(i32 %133, i32 255)
  %135 = shl i32 %134, 23
  %136 = bitcast i32 %135 to float
  %137 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %75, float %136)
  %138 = udiv i32 %53, 4
  %139 = mul i32 %138, 4
  %140 = sub i32 %53, %139
  %141 = trunc i32 %134 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %137, ptr addrspace(8) %61, i32 %71, i32 0, i32 0)
  br i1 %56, label %142, label %146

142:                                              ; preds = %68
  %143 = add i32 %46, %138
  %144 = mul i32 %143, 4
  %145 = add i32 %144, %140
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %141, ptr addrspace(8) %50, i32 %145, i32 0, i32 0)
  br label %146

146:                                              ; preds = %142, %68
  br label %147

147:                                              ; preds = %146, %32
  %148 = add i32 %53, 8
  %149 = icmp ult i32 %148, 224
  br i1 %149, label %150, label %229

150:                                              ; preds = %147
  %151 = mul i32 %148, 32
  %152 = mul i32 %55, 8
  %153 = add i32 %151, %152
  %154 = lshr i32 %153, 1
  %155 = mul i32 %154, 4
  %156 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %155, i32 0, i32 0)
  %157 = bitcast <4 x i32> %156 to <8 x bfloat>
  %158 = fpext <8 x bfloat> %157 to <8 x float>
  %159 = extractelement <8 x float> %158, i64 0
  %160 = call float @llvm.fabs.f32(float %159)
  %161 = call float @llvm.maximum.f32(float %160, float 0.000000e+00)
  %162 = extractelement <8 x float> %158, i64 1
  %163 = call float @llvm.fabs.f32(float %162)
  %164 = call float @llvm.maximum.f32(float %161, float %163)
  %165 = extractelement <8 x float> %158, i64 2
  %166 = call float @llvm.fabs.f32(float %165)
  %167 = call float @llvm.maximum.f32(float %164, float %166)
  %168 = extractelement <8 x float> %158, i64 3
  %169 = call float @llvm.fabs.f32(float %168)
  %170 = call float @llvm.maximum.f32(float %167, float %169)
  %171 = extractelement <8 x float> %158, i64 4
  %172 = call float @llvm.fabs.f32(float %171)
  %173 = call float @llvm.maximum.f32(float %170, float %172)
  %174 = extractelement <8 x float> %158, i64 5
  %175 = call float @llvm.fabs.f32(float %174)
  %176 = call float @llvm.maximum.f32(float %173, float %175)
  %177 = extractelement <8 x float> %158, i64 6
  %178 = call float @llvm.fabs.f32(float %177)
  %179 = call float @llvm.maximum.f32(float %176, float %178)
  %180 = extractelement <8 x float> %158, i64 7
  %181 = call float @llvm.fabs.f32(float %180)
  %182 = call float @llvm.maximum.f32(float %179, float %181)
  %183 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %184 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %183)
  %185 = add i32 %184, 32
  %186 = and i32 %185, -32
  %187 = xor i32 %184, 1
  %188 = icmp slt i32 %187, %186
  %189 = select i1 %188, i32 %187, i32 %184
  %190 = shl i32 %189, 2
  %191 = bitcast float %182 to i32
  %192 = call i32 @llvm.amdgcn.ds.bpermute(i32 %190, i32 %191)
  %193 = bitcast i32 %192 to float
  %194 = call float @llvm.maximum.f32(float %182, float %193)
  %195 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %196 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %195)
  %197 = add i32 %196, 32
  %198 = and i32 %197, -32
  %199 = xor i32 %196, 2
  %200 = icmp slt i32 %199, %198
  %201 = select i1 %200, i32 %199, i32 %196
  %202 = shl i32 %201, 2
  %203 = bitcast float %194 to i32
  %204 = call i32 @llvm.amdgcn.ds.bpermute(i32 %202, i32 %203)
  %205 = bitcast i32 %204 to float
  %206 = call float @llvm.maximum.f32(float %194, float %205)
  %207 = fmul float %206, 0x3F624924A0000000
  %208 = bitcast float %207 to i32
  %209 = and i32 %208, 8388607
  %210 = lshr i32 %208, 23
  %211 = and i32 %210, 255
  %212 = icmp ne i32 %209, 0
  %213 = add i32 %211, 1
  %214 = select i1 %212, i32 %213, i32 %211
  %215 = call i32 @llvm.smax.i32(i32 %214, i32 0)
  %216 = call i32 @llvm.smin.i32(i32 %215, i32 255)
  %217 = shl i32 %216, 23
  %218 = bitcast i32 %217 to float
  %219 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %157, float %218)
  %220 = udiv i32 %148, 4
  %221 = mul i32 %220, 4
  %222 = sub i32 %148, %221
  %223 = trunc i32 %216 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %219, ptr addrspace(8) %61, i32 %153, i32 0, i32 0)
  br i1 %56, label %224, label %228

224:                                              ; preds = %150
  %225 = add i32 %46, %220
  %226 = mul i32 %225, 4
  %227 = add i32 %226, %222
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %223, ptr addrspace(8) %50, i32 %227, i32 0, i32 0)
  br label %228

228:                                              ; preds = %224, %150
  br label %229

229:                                              ; preds = %228, %147
  %230 = add i32 %53, 16
  %231 = icmp ult i32 %230, 224
  br i1 %231, label %232, label %311

232:                                              ; preds = %229
  %233 = mul i32 %230, 32
  %234 = mul i32 %55, 8
  %235 = add i32 %233, %234
  %236 = lshr i32 %235, 1
  %237 = mul i32 %236, 4
  %238 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %237, i32 0, i32 0)
  %239 = bitcast <4 x i32> %238 to <8 x bfloat>
  %240 = fpext <8 x bfloat> %239 to <8 x float>
  %241 = extractelement <8 x float> %240, i64 0
  %242 = call float @llvm.fabs.f32(float %241)
  %243 = call float @llvm.maximum.f32(float %242, float 0.000000e+00)
  %244 = extractelement <8 x float> %240, i64 1
  %245 = call float @llvm.fabs.f32(float %244)
  %246 = call float @llvm.maximum.f32(float %243, float %245)
  %247 = extractelement <8 x float> %240, i64 2
  %248 = call float @llvm.fabs.f32(float %247)
  %249 = call float @llvm.maximum.f32(float %246, float %248)
  %250 = extractelement <8 x float> %240, i64 3
  %251 = call float @llvm.fabs.f32(float %250)
  %252 = call float @llvm.maximum.f32(float %249, float %251)
  %253 = extractelement <8 x float> %240, i64 4
  %254 = call float @llvm.fabs.f32(float %253)
  %255 = call float @llvm.maximum.f32(float %252, float %254)
  %256 = extractelement <8 x float> %240, i64 5
  %257 = call float @llvm.fabs.f32(float %256)
  %258 = call float @llvm.maximum.f32(float %255, float %257)
  %259 = extractelement <8 x float> %240, i64 6
  %260 = call float @llvm.fabs.f32(float %259)
  %261 = call float @llvm.maximum.f32(float %258, float %260)
  %262 = extractelement <8 x float> %240, i64 7
  %263 = call float @llvm.fabs.f32(float %262)
  %264 = call float @llvm.maximum.f32(float %261, float %263)
  %265 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %266 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %265)
  %267 = add i32 %266, 32
  %268 = and i32 %267, -32
  %269 = xor i32 %266, 1
  %270 = icmp slt i32 %269, %268
  %271 = select i1 %270, i32 %269, i32 %266
  %272 = shl i32 %271, 2
  %273 = bitcast float %264 to i32
  %274 = call i32 @llvm.amdgcn.ds.bpermute(i32 %272, i32 %273)
  %275 = bitcast i32 %274 to float
  %276 = call float @llvm.maximum.f32(float %264, float %275)
  %277 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %278 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %277)
  %279 = add i32 %278, 32
  %280 = and i32 %279, -32
  %281 = xor i32 %278, 2
  %282 = icmp slt i32 %281, %280
  %283 = select i1 %282, i32 %281, i32 %278
  %284 = shl i32 %283, 2
  %285 = bitcast float %276 to i32
  %286 = call i32 @llvm.amdgcn.ds.bpermute(i32 %284, i32 %285)
  %287 = bitcast i32 %286 to float
  %288 = call float @llvm.maximum.f32(float %276, float %287)
  %289 = fmul float %288, 0x3F624924A0000000
  %290 = bitcast float %289 to i32
  %291 = and i32 %290, 8388607
  %292 = lshr i32 %290, 23
  %293 = and i32 %292, 255
  %294 = icmp ne i32 %291, 0
  %295 = add i32 %293, 1
  %296 = select i1 %294, i32 %295, i32 %293
  %297 = call i32 @llvm.smax.i32(i32 %296, i32 0)
  %298 = call i32 @llvm.smin.i32(i32 %297, i32 255)
  %299 = shl i32 %298, 23
  %300 = bitcast i32 %299 to float
  %301 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %239, float %300)
  %302 = udiv i32 %230, 4
  %303 = mul i32 %302, 4
  %304 = sub i32 %230, %303
  %305 = trunc i32 %298 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %301, ptr addrspace(8) %61, i32 %235, i32 0, i32 0)
  br i1 %56, label %306, label %310

306:                                              ; preds = %232
  %307 = add i32 %46, %302
  %308 = mul i32 %307, 4
  %309 = add i32 %308, %304
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %305, ptr addrspace(8) %50, i32 %309, i32 0, i32 0)
  br label %310

310:                                              ; preds = %306, %232
  br label %311

311:                                              ; preds = %310, %229
  %312 = add i32 %53, 24
  %313 = icmp ult i32 %312, 224
  br i1 %313, label %314, label %393

314:                                              ; preds = %311
  %315 = mul i32 %312, 32
  %316 = mul i32 %55, 8
  %317 = add i32 %315, %316
  %318 = lshr i32 %317, 1
  %319 = mul i32 %318, 4
  %320 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %319, i32 0, i32 0)
  %321 = bitcast <4 x i32> %320 to <8 x bfloat>
  %322 = fpext <8 x bfloat> %321 to <8 x float>
  %323 = extractelement <8 x float> %322, i64 0
  %324 = call float @llvm.fabs.f32(float %323)
  %325 = call float @llvm.maximum.f32(float %324, float 0.000000e+00)
  %326 = extractelement <8 x float> %322, i64 1
  %327 = call float @llvm.fabs.f32(float %326)
  %328 = call float @llvm.maximum.f32(float %325, float %327)
  %329 = extractelement <8 x float> %322, i64 2
  %330 = call float @llvm.fabs.f32(float %329)
  %331 = call float @llvm.maximum.f32(float %328, float %330)
  %332 = extractelement <8 x float> %322, i64 3
  %333 = call float @llvm.fabs.f32(float %332)
  %334 = call float @llvm.maximum.f32(float %331, float %333)
  %335 = extractelement <8 x float> %322, i64 4
  %336 = call float @llvm.fabs.f32(float %335)
  %337 = call float @llvm.maximum.f32(float %334, float %336)
  %338 = extractelement <8 x float> %322, i64 5
  %339 = call float @llvm.fabs.f32(float %338)
  %340 = call float @llvm.maximum.f32(float %337, float %339)
  %341 = extractelement <8 x float> %322, i64 6
  %342 = call float @llvm.fabs.f32(float %341)
  %343 = call float @llvm.maximum.f32(float %340, float %342)
  %344 = extractelement <8 x float> %322, i64 7
  %345 = call float @llvm.fabs.f32(float %344)
  %346 = call float @llvm.maximum.f32(float %343, float %345)
  %347 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %348 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %347)
  %349 = add i32 %348, 32
  %350 = and i32 %349, -32
  %351 = xor i32 %348, 1
  %352 = icmp slt i32 %351, %350
  %353 = select i1 %352, i32 %351, i32 %348
  %354 = shl i32 %353, 2
  %355 = bitcast float %346 to i32
  %356 = call i32 @llvm.amdgcn.ds.bpermute(i32 %354, i32 %355)
  %357 = bitcast i32 %356 to float
  %358 = call float @llvm.maximum.f32(float %346, float %357)
  %359 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %360 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %359)
  %361 = add i32 %360, 32
  %362 = and i32 %361, -32
  %363 = xor i32 %360, 2
  %364 = icmp slt i32 %363, %362
  %365 = select i1 %364, i32 %363, i32 %360
  %366 = shl i32 %365, 2
  %367 = bitcast float %358 to i32
  %368 = call i32 @llvm.amdgcn.ds.bpermute(i32 %366, i32 %367)
  %369 = bitcast i32 %368 to float
  %370 = call float @llvm.maximum.f32(float %358, float %369)
  %371 = fmul float %370, 0x3F624924A0000000
  %372 = bitcast float %371 to i32
  %373 = and i32 %372, 8388607
  %374 = lshr i32 %372, 23
  %375 = and i32 %374, 255
  %376 = icmp ne i32 %373, 0
  %377 = add i32 %375, 1
  %378 = select i1 %376, i32 %377, i32 %375
  %379 = call i32 @llvm.smax.i32(i32 %378, i32 0)
  %380 = call i32 @llvm.smin.i32(i32 %379, i32 255)
  %381 = shl i32 %380, 23
  %382 = bitcast i32 %381 to float
  %383 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %321, float %382)
  %384 = udiv i32 %312, 4
  %385 = mul i32 %384, 4
  %386 = sub i32 %312, %385
  %387 = trunc i32 %380 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %383, ptr addrspace(8) %61, i32 %317, i32 0, i32 0)
  br i1 %56, label %388, label %392

388:                                              ; preds = %314
  %389 = add i32 %46, %384
  %390 = mul i32 %389, 4
  %391 = add i32 %390, %386
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %387, ptr addrspace(8) %50, i32 %391, i32 0, i32 0)
  br label %392

392:                                              ; preds = %388, %314
  br label %393

393:                                              ; preds = %392, %311
  %394 = add i32 %53, 32
  %395 = icmp ult i32 %394, 224
  br i1 %395, label %396, label %475

396:                                              ; preds = %393
  %397 = mul i32 %394, 32
  %398 = mul i32 %55, 8
  %399 = add i32 %397, %398
  %400 = lshr i32 %399, 1
  %401 = mul i32 %400, 4
  %402 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %401, i32 0, i32 0)
  %403 = bitcast <4 x i32> %402 to <8 x bfloat>
  %404 = fpext <8 x bfloat> %403 to <8 x float>
  %405 = extractelement <8 x float> %404, i64 0
  %406 = call float @llvm.fabs.f32(float %405)
  %407 = call float @llvm.maximum.f32(float %406, float 0.000000e+00)
  %408 = extractelement <8 x float> %404, i64 1
  %409 = call float @llvm.fabs.f32(float %408)
  %410 = call float @llvm.maximum.f32(float %407, float %409)
  %411 = extractelement <8 x float> %404, i64 2
  %412 = call float @llvm.fabs.f32(float %411)
  %413 = call float @llvm.maximum.f32(float %410, float %412)
  %414 = extractelement <8 x float> %404, i64 3
  %415 = call float @llvm.fabs.f32(float %414)
  %416 = call float @llvm.maximum.f32(float %413, float %415)
  %417 = extractelement <8 x float> %404, i64 4
  %418 = call float @llvm.fabs.f32(float %417)
  %419 = call float @llvm.maximum.f32(float %416, float %418)
  %420 = extractelement <8 x float> %404, i64 5
  %421 = call float @llvm.fabs.f32(float %420)
  %422 = call float @llvm.maximum.f32(float %419, float %421)
  %423 = extractelement <8 x float> %404, i64 6
  %424 = call float @llvm.fabs.f32(float %423)
  %425 = call float @llvm.maximum.f32(float %422, float %424)
  %426 = extractelement <8 x float> %404, i64 7
  %427 = call float @llvm.fabs.f32(float %426)
  %428 = call float @llvm.maximum.f32(float %425, float %427)
  %429 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %430 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %429)
  %431 = add i32 %430, 32
  %432 = and i32 %431, -32
  %433 = xor i32 %430, 1
  %434 = icmp slt i32 %433, %432
  %435 = select i1 %434, i32 %433, i32 %430
  %436 = shl i32 %435, 2
  %437 = bitcast float %428 to i32
  %438 = call i32 @llvm.amdgcn.ds.bpermute(i32 %436, i32 %437)
  %439 = bitcast i32 %438 to float
  %440 = call float @llvm.maximum.f32(float %428, float %439)
  %441 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %442 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %441)
  %443 = add i32 %442, 32
  %444 = and i32 %443, -32
  %445 = xor i32 %442, 2
  %446 = icmp slt i32 %445, %444
  %447 = select i1 %446, i32 %445, i32 %442
  %448 = shl i32 %447, 2
  %449 = bitcast float %440 to i32
  %450 = call i32 @llvm.amdgcn.ds.bpermute(i32 %448, i32 %449)
  %451 = bitcast i32 %450 to float
  %452 = call float @llvm.maximum.f32(float %440, float %451)
  %453 = fmul float %452, 0x3F624924A0000000
  %454 = bitcast float %453 to i32
  %455 = and i32 %454, 8388607
  %456 = lshr i32 %454, 23
  %457 = and i32 %456, 255
  %458 = icmp ne i32 %455, 0
  %459 = add i32 %457, 1
  %460 = select i1 %458, i32 %459, i32 %457
  %461 = call i32 @llvm.smax.i32(i32 %460, i32 0)
  %462 = call i32 @llvm.smin.i32(i32 %461, i32 255)
  %463 = shl i32 %462, 23
  %464 = bitcast i32 %463 to float
  %465 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %403, float %464)
  %466 = udiv i32 %394, 4
  %467 = mul i32 %466, 4
  %468 = sub i32 %394, %467
  %469 = trunc i32 %462 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %465, ptr addrspace(8) %61, i32 %399, i32 0, i32 0)
  br i1 %56, label %470, label %474

470:                                              ; preds = %396
  %471 = add i32 %46, %466
  %472 = mul i32 %471, 4
  %473 = add i32 %472, %468
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %469, ptr addrspace(8) %50, i32 %473, i32 0, i32 0)
  br label %474

474:                                              ; preds = %470, %396
  br label %475

475:                                              ; preds = %474, %393
  %476 = add i32 %53, 40
  %477 = icmp ult i32 %476, 224
  br i1 %477, label %478, label %557

478:                                              ; preds = %475
  %479 = mul i32 %476, 32
  %480 = mul i32 %55, 8
  %481 = add i32 %479, %480
  %482 = lshr i32 %481, 1
  %483 = mul i32 %482, 4
  %484 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %483, i32 0, i32 0)
  %485 = bitcast <4 x i32> %484 to <8 x bfloat>
  %486 = fpext <8 x bfloat> %485 to <8 x float>
  %487 = extractelement <8 x float> %486, i64 0
  %488 = call float @llvm.fabs.f32(float %487)
  %489 = call float @llvm.maximum.f32(float %488, float 0.000000e+00)
  %490 = extractelement <8 x float> %486, i64 1
  %491 = call float @llvm.fabs.f32(float %490)
  %492 = call float @llvm.maximum.f32(float %489, float %491)
  %493 = extractelement <8 x float> %486, i64 2
  %494 = call float @llvm.fabs.f32(float %493)
  %495 = call float @llvm.maximum.f32(float %492, float %494)
  %496 = extractelement <8 x float> %486, i64 3
  %497 = call float @llvm.fabs.f32(float %496)
  %498 = call float @llvm.maximum.f32(float %495, float %497)
  %499 = extractelement <8 x float> %486, i64 4
  %500 = call float @llvm.fabs.f32(float %499)
  %501 = call float @llvm.maximum.f32(float %498, float %500)
  %502 = extractelement <8 x float> %486, i64 5
  %503 = call float @llvm.fabs.f32(float %502)
  %504 = call float @llvm.maximum.f32(float %501, float %503)
  %505 = extractelement <8 x float> %486, i64 6
  %506 = call float @llvm.fabs.f32(float %505)
  %507 = call float @llvm.maximum.f32(float %504, float %506)
  %508 = extractelement <8 x float> %486, i64 7
  %509 = call float @llvm.fabs.f32(float %508)
  %510 = call float @llvm.maximum.f32(float %507, float %509)
  %511 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %512 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %511)
  %513 = add i32 %512, 32
  %514 = and i32 %513, -32
  %515 = xor i32 %512, 1
  %516 = icmp slt i32 %515, %514
  %517 = select i1 %516, i32 %515, i32 %512
  %518 = shl i32 %517, 2
  %519 = bitcast float %510 to i32
  %520 = call i32 @llvm.amdgcn.ds.bpermute(i32 %518, i32 %519)
  %521 = bitcast i32 %520 to float
  %522 = call float @llvm.maximum.f32(float %510, float %521)
  %523 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %524 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %523)
  %525 = add i32 %524, 32
  %526 = and i32 %525, -32
  %527 = xor i32 %524, 2
  %528 = icmp slt i32 %527, %526
  %529 = select i1 %528, i32 %527, i32 %524
  %530 = shl i32 %529, 2
  %531 = bitcast float %522 to i32
  %532 = call i32 @llvm.amdgcn.ds.bpermute(i32 %530, i32 %531)
  %533 = bitcast i32 %532 to float
  %534 = call float @llvm.maximum.f32(float %522, float %533)
  %535 = fmul float %534, 0x3F624924A0000000
  %536 = bitcast float %535 to i32
  %537 = and i32 %536, 8388607
  %538 = lshr i32 %536, 23
  %539 = and i32 %538, 255
  %540 = icmp ne i32 %537, 0
  %541 = add i32 %539, 1
  %542 = select i1 %540, i32 %541, i32 %539
  %543 = call i32 @llvm.smax.i32(i32 %542, i32 0)
  %544 = call i32 @llvm.smin.i32(i32 %543, i32 255)
  %545 = shl i32 %544, 23
  %546 = bitcast i32 %545 to float
  %547 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %485, float %546)
  %548 = udiv i32 %476, 4
  %549 = mul i32 %548, 4
  %550 = sub i32 %476, %549
  %551 = trunc i32 %544 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %547, ptr addrspace(8) %61, i32 %481, i32 0, i32 0)
  br i1 %56, label %552, label %556

552:                                              ; preds = %478
  %553 = add i32 %46, %548
  %554 = mul i32 %553, 4
  %555 = add i32 %554, %550
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %551, ptr addrspace(8) %50, i32 %555, i32 0, i32 0)
  br label %556

556:                                              ; preds = %552, %478
  br label %557

557:                                              ; preds = %556, %475
  %558 = add i32 %53, 48
  %559 = icmp ult i32 %558, 224
  br i1 %559, label %560, label %639

560:                                              ; preds = %557
  %561 = mul i32 %558, 32
  %562 = mul i32 %55, 8
  %563 = add i32 %561, %562
  %564 = lshr i32 %563, 1
  %565 = mul i32 %564, 4
  %566 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %565, i32 0, i32 0)
  %567 = bitcast <4 x i32> %566 to <8 x bfloat>
  %568 = fpext <8 x bfloat> %567 to <8 x float>
  %569 = extractelement <8 x float> %568, i64 0
  %570 = call float @llvm.fabs.f32(float %569)
  %571 = call float @llvm.maximum.f32(float %570, float 0.000000e+00)
  %572 = extractelement <8 x float> %568, i64 1
  %573 = call float @llvm.fabs.f32(float %572)
  %574 = call float @llvm.maximum.f32(float %571, float %573)
  %575 = extractelement <8 x float> %568, i64 2
  %576 = call float @llvm.fabs.f32(float %575)
  %577 = call float @llvm.maximum.f32(float %574, float %576)
  %578 = extractelement <8 x float> %568, i64 3
  %579 = call float @llvm.fabs.f32(float %578)
  %580 = call float @llvm.maximum.f32(float %577, float %579)
  %581 = extractelement <8 x float> %568, i64 4
  %582 = call float @llvm.fabs.f32(float %581)
  %583 = call float @llvm.maximum.f32(float %580, float %582)
  %584 = extractelement <8 x float> %568, i64 5
  %585 = call float @llvm.fabs.f32(float %584)
  %586 = call float @llvm.maximum.f32(float %583, float %585)
  %587 = extractelement <8 x float> %568, i64 6
  %588 = call float @llvm.fabs.f32(float %587)
  %589 = call float @llvm.maximum.f32(float %586, float %588)
  %590 = extractelement <8 x float> %568, i64 7
  %591 = call float @llvm.fabs.f32(float %590)
  %592 = call float @llvm.maximum.f32(float %589, float %591)
  %593 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %594 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %593)
  %595 = add i32 %594, 32
  %596 = and i32 %595, -32
  %597 = xor i32 %594, 1
  %598 = icmp slt i32 %597, %596
  %599 = select i1 %598, i32 %597, i32 %594
  %600 = shl i32 %599, 2
  %601 = bitcast float %592 to i32
  %602 = call i32 @llvm.amdgcn.ds.bpermute(i32 %600, i32 %601)
  %603 = bitcast i32 %602 to float
  %604 = call float @llvm.maximum.f32(float %592, float %603)
  %605 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %606 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %605)
  %607 = add i32 %606, 32
  %608 = and i32 %607, -32
  %609 = xor i32 %606, 2
  %610 = icmp slt i32 %609, %608
  %611 = select i1 %610, i32 %609, i32 %606
  %612 = shl i32 %611, 2
  %613 = bitcast float %604 to i32
  %614 = call i32 @llvm.amdgcn.ds.bpermute(i32 %612, i32 %613)
  %615 = bitcast i32 %614 to float
  %616 = call float @llvm.maximum.f32(float %604, float %615)
  %617 = fmul float %616, 0x3F624924A0000000
  %618 = bitcast float %617 to i32
  %619 = and i32 %618, 8388607
  %620 = lshr i32 %618, 23
  %621 = and i32 %620, 255
  %622 = icmp ne i32 %619, 0
  %623 = add i32 %621, 1
  %624 = select i1 %622, i32 %623, i32 %621
  %625 = call i32 @llvm.smax.i32(i32 %624, i32 0)
  %626 = call i32 @llvm.smin.i32(i32 %625, i32 255)
  %627 = shl i32 %626, 23
  %628 = bitcast i32 %627 to float
  %629 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %567, float %628)
  %630 = udiv i32 %558, 4
  %631 = mul i32 %630, 4
  %632 = sub i32 %558, %631
  %633 = trunc i32 %626 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %629, ptr addrspace(8) %61, i32 %563, i32 0, i32 0)
  br i1 %56, label %634, label %638

634:                                              ; preds = %560
  %635 = add i32 %46, %630
  %636 = mul i32 %635, 4
  %637 = add i32 %636, %632
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %633, ptr addrspace(8) %50, i32 %637, i32 0, i32 0)
  br label %638

638:                                              ; preds = %634, %560
  br label %639

639:                                              ; preds = %638, %557
  %640 = add i32 %53, 56
  %641 = icmp ult i32 %640, 224
  br i1 %641, label %642, label %721

642:                                              ; preds = %639
  %643 = mul i32 %640, 32
  %644 = mul i32 %55, 8
  %645 = add i32 %643, %644
  %646 = lshr i32 %645, 1
  %647 = mul i32 %646, 4
  %648 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %647, i32 0, i32 0)
  %649 = bitcast <4 x i32> %648 to <8 x bfloat>
  %650 = fpext <8 x bfloat> %649 to <8 x float>
  %651 = extractelement <8 x float> %650, i64 0
  %652 = call float @llvm.fabs.f32(float %651)
  %653 = call float @llvm.maximum.f32(float %652, float 0.000000e+00)
  %654 = extractelement <8 x float> %650, i64 1
  %655 = call float @llvm.fabs.f32(float %654)
  %656 = call float @llvm.maximum.f32(float %653, float %655)
  %657 = extractelement <8 x float> %650, i64 2
  %658 = call float @llvm.fabs.f32(float %657)
  %659 = call float @llvm.maximum.f32(float %656, float %658)
  %660 = extractelement <8 x float> %650, i64 3
  %661 = call float @llvm.fabs.f32(float %660)
  %662 = call float @llvm.maximum.f32(float %659, float %661)
  %663 = extractelement <8 x float> %650, i64 4
  %664 = call float @llvm.fabs.f32(float %663)
  %665 = call float @llvm.maximum.f32(float %662, float %664)
  %666 = extractelement <8 x float> %650, i64 5
  %667 = call float @llvm.fabs.f32(float %666)
  %668 = call float @llvm.maximum.f32(float %665, float %667)
  %669 = extractelement <8 x float> %650, i64 6
  %670 = call float @llvm.fabs.f32(float %669)
  %671 = call float @llvm.maximum.f32(float %668, float %670)
  %672 = extractelement <8 x float> %650, i64 7
  %673 = call float @llvm.fabs.f32(float %672)
  %674 = call float @llvm.maximum.f32(float %671, float %673)
  %675 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %676 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %675)
  %677 = add i32 %676, 32
  %678 = and i32 %677, -32
  %679 = xor i32 %676, 1
  %680 = icmp slt i32 %679, %678
  %681 = select i1 %680, i32 %679, i32 %676
  %682 = shl i32 %681, 2
  %683 = bitcast float %674 to i32
  %684 = call i32 @llvm.amdgcn.ds.bpermute(i32 %682, i32 %683)
  %685 = bitcast i32 %684 to float
  %686 = call float @llvm.maximum.f32(float %674, float %685)
  %687 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %688 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %687)
  %689 = add i32 %688, 32
  %690 = and i32 %689, -32
  %691 = xor i32 %688, 2
  %692 = icmp slt i32 %691, %690
  %693 = select i1 %692, i32 %691, i32 %688
  %694 = shl i32 %693, 2
  %695 = bitcast float %686 to i32
  %696 = call i32 @llvm.amdgcn.ds.bpermute(i32 %694, i32 %695)
  %697 = bitcast i32 %696 to float
  %698 = call float @llvm.maximum.f32(float %686, float %697)
  %699 = fmul float %698, 0x3F624924A0000000
  %700 = bitcast float %699 to i32
  %701 = and i32 %700, 8388607
  %702 = lshr i32 %700, 23
  %703 = and i32 %702, 255
  %704 = icmp ne i32 %701, 0
  %705 = add i32 %703, 1
  %706 = select i1 %704, i32 %705, i32 %703
  %707 = call i32 @llvm.smax.i32(i32 %706, i32 0)
  %708 = call i32 @llvm.smin.i32(i32 %707, i32 255)
  %709 = shl i32 %708, 23
  %710 = bitcast i32 %709 to float
  %711 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %649, float %710)
  %712 = udiv i32 %640, 4
  %713 = mul i32 %712, 4
  %714 = sub i32 %640, %713
  %715 = trunc i32 %708 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %711, ptr addrspace(8) %61, i32 %645, i32 0, i32 0)
  br i1 %56, label %716, label %720

716:                                              ; preds = %642
  %717 = add i32 %46, %712
  %718 = mul i32 %717, 4
  %719 = add i32 %718, %714
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %715, ptr addrspace(8) %50, i32 %719, i32 0, i32 0)
  br label %720

720:                                              ; preds = %716, %642
  br label %721

721:                                              ; preds = %720, %639
  %722 = add i32 %53, 64
  %723 = icmp ult i32 %722, 224
  br i1 %723, label %724, label %803

724:                                              ; preds = %721
  %725 = mul i32 %722, 32
  %726 = mul i32 %55, 8
  %727 = add i32 %725, %726
  %728 = lshr i32 %727, 1
  %729 = mul i32 %728, 4
  %730 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %729, i32 0, i32 0)
  %731 = bitcast <4 x i32> %730 to <8 x bfloat>
  %732 = fpext <8 x bfloat> %731 to <8 x float>
  %733 = extractelement <8 x float> %732, i64 0
  %734 = call float @llvm.fabs.f32(float %733)
  %735 = call float @llvm.maximum.f32(float %734, float 0.000000e+00)
  %736 = extractelement <8 x float> %732, i64 1
  %737 = call float @llvm.fabs.f32(float %736)
  %738 = call float @llvm.maximum.f32(float %735, float %737)
  %739 = extractelement <8 x float> %732, i64 2
  %740 = call float @llvm.fabs.f32(float %739)
  %741 = call float @llvm.maximum.f32(float %738, float %740)
  %742 = extractelement <8 x float> %732, i64 3
  %743 = call float @llvm.fabs.f32(float %742)
  %744 = call float @llvm.maximum.f32(float %741, float %743)
  %745 = extractelement <8 x float> %732, i64 4
  %746 = call float @llvm.fabs.f32(float %745)
  %747 = call float @llvm.maximum.f32(float %744, float %746)
  %748 = extractelement <8 x float> %732, i64 5
  %749 = call float @llvm.fabs.f32(float %748)
  %750 = call float @llvm.maximum.f32(float %747, float %749)
  %751 = extractelement <8 x float> %732, i64 6
  %752 = call float @llvm.fabs.f32(float %751)
  %753 = call float @llvm.maximum.f32(float %750, float %752)
  %754 = extractelement <8 x float> %732, i64 7
  %755 = call float @llvm.fabs.f32(float %754)
  %756 = call float @llvm.maximum.f32(float %753, float %755)
  %757 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %758 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %757)
  %759 = add i32 %758, 32
  %760 = and i32 %759, -32
  %761 = xor i32 %758, 1
  %762 = icmp slt i32 %761, %760
  %763 = select i1 %762, i32 %761, i32 %758
  %764 = shl i32 %763, 2
  %765 = bitcast float %756 to i32
  %766 = call i32 @llvm.amdgcn.ds.bpermute(i32 %764, i32 %765)
  %767 = bitcast i32 %766 to float
  %768 = call float @llvm.maximum.f32(float %756, float %767)
  %769 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %770 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %769)
  %771 = add i32 %770, 32
  %772 = and i32 %771, -32
  %773 = xor i32 %770, 2
  %774 = icmp slt i32 %773, %772
  %775 = select i1 %774, i32 %773, i32 %770
  %776 = shl i32 %775, 2
  %777 = bitcast float %768 to i32
  %778 = call i32 @llvm.amdgcn.ds.bpermute(i32 %776, i32 %777)
  %779 = bitcast i32 %778 to float
  %780 = call float @llvm.maximum.f32(float %768, float %779)
  %781 = fmul float %780, 0x3F624924A0000000
  %782 = bitcast float %781 to i32
  %783 = and i32 %782, 8388607
  %784 = lshr i32 %782, 23
  %785 = and i32 %784, 255
  %786 = icmp ne i32 %783, 0
  %787 = add i32 %785, 1
  %788 = select i1 %786, i32 %787, i32 %785
  %789 = call i32 @llvm.smax.i32(i32 %788, i32 0)
  %790 = call i32 @llvm.smin.i32(i32 %789, i32 255)
  %791 = shl i32 %790, 23
  %792 = bitcast i32 %791 to float
  %793 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %731, float %792)
  %794 = udiv i32 %722, 4
  %795 = mul i32 %794, 4
  %796 = sub i32 %722, %795
  %797 = trunc i32 %790 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %793, ptr addrspace(8) %61, i32 %727, i32 0, i32 0)
  br i1 %56, label %798, label %802

798:                                              ; preds = %724
  %799 = add i32 %46, %794
  %800 = mul i32 %799, 4
  %801 = add i32 %800, %796
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %797, ptr addrspace(8) %50, i32 %801, i32 0, i32 0)
  br label %802

802:                                              ; preds = %798, %724
  br label %803

803:                                              ; preds = %802, %721
  %804 = add i32 %53, 72
  %805 = icmp ult i32 %804, 224
  br i1 %805, label %806, label %885

806:                                              ; preds = %803
  %807 = mul i32 %804, 32
  %808 = mul i32 %55, 8
  %809 = add i32 %807, %808
  %810 = lshr i32 %809, 1
  %811 = mul i32 %810, 4
  %812 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %811, i32 0, i32 0)
  %813 = bitcast <4 x i32> %812 to <8 x bfloat>
  %814 = fpext <8 x bfloat> %813 to <8 x float>
  %815 = extractelement <8 x float> %814, i64 0
  %816 = call float @llvm.fabs.f32(float %815)
  %817 = call float @llvm.maximum.f32(float %816, float 0.000000e+00)
  %818 = extractelement <8 x float> %814, i64 1
  %819 = call float @llvm.fabs.f32(float %818)
  %820 = call float @llvm.maximum.f32(float %817, float %819)
  %821 = extractelement <8 x float> %814, i64 2
  %822 = call float @llvm.fabs.f32(float %821)
  %823 = call float @llvm.maximum.f32(float %820, float %822)
  %824 = extractelement <8 x float> %814, i64 3
  %825 = call float @llvm.fabs.f32(float %824)
  %826 = call float @llvm.maximum.f32(float %823, float %825)
  %827 = extractelement <8 x float> %814, i64 4
  %828 = call float @llvm.fabs.f32(float %827)
  %829 = call float @llvm.maximum.f32(float %826, float %828)
  %830 = extractelement <8 x float> %814, i64 5
  %831 = call float @llvm.fabs.f32(float %830)
  %832 = call float @llvm.maximum.f32(float %829, float %831)
  %833 = extractelement <8 x float> %814, i64 6
  %834 = call float @llvm.fabs.f32(float %833)
  %835 = call float @llvm.maximum.f32(float %832, float %834)
  %836 = extractelement <8 x float> %814, i64 7
  %837 = call float @llvm.fabs.f32(float %836)
  %838 = call float @llvm.maximum.f32(float %835, float %837)
  %839 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %840 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %839)
  %841 = add i32 %840, 32
  %842 = and i32 %841, -32
  %843 = xor i32 %840, 1
  %844 = icmp slt i32 %843, %842
  %845 = select i1 %844, i32 %843, i32 %840
  %846 = shl i32 %845, 2
  %847 = bitcast float %838 to i32
  %848 = call i32 @llvm.amdgcn.ds.bpermute(i32 %846, i32 %847)
  %849 = bitcast i32 %848 to float
  %850 = call float @llvm.maximum.f32(float %838, float %849)
  %851 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %852 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %851)
  %853 = add i32 %852, 32
  %854 = and i32 %853, -32
  %855 = xor i32 %852, 2
  %856 = icmp slt i32 %855, %854
  %857 = select i1 %856, i32 %855, i32 %852
  %858 = shl i32 %857, 2
  %859 = bitcast float %850 to i32
  %860 = call i32 @llvm.amdgcn.ds.bpermute(i32 %858, i32 %859)
  %861 = bitcast i32 %860 to float
  %862 = call float @llvm.maximum.f32(float %850, float %861)
  %863 = fmul float %862, 0x3F624924A0000000
  %864 = bitcast float %863 to i32
  %865 = and i32 %864, 8388607
  %866 = lshr i32 %864, 23
  %867 = and i32 %866, 255
  %868 = icmp ne i32 %865, 0
  %869 = add i32 %867, 1
  %870 = select i1 %868, i32 %869, i32 %867
  %871 = call i32 @llvm.smax.i32(i32 %870, i32 0)
  %872 = call i32 @llvm.smin.i32(i32 %871, i32 255)
  %873 = shl i32 %872, 23
  %874 = bitcast i32 %873 to float
  %875 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %813, float %874)
  %876 = udiv i32 %804, 4
  %877 = mul i32 %876, 4
  %878 = sub i32 %804, %877
  %879 = trunc i32 %872 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %875, ptr addrspace(8) %61, i32 %809, i32 0, i32 0)
  br i1 %56, label %880, label %884

880:                                              ; preds = %806
  %881 = add i32 %46, %876
  %882 = mul i32 %881, 4
  %883 = add i32 %882, %878
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %879, ptr addrspace(8) %50, i32 %883, i32 0, i32 0)
  br label %884

884:                                              ; preds = %880, %806
  br label %885

885:                                              ; preds = %884, %803
  %886 = add i32 %53, 80
  %887 = icmp ult i32 %886, 224
  br i1 %887, label %888, label %967

888:                                              ; preds = %885
  %889 = mul i32 %886, 32
  %890 = mul i32 %55, 8
  %891 = add i32 %889, %890
  %892 = lshr i32 %891, 1
  %893 = mul i32 %892, 4
  %894 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %893, i32 0, i32 0)
  %895 = bitcast <4 x i32> %894 to <8 x bfloat>
  %896 = fpext <8 x bfloat> %895 to <8 x float>
  %897 = extractelement <8 x float> %896, i64 0
  %898 = call float @llvm.fabs.f32(float %897)
  %899 = call float @llvm.maximum.f32(float %898, float 0.000000e+00)
  %900 = extractelement <8 x float> %896, i64 1
  %901 = call float @llvm.fabs.f32(float %900)
  %902 = call float @llvm.maximum.f32(float %899, float %901)
  %903 = extractelement <8 x float> %896, i64 2
  %904 = call float @llvm.fabs.f32(float %903)
  %905 = call float @llvm.maximum.f32(float %902, float %904)
  %906 = extractelement <8 x float> %896, i64 3
  %907 = call float @llvm.fabs.f32(float %906)
  %908 = call float @llvm.maximum.f32(float %905, float %907)
  %909 = extractelement <8 x float> %896, i64 4
  %910 = call float @llvm.fabs.f32(float %909)
  %911 = call float @llvm.maximum.f32(float %908, float %910)
  %912 = extractelement <8 x float> %896, i64 5
  %913 = call float @llvm.fabs.f32(float %912)
  %914 = call float @llvm.maximum.f32(float %911, float %913)
  %915 = extractelement <8 x float> %896, i64 6
  %916 = call float @llvm.fabs.f32(float %915)
  %917 = call float @llvm.maximum.f32(float %914, float %916)
  %918 = extractelement <8 x float> %896, i64 7
  %919 = call float @llvm.fabs.f32(float %918)
  %920 = call float @llvm.maximum.f32(float %917, float %919)
  %921 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %922 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %921)
  %923 = add i32 %922, 32
  %924 = and i32 %923, -32
  %925 = xor i32 %922, 1
  %926 = icmp slt i32 %925, %924
  %927 = select i1 %926, i32 %925, i32 %922
  %928 = shl i32 %927, 2
  %929 = bitcast float %920 to i32
  %930 = call i32 @llvm.amdgcn.ds.bpermute(i32 %928, i32 %929)
  %931 = bitcast i32 %930 to float
  %932 = call float @llvm.maximum.f32(float %920, float %931)
  %933 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %934 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %933)
  %935 = add i32 %934, 32
  %936 = and i32 %935, -32
  %937 = xor i32 %934, 2
  %938 = icmp slt i32 %937, %936
  %939 = select i1 %938, i32 %937, i32 %934
  %940 = shl i32 %939, 2
  %941 = bitcast float %932 to i32
  %942 = call i32 @llvm.amdgcn.ds.bpermute(i32 %940, i32 %941)
  %943 = bitcast i32 %942 to float
  %944 = call float @llvm.maximum.f32(float %932, float %943)
  %945 = fmul float %944, 0x3F624924A0000000
  %946 = bitcast float %945 to i32
  %947 = and i32 %946, 8388607
  %948 = lshr i32 %946, 23
  %949 = and i32 %948, 255
  %950 = icmp ne i32 %947, 0
  %951 = add i32 %949, 1
  %952 = select i1 %950, i32 %951, i32 %949
  %953 = call i32 @llvm.smax.i32(i32 %952, i32 0)
  %954 = call i32 @llvm.smin.i32(i32 %953, i32 255)
  %955 = shl i32 %954, 23
  %956 = bitcast i32 %955 to float
  %957 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %895, float %956)
  %958 = udiv i32 %886, 4
  %959 = mul i32 %958, 4
  %960 = sub i32 %886, %959
  %961 = trunc i32 %954 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %957, ptr addrspace(8) %61, i32 %891, i32 0, i32 0)
  br i1 %56, label %962, label %966

962:                                              ; preds = %888
  %963 = add i32 %46, %958
  %964 = mul i32 %963, 4
  %965 = add i32 %964, %960
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %961, ptr addrspace(8) %50, i32 %965, i32 0, i32 0)
  br label %966

966:                                              ; preds = %962, %888
  br label %967

967:                                              ; preds = %966, %885
  %968 = add i32 %53, 88
  %969 = icmp ult i32 %968, 224
  br i1 %969, label %970, label %1049

970:                                              ; preds = %967
  %971 = mul i32 %968, 32
  %972 = mul i32 %55, 8
  %973 = add i32 %971, %972
  %974 = lshr i32 %973, 1
  %975 = mul i32 %974, 4
  %976 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %975, i32 0, i32 0)
  %977 = bitcast <4 x i32> %976 to <8 x bfloat>
  %978 = fpext <8 x bfloat> %977 to <8 x float>
  %979 = extractelement <8 x float> %978, i64 0
  %980 = call float @llvm.fabs.f32(float %979)
  %981 = call float @llvm.maximum.f32(float %980, float 0.000000e+00)
  %982 = extractelement <8 x float> %978, i64 1
  %983 = call float @llvm.fabs.f32(float %982)
  %984 = call float @llvm.maximum.f32(float %981, float %983)
  %985 = extractelement <8 x float> %978, i64 2
  %986 = call float @llvm.fabs.f32(float %985)
  %987 = call float @llvm.maximum.f32(float %984, float %986)
  %988 = extractelement <8 x float> %978, i64 3
  %989 = call float @llvm.fabs.f32(float %988)
  %990 = call float @llvm.maximum.f32(float %987, float %989)
  %991 = extractelement <8 x float> %978, i64 4
  %992 = call float @llvm.fabs.f32(float %991)
  %993 = call float @llvm.maximum.f32(float %990, float %992)
  %994 = extractelement <8 x float> %978, i64 5
  %995 = call float @llvm.fabs.f32(float %994)
  %996 = call float @llvm.maximum.f32(float %993, float %995)
  %997 = extractelement <8 x float> %978, i64 6
  %998 = call float @llvm.fabs.f32(float %997)
  %999 = call float @llvm.maximum.f32(float %996, float %998)
  %1000 = extractelement <8 x float> %978, i64 7
  %1001 = call float @llvm.fabs.f32(float %1000)
  %1002 = call float @llvm.maximum.f32(float %999, float %1001)
  %1003 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1004 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1003)
  %1005 = add i32 %1004, 32
  %1006 = and i32 %1005, -32
  %1007 = xor i32 %1004, 1
  %1008 = icmp slt i32 %1007, %1006
  %1009 = select i1 %1008, i32 %1007, i32 %1004
  %1010 = shl i32 %1009, 2
  %1011 = bitcast float %1002 to i32
  %1012 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1010, i32 %1011)
  %1013 = bitcast i32 %1012 to float
  %1014 = call float @llvm.maximum.f32(float %1002, float %1013)
  %1015 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1016 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1015)
  %1017 = add i32 %1016, 32
  %1018 = and i32 %1017, -32
  %1019 = xor i32 %1016, 2
  %1020 = icmp slt i32 %1019, %1018
  %1021 = select i1 %1020, i32 %1019, i32 %1016
  %1022 = shl i32 %1021, 2
  %1023 = bitcast float %1014 to i32
  %1024 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1022, i32 %1023)
  %1025 = bitcast i32 %1024 to float
  %1026 = call float @llvm.maximum.f32(float %1014, float %1025)
  %1027 = fmul float %1026, 0x3F624924A0000000
  %1028 = bitcast float %1027 to i32
  %1029 = and i32 %1028, 8388607
  %1030 = lshr i32 %1028, 23
  %1031 = and i32 %1030, 255
  %1032 = icmp ne i32 %1029, 0
  %1033 = add i32 %1031, 1
  %1034 = select i1 %1032, i32 %1033, i32 %1031
  %1035 = call i32 @llvm.smax.i32(i32 %1034, i32 0)
  %1036 = call i32 @llvm.smin.i32(i32 %1035, i32 255)
  %1037 = shl i32 %1036, 23
  %1038 = bitcast i32 %1037 to float
  %1039 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %977, float %1038)
  %1040 = udiv i32 %968, 4
  %1041 = mul i32 %1040, 4
  %1042 = sub i32 %968, %1041
  %1043 = trunc i32 %1036 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %1039, ptr addrspace(8) %61, i32 %973, i32 0, i32 0)
  br i1 %56, label %1044, label %1048

1044:                                             ; preds = %970
  %1045 = add i32 %46, %1040
  %1046 = mul i32 %1045, 4
  %1047 = add i32 %1046, %1042
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %1043, ptr addrspace(8) %50, i32 %1047, i32 0, i32 0)
  br label %1048

1048:                                             ; preds = %1044, %970
  br label %1049

1049:                                             ; preds = %1048, %967
  %1050 = add i32 %53, 96
  %1051 = icmp ult i32 %1050, 224
  br i1 %1051, label %1052, label %1131

1052:                                             ; preds = %1049
  %1053 = mul i32 %1050, 32
  %1054 = mul i32 %55, 8
  %1055 = add i32 %1053, %1054
  %1056 = lshr i32 %1055, 1
  %1057 = mul i32 %1056, 4
  %1058 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %1057, i32 0, i32 0)
  %1059 = bitcast <4 x i32> %1058 to <8 x bfloat>
  %1060 = fpext <8 x bfloat> %1059 to <8 x float>
  %1061 = extractelement <8 x float> %1060, i64 0
  %1062 = call float @llvm.fabs.f32(float %1061)
  %1063 = call float @llvm.maximum.f32(float %1062, float 0.000000e+00)
  %1064 = extractelement <8 x float> %1060, i64 1
  %1065 = call float @llvm.fabs.f32(float %1064)
  %1066 = call float @llvm.maximum.f32(float %1063, float %1065)
  %1067 = extractelement <8 x float> %1060, i64 2
  %1068 = call float @llvm.fabs.f32(float %1067)
  %1069 = call float @llvm.maximum.f32(float %1066, float %1068)
  %1070 = extractelement <8 x float> %1060, i64 3
  %1071 = call float @llvm.fabs.f32(float %1070)
  %1072 = call float @llvm.maximum.f32(float %1069, float %1071)
  %1073 = extractelement <8 x float> %1060, i64 4
  %1074 = call float @llvm.fabs.f32(float %1073)
  %1075 = call float @llvm.maximum.f32(float %1072, float %1074)
  %1076 = extractelement <8 x float> %1060, i64 5
  %1077 = call float @llvm.fabs.f32(float %1076)
  %1078 = call float @llvm.maximum.f32(float %1075, float %1077)
  %1079 = extractelement <8 x float> %1060, i64 6
  %1080 = call float @llvm.fabs.f32(float %1079)
  %1081 = call float @llvm.maximum.f32(float %1078, float %1080)
  %1082 = extractelement <8 x float> %1060, i64 7
  %1083 = call float @llvm.fabs.f32(float %1082)
  %1084 = call float @llvm.maximum.f32(float %1081, float %1083)
  %1085 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1086 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1085)
  %1087 = add i32 %1086, 32
  %1088 = and i32 %1087, -32
  %1089 = xor i32 %1086, 1
  %1090 = icmp slt i32 %1089, %1088
  %1091 = select i1 %1090, i32 %1089, i32 %1086
  %1092 = shl i32 %1091, 2
  %1093 = bitcast float %1084 to i32
  %1094 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1092, i32 %1093)
  %1095 = bitcast i32 %1094 to float
  %1096 = call float @llvm.maximum.f32(float %1084, float %1095)
  %1097 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1098 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1097)
  %1099 = add i32 %1098, 32
  %1100 = and i32 %1099, -32
  %1101 = xor i32 %1098, 2
  %1102 = icmp slt i32 %1101, %1100
  %1103 = select i1 %1102, i32 %1101, i32 %1098
  %1104 = shl i32 %1103, 2
  %1105 = bitcast float %1096 to i32
  %1106 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1104, i32 %1105)
  %1107 = bitcast i32 %1106 to float
  %1108 = call float @llvm.maximum.f32(float %1096, float %1107)
  %1109 = fmul float %1108, 0x3F624924A0000000
  %1110 = bitcast float %1109 to i32
  %1111 = and i32 %1110, 8388607
  %1112 = lshr i32 %1110, 23
  %1113 = and i32 %1112, 255
  %1114 = icmp ne i32 %1111, 0
  %1115 = add i32 %1113, 1
  %1116 = select i1 %1114, i32 %1115, i32 %1113
  %1117 = call i32 @llvm.smax.i32(i32 %1116, i32 0)
  %1118 = call i32 @llvm.smin.i32(i32 %1117, i32 255)
  %1119 = shl i32 %1118, 23
  %1120 = bitcast i32 %1119 to float
  %1121 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %1059, float %1120)
  %1122 = udiv i32 %1050, 4
  %1123 = mul i32 %1122, 4
  %1124 = sub i32 %1050, %1123
  %1125 = trunc i32 %1118 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %1121, ptr addrspace(8) %61, i32 %1055, i32 0, i32 0)
  br i1 %56, label %1126, label %1130

1126:                                             ; preds = %1052
  %1127 = add i32 %46, %1122
  %1128 = mul i32 %1127, 4
  %1129 = add i32 %1128, %1124
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %1125, ptr addrspace(8) %50, i32 %1129, i32 0, i32 0)
  br label %1130

1130:                                             ; preds = %1126, %1052
  br label %1131

1131:                                             ; preds = %1130, %1049
  %1132 = add i32 %53, 104
  %1133 = icmp ult i32 %1132, 224
  br i1 %1133, label %1134, label %1213

1134:                                             ; preds = %1131
  %1135 = mul i32 %1132, 32
  %1136 = mul i32 %55, 8
  %1137 = add i32 %1135, %1136
  %1138 = lshr i32 %1137, 1
  %1139 = mul i32 %1138, 4
  %1140 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %1139, i32 0, i32 0)
  %1141 = bitcast <4 x i32> %1140 to <8 x bfloat>
  %1142 = fpext <8 x bfloat> %1141 to <8 x float>
  %1143 = extractelement <8 x float> %1142, i64 0
  %1144 = call float @llvm.fabs.f32(float %1143)
  %1145 = call float @llvm.maximum.f32(float %1144, float 0.000000e+00)
  %1146 = extractelement <8 x float> %1142, i64 1
  %1147 = call float @llvm.fabs.f32(float %1146)
  %1148 = call float @llvm.maximum.f32(float %1145, float %1147)
  %1149 = extractelement <8 x float> %1142, i64 2
  %1150 = call float @llvm.fabs.f32(float %1149)
  %1151 = call float @llvm.maximum.f32(float %1148, float %1150)
  %1152 = extractelement <8 x float> %1142, i64 3
  %1153 = call float @llvm.fabs.f32(float %1152)
  %1154 = call float @llvm.maximum.f32(float %1151, float %1153)
  %1155 = extractelement <8 x float> %1142, i64 4
  %1156 = call float @llvm.fabs.f32(float %1155)
  %1157 = call float @llvm.maximum.f32(float %1154, float %1156)
  %1158 = extractelement <8 x float> %1142, i64 5
  %1159 = call float @llvm.fabs.f32(float %1158)
  %1160 = call float @llvm.maximum.f32(float %1157, float %1159)
  %1161 = extractelement <8 x float> %1142, i64 6
  %1162 = call float @llvm.fabs.f32(float %1161)
  %1163 = call float @llvm.maximum.f32(float %1160, float %1162)
  %1164 = extractelement <8 x float> %1142, i64 7
  %1165 = call float @llvm.fabs.f32(float %1164)
  %1166 = call float @llvm.maximum.f32(float %1163, float %1165)
  %1167 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1168 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1167)
  %1169 = add i32 %1168, 32
  %1170 = and i32 %1169, -32
  %1171 = xor i32 %1168, 1
  %1172 = icmp slt i32 %1171, %1170
  %1173 = select i1 %1172, i32 %1171, i32 %1168
  %1174 = shl i32 %1173, 2
  %1175 = bitcast float %1166 to i32
  %1176 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1174, i32 %1175)
  %1177 = bitcast i32 %1176 to float
  %1178 = call float @llvm.maximum.f32(float %1166, float %1177)
  %1179 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1180 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1179)
  %1181 = add i32 %1180, 32
  %1182 = and i32 %1181, -32
  %1183 = xor i32 %1180, 2
  %1184 = icmp slt i32 %1183, %1182
  %1185 = select i1 %1184, i32 %1183, i32 %1180
  %1186 = shl i32 %1185, 2
  %1187 = bitcast float %1178 to i32
  %1188 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1187)
  %1189 = bitcast i32 %1188 to float
  %1190 = call float @llvm.maximum.f32(float %1178, float %1189)
  %1191 = fmul float %1190, 0x3F624924A0000000
  %1192 = bitcast float %1191 to i32
  %1193 = and i32 %1192, 8388607
  %1194 = lshr i32 %1192, 23
  %1195 = and i32 %1194, 255
  %1196 = icmp ne i32 %1193, 0
  %1197 = add i32 %1195, 1
  %1198 = select i1 %1196, i32 %1197, i32 %1195
  %1199 = call i32 @llvm.smax.i32(i32 %1198, i32 0)
  %1200 = call i32 @llvm.smin.i32(i32 %1199, i32 255)
  %1201 = shl i32 %1200, 23
  %1202 = bitcast i32 %1201 to float
  %1203 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %1141, float %1202)
  %1204 = udiv i32 %1132, 4
  %1205 = mul i32 %1204, 4
  %1206 = sub i32 %1132, %1205
  %1207 = trunc i32 %1200 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %1203, ptr addrspace(8) %61, i32 %1137, i32 0, i32 0)
  br i1 %56, label %1208, label %1212

1208:                                             ; preds = %1134
  %1209 = add i32 %46, %1204
  %1210 = mul i32 %1209, 4
  %1211 = add i32 %1210, %1206
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %1207, ptr addrspace(8) %50, i32 %1211, i32 0, i32 0)
  br label %1212

1212:                                             ; preds = %1208, %1134
  br label %1213

1213:                                             ; preds = %1212, %1131
  %1214 = add i32 %53, 112
  %1215 = icmp ult i32 %1214, 224
  br i1 %1215, label %1216, label %1295

1216:                                             ; preds = %1213
  %1217 = mul i32 %1214, 32
  %1218 = mul i32 %55, 8
  %1219 = add i32 %1217, %1218
  %1220 = lshr i32 %1219, 1
  %1221 = mul i32 %1220, 4
  %1222 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %1221, i32 0, i32 0)
  %1223 = bitcast <4 x i32> %1222 to <8 x bfloat>
  %1224 = fpext <8 x bfloat> %1223 to <8 x float>
  %1225 = extractelement <8 x float> %1224, i64 0
  %1226 = call float @llvm.fabs.f32(float %1225)
  %1227 = call float @llvm.maximum.f32(float %1226, float 0.000000e+00)
  %1228 = extractelement <8 x float> %1224, i64 1
  %1229 = call float @llvm.fabs.f32(float %1228)
  %1230 = call float @llvm.maximum.f32(float %1227, float %1229)
  %1231 = extractelement <8 x float> %1224, i64 2
  %1232 = call float @llvm.fabs.f32(float %1231)
  %1233 = call float @llvm.maximum.f32(float %1230, float %1232)
  %1234 = extractelement <8 x float> %1224, i64 3
  %1235 = call float @llvm.fabs.f32(float %1234)
  %1236 = call float @llvm.maximum.f32(float %1233, float %1235)
  %1237 = extractelement <8 x float> %1224, i64 4
  %1238 = call float @llvm.fabs.f32(float %1237)
  %1239 = call float @llvm.maximum.f32(float %1236, float %1238)
  %1240 = extractelement <8 x float> %1224, i64 5
  %1241 = call float @llvm.fabs.f32(float %1240)
  %1242 = call float @llvm.maximum.f32(float %1239, float %1241)
  %1243 = extractelement <8 x float> %1224, i64 6
  %1244 = call float @llvm.fabs.f32(float %1243)
  %1245 = call float @llvm.maximum.f32(float %1242, float %1244)
  %1246 = extractelement <8 x float> %1224, i64 7
  %1247 = call float @llvm.fabs.f32(float %1246)
  %1248 = call float @llvm.maximum.f32(float %1245, float %1247)
  %1249 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1250 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1249)
  %1251 = add i32 %1250, 32
  %1252 = and i32 %1251, -32
  %1253 = xor i32 %1250, 1
  %1254 = icmp slt i32 %1253, %1252
  %1255 = select i1 %1254, i32 %1253, i32 %1250
  %1256 = shl i32 %1255, 2
  %1257 = bitcast float %1248 to i32
  %1258 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1256, i32 %1257)
  %1259 = bitcast i32 %1258 to float
  %1260 = call float @llvm.maximum.f32(float %1248, float %1259)
  %1261 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1262 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1261)
  %1263 = add i32 %1262, 32
  %1264 = and i32 %1263, -32
  %1265 = xor i32 %1262, 2
  %1266 = icmp slt i32 %1265, %1264
  %1267 = select i1 %1266, i32 %1265, i32 %1262
  %1268 = shl i32 %1267, 2
  %1269 = bitcast float %1260 to i32
  %1270 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1268, i32 %1269)
  %1271 = bitcast i32 %1270 to float
  %1272 = call float @llvm.maximum.f32(float %1260, float %1271)
  %1273 = fmul float %1272, 0x3F624924A0000000
  %1274 = bitcast float %1273 to i32
  %1275 = and i32 %1274, 8388607
  %1276 = lshr i32 %1274, 23
  %1277 = and i32 %1276, 255
  %1278 = icmp ne i32 %1275, 0
  %1279 = add i32 %1277, 1
  %1280 = select i1 %1278, i32 %1279, i32 %1277
  %1281 = call i32 @llvm.smax.i32(i32 %1280, i32 0)
  %1282 = call i32 @llvm.smin.i32(i32 %1281, i32 255)
  %1283 = shl i32 %1282, 23
  %1284 = bitcast i32 %1283 to float
  %1285 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %1223, float %1284)
  %1286 = udiv i32 %1214, 4
  %1287 = mul i32 %1286, 4
  %1288 = sub i32 %1214, %1287
  %1289 = trunc i32 %1282 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %1285, ptr addrspace(8) %61, i32 %1219, i32 0, i32 0)
  br i1 %56, label %1290, label %1294

1290:                                             ; preds = %1216
  %1291 = add i32 %46, %1286
  %1292 = mul i32 %1291, 4
  %1293 = add i32 %1292, %1288
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %1289, ptr addrspace(8) %50, i32 %1293, i32 0, i32 0)
  br label %1294

1294:                                             ; preds = %1290, %1216
  br label %1295

1295:                                             ; preds = %1294, %1213
  %1296 = add i32 %53, 120
  %1297 = icmp ult i32 %1296, 224
  br i1 %1297, label %1298, label %1377

1298:                                             ; preds = %1295
  %1299 = mul i32 %1296, 32
  %1300 = mul i32 %55, 8
  %1301 = add i32 %1299, %1300
  %1302 = lshr i32 %1301, 1
  %1303 = mul i32 %1302, 4
  %1304 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %1303, i32 0, i32 0)
  %1305 = bitcast <4 x i32> %1304 to <8 x bfloat>
  %1306 = fpext <8 x bfloat> %1305 to <8 x float>
  %1307 = extractelement <8 x float> %1306, i64 0
  %1308 = call float @llvm.fabs.f32(float %1307)
  %1309 = call float @llvm.maximum.f32(float %1308, float 0.000000e+00)
  %1310 = extractelement <8 x float> %1306, i64 1
  %1311 = call float @llvm.fabs.f32(float %1310)
  %1312 = call float @llvm.maximum.f32(float %1309, float %1311)
  %1313 = extractelement <8 x float> %1306, i64 2
  %1314 = call float @llvm.fabs.f32(float %1313)
  %1315 = call float @llvm.maximum.f32(float %1312, float %1314)
  %1316 = extractelement <8 x float> %1306, i64 3
  %1317 = call float @llvm.fabs.f32(float %1316)
  %1318 = call float @llvm.maximum.f32(float %1315, float %1317)
  %1319 = extractelement <8 x float> %1306, i64 4
  %1320 = call float @llvm.fabs.f32(float %1319)
  %1321 = call float @llvm.maximum.f32(float %1318, float %1320)
  %1322 = extractelement <8 x float> %1306, i64 5
  %1323 = call float @llvm.fabs.f32(float %1322)
  %1324 = call float @llvm.maximum.f32(float %1321, float %1323)
  %1325 = extractelement <8 x float> %1306, i64 6
  %1326 = call float @llvm.fabs.f32(float %1325)
  %1327 = call float @llvm.maximum.f32(float %1324, float %1326)
  %1328 = extractelement <8 x float> %1306, i64 7
  %1329 = call float @llvm.fabs.f32(float %1328)
  %1330 = call float @llvm.maximum.f32(float %1327, float %1329)
  %1331 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1332 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1331)
  %1333 = add i32 %1332, 32
  %1334 = and i32 %1333, -32
  %1335 = xor i32 %1332, 1
  %1336 = icmp slt i32 %1335, %1334
  %1337 = select i1 %1336, i32 %1335, i32 %1332
  %1338 = shl i32 %1337, 2
  %1339 = bitcast float %1330 to i32
  %1340 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1338, i32 %1339)
  %1341 = bitcast i32 %1340 to float
  %1342 = call float @llvm.maximum.f32(float %1330, float %1341)
  %1343 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1344 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1343)
  %1345 = add i32 %1344, 32
  %1346 = and i32 %1345, -32
  %1347 = xor i32 %1344, 2
  %1348 = icmp slt i32 %1347, %1346
  %1349 = select i1 %1348, i32 %1347, i32 %1344
  %1350 = shl i32 %1349, 2
  %1351 = bitcast float %1342 to i32
  %1352 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1350, i32 %1351)
  %1353 = bitcast i32 %1352 to float
  %1354 = call float @llvm.maximum.f32(float %1342, float %1353)
  %1355 = fmul float %1354, 0x3F624924A0000000
  %1356 = bitcast float %1355 to i32
  %1357 = and i32 %1356, 8388607
  %1358 = lshr i32 %1356, 23
  %1359 = and i32 %1358, 255
  %1360 = icmp ne i32 %1357, 0
  %1361 = add i32 %1359, 1
  %1362 = select i1 %1360, i32 %1361, i32 %1359
  %1363 = call i32 @llvm.smax.i32(i32 %1362, i32 0)
  %1364 = call i32 @llvm.smin.i32(i32 %1363, i32 255)
  %1365 = shl i32 %1364, 23
  %1366 = bitcast i32 %1365 to float
  %1367 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %1305, float %1366)
  %1368 = udiv i32 %1296, 4
  %1369 = mul i32 %1368, 4
  %1370 = sub i32 %1296, %1369
  %1371 = trunc i32 %1364 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %1367, ptr addrspace(8) %61, i32 %1301, i32 0, i32 0)
  br i1 %56, label %1372, label %1376

1372:                                             ; preds = %1298
  %1373 = add i32 %46, %1368
  %1374 = mul i32 %1373, 4
  %1375 = add i32 %1374, %1370
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %1371, ptr addrspace(8) %50, i32 %1375, i32 0, i32 0)
  br label %1376

1376:                                             ; preds = %1372, %1298
  br label %1377

1377:                                             ; preds = %1376, %1295
  %1378 = add i32 %53, 128
  %1379 = icmp ult i32 %1378, 224
  br i1 %1379, label %1380, label %1459

1380:                                             ; preds = %1377
  %1381 = mul i32 %1378, 32
  %1382 = mul i32 %55, 8
  %1383 = add i32 %1381, %1382
  %1384 = lshr i32 %1383, 1
  %1385 = mul i32 %1384, 4
  %1386 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %1385, i32 0, i32 0)
  %1387 = bitcast <4 x i32> %1386 to <8 x bfloat>
  %1388 = fpext <8 x bfloat> %1387 to <8 x float>
  %1389 = extractelement <8 x float> %1388, i64 0
  %1390 = call float @llvm.fabs.f32(float %1389)
  %1391 = call float @llvm.maximum.f32(float %1390, float 0.000000e+00)
  %1392 = extractelement <8 x float> %1388, i64 1
  %1393 = call float @llvm.fabs.f32(float %1392)
  %1394 = call float @llvm.maximum.f32(float %1391, float %1393)
  %1395 = extractelement <8 x float> %1388, i64 2
  %1396 = call float @llvm.fabs.f32(float %1395)
  %1397 = call float @llvm.maximum.f32(float %1394, float %1396)
  %1398 = extractelement <8 x float> %1388, i64 3
  %1399 = call float @llvm.fabs.f32(float %1398)
  %1400 = call float @llvm.maximum.f32(float %1397, float %1399)
  %1401 = extractelement <8 x float> %1388, i64 4
  %1402 = call float @llvm.fabs.f32(float %1401)
  %1403 = call float @llvm.maximum.f32(float %1400, float %1402)
  %1404 = extractelement <8 x float> %1388, i64 5
  %1405 = call float @llvm.fabs.f32(float %1404)
  %1406 = call float @llvm.maximum.f32(float %1403, float %1405)
  %1407 = extractelement <8 x float> %1388, i64 6
  %1408 = call float @llvm.fabs.f32(float %1407)
  %1409 = call float @llvm.maximum.f32(float %1406, float %1408)
  %1410 = extractelement <8 x float> %1388, i64 7
  %1411 = call float @llvm.fabs.f32(float %1410)
  %1412 = call float @llvm.maximum.f32(float %1409, float %1411)
  %1413 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1414 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1413)
  %1415 = add i32 %1414, 32
  %1416 = and i32 %1415, -32
  %1417 = xor i32 %1414, 1
  %1418 = icmp slt i32 %1417, %1416
  %1419 = select i1 %1418, i32 %1417, i32 %1414
  %1420 = shl i32 %1419, 2
  %1421 = bitcast float %1412 to i32
  %1422 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1420, i32 %1421)
  %1423 = bitcast i32 %1422 to float
  %1424 = call float @llvm.maximum.f32(float %1412, float %1423)
  %1425 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1426 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1425)
  %1427 = add i32 %1426, 32
  %1428 = and i32 %1427, -32
  %1429 = xor i32 %1426, 2
  %1430 = icmp slt i32 %1429, %1428
  %1431 = select i1 %1430, i32 %1429, i32 %1426
  %1432 = shl i32 %1431, 2
  %1433 = bitcast float %1424 to i32
  %1434 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1432, i32 %1433)
  %1435 = bitcast i32 %1434 to float
  %1436 = call float @llvm.maximum.f32(float %1424, float %1435)
  %1437 = fmul float %1436, 0x3F624924A0000000
  %1438 = bitcast float %1437 to i32
  %1439 = and i32 %1438, 8388607
  %1440 = lshr i32 %1438, 23
  %1441 = and i32 %1440, 255
  %1442 = icmp ne i32 %1439, 0
  %1443 = add i32 %1441, 1
  %1444 = select i1 %1442, i32 %1443, i32 %1441
  %1445 = call i32 @llvm.smax.i32(i32 %1444, i32 0)
  %1446 = call i32 @llvm.smin.i32(i32 %1445, i32 255)
  %1447 = shl i32 %1446, 23
  %1448 = bitcast i32 %1447 to float
  %1449 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %1387, float %1448)
  %1450 = udiv i32 %1378, 4
  %1451 = mul i32 %1450, 4
  %1452 = sub i32 %1378, %1451
  %1453 = trunc i32 %1446 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %1449, ptr addrspace(8) %61, i32 %1383, i32 0, i32 0)
  br i1 %56, label %1454, label %1458

1454:                                             ; preds = %1380
  %1455 = add i32 %46, %1450
  %1456 = mul i32 %1455, 4
  %1457 = add i32 %1456, %1452
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %1453, ptr addrspace(8) %50, i32 %1457, i32 0, i32 0)
  br label %1458

1458:                                             ; preds = %1454, %1380
  br label %1459

1459:                                             ; preds = %1458, %1377
  %1460 = add i32 %53, 136
  %1461 = icmp ult i32 %1460, 224
  br i1 %1461, label %1462, label %1541

1462:                                             ; preds = %1459
  %1463 = mul i32 %1460, 32
  %1464 = mul i32 %55, 8
  %1465 = add i32 %1463, %1464
  %1466 = lshr i32 %1465, 1
  %1467 = mul i32 %1466, 4
  %1468 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %1467, i32 0, i32 0)
  %1469 = bitcast <4 x i32> %1468 to <8 x bfloat>
  %1470 = fpext <8 x bfloat> %1469 to <8 x float>
  %1471 = extractelement <8 x float> %1470, i64 0
  %1472 = call float @llvm.fabs.f32(float %1471)
  %1473 = call float @llvm.maximum.f32(float %1472, float 0.000000e+00)
  %1474 = extractelement <8 x float> %1470, i64 1
  %1475 = call float @llvm.fabs.f32(float %1474)
  %1476 = call float @llvm.maximum.f32(float %1473, float %1475)
  %1477 = extractelement <8 x float> %1470, i64 2
  %1478 = call float @llvm.fabs.f32(float %1477)
  %1479 = call float @llvm.maximum.f32(float %1476, float %1478)
  %1480 = extractelement <8 x float> %1470, i64 3
  %1481 = call float @llvm.fabs.f32(float %1480)
  %1482 = call float @llvm.maximum.f32(float %1479, float %1481)
  %1483 = extractelement <8 x float> %1470, i64 4
  %1484 = call float @llvm.fabs.f32(float %1483)
  %1485 = call float @llvm.maximum.f32(float %1482, float %1484)
  %1486 = extractelement <8 x float> %1470, i64 5
  %1487 = call float @llvm.fabs.f32(float %1486)
  %1488 = call float @llvm.maximum.f32(float %1485, float %1487)
  %1489 = extractelement <8 x float> %1470, i64 6
  %1490 = call float @llvm.fabs.f32(float %1489)
  %1491 = call float @llvm.maximum.f32(float %1488, float %1490)
  %1492 = extractelement <8 x float> %1470, i64 7
  %1493 = call float @llvm.fabs.f32(float %1492)
  %1494 = call float @llvm.maximum.f32(float %1491, float %1493)
  %1495 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1496 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1495)
  %1497 = add i32 %1496, 32
  %1498 = and i32 %1497, -32
  %1499 = xor i32 %1496, 1
  %1500 = icmp slt i32 %1499, %1498
  %1501 = select i1 %1500, i32 %1499, i32 %1496
  %1502 = shl i32 %1501, 2
  %1503 = bitcast float %1494 to i32
  %1504 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1502, i32 %1503)
  %1505 = bitcast i32 %1504 to float
  %1506 = call float @llvm.maximum.f32(float %1494, float %1505)
  %1507 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1508 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1507)
  %1509 = add i32 %1508, 32
  %1510 = and i32 %1509, -32
  %1511 = xor i32 %1508, 2
  %1512 = icmp slt i32 %1511, %1510
  %1513 = select i1 %1512, i32 %1511, i32 %1508
  %1514 = shl i32 %1513, 2
  %1515 = bitcast float %1506 to i32
  %1516 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1514, i32 %1515)
  %1517 = bitcast i32 %1516 to float
  %1518 = call float @llvm.maximum.f32(float %1506, float %1517)
  %1519 = fmul float %1518, 0x3F624924A0000000
  %1520 = bitcast float %1519 to i32
  %1521 = and i32 %1520, 8388607
  %1522 = lshr i32 %1520, 23
  %1523 = and i32 %1522, 255
  %1524 = icmp ne i32 %1521, 0
  %1525 = add i32 %1523, 1
  %1526 = select i1 %1524, i32 %1525, i32 %1523
  %1527 = call i32 @llvm.smax.i32(i32 %1526, i32 0)
  %1528 = call i32 @llvm.smin.i32(i32 %1527, i32 255)
  %1529 = shl i32 %1528, 23
  %1530 = bitcast i32 %1529 to float
  %1531 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %1469, float %1530)
  %1532 = udiv i32 %1460, 4
  %1533 = mul i32 %1532, 4
  %1534 = sub i32 %1460, %1533
  %1535 = trunc i32 %1528 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %1531, ptr addrspace(8) %61, i32 %1465, i32 0, i32 0)
  br i1 %56, label %1536, label %1540

1536:                                             ; preds = %1462
  %1537 = add i32 %46, %1532
  %1538 = mul i32 %1537, 4
  %1539 = add i32 %1538, %1534
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %1535, ptr addrspace(8) %50, i32 %1539, i32 0, i32 0)
  br label %1540

1540:                                             ; preds = %1536, %1462
  br label %1541

1541:                                             ; preds = %1540, %1459
  %1542 = add i32 %53, 144
  %1543 = icmp ult i32 %1542, 224
  br i1 %1543, label %1544, label %1623

1544:                                             ; preds = %1541
  %1545 = mul i32 %1542, 32
  %1546 = mul i32 %55, 8
  %1547 = add i32 %1545, %1546
  %1548 = lshr i32 %1547, 1
  %1549 = mul i32 %1548, 4
  %1550 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %1549, i32 0, i32 0)
  %1551 = bitcast <4 x i32> %1550 to <8 x bfloat>
  %1552 = fpext <8 x bfloat> %1551 to <8 x float>
  %1553 = extractelement <8 x float> %1552, i64 0
  %1554 = call float @llvm.fabs.f32(float %1553)
  %1555 = call float @llvm.maximum.f32(float %1554, float 0.000000e+00)
  %1556 = extractelement <8 x float> %1552, i64 1
  %1557 = call float @llvm.fabs.f32(float %1556)
  %1558 = call float @llvm.maximum.f32(float %1555, float %1557)
  %1559 = extractelement <8 x float> %1552, i64 2
  %1560 = call float @llvm.fabs.f32(float %1559)
  %1561 = call float @llvm.maximum.f32(float %1558, float %1560)
  %1562 = extractelement <8 x float> %1552, i64 3
  %1563 = call float @llvm.fabs.f32(float %1562)
  %1564 = call float @llvm.maximum.f32(float %1561, float %1563)
  %1565 = extractelement <8 x float> %1552, i64 4
  %1566 = call float @llvm.fabs.f32(float %1565)
  %1567 = call float @llvm.maximum.f32(float %1564, float %1566)
  %1568 = extractelement <8 x float> %1552, i64 5
  %1569 = call float @llvm.fabs.f32(float %1568)
  %1570 = call float @llvm.maximum.f32(float %1567, float %1569)
  %1571 = extractelement <8 x float> %1552, i64 6
  %1572 = call float @llvm.fabs.f32(float %1571)
  %1573 = call float @llvm.maximum.f32(float %1570, float %1572)
  %1574 = extractelement <8 x float> %1552, i64 7
  %1575 = call float @llvm.fabs.f32(float %1574)
  %1576 = call float @llvm.maximum.f32(float %1573, float %1575)
  %1577 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1578 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1577)
  %1579 = add i32 %1578, 32
  %1580 = and i32 %1579, -32
  %1581 = xor i32 %1578, 1
  %1582 = icmp slt i32 %1581, %1580
  %1583 = select i1 %1582, i32 %1581, i32 %1578
  %1584 = shl i32 %1583, 2
  %1585 = bitcast float %1576 to i32
  %1586 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1584, i32 %1585)
  %1587 = bitcast i32 %1586 to float
  %1588 = call float @llvm.maximum.f32(float %1576, float %1587)
  %1589 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1590 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1589)
  %1591 = add i32 %1590, 32
  %1592 = and i32 %1591, -32
  %1593 = xor i32 %1590, 2
  %1594 = icmp slt i32 %1593, %1592
  %1595 = select i1 %1594, i32 %1593, i32 %1590
  %1596 = shl i32 %1595, 2
  %1597 = bitcast float %1588 to i32
  %1598 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1596, i32 %1597)
  %1599 = bitcast i32 %1598 to float
  %1600 = call float @llvm.maximum.f32(float %1588, float %1599)
  %1601 = fmul float %1600, 0x3F624924A0000000
  %1602 = bitcast float %1601 to i32
  %1603 = and i32 %1602, 8388607
  %1604 = lshr i32 %1602, 23
  %1605 = and i32 %1604, 255
  %1606 = icmp ne i32 %1603, 0
  %1607 = add i32 %1605, 1
  %1608 = select i1 %1606, i32 %1607, i32 %1605
  %1609 = call i32 @llvm.smax.i32(i32 %1608, i32 0)
  %1610 = call i32 @llvm.smin.i32(i32 %1609, i32 255)
  %1611 = shl i32 %1610, 23
  %1612 = bitcast i32 %1611 to float
  %1613 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %1551, float %1612)
  %1614 = udiv i32 %1542, 4
  %1615 = mul i32 %1614, 4
  %1616 = sub i32 %1542, %1615
  %1617 = trunc i32 %1610 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %1613, ptr addrspace(8) %61, i32 %1547, i32 0, i32 0)
  br i1 %56, label %1618, label %1622

1618:                                             ; preds = %1544
  %1619 = add i32 %46, %1614
  %1620 = mul i32 %1619, 4
  %1621 = add i32 %1620, %1616
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %1617, ptr addrspace(8) %50, i32 %1621, i32 0, i32 0)
  br label %1622

1622:                                             ; preds = %1618, %1544
  br label %1623

1623:                                             ; preds = %1622, %1541
  %1624 = add i32 %53, 152
  %1625 = icmp ult i32 %1624, 224
  br i1 %1625, label %1626, label %1705

1626:                                             ; preds = %1623
  %1627 = mul i32 %1624, 32
  %1628 = mul i32 %55, 8
  %1629 = add i32 %1627, %1628
  %1630 = lshr i32 %1629, 1
  %1631 = mul i32 %1630, 4
  %1632 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %1631, i32 0, i32 0)
  %1633 = bitcast <4 x i32> %1632 to <8 x bfloat>
  %1634 = fpext <8 x bfloat> %1633 to <8 x float>
  %1635 = extractelement <8 x float> %1634, i64 0
  %1636 = call float @llvm.fabs.f32(float %1635)
  %1637 = call float @llvm.maximum.f32(float %1636, float 0.000000e+00)
  %1638 = extractelement <8 x float> %1634, i64 1
  %1639 = call float @llvm.fabs.f32(float %1638)
  %1640 = call float @llvm.maximum.f32(float %1637, float %1639)
  %1641 = extractelement <8 x float> %1634, i64 2
  %1642 = call float @llvm.fabs.f32(float %1641)
  %1643 = call float @llvm.maximum.f32(float %1640, float %1642)
  %1644 = extractelement <8 x float> %1634, i64 3
  %1645 = call float @llvm.fabs.f32(float %1644)
  %1646 = call float @llvm.maximum.f32(float %1643, float %1645)
  %1647 = extractelement <8 x float> %1634, i64 4
  %1648 = call float @llvm.fabs.f32(float %1647)
  %1649 = call float @llvm.maximum.f32(float %1646, float %1648)
  %1650 = extractelement <8 x float> %1634, i64 5
  %1651 = call float @llvm.fabs.f32(float %1650)
  %1652 = call float @llvm.maximum.f32(float %1649, float %1651)
  %1653 = extractelement <8 x float> %1634, i64 6
  %1654 = call float @llvm.fabs.f32(float %1653)
  %1655 = call float @llvm.maximum.f32(float %1652, float %1654)
  %1656 = extractelement <8 x float> %1634, i64 7
  %1657 = call float @llvm.fabs.f32(float %1656)
  %1658 = call float @llvm.maximum.f32(float %1655, float %1657)
  %1659 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1660 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1659)
  %1661 = add i32 %1660, 32
  %1662 = and i32 %1661, -32
  %1663 = xor i32 %1660, 1
  %1664 = icmp slt i32 %1663, %1662
  %1665 = select i1 %1664, i32 %1663, i32 %1660
  %1666 = shl i32 %1665, 2
  %1667 = bitcast float %1658 to i32
  %1668 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1666, i32 %1667)
  %1669 = bitcast i32 %1668 to float
  %1670 = call float @llvm.maximum.f32(float %1658, float %1669)
  %1671 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1672 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1671)
  %1673 = add i32 %1672, 32
  %1674 = and i32 %1673, -32
  %1675 = xor i32 %1672, 2
  %1676 = icmp slt i32 %1675, %1674
  %1677 = select i1 %1676, i32 %1675, i32 %1672
  %1678 = shl i32 %1677, 2
  %1679 = bitcast float %1670 to i32
  %1680 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1678, i32 %1679)
  %1681 = bitcast i32 %1680 to float
  %1682 = call float @llvm.maximum.f32(float %1670, float %1681)
  %1683 = fmul float %1682, 0x3F624924A0000000
  %1684 = bitcast float %1683 to i32
  %1685 = and i32 %1684, 8388607
  %1686 = lshr i32 %1684, 23
  %1687 = and i32 %1686, 255
  %1688 = icmp ne i32 %1685, 0
  %1689 = add i32 %1687, 1
  %1690 = select i1 %1688, i32 %1689, i32 %1687
  %1691 = call i32 @llvm.smax.i32(i32 %1690, i32 0)
  %1692 = call i32 @llvm.smin.i32(i32 %1691, i32 255)
  %1693 = shl i32 %1692, 23
  %1694 = bitcast i32 %1693 to float
  %1695 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %1633, float %1694)
  %1696 = udiv i32 %1624, 4
  %1697 = mul i32 %1696, 4
  %1698 = sub i32 %1624, %1697
  %1699 = trunc i32 %1692 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %1695, ptr addrspace(8) %61, i32 %1629, i32 0, i32 0)
  br i1 %56, label %1700, label %1704

1700:                                             ; preds = %1626
  %1701 = add i32 %46, %1696
  %1702 = mul i32 %1701, 4
  %1703 = add i32 %1702, %1698
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %1699, ptr addrspace(8) %50, i32 %1703, i32 0, i32 0)
  br label %1704

1704:                                             ; preds = %1700, %1626
  br label %1705

1705:                                             ; preds = %1704, %1623
  %1706 = add i32 %53, 160
  %1707 = icmp ult i32 %1706, 224
  br i1 %1707, label %1708, label %1787

1708:                                             ; preds = %1705
  %1709 = mul i32 %1706, 32
  %1710 = mul i32 %55, 8
  %1711 = add i32 %1709, %1710
  %1712 = lshr i32 %1711, 1
  %1713 = mul i32 %1712, 4
  %1714 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %1713, i32 0, i32 0)
  %1715 = bitcast <4 x i32> %1714 to <8 x bfloat>
  %1716 = fpext <8 x bfloat> %1715 to <8 x float>
  %1717 = extractelement <8 x float> %1716, i64 0
  %1718 = call float @llvm.fabs.f32(float %1717)
  %1719 = call float @llvm.maximum.f32(float %1718, float 0.000000e+00)
  %1720 = extractelement <8 x float> %1716, i64 1
  %1721 = call float @llvm.fabs.f32(float %1720)
  %1722 = call float @llvm.maximum.f32(float %1719, float %1721)
  %1723 = extractelement <8 x float> %1716, i64 2
  %1724 = call float @llvm.fabs.f32(float %1723)
  %1725 = call float @llvm.maximum.f32(float %1722, float %1724)
  %1726 = extractelement <8 x float> %1716, i64 3
  %1727 = call float @llvm.fabs.f32(float %1726)
  %1728 = call float @llvm.maximum.f32(float %1725, float %1727)
  %1729 = extractelement <8 x float> %1716, i64 4
  %1730 = call float @llvm.fabs.f32(float %1729)
  %1731 = call float @llvm.maximum.f32(float %1728, float %1730)
  %1732 = extractelement <8 x float> %1716, i64 5
  %1733 = call float @llvm.fabs.f32(float %1732)
  %1734 = call float @llvm.maximum.f32(float %1731, float %1733)
  %1735 = extractelement <8 x float> %1716, i64 6
  %1736 = call float @llvm.fabs.f32(float %1735)
  %1737 = call float @llvm.maximum.f32(float %1734, float %1736)
  %1738 = extractelement <8 x float> %1716, i64 7
  %1739 = call float @llvm.fabs.f32(float %1738)
  %1740 = call float @llvm.maximum.f32(float %1737, float %1739)
  %1741 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1742 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1741)
  %1743 = add i32 %1742, 32
  %1744 = and i32 %1743, -32
  %1745 = xor i32 %1742, 1
  %1746 = icmp slt i32 %1745, %1744
  %1747 = select i1 %1746, i32 %1745, i32 %1742
  %1748 = shl i32 %1747, 2
  %1749 = bitcast float %1740 to i32
  %1750 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1748, i32 %1749)
  %1751 = bitcast i32 %1750 to float
  %1752 = call float @llvm.maximum.f32(float %1740, float %1751)
  %1753 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1754 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1753)
  %1755 = add i32 %1754, 32
  %1756 = and i32 %1755, -32
  %1757 = xor i32 %1754, 2
  %1758 = icmp slt i32 %1757, %1756
  %1759 = select i1 %1758, i32 %1757, i32 %1754
  %1760 = shl i32 %1759, 2
  %1761 = bitcast float %1752 to i32
  %1762 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1760, i32 %1761)
  %1763 = bitcast i32 %1762 to float
  %1764 = call float @llvm.maximum.f32(float %1752, float %1763)
  %1765 = fmul float %1764, 0x3F624924A0000000
  %1766 = bitcast float %1765 to i32
  %1767 = and i32 %1766, 8388607
  %1768 = lshr i32 %1766, 23
  %1769 = and i32 %1768, 255
  %1770 = icmp ne i32 %1767, 0
  %1771 = add i32 %1769, 1
  %1772 = select i1 %1770, i32 %1771, i32 %1769
  %1773 = call i32 @llvm.smax.i32(i32 %1772, i32 0)
  %1774 = call i32 @llvm.smin.i32(i32 %1773, i32 255)
  %1775 = shl i32 %1774, 23
  %1776 = bitcast i32 %1775 to float
  %1777 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %1715, float %1776)
  %1778 = udiv i32 %1706, 4
  %1779 = mul i32 %1778, 4
  %1780 = sub i32 %1706, %1779
  %1781 = trunc i32 %1774 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %1777, ptr addrspace(8) %61, i32 %1711, i32 0, i32 0)
  br i1 %56, label %1782, label %1786

1782:                                             ; preds = %1708
  %1783 = add i32 %46, %1778
  %1784 = mul i32 %1783, 4
  %1785 = add i32 %1784, %1780
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %1781, ptr addrspace(8) %50, i32 %1785, i32 0, i32 0)
  br label %1786

1786:                                             ; preds = %1782, %1708
  br label %1787

1787:                                             ; preds = %1786, %1705
  %1788 = add i32 %53, 168
  %1789 = icmp ult i32 %1788, 224
  br i1 %1789, label %1790, label %1869

1790:                                             ; preds = %1787
  %1791 = mul i32 %1788, 32
  %1792 = mul i32 %55, 8
  %1793 = add i32 %1791, %1792
  %1794 = lshr i32 %1793, 1
  %1795 = mul i32 %1794, 4
  %1796 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %1795, i32 0, i32 0)
  %1797 = bitcast <4 x i32> %1796 to <8 x bfloat>
  %1798 = fpext <8 x bfloat> %1797 to <8 x float>
  %1799 = extractelement <8 x float> %1798, i64 0
  %1800 = call float @llvm.fabs.f32(float %1799)
  %1801 = call float @llvm.maximum.f32(float %1800, float 0.000000e+00)
  %1802 = extractelement <8 x float> %1798, i64 1
  %1803 = call float @llvm.fabs.f32(float %1802)
  %1804 = call float @llvm.maximum.f32(float %1801, float %1803)
  %1805 = extractelement <8 x float> %1798, i64 2
  %1806 = call float @llvm.fabs.f32(float %1805)
  %1807 = call float @llvm.maximum.f32(float %1804, float %1806)
  %1808 = extractelement <8 x float> %1798, i64 3
  %1809 = call float @llvm.fabs.f32(float %1808)
  %1810 = call float @llvm.maximum.f32(float %1807, float %1809)
  %1811 = extractelement <8 x float> %1798, i64 4
  %1812 = call float @llvm.fabs.f32(float %1811)
  %1813 = call float @llvm.maximum.f32(float %1810, float %1812)
  %1814 = extractelement <8 x float> %1798, i64 5
  %1815 = call float @llvm.fabs.f32(float %1814)
  %1816 = call float @llvm.maximum.f32(float %1813, float %1815)
  %1817 = extractelement <8 x float> %1798, i64 6
  %1818 = call float @llvm.fabs.f32(float %1817)
  %1819 = call float @llvm.maximum.f32(float %1816, float %1818)
  %1820 = extractelement <8 x float> %1798, i64 7
  %1821 = call float @llvm.fabs.f32(float %1820)
  %1822 = call float @llvm.maximum.f32(float %1819, float %1821)
  %1823 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1824 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1823)
  %1825 = add i32 %1824, 32
  %1826 = and i32 %1825, -32
  %1827 = xor i32 %1824, 1
  %1828 = icmp slt i32 %1827, %1826
  %1829 = select i1 %1828, i32 %1827, i32 %1824
  %1830 = shl i32 %1829, 2
  %1831 = bitcast float %1822 to i32
  %1832 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1830, i32 %1831)
  %1833 = bitcast i32 %1832 to float
  %1834 = call float @llvm.maximum.f32(float %1822, float %1833)
  %1835 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1836 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1835)
  %1837 = add i32 %1836, 32
  %1838 = and i32 %1837, -32
  %1839 = xor i32 %1836, 2
  %1840 = icmp slt i32 %1839, %1838
  %1841 = select i1 %1840, i32 %1839, i32 %1836
  %1842 = shl i32 %1841, 2
  %1843 = bitcast float %1834 to i32
  %1844 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1842, i32 %1843)
  %1845 = bitcast i32 %1844 to float
  %1846 = call float @llvm.maximum.f32(float %1834, float %1845)
  %1847 = fmul float %1846, 0x3F624924A0000000
  %1848 = bitcast float %1847 to i32
  %1849 = and i32 %1848, 8388607
  %1850 = lshr i32 %1848, 23
  %1851 = and i32 %1850, 255
  %1852 = icmp ne i32 %1849, 0
  %1853 = add i32 %1851, 1
  %1854 = select i1 %1852, i32 %1853, i32 %1851
  %1855 = call i32 @llvm.smax.i32(i32 %1854, i32 0)
  %1856 = call i32 @llvm.smin.i32(i32 %1855, i32 255)
  %1857 = shl i32 %1856, 23
  %1858 = bitcast i32 %1857 to float
  %1859 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %1797, float %1858)
  %1860 = udiv i32 %1788, 4
  %1861 = mul i32 %1860, 4
  %1862 = sub i32 %1788, %1861
  %1863 = trunc i32 %1856 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %1859, ptr addrspace(8) %61, i32 %1793, i32 0, i32 0)
  br i1 %56, label %1864, label %1868

1864:                                             ; preds = %1790
  %1865 = add i32 %46, %1860
  %1866 = mul i32 %1865, 4
  %1867 = add i32 %1866, %1862
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %1863, ptr addrspace(8) %50, i32 %1867, i32 0, i32 0)
  br label %1868

1868:                                             ; preds = %1864, %1790
  br label %1869

1869:                                             ; preds = %1868, %1787
  %1870 = add i32 %53, 176
  %1871 = icmp ult i32 %1870, 224
  br i1 %1871, label %1872, label %1951

1872:                                             ; preds = %1869
  %1873 = mul i32 %1870, 32
  %1874 = mul i32 %55, 8
  %1875 = add i32 %1873, %1874
  %1876 = lshr i32 %1875, 1
  %1877 = mul i32 %1876, 4
  %1878 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %1877, i32 0, i32 0)
  %1879 = bitcast <4 x i32> %1878 to <8 x bfloat>
  %1880 = fpext <8 x bfloat> %1879 to <8 x float>
  %1881 = extractelement <8 x float> %1880, i64 0
  %1882 = call float @llvm.fabs.f32(float %1881)
  %1883 = call float @llvm.maximum.f32(float %1882, float 0.000000e+00)
  %1884 = extractelement <8 x float> %1880, i64 1
  %1885 = call float @llvm.fabs.f32(float %1884)
  %1886 = call float @llvm.maximum.f32(float %1883, float %1885)
  %1887 = extractelement <8 x float> %1880, i64 2
  %1888 = call float @llvm.fabs.f32(float %1887)
  %1889 = call float @llvm.maximum.f32(float %1886, float %1888)
  %1890 = extractelement <8 x float> %1880, i64 3
  %1891 = call float @llvm.fabs.f32(float %1890)
  %1892 = call float @llvm.maximum.f32(float %1889, float %1891)
  %1893 = extractelement <8 x float> %1880, i64 4
  %1894 = call float @llvm.fabs.f32(float %1893)
  %1895 = call float @llvm.maximum.f32(float %1892, float %1894)
  %1896 = extractelement <8 x float> %1880, i64 5
  %1897 = call float @llvm.fabs.f32(float %1896)
  %1898 = call float @llvm.maximum.f32(float %1895, float %1897)
  %1899 = extractelement <8 x float> %1880, i64 6
  %1900 = call float @llvm.fabs.f32(float %1899)
  %1901 = call float @llvm.maximum.f32(float %1898, float %1900)
  %1902 = extractelement <8 x float> %1880, i64 7
  %1903 = call float @llvm.fabs.f32(float %1902)
  %1904 = call float @llvm.maximum.f32(float %1901, float %1903)
  %1905 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1906 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1905)
  %1907 = add i32 %1906, 32
  %1908 = and i32 %1907, -32
  %1909 = xor i32 %1906, 1
  %1910 = icmp slt i32 %1909, %1908
  %1911 = select i1 %1910, i32 %1909, i32 %1906
  %1912 = shl i32 %1911, 2
  %1913 = bitcast float %1904 to i32
  %1914 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1912, i32 %1913)
  %1915 = bitcast i32 %1914 to float
  %1916 = call float @llvm.maximum.f32(float %1904, float %1915)
  %1917 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1918 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1917)
  %1919 = add i32 %1918, 32
  %1920 = and i32 %1919, -32
  %1921 = xor i32 %1918, 2
  %1922 = icmp slt i32 %1921, %1920
  %1923 = select i1 %1922, i32 %1921, i32 %1918
  %1924 = shl i32 %1923, 2
  %1925 = bitcast float %1916 to i32
  %1926 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1924, i32 %1925)
  %1927 = bitcast i32 %1926 to float
  %1928 = call float @llvm.maximum.f32(float %1916, float %1927)
  %1929 = fmul float %1928, 0x3F624924A0000000
  %1930 = bitcast float %1929 to i32
  %1931 = and i32 %1930, 8388607
  %1932 = lshr i32 %1930, 23
  %1933 = and i32 %1932, 255
  %1934 = icmp ne i32 %1931, 0
  %1935 = add i32 %1933, 1
  %1936 = select i1 %1934, i32 %1935, i32 %1933
  %1937 = call i32 @llvm.smax.i32(i32 %1936, i32 0)
  %1938 = call i32 @llvm.smin.i32(i32 %1937, i32 255)
  %1939 = shl i32 %1938, 23
  %1940 = bitcast i32 %1939 to float
  %1941 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %1879, float %1940)
  %1942 = udiv i32 %1870, 4
  %1943 = mul i32 %1942, 4
  %1944 = sub i32 %1870, %1943
  %1945 = trunc i32 %1938 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %1941, ptr addrspace(8) %61, i32 %1875, i32 0, i32 0)
  br i1 %56, label %1946, label %1950

1946:                                             ; preds = %1872
  %1947 = add i32 %46, %1942
  %1948 = mul i32 %1947, 4
  %1949 = add i32 %1948, %1944
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %1945, ptr addrspace(8) %50, i32 %1949, i32 0, i32 0)
  br label %1950

1950:                                             ; preds = %1946, %1872
  br label %1951

1951:                                             ; preds = %1950, %1869
  %1952 = add i32 %53, 184
  %1953 = icmp ult i32 %1952, 224
  br i1 %1953, label %1954, label %2033

1954:                                             ; preds = %1951
  %1955 = mul i32 %1952, 32
  %1956 = mul i32 %55, 8
  %1957 = add i32 %1955, %1956
  %1958 = lshr i32 %1957, 1
  %1959 = mul i32 %1958, 4
  %1960 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %1959, i32 0, i32 0)
  %1961 = bitcast <4 x i32> %1960 to <8 x bfloat>
  %1962 = fpext <8 x bfloat> %1961 to <8 x float>
  %1963 = extractelement <8 x float> %1962, i64 0
  %1964 = call float @llvm.fabs.f32(float %1963)
  %1965 = call float @llvm.maximum.f32(float %1964, float 0.000000e+00)
  %1966 = extractelement <8 x float> %1962, i64 1
  %1967 = call float @llvm.fabs.f32(float %1966)
  %1968 = call float @llvm.maximum.f32(float %1965, float %1967)
  %1969 = extractelement <8 x float> %1962, i64 2
  %1970 = call float @llvm.fabs.f32(float %1969)
  %1971 = call float @llvm.maximum.f32(float %1968, float %1970)
  %1972 = extractelement <8 x float> %1962, i64 3
  %1973 = call float @llvm.fabs.f32(float %1972)
  %1974 = call float @llvm.maximum.f32(float %1971, float %1973)
  %1975 = extractelement <8 x float> %1962, i64 4
  %1976 = call float @llvm.fabs.f32(float %1975)
  %1977 = call float @llvm.maximum.f32(float %1974, float %1976)
  %1978 = extractelement <8 x float> %1962, i64 5
  %1979 = call float @llvm.fabs.f32(float %1978)
  %1980 = call float @llvm.maximum.f32(float %1977, float %1979)
  %1981 = extractelement <8 x float> %1962, i64 6
  %1982 = call float @llvm.fabs.f32(float %1981)
  %1983 = call float @llvm.maximum.f32(float %1980, float %1982)
  %1984 = extractelement <8 x float> %1962, i64 7
  %1985 = call float @llvm.fabs.f32(float %1984)
  %1986 = call float @llvm.maximum.f32(float %1983, float %1985)
  %1987 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %1988 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1987)
  %1989 = add i32 %1988, 32
  %1990 = and i32 %1989, -32
  %1991 = xor i32 %1988, 1
  %1992 = icmp slt i32 %1991, %1990
  %1993 = select i1 %1992, i32 %1991, i32 %1988
  %1994 = shl i32 %1993, 2
  %1995 = bitcast float %1986 to i32
  %1996 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1994, i32 %1995)
  %1997 = bitcast i32 %1996 to float
  %1998 = call float @llvm.maximum.f32(float %1986, float %1997)
  %1999 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %2000 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1999)
  %2001 = add i32 %2000, 32
  %2002 = and i32 %2001, -32
  %2003 = xor i32 %2000, 2
  %2004 = icmp slt i32 %2003, %2002
  %2005 = select i1 %2004, i32 %2003, i32 %2000
  %2006 = shl i32 %2005, 2
  %2007 = bitcast float %1998 to i32
  %2008 = call i32 @llvm.amdgcn.ds.bpermute(i32 %2006, i32 %2007)
  %2009 = bitcast i32 %2008 to float
  %2010 = call float @llvm.maximum.f32(float %1998, float %2009)
  %2011 = fmul float %2010, 0x3F624924A0000000
  %2012 = bitcast float %2011 to i32
  %2013 = and i32 %2012, 8388607
  %2014 = lshr i32 %2012, 23
  %2015 = and i32 %2014, 255
  %2016 = icmp ne i32 %2013, 0
  %2017 = add i32 %2015, 1
  %2018 = select i1 %2016, i32 %2017, i32 %2015
  %2019 = call i32 @llvm.smax.i32(i32 %2018, i32 0)
  %2020 = call i32 @llvm.smin.i32(i32 %2019, i32 255)
  %2021 = shl i32 %2020, 23
  %2022 = bitcast i32 %2021 to float
  %2023 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %1961, float %2022)
  %2024 = udiv i32 %1952, 4
  %2025 = mul i32 %2024, 4
  %2026 = sub i32 %1952, %2025
  %2027 = trunc i32 %2020 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %2023, ptr addrspace(8) %61, i32 %1957, i32 0, i32 0)
  br i1 %56, label %2028, label %2032

2028:                                             ; preds = %1954
  %2029 = add i32 %46, %2024
  %2030 = mul i32 %2029, 4
  %2031 = add i32 %2030, %2026
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %2027, ptr addrspace(8) %50, i32 %2031, i32 0, i32 0)
  br label %2032

2032:                                             ; preds = %2028, %1954
  br label %2033

2033:                                             ; preds = %2032, %1951
  %2034 = add i32 %53, 192
  %2035 = icmp ult i32 %2034, 224
  br i1 %2035, label %2036, label %2115

2036:                                             ; preds = %2033
  %2037 = mul i32 %2034, 32
  %2038 = mul i32 %55, 8
  %2039 = add i32 %2037, %2038
  %2040 = lshr i32 %2039, 1
  %2041 = mul i32 %2040, 4
  %2042 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %2041, i32 0, i32 0)
  %2043 = bitcast <4 x i32> %2042 to <8 x bfloat>
  %2044 = fpext <8 x bfloat> %2043 to <8 x float>
  %2045 = extractelement <8 x float> %2044, i64 0
  %2046 = call float @llvm.fabs.f32(float %2045)
  %2047 = call float @llvm.maximum.f32(float %2046, float 0.000000e+00)
  %2048 = extractelement <8 x float> %2044, i64 1
  %2049 = call float @llvm.fabs.f32(float %2048)
  %2050 = call float @llvm.maximum.f32(float %2047, float %2049)
  %2051 = extractelement <8 x float> %2044, i64 2
  %2052 = call float @llvm.fabs.f32(float %2051)
  %2053 = call float @llvm.maximum.f32(float %2050, float %2052)
  %2054 = extractelement <8 x float> %2044, i64 3
  %2055 = call float @llvm.fabs.f32(float %2054)
  %2056 = call float @llvm.maximum.f32(float %2053, float %2055)
  %2057 = extractelement <8 x float> %2044, i64 4
  %2058 = call float @llvm.fabs.f32(float %2057)
  %2059 = call float @llvm.maximum.f32(float %2056, float %2058)
  %2060 = extractelement <8 x float> %2044, i64 5
  %2061 = call float @llvm.fabs.f32(float %2060)
  %2062 = call float @llvm.maximum.f32(float %2059, float %2061)
  %2063 = extractelement <8 x float> %2044, i64 6
  %2064 = call float @llvm.fabs.f32(float %2063)
  %2065 = call float @llvm.maximum.f32(float %2062, float %2064)
  %2066 = extractelement <8 x float> %2044, i64 7
  %2067 = call float @llvm.fabs.f32(float %2066)
  %2068 = call float @llvm.maximum.f32(float %2065, float %2067)
  %2069 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %2070 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %2069)
  %2071 = add i32 %2070, 32
  %2072 = and i32 %2071, -32
  %2073 = xor i32 %2070, 1
  %2074 = icmp slt i32 %2073, %2072
  %2075 = select i1 %2074, i32 %2073, i32 %2070
  %2076 = shl i32 %2075, 2
  %2077 = bitcast float %2068 to i32
  %2078 = call i32 @llvm.amdgcn.ds.bpermute(i32 %2076, i32 %2077)
  %2079 = bitcast i32 %2078 to float
  %2080 = call float @llvm.maximum.f32(float %2068, float %2079)
  %2081 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %2082 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %2081)
  %2083 = add i32 %2082, 32
  %2084 = and i32 %2083, -32
  %2085 = xor i32 %2082, 2
  %2086 = icmp slt i32 %2085, %2084
  %2087 = select i1 %2086, i32 %2085, i32 %2082
  %2088 = shl i32 %2087, 2
  %2089 = bitcast float %2080 to i32
  %2090 = call i32 @llvm.amdgcn.ds.bpermute(i32 %2088, i32 %2089)
  %2091 = bitcast i32 %2090 to float
  %2092 = call float @llvm.maximum.f32(float %2080, float %2091)
  %2093 = fmul float %2092, 0x3F624924A0000000
  %2094 = bitcast float %2093 to i32
  %2095 = and i32 %2094, 8388607
  %2096 = lshr i32 %2094, 23
  %2097 = and i32 %2096, 255
  %2098 = icmp ne i32 %2095, 0
  %2099 = add i32 %2097, 1
  %2100 = select i1 %2098, i32 %2099, i32 %2097
  %2101 = call i32 @llvm.smax.i32(i32 %2100, i32 0)
  %2102 = call i32 @llvm.smin.i32(i32 %2101, i32 255)
  %2103 = shl i32 %2102, 23
  %2104 = bitcast i32 %2103 to float
  %2105 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %2043, float %2104)
  %2106 = udiv i32 %2034, 4
  %2107 = mul i32 %2106, 4
  %2108 = sub i32 %2034, %2107
  %2109 = trunc i32 %2102 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %2105, ptr addrspace(8) %61, i32 %2039, i32 0, i32 0)
  br i1 %56, label %2110, label %2114

2110:                                             ; preds = %2036
  %2111 = add i32 %46, %2106
  %2112 = mul i32 %2111, 4
  %2113 = add i32 %2112, %2108
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %2109, ptr addrspace(8) %50, i32 %2113, i32 0, i32 0)
  br label %2114

2114:                                             ; preds = %2110, %2036
  br label %2115

2115:                                             ; preds = %2114, %2033
  %2116 = add i32 %53, 200
  %2117 = icmp ult i32 %2116, 224
  br i1 %2117, label %2118, label %2197

2118:                                             ; preds = %2115
  %2119 = mul i32 %2116, 32
  %2120 = mul i32 %55, 8
  %2121 = add i32 %2119, %2120
  %2122 = lshr i32 %2121, 1
  %2123 = mul i32 %2122, 4
  %2124 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %2123, i32 0, i32 0)
  %2125 = bitcast <4 x i32> %2124 to <8 x bfloat>
  %2126 = fpext <8 x bfloat> %2125 to <8 x float>
  %2127 = extractelement <8 x float> %2126, i64 0
  %2128 = call float @llvm.fabs.f32(float %2127)
  %2129 = call float @llvm.maximum.f32(float %2128, float 0.000000e+00)
  %2130 = extractelement <8 x float> %2126, i64 1
  %2131 = call float @llvm.fabs.f32(float %2130)
  %2132 = call float @llvm.maximum.f32(float %2129, float %2131)
  %2133 = extractelement <8 x float> %2126, i64 2
  %2134 = call float @llvm.fabs.f32(float %2133)
  %2135 = call float @llvm.maximum.f32(float %2132, float %2134)
  %2136 = extractelement <8 x float> %2126, i64 3
  %2137 = call float @llvm.fabs.f32(float %2136)
  %2138 = call float @llvm.maximum.f32(float %2135, float %2137)
  %2139 = extractelement <8 x float> %2126, i64 4
  %2140 = call float @llvm.fabs.f32(float %2139)
  %2141 = call float @llvm.maximum.f32(float %2138, float %2140)
  %2142 = extractelement <8 x float> %2126, i64 5
  %2143 = call float @llvm.fabs.f32(float %2142)
  %2144 = call float @llvm.maximum.f32(float %2141, float %2143)
  %2145 = extractelement <8 x float> %2126, i64 6
  %2146 = call float @llvm.fabs.f32(float %2145)
  %2147 = call float @llvm.maximum.f32(float %2144, float %2146)
  %2148 = extractelement <8 x float> %2126, i64 7
  %2149 = call float @llvm.fabs.f32(float %2148)
  %2150 = call float @llvm.maximum.f32(float %2147, float %2149)
  %2151 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %2152 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %2151)
  %2153 = add i32 %2152, 32
  %2154 = and i32 %2153, -32
  %2155 = xor i32 %2152, 1
  %2156 = icmp slt i32 %2155, %2154
  %2157 = select i1 %2156, i32 %2155, i32 %2152
  %2158 = shl i32 %2157, 2
  %2159 = bitcast float %2150 to i32
  %2160 = call i32 @llvm.amdgcn.ds.bpermute(i32 %2158, i32 %2159)
  %2161 = bitcast i32 %2160 to float
  %2162 = call float @llvm.maximum.f32(float %2150, float %2161)
  %2163 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %2164 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %2163)
  %2165 = add i32 %2164, 32
  %2166 = and i32 %2165, -32
  %2167 = xor i32 %2164, 2
  %2168 = icmp slt i32 %2167, %2166
  %2169 = select i1 %2168, i32 %2167, i32 %2164
  %2170 = shl i32 %2169, 2
  %2171 = bitcast float %2162 to i32
  %2172 = call i32 @llvm.amdgcn.ds.bpermute(i32 %2170, i32 %2171)
  %2173 = bitcast i32 %2172 to float
  %2174 = call float @llvm.maximum.f32(float %2162, float %2173)
  %2175 = fmul float %2174, 0x3F624924A0000000
  %2176 = bitcast float %2175 to i32
  %2177 = and i32 %2176, 8388607
  %2178 = lshr i32 %2176, 23
  %2179 = and i32 %2178, 255
  %2180 = icmp ne i32 %2177, 0
  %2181 = add i32 %2179, 1
  %2182 = select i1 %2180, i32 %2181, i32 %2179
  %2183 = call i32 @llvm.smax.i32(i32 %2182, i32 0)
  %2184 = call i32 @llvm.smin.i32(i32 %2183, i32 255)
  %2185 = shl i32 %2184, 23
  %2186 = bitcast i32 %2185 to float
  %2187 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %2125, float %2186)
  %2188 = udiv i32 %2116, 4
  %2189 = mul i32 %2188, 4
  %2190 = sub i32 %2116, %2189
  %2191 = trunc i32 %2184 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %2187, ptr addrspace(8) %61, i32 %2121, i32 0, i32 0)
  br i1 %56, label %2192, label %2196

2192:                                             ; preds = %2118
  %2193 = add i32 %46, %2188
  %2194 = mul i32 %2193, 4
  %2195 = add i32 %2194, %2190
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %2191, ptr addrspace(8) %50, i32 %2195, i32 0, i32 0)
  br label %2196

2196:                                             ; preds = %2192, %2118
  br label %2197

2197:                                             ; preds = %2196, %2115
  %2198 = add i32 %53, 208
  %2199 = icmp ult i32 %2198, 224
  br i1 %2199, label %2200, label %2279

2200:                                             ; preds = %2197
  %2201 = mul i32 %2198, 32
  %2202 = mul i32 %55, 8
  %2203 = add i32 %2201, %2202
  %2204 = lshr i32 %2203, 1
  %2205 = mul i32 %2204, 4
  %2206 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %2205, i32 0, i32 0)
  %2207 = bitcast <4 x i32> %2206 to <8 x bfloat>
  %2208 = fpext <8 x bfloat> %2207 to <8 x float>
  %2209 = extractelement <8 x float> %2208, i64 0
  %2210 = call float @llvm.fabs.f32(float %2209)
  %2211 = call float @llvm.maximum.f32(float %2210, float 0.000000e+00)
  %2212 = extractelement <8 x float> %2208, i64 1
  %2213 = call float @llvm.fabs.f32(float %2212)
  %2214 = call float @llvm.maximum.f32(float %2211, float %2213)
  %2215 = extractelement <8 x float> %2208, i64 2
  %2216 = call float @llvm.fabs.f32(float %2215)
  %2217 = call float @llvm.maximum.f32(float %2214, float %2216)
  %2218 = extractelement <8 x float> %2208, i64 3
  %2219 = call float @llvm.fabs.f32(float %2218)
  %2220 = call float @llvm.maximum.f32(float %2217, float %2219)
  %2221 = extractelement <8 x float> %2208, i64 4
  %2222 = call float @llvm.fabs.f32(float %2221)
  %2223 = call float @llvm.maximum.f32(float %2220, float %2222)
  %2224 = extractelement <8 x float> %2208, i64 5
  %2225 = call float @llvm.fabs.f32(float %2224)
  %2226 = call float @llvm.maximum.f32(float %2223, float %2225)
  %2227 = extractelement <8 x float> %2208, i64 6
  %2228 = call float @llvm.fabs.f32(float %2227)
  %2229 = call float @llvm.maximum.f32(float %2226, float %2228)
  %2230 = extractelement <8 x float> %2208, i64 7
  %2231 = call float @llvm.fabs.f32(float %2230)
  %2232 = call float @llvm.maximum.f32(float %2229, float %2231)
  %2233 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %2234 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %2233)
  %2235 = add i32 %2234, 32
  %2236 = and i32 %2235, -32
  %2237 = xor i32 %2234, 1
  %2238 = icmp slt i32 %2237, %2236
  %2239 = select i1 %2238, i32 %2237, i32 %2234
  %2240 = shl i32 %2239, 2
  %2241 = bitcast float %2232 to i32
  %2242 = call i32 @llvm.amdgcn.ds.bpermute(i32 %2240, i32 %2241)
  %2243 = bitcast i32 %2242 to float
  %2244 = call float @llvm.maximum.f32(float %2232, float %2243)
  %2245 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %2246 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %2245)
  %2247 = add i32 %2246, 32
  %2248 = and i32 %2247, -32
  %2249 = xor i32 %2246, 2
  %2250 = icmp slt i32 %2249, %2248
  %2251 = select i1 %2250, i32 %2249, i32 %2246
  %2252 = shl i32 %2251, 2
  %2253 = bitcast float %2244 to i32
  %2254 = call i32 @llvm.amdgcn.ds.bpermute(i32 %2252, i32 %2253)
  %2255 = bitcast i32 %2254 to float
  %2256 = call float @llvm.maximum.f32(float %2244, float %2255)
  %2257 = fmul float %2256, 0x3F624924A0000000
  %2258 = bitcast float %2257 to i32
  %2259 = and i32 %2258, 8388607
  %2260 = lshr i32 %2258, 23
  %2261 = and i32 %2260, 255
  %2262 = icmp ne i32 %2259, 0
  %2263 = add i32 %2261, 1
  %2264 = select i1 %2262, i32 %2263, i32 %2261
  %2265 = call i32 @llvm.smax.i32(i32 %2264, i32 0)
  %2266 = call i32 @llvm.smin.i32(i32 %2265, i32 255)
  %2267 = shl i32 %2266, 23
  %2268 = bitcast i32 %2267 to float
  %2269 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %2207, float %2268)
  %2270 = udiv i32 %2198, 4
  %2271 = mul i32 %2270, 4
  %2272 = sub i32 %2198, %2271
  %2273 = trunc i32 %2266 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %2269, ptr addrspace(8) %61, i32 %2203, i32 0, i32 0)
  br i1 %56, label %2274, label %2278

2274:                                             ; preds = %2200
  %2275 = add i32 %46, %2270
  %2276 = mul i32 %2275, 4
  %2277 = add i32 %2276, %2272
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %2273, ptr addrspace(8) %50, i32 %2277, i32 0, i32 0)
  br label %2278

2278:                                             ; preds = %2274, %2200
  br label %2279

2279:                                             ; preds = %2278, %2197
  %2280 = add i32 %53, 216
  %2281 = icmp ult i32 %2280, 224
  br i1 %2281, label %2282, label %2361

2282:                                             ; preds = %2279
  %2283 = mul i32 %2280, 32
  %2284 = mul i32 %55, 8
  %2285 = add i32 %2283, %2284
  %2286 = lshr i32 %2285, 1
  %2287 = mul i32 %2286, 4
  %2288 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %66, i32 %2287, i32 0, i32 0)
  %2289 = bitcast <4 x i32> %2288 to <8 x bfloat>
  %2290 = fpext <8 x bfloat> %2289 to <8 x float>
  %2291 = extractelement <8 x float> %2290, i64 0
  %2292 = call float @llvm.fabs.f32(float %2291)
  %2293 = call float @llvm.maximum.f32(float %2292, float 0.000000e+00)
  %2294 = extractelement <8 x float> %2290, i64 1
  %2295 = call float @llvm.fabs.f32(float %2294)
  %2296 = call float @llvm.maximum.f32(float %2293, float %2295)
  %2297 = extractelement <8 x float> %2290, i64 2
  %2298 = call float @llvm.fabs.f32(float %2297)
  %2299 = call float @llvm.maximum.f32(float %2296, float %2298)
  %2300 = extractelement <8 x float> %2290, i64 3
  %2301 = call float @llvm.fabs.f32(float %2300)
  %2302 = call float @llvm.maximum.f32(float %2299, float %2301)
  %2303 = extractelement <8 x float> %2290, i64 4
  %2304 = call float @llvm.fabs.f32(float %2303)
  %2305 = call float @llvm.maximum.f32(float %2302, float %2304)
  %2306 = extractelement <8 x float> %2290, i64 5
  %2307 = call float @llvm.fabs.f32(float %2306)
  %2308 = call float @llvm.maximum.f32(float %2305, float %2307)
  %2309 = extractelement <8 x float> %2290, i64 6
  %2310 = call float @llvm.fabs.f32(float %2309)
  %2311 = call float @llvm.maximum.f32(float %2308, float %2310)
  %2312 = extractelement <8 x float> %2290, i64 7
  %2313 = call float @llvm.fabs.f32(float %2312)
  %2314 = call float @llvm.maximum.f32(float %2311, float %2313)
  %2315 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %2316 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %2315)
  %2317 = add i32 %2316, 32
  %2318 = and i32 %2317, -32
  %2319 = xor i32 %2316, 1
  %2320 = icmp slt i32 %2319, %2318
  %2321 = select i1 %2320, i32 %2319, i32 %2316
  %2322 = shl i32 %2321, 2
  %2323 = bitcast float %2314 to i32
  %2324 = call i32 @llvm.amdgcn.ds.bpermute(i32 %2322, i32 %2323)
  %2325 = bitcast i32 %2324 to float
  %2326 = call float @llvm.maximum.f32(float %2314, float %2325)
  %2327 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %2328 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %2327)
  %2329 = add i32 %2328, 32
  %2330 = and i32 %2329, -32
  %2331 = xor i32 %2328, 2
  %2332 = icmp slt i32 %2331, %2330
  %2333 = select i1 %2332, i32 %2331, i32 %2328
  %2334 = shl i32 %2333, 2
  %2335 = bitcast float %2326 to i32
  %2336 = call i32 @llvm.amdgcn.ds.bpermute(i32 %2334, i32 %2335)
  %2337 = bitcast i32 %2336 to float
  %2338 = call float @llvm.maximum.f32(float %2326, float %2337)
  %2339 = fmul float %2338, 0x3F624924A0000000
  %2340 = bitcast float %2339 to i32
  %2341 = and i32 %2340, 8388607
  %2342 = lshr i32 %2340, 23
  %2343 = and i32 %2342, 255
  %2344 = icmp ne i32 %2341, 0
  %2345 = add i32 %2343, 1
  %2346 = select i1 %2344, i32 %2345, i32 %2343
  %2347 = call i32 @llvm.smax.i32(i32 %2346, i32 0)
  %2348 = call i32 @llvm.smin.i32(i32 %2347, i32 255)
  %2349 = shl i32 %2348, 23
  %2350 = bitcast i32 %2349 to float
  %2351 = call <2 x i32> asm "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2", "=v,v,v"(<8 x bfloat> %2289, float %2350)
  %2352 = udiv i32 %2280, 4
  %2353 = mul i32 %2352, 4
  %2354 = sub i32 %2280, %2353
  %2355 = trunc i32 %2348 to i8
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32> %2351, ptr addrspace(8) %61, i32 %2285, i32 0, i32 0)
  br i1 %56, label %2356, label %2360

2356:                                             ; preds = %2282
  %2357 = add i32 %46, %2352
  %2358 = mul i32 %2357, 4
  %2359 = add i32 %2358, %2354
  call void @llvm.amdgcn.raw.ptr.buffer.store.i8(i8 %2355, ptr addrspace(8) %50, i32 %2359, i32 0, i32 0)
  br label %2360

2360:                                             ; preds = %2356, %2282
  br label %2361

2361:                                             ; preds = %2360, %2279
  br label %2362

2362:                                             ; preds = %2361, %8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr readnone, i16, i64, i32) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #3

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
