; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

@__dynamic_shared_0 = external dso_local addrspace(3) global [0 x i8], align 1024

define amdgpu_kernel void @gemm_a8w8_mx128_t16x64x512_mw1_nw4_nb4_sk1_cm1_cn1(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10) #0 !reqd_work_group_size !1 {
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
  %61 = mul i32 %31, 64
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
  %144 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 536903679, i32 4, i32 poison, i32 poison, i32 poison>, i32 %140, i64 5
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
  %157 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 41216) to i32), i32 poison, i32 poison>, i32 %153, i64 2
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
  %186 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 41280) to i32), i32 poison, i32 poison>, i32 %182, i64 2
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
  br i1 %100, label %196, label %224

196:                                              ; preds = %195
  %197 = icmp eq i64 %21, -2147483648
  %198 = select i1 %197, i64 512, i64 %21
  %199 = ptrtoint ptr addrspace(1) %77 to i64
  %200 = add i64 %199, 512
  %201 = trunc i64 %200 to i32
  %202 = lshr i64 %200, 32
  %203 = trunc i64 %202 to i32
  %204 = or i32 %203, -2147483648
  %205 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 41984) to i32), i32 poison, i32 poison>, i32 %201, i64 2
  %206 = insertelement <4 x i32> %205, i32 %204, i64 3
  %207 = call i32 @llvm.smax.i32(i32 %64, i32 0)
  %208 = and i32 %207, 65535
  %209 = shl i32 %208, 16
  %210 = or i32 %209, 32767
  %211 = lshr i32 %207, 16
  %212 = and i32 %211, 65535
  %213 = or i32 %212, 33554432
  %214 = trunc i64 %198 to i32
  %215 = lshr i64 %198, 32
  %216 = trunc i64 %215 to i32
  %217 = and i32 %216, 65535
  %218 = insertelement <8 x i32> <i32 128974848, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %210, i64 2
  %219 = insertelement <8 x i32> %218, i32 %213, i64 3
  %220 = insertelement <8 x i32> %219, i32 16, i64 4
  %221 = insertelement <8 x i32> %220, i32 %214, i64 5
  %222 = insertelement <8 x i32> %221, i32 %217, i64 6
  %223 = insertelement <8 x i32> %222, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %206, <8 x i32> %223, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %224

224:                                              ; preds = %196, %195
  br i1 %129, label %225, label %243

225:                                              ; preds = %224
  %226 = icmp eq i64 %75, -2147483648
  %227 = select i1 %226, i64 8192, i64 %75
  %228 = ptrtoint ptr addrspace(1) %78 to i64
  %229 = add i64 %228, 8192
  %230 = trunc i64 %229 to i32
  %231 = lshr i64 %229, 32
  %232 = trunc i64 %231 to i32
  %233 = or i32 %232, -2147483648
  %234 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 50432) to i32), i32 poison, i32 poison>, i32 %230, i64 2
  %235 = insertelement <4 x i32> %234, i32 %233, i64 3
  %236 = trunc i64 %227 to i32
  %237 = lshr i64 %227, 32
  %238 = trunc i64 %237 to i32
  %239 = and i32 %238, 65535
  %240 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 536903679, i32 4, i32 poison, i32 poison, i32 poison>, i32 %236, i64 5
  %241 = insertelement <8 x i32> %240, i32 %239, i64 6
  %242 = insertelement <8 x i32> %241, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %235, <8 x i32> %242, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %243

243:                                              ; preds = %225, %224
  %244 = mul i64 %65, 4
  br i1 %148, label %245, label %273

245:                                              ; preds = %243
  %246 = icmp eq i64 %65, -2147483648
  %247 = select i1 %246, i64 16, i64 %65
  %248 = ptrtoint ptr addrspace(1) %96 to i64
  %249 = add i64 %248, %244
  %250 = trunc i64 %249 to i32
  %251 = lshr i64 %249, 32
  %252 = trunc i64 %251 to i32
  %253 = or i32 %252, -2147483648
  %254 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 83200) to i32), i32 poison, i32 poison>, i32 %250, i64 2
  %255 = insertelement <4 x i32> %254, i32 %253, i64 3
  %256 = call i32 @llvm.smax.i32(i32 %64, i32 0)
  %257 = and i32 %256, 65535
  %258 = shl i32 %257, 16
  %259 = lshr i32 %256, 16
  %260 = and i32 %259, 65535
  %261 = or i32 %260, -65536
  %262 = trunc i64 %247 to i32
  %263 = lshr i64 %247, 32
  %264 = trunc i64 %263 to i32
  %265 = and i32 %264, 65535
  %266 = insertelement <8 x i32> <i32 2097152, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %258, i64 1
  %267 = insertelement <8 x i32> %266, i32 %261, i64 2
  %268 = insertelement <8 x i32> %267, i32 1081343, i64 3
  %269 = insertelement <8 x i32> %268, i32 4, i64 4
  %270 = insertelement <8 x i32> %269, i32 %262, i64 5
  %271 = insertelement <8 x i32> %270, i32 %265, i64 6
  %272 = insertelement <8 x i32> %271, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %255, <8 x i32> %272, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %273

273:                                              ; preds = %245, %243
  br i1 %177, label %274, label %292

274:                                              ; preds = %273
  %275 = icmp eq i64 %94, -2147483648
  %276 = select i1 %275, i64 4, i64 %94
  %277 = ptrtoint ptr addrspace(1) %97 to i64
  %278 = add i64 %277, 4
  %279 = trunc i64 %278 to i32
  %280 = lshr i64 %278, 32
  %281 = trunc i64 %280 to i32
  %282 = or i32 %281, -2147483648
  %283 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 83264) to i32), i32 poison, i32 poison>, i32 %279, i64 2
  %284 = insertelement <4 x i32> %283, i32 %282, i64 3
  %285 = trunc i64 %276 to i32
  %286 = lshr i64 %276, 32
  %287 = trunc i64 %286 to i32
  %288 = and i32 %287, 65535
  %289 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 294911, i32 1, i32 poison, i32 poison, i32 poison>, i32 %285, i64 5
  %290 = insertelement <8 x i32> %289, i32 %288, i64 6
  %291 = insertelement <8 x i32> %290, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %284, <8 x i32> %291, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %292

292:                                              ; preds = %274, %273
  br i1 %100, label %293, label %321

293:                                              ; preds = %292
  %294 = icmp eq i64 %21, -2147483648
  %295 = select i1 %294, i64 512, i64 %21
  %296 = ptrtoint ptr addrspace(1) %77 to i64
  %297 = add i64 %296, 1024
  %298 = trunc i64 %297 to i32
  %299 = lshr i64 %297, 32
  %300 = trunc i64 %299 to i32
  %301 = or i32 %300, -2147483648
  %302 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 83968) to i32), i32 poison, i32 poison>, i32 %298, i64 2
  %303 = insertelement <4 x i32> %302, i32 %301, i64 3
  %304 = call i32 @llvm.smax.i32(i32 %64, i32 0)
  %305 = and i32 %304, 65535
  %306 = shl i32 %305, 16
  %307 = or i32 %306, 32767
  %308 = lshr i32 %304, 16
  %309 = and i32 %308, 65535
  %310 = or i32 %309, 33554432
  %311 = trunc i64 %295 to i32
  %312 = lshr i64 %295, 32
  %313 = trunc i64 %312 to i32
  %314 = and i32 %313, 65535
  %315 = insertelement <8 x i32> <i32 128974848, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %307, i64 2
  %316 = insertelement <8 x i32> %315, i32 %310, i64 3
  %317 = insertelement <8 x i32> %316, i32 16, i64 4
  %318 = insertelement <8 x i32> %317, i32 %311, i64 5
  %319 = insertelement <8 x i32> %318, i32 %314, i64 6
  %320 = insertelement <8 x i32> %319, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %303, <8 x i32> %320, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %321

321:                                              ; preds = %293, %292
  br i1 %129, label %322, label %340

322:                                              ; preds = %321
  %323 = icmp eq i64 %75, -2147483648
  %324 = select i1 %323, i64 8192, i64 %75
  %325 = ptrtoint ptr addrspace(1) %78 to i64
  %326 = add i64 %325, 16384
  %327 = trunc i64 %326 to i32
  %328 = lshr i64 %326, 32
  %329 = trunc i64 %328 to i32
  %330 = or i32 %329, -2147483648
  %331 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 92416) to i32), i32 poison, i32 poison>, i32 %327, i64 2
  %332 = insertelement <4 x i32> %331, i32 %330, i64 3
  %333 = trunc i64 %324 to i32
  %334 = lshr i64 %324, 32
  %335 = trunc i64 %334 to i32
  %336 = and i32 %335, 65535
  %337 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 536903679, i32 4, i32 poison, i32 poison, i32 poison>, i32 %333, i64 5
  %338 = insertelement <8 x i32> %337, i32 %336, i64 6
  %339 = insertelement <8 x i32> %338, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %332, <8 x i32> %339, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %340

340:                                              ; preds = %322, %321
  %341 = mul i64 %65, 8
  br i1 %148, label %342, label %370

342:                                              ; preds = %340
  %343 = icmp eq i64 %65, -2147483648
  %344 = select i1 %343, i64 16, i64 %65
  %345 = ptrtoint ptr addrspace(1) %96 to i64
  %346 = add i64 %345, %341
  %347 = trunc i64 %346 to i32
  %348 = lshr i64 %346, 32
  %349 = trunc i64 %348 to i32
  %350 = or i32 %349, -2147483648
  %351 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 125184) to i32), i32 poison, i32 poison>, i32 %347, i64 2
  %352 = insertelement <4 x i32> %351, i32 %350, i64 3
  %353 = call i32 @llvm.smax.i32(i32 %64, i32 0)
  %354 = and i32 %353, 65535
  %355 = shl i32 %354, 16
  %356 = lshr i32 %353, 16
  %357 = and i32 %356, 65535
  %358 = or i32 %357, -65536
  %359 = trunc i64 %344 to i32
  %360 = lshr i64 %344, 32
  %361 = trunc i64 %360 to i32
  %362 = and i32 %361, 65535
  %363 = insertelement <8 x i32> <i32 2097152, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %355, i64 1
  %364 = insertelement <8 x i32> %363, i32 %358, i64 2
  %365 = insertelement <8 x i32> %364, i32 1081343, i64 3
  %366 = insertelement <8 x i32> %365, i32 4, i64 4
  %367 = insertelement <8 x i32> %366, i32 %359, i64 5
  %368 = insertelement <8 x i32> %367, i32 %362, i64 6
  %369 = insertelement <8 x i32> %368, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %352, <8 x i32> %369, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %370

370:                                              ; preds = %342, %340
  br i1 %177, label %371, label %389

371:                                              ; preds = %370
  %372 = icmp eq i64 %94, -2147483648
  %373 = select i1 %372, i64 4, i64 %94
  %374 = ptrtoint ptr addrspace(1) %97 to i64
  %375 = add i64 %374, 8
  %376 = trunc i64 %375 to i32
  %377 = lshr i64 %375, 32
  %378 = trunc i64 %377 to i32
  %379 = or i32 %378, -2147483648
  %380 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 125248) to i32), i32 poison, i32 poison>, i32 %376, i64 2
  %381 = insertelement <4 x i32> %380, i32 %379, i64 3
  %382 = trunc i64 %373 to i32
  %383 = lshr i64 %373, 32
  %384 = trunc i64 %383 to i32
  %385 = and i32 %384, 65535
  %386 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 294911, i32 1, i32 poison, i32 poison, i32 poison>, i32 %382, i64 5
  %387 = insertelement <8 x i32> %386, i32 %385, i64 6
  %388 = insertelement <8 x i32> %387, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %381, <8 x i32> %388, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %389

389:                                              ; preds = %371, %370
  %390 = sub i32 %19, 3
  br label %391

391:                                              ; preds = %599, %389
  %392 = phi i32 [ %786, %599 ], [ 0, %389 ]
  %393 = phi <8 x float> [ %785, %599 ], [ zeroinitializer, %389 ]
  %394 = icmp slt i32 %392, %390
  br i1 %394, label %395, label %787

395:                                              ; preds = %391
  %396 = srem i32 %392, 4
  %397 = mul i32 %396, 41984
  %398 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %397
  %399 = ptrtoint ptr addrspace(3) %398 to i32
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 2)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %400 = add i32 %392, 3
  %401 = sdiv i32 %99, 16
  %402 = mul i32 %401, 16
  %403 = icmp ne i32 %99, %402
  %404 = icmp slt i32 %99, 0
  %405 = icmp ne i1 %404, false
  %406 = and i1 %403, %405
  %407 = add i32 %401, -1
  %408 = select i1 %406, i32 %407, i32 %401
  %409 = mul i32 %408, 8192
  %410 = add i32 %409, 8448
  %411 = mul i32 %50, 256
  %412 = add i32 %410, %411
  %413 = mul i32 %42, 16
  %414 = add i32 %412, %413
  %415 = sext i32 %414 to i64
  %416 = add i32 %399, %414
  %417 = inttoptr i32 %416 to ptr addrspace(3)
  %418 = load <4 x i32>, ptr addrspace(3) %417, align 16
  %419 = add i64 %415, 512
  %420 = trunc i64 %419 to i32
  %421 = add i32 %399, %420
  %422 = inttoptr i32 %421 to ptr addrspace(3)
  %423 = load <4 x i32>, ptr addrspace(3) %422, align 16
  %424 = add i64 %415, 1024
  %425 = trunc i64 %424 to i32
  %426 = add i32 %399, %425
  %427 = inttoptr i32 %426 to ptr addrspace(3)
  %428 = load <4 x i32>, ptr addrspace(3) %427, align 16
  %429 = add i64 %415, 1536
  %430 = trunc i64 %429 to i32
  %431 = add i32 %399, %430
  %432 = inttoptr i32 %431 to ptr addrspace(3)
  %433 = load <4 x i32>, ptr addrspace(3) %432, align 16
  %434 = shufflevector <4 x i32> %418, <4 x i32> %423, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %435 = shufflevector <4 x i32> %428, <4 x i32> %433, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %436 = shufflevector <8 x i32> %434, <8 x i32> %435, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %437 = sdiv i32 %99, 128
  %438 = mul i32 %437, 128
  %439 = icmp ne i32 %99, %438
  %440 = icmp slt i32 %99, 0
  %441 = icmp ne i1 %440, false
  %442 = and i1 %439, %441
  %443 = add i32 %437, -1
  %444 = select i1 %442, i32 %443, i32 %437
  %445 = mul i32 %444, 4
  %446 = add i32 %445, 41280
  %447 = add i32 %397, %446
  %448 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %447
  %449 = load i8, ptr addrspace(3) %448, align 1
  %450 = zext i8 %449 to i32
  %451 = call i32 @llvm.amdgcn.perm(i32 %450, i32 %450, i32 0)
  %452 = add i32 %98, %42
  %453 = add i32 %452, 41216
  %454 = add i32 %397, %453
  %455 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %454
  %456 = load i8, ptr addrspace(3) %455, align 1
  %457 = zext i8 %456 to i32
  %458 = call i32 @llvm.amdgcn.perm(i32 %457, i32 %457, i32 0)
  %459 = mul i32 %452, 528
  %460 = mul i32 %50, 16
  %461 = add i32 %459, %460
  %462 = sext i32 %461 to i64
  %463 = add i32 %399, %461
  %464 = inttoptr i32 %463 to ptr addrspace(3)
  %465 = load <4 x i32>, ptr addrspace(3) %464, align 16
  %466 = add i64 %462, 32
  %467 = trunc i64 %466 to i32
  %468 = add i32 %399, %467
  %469 = inttoptr i32 %468 to ptr addrspace(3)
  %470 = load <4 x i32>, ptr addrspace(3) %469, align 16
  %471 = add i64 %462, 64
  %472 = trunc i64 %471 to i32
  %473 = add i32 %399, %472
  %474 = inttoptr i32 %473 to ptr addrspace(3)
  %475 = load <4 x i32>, ptr addrspace(3) %474, align 16
  %476 = add i64 %462, 96
  %477 = trunc i64 %476 to i32
  %478 = add i32 %399, %477
  %479 = inttoptr i32 %478 to ptr addrspace(3)
  %480 = load <4 x i32>, ptr addrspace(3) %479, align 16
  %481 = shufflevector <4 x i32> %465, <4 x i32> %470, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %482 = shufflevector <4 x i32> %475, <4 x i32> %480, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %483 = shufflevector <8 x i32> %481, <8 x i32> %482, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %484 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %436, i32 0, <16 x i32> %483, i16 0, <8 x float> %393, i32 0, i32 0, i32 %451, i32 0, i32 0, i32 %458, i1 false, i1 false)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %485 = srem i32 %400, 4
  %486 = mul i32 %485, 41984
  %487 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %486
  %488 = sext i32 %400 to i64
  %489 = mul i64 %488, 512
  br i1 %100, label %490, label %520

490:                                              ; preds = %395
  %491 = icmp eq i64 %21, -2147483648
  %492 = select i1 %491, i64 512, i64 %21
  %493 = ptrtoint ptr addrspace(1) %77 to i64
  %494 = add i64 %493, %489
  %495 = ptrtoint ptr addrspace(3) %487 to i32
  %496 = trunc i64 %494 to i32
  %497 = lshr i64 %494, 32
  %498 = trunc i64 %497 to i32
  %499 = or i32 %498, -2147483648
  %500 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %495, i64 1
  %501 = insertelement <4 x i32> %500, i32 %496, i64 2
  %502 = insertelement <4 x i32> %501, i32 %499, i64 3
  %503 = call i32 @llvm.smax.i32(i32 %64, i32 0)
  %504 = and i32 %503, 65535
  %505 = shl i32 %504, 16
  %506 = or i32 %505, 32767
  %507 = lshr i32 %503, 16
  %508 = and i32 %507, 65535
  %509 = or i32 %508, 33554432
  %510 = trunc i64 %492 to i32
  %511 = lshr i64 %492, 32
  %512 = trunc i64 %511 to i32
  %513 = and i32 %512, 65535
  %514 = insertelement <8 x i32> <i32 128974848, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %506, i64 2
  %515 = insertelement <8 x i32> %514, i32 %509, i64 3
  %516 = insertelement <8 x i32> %515, i32 16, i64 4
  %517 = insertelement <8 x i32> %516, i32 %510, i64 5
  %518 = insertelement <8 x i32> %517, i32 %513, i64 6
  %519 = insertelement <8 x i32> %518, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %502, <8 x i32> %519, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %520

520:                                              ; preds = %490, %395
  %521 = getelementptr i8, ptr addrspace(3) %487, i32 8448
  %522 = mul i64 %488, 8192
  br i1 %129, label %523, label %543

523:                                              ; preds = %520
  %524 = icmp eq i64 %75, -2147483648
  %525 = select i1 %524, i64 8192, i64 %75
  %526 = ptrtoint ptr addrspace(1) %78 to i64
  %527 = add i64 %526, %522
  %528 = ptrtoint ptr addrspace(3) %521 to i32
  %529 = trunc i64 %527 to i32
  %530 = lshr i64 %527, 32
  %531 = trunc i64 %530 to i32
  %532 = or i32 %531, -2147483648
  %533 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %528, i64 1
  %534 = insertelement <4 x i32> %533, i32 %529, i64 2
  %535 = insertelement <4 x i32> %534, i32 %532, i64 3
  %536 = trunc i64 %525 to i32
  %537 = lshr i64 %525, 32
  %538 = trunc i64 %537 to i32
  %539 = and i32 %538, 65535
  %540 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 536903679, i32 4, i32 poison, i32 poison, i32 poison>, i32 %536, i64 5
  %541 = insertelement <8 x i32> %540, i32 %539, i64 6
  %542 = insertelement <8 x i32> %541, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %535, <8 x i32> %542, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %543

543:                                              ; preds = %523, %520
  %544 = getelementptr i8, ptr addrspace(3) %487, i32 41216
  %545 = mul i64 %488, 4
  %546 = mul i64 %545, %65
  br i1 %148, label %547, label %577

547:                                              ; preds = %543
  %548 = icmp eq i64 %65, -2147483648
  %549 = select i1 %548, i64 16, i64 %65
  %550 = ptrtoint ptr addrspace(1) %96 to i64
  %551 = add i64 %550, %546
  %552 = ptrtoint ptr addrspace(3) %544 to i32
  %553 = trunc i64 %551 to i32
  %554 = lshr i64 %551, 32
  %555 = trunc i64 %554 to i32
  %556 = or i32 %555, -2147483648
  %557 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %552, i64 1
  %558 = insertelement <4 x i32> %557, i32 %553, i64 2
  %559 = insertelement <4 x i32> %558, i32 %556, i64 3
  %560 = call i32 @llvm.smax.i32(i32 %64, i32 0)
  %561 = and i32 %560, 65535
  %562 = shl i32 %561, 16
  %563 = lshr i32 %560, 16
  %564 = and i32 %563, 65535
  %565 = or i32 %564, -65536
  %566 = trunc i64 %549 to i32
  %567 = lshr i64 %549, 32
  %568 = trunc i64 %567 to i32
  %569 = and i32 %568, 65535
  %570 = insertelement <8 x i32> <i32 2097152, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %562, i64 1
  %571 = insertelement <8 x i32> %570, i32 %565, i64 2
  %572 = insertelement <8 x i32> %571, i32 1081343, i64 3
  %573 = insertelement <8 x i32> %572, i32 4, i64 4
  %574 = insertelement <8 x i32> %573, i32 %566, i64 5
  %575 = insertelement <8 x i32> %574, i32 %569, i64 6
  %576 = insertelement <8 x i32> %575, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %559, <8 x i32> %576, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %577

577:                                              ; preds = %547, %543
  %578 = getelementptr i8, ptr addrspace(3) %487, i32 41280
  br i1 %177, label %579, label %599

579:                                              ; preds = %577
  %580 = icmp eq i64 %94, -2147483648
  %581 = select i1 %580, i64 4, i64 %94
  %582 = ptrtoint ptr addrspace(1) %97 to i64
  %583 = add i64 %582, %545
  %584 = ptrtoint ptr addrspace(3) %578 to i32
  %585 = trunc i64 %583 to i32
  %586 = lshr i64 %583, 32
  %587 = trunc i64 %586 to i32
  %588 = or i32 %587, -2147483648
  %589 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %584, i64 1
  %590 = insertelement <4 x i32> %589, i32 %585, i64 2
  %591 = insertelement <4 x i32> %590, i32 %588, i64 3
  %592 = trunc i64 %581 to i32
  %593 = lshr i64 %581, 32
  %594 = trunc i64 %593 to i32
  %595 = and i32 %594, 65535
  %596 = insertelement <8 x i32> <i32 2097152, i32 -65536, i32 -32769, i32 294911, i32 1, i32 poison, i32 poison, i32 poison>, i32 %592, i64 5
  %597 = insertelement <8 x i32> %596, i32 %595, i64 6
  %598 = insertelement <8 x i32> %597, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %591, <8 x i32> %598, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  br label %599

599:                                              ; preds = %579, %577
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %600 = add i32 %409, 10496
  %601 = add i32 %600, %411
  %602 = add i32 %601, %413
  %603 = sext i32 %602 to i64
  %604 = add i32 %399, %602
  %605 = inttoptr i32 %604 to ptr addrspace(3)
  %606 = load <4 x i32>, ptr addrspace(3) %605, align 16
  %607 = add i64 %603, 512
  %608 = trunc i64 %607 to i32
  %609 = add i32 %399, %608
  %610 = inttoptr i32 %609 to ptr addrspace(3)
  %611 = load <4 x i32>, ptr addrspace(3) %610, align 16
  %612 = add i64 %603, 1024
  %613 = trunc i64 %612 to i32
  %614 = add i32 %399, %613
  %615 = inttoptr i32 %614 to ptr addrspace(3)
  %616 = load <4 x i32>, ptr addrspace(3) %615, align 16
  %617 = add i64 %603, 1536
  %618 = trunc i64 %617 to i32
  %619 = add i32 %399, %618
  %620 = inttoptr i32 %619 to ptr addrspace(3)
  %621 = load <4 x i32>, ptr addrspace(3) %620, align 16
  %622 = shufflevector <4 x i32> %606, <4 x i32> %611, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %623 = shufflevector <4 x i32> %616, <4 x i32> %621, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %624 = shufflevector <8 x i32> %622, <8 x i32> %623, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %625 = add i32 %445, 41281
  %626 = add i32 %397, %625
  %627 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %626
  %628 = load i8, ptr addrspace(3) %627, align 1
  %629 = zext i8 %628 to i32
  %630 = call i32 @llvm.amdgcn.perm(i32 %629, i32 %629, i32 0)
  %631 = add i32 %452, 41232
  %632 = add i32 %397, %631
  %633 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %632
  %634 = load i8, ptr addrspace(3) %633, align 1
  %635 = zext i8 %634 to i32
  %636 = call i32 @llvm.amdgcn.perm(i32 %635, i32 %635, i32 0)
  %637 = add i32 %459, 128
  %638 = add i32 %637, %460
  %639 = sext i32 %638 to i64
  %640 = add i32 %399, %638
  %641 = inttoptr i32 %640 to ptr addrspace(3)
  %642 = load <4 x i32>, ptr addrspace(3) %641, align 16
  %643 = add i64 %639, 32
  %644 = trunc i64 %643 to i32
  %645 = add i32 %399, %644
  %646 = inttoptr i32 %645 to ptr addrspace(3)
  %647 = load <4 x i32>, ptr addrspace(3) %646, align 16
  %648 = add i64 %639, 64
  %649 = trunc i64 %648 to i32
  %650 = add i32 %399, %649
  %651 = inttoptr i32 %650 to ptr addrspace(3)
  %652 = load <4 x i32>, ptr addrspace(3) %651, align 16
  %653 = add i64 %639, 96
  %654 = trunc i64 %653 to i32
  %655 = add i32 %399, %654
  %656 = inttoptr i32 %655 to ptr addrspace(3)
  %657 = load <4 x i32>, ptr addrspace(3) %656, align 16
  %658 = shufflevector <4 x i32> %642, <4 x i32> %647, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %659 = shufflevector <4 x i32> %652, <4 x i32> %657, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %660 = shufflevector <8 x i32> %658, <8 x i32> %659, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %661 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %624, i32 0, <16 x i32> %660, i16 0, <8 x float> %484, i32 0, i32 0, i32 %630, i32 0, i32 0, i32 %636, i1 false, i1 false)
  %662 = add i32 %409, 12544
  %663 = add i32 %662, %411
  %664 = add i32 %663, %413
  %665 = sext i32 %664 to i64
  %666 = add i32 %399, %664
  %667 = inttoptr i32 %666 to ptr addrspace(3)
  %668 = load <4 x i32>, ptr addrspace(3) %667, align 16
  %669 = add i64 %665, 512
  %670 = trunc i64 %669 to i32
  %671 = add i32 %399, %670
  %672 = inttoptr i32 %671 to ptr addrspace(3)
  %673 = load <4 x i32>, ptr addrspace(3) %672, align 16
  %674 = add i64 %665, 1024
  %675 = trunc i64 %674 to i32
  %676 = add i32 %399, %675
  %677 = inttoptr i32 %676 to ptr addrspace(3)
  %678 = load <4 x i32>, ptr addrspace(3) %677, align 16
  %679 = add i64 %665, 1536
  %680 = trunc i64 %679 to i32
  %681 = add i32 %399, %680
  %682 = inttoptr i32 %681 to ptr addrspace(3)
  %683 = load <4 x i32>, ptr addrspace(3) %682, align 16
  %684 = shufflevector <4 x i32> %668, <4 x i32> %673, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %685 = shufflevector <4 x i32> %678, <4 x i32> %683, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %686 = shufflevector <8 x i32> %684, <8 x i32> %685, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %687 = add i32 %445, 41282
  %688 = add i32 %397, %687
  %689 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %688
  %690 = load i8, ptr addrspace(3) %689, align 1
  %691 = zext i8 %690 to i32
  %692 = call i32 @llvm.amdgcn.perm(i32 %691, i32 %691, i32 0)
  %693 = add i32 %452, 41248
  %694 = add i32 %397, %693
  %695 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %694
  %696 = load i8, ptr addrspace(3) %695, align 1
  %697 = zext i8 %696 to i32
  %698 = call i32 @llvm.amdgcn.perm(i32 %697, i32 %697, i32 0)
  %699 = add i32 %459, 256
  %700 = add i32 %699, %460
  %701 = sext i32 %700 to i64
  %702 = add i32 %399, %700
  %703 = inttoptr i32 %702 to ptr addrspace(3)
  %704 = load <4 x i32>, ptr addrspace(3) %703, align 16
  %705 = add i64 %701, 32
  %706 = trunc i64 %705 to i32
  %707 = add i32 %399, %706
  %708 = inttoptr i32 %707 to ptr addrspace(3)
  %709 = load <4 x i32>, ptr addrspace(3) %708, align 16
  %710 = add i64 %701, 64
  %711 = trunc i64 %710 to i32
  %712 = add i32 %399, %711
  %713 = inttoptr i32 %712 to ptr addrspace(3)
  %714 = load <4 x i32>, ptr addrspace(3) %713, align 16
  %715 = add i64 %701, 96
  %716 = trunc i64 %715 to i32
  %717 = add i32 %399, %716
  %718 = inttoptr i32 %717 to ptr addrspace(3)
  %719 = load <4 x i32>, ptr addrspace(3) %718, align 16
  %720 = shufflevector <4 x i32> %704, <4 x i32> %709, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %721 = shufflevector <4 x i32> %714, <4 x i32> %719, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %722 = shufflevector <8 x i32> %720, <8 x i32> %721, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %723 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %686, i32 0, <16 x i32> %722, i16 0, <8 x float> %661, i32 0, i32 0, i32 %692, i32 0, i32 0, i32 %698, i1 false, i1 false)
  %724 = add i32 %409, 14592
  %725 = add i32 %724, %411
  %726 = add i32 %725, %413
  %727 = sext i32 %726 to i64
  %728 = add i32 %399, %726
  %729 = inttoptr i32 %728 to ptr addrspace(3)
  %730 = load <4 x i32>, ptr addrspace(3) %729, align 16
  %731 = add i64 %727, 512
  %732 = trunc i64 %731 to i32
  %733 = add i32 %399, %732
  %734 = inttoptr i32 %733 to ptr addrspace(3)
  %735 = load <4 x i32>, ptr addrspace(3) %734, align 16
  %736 = add i64 %727, 1024
  %737 = trunc i64 %736 to i32
  %738 = add i32 %399, %737
  %739 = inttoptr i32 %738 to ptr addrspace(3)
  %740 = load <4 x i32>, ptr addrspace(3) %739, align 16
  %741 = add i64 %727, 1536
  %742 = trunc i64 %741 to i32
  %743 = add i32 %399, %742
  %744 = inttoptr i32 %743 to ptr addrspace(3)
  %745 = load <4 x i32>, ptr addrspace(3) %744, align 16
  %746 = shufflevector <4 x i32> %730, <4 x i32> %735, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %747 = shufflevector <4 x i32> %740, <4 x i32> %745, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %748 = shufflevector <8 x i32> %746, <8 x i32> %747, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %749 = add i32 %445, 41283
  %750 = add i32 %397, %749
  %751 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %750
  %752 = load i8, ptr addrspace(3) %751, align 1
  %753 = zext i8 %752 to i32
  %754 = call i32 @llvm.amdgcn.perm(i32 %753, i32 %753, i32 0)
  %755 = add i32 %452, 41264
  %756 = add i32 %397, %755
  %757 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %756
  %758 = load i8, ptr addrspace(3) %757, align 1
  %759 = zext i8 %758 to i32
  %760 = call i32 @llvm.amdgcn.perm(i32 %759, i32 %759, i32 0)
  %761 = add i32 %459, 384
  %762 = add i32 %761, %460
  %763 = sext i32 %762 to i64
  %764 = add i32 %399, %762
  %765 = inttoptr i32 %764 to ptr addrspace(3)
  %766 = load <4 x i32>, ptr addrspace(3) %765, align 16
  %767 = add i64 %763, 32
  %768 = trunc i64 %767 to i32
  %769 = add i32 %399, %768
  %770 = inttoptr i32 %769 to ptr addrspace(3)
  %771 = load <4 x i32>, ptr addrspace(3) %770, align 16
  %772 = add i64 %763, 64
  %773 = trunc i64 %772 to i32
  %774 = add i32 %399, %773
  %775 = inttoptr i32 %774 to ptr addrspace(3)
  %776 = load <4 x i32>, ptr addrspace(3) %775, align 16
  %777 = add i64 %763, 96
  %778 = trunc i64 %777 to i32
  %779 = add i32 %399, %778
  %780 = inttoptr i32 %779 to ptr addrspace(3)
  %781 = load <4 x i32>, ptr addrspace(3) %780, align 16
  %782 = shufflevector <4 x i32> %766, <4 x i32> %771, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %783 = shufflevector <4 x i32> %776, <4 x i32> %781, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %784 = shufflevector <8 x i32> %782, <8 x i32> %783, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %785 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %748, i32 0, <16 x i32> %784, i16 0, <8 x float> %723, i32 0, i32 0, i32 %754, i32 0, i32 0, i32 %760, i1 false, i1 false)
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
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %786 = add i32 %392, 1
  br label %391

787:                                              ; preds = %391
  %788 = srem i32 %390, 4
  %789 = mul i32 %788, 41984
  %790 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %789
  %791 = ptrtoint ptr addrspace(3) %790 to i32
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 2)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %792 = sdiv i32 %99, 16
  %793 = mul i32 %792, 16
  %794 = icmp ne i32 %99, %793
  %795 = icmp slt i32 %99, 0
  %796 = icmp ne i1 %795, false
  %797 = and i1 %794, %796
  %798 = add i32 %792, -1
  %799 = select i1 %797, i32 %798, i32 %792
  %800 = mul i32 %799, 8192
  %801 = add i32 %800, 8448
  %802 = mul i32 %50, 256
  %803 = add i32 %801, %802
  %804 = mul i32 %42, 16
  %805 = add i32 %803, %804
  %806 = sext i32 %805 to i64
  %807 = add i32 %791, %805
  %808 = inttoptr i32 %807 to ptr addrspace(3)
  %809 = load <4 x i32>, ptr addrspace(3) %808, align 16
  %810 = add i64 %806, 512
  %811 = trunc i64 %810 to i32
  %812 = add i32 %791, %811
  %813 = inttoptr i32 %812 to ptr addrspace(3)
  %814 = load <4 x i32>, ptr addrspace(3) %813, align 16
  %815 = add i64 %806, 1024
  %816 = trunc i64 %815 to i32
  %817 = add i32 %791, %816
  %818 = inttoptr i32 %817 to ptr addrspace(3)
  %819 = load <4 x i32>, ptr addrspace(3) %818, align 16
  %820 = add i64 %806, 1536
  %821 = trunc i64 %820 to i32
  %822 = add i32 %791, %821
  %823 = inttoptr i32 %822 to ptr addrspace(3)
  %824 = load <4 x i32>, ptr addrspace(3) %823, align 16
  %825 = shufflevector <4 x i32> %809, <4 x i32> %814, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %826 = shufflevector <4 x i32> %819, <4 x i32> %824, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %827 = shufflevector <8 x i32> %825, <8 x i32> %826, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %828 = sdiv i32 %99, 128
  %829 = mul i32 %828, 128
  %830 = icmp ne i32 %99, %829
  %831 = icmp slt i32 %99, 0
  %832 = icmp ne i1 %831, false
  %833 = and i1 %830, %832
  %834 = add i32 %828, -1
  %835 = select i1 %833, i32 %834, i32 %828
  %836 = mul i32 %835, 4
  %837 = add i32 %836, 41280
  %838 = add i32 %789, %837
  %839 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %838
  %840 = load i8, ptr addrspace(3) %839, align 1
  %841 = zext i8 %840 to i32
  %842 = call i32 @llvm.amdgcn.perm(i32 %841, i32 %841, i32 0)
  %843 = add i32 %98, %42
  %844 = add i32 %843, 41216
  %845 = add i32 %789, %844
  %846 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %845
  %847 = load i8, ptr addrspace(3) %846, align 1
  %848 = zext i8 %847 to i32
  %849 = call i32 @llvm.amdgcn.perm(i32 %848, i32 %848, i32 0)
  %850 = mul i32 %843, 528
  %851 = mul i32 %50, 16
  %852 = add i32 %850, %851
  %853 = sext i32 %852 to i64
  %854 = add i32 %791, %852
  %855 = inttoptr i32 %854 to ptr addrspace(3)
  %856 = load <4 x i32>, ptr addrspace(3) %855, align 16
  %857 = add i64 %853, 32
  %858 = trunc i64 %857 to i32
  %859 = add i32 %791, %858
  %860 = inttoptr i32 %859 to ptr addrspace(3)
  %861 = load <4 x i32>, ptr addrspace(3) %860, align 16
  %862 = add i64 %853, 64
  %863 = trunc i64 %862 to i32
  %864 = add i32 %791, %863
  %865 = inttoptr i32 %864 to ptr addrspace(3)
  %866 = load <4 x i32>, ptr addrspace(3) %865, align 16
  %867 = add i64 %853, 96
  %868 = trunc i64 %867 to i32
  %869 = add i32 %791, %868
  %870 = inttoptr i32 %869 to ptr addrspace(3)
  %871 = load <4 x i32>, ptr addrspace(3) %870, align 16
  %872 = shufflevector <4 x i32> %856, <4 x i32> %861, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %873 = shufflevector <4 x i32> %866, <4 x i32> %871, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %874 = shufflevector <8 x i32> %872, <8 x i32> %873, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %875 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %827, i32 0, <16 x i32> %874, i16 0, <8 x float> %393, i32 0, i32 0, i32 %842, i32 0, i32 0, i32 %849, i1 false, i1 false)
  %876 = add i32 %800, 10496
  %877 = add i32 %876, %802
  %878 = add i32 %877, %804
  %879 = sext i32 %878 to i64
  %880 = add i32 %791, %878
  %881 = inttoptr i32 %880 to ptr addrspace(3)
  %882 = load <4 x i32>, ptr addrspace(3) %881, align 16
  %883 = add i64 %879, 512
  %884 = trunc i64 %883 to i32
  %885 = add i32 %791, %884
  %886 = inttoptr i32 %885 to ptr addrspace(3)
  %887 = load <4 x i32>, ptr addrspace(3) %886, align 16
  %888 = add i64 %879, 1024
  %889 = trunc i64 %888 to i32
  %890 = add i32 %791, %889
  %891 = inttoptr i32 %890 to ptr addrspace(3)
  %892 = load <4 x i32>, ptr addrspace(3) %891, align 16
  %893 = add i64 %879, 1536
  %894 = trunc i64 %893 to i32
  %895 = add i32 %791, %894
  %896 = inttoptr i32 %895 to ptr addrspace(3)
  %897 = load <4 x i32>, ptr addrspace(3) %896, align 16
  %898 = shufflevector <4 x i32> %882, <4 x i32> %887, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %899 = shufflevector <4 x i32> %892, <4 x i32> %897, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %900 = shufflevector <8 x i32> %898, <8 x i32> %899, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %901 = add i32 %836, 41281
  %902 = add i32 %789, %901
  %903 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %902
  %904 = load i8, ptr addrspace(3) %903, align 1
  %905 = zext i8 %904 to i32
  %906 = call i32 @llvm.amdgcn.perm(i32 %905, i32 %905, i32 0)
  %907 = add i32 %843, 41232
  %908 = add i32 %789, %907
  %909 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %908
  %910 = load i8, ptr addrspace(3) %909, align 1
  %911 = zext i8 %910 to i32
  %912 = call i32 @llvm.amdgcn.perm(i32 %911, i32 %911, i32 0)
  %913 = add i32 %850, 128
  %914 = add i32 %913, %851
  %915 = sext i32 %914 to i64
  %916 = add i32 %791, %914
  %917 = inttoptr i32 %916 to ptr addrspace(3)
  %918 = load <4 x i32>, ptr addrspace(3) %917, align 16
  %919 = add i64 %915, 32
  %920 = trunc i64 %919 to i32
  %921 = add i32 %791, %920
  %922 = inttoptr i32 %921 to ptr addrspace(3)
  %923 = load <4 x i32>, ptr addrspace(3) %922, align 16
  %924 = add i64 %915, 64
  %925 = trunc i64 %924 to i32
  %926 = add i32 %791, %925
  %927 = inttoptr i32 %926 to ptr addrspace(3)
  %928 = load <4 x i32>, ptr addrspace(3) %927, align 16
  %929 = add i64 %915, 96
  %930 = trunc i64 %929 to i32
  %931 = add i32 %791, %930
  %932 = inttoptr i32 %931 to ptr addrspace(3)
  %933 = load <4 x i32>, ptr addrspace(3) %932, align 16
  %934 = shufflevector <4 x i32> %918, <4 x i32> %923, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %935 = shufflevector <4 x i32> %928, <4 x i32> %933, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %936 = shufflevector <8 x i32> %934, <8 x i32> %935, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %937 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %900, i32 0, <16 x i32> %936, i16 0, <8 x float> %875, i32 0, i32 0, i32 %906, i32 0, i32 0, i32 %912, i1 false, i1 false)
  %938 = add i32 %800, 12544
  %939 = add i32 %938, %802
  %940 = add i32 %939, %804
  %941 = sext i32 %940 to i64
  %942 = add i32 %791, %940
  %943 = inttoptr i32 %942 to ptr addrspace(3)
  %944 = load <4 x i32>, ptr addrspace(3) %943, align 16
  %945 = add i64 %941, 512
  %946 = trunc i64 %945 to i32
  %947 = add i32 %791, %946
  %948 = inttoptr i32 %947 to ptr addrspace(3)
  %949 = load <4 x i32>, ptr addrspace(3) %948, align 16
  %950 = add i64 %941, 1024
  %951 = trunc i64 %950 to i32
  %952 = add i32 %791, %951
  %953 = inttoptr i32 %952 to ptr addrspace(3)
  %954 = load <4 x i32>, ptr addrspace(3) %953, align 16
  %955 = add i64 %941, 1536
  %956 = trunc i64 %955 to i32
  %957 = add i32 %791, %956
  %958 = inttoptr i32 %957 to ptr addrspace(3)
  %959 = load <4 x i32>, ptr addrspace(3) %958, align 16
  %960 = shufflevector <4 x i32> %944, <4 x i32> %949, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %961 = shufflevector <4 x i32> %954, <4 x i32> %959, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %962 = shufflevector <8 x i32> %960, <8 x i32> %961, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %963 = add i32 %836, 41282
  %964 = add i32 %789, %963
  %965 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %964
  %966 = load i8, ptr addrspace(3) %965, align 1
  %967 = zext i8 %966 to i32
  %968 = call i32 @llvm.amdgcn.perm(i32 %967, i32 %967, i32 0)
  %969 = add i32 %843, 41248
  %970 = add i32 %789, %969
  %971 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %970
  %972 = load i8, ptr addrspace(3) %971, align 1
  %973 = zext i8 %972 to i32
  %974 = call i32 @llvm.amdgcn.perm(i32 %973, i32 %973, i32 0)
  %975 = add i32 %850, 256
  %976 = add i32 %975, %851
  %977 = sext i32 %976 to i64
  %978 = add i32 %791, %976
  %979 = inttoptr i32 %978 to ptr addrspace(3)
  %980 = load <4 x i32>, ptr addrspace(3) %979, align 16
  %981 = add i64 %977, 32
  %982 = trunc i64 %981 to i32
  %983 = add i32 %791, %982
  %984 = inttoptr i32 %983 to ptr addrspace(3)
  %985 = load <4 x i32>, ptr addrspace(3) %984, align 16
  %986 = add i64 %977, 64
  %987 = trunc i64 %986 to i32
  %988 = add i32 %791, %987
  %989 = inttoptr i32 %988 to ptr addrspace(3)
  %990 = load <4 x i32>, ptr addrspace(3) %989, align 16
  %991 = add i64 %977, 96
  %992 = trunc i64 %991 to i32
  %993 = add i32 %791, %992
  %994 = inttoptr i32 %993 to ptr addrspace(3)
  %995 = load <4 x i32>, ptr addrspace(3) %994, align 16
  %996 = shufflevector <4 x i32> %980, <4 x i32> %985, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %997 = shufflevector <4 x i32> %990, <4 x i32> %995, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %998 = shufflevector <8 x i32> %996, <8 x i32> %997, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %999 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %962, i32 0, <16 x i32> %998, i16 0, <8 x float> %937, i32 0, i32 0, i32 %968, i32 0, i32 0, i32 %974, i1 false, i1 false)
  %1000 = add i32 %800, 14592
  %1001 = add i32 %1000, %802
  %1002 = add i32 %1001, %804
  %1003 = sext i32 %1002 to i64
  %1004 = add i32 %791, %1002
  %1005 = inttoptr i32 %1004 to ptr addrspace(3)
  %1006 = load <4 x i32>, ptr addrspace(3) %1005, align 16
  %1007 = add i64 %1003, 512
  %1008 = trunc i64 %1007 to i32
  %1009 = add i32 %791, %1008
  %1010 = inttoptr i32 %1009 to ptr addrspace(3)
  %1011 = load <4 x i32>, ptr addrspace(3) %1010, align 16
  %1012 = add i64 %1003, 1024
  %1013 = trunc i64 %1012 to i32
  %1014 = add i32 %791, %1013
  %1015 = inttoptr i32 %1014 to ptr addrspace(3)
  %1016 = load <4 x i32>, ptr addrspace(3) %1015, align 16
  %1017 = add i64 %1003, 1536
  %1018 = trunc i64 %1017 to i32
  %1019 = add i32 %791, %1018
  %1020 = inttoptr i32 %1019 to ptr addrspace(3)
  %1021 = load <4 x i32>, ptr addrspace(3) %1020, align 16
  %1022 = shufflevector <4 x i32> %1006, <4 x i32> %1011, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1023 = shufflevector <4 x i32> %1016, <4 x i32> %1021, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1024 = shufflevector <8 x i32> %1022, <8 x i32> %1023, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1025 = add i32 %836, 41283
  %1026 = add i32 %789, %1025
  %1027 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1026
  %1028 = load i8, ptr addrspace(3) %1027, align 1
  %1029 = zext i8 %1028 to i32
  %1030 = call i32 @llvm.amdgcn.perm(i32 %1029, i32 %1029, i32 0)
  %1031 = add i32 %843, 41264
  %1032 = add i32 %789, %1031
  %1033 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1032
  %1034 = load i8, ptr addrspace(3) %1033, align 1
  %1035 = zext i8 %1034 to i32
  %1036 = call i32 @llvm.amdgcn.perm(i32 %1035, i32 %1035, i32 0)
  %1037 = add i32 %850, 384
  %1038 = add i32 %1037, %851
  %1039 = sext i32 %1038 to i64
  %1040 = add i32 %791, %1038
  %1041 = inttoptr i32 %1040 to ptr addrspace(3)
  %1042 = load <4 x i32>, ptr addrspace(3) %1041, align 16
  %1043 = add i64 %1039, 32
  %1044 = trunc i64 %1043 to i32
  %1045 = add i32 %791, %1044
  %1046 = inttoptr i32 %1045 to ptr addrspace(3)
  %1047 = load <4 x i32>, ptr addrspace(3) %1046, align 16
  %1048 = add i64 %1039, 64
  %1049 = trunc i64 %1048 to i32
  %1050 = add i32 %791, %1049
  %1051 = inttoptr i32 %1050 to ptr addrspace(3)
  %1052 = load <4 x i32>, ptr addrspace(3) %1051, align 16
  %1053 = add i64 %1039, 96
  %1054 = trunc i64 %1053 to i32
  %1055 = add i32 %791, %1054
  %1056 = inttoptr i32 %1055 to ptr addrspace(3)
  %1057 = load <4 x i32>, ptr addrspace(3) %1056, align 16
  %1058 = shufflevector <4 x i32> %1042, <4 x i32> %1047, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1059 = shufflevector <4 x i32> %1052, <4 x i32> %1057, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1060 = shufflevector <8 x i32> %1058, <8 x i32> %1059, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1061 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1024, i32 0, <16 x i32> %1060, i16 0, <8 x float> %999, i32 0, i32 0, i32 %1030, i32 0, i32 0, i32 %1036, i1 false, i1 false)
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
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1062 = add i32 %19, -2
  %1063 = srem i32 %1062, 4
  %1064 = mul i32 %1063, 41984
  %1065 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1064
  %1066 = ptrtoint ptr addrspace(3) %1065 to i32
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 1)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %1067 = add i32 %1066, %805
  %1068 = inttoptr i32 %1067 to ptr addrspace(3)
  %1069 = load <4 x i32>, ptr addrspace(3) %1068, align 16
  %1070 = add i32 %1066, %811
  %1071 = inttoptr i32 %1070 to ptr addrspace(3)
  %1072 = load <4 x i32>, ptr addrspace(3) %1071, align 16
  %1073 = add i32 %1066, %816
  %1074 = inttoptr i32 %1073 to ptr addrspace(3)
  %1075 = load <4 x i32>, ptr addrspace(3) %1074, align 16
  %1076 = add i32 %1066, %821
  %1077 = inttoptr i32 %1076 to ptr addrspace(3)
  %1078 = load <4 x i32>, ptr addrspace(3) %1077, align 16
  %1079 = shufflevector <4 x i32> %1069, <4 x i32> %1072, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1080 = shufflevector <4 x i32> %1075, <4 x i32> %1078, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1081 = shufflevector <8 x i32> %1079, <8 x i32> %1080, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1082 = add i32 %1064, %837
  %1083 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1082
  %1084 = load i8, ptr addrspace(3) %1083, align 1
  %1085 = zext i8 %1084 to i32
  %1086 = call i32 @llvm.amdgcn.perm(i32 %1085, i32 %1085, i32 0)
  %1087 = add i32 %1064, %844
  %1088 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1087
  %1089 = load i8, ptr addrspace(3) %1088, align 1
  %1090 = zext i8 %1089 to i32
  %1091 = call i32 @llvm.amdgcn.perm(i32 %1090, i32 %1090, i32 0)
  %1092 = add i32 %1066, %852
  %1093 = inttoptr i32 %1092 to ptr addrspace(3)
  %1094 = load <4 x i32>, ptr addrspace(3) %1093, align 16
  %1095 = add i32 %1066, %858
  %1096 = inttoptr i32 %1095 to ptr addrspace(3)
  %1097 = load <4 x i32>, ptr addrspace(3) %1096, align 16
  %1098 = add i32 %1066, %863
  %1099 = inttoptr i32 %1098 to ptr addrspace(3)
  %1100 = load <4 x i32>, ptr addrspace(3) %1099, align 16
  %1101 = add i32 %1066, %868
  %1102 = inttoptr i32 %1101 to ptr addrspace(3)
  %1103 = load <4 x i32>, ptr addrspace(3) %1102, align 16
  %1104 = shufflevector <4 x i32> %1094, <4 x i32> %1097, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1105 = shufflevector <4 x i32> %1100, <4 x i32> %1103, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1106 = shufflevector <8 x i32> %1104, <8 x i32> %1105, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1107 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1081, i32 0, <16 x i32> %1106, i16 0, <8 x float> %1061, i32 0, i32 0, i32 %1086, i32 0, i32 0, i32 %1091, i1 false, i1 false)
  %1108 = add i32 %1066, %878
  %1109 = inttoptr i32 %1108 to ptr addrspace(3)
  %1110 = load <4 x i32>, ptr addrspace(3) %1109, align 16
  %1111 = add i32 %1066, %884
  %1112 = inttoptr i32 %1111 to ptr addrspace(3)
  %1113 = load <4 x i32>, ptr addrspace(3) %1112, align 16
  %1114 = add i32 %1066, %889
  %1115 = inttoptr i32 %1114 to ptr addrspace(3)
  %1116 = load <4 x i32>, ptr addrspace(3) %1115, align 16
  %1117 = add i32 %1066, %894
  %1118 = inttoptr i32 %1117 to ptr addrspace(3)
  %1119 = load <4 x i32>, ptr addrspace(3) %1118, align 16
  %1120 = shufflevector <4 x i32> %1110, <4 x i32> %1113, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1121 = shufflevector <4 x i32> %1116, <4 x i32> %1119, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1122 = shufflevector <8 x i32> %1120, <8 x i32> %1121, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1123 = add i32 %1064, %901
  %1124 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1123
  %1125 = load i8, ptr addrspace(3) %1124, align 1
  %1126 = zext i8 %1125 to i32
  %1127 = call i32 @llvm.amdgcn.perm(i32 %1126, i32 %1126, i32 0)
  %1128 = add i32 %1064, %907
  %1129 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1128
  %1130 = load i8, ptr addrspace(3) %1129, align 1
  %1131 = zext i8 %1130 to i32
  %1132 = call i32 @llvm.amdgcn.perm(i32 %1131, i32 %1131, i32 0)
  %1133 = add i32 %1066, %914
  %1134 = inttoptr i32 %1133 to ptr addrspace(3)
  %1135 = load <4 x i32>, ptr addrspace(3) %1134, align 16
  %1136 = add i32 %1066, %920
  %1137 = inttoptr i32 %1136 to ptr addrspace(3)
  %1138 = load <4 x i32>, ptr addrspace(3) %1137, align 16
  %1139 = add i32 %1066, %925
  %1140 = inttoptr i32 %1139 to ptr addrspace(3)
  %1141 = load <4 x i32>, ptr addrspace(3) %1140, align 16
  %1142 = add i32 %1066, %930
  %1143 = inttoptr i32 %1142 to ptr addrspace(3)
  %1144 = load <4 x i32>, ptr addrspace(3) %1143, align 16
  %1145 = shufflevector <4 x i32> %1135, <4 x i32> %1138, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1146 = shufflevector <4 x i32> %1141, <4 x i32> %1144, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1147 = shufflevector <8 x i32> %1145, <8 x i32> %1146, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1148 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1122, i32 0, <16 x i32> %1147, i16 0, <8 x float> %1107, i32 0, i32 0, i32 %1127, i32 0, i32 0, i32 %1132, i1 false, i1 false)
  %1149 = add i32 %1066, %940
  %1150 = inttoptr i32 %1149 to ptr addrspace(3)
  %1151 = load <4 x i32>, ptr addrspace(3) %1150, align 16
  %1152 = add i32 %1066, %946
  %1153 = inttoptr i32 %1152 to ptr addrspace(3)
  %1154 = load <4 x i32>, ptr addrspace(3) %1153, align 16
  %1155 = add i32 %1066, %951
  %1156 = inttoptr i32 %1155 to ptr addrspace(3)
  %1157 = load <4 x i32>, ptr addrspace(3) %1156, align 16
  %1158 = add i32 %1066, %956
  %1159 = inttoptr i32 %1158 to ptr addrspace(3)
  %1160 = load <4 x i32>, ptr addrspace(3) %1159, align 16
  %1161 = shufflevector <4 x i32> %1151, <4 x i32> %1154, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1162 = shufflevector <4 x i32> %1157, <4 x i32> %1160, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1163 = shufflevector <8 x i32> %1161, <8 x i32> %1162, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1164 = add i32 %1064, %963
  %1165 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1164
  %1166 = load i8, ptr addrspace(3) %1165, align 1
  %1167 = zext i8 %1166 to i32
  %1168 = call i32 @llvm.amdgcn.perm(i32 %1167, i32 %1167, i32 0)
  %1169 = add i32 %1064, %969
  %1170 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1169
  %1171 = load i8, ptr addrspace(3) %1170, align 1
  %1172 = zext i8 %1171 to i32
  %1173 = call i32 @llvm.amdgcn.perm(i32 %1172, i32 %1172, i32 0)
  %1174 = add i32 %1066, %976
  %1175 = inttoptr i32 %1174 to ptr addrspace(3)
  %1176 = load <4 x i32>, ptr addrspace(3) %1175, align 16
  %1177 = add i32 %1066, %982
  %1178 = inttoptr i32 %1177 to ptr addrspace(3)
  %1179 = load <4 x i32>, ptr addrspace(3) %1178, align 16
  %1180 = add i32 %1066, %987
  %1181 = inttoptr i32 %1180 to ptr addrspace(3)
  %1182 = load <4 x i32>, ptr addrspace(3) %1181, align 16
  %1183 = add i32 %1066, %992
  %1184 = inttoptr i32 %1183 to ptr addrspace(3)
  %1185 = load <4 x i32>, ptr addrspace(3) %1184, align 16
  %1186 = shufflevector <4 x i32> %1176, <4 x i32> %1179, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1187 = shufflevector <4 x i32> %1182, <4 x i32> %1185, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1188 = shufflevector <8 x i32> %1186, <8 x i32> %1187, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1189 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1163, i32 0, <16 x i32> %1188, i16 0, <8 x float> %1148, i32 0, i32 0, i32 %1168, i32 0, i32 0, i32 %1173, i1 false, i1 false)
  %1190 = add i32 %1066, %1002
  %1191 = inttoptr i32 %1190 to ptr addrspace(3)
  %1192 = load <4 x i32>, ptr addrspace(3) %1191, align 16
  %1193 = add i32 %1066, %1008
  %1194 = inttoptr i32 %1193 to ptr addrspace(3)
  %1195 = load <4 x i32>, ptr addrspace(3) %1194, align 16
  %1196 = add i32 %1066, %1013
  %1197 = inttoptr i32 %1196 to ptr addrspace(3)
  %1198 = load <4 x i32>, ptr addrspace(3) %1197, align 16
  %1199 = add i32 %1066, %1018
  %1200 = inttoptr i32 %1199 to ptr addrspace(3)
  %1201 = load <4 x i32>, ptr addrspace(3) %1200, align 16
  %1202 = shufflevector <4 x i32> %1192, <4 x i32> %1195, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1203 = shufflevector <4 x i32> %1198, <4 x i32> %1201, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1204 = shufflevector <8 x i32> %1202, <8 x i32> %1203, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1205 = add i32 %1064, %1025
  %1206 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1205
  %1207 = load i8, ptr addrspace(3) %1206, align 1
  %1208 = zext i8 %1207 to i32
  %1209 = call i32 @llvm.amdgcn.perm(i32 %1208, i32 %1208, i32 0)
  %1210 = add i32 %1064, %1031
  %1211 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1210
  %1212 = load i8, ptr addrspace(3) %1211, align 1
  %1213 = zext i8 %1212 to i32
  %1214 = call i32 @llvm.amdgcn.perm(i32 %1213, i32 %1213, i32 0)
  %1215 = add i32 %1066, %1038
  %1216 = inttoptr i32 %1215 to ptr addrspace(3)
  %1217 = load <4 x i32>, ptr addrspace(3) %1216, align 16
  %1218 = add i32 %1066, %1044
  %1219 = inttoptr i32 %1218 to ptr addrspace(3)
  %1220 = load <4 x i32>, ptr addrspace(3) %1219, align 16
  %1221 = add i32 %1066, %1049
  %1222 = inttoptr i32 %1221 to ptr addrspace(3)
  %1223 = load <4 x i32>, ptr addrspace(3) %1222, align 16
  %1224 = add i32 %1066, %1054
  %1225 = inttoptr i32 %1224 to ptr addrspace(3)
  %1226 = load <4 x i32>, ptr addrspace(3) %1225, align 16
  %1227 = shufflevector <4 x i32> %1217, <4 x i32> %1220, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1228 = shufflevector <4 x i32> %1223, <4 x i32> %1226, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1229 = shufflevector <8 x i32> %1227, <8 x i32> %1228, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1230 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1204, i32 0, <16 x i32> %1229, i16 0, <8 x float> %1189, i32 0, i32 0, i32 %1209, i32 0, i32 0, i32 %1214, i1 false, i1 false)
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
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1231 = add i32 %19, -1
  %1232 = srem i32 %1231, 4
  %1233 = mul i32 %1232, 41984
  %1234 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1233
  %1235 = ptrtoint ptr addrspace(3) %1234 to i32
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %1236 = add i32 %1235, %805
  %1237 = inttoptr i32 %1236 to ptr addrspace(3)
  %1238 = load <4 x i32>, ptr addrspace(3) %1237, align 16
  %1239 = add i32 %1235, %811
  %1240 = inttoptr i32 %1239 to ptr addrspace(3)
  %1241 = load <4 x i32>, ptr addrspace(3) %1240, align 16
  %1242 = add i32 %1235, %816
  %1243 = inttoptr i32 %1242 to ptr addrspace(3)
  %1244 = load <4 x i32>, ptr addrspace(3) %1243, align 16
  %1245 = add i32 %1235, %821
  %1246 = inttoptr i32 %1245 to ptr addrspace(3)
  %1247 = load <4 x i32>, ptr addrspace(3) %1246, align 16
  %1248 = shufflevector <4 x i32> %1238, <4 x i32> %1241, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1249 = shufflevector <4 x i32> %1244, <4 x i32> %1247, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1250 = shufflevector <8 x i32> %1248, <8 x i32> %1249, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1251 = add i32 %1233, %837
  %1252 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1251
  %1253 = load i8, ptr addrspace(3) %1252, align 1
  %1254 = zext i8 %1253 to i32
  %1255 = call i32 @llvm.amdgcn.perm(i32 %1254, i32 %1254, i32 0)
  %1256 = add i32 %1233, %844
  %1257 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1256
  %1258 = load i8, ptr addrspace(3) %1257, align 1
  %1259 = zext i8 %1258 to i32
  %1260 = call i32 @llvm.amdgcn.perm(i32 %1259, i32 %1259, i32 0)
  %1261 = add i32 %1235, %852
  %1262 = inttoptr i32 %1261 to ptr addrspace(3)
  %1263 = load <4 x i32>, ptr addrspace(3) %1262, align 16
  %1264 = add i32 %1235, %858
  %1265 = inttoptr i32 %1264 to ptr addrspace(3)
  %1266 = load <4 x i32>, ptr addrspace(3) %1265, align 16
  %1267 = add i32 %1235, %863
  %1268 = inttoptr i32 %1267 to ptr addrspace(3)
  %1269 = load <4 x i32>, ptr addrspace(3) %1268, align 16
  %1270 = add i32 %1235, %868
  %1271 = inttoptr i32 %1270 to ptr addrspace(3)
  %1272 = load <4 x i32>, ptr addrspace(3) %1271, align 16
  %1273 = shufflevector <4 x i32> %1263, <4 x i32> %1266, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1274 = shufflevector <4 x i32> %1269, <4 x i32> %1272, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1275 = shufflevector <8 x i32> %1273, <8 x i32> %1274, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1276 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1250, i32 0, <16 x i32> %1275, i16 0, <8 x float> %1230, i32 0, i32 0, i32 %1255, i32 0, i32 0, i32 %1260, i1 false, i1 false)
  %1277 = add i32 %1235, %878
  %1278 = inttoptr i32 %1277 to ptr addrspace(3)
  %1279 = load <4 x i32>, ptr addrspace(3) %1278, align 16
  %1280 = add i32 %1235, %884
  %1281 = inttoptr i32 %1280 to ptr addrspace(3)
  %1282 = load <4 x i32>, ptr addrspace(3) %1281, align 16
  %1283 = add i32 %1235, %889
  %1284 = inttoptr i32 %1283 to ptr addrspace(3)
  %1285 = load <4 x i32>, ptr addrspace(3) %1284, align 16
  %1286 = add i32 %1235, %894
  %1287 = inttoptr i32 %1286 to ptr addrspace(3)
  %1288 = load <4 x i32>, ptr addrspace(3) %1287, align 16
  %1289 = shufflevector <4 x i32> %1279, <4 x i32> %1282, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1290 = shufflevector <4 x i32> %1285, <4 x i32> %1288, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1291 = shufflevector <8 x i32> %1289, <8 x i32> %1290, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1292 = add i32 %1233, %901
  %1293 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1292
  %1294 = load i8, ptr addrspace(3) %1293, align 1
  %1295 = zext i8 %1294 to i32
  %1296 = call i32 @llvm.amdgcn.perm(i32 %1295, i32 %1295, i32 0)
  %1297 = add i32 %1233, %907
  %1298 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1297
  %1299 = load i8, ptr addrspace(3) %1298, align 1
  %1300 = zext i8 %1299 to i32
  %1301 = call i32 @llvm.amdgcn.perm(i32 %1300, i32 %1300, i32 0)
  %1302 = add i32 %1235, %914
  %1303 = inttoptr i32 %1302 to ptr addrspace(3)
  %1304 = load <4 x i32>, ptr addrspace(3) %1303, align 16
  %1305 = add i32 %1235, %920
  %1306 = inttoptr i32 %1305 to ptr addrspace(3)
  %1307 = load <4 x i32>, ptr addrspace(3) %1306, align 16
  %1308 = add i32 %1235, %925
  %1309 = inttoptr i32 %1308 to ptr addrspace(3)
  %1310 = load <4 x i32>, ptr addrspace(3) %1309, align 16
  %1311 = add i32 %1235, %930
  %1312 = inttoptr i32 %1311 to ptr addrspace(3)
  %1313 = load <4 x i32>, ptr addrspace(3) %1312, align 16
  %1314 = shufflevector <4 x i32> %1304, <4 x i32> %1307, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1315 = shufflevector <4 x i32> %1310, <4 x i32> %1313, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1316 = shufflevector <8 x i32> %1314, <8 x i32> %1315, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1317 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1291, i32 0, <16 x i32> %1316, i16 0, <8 x float> %1276, i32 0, i32 0, i32 %1296, i32 0, i32 0, i32 %1301, i1 false, i1 false)
  %1318 = add i32 %1235, %940
  %1319 = inttoptr i32 %1318 to ptr addrspace(3)
  %1320 = load <4 x i32>, ptr addrspace(3) %1319, align 16
  %1321 = add i32 %1235, %946
  %1322 = inttoptr i32 %1321 to ptr addrspace(3)
  %1323 = load <4 x i32>, ptr addrspace(3) %1322, align 16
  %1324 = add i32 %1235, %951
  %1325 = inttoptr i32 %1324 to ptr addrspace(3)
  %1326 = load <4 x i32>, ptr addrspace(3) %1325, align 16
  %1327 = add i32 %1235, %956
  %1328 = inttoptr i32 %1327 to ptr addrspace(3)
  %1329 = load <4 x i32>, ptr addrspace(3) %1328, align 16
  %1330 = shufflevector <4 x i32> %1320, <4 x i32> %1323, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1331 = shufflevector <4 x i32> %1326, <4 x i32> %1329, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1332 = shufflevector <8 x i32> %1330, <8 x i32> %1331, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1333 = add i32 %1233, %963
  %1334 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1333
  %1335 = load i8, ptr addrspace(3) %1334, align 1
  %1336 = zext i8 %1335 to i32
  %1337 = call i32 @llvm.amdgcn.perm(i32 %1336, i32 %1336, i32 0)
  %1338 = add i32 %1233, %969
  %1339 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1338
  %1340 = load i8, ptr addrspace(3) %1339, align 1
  %1341 = zext i8 %1340 to i32
  %1342 = call i32 @llvm.amdgcn.perm(i32 %1341, i32 %1341, i32 0)
  %1343 = add i32 %1235, %976
  %1344 = inttoptr i32 %1343 to ptr addrspace(3)
  %1345 = load <4 x i32>, ptr addrspace(3) %1344, align 16
  %1346 = add i32 %1235, %982
  %1347 = inttoptr i32 %1346 to ptr addrspace(3)
  %1348 = load <4 x i32>, ptr addrspace(3) %1347, align 16
  %1349 = add i32 %1235, %987
  %1350 = inttoptr i32 %1349 to ptr addrspace(3)
  %1351 = load <4 x i32>, ptr addrspace(3) %1350, align 16
  %1352 = add i32 %1235, %992
  %1353 = inttoptr i32 %1352 to ptr addrspace(3)
  %1354 = load <4 x i32>, ptr addrspace(3) %1353, align 16
  %1355 = shufflevector <4 x i32> %1345, <4 x i32> %1348, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1356 = shufflevector <4 x i32> %1351, <4 x i32> %1354, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1357 = shufflevector <8 x i32> %1355, <8 x i32> %1356, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1358 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1332, i32 0, <16 x i32> %1357, i16 0, <8 x float> %1317, i32 0, i32 0, i32 %1337, i32 0, i32 0, i32 %1342, i1 false, i1 false)
  %1359 = add i32 %1235, %1002
  %1360 = inttoptr i32 %1359 to ptr addrspace(3)
  %1361 = load <4 x i32>, ptr addrspace(3) %1360, align 16
  %1362 = add i32 %1235, %1008
  %1363 = inttoptr i32 %1362 to ptr addrspace(3)
  %1364 = load <4 x i32>, ptr addrspace(3) %1363, align 16
  %1365 = add i32 %1235, %1013
  %1366 = inttoptr i32 %1365 to ptr addrspace(3)
  %1367 = load <4 x i32>, ptr addrspace(3) %1366, align 16
  %1368 = add i32 %1235, %1018
  %1369 = inttoptr i32 %1368 to ptr addrspace(3)
  %1370 = load <4 x i32>, ptr addrspace(3) %1369, align 16
  %1371 = shufflevector <4 x i32> %1361, <4 x i32> %1364, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1372 = shufflevector <4 x i32> %1367, <4 x i32> %1370, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1373 = shufflevector <8 x i32> %1371, <8 x i32> %1372, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1374 = add i32 %1233, %1025
  %1375 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1374
  %1376 = load i8, ptr addrspace(3) %1375, align 1
  %1377 = zext i8 %1376 to i32
  %1378 = call i32 @llvm.amdgcn.perm(i32 %1377, i32 %1377, i32 0)
  %1379 = add i32 %1233, %1031
  %1380 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1379
  %1381 = load i8, ptr addrspace(3) %1380, align 1
  %1382 = zext i8 %1381 to i32
  %1383 = call i32 @llvm.amdgcn.perm(i32 %1382, i32 %1382, i32 0)
  %1384 = add i32 %1235, %1038
  %1385 = inttoptr i32 %1384 to ptr addrspace(3)
  %1386 = load <4 x i32>, ptr addrspace(3) %1385, align 16
  %1387 = add i32 %1235, %1044
  %1388 = inttoptr i32 %1387 to ptr addrspace(3)
  %1389 = load <4 x i32>, ptr addrspace(3) %1388, align 16
  %1390 = add i32 %1235, %1049
  %1391 = inttoptr i32 %1390 to ptr addrspace(3)
  %1392 = load <4 x i32>, ptr addrspace(3) %1391, align 16
  %1393 = add i32 %1235, %1054
  %1394 = inttoptr i32 %1393 to ptr addrspace(3)
  %1395 = load <4 x i32>, ptr addrspace(3) %1394, align 16
  %1396 = shufflevector <4 x i32> %1386, <4 x i32> %1389, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1397 = shufflevector <4 x i32> %1392, <4 x i32> %1395, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1398 = shufflevector <8 x i32> %1396, <8 x i32> %1397, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.s.wait.dscnt(i16 0)
  %1399 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %1373, i32 0, <16 x i32> %1398, i16 0, <8 x float> %1358, i32 0, i32 0, i32 %1378, i32 0, i32 0, i32 %1383, i1 false, i1 false)
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
  call void @llvm.amdgcn.sched.barrier(i32 0)
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %1400 = mul i32 %50, 8
  %1401 = add i32 %99, %1400
  %1402 = fptrunc <8 x float> %1399 to <8 x bfloat>
  %1403 = bitcast <8 x bfloat> %1402 to <16 x i8>
  %1404 = mul i32 %843, 64
  %1405 = add i32 %1404, %1401
  %1406 = mul i32 %1405, 2
  %1407 = getelementptr i8, ptr addrspace(3) @__dynamic_shared_0, i32 %1406
  store <16 x i8> %1403, ptr addrspace(3) %1407, align 1
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %1408 = mul i64 %62, %22
  %1409 = add i64 %1408, %63
  %1410 = getelementptr bfloat, ptr addrspace(1) %0, i64 %1409
  %1411 = icmp eq i64 %22, -2147483648
  %1412 = select i1 %1411, i64 64, i64 %22
  %1413 = ptrtoint ptr addrspace(1) %1410 to i64
  %1414 = call i32 @llvm.amdgcn.wave.id()
  %1415 = urem i32 %1414, 4
  %1416 = mul i32 %1415, 4
  %1417 = zext i32 %1416 to i64
  %1418 = mul i64 %1417, %1412
  %1419 = mul i64 %1418, 2
  %1420 = add i64 %1413, %1419
  %1421 = mul i32 %1415, 512
  %1422 = add i32 ptrtoint (ptr addrspace(3) @__dynamic_shared_0 to i32), %1421
  %1423 = trunc i64 %1420 to i32
  %1424 = lshr i64 %1420, 32
  %1425 = trunc i64 %1424 to i32
  %1426 = or i32 %1425, -2147483648
  %1427 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %1422, i64 1
  %1428 = insertelement <4 x i32> %1427, i32 %1423, i64 2
  %1429 = insertelement <4 x i32> %1428, i32 %1426, i64 3
  %1430 = sub i32 %64, %1416
  %1431 = call i32 @llvm.smax.i32(i32 %1430, i32 0)
  %1432 = and i32 %1431, 65535
  %1433 = shl i32 %1432, 16
  %1434 = or i32 %1433, 32767
  %1435 = lshr i32 %1431, 16
  %1436 = and i32 %1435, 65535
  %1437 = or i32 %1436, 4194304
  %1438 = trunc i64 %1412 to i32
  %1439 = lshr i64 %1412, 32
  %1440 = trunc i64 %1439 to i32
  %1441 = and i32 %1440, 65535
  %1442 = insertelement <8 x i32> <i32 65536, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %1434, i64 2
  %1443 = insertelement <8 x i32> %1442, i32 %1437, i64 3
  %1444 = insertelement <8 x i32> %1443, i32 4, i64 4
  %1445 = insertelement <8 x i32> %1444, i32 %1438, i64 5
  %1446 = insertelement <8 x i32> %1445, i32 %1441, i64 6
  %1447 = insertelement <8 x i32> %1446, i32 0, i64 7
  call void @llvm.amdgcn.tensor.store.from.lds(<4 x i32> %1429, <8 x i32> %1447, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
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
