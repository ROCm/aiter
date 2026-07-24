; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

@moe_contiguous_psum_remap_smem = external addrspace(3) global [4096 x i8], align 1024

define amdgpu_kernel void @moe_contiguous_psum_remap(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, i32 %5, i32 %6, i32 %7, i32 %8, ptr addrspace(1) %9) #0 !reqd_work_group_size !1 {
  %11 = call range(i32 0, 512) i32 @llvm.amdgcn.workitem.id.x()
  %12 = sext i32 %11 to i64
  %13 = trunc i64 %12 to i32
  %14 = sub i32 %8, 1
  %15 = ptrtoint ptr addrspace(1) %0 to i64
  %16 = inttoptr i64 %15 to ptr
  %17 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %16, i16 0, i64 4294967295, i32 159744)
  %18 = ptrtoint ptr addrspace(1) %1 to i64
  %19 = inttoptr i64 %18 to ptr
  %20 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %19, i16 0, i64 4294967295, i32 159744)
  %21 = ptrtoint ptr addrspace(1) %2 to i64
  %22 = inttoptr i64 %21 to ptr
  %23 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %22, i16 0, i64 4294967295, i32 159744)
  %24 = ptrtoint ptr addrspace(1) %3 to i64
  %25 = inttoptr i64 %24 to ptr
  %26 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %25, i16 0, i64 4294967295, i32 159744)
  %27 = ptrtoint ptr addrspace(1) %4 to i64
  %28 = inttoptr i64 %27 to ptr
  %29 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %28, i16 0, i64 4294967295, i32 159744)
  %30 = icmp ult i32 %13, %6
  br i1 %30, label %31, label %39

31:                                               ; preds = %10
  %32 = mul i32 %13, 4
  %33 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %17, i32 %32, i32 0, i32 0)
  %34 = add i32 %33, %14
  %35 = udiv i32 %34, %8
  %36 = mul i32 %35, %8
  %37 = insertelement <1 x i32> poison, i32 %36, i32 0
  %38 = getelementptr i32, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i64 %12
  store <1 x i32> %37, ptr addrspace(3) %38, align 16
  br label %39

39:                                               ; preds = %31, %10
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  br i1 %30, label %40, label %58

40:                                               ; preds = %39
  %41 = getelementptr i32, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i64 %12
  %42 = load <1 x i32>, ptr addrspace(3) %41, align 4
  %43 = extractelement <1 x i32> %42, i64 0
  %44 = icmp uge i32 %13, 1
  br i1 %44, label %45, label %51

45:                                               ; preds = %40
  %46 = sub i32 %13, 1
  %47 = sext i32 %46 to i64
  %48 = getelementptr i32, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i64 %47
  %49 = load <1 x i32>, ptr addrspace(3) %48, align 4
  %50 = extractelement <1 x i32> %49, i64 0
  br label %52

51:                                               ; preds = %40
  br label %52

52:                                               ; preds = %45, %51
  %53 = phi i32 [ 0, %51 ], [ %50, %45 ]
  br label %54

54:                                               ; preds = %52
  %55 = add i32 %43, %53
  %56 = insertelement <1 x i32> poison, i32 %55, i32 0
  %57 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i32 2048), i64 %12
  store <1 x i32> %56, ptr addrspace(3) %57, align 16
  br label %58

58:                                               ; preds = %54, %39
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  br i1 %30, label %59, label %77

59:                                               ; preds = %58
  %60 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i32 2048), i64 %12
  %61 = load <1 x i32>, ptr addrspace(3) %60, align 4
  %62 = extractelement <1 x i32> %61, i64 0
  %63 = icmp uge i32 %13, 2
  br i1 %63, label %64, label %70

64:                                               ; preds = %59
  %65 = sub i32 %13, 2
  %66 = sext i32 %65 to i64
  %67 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i32 2048), i64 %66
  %68 = load <1 x i32>, ptr addrspace(3) %67, align 4
  %69 = extractelement <1 x i32> %68, i64 0
  br label %71

70:                                               ; preds = %59
  br label %71

71:                                               ; preds = %64, %70
  %72 = phi i32 [ 0, %70 ], [ %69, %64 ]
  br label %73

73:                                               ; preds = %71
  %74 = add i32 %62, %72
  %75 = insertelement <1 x i32> poison, i32 %74, i32 0
  %76 = getelementptr i32, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i64 %12
  store <1 x i32> %75, ptr addrspace(3) %76, align 16
  br label %77

77:                                               ; preds = %73, %58
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  br i1 %30, label %78, label %96

78:                                               ; preds = %77
  %79 = getelementptr i32, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i64 %12
  %80 = load <1 x i32>, ptr addrspace(3) %79, align 4
  %81 = extractelement <1 x i32> %80, i64 0
  %82 = icmp uge i32 %13, 4
  br i1 %82, label %83, label %89

83:                                               ; preds = %78
  %84 = sub i32 %13, 4
  %85 = sext i32 %84 to i64
  %86 = getelementptr i32, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i64 %85
  %87 = load <1 x i32>, ptr addrspace(3) %86, align 4
  %88 = extractelement <1 x i32> %87, i64 0
  br label %90

89:                                               ; preds = %78
  br label %90

90:                                               ; preds = %83, %89
  %91 = phi i32 [ 0, %89 ], [ %88, %83 ]
  br label %92

92:                                               ; preds = %90
  %93 = add i32 %81, %91
  %94 = insertelement <1 x i32> poison, i32 %93, i32 0
  %95 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i32 2048), i64 %12
  store <1 x i32> %94, ptr addrspace(3) %95, align 16
  br label %96

96:                                               ; preds = %92, %77
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  br i1 %30, label %97, label %115

97:                                               ; preds = %96
  %98 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i32 2048), i64 %12
  %99 = load <1 x i32>, ptr addrspace(3) %98, align 4
  %100 = extractelement <1 x i32> %99, i64 0
  %101 = icmp uge i32 %13, 8
  br i1 %101, label %102, label %108

102:                                              ; preds = %97
  %103 = sub i32 %13, 8
  %104 = sext i32 %103 to i64
  %105 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i32 2048), i64 %104
  %106 = load <1 x i32>, ptr addrspace(3) %105, align 4
  %107 = extractelement <1 x i32> %106, i64 0
  br label %109

108:                                              ; preds = %97
  br label %109

109:                                              ; preds = %102, %108
  %110 = phi i32 [ 0, %108 ], [ %107, %102 ]
  br label %111

111:                                              ; preds = %109
  %112 = add i32 %100, %110
  %113 = insertelement <1 x i32> poison, i32 %112, i32 0
  %114 = getelementptr i32, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i64 %12
  store <1 x i32> %113, ptr addrspace(3) %114, align 16
  br label %115

115:                                              ; preds = %111, %96
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  br i1 %30, label %116, label %134

116:                                              ; preds = %115
  %117 = getelementptr i32, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i64 %12
  %118 = load <1 x i32>, ptr addrspace(3) %117, align 4
  %119 = extractelement <1 x i32> %118, i64 0
  %120 = icmp uge i32 %13, 16
  br i1 %120, label %121, label %127

121:                                              ; preds = %116
  %122 = sub i32 %13, 16
  %123 = sext i32 %122 to i64
  %124 = getelementptr i32, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i64 %123
  %125 = load <1 x i32>, ptr addrspace(3) %124, align 4
  %126 = extractelement <1 x i32> %125, i64 0
  br label %128

127:                                              ; preds = %116
  br label %128

128:                                              ; preds = %121, %127
  %129 = phi i32 [ 0, %127 ], [ %126, %121 ]
  br label %130

130:                                              ; preds = %128
  %131 = add i32 %119, %129
  %132 = insertelement <1 x i32> poison, i32 %131, i32 0
  %133 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i32 2048), i64 %12
  store <1 x i32> %132, ptr addrspace(3) %133, align 16
  br label %134

134:                                              ; preds = %130, %115
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  br i1 %30, label %135, label %153

135:                                              ; preds = %134
  %136 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i32 2048), i64 %12
  %137 = load <1 x i32>, ptr addrspace(3) %136, align 4
  %138 = extractelement <1 x i32> %137, i64 0
  %139 = icmp uge i32 %13, 32
  br i1 %139, label %140, label %146

140:                                              ; preds = %135
  %141 = sub i32 %13, 32
  %142 = sext i32 %141 to i64
  %143 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i32 2048), i64 %142
  %144 = load <1 x i32>, ptr addrspace(3) %143, align 4
  %145 = extractelement <1 x i32> %144, i64 0
  br label %147

146:                                              ; preds = %135
  br label %147

147:                                              ; preds = %140, %146
  %148 = phi i32 [ 0, %146 ], [ %145, %140 ]
  br label %149

149:                                              ; preds = %147
  %150 = add i32 %138, %148
  %151 = insertelement <1 x i32> poison, i32 %150, i32 0
  %152 = getelementptr i32, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i64 %12
  store <1 x i32> %151, ptr addrspace(3) %152, align 16
  br label %153

153:                                              ; preds = %149, %134
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  br i1 %30, label %154, label %172

154:                                              ; preds = %153
  %155 = getelementptr i32, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i64 %12
  %156 = load <1 x i32>, ptr addrspace(3) %155, align 4
  %157 = extractelement <1 x i32> %156, i64 0
  %158 = icmp uge i32 %13, 64
  br i1 %158, label %159, label %165

159:                                              ; preds = %154
  %160 = sub i32 %13, 64
  %161 = sext i32 %160 to i64
  %162 = getelementptr i32, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i64 %161
  %163 = load <1 x i32>, ptr addrspace(3) %162, align 4
  %164 = extractelement <1 x i32> %163, i64 0
  br label %166

165:                                              ; preds = %154
  br label %166

166:                                              ; preds = %159, %165
  %167 = phi i32 [ 0, %165 ], [ %164, %159 ]
  br label %168

168:                                              ; preds = %166
  %169 = add i32 %157, %167
  %170 = insertelement <1 x i32> poison, i32 %169, i32 0
  %171 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i32 2048), i64 %12
  store <1 x i32> %170, ptr addrspace(3) %171, align 16
  br label %172

172:                                              ; preds = %168, %153
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  br i1 %30, label %173, label %191

173:                                              ; preds = %172
  %174 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i32 2048), i64 %12
  %175 = load <1 x i32>, ptr addrspace(3) %174, align 4
  %176 = extractelement <1 x i32> %175, i64 0
  %177 = icmp uge i32 %13, 128
  br i1 %177, label %178, label %184

178:                                              ; preds = %173
  %179 = sub i32 %13, 128
  %180 = sext i32 %179 to i64
  %181 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i32 2048), i64 %180
  %182 = load <1 x i32>, ptr addrspace(3) %181, align 4
  %183 = extractelement <1 x i32> %182, i64 0
  br label %185

184:                                              ; preds = %173
  br label %185

185:                                              ; preds = %178, %184
  %186 = phi i32 [ 0, %184 ], [ %183, %178 ]
  br label %187

187:                                              ; preds = %185
  %188 = add i32 %176, %186
  %189 = insertelement <1 x i32> poison, i32 %188, i32 0
  %190 = getelementptr i32, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i64 %12
  store <1 x i32> %189, ptr addrspace(3) %190, align 16
  br label %191

191:                                              ; preds = %187, %172
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  br i1 %30, label %192, label %210

192:                                              ; preds = %191
  %193 = getelementptr i32, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i64 %12
  %194 = load <1 x i32>, ptr addrspace(3) %193, align 4
  %195 = extractelement <1 x i32> %194, i64 0
  %196 = icmp uge i32 %13, 256
  br i1 %196, label %197, label %203

197:                                              ; preds = %192
  %198 = sub i32 %13, 256
  %199 = sext i32 %198 to i64
  %200 = getelementptr i32, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i64 %199
  %201 = load <1 x i32>, ptr addrspace(3) %200, align 4
  %202 = extractelement <1 x i32> %201, i64 0
  br label %204

203:                                              ; preds = %192
  br label %204

204:                                              ; preds = %197, %203
  %205 = phi i32 [ 0, %203 ], [ %202, %197 ]
  br label %206

206:                                              ; preds = %204
  %207 = add i32 %195, %205
  %208 = insertelement <1 x i32> poison, i32 %207, i32 0
  %209 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i32 2048), i64 %12
  store <1 x i32> %208, ptr addrspace(3) %209, align 16
  br label %210

210:                                              ; preds = %206, %191
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  br i1 %30, label %211, label %235

211:                                              ; preds = %210
  %212 = icmp eq i32 %13, 0
  br i1 %212, label %213, label %214

213:                                              ; preds = %211
  br label %220

214:                                              ; preds = %211
  %215 = sub i32 %13, 1
  %216 = sext i32 %215 to i64
  %217 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i32 2048), i64 %216
  %218 = load <1 x i32>, ptr addrspace(3) %217, align 4
  %219 = extractelement <1 x i32> %218, i64 0
  br label %220

220:                                              ; preds = %213, %214
  %221 = phi i32 [ %219, %214 ], [ 0, %213 ]
  br label %222

222:                                              ; preds = %220
  %223 = mul i32 %13, 4
  %224 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %17, i32 %223, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %221, ptr addrspace(8) %23, i32 %223, i32 0, i32 0)
  %225 = add i32 %221, %224
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %225, ptr addrspace(8) %26, i32 %223, i32 0, i32 0)
  %226 = sub i32 %6, 1
  %227 = icmp eq i32 %13, %226
  br i1 %227, label %228, label %234

228:                                              ; preds = %222
  %229 = getelementptr i32, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @moe_contiguous_psum_remap_smem, i32 2048), i64 %12
  %230 = load <1 x i32>, ptr addrspace(3) %229, align 4
  %231 = extractelement <1 x i32> %230, i64 0
  %232 = icmp sgt i32 %231, %8
  %233 = select i1 %232, i32 %231, i32 %8
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %233, ptr addrspace(8) %29, i32 0, i32 0, i32 0)
  br label %234

234:                                              ; preds = %228, %222
  br label %235

235:                                              ; preds = %234, %210
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %236 = ptrtoint ptr addrspace(1) %9 to i64
  %237 = inttoptr i64 %236 to ptr
  %238 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %237, i16 0, i64 4294967295, i32 159744)
  %239 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %238, i32 0, i32 0, i32 0)
  %240 = sext i32 %239 to i64
  br label %241

241:                                              ; preds = %244, %235
  %242 = phi i64 [ %254, %244 ], [ %12, %235 ]
  %243 = icmp slt i64 %242, %240
  br i1 %243, label %244, label %255

244:                                              ; preds = %241
  %245 = trunc i64 %242 to i32
  %246 = mul i32 %245, 4
  %247 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %20, i32 %246, i32 0, i32 0)
  %248 = udiv i32 %247, %7
  %249 = mul i32 %248, %7
  %250 = sub i32 %247, %249
  %251 = mul i32 %248, 4
  %252 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %23, i32 %251, i32 0, i32 0)
  %253 = add i32 %252, %250
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %253, ptr addrspace(8) %20, i32 %246, i32 0, i32 0)
  %254 = add i64 %242, 512
  br label %241

255:                                              ; preds = %241
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr readnone, i16, i64, i32) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #3

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier.signal(i32 immarg) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier.wait(i16 immarg) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #5

attributes #0 = { "amdgpu-flat-work-group-size"="512,512" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #4 = { convergent nocallback nofree nounwind willreturn }
attributes #5 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 512, i32 1, i32 1}
