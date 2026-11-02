; ModuleID = '/work/swa-fourwave/native_pre.bc'
source_filename = "/work/swa-fourwave/shuffled_pre.cpp"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

$_ZN7ck_tileL11flush_cacheEv.intern.2b4ba03da513173f = comdat any

$_ZZN5aiter26mhc_pre_gemm_sqrsum_kernelIDF16bLi4ELi64ELi16ELi64ELb1ELb1EEEvPfS1_PT_S1_iiiiiijjjiE4s_fn = comdat any

$_ZZN5aiter26mhc_pre_gemm_sqrsum_kernelIDF16bLi4ELi64ELi16ELi64ELb1ELb1EEEvPfS1_PT_S1_iiiiiijjjiE3s_x = comdat any

$_ZZN5aiter31mhc_pre_big_fuse_rmsnorm_kernelIDF16bLi9ELi4ELi1ELi4096ELi1024ELi1024ELb0EEEvPfS1_PT_S1_S1_S1_S1_S3_S3_iiiifffffiijPcE10s_hc_mult3 = comdat any

$_ZZ12block_reduceIfZN5aiter31mhc_pre_big_fuse_rmsnorm_kernelIDF16bLi9ELi4ELi1ELi4096ELi1024ELi1024ELb0EEEvPfS2_PT_S2_S2_S2_S2_S4_S4_iiiifffffiijPcEUlffE_Li256ELb1EES3_S3_T0_E4smem = comdat any

@__const.__assert_fail.fmt = private unnamed_addr addrspace(4) constant [47 x i8] c"%s:%u: %s: Device-side assertion `%s' failed.\0A\00", align 16
@__hip_cuid_2b4ba03da513173f = addrspace(1) global i8 0

; Function Attrs: mustprogress noreturn nounwind uwtable
define weak void @__cxa_pure_virtual() local_unnamed_addr #0 {
  tail call void @llvm.trap()
  unreachable
}

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #1

; Function Attrs: mustprogress noreturn nounwind uwtable
define weak void @__cxa_deleted_virtual() local_unnamed_addr #0 {
  tail call void @llvm.trap()
  unreachable
}

; Function Attrs: convergent noinline nounwind uwtable
define weak hidden void @__assert_fail(ptr noundef %0, ptr noundef %1, i32 noundef %2, ptr noundef %3) local_unnamed_addr #2 {
  %5 = alloca [47 x i8], align 16, addrspace(5)
  %6 = addrspacecast ptr addrspace(5) %5 to ptr
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %5) #23
  call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) noundef align 16 dereferenceable(47) %5, ptr addrspace(4) noundef align 16 dereferenceable(47) @__const.__assert_fail.fmt, i64 47, i1 false)
  %7 = tail call dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %8 = getelementptr inbounds nuw i8, ptr addrspace(4) %7, i64 80
  %9 = load i64, ptr addrspace(4) %8, align 8, !tbaa !14
  %10 = inttoptr i64 %9 to ptr addrspace(1)
  %11 = tail call range(i32 0, 33) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %12 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %11)
  %13 = icmp eq i32 %11, %12
  br i1 %13, label %14, label %36

14:                                               ; preds = %4
  %15 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 24
  %16 = load atomic i64, ptr addrspace(1) %15 syncscope("one-as") acquire, align 8
  %17 = getelementptr i8, ptr addrspace(1) %10, i64 40
  %18 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %19 = load i64, ptr addrspace(1) %17, align 8, !tbaa !20
  %20 = and i64 %19, %16
  %21 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %18, i64 %20
  %22 = load atomic i64, ptr addrspace(1) %21 syncscope("one-as") monotonic, align 8
  %23 = cmpxchg ptr addrspace(1) %15, i64 %16, i64 %22 syncscope("one-as") acquire monotonic, align 8
  %24 = extractvalue { i64, i1 } %23, 1
  %25 = extractvalue { i64, i1 } %23, 0
  br i1 %24, label %36, label %26

26:                                               ; preds = %14, %26
  %27 = phi i64 [ %35, %26 ], [ %25, %14 ]
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  %28 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %29 = load i64, ptr addrspace(1) %17, align 8, !tbaa !20
  %30 = and i64 %29, %27
  %31 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %28, i64 %30
  %32 = load atomic i64, ptr addrspace(1) %31 syncscope("one-as") monotonic, align 8
  %33 = cmpxchg ptr addrspace(1) %15, i64 %27, i64 %32 syncscope("one-as") acquire monotonic, align 8
  %34 = extractvalue { i64, i1 } %33, 1
  %35 = extractvalue { i64, i1 } %33, 0
  br i1 %34, label %36, label %26

36:                                               ; preds = %26, %14, %4
  %37 = phi i64 [ 0, %4 ], [ %25, %14 ], [ %35, %26 ]
  %38 = tail call i64 @llvm.amdgcn.readfirstlane.i64(i64 %37)
  %39 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %40 = getelementptr i8, ptr addrspace(1) %10, i64 40
  %41 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %42 = and i64 %41, %38
  %43 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %39, i64 %42
  %44 = getelementptr i8, ptr addrspace(1) %10, i64 8
  %45 = load ptr addrspace(1), ptr addrspace(1) %44, align 8, !tbaa !21
  %46 = getelementptr inbounds nuw [4096 x i8], ptr addrspace(1) %45, i64 %42
  %47 = tail call i32 @llvm.amdgcn.ballot.i32(i1 true)
  br i1 %13, label %48, label %52

48:                                               ; preds = %36
  %49 = zext i32 %47 to i64
  %50 = getelementptr inbounds nuw i8, ptr addrspace(1) %43, i64 16
  %51 = getelementptr inbounds nuw i8, ptr addrspace(1) %43, i64 8
  store i64 %49, ptr addrspace(1) %51, align 8, !tbaa !22
  store <2 x i32> <i32 2, i32 1>, ptr addrspace(1) %50, align 8, !tbaa !9
  br label %52

52:                                               ; preds = %48, %36
  %53 = zext nneg i32 %11 to i64
  %54 = getelementptr inbounds nuw [64 x i8], ptr addrspace(1) %46, i64 %53
  store i64 33, ptr addrspace(1) %54, align 8, !tbaa !14
  %55 = getelementptr inbounds nuw i8, ptr addrspace(1) %54, i64 8
  store i64 1, ptr addrspace(1) %55, align 8, !tbaa !14
  %56 = getelementptr inbounds nuw i8, ptr addrspace(1) %54, i64 16
  store i64 0, ptr addrspace(1) %56, align 8, !tbaa !14
  %57 = getelementptr inbounds nuw i8, ptr addrspace(1) %54, i64 24
  store i64 0, ptr addrspace(1) %57, align 8, !tbaa !14
  %58 = getelementptr inbounds nuw i8, ptr addrspace(1) %54, i64 32
  store i64 0, ptr addrspace(1) %58, align 8, !tbaa !14
  %59 = getelementptr inbounds nuw i8, ptr addrspace(1) %54, i64 40
  store i64 0, ptr addrspace(1) %59, align 8, !tbaa !14
  %60 = getelementptr inbounds nuw i8, ptr addrspace(1) %54, i64 48
  store i64 0, ptr addrspace(1) %60, align 8, !tbaa !14
  %61 = getelementptr inbounds nuw i8, ptr addrspace(1) %54, i64 56
  store i64 0, ptr addrspace(1) %61, align 8, !tbaa !14
  br i1 %13, label %62, label %91

62:                                               ; preds = %52
  %63 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 32
  %64 = load atomic i64, ptr addrspace(1) %63 syncscope("one-as") monotonic, align 8
  %65 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %66 = and i64 %65, %38
  %67 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %39, i64 %66
  store i64 %64, ptr addrspace(1) %67, align 8, !tbaa !24
  %68 = cmpxchg ptr addrspace(1) %63, i64 %64, i64 %38 syncscope("one-as") release monotonic, align 8
  %69 = extractvalue { i64, i1 } %68, 1
  br i1 %69, label %75, label %70

70:                                               ; preds = %62, %70
  %71 = phi { i64, i1 } [ %73, %70 ], [ %68, %62 ]
  %72 = extractvalue { i64, i1 } %71, 0
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %72, ptr addrspace(1) %67, align 8, !tbaa !24
  %73 = cmpxchg ptr addrspace(1) %63, i64 %72, i64 %38 syncscope("one-as") release monotonic, align 8
  %74 = extractvalue { i64, i1 } %73, 1
  br i1 %74, label %75, label %70

75:                                               ; preds = %70, %62
  %76 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 16
  %77 = load i64, ptr addrspace(1) %76, align 8
  %78 = inttoptr i64 %77 to ptr addrspace(1)
  %79 = getelementptr inbounds nuw i8, ptr addrspace(1) %78, i64 8
  %80 = atomicrmw add ptr addrspace(1) %79, i64 1 syncscope("one-as") release, align 8, !amdgpu.no.fine.grained.memory !25, !amdgpu.no.remote.memory !25
  %81 = getelementptr inbounds nuw i8, ptr addrspace(1) %78, i64 16
  %82 = load i64, ptr addrspace(1) %81, align 16, !tbaa !26
  %83 = icmp eq i64 %82, 0
  br i1 %83, label %91, label %84

84:                                               ; preds = %75
  %85 = inttoptr i64 %82 to ptr addrspace(1)
  %86 = getelementptr inbounds nuw i8, ptr addrspace(1) %78, i64 24
  %87 = load i32, ptr addrspace(1) %86, align 8, !tbaa !28
  %88 = zext i32 %87 to i64
  store atomic i64 %88, ptr addrspace(1) %85 syncscope("one-as") release, align 8
  %89 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %87)
  %90 = and i32 %89, 16777215
  tail call void @llvm.amdgcn.s.sendmsg(i32 1, i32 %90)
  br label %91

91:                                               ; preds = %84, %75, %52
  %92 = getelementptr inbounds nuw i8, ptr addrspace(1) %43, i64 20
  br label %93

93:                                               ; preds = %101, %91
  br i1 %13, label %94, label %97

94:                                               ; preds = %93
  %95 = load atomic i32, ptr addrspace(1) %92 syncscope("one-as") acquire, align 4
  %96 = and i32 %95, 1
  br label %97

97:                                               ; preds = %94, %93
  %98 = phi i32 [ %96, %94 ], [ 1, %93 ]
  %99 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %98)
  %100 = icmp eq i32 %99, 0
  br i1 %100, label %102, label %101

101:                                              ; preds = %97
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  br label %93

102:                                              ; preds = %97
  %103 = load i64, ptr addrspace(1) %54, align 8, !tbaa !14
  br i1 %13, label %104, label %122

104:                                              ; preds = %102
  %105 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %106 = add i64 %105, 1
  %107 = add i64 %106, %38
  %108 = icmp eq i64 %107, 0
  %109 = select i1 %108, i64 %106, i64 %107
  %110 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 24
  %111 = load atomic i64, ptr addrspace(1) %110 syncscope("one-as") monotonic, align 8
  %112 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %113 = and i64 %109, %105
  %114 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %112, i64 %113
  store i64 %111, ptr addrspace(1) %114, align 8, !tbaa !24
  %115 = cmpxchg ptr addrspace(1) %110, i64 %111, i64 %109 syncscope("one-as") release monotonic, align 8
  %116 = extractvalue { i64, i1 } %115, 1
  br i1 %116, label %122, label %117

117:                                              ; preds = %104, %117
  %118 = phi { i64, i1 } [ %120, %117 ], [ %115, %104 ]
  %119 = extractvalue { i64, i1 } %118, 0
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %119, ptr addrspace(1) %114, align 8, !tbaa !24
  %120 = cmpxchg ptr addrspace(1) %110, i64 %119, i64 %109 syncscope("one-as") release monotonic, align 8
  %121 = extractvalue { i64, i1 } %120, 1
  br i1 %121, label %122, label %117

122:                                              ; preds = %117, %102, %104
  br label %123

123:                                              ; preds = %122, %123
  %124 = phi ptr addrspace(5) [ %125, %123 ], [ %5, %122 ]
  %125 = getelementptr inbounds nuw i8, ptr addrspace(5) %124, i32 1
  %126 = load i8, ptr addrspace(5) %124, align 1, !tbaa !29
  %127 = icmp eq i8 %126, 0
  br i1 %127, label %128, label %123, !llvm.loop !30

128:                                              ; preds = %123
  %129 = icmp eq ptr addrspace(5) %5, addrspacecast (ptr null to ptr addrspace(5))
  br i1 %129, label %130, label %243

130:                                              ; preds = %128
  %131 = and i64 %103, -225
  %132 = or disjoint i64 %131, 32
  %133 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %11)
  %134 = icmp eq i32 %11, %133
  br i1 %134, label %135, label %156

135:                                              ; preds = %130
  %136 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 24
  %137 = load atomic i64, ptr addrspace(1) %136 syncscope("one-as") acquire, align 8
  %138 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %139 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %140 = and i64 %139, %137
  %141 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %138, i64 %140
  %142 = load atomic i64, ptr addrspace(1) %141 syncscope("one-as") monotonic, align 8
  %143 = cmpxchg ptr addrspace(1) %136, i64 %137, i64 %142 syncscope("one-as") acquire monotonic, align 8
  %144 = extractvalue { i64, i1 } %143, 1
  %145 = extractvalue { i64, i1 } %143, 0
  br i1 %144, label %156, label %146

146:                                              ; preds = %135, %146
  %147 = phi i64 [ %155, %146 ], [ %145, %135 ]
  call void @llvm.amdgcn.s.sleep(i32 1)
  %148 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %149 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %150 = and i64 %149, %147
  %151 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %148, i64 %150
  %152 = load atomic i64, ptr addrspace(1) %151 syncscope("one-as") monotonic, align 8
  %153 = cmpxchg ptr addrspace(1) %136, i64 %147, i64 %152 syncscope("one-as") acquire monotonic, align 8
  %154 = extractvalue { i64, i1 } %153, 1
  %155 = extractvalue { i64, i1 } %153, 0
  br i1 %154, label %156, label %146

156:                                              ; preds = %146, %135, %130
  %157 = phi i64 [ 0, %130 ], [ %145, %135 ], [ %155, %146 ]
  %158 = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %157)
  %159 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %160 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %161 = and i64 %160, %158
  %162 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %159, i64 %161
  %163 = load ptr addrspace(1), ptr addrspace(1) %44, align 8, !tbaa !21
  %164 = getelementptr inbounds nuw [4096 x i8], ptr addrspace(1) %163, i64 %161
  %165 = call i32 @llvm.amdgcn.ballot.i32(i1 true)
  br i1 %134, label %166, label %170

166:                                              ; preds = %156
  %167 = zext i32 %165 to i64
  %168 = getelementptr inbounds nuw i8, ptr addrspace(1) %162, i64 16
  %169 = getelementptr inbounds nuw i8, ptr addrspace(1) %162, i64 8
  store i64 %167, ptr addrspace(1) %169, align 8, !tbaa !22
  store <2 x i32> <i32 2, i32 1>, ptr addrspace(1) %168, align 8, !tbaa !9
  br label %170

170:                                              ; preds = %166, %156
  %171 = getelementptr inbounds nuw [64 x i8], ptr addrspace(1) %164, i64 %53
  store i64 %132, ptr addrspace(1) %171, align 8, !tbaa !14
  %172 = getelementptr inbounds nuw i8, ptr addrspace(1) %171, i64 8
  store i64 0, ptr addrspace(1) %172, align 8, !tbaa !14
  %173 = getelementptr inbounds nuw i8, ptr addrspace(1) %171, i64 16
  store i64 0, ptr addrspace(1) %173, align 8, !tbaa !14
  %174 = getelementptr inbounds nuw i8, ptr addrspace(1) %171, i64 24
  store i64 0, ptr addrspace(1) %174, align 8, !tbaa !14
  %175 = getelementptr inbounds nuw i8, ptr addrspace(1) %171, i64 32
  store i64 0, ptr addrspace(1) %175, align 8, !tbaa !14
  %176 = getelementptr inbounds nuw i8, ptr addrspace(1) %171, i64 40
  store i64 0, ptr addrspace(1) %176, align 8, !tbaa !14
  %177 = getelementptr inbounds nuw i8, ptr addrspace(1) %171, i64 48
  store i64 0, ptr addrspace(1) %177, align 8, !tbaa !14
  %178 = getelementptr inbounds nuw i8, ptr addrspace(1) %171, i64 56
  store i64 0, ptr addrspace(1) %178, align 8, !tbaa !14
  br i1 %134, label %179, label %208

179:                                              ; preds = %170
  %180 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 32
  %181 = load atomic i64, ptr addrspace(1) %180 syncscope("one-as") monotonic, align 8
  %182 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %183 = and i64 %182, %158
  %184 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %159, i64 %183
  store i64 %181, ptr addrspace(1) %184, align 8, !tbaa !24
  %185 = cmpxchg ptr addrspace(1) %180, i64 %181, i64 %158 syncscope("one-as") release monotonic, align 8
  %186 = extractvalue { i64, i1 } %185, 1
  br i1 %186, label %192, label %187

187:                                              ; preds = %179, %187
  %188 = phi { i64, i1 } [ %190, %187 ], [ %185, %179 ]
  %189 = extractvalue { i64, i1 } %188, 0
  call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %189, ptr addrspace(1) %184, align 8, !tbaa !24
  %190 = cmpxchg ptr addrspace(1) %180, i64 %189, i64 %158 syncscope("one-as") release monotonic, align 8
  %191 = extractvalue { i64, i1 } %190, 1
  br i1 %191, label %192, label %187

192:                                              ; preds = %187, %179
  %193 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 16
  %194 = load i64, ptr addrspace(1) %193, align 8
  %195 = inttoptr i64 %194 to ptr addrspace(1)
  %196 = getelementptr inbounds nuw i8, ptr addrspace(1) %195, i64 8
  %197 = atomicrmw add ptr addrspace(1) %196, i64 1 syncscope("one-as") release, align 8, !amdgpu.no.fine.grained.memory !25, !amdgpu.no.remote.memory !25
  %198 = getelementptr inbounds nuw i8, ptr addrspace(1) %195, i64 16
  %199 = load i64, ptr addrspace(1) %198, align 16, !tbaa !26
  %200 = icmp eq i64 %199, 0
  br i1 %200, label %208, label %201

201:                                              ; preds = %192
  %202 = inttoptr i64 %199 to ptr addrspace(1)
  %203 = getelementptr inbounds nuw i8, ptr addrspace(1) %195, i64 24
  %204 = load i32, ptr addrspace(1) %203, align 8, !tbaa !28
  %205 = zext i32 %204 to i64
  store atomic i64 %205, ptr addrspace(1) %202 syncscope("one-as") release, align 8
  %206 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %204)
  %207 = and i32 %206, 16777215
  call void @llvm.amdgcn.s.sendmsg(i32 1, i32 %207)
  br label %208

208:                                              ; preds = %201, %192, %170
  %209 = getelementptr inbounds nuw i8, ptr addrspace(1) %162, i64 20
  br label %210

210:                                              ; preds = %218, %208
  br i1 %134, label %211, label %214

211:                                              ; preds = %210
  %212 = load atomic i32, ptr addrspace(1) %209 syncscope("one-as") acquire, align 4
  %213 = and i32 %212, 1
  br label %214

214:                                              ; preds = %211, %210
  %215 = phi i32 [ %213, %211 ], [ 1, %210 ]
  %216 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %215)
  %217 = icmp eq i32 %216, 0
  br i1 %217, label %219, label %218

218:                                              ; preds = %214
  call void @llvm.amdgcn.s.sleep(i32 1)
  br label %210

219:                                              ; preds = %214
  %220 = load i64, ptr addrspace(1) %171, align 8, !tbaa !14
  %221 = load i64, ptr addrspace(1) %172, align 8, !tbaa !14
  br i1 %134, label %222, label %240

222:                                              ; preds = %219
  %223 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %224 = add i64 %223, 1
  %225 = add i64 %224, %158
  %226 = icmp eq i64 %225, 0
  %227 = select i1 %226, i64 %224, i64 %225
  %228 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 24
  %229 = load atomic i64, ptr addrspace(1) %228 syncscope("one-as") monotonic, align 8
  %230 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %231 = and i64 %227, %223
  %232 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %230, i64 %231
  store i64 %229, ptr addrspace(1) %232, align 8, !tbaa !24
  %233 = cmpxchg ptr addrspace(1) %228, i64 %229, i64 %227 syncscope("one-as") release monotonic, align 8
  %234 = extractvalue { i64, i1 } %233, 1
  br i1 %234, label %240, label %235

235:                                              ; preds = %222, %235
  %236 = phi { i64, i1 } [ %238, %235 ], [ %233, %222 ]
  %237 = extractvalue { i64, i1 } %236, 0
  call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %237, ptr addrspace(1) %232, align 8, !tbaa !24
  %238 = cmpxchg ptr addrspace(1) %228, i64 %237, i64 %227 syncscope("one-as") release monotonic, align 8
  %239 = extractvalue { i64, i1 } %238, 1
  br i1 %239, label %240, label %235

240:                                              ; preds = %235, %222, %219
  %241 = insertelement <2 x i64> poison, i64 %220, i64 0
  %242 = insertelement <2 x i64> %241, i64 %221, i64 1
  br label %537

243:                                              ; preds = %128
  %244 = addrspacecast ptr addrspace(5) %125 to ptr
  %245 = ptrtoint ptr %244 to i64
  %246 = ptrtoint ptr %6 to i64
  %247 = sub i64 %245, %246
  %248 = shl i64 %247, 32
  %249 = ashr exact i64 %248, 32
  %250 = and i64 %103, 2
  %251 = and i64 %103, -3
  %252 = insertelement <2 x i64> <i64 poison, i64 0>, i64 %251, i64 0
  %253 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 24
  %254 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 32
  %255 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 16
  br label %256

256:                                              ; preds = %531, %243
  %257 = phi i64 [ %249, %243 ], [ %534, %531 ]
  %258 = phi ptr addrspace(5) [ %5, %243 ], [ %535, %531 ]
  %259 = phi <2 x i64> [ %252, %243 ], [ %533, %531 ]
  %260 = icmp ugt i64 %257, 56
  %261 = extractelement <2 x i64> %259, i64 0
  %262 = call i64 @llvm.umin.i64(i64 %257, i64 56)
  %263 = trunc nuw nsw i64 %262 to i32
  %264 = select i1 %260, i64 0, i64 %250
  %265 = icmp ugt i64 %257, 7
  br i1 %265, label %268, label %266

266:                                              ; preds = %256
  %267 = icmp eq i64 %257, 0
  br i1 %267, label %284, label %272

268:                                              ; preds = %256
  %269 = load i64, ptr addrspace(5) %258, align 1
  %270 = add nsw i32 %263, -8
  %271 = getelementptr inbounds nuw i8, ptr addrspace(5) %258, i32 8
  br label %284

272:                                              ; preds = %266, %272
  %273 = phi i32 [ %282, %272 ], [ 0, %266 ]
  %274 = phi i64 [ %281, %272 ], [ 0, %266 ]
  %275 = getelementptr inbounds nuw i8, ptr addrspace(5) %258, i32 %273
  %276 = load i8, ptr addrspace(5) %275, align 1, !tbaa !32
  %277 = zext i8 %276 to i64
  %278 = shl i32 %273, 3
  %279 = zext nneg i32 %278 to i64
  %280 = shl nuw i64 %277, %279
  %281 = or i64 %280, %274
  %282 = add nuw nsw i32 %273, 1
  %283 = icmp eq i32 %282, %263
  br i1 %283, label %284, label %272

284:                                              ; preds = %272, %268, %266
  %285 = phi ptr addrspace(5) [ %271, %268 ], [ %258, %266 ], [ %258, %272 ]
  %286 = phi i32 [ %270, %268 ], [ 0, %266 ], [ 0, %272 ]
  %287 = phi i64 [ %269, %268 ], [ 0, %266 ], [ %281, %272 ]
  %288 = icmp ugt i32 %286, 7
  br i1 %288, label %291, label %289

289:                                              ; preds = %284
  %290 = icmp eq i32 %286, 0
  br i1 %290, label %307, label %295

291:                                              ; preds = %284
  %292 = load i64, ptr addrspace(5) %285, align 1
  %293 = add nsw i32 %286, -8
  %294 = getelementptr inbounds nuw i8, ptr addrspace(5) %285, i32 8
  br label %307

295:                                              ; preds = %289, %295
  %296 = phi i32 [ %305, %295 ], [ 0, %289 ]
  %297 = phi i64 [ %304, %295 ], [ 0, %289 ]
  %298 = getelementptr inbounds nuw i8, ptr addrspace(5) %285, i32 %296
  %299 = load i8, ptr addrspace(5) %298, align 1, !tbaa !32
  %300 = zext i8 %299 to i64
  %301 = shl i32 %296, 3
  %302 = zext nneg i32 %301 to i64
  %303 = shl nuw i64 %300, %302
  %304 = or i64 %303, %297
  %305 = add nuw nsw i32 %296, 1
  %306 = icmp eq i32 %305, %286
  br i1 %306, label %307, label %295

307:                                              ; preds = %295, %291, %289
  %308 = phi ptr addrspace(5) [ %294, %291 ], [ %285, %289 ], [ %285, %295 ]
  %309 = phi i32 [ %293, %291 ], [ 0, %289 ], [ 0, %295 ]
  %310 = phi i64 [ %292, %291 ], [ 0, %289 ], [ %304, %295 ]
  %311 = icmp ugt i32 %309, 7
  br i1 %311, label %314, label %312

312:                                              ; preds = %307
  %313 = icmp eq i32 %309, 0
  br i1 %313, label %330, label %318

314:                                              ; preds = %307
  %315 = load i64, ptr addrspace(5) %308, align 1
  %316 = add nsw i32 %309, -8
  %317 = getelementptr inbounds nuw i8, ptr addrspace(5) %308, i32 8
  br label %330

318:                                              ; preds = %312, %318
  %319 = phi i32 [ %328, %318 ], [ 0, %312 ]
  %320 = phi i64 [ %327, %318 ], [ 0, %312 ]
  %321 = getelementptr inbounds nuw i8, ptr addrspace(5) %308, i32 %319
  %322 = load i8, ptr addrspace(5) %321, align 1, !tbaa !32
  %323 = zext i8 %322 to i64
  %324 = shl i32 %319, 3
  %325 = zext nneg i32 %324 to i64
  %326 = shl nuw i64 %323, %325
  %327 = or i64 %326, %320
  %328 = add nuw nsw i32 %319, 1
  %329 = icmp eq i32 %328, %309
  br i1 %329, label %330, label %318

330:                                              ; preds = %318, %314, %312
  %331 = phi ptr addrspace(5) [ %317, %314 ], [ %308, %312 ], [ %308, %318 ]
  %332 = phi i32 [ %316, %314 ], [ 0, %312 ], [ 0, %318 ]
  %333 = phi i64 [ %315, %314 ], [ 0, %312 ], [ %327, %318 ]
  %334 = icmp ugt i32 %332, 7
  br i1 %334, label %337, label %335

335:                                              ; preds = %330
  %336 = icmp eq i32 %332, 0
  br i1 %336, label %353, label %341

337:                                              ; preds = %330
  %338 = load i64, ptr addrspace(5) %331, align 1
  %339 = add nsw i32 %332, -8
  %340 = getelementptr inbounds nuw i8, ptr addrspace(5) %331, i32 8
  br label %353

341:                                              ; preds = %335, %341
  %342 = phi i32 [ %351, %341 ], [ 0, %335 ]
  %343 = phi i64 [ %350, %341 ], [ 0, %335 ]
  %344 = getelementptr inbounds nuw i8, ptr addrspace(5) %331, i32 %342
  %345 = load i8, ptr addrspace(5) %344, align 1, !tbaa !32
  %346 = zext i8 %345 to i64
  %347 = shl i32 %342, 3
  %348 = zext nneg i32 %347 to i64
  %349 = shl nuw i64 %346, %348
  %350 = or i64 %349, %343
  %351 = add nuw nsw i32 %342, 1
  %352 = icmp eq i32 %351, %332
  br i1 %352, label %353, label %341

353:                                              ; preds = %341, %337, %335
  %354 = phi ptr addrspace(5) [ %340, %337 ], [ %331, %335 ], [ %331, %341 ]
  %355 = phi i32 [ %339, %337 ], [ 0, %335 ], [ 0, %341 ]
  %356 = phi i64 [ %338, %337 ], [ 0, %335 ], [ %350, %341 ]
  %357 = icmp ugt i32 %355, 7
  br i1 %357, label %360, label %358

358:                                              ; preds = %353
  %359 = icmp eq i32 %355, 0
  br i1 %359, label %376, label %364

360:                                              ; preds = %353
  %361 = load i64, ptr addrspace(5) %354, align 1
  %362 = add nsw i32 %355, -8
  %363 = getelementptr inbounds nuw i8, ptr addrspace(5) %354, i32 8
  br label %376

364:                                              ; preds = %358, %364
  %365 = phi i32 [ %374, %364 ], [ 0, %358 ]
  %366 = phi i64 [ %373, %364 ], [ 0, %358 ]
  %367 = getelementptr inbounds nuw i8, ptr addrspace(5) %354, i32 %365
  %368 = load i8, ptr addrspace(5) %367, align 1, !tbaa !32
  %369 = zext i8 %368 to i64
  %370 = shl i32 %365, 3
  %371 = zext nneg i32 %370 to i64
  %372 = shl nuw i64 %369, %371
  %373 = or i64 %372, %366
  %374 = add nuw nsw i32 %365, 1
  %375 = icmp eq i32 %374, %355
  br i1 %375, label %376, label %364

376:                                              ; preds = %364, %360, %358
  %377 = phi ptr addrspace(5) [ %363, %360 ], [ %354, %358 ], [ %354, %364 ]
  %378 = phi i32 [ %362, %360 ], [ 0, %358 ], [ 0, %364 ]
  %379 = phi i64 [ %361, %360 ], [ 0, %358 ], [ %373, %364 ]
  %380 = icmp ugt i32 %378, 7
  br i1 %380, label %383, label %381

381:                                              ; preds = %376
  %382 = icmp eq i32 %378, 0
  br i1 %382, label %399, label %387

383:                                              ; preds = %376
  %384 = load i64, ptr addrspace(5) %377, align 1
  %385 = add nsw i32 %378, -8
  %386 = getelementptr inbounds nuw i8, ptr addrspace(5) %377, i32 8
  br label %399

387:                                              ; preds = %381, %387
  %388 = phi i32 [ %397, %387 ], [ 0, %381 ]
  %389 = phi i64 [ %396, %387 ], [ 0, %381 ]
  %390 = getelementptr inbounds nuw i8, ptr addrspace(5) %377, i32 %388
  %391 = load i8, ptr addrspace(5) %390, align 1, !tbaa !32
  %392 = zext i8 %391 to i64
  %393 = shl i32 %388, 3
  %394 = zext nneg i32 %393 to i64
  %395 = shl nuw i64 %392, %394
  %396 = or i64 %395, %389
  %397 = add nuw nsw i32 %388, 1
  %398 = icmp eq i32 %397, %378
  br i1 %398, label %399, label %387

399:                                              ; preds = %387, %383, %381
  %400 = phi ptr addrspace(5) [ %386, %383 ], [ %377, %381 ], [ %377, %387 ]
  %401 = phi i32 [ %385, %383 ], [ 0, %381 ], [ 0, %387 ]
  %402 = phi i64 [ %384, %383 ], [ 0, %381 ], [ %396, %387 ]
  %403 = icmp ugt i32 %401, 7
  br i1 %403, label %406, label %404

404:                                              ; preds = %399
  %405 = icmp eq i32 %401, 0
  br i1 %405, label %420, label %408

406:                                              ; preds = %399
  %407 = load i64, ptr addrspace(5) %400, align 1
  br label %420

408:                                              ; preds = %404, %408
  %409 = phi i32 [ %418, %408 ], [ 0, %404 ]
  %410 = phi i64 [ %417, %408 ], [ 0, %404 ]
  %411 = getelementptr inbounds nuw i8, ptr addrspace(5) %400, i32 %409
  %412 = load i8, ptr addrspace(5) %411, align 1, !tbaa !32
  %413 = zext i8 %412 to i64
  %414 = shl i32 %409, 3
  %415 = zext nneg i32 %414 to i64
  %416 = shl nuw i64 %413, %415
  %417 = or i64 %416, %410
  %418 = add nuw nsw i32 %409, 1
  %419 = icmp eq i32 %418, %401
  br i1 %419, label %420, label %408

420:                                              ; preds = %408, %406, %404
  %421 = phi i64 [ %407, %406 ], [ 0, %404 ], [ %417, %408 ]
  %422 = shl nuw nsw i64 %262, 2
  %423 = add nuw nsw i64 %422, 28
  %424 = and i64 %423, 480
  %425 = and i64 %261, -225
  %426 = or i64 %425, %264
  %427 = or i64 %426, %424
  %428 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %11)
  %429 = icmp eq i32 %11, %428
  br i1 %429, label %430, label %450

430:                                              ; preds = %420
  %431 = load atomic i64, ptr addrspace(1) %253 syncscope("one-as") acquire, align 8
  %432 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %433 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %434 = and i64 %433, %431
  %435 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %432, i64 %434
  %436 = load atomic i64, ptr addrspace(1) %435 syncscope("one-as") monotonic, align 8
  %437 = cmpxchg ptr addrspace(1) %253, i64 %431, i64 %436 syncscope("one-as") acquire monotonic, align 8
  %438 = extractvalue { i64, i1 } %437, 1
  %439 = extractvalue { i64, i1 } %437, 0
  br i1 %438, label %450, label %440

440:                                              ; preds = %430, %440
  %441 = phi i64 [ %449, %440 ], [ %439, %430 ]
  call void @llvm.amdgcn.s.sleep(i32 1)
  %442 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %443 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %444 = and i64 %443, %441
  %445 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %442, i64 %444
  %446 = load atomic i64, ptr addrspace(1) %445 syncscope("one-as") monotonic, align 8
  %447 = cmpxchg ptr addrspace(1) %253, i64 %441, i64 %446 syncscope("one-as") acquire monotonic, align 8
  %448 = extractvalue { i64, i1 } %447, 1
  %449 = extractvalue { i64, i1 } %447, 0
  br i1 %448, label %450, label %440

450:                                              ; preds = %440, %430, %420
  %451 = phi i64 [ 0, %420 ], [ %439, %430 ], [ %449, %440 ]
  %452 = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %451)
  %453 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %454 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %455 = and i64 %454, %452
  %456 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %453, i64 %455
  %457 = load ptr addrspace(1), ptr addrspace(1) %44, align 8, !tbaa !21
  %458 = getelementptr inbounds nuw [4096 x i8], ptr addrspace(1) %457, i64 %455
  %459 = call i32 @llvm.amdgcn.ballot.i32(i1 true)
  br i1 %429, label %460, label %464

460:                                              ; preds = %450
  %461 = zext i32 %459 to i64
  %462 = getelementptr inbounds nuw i8, ptr addrspace(1) %456, i64 16
  %463 = getelementptr inbounds nuw i8, ptr addrspace(1) %456, i64 8
  store i64 %461, ptr addrspace(1) %463, align 8, !tbaa !22
  store <2 x i32> <i32 2, i32 1>, ptr addrspace(1) %462, align 8, !tbaa !9
  br label %464

464:                                              ; preds = %460, %450
  %465 = getelementptr inbounds nuw [64 x i8], ptr addrspace(1) %458, i64 %53
  store i64 %427, ptr addrspace(1) %465, align 8, !tbaa !14
  %466 = getelementptr inbounds nuw i8, ptr addrspace(1) %465, i64 8
  store i64 %287, ptr addrspace(1) %466, align 8, !tbaa !14
  %467 = getelementptr inbounds nuw i8, ptr addrspace(1) %465, i64 16
  store i64 %310, ptr addrspace(1) %467, align 8, !tbaa !14
  %468 = getelementptr inbounds nuw i8, ptr addrspace(1) %465, i64 24
  store i64 %333, ptr addrspace(1) %468, align 8, !tbaa !14
  %469 = getelementptr inbounds nuw i8, ptr addrspace(1) %465, i64 32
  store i64 %356, ptr addrspace(1) %469, align 8, !tbaa !14
  %470 = getelementptr inbounds nuw i8, ptr addrspace(1) %465, i64 40
  store i64 %379, ptr addrspace(1) %470, align 8, !tbaa !14
  %471 = getelementptr inbounds nuw i8, ptr addrspace(1) %465, i64 48
  store i64 %402, ptr addrspace(1) %471, align 8, !tbaa !14
  %472 = getelementptr inbounds nuw i8, ptr addrspace(1) %465, i64 56
  store i64 %421, ptr addrspace(1) %472, align 8, !tbaa !14
  br i1 %429, label %473, label %500

473:                                              ; preds = %464
  %474 = load atomic i64, ptr addrspace(1) %254 syncscope("one-as") monotonic, align 8
  %475 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %476 = and i64 %475, %452
  %477 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %453, i64 %476
  store i64 %474, ptr addrspace(1) %477, align 8, !tbaa !24
  %478 = cmpxchg ptr addrspace(1) %254, i64 %474, i64 %452 syncscope("one-as") release monotonic, align 8
  %479 = extractvalue { i64, i1 } %478, 1
  br i1 %479, label %485, label %480

480:                                              ; preds = %473, %480
  %481 = phi { i64, i1 } [ %483, %480 ], [ %478, %473 ]
  %482 = extractvalue { i64, i1 } %481, 0
  call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %482, ptr addrspace(1) %477, align 8, !tbaa !24
  %483 = cmpxchg ptr addrspace(1) %254, i64 %482, i64 %452 syncscope("one-as") release monotonic, align 8
  %484 = extractvalue { i64, i1 } %483, 1
  br i1 %484, label %485, label %480

485:                                              ; preds = %480, %473
  %486 = load i64, ptr addrspace(1) %255, align 8
  %487 = inttoptr i64 %486 to ptr addrspace(1)
  %488 = getelementptr inbounds nuw i8, ptr addrspace(1) %487, i64 8
  %489 = atomicrmw add ptr addrspace(1) %488, i64 1 syncscope("one-as") release, align 8, !amdgpu.no.fine.grained.memory !25, !amdgpu.no.remote.memory !25
  %490 = getelementptr inbounds nuw i8, ptr addrspace(1) %487, i64 16
  %491 = load i64, ptr addrspace(1) %490, align 16, !tbaa !26
  %492 = icmp eq i64 %491, 0
  br i1 %492, label %500, label %493

493:                                              ; preds = %485
  %494 = inttoptr i64 %491 to ptr addrspace(1)
  %495 = getelementptr inbounds nuw i8, ptr addrspace(1) %487, i64 24
  %496 = load i32, ptr addrspace(1) %495, align 8, !tbaa !28
  %497 = zext i32 %496 to i64
  store atomic i64 %497, ptr addrspace(1) %494 syncscope("one-as") release, align 8
  %498 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %496)
  %499 = and i32 %498, 16777215
  call void @llvm.amdgcn.s.sendmsg(i32 1, i32 %499)
  br label %500

500:                                              ; preds = %493, %485, %464
  %501 = getelementptr inbounds nuw i8, ptr addrspace(1) %456, i64 20
  br label %502

502:                                              ; preds = %510, %500
  br i1 %429, label %503, label %506

503:                                              ; preds = %502
  %504 = load atomic i32, ptr addrspace(1) %501 syncscope("one-as") acquire, align 4
  %505 = and i32 %504, 1
  br label %506

506:                                              ; preds = %503, %502
  %507 = phi i32 [ %505, %503 ], [ 1, %502 ]
  %508 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %507)
  %509 = icmp eq i32 %508, 0
  br i1 %509, label %511, label %510

510:                                              ; preds = %506
  call void @llvm.amdgcn.s.sleep(i32 1)
  br label %502

511:                                              ; preds = %506
  %512 = load i64, ptr addrspace(1) %465, align 8, !tbaa !14
  %513 = load i64, ptr addrspace(1) %466, align 8, !tbaa !14
  br i1 %429, label %514, label %531

514:                                              ; preds = %511
  %515 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %516 = add i64 %515, 1
  %517 = add i64 %516, %452
  %518 = icmp eq i64 %517, 0
  %519 = select i1 %518, i64 %516, i64 %517
  %520 = load atomic i64, ptr addrspace(1) %253 syncscope("one-as") monotonic, align 8
  %521 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %522 = and i64 %519, %515
  %523 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %521, i64 %522
  store i64 %520, ptr addrspace(1) %523, align 8, !tbaa !24
  %524 = cmpxchg ptr addrspace(1) %253, i64 %520, i64 %519 syncscope("one-as") release monotonic, align 8
  %525 = extractvalue { i64, i1 } %524, 1
  br i1 %525, label %531, label %526

526:                                              ; preds = %514, %526
  %527 = phi { i64, i1 } [ %529, %526 ], [ %524, %514 ]
  %528 = extractvalue { i64, i1 } %527, 0
  call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %528, ptr addrspace(1) %523, align 8, !tbaa !24
  %529 = cmpxchg ptr addrspace(1) %253, i64 %528, i64 %519 syncscope("one-as") release monotonic, align 8
  %530 = extractvalue { i64, i1 } %529, 1
  br i1 %530, label %531, label %526

531:                                              ; preds = %526, %514, %511
  %532 = insertelement <2 x i64> poison, i64 %512, i64 0
  %533 = insertelement <2 x i64> %532, i64 %513, i64 1
  %534 = sub i64 %257, %262
  %535 = getelementptr inbounds nuw i8, ptr addrspace(5) %258, i32 %263
  %536 = icmp eq i64 %534, 0
  br i1 %536, label %537, label %256

537:                                              ; preds = %531, %240
  %538 = phi <2 x i64> [ %242, %240 ], [ %533, %531 ]
  br label %539

539:                                              ; preds = %539, %537
  %540 = phi ptr [ %1, %537 ], [ %541, %539 ]
  %541 = getelementptr inbounds nuw i8, ptr %540, i64 1
  %542 = load i8, ptr %540, align 1, !tbaa !29
  %543 = icmp eq i8 %542, 0
  br i1 %543, label %544, label %539, !llvm.loop !33

544:                                              ; preds = %539
  %545 = extractelement <2 x i64> %538, i64 0
  %546 = icmp eq ptr %1, null
  br i1 %546, label %547, label %660

547:                                              ; preds = %544
  %548 = and i64 %545, -225
  %549 = or disjoint i64 %548, 32
  %550 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %11)
  %551 = icmp eq i32 %11, %550
  br i1 %551, label %552, label %573

552:                                              ; preds = %547
  %553 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 24
  %554 = load atomic i64, ptr addrspace(1) %553 syncscope("one-as") acquire, align 8
  %555 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %556 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %557 = and i64 %556, %554
  %558 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %555, i64 %557
  %559 = load atomic i64, ptr addrspace(1) %558 syncscope("one-as") monotonic, align 8
  %560 = cmpxchg ptr addrspace(1) %553, i64 %554, i64 %559 syncscope("one-as") acquire monotonic, align 8
  %561 = extractvalue { i64, i1 } %560, 1
  %562 = extractvalue { i64, i1 } %560, 0
  br i1 %561, label %573, label %563

563:                                              ; preds = %552, %563
  %564 = phi i64 [ %572, %563 ], [ %562, %552 ]
  call void @llvm.amdgcn.s.sleep(i32 1)
  %565 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %566 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %567 = and i64 %566, %564
  %568 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %565, i64 %567
  %569 = load atomic i64, ptr addrspace(1) %568 syncscope("one-as") monotonic, align 8
  %570 = cmpxchg ptr addrspace(1) %553, i64 %564, i64 %569 syncscope("one-as") acquire monotonic, align 8
  %571 = extractvalue { i64, i1 } %570, 1
  %572 = extractvalue { i64, i1 } %570, 0
  br i1 %571, label %573, label %563

573:                                              ; preds = %563, %552, %547
  %574 = phi i64 [ 0, %547 ], [ %562, %552 ], [ %572, %563 ]
  %575 = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %574)
  %576 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %577 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %578 = and i64 %577, %575
  %579 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %576, i64 %578
  %580 = load ptr addrspace(1), ptr addrspace(1) %44, align 8, !tbaa !21
  %581 = getelementptr inbounds nuw [4096 x i8], ptr addrspace(1) %580, i64 %578
  %582 = call i32 @llvm.amdgcn.ballot.i32(i1 true)
  br i1 %551, label %583, label %587

583:                                              ; preds = %573
  %584 = zext i32 %582 to i64
  %585 = getelementptr inbounds nuw i8, ptr addrspace(1) %579, i64 16
  %586 = getelementptr inbounds nuw i8, ptr addrspace(1) %579, i64 8
  store i64 %584, ptr addrspace(1) %586, align 8, !tbaa !22
  store <2 x i32> <i32 2, i32 1>, ptr addrspace(1) %585, align 8, !tbaa !9
  br label %587

587:                                              ; preds = %583, %573
  %588 = getelementptr inbounds nuw [64 x i8], ptr addrspace(1) %581, i64 %53
  store i64 %549, ptr addrspace(1) %588, align 8, !tbaa !14
  %589 = getelementptr inbounds nuw i8, ptr addrspace(1) %588, i64 8
  store i64 0, ptr addrspace(1) %589, align 8, !tbaa !14
  %590 = getelementptr inbounds nuw i8, ptr addrspace(1) %588, i64 16
  store i64 0, ptr addrspace(1) %590, align 8, !tbaa !14
  %591 = getelementptr inbounds nuw i8, ptr addrspace(1) %588, i64 24
  store i64 0, ptr addrspace(1) %591, align 8, !tbaa !14
  %592 = getelementptr inbounds nuw i8, ptr addrspace(1) %588, i64 32
  store i64 0, ptr addrspace(1) %592, align 8, !tbaa !14
  %593 = getelementptr inbounds nuw i8, ptr addrspace(1) %588, i64 40
  store i64 0, ptr addrspace(1) %593, align 8, !tbaa !14
  %594 = getelementptr inbounds nuw i8, ptr addrspace(1) %588, i64 48
  store i64 0, ptr addrspace(1) %594, align 8, !tbaa !14
  %595 = getelementptr inbounds nuw i8, ptr addrspace(1) %588, i64 56
  store i64 0, ptr addrspace(1) %595, align 8, !tbaa !14
  br i1 %551, label %596, label %625

596:                                              ; preds = %587
  %597 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 32
  %598 = load atomic i64, ptr addrspace(1) %597 syncscope("one-as") monotonic, align 8
  %599 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %600 = and i64 %599, %575
  %601 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %576, i64 %600
  store i64 %598, ptr addrspace(1) %601, align 8, !tbaa !24
  %602 = cmpxchg ptr addrspace(1) %597, i64 %598, i64 %575 syncscope("one-as") release monotonic, align 8
  %603 = extractvalue { i64, i1 } %602, 1
  br i1 %603, label %609, label %604

604:                                              ; preds = %596, %604
  %605 = phi { i64, i1 } [ %607, %604 ], [ %602, %596 ]
  %606 = extractvalue { i64, i1 } %605, 0
  call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %606, ptr addrspace(1) %601, align 8, !tbaa !24
  %607 = cmpxchg ptr addrspace(1) %597, i64 %606, i64 %575 syncscope("one-as") release monotonic, align 8
  %608 = extractvalue { i64, i1 } %607, 1
  br i1 %608, label %609, label %604

609:                                              ; preds = %604, %596
  %610 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 16
  %611 = load i64, ptr addrspace(1) %610, align 8
  %612 = inttoptr i64 %611 to ptr addrspace(1)
  %613 = getelementptr inbounds nuw i8, ptr addrspace(1) %612, i64 8
  %614 = atomicrmw add ptr addrspace(1) %613, i64 1 syncscope("one-as") release, align 8, !amdgpu.no.fine.grained.memory !25, !amdgpu.no.remote.memory !25
  %615 = getelementptr inbounds nuw i8, ptr addrspace(1) %612, i64 16
  %616 = load i64, ptr addrspace(1) %615, align 16, !tbaa !26
  %617 = icmp eq i64 %616, 0
  br i1 %617, label %625, label %618

618:                                              ; preds = %609
  %619 = inttoptr i64 %616 to ptr addrspace(1)
  %620 = getelementptr inbounds nuw i8, ptr addrspace(1) %612, i64 24
  %621 = load i32, ptr addrspace(1) %620, align 8, !tbaa !28
  %622 = zext i32 %621 to i64
  store atomic i64 %622, ptr addrspace(1) %619 syncscope("one-as") release, align 8
  %623 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %621)
  %624 = and i32 %623, 16777215
  call void @llvm.amdgcn.s.sendmsg(i32 1, i32 %624)
  br label %625

625:                                              ; preds = %618, %609, %587
  %626 = getelementptr inbounds nuw i8, ptr addrspace(1) %579, i64 20
  br label %627

627:                                              ; preds = %635, %625
  br i1 %551, label %628, label %631

628:                                              ; preds = %627
  %629 = load atomic i32, ptr addrspace(1) %626 syncscope("one-as") acquire, align 4
  %630 = and i32 %629, 1
  br label %631

631:                                              ; preds = %628, %627
  %632 = phi i32 [ %630, %628 ], [ 1, %627 ]
  %633 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %632)
  %634 = icmp eq i32 %633, 0
  br i1 %634, label %636, label %635

635:                                              ; preds = %631
  call void @llvm.amdgcn.s.sleep(i32 1)
  br label %627

636:                                              ; preds = %631
  %637 = load i64, ptr addrspace(1) %588, align 8, !tbaa !14
  %638 = load i64, ptr addrspace(1) %589, align 8, !tbaa !14
  br i1 %551, label %639, label %657

639:                                              ; preds = %636
  %640 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %641 = add i64 %640, 1
  %642 = add i64 %641, %575
  %643 = icmp eq i64 %642, 0
  %644 = select i1 %643, i64 %641, i64 %642
  %645 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 24
  %646 = load atomic i64, ptr addrspace(1) %645 syncscope("one-as") monotonic, align 8
  %647 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %648 = and i64 %644, %640
  %649 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %647, i64 %648
  store i64 %646, ptr addrspace(1) %649, align 8, !tbaa !24
  %650 = cmpxchg ptr addrspace(1) %645, i64 %646, i64 %644 syncscope("one-as") release monotonic, align 8
  %651 = extractvalue { i64, i1 } %650, 1
  br i1 %651, label %657, label %652

652:                                              ; preds = %639, %652
  %653 = phi { i64, i1 } [ %655, %652 ], [ %650, %639 ]
  %654 = extractvalue { i64, i1 } %653, 0
  call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %654, ptr addrspace(1) %649, align 8, !tbaa !24
  %655 = cmpxchg ptr addrspace(1) %645, i64 %654, i64 %644 syncscope("one-as") release monotonic, align 8
  %656 = extractvalue { i64, i1 } %655, 1
  br i1 %656, label %657, label %652

657:                                              ; preds = %652, %639, %636
  %658 = insertelement <2 x i64> poison, i64 %637, i64 0
  %659 = insertelement <2 x i64> %658, i64 %638, i64 1
  br label %960

660:                                              ; preds = %544
  %661 = ptrtoint ptr %541 to i64
  %662 = ptrtoint ptr %1 to i64
  %663 = sub i64 %661, %662
  %664 = shl i64 %663, 32
  %665 = ashr exact i64 %664, 32
  %666 = and i64 %545, 2
  %667 = and i64 %545, -3
  %668 = insertelement <2 x i64> <i64 poison, i64 0>, i64 %667, i64 0
  %669 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 24
  %670 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 32
  %671 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 16
  br label %672

672:                                              ; preds = %954, %660
  %673 = phi i64 [ %665, %660 ], [ %957, %954 ]
  %674 = phi ptr [ %1, %660 ], [ %958, %954 ]
  %675 = phi <2 x i64> [ %668, %660 ], [ %956, %954 ]
  %676 = icmp ugt i64 %673, 56
  %677 = extractelement <2 x i64> %675, i64 0
  %678 = call i64 @llvm.umin.i64(i64 %673, i64 56)
  %679 = trunc nuw nsw i64 %678 to i32
  %680 = select i1 %676, i64 0, i64 %666
  %681 = icmp ugt i64 %673, 7
  br i1 %681, label %684, label %682

682:                                              ; preds = %672
  %683 = icmp eq i64 %673, 0
  br i1 %683, label %701, label %688

684:                                              ; preds = %672
  %685 = load i64, ptr %674, align 1
  %686 = add nsw i32 %679, -8
  %687 = getelementptr inbounds nuw i8, ptr %674, i64 8
  br label %701

688:                                              ; preds = %682, %688
  %689 = phi i32 [ %699, %688 ], [ 0, %682 ]
  %690 = phi i64 [ %698, %688 ], [ 0, %682 ]
  %691 = zext nneg i32 %689 to i64
  %692 = getelementptr inbounds nuw i8, ptr %674, i64 %691
  %693 = load i8, ptr %692, align 1, !tbaa !32
  %694 = zext i8 %693 to i64
  %695 = shl i32 %689, 3
  %696 = zext nneg i32 %695 to i64
  %697 = shl nuw i64 %694, %696
  %698 = or i64 %697, %690
  %699 = add nuw nsw i32 %689, 1
  %700 = icmp eq i32 %699, %679
  br i1 %700, label %701, label %688

701:                                              ; preds = %688, %684, %682
  %702 = phi ptr [ %687, %684 ], [ %674, %682 ], [ %674, %688 ]
  %703 = phi i32 [ %686, %684 ], [ 0, %682 ], [ 0, %688 ]
  %704 = phi i64 [ %685, %684 ], [ 0, %682 ], [ %698, %688 ]
  %705 = icmp ugt i32 %703, 7
  br i1 %705, label %708, label %706

706:                                              ; preds = %701
  %707 = icmp eq i32 %703, 0
  br i1 %707, label %725, label %712

708:                                              ; preds = %701
  %709 = load i64, ptr %702, align 1
  %710 = add nsw i32 %703, -8
  %711 = getelementptr inbounds nuw i8, ptr %702, i64 8
  br label %725

712:                                              ; preds = %706, %712
  %713 = phi i32 [ %723, %712 ], [ 0, %706 ]
  %714 = phi i64 [ %722, %712 ], [ 0, %706 ]
  %715 = zext nneg i32 %713 to i64
  %716 = getelementptr inbounds nuw i8, ptr %702, i64 %715
  %717 = load i8, ptr %716, align 1, !tbaa !32
  %718 = zext i8 %717 to i64
  %719 = shl i32 %713, 3
  %720 = zext nneg i32 %719 to i64
  %721 = shl nuw i64 %718, %720
  %722 = or i64 %721, %714
  %723 = add nuw nsw i32 %713, 1
  %724 = icmp eq i32 %723, %703
  br i1 %724, label %725, label %712

725:                                              ; preds = %712, %708, %706
  %726 = phi ptr [ %711, %708 ], [ %702, %706 ], [ %702, %712 ]
  %727 = phi i32 [ %710, %708 ], [ 0, %706 ], [ 0, %712 ]
  %728 = phi i64 [ %709, %708 ], [ 0, %706 ], [ %722, %712 ]
  %729 = icmp ugt i32 %727, 7
  br i1 %729, label %732, label %730

730:                                              ; preds = %725
  %731 = icmp eq i32 %727, 0
  br i1 %731, label %749, label %736

732:                                              ; preds = %725
  %733 = load i64, ptr %726, align 1
  %734 = add nsw i32 %727, -8
  %735 = getelementptr inbounds nuw i8, ptr %726, i64 8
  br label %749

736:                                              ; preds = %730, %736
  %737 = phi i32 [ %747, %736 ], [ 0, %730 ]
  %738 = phi i64 [ %746, %736 ], [ 0, %730 ]
  %739 = zext nneg i32 %737 to i64
  %740 = getelementptr inbounds nuw i8, ptr %726, i64 %739
  %741 = load i8, ptr %740, align 1, !tbaa !32
  %742 = zext i8 %741 to i64
  %743 = shl i32 %737, 3
  %744 = zext nneg i32 %743 to i64
  %745 = shl nuw i64 %742, %744
  %746 = or i64 %745, %738
  %747 = add nuw nsw i32 %737, 1
  %748 = icmp eq i32 %747, %727
  br i1 %748, label %749, label %736

749:                                              ; preds = %736, %732, %730
  %750 = phi ptr [ %735, %732 ], [ %726, %730 ], [ %726, %736 ]
  %751 = phi i32 [ %734, %732 ], [ 0, %730 ], [ 0, %736 ]
  %752 = phi i64 [ %733, %732 ], [ 0, %730 ], [ %746, %736 ]
  %753 = icmp ugt i32 %751, 7
  br i1 %753, label %756, label %754

754:                                              ; preds = %749
  %755 = icmp eq i32 %751, 0
  br i1 %755, label %773, label %760

756:                                              ; preds = %749
  %757 = load i64, ptr %750, align 1
  %758 = add nsw i32 %751, -8
  %759 = getelementptr inbounds nuw i8, ptr %750, i64 8
  br label %773

760:                                              ; preds = %754, %760
  %761 = phi i32 [ %771, %760 ], [ 0, %754 ]
  %762 = phi i64 [ %770, %760 ], [ 0, %754 ]
  %763 = zext nneg i32 %761 to i64
  %764 = getelementptr inbounds nuw i8, ptr %750, i64 %763
  %765 = load i8, ptr %764, align 1, !tbaa !32
  %766 = zext i8 %765 to i64
  %767 = shl i32 %761, 3
  %768 = zext nneg i32 %767 to i64
  %769 = shl nuw i64 %766, %768
  %770 = or i64 %769, %762
  %771 = add nuw nsw i32 %761, 1
  %772 = icmp eq i32 %771, %751
  br i1 %772, label %773, label %760

773:                                              ; preds = %760, %756, %754
  %774 = phi ptr [ %759, %756 ], [ %750, %754 ], [ %750, %760 ]
  %775 = phi i32 [ %758, %756 ], [ 0, %754 ], [ 0, %760 ]
  %776 = phi i64 [ %757, %756 ], [ 0, %754 ], [ %770, %760 ]
  %777 = icmp ugt i32 %775, 7
  br i1 %777, label %780, label %778

778:                                              ; preds = %773
  %779 = icmp eq i32 %775, 0
  br i1 %779, label %797, label %784

780:                                              ; preds = %773
  %781 = load i64, ptr %774, align 1
  %782 = add nsw i32 %775, -8
  %783 = getelementptr inbounds nuw i8, ptr %774, i64 8
  br label %797

784:                                              ; preds = %778, %784
  %785 = phi i32 [ %795, %784 ], [ 0, %778 ]
  %786 = phi i64 [ %794, %784 ], [ 0, %778 ]
  %787 = zext nneg i32 %785 to i64
  %788 = getelementptr inbounds nuw i8, ptr %774, i64 %787
  %789 = load i8, ptr %788, align 1, !tbaa !32
  %790 = zext i8 %789 to i64
  %791 = shl i32 %785, 3
  %792 = zext nneg i32 %791 to i64
  %793 = shl nuw i64 %790, %792
  %794 = or i64 %793, %786
  %795 = add nuw nsw i32 %785, 1
  %796 = icmp eq i32 %795, %775
  br i1 %796, label %797, label %784

797:                                              ; preds = %784, %780, %778
  %798 = phi ptr [ %783, %780 ], [ %774, %778 ], [ %774, %784 ]
  %799 = phi i32 [ %782, %780 ], [ 0, %778 ], [ 0, %784 ]
  %800 = phi i64 [ %781, %780 ], [ 0, %778 ], [ %794, %784 ]
  %801 = icmp ugt i32 %799, 7
  br i1 %801, label %804, label %802

802:                                              ; preds = %797
  %803 = icmp eq i32 %799, 0
  br i1 %803, label %821, label %808

804:                                              ; preds = %797
  %805 = load i64, ptr %798, align 1
  %806 = add nsw i32 %799, -8
  %807 = getelementptr inbounds nuw i8, ptr %798, i64 8
  br label %821

808:                                              ; preds = %802, %808
  %809 = phi i32 [ %819, %808 ], [ 0, %802 ]
  %810 = phi i64 [ %818, %808 ], [ 0, %802 ]
  %811 = zext nneg i32 %809 to i64
  %812 = getelementptr inbounds nuw i8, ptr %798, i64 %811
  %813 = load i8, ptr %812, align 1, !tbaa !32
  %814 = zext i8 %813 to i64
  %815 = shl i32 %809, 3
  %816 = zext nneg i32 %815 to i64
  %817 = shl nuw i64 %814, %816
  %818 = or i64 %817, %810
  %819 = add nuw nsw i32 %809, 1
  %820 = icmp eq i32 %819, %799
  br i1 %820, label %821, label %808

821:                                              ; preds = %808, %804, %802
  %822 = phi ptr [ %807, %804 ], [ %798, %802 ], [ %798, %808 ]
  %823 = phi i32 [ %806, %804 ], [ 0, %802 ], [ 0, %808 ]
  %824 = phi i64 [ %805, %804 ], [ 0, %802 ], [ %818, %808 ]
  %825 = icmp ugt i32 %823, 7
  br i1 %825, label %828, label %826

826:                                              ; preds = %821
  %827 = icmp eq i32 %823, 0
  br i1 %827, label %843, label %830

828:                                              ; preds = %821
  %829 = load i64, ptr %822, align 1
  br label %843

830:                                              ; preds = %826, %830
  %831 = phi i32 [ %841, %830 ], [ 0, %826 ]
  %832 = phi i64 [ %840, %830 ], [ 0, %826 ]
  %833 = zext nneg i32 %831 to i64
  %834 = getelementptr inbounds nuw i8, ptr %822, i64 %833
  %835 = load i8, ptr %834, align 1, !tbaa !32
  %836 = zext i8 %835 to i64
  %837 = shl i32 %831, 3
  %838 = zext nneg i32 %837 to i64
  %839 = shl nuw i64 %836, %838
  %840 = or i64 %839, %832
  %841 = add nuw nsw i32 %831, 1
  %842 = icmp eq i32 %841, %823
  br i1 %842, label %843, label %830

843:                                              ; preds = %830, %828, %826
  %844 = phi i64 [ %829, %828 ], [ 0, %826 ], [ %840, %830 ]
  %845 = shl nuw nsw i64 %678, 2
  %846 = add nuw nsw i64 %845, 28
  %847 = and i64 %846, 480
  %848 = and i64 %677, -225
  %849 = or i64 %848, %680
  %850 = or i64 %849, %847
  %851 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %11)
  %852 = icmp eq i32 %11, %851
  br i1 %852, label %853, label %873

853:                                              ; preds = %843
  %854 = load atomic i64, ptr addrspace(1) %669 syncscope("one-as") acquire, align 8
  %855 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %856 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %857 = and i64 %856, %854
  %858 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %855, i64 %857
  %859 = load atomic i64, ptr addrspace(1) %858 syncscope("one-as") monotonic, align 8
  %860 = cmpxchg ptr addrspace(1) %669, i64 %854, i64 %859 syncscope("one-as") acquire monotonic, align 8
  %861 = extractvalue { i64, i1 } %860, 1
  %862 = extractvalue { i64, i1 } %860, 0
  br i1 %861, label %873, label %863

863:                                              ; preds = %853, %863
  %864 = phi i64 [ %872, %863 ], [ %862, %853 ]
  call void @llvm.amdgcn.s.sleep(i32 1)
  %865 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %866 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %867 = and i64 %866, %864
  %868 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %865, i64 %867
  %869 = load atomic i64, ptr addrspace(1) %868 syncscope("one-as") monotonic, align 8
  %870 = cmpxchg ptr addrspace(1) %669, i64 %864, i64 %869 syncscope("one-as") acquire monotonic, align 8
  %871 = extractvalue { i64, i1 } %870, 1
  %872 = extractvalue { i64, i1 } %870, 0
  br i1 %871, label %873, label %863

873:                                              ; preds = %863, %853, %843
  %874 = phi i64 [ 0, %843 ], [ %862, %853 ], [ %872, %863 ]
  %875 = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %874)
  %876 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %877 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %878 = and i64 %877, %875
  %879 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %876, i64 %878
  %880 = load ptr addrspace(1), ptr addrspace(1) %44, align 8, !tbaa !21
  %881 = getelementptr inbounds nuw [4096 x i8], ptr addrspace(1) %880, i64 %878
  %882 = call i32 @llvm.amdgcn.ballot.i32(i1 true)
  br i1 %852, label %883, label %887

883:                                              ; preds = %873
  %884 = zext i32 %882 to i64
  %885 = getelementptr inbounds nuw i8, ptr addrspace(1) %879, i64 16
  %886 = getelementptr inbounds nuw i8, ptr addrspace(1) %879, i64 8
  store i64 %884, ptr addrspace(1) %886, align 8, !tbaa !22
  store <2 x i32> <i32 2, i32 1>, ptr addrspace(1) %885, align 8, !tbaa !9
  br label %887

887:                                              ; preds = %883, %873
  %888 = getelementptr inbounds nuw [64 x i8], ptr addrspace(1) %881, i64 %53
  store i64 %850, ptr addrspace(1) %888, align 8, !tbaa !14
  %889 = getelementptr inbounds nuw i8, ptr addrspace(1) %888, i64 8
  store i64 %704, ptr addrspace(1) %889, align 8, !tbaa !14
  %890 = getelementptr inbounds nuw i8, ptr addrspace(1) %888, i64 16
  store i64 %728, ptr addrspace(1) %890, align 8, !tbaa !14
  %891 = getelementptr inbounds nuw i8, ptr addrspace(1) %888, i64 24
  store i64 %752, ptr addrspace(1) %891, align 8, !tbaa !14
  %892 = getelementptr inbounds nuw i8, ptr addrspace(1) %888, i64 32
  store i64 %776, ptr addrspace(1) %892, align 8, !tbaa !14
  %893 = getelementptr inbounds nuw i8, ptr addrspace(1) %888, i64 40
  store i64 %800, ptr addrspace(1) %893, align 8, !tbaa !14
  %894 = getelementptr inbounds nuw i8, ptr addrspace(1) %888, i64 48
  store i64 %824, ptr addrspace(1) %894, align 8, !tbaa !14
  %895 = getelementptr inbounds nuw i8, ptr addrspace(1) %888, i64 56
  store i64 %844, ptr addrspace(1) %895, align 8, !tbaa !14
  br i1 %852, label %896, label %923

896:                                              ; preds = %887
  %897 = load atomic i64, ptr addrspace(1) %670 syncscope("one-as") monotonic, align 8
  %898 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %899 = and i64 %898, %875
  %900 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %876, i64 %899
  store i64 %897, ptr addrspace(1) %900, align 8, !tbaa !24
  %901 = cmpxchg ptr addrspace(1) %670, i64 %897, i64 %875 syncscope("one-as") release monotonic, align 8
  %902 = extractvalue { i64, i1 } %901, 1
  br i1 %902, label %908, label %903

903:                                              ; preds = %896, %903
  %904 = phi { i64, i1 } [ %906, %903 ], [ %901, %896 ]
  %905 = extractvalue { i64, i1 } %904, 0
  call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %905, ptr addrspace(1) %900, align 8, !tbaa !24
  %906 = cmpxchg ptr addrspace(1) %670, i64 %905, i64 %875 syncscope("one-as") release monotonic, align 8
  %907 = extractvalue { i64, i1 } %906, 1
  br i1 %907, label %908, label %903

908:                                              ; preds = %903, %896
  %909 = load i64, ptr addrspace(1) %671, align 8
  %910 = inttoptr i64 %909 to ptr addrspace(1)
  %911 = getelementptr inbounds nuw i8, ptr addrspace(1) %910, i64 8
  %912 = atomicrmw add ptr addrspace(1) %911, i64 1 syncscope("one-as") release, align 8, !amdgpu.no.fine.grained.memory !25, !amdgpu.no.remote.memory !25
  %913 = getelementptr inbounds nuw i8, ptr addrspace(1) %910, i64 16
  %914 = load i64, ptr addrspace(1) %913, align 16, !tbaa !26
  %915 = icmp eq i64 %914, 0
  br i1 %915, label %923, label %916

916:                                              ; preds = %908
  %917 = inttoptr i64 %914 to ptr addrspace(1)
  %918 = getelementptr inbounds nuw i8, ptr addrspace(1) %910, i64 24
  %919 = load i32, ptr addrspace(1) %918, align 8, !tbaa !28
  %920 = zext i32 %919 to i64
  store atomic i64 %920, ptr addrspace(1) %917 syncscope("one-as") release, align 8
  %921 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %919)
  %922 = and i32 %921, 16777215
  call void @llvm.amdgcn.s.sendmsg(i32 1, i32 %922)
  br label %923

923:                                              ; preds = %916, %908, %887
  %924 = getelementptr inbounds nuw i8, ptr addrspace(1) %879, i64 20
  br label %925

925:                                              ; preds = %933, %923
  br i1 %852, label %926, label %929

926:                                              ; preds = %925
  %927 = load atomic i32, ptr addrspace(1) %924 syncscope("one-as") acquire, align 4
  %928 = and i32 %927, 1
  br label %929

929:                                              ; preds = %926, %925
  %930 = phi i32 [ %928, %926 ], [ 1, %925 ]
  %931 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %930)
  %932 = icmp eq i32 %931, 0
  br i1 %932, label %934, label %933

933:                                              ; preds = %929
  call void @llvm.amdgcn.s.sleep(i32 1)
  br label %925

934:                                              ; preds = %929
  %935 = load i64, ptr addrspace(1) %888, align 8, !tbaa !14
  %936 = load i64, ptr addrspace(1) %889, align 8, !tbaa !14
  br i1 %852, label %937, label %954

937:                                              ; preds = %934
  %938 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %939 = add i64 %938, 1
  %940 = add i64 %939, %875
  %941 = icmp eq i64 %940, 0
  %942 = select i1 %941, i64 %939, i64 %940
  %943 = load atomic i64, ptr addrspace(1) %669 syncscope("one-as") monotonic, align 8
  %944 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %945 = and i64 %942, %938
  %946 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %944, i64 %945
  store i64 %943, ptr addrspace(1) %946, align 8, !tbaa !24
  %947 = cmpxchg ptr addrspace(1) %669, i64 %943, i64 %942 syncscope("one-as") release monotonic, align 8
  %948 = extractvalue { i64, i1 } %947, 1
  br i1 %948, label %954, label %949

949:                                              ; preds = %937, %949
  %950 = phi { i64, i1 } [ %952, %949 ], [ %947, %937 ]
  %951 = extractvalue { i64, i1 } %950, 0
  call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %951, ptr addrspace(1) %946, align 8, !tbaa !24
  %952 = cmpxchg ptr addrspace(1) %669, i64 %951, i64 %942 syncscope("one-as") release monotonic, align 8
  %953 = extractvalue { i64, i1 } %952, 1
  br i1 %953, label %954, label %949

954:                                              ; preds = %949, %937, %934
  %955 = insertelement <2 x i64> poison, i64 %935, i64 0
  %956 = insertelement <2 x i64> %955, i64 %936, i64 1
  %957 = sub i64 %673, %678
  %958 = getelementptr inbounds nuw i8, ptr %674, i64 %678
  %959 = icmp eq i64 %957, 0
  br i1 %959, label %960, label %672

960:                                              ; preds = %954, %657
  %961 = phi <2 x i64> [ %659, %657 ], [ %956, %954 ]
  %962 = extractelement <2 x i64> %961, i64 0
  %963 = zext i32 %2 to i64
  %964 = and i64 %962, -225
  %965 = or disjoint i64 %964, 32
  %966 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %11)
  %967 = icmp eq i32 %11, %966
  br i1 %967, label %968, label %989

968:                                              ; preds = %960
  %969 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 24
  %970 = load atomic i64, ptr addrspace(1) %969 syncscope("one-as") acquire, align 8
  %971 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %972 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %973 = and i64 %972, %970
  %974 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %971, i64 %973
  %975 = load atomic i64, ptr addrspace(1) %974 syncscope("one-as") monotonic, align 8
  %976 = cmpxchg ptr addrspace(1) %969, i64 %970, i64 %975 syncscope("one-as") acquire monotonic, align 8
  %977 = extractvalue { i64, i1 } %976, 1
  %978 = extractvalue { i64, i1 } %976, 0
  br i1 %977, label %989, label %979

979:                                              ; preds = %968, %979
  %980 = phi i64 [ %988, %979 ], [ %978, %968 ]
  call void @llvm.amdgcn.s.sleep(i32 1)
  %981 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %982 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %983 = and i64 %982, %980
  %984 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %981, i64 %983
  %985 = load atomic i64, ptr addrspace(1) %984 syncscope("one-as") monotonic, align 8
  %986 = cmpxchg ptr addrspace(1) %969, i64 %980, i64 %985 syncscope("one-as") acquire monotonic, align 8
  %987 = extractvalue { i64, i1 } %986, 1
  %988 = extractvalue { i64, i1 } %986, 0
  br i1 %987, label %989, label %979

989:                                              ; preds = %979, %968, %960
  %990 = phi i64 [ 0, %960 ], [ %978, %968 ], [ %988, %979 ]
  %991 = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %990)
  %992 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %993 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %994 = and i64 %993, %991
  %995 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %992, i64 %994
  %996 = load ptr addrspace(1), ptr addrspace(1) %44, align 8, !tbaa !21
  %997 = getelementptr inbounds nuw [4096 x i8], ptr addrspace(1) %996, i64 %994
  %998 = call i32 @llvm.amdgcn.ballot.i32(i1 true)
  br i1 %967, label %999, label %1003

999:                                              ; preds = %989
  %1000 = zext i32 %998 to i64
  %1001 = getelementptr inbounds nuw i8, ptr addrspace(1) %995, i64 16
  %1002 = getelementptr inbounds nuw i8, ptr addrspace(1) %995, i64 8
  store i64 %1000, ptr addrspace(1) %1002, align 8, !tbaa !22
  store <2 x i32> <i32 2, i32 1>, ptr addrspace(1) %1001, align 8, !tbaa !9
  br label %1003

1003:                                             ; preds = %999, %989
  %1004 = getelementptr inbounds nuw [64 x i8], ptr addrspace(1) %997, i64 %53
  store i64 %965, ptr addrspace(1) %1004, align 8, !tbaa !14
  %1005 = getelementptr inbounds nuw i8, ptr addrspace(1) %1004, i64 8
  store i64 %963, ptr addrspace(1) %1005, align 8, !tbaa !14
  %1006 = getelementptr inbounds nuw i8, ptr addrspace(1) %1004, i64 16
  store i64 0, ptr addrspace(1) %1006, align 8, !tbaa !14
  %1007 = getelementptr inbounds nuw i8, ptr addrspace(1) %1004, i64 24
  store i64 0, ptr addrspace(1) %1007, align 8, !tbaa !14
  %1008 = getelementptr inbounds nuw i8, ptr addrspace(1) %1004, i64 32
  store i64 0, ptr addrspace(1) %1008, align 8, !tbaa !14
  %1009 = getelementptr inbounds nuw i8, ptr addrspace(1) %1004, i64 40
  store i64 0, ptr addrspace(1) %1009, align 8, !tbaa !14
  %1010 = getelementptr inbounds nuw i8, ptr addrspace(1) %1004, i64 48
  store i64 0, ptr addrspace(1) %1010, align 8, !tbaa !14
  %1011 = getelementptr inbounds nuw i8, ptr addrspace(1) %1004, i64 56
  store i64 0, ptr addrspace(1) %1011, align 8, !tbaa !14
  br i1 %967, label %1012, label %1041

1012:                                             ; preds = %1003
  %1013 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 32
  %1014 = load atomic i64, ptr addrspace(1) %1013 syncscope("one-as") monotonic, align 8
  %1015 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %1016 = and i64 %1015, %991
  %1017 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %992, i64 %1016
  store i64 %1014, ptr addrspace(1) %1017, align 8, !tbaa !24
  %1018 = cmpxchg ptr addrspace(1) %1013, i64 %1014, i64 %991 syncscope("one-as") release monotonic, align 8
  %1019 = extractvalue { i64, i1 } %1018, 1
  br i1 %1019, label %1025, label %1020

1020:                                             ; preds = %1012, %1020
  %1021 = phi { i64, i1 } [ %1023, %1020 ], [ %1018, %1012 ]
  %1022 = extractvalue { i64, i1 } %1021, 0
  call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %1022, ptr addrspace(1) %1017, align 8, !tbaa !24
  %1023 = cmpxchg ptr addrspace(1) %1013, i64 %1022, i64 %991 syncscope("one-as") release monotonic, align 8
  %1024 = extractvalue { i64, i1 } %1023, 1
  br i1 %1024, label %1025, label %1020

1025:                                             ; preds = %1020, %1012
  %1026 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 16
  %1027 = load i64, ptr addrspace(1) %1026, align 8
  %1028 = inttoptr i64 %1027 to ptr addrspace(1)
  %1029 = getelementptr inbounds nuw i8, ptr addrspace(1) %1028, i64 8
  %1030 = atomicrmw add ptr addrspace(1) %1029, i64 1 syncscope("one-as") release, align 8, !amdgpu.no.fine.grained.memory !25, !amdgpu.no.remote.memory !25
  %1031 = getelementptr inbounds nuw i8, ptr addrspace(1) %1028, i64 16
  %1032 = load i64, ptr addrspace(1) %1031, align 16, !tbaa !26
  %1033 = icmp eq i64 %1032, 0
  br i1 %1033, label %1041, label %1034

1034:                                             ; preds = %1025
  %1035 = inttoptr i64 %1032 to ptr addrspace(1)
  %1036 = getelementptr inbounds nuw i8, ptr addrspace(1) %1028, i64 24
  %1037 = load i32, ptr addrspace(1) %1036, align 8, !tbaa !28
  %1038 = zext i32 %1037 to i64
  store atomic i64 %1038, ptr addrspace(1) %1035 syncscope("one-as") release, align 8
  %1039 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1037)
  %1040 = and i32 %1039, 16777215
  call void @llvm.amdgcn.s.sendmsg(i32 1, i32 %1040)
  br label %1041

1041:                                             ; preds = %1034, %1025, %1003
  %1042 = getelementptr inbounds nuw i8, ptr addrspace(1) %995, i64 20
  br label %1043

1043:                                             ; preds = %1051, %1041
  br i1 %967, label %1044, label %1047

1044:                                             ; preds = %1043
  %1045 = load atomic i32, ptr addrspace(1) %1042 syncscope("one-as") acquire, align 4
  %1046 = and i32 %1045, 1
  br label %1047

1047:                                             ; preds = %1044, %1043
  %1048 = phi i32 [ %1046, %1044 ], [ 1, %1043 ]
  %1049 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1048)
  %1050 = icmp eq i32 %1049, 0
  br i1 %1050, label %1052, label %1051

1051:                                             ; preds = %1047
  call void @llvm.amdgcn.s.sleep(i32 1)
  br label %1043

1052:                                             ; preds = %1047
  %1053 = load i64, ptr addrspace(1) %1004, align 8, !tbaa !14
  br i1 %967, label %1054, label %1072

1054:                                             ; preds = %1052
  %1055 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %1056 = add i64 %1055, 1
  %1057 = add i64 %1056, %991
  %1058 = icmp eq i64 %1057, 0
  %1059 = select i1 %1058, i64 %1056, i64 %1057
  %1060 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 24
  %1061 = load atomic i64, ptr addrspace(1) %1060 syncscope("one-as") monotonic, align 8
  %1062 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %1063 = and i64 %1059, %1055
  %1064 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %1062, i64 %1063
  store i64 %1061, ptr addrspace(1) %1064, align 8, !tbaa !24
  %1065 = cmpxchg ptr addrspace(1) %1060, i64 %1061, i64 %1059 syncscope("one-as") release monotonic, align 8
  %1066 = extractvalue { i64, i1 } %1065, 1
  br i1 %1066, label %1072, label %1067

1067:                                             ; preds = %1054, %1067
  %1068 = phi { i64, i1 } [ %1070, %1067 ], [ %1065, %1054 ]
  %1069 = extractvalue { i64, i1 } %1068, 0
  call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %1069, ptr addrspace(1) %1064, align 8, !tbaa !24
  %1070 = cmpxchg ptr addrspace(1) %1060, i64 %1069, i64 %1059 syncscope("one-as") release monotonic, align 8
  %1071 = extractvalue { i64, i1 } %1070, 1
  br i1 %1071, label %1072, label %1067

1072:                                             ; preds = %1067, %1052, %1054
  br label %1073

1073:                                             ; preds = %1072, %1073
  %1074 = phi ptr [ %1075, %1073 ], [ %3, %1072 ]
  %1075 = getelementptr inbounds nuw i8, ptr %1074, i64 1
  %1076 = load i8, ptr %1074, align 1, !tbaa !29
  %1077 = icmp eq i8 %1076, 0
  br i1 %1077, label %1078, label %1073, !llvm.loop !34

1078:                                             ; preds = %1073
  %1079 = icmp eq ptr %3, null
  br i1 %1079, label %1080, label %1193

1080:                                             ; preds = %1078
  %1081 = and i64 %1053, -225
  %1082 = or disjoint i64 %1081, 32
  %1083 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %11)
  %1084 = icmp eq i32 %11, %1083
  br i1 %1084, label %1085, label %1106

1085:                                             ; preds = %1080
  %1086 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 24
  %1087 = load atomic i64, ptr addrspace(1) %1086 syncscope("one-as") acquire, align 8
  %1088 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %1089 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %1090 = and i64 %1089, %1087
  %1091 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %1088, i64 %1090
  %1092 = load atomic i64, ptr addrspace(1) %1091 syncscope("one-as") monotonic, align 8
  %1093 = cmpxchg ptr addrspace(1) %1086, i64 %1087, i64 %1092 syncscope("one-as") acquire monotonic, align 8
  %1094 = extractvalue { i64, i1 } %1093, 1
  %1095 = extractvalue { i64, i1 } %1093, 0
  br i1 %1094, label %1106, label %1096

1096:                                             ; preds = %1085, %1096
  %1097 = phi i64 [ %1105, %1096 ], [ %1095, %1085 ]
  call void @llvm.amdgcn.s.sleep(i32 1)
  %1098 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %1099 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %1100 = and i64 %1099, %1097
  %1101 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %1098, i64 %1100
  %1102 = load atomic i64, ptr addrspace(1) %1101 syncscope("one-as") monotonic, align 8
  %1103 = cmpxchg ptr addrspace(1) %1086, i64 %1097, i64 %1102 syncscope("one-as") acquire monotonic, align 8
  %1104 = extractvalue { i64, i1 } %1103, 1
  %1105 = extractvalue { i64, i1 } %1103, 0
  br i1 %1104, label %1106, label %1096

1106:                                             ; preds = %1096, %1085, %1080
  %1107 = phi i64 [ 0, %1080 ], [ %1095, %1085 ], [ %1105, %1096 ]
  %1108 = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %1107)
  %1109 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %1110 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %1111 = and i64 %1110, %1108
  %1112 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %1109, i64 %1111
  %1113 = load ptr addrspace(1), ptr addrspace(1) %44, align 8, !tbaa !21
  %1114 = getelementptr inbounds nuw [4096 x i8], ptr addrspace(1) %1113, i64 %1111
  %1115 = call i32 @llvm.amdgcn.ballot.i32(i1 true)
  br i1 %1084, label %1116, label %1120

1116:                                             ; preds = %1106
  %1117 = zext i32 %1115 to i64
  %1118 = getelementptr inbounds nuw i8, ptr addrspace(1) %1112, i64 16
  %1119 = getelementptr inbounds nuw i8, ptr addrspace(1) %1112, i64 8
  store i64 %1117, ptr addrspace(1) %1119, align 8, !tbaa !22
  store <2 x i32> <i32 2, i32 1>, ptr addrspace(1) %1118, align 8, !tbaa !9
  br label %1120

1120:                                             ; preds = %1116, %1106
  %1121 = getelementptr inbounds nuw [64 x i8], ptr addrspace(1) %1114, i64 %53
  store i64 %1082, ptr addrspace(1) %1121, align 8, !tbaa !14
  %1122 = getelementptr inbounds nuw i8, ptr addrspace(1) %1121, i64 8
  store i64 0, ptr addrspace(1) %1122, align 8, !tbaa !14
  %1123 = getelementptr inbounds nuw i8, ptr addrspace(1) %1121, i64 16
  store i64 0, ptr addrspace(1) %1123, align 8, !tbaa !14
  %1124 = getelementptr inbounds nuw i8, ptr addrspace(1) %1121, i64 24
  store i64 0, ptr addrspace(1) %1124, align 8, !tbaa !14
  %1125 = getelementptr inbounds nuw i8, ptr addrspace(1) %1121, i64 32
  store i64 0, ptr addrspace(1) %1125, align 8, !tbaa !14
  %1126 = getelementptr inbounds nuw i8, ptr addrspace(1) %1121, i64 40
  store i64 0, ptr addrspace(1) %1126, align 8, !tbaa !14
  %1127 = getelementptr inbounds nuw i8, ptr addrspace(1) %1121, i64 48
  store i64 0, ptr addrspace(1) %1127, align 8, !tbaa !14
  %1128 = getelementptr inbounds nuw i8, ptr addrspace(1) %1121, i64 56
  store i64 0, ptr addrspace(1) %1128, align 8, !tbaa !14
  br i1 %1084, label %1129, label %1158

1129:                                             ; preds = %1120
  %1130 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 32
  %1131 = load atomic i64, ptr addrspace(1) %1130 syncscope("one-as") monotonic, align 8
  %1132 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %1133 = and i64 %1132, %1108
  %1134 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %1109, i64 %1133
  store i64 %1131, ptr addrspace(1) %1134, align 8, !tbaa !24
  %1135 = cmpxchg ptr addrspace(1) %1130, i64 %1131, i64 %1108 syncscope("one-as") release monotonic, align 8
  %1136 = extractvalue { i64, i1 } %1135, 1
  br i1 %1136, label %1142, label %1137

1137:                                             ; preds = %1129, %1137
  %1138 = phi { i64, i1 } [ %1140, %1137 ], [ %1135, %1129 ]
  %1139 = extractvalue { i64, i1 } %1138, 0
  call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %1139, ptr addrspace(1) %1134, align 8, !tbaa !24
  %1140 = cmpxchg ptr addrspace(1) %1130, i64 %1139, i64 %1108 syncscope("one-as") release monotonic, align 8
  %1141 = extractvalue { i64, i1 } %1140, 1
  br i1 %1141, label %1142, label %1137

1142:                                             ; preds = %1137, %1129
  %1143 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 16
  %1144 = load i64, ptr addrspace(1) %1143, align 8
  %1145 = inttoptr i64 %1144 to ptr addrspace(1)
  %1146 = getelementptr inbounds nuw i8, ptr addrspace(1) %1145, i64 8
  %1147 = atomicrmw add ptr addrspace(1) %1146, i64 1 syncscope("one-as") release, align 8, !amdgpu.no.fine.grained.memory !25, !amdgpu.no.remote.memory !25
  %1148 = getelementptr inbounds nuw i8, ptr addrspace(1) %1145, i64 16
  %1149 = load i64, ptr addrspace(1) %1148, align 16, !tbaa !26
  %1150 = icmp eq i64 %1149, 0
  br i1 %1150, label %1158, label %1151

1151:                                             ; preds = %1142
  %1152 = inttoptr i64 %1149 to ptr addrspace(1)
  %1153 = getelementptr inbounds nuw i8, ptr addrspace(1) %1145, i64 24
  %1154 = load i32, ptr addrspace(1) %1153, align 8, !tbaa !28
  %1155 = zext i32 %1154 to i64
  store atomic i64 %1155, ptr addrspace(1) %1152 syncscope("one-as") release, align 8
  %1156 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1154)
  %1157 = and i32 %1156, 16777215
  call void @llvm.amdgcn.s.sendmsg(i32 1, i32 %1157)
  br label %1158

1158:                                             ; preds = %1151, %1142, %1120
  %1159 = getelementptr inbounds nuw i8, ptr addrspace(1) %1112, i64 20
  br label %1160

1160:                                             ; preds = %1168, %1158
  br i1 %1084, label %1161, label %1164

1161:                                             ; preds = %1160
  %1162 = load atomic i32, ptr addrspace(1) %1159 syncscope("one-as") acquire, align 4
  %1163 = and i32 %1162, 1
  br label %1164

1164:                                             ; preds = %1161, %1160
  %1165 = phi i32 [ %1163, %1161 ], [ 1, %1160 ]
  %1166 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1165)
  %1167 = icmp eq i32 %1166, 0
  br i1 %1167, label %1169, label %1168

1168:                                             ; preds = %1164
  call void @llvm.amdgcn.s.sleep(i32 1)
  br label %1160

1169:                                             ; preds = %1164
  %1170 = load i64, ptr addrspace(1) %1121, align 8, !tbaa !14
  %1171 = load i64, ptr addrspace(1) %1122, align 8, !tbaa !14
  br i1 %1084, label %1172, label %1190

1172:                                             ; preds = %1169
  %1173 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %1174 = add i64 %1173, 1
  %1175 = add i64 %1174, %1108
  %1176 = icmp eq i64 %1175, 0
  %1177 = select i1 %1176, i64 %1174, i64 %1175
  %1178 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 24
  %1179 = load atomic i64, ptr addrspace(1) %1178 syncscope("one-as") monotonic, align 8
  %1180 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %1181 = and i64 %1177, %1173
  %1182 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %1180, i64 %1181
  store i64 %1179, ptr addrspace(1) %1182, align 8, !tbaa !24
  %1183 = cmpxchg ptr addrspace(1) %1178, i64 %1179, i64 %1177 syncscope("one-as") release monotonic, align 8
  %1184 = extractvalue { i64, i1 } %1183, 1
  br i1 %1184, label %1190, label %1185

1185:                                             ; preds = %1172, %1185
  %1186 = phi { i64, i1 } [ %1188, %1185 ], [ %1183, %1172 ]
  %1187 = extractvalue { i64, i1 } %1186, 0
  call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %1187, ptr addrspace(1) %1182, align 8, !tbaa !24
  %1188 = cmpxchg ptr addrspace(1) %1178, i64 %1187, i64 %1177 syncscope("one-as") release monotonic, align 8
  %1189 = extractvalue { i64, i1 } %1188, 1
  br i1 %1189, label %1190, label %1185

1190:                                             ; preds = %1185, %1172, %1169
  %1191 = insertelement <2 x i64> poison, i64 %1170, i64 0
  %1192 = insertelement <2 x i64> %1191, i64 %1171, i64 1
  br label %1493

1193:                                             ; preds = %1078
  %1194 = ptrtoint ptr %1075 to i64
  %1195 = ptrtoint ptr %3 to i64
  %1196 = sub i64 %1194, %1195
  %1197 = shl i64 %1196, 32
  %1198 = ashr exact i64 %1197, 32
  %1199 = and i64 %1053, 2
  %1200 = and i64 %1053, -3
  %1201 = insertelement <2 x i64> <i64 poison, i64 0>, i64 %1200, i64 0
  %1202 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 24
  %1203 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 32
  %1204 = getelementptr inbounds nuw i8, ptr addrspace(1) %10, i64 16
  br label %1205

1205:                                             ; preds = %1487, %1193
  %1206 = phi i64 [ %1198, %1193 ], [ %1490, %1487 ]
  %1207 = phi ptr [ %3, %1193 ], [ %1491, %1487 ]
  %1208 = phi <2 x i64> [ %1201, %1193 ], [ %1489, %1487 ]
  %1209 = icmp ugt i64 %1206, 56
  %1210 = extractelement <2 x i64> %1208, i64 0
  %1211 = call i64 @llvm.umin.i64(i64 %1206, i64 56)
  %1212 = trunc nuw nsw i64 %1211 to i32
  %1213 = select i1 %1209, i64 0, i64 %1199
  %1214 = icmp ugt i64 %1206, 7
  br i1 %1214, label %1217, label %1215

1215:                                             ; preds = %1205
  %1216 = icmp eq i64 %1206, 0
  br i1 %1216, label %1234, label %1221

1217:                                             ; preds = %1205
  %1218 = load i64, ptr %1207, align 1
  %1219 = add nsw i32 %1212, -8
  %1220 = getelementptr inbounds nuw i8, ptr %1207, i64 8
  br label %1234

1221:                                             ; preds = %1215, %1221
  %1222 = phi i32 [ %1232, %1221 ], [ 0, %1215 ]
  %1223 = phi i64 [ %1231, %1221 ], [ 0, %1215 ]
  %1224 = zext nneg i32 %1222 to i64
  %1225 = getelementptr inbounds nuw i8, ptr %1207, i64 %1224
  %1226 = load i8, ptr %1225, align 1, !tbaa !32
  %1227 = zext i8 %1226 to i64
  %1228 = shl i32 %1222, 3
  %1229 = zext nneg i32 %1228 to i64
  %1230 = shl nuw i64 %1227, %1229
  %1231 = or i64 %1230, %1223
  %1232 = add nuw nsw i32 %1222, 1
  %1233 = icmp eq i32 %1232, %1212
  br i1 %1233, label %1234, label %1221

1234:                                             ; preds = %1221, %1217, %1215
  %1235 = phi ptr [ %1220, %1217 ], [ %1207, %1215 ], [ %1207, %1221 ]
  %1236 = phi i32 [ %1219, %1217 ], [ 0, %1215 ], [ 0, %1221 ]
  %1237 = phi i64 [ %1218, %1217 ], [ 0, %1215 ], [ %1231, %1221 ]
  %1238 = icmp ugt i32 %1236, 7
  br i1 %1238, label %1241, label %1239

1239:                                             ; preds = %1234
  %1240 = icmp eq i32 %1236, 0
  br i1 %1240, label %1258, label %1245

1241:                                             ; preds = %1234
  %1242 = load i64, ptr %1235, align 1
  %1243 = add nsw i32 %1236, -8
  %1244 = getelementptr inbounds nuw i8, ptr %1235, i64 8
  br label %1258

1245:                                             ; preds = %1239, %1245
  %1246 = phi i32 [ %1256, %1245 ], [ 0, %1239 ]
  %1247 = phi i64 [ %1255, %1245 ], [ 0, %1239 ]
  %1248 = zext nneg i32 %1246 to i64
  %1249 = getelementptr inbounds nuw i8, ptr %1235, i64 %1248
  %1250 = load i8, ptr %1249, align 1, !tbaa !32
  %1251 = zext i8 %1250 to i64
  %1252 = shl i32 %1246, 3
  %1253 = zext nneg i32 %1252 to i64
  %1254 = shl nuw i64 %1251, %1253
  %1255 = or i64 %1254, %1247
  %1256 = add nuw nsw i32 %1246, 1
  %1257 = icmp eq i32 %1256, %1236
  br i1 %1257, label %1258, label %1245

1258:                                             ; preds = %1245, %1241, %1239
  %1259 = phi ptr [ %1244, %1241 ], [ %1235, %1239 ], [ %1235, %1245 ]
  %1260 = phi i32 [ %1243, %1241 ], [ 0, %1239 ], [ 0, %1245 ]
  %1261 = phi i64 [ %1242, %1241 ], [ 0, %1239 ], [ %1255, %1245 ]
  %1262 = icmp ugt i32 %1260, 7
  br i1 %1262, label %1265, label %1263

1263:                                             ; preds = %1258
  %1264 = icmp eq i32 %1260, 0
  br i1 %1264, label %1282, label %1269

1265:                                             ; preds = %1258
  %1266 = load i64, ptr %1259, align 1
  %1267 = add nsw i32 %1260, -8
  %1268 = getelementptr inbounds nuw i8, ptr %1259, i64 8
  br label %1282

1269:                                             ; preds = %1263, %1269
  %1270 = phi i32 [ %1280, %1269 ], [ 0, %1263 ]
  %1271 = phi i64 [ %1279, %1269 ], [ 0, %1263 ]
  %1272 = zext nneg i32 %1270 to i64
  %1273 = getelementptr inbounds nuw i8, ptr %1259, i64 %1272
  %1274 = load i8, ptr %1273, align 1, !tbaa !32
  %1275 = zext i8 %1274 to i64
  %1276 = shl i32 %1270, 3
  %1277 = zext nneg i32 %1276 to i64
  %1278 = shl nuw i64 %1275, %1277
  %1279 = or i64 %1278, %1271
  %1280 = add nuw nsw i32 %1270, 1
  %1281 = icmp eq i32 %1280, %1260
  br i1 %1281, label %1282, label %1269

1282:                                             ; preds = %1269, %1265, %1263
  %1283 = phi ptr [ %1268, %1265 ], [ %1259, %1263 ], [ %1259, %1269 ]
  %1284 = phi i32 [ %1267, %1265 ], [ 0, %1263 ], [ 0, %1269 ]
  %1285 = phi i64 [ %1266, %1265 ], [ 0, %1263 ], [ %1279, %1269 ]
  %1286 = icmp ugt i32 %1284, 7
  br i1 %1286, label %1289, label %1287

1287:                                             ; preds = %1282
  %1288 = icmp eq i32 %1284, 0
  br i1 %1288, label %1306, label %1293

1289:                                             ; preds = %1282
  %1290 = load i64, ptr %1283, align 1
  %1291 = add nsw i32 %1284, -8
  %1292 = getelementptr inbounds nuw i8, ptr %1283, i64 8
  br label %1306

1293:                                             ; preds = %1287, %1293
  %1294 = phi i32 [ %1304, %1293 ], [ 0, %1287 ]
  %1295 = phi i64 [ %1303, %1293 ], [ 0, %1287 ]
  %1296 = zext nneg i32 %1294 to i64
  %1297 = getelementptr inbounds nuw i8, ptr %1283, i64 %1296
  %1298 = load i8, ptr %1297, align 1, !tbaa !32
  %1299 = zext i8 %1298 to i64
  %1300 = shl i32 %1294, 3
  %1301 = zext nneg i32 %1300 to i64
  %1302 = shl nuw i64 %1299, %1301
  %1303 = or i64 %1302, %1295
  %1304 = add nuw nsw i32 %1294, 1
  %1305 = icmp eq i32 %1304, %1284
  br i1 %1305, label %1306, label %1293

1306:                                             ; preds = %1293, %1289, %1287
  %1307 = phi ptr [ %1292, %1289 ], [ %1283, %1287 ], [ %1283, %1293 ]
  %1308 = phi i32 [ %1291, %1289 ], [ 0, %1287 ], [ 0, %1293 ]
  %1309 = phi i64 [ %1290, %1289 ], [ 0, %1287 ], [ %1303, %1293 ]
  %1310 = icmp ugt i32 %1308, 7
  br i1 %1310, label %1313, label %1311

1311:                                             ; preds = %1306
  %1312 = icmp eq i32 %1308, 0
  br i1 %1312, label %1330, label %1317

1313:                                             ; preds = %1306
  %1314 = load i64, ptr %1307, align 1
  %1315 = add nsw i32 %1308, -8
  %1316 = getelementptr inbounds nuw i8, ptr %1307, i64 8
  br label %1330

1317:                                             ; preds = %1311, %1317
  %1318 = phi i32 [ %1328, %1317 ], [ 0, %1311 ]
  %1319 = phi i64 [ %1327, %1317 ], [ 0, %1311 ]
  %1320 = zext nneg i32 %1318 to i64
  %1321 = getelementptr inbounds nuw i8, ptr %1307, i64 %1320
  %1322 = load i8, ptr %1321, align 1, !tbaa !32
  %1323 = zext i8 %1322 to i64
  %1324 = shl i32 %1318, 3
  %1325 = zext nneg i32 %1324 to i64
  %1326 = shl nuw i64 %1323, %1325
  %1327 = or i64 %1326, %1319
  %1328 = add nuw nsw i32 %1318, 1
  %1329 = icmp eq i32 %1328, %1308
  br i1 %1329, label %1330, label %1317

1330:                                             ; preds = %1317, %1313, %1311
  %1331 = phi ptr [ %1316, %1313 ], [ %1307, %1311 ], [ %1307, %1317 ]
  %1332 = phi i32 [ %1315, %1313 ], [ 0, %1311 ], [ 0, %1317 ]
  %1333 = phi i64 [ %1314, %1313 ], [ 0, %1311 ], [ %1327, %1317 ]
  %1334 = icmp ugt i32 %1332, 7
  br i1 %1334, label %1337, label %1335

1335:                                             ; preds = %1330
  %1336 = icmp eq i32 %1332, 0
  br i1 %1336, label %1354, label %1341

1337:                                             ; preds = %1330
  %1338 = load i64, ptr %1331, align 1
  %1339 = add nsw i32 %1332, -8
  %1340 = getelementptr inbounds nuw i8, ptr %1331, i64 8
  br label %1354

1341:                                             ; preds = %1335, %1341
  %1342 = phi i32 [ %1352, %1341 ], [ 0, %1335 ]
  %1343 = phi i64 [ %1351, %1341 ], [ 0, %1335 ]
  %1344 = zext nneg i32 %1342 to i64
  %1345 = getelementptr inbounds nuw i8, ptr %1331, i64 %1344
  %1346 = load i8, ptr %1345, align 1, !tbaa !32
  %1347 = zext i8 %1346 to i64
  %1348 = shl i32 %1342, 3
  %1349 = zext nneg i32 %1348 to i64
  %1350 = shl nuw i64 %1347, %1349
  %1351 = or i64 %1350, %1343
  %1352 = add nuw nsw i32 %1342, 1
  %1353 = icmp eq i32 %1352, %1332
  br i1 %1353, label %1354, label %1341

1354:                                             ; preds = %1341, %1337, %1335
  %1355 = phi ptr [ %1340, %1337 ], [ %1331, %1335 ], [ %1331, %1341 ]
  %1356 = phi i32 [ %1339, %1337 ], [ 0, %1335 ], [ 0, %1341 ]
  %1357 = phi i64 [ %1338, %1337 ], [ 0, %1335 ], [ %1351, %1341 ]
  %1358 = icmp ugt i32 %1356, 7
  br i1 %1358, label %1361, label %1359

1359:                                             ; preds = %1354
  %1360 = icmp eq i32 %1356, 0
  br i1 %1360, label %1376, label %1363

1361:                                             ; preds = %1354
  %1362 = load i64, ptr %1355, align 1
  br label %1376

1363:                                             ; preds = %1359, %1363
  %1364 = phi i32 [ %1374, %1363 ], [ 0, %1359 ]
  %1365 = phi i64 [ %1373, %1363 ], [ 0, %1359 ]
  %1366 = zext nneg i32 %1364 to i64
  %1367 = getelementptr inbounds nuw i8, ptr %1355, i64 %1366
  %1368 = load i8, ptr %1367, align 1, !tbaa !32
  %1369 = zext i8 %1368 to i64
  %1370 = shl i32 %1364, 3
  %1371 = zext nneg i32 %1370 to i64
  %1372 = shl nuw i64 %1369, %1371
  %1373 = or i64 %1372, %1365
  %1374 = add nuw nsw i32 %1364, 1
  %1375 = icmp eq i32 %1374, %1356
  br i1 %1375, label %1376, label %1363

1376:                                             ; preds = %1363, %1361, %1359
  %1377 = phi i64 [ %1362, %1361 ], [ 0, %1359 ], [ %1373, %1363 ]
  %1378 = shl nuw nsw i64 %1211, 2
  %1379 = add nuw nsw i64 %1378, 28
  %1380 = and i64 %1379, 480
  %1381 = and i64 %1210, -225
  %1382 = or i64 %1381, %1213
  %1383 = or i64 %1382, %1380
  %1384 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %11)
  %1385 = icmp eq i32 %11, %1384
  br i1 %1385, label %1386, label %1406

1386:                                             ; preds = %1376
  %1387 = load atomic i64, ptr addrspace(1) %1202 syncscope("one-as") acquire, align 8
  %1388 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %1389 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %1390 = and i64 %1389, %1387
  %1391 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %1388, i64 %1390
  %1392 = load atomic i64, ptr addrspace(1) %1391 syncscope("one-as") monotonic, align 8
  %1393 = cmpxchg ptr addrspace(1) %1202, i64 %1387, i64 %1392 syncscope("one-as") acquire monotonic, align 8
  %1394 = extractvalue { i64, i1 } %1393, 1
  %1395 = extractvalue { i64, i1 } %1393, 0
  br i1 %1394, label %1406, label %1396

1396:                                             ; preds = %1386, %1396
  %1397 = phi i64 [ %1405, %1396 ], [ %1395, %1386 ]
  call void @llvm.amdgcn.s.sleep(i32 1)
  %1398 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %1399 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %1400 = and i64 %1399, %1397
  %1401 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %1398, i64 %1400
  %1402 = load atomic i64, ptr addrspace(1) %1401 syncscope("one-as") monotonic, align 8
  %1403 = cmpxchg ptr addrspace(1) %1202, i64 %1397, i64 %1402 syncscope("one-as") acquire monotonic, align 8
  %1404 = extractvalue { i64, i1 } %1403, 1
  %1405 = extractvalue { i64, i1 } %1403, 0
  br i1 %1404, label %1406, label %1396

1406:                                             ; preds = %1396, %1386, %1376
  %1407 = phi i64 [ 0, %1376 ], [ %1395, %1386 ], [ %1405, %1396 ]
  %1408 = call i64 @llvm.amdgcn.readfirstlane.i64(i64 %1407)
  %1409 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %1410 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %1411 = and i64 %1410, %1408
  %1412 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %1409, i64 %1411
  %1413 = load ptr addrspace(1), ptr addrspace(1) %44, align 8, !tbaa !21
  %1414 = getelementptr inbounds nuw [4096 x i8], ptr addrspace(1) %1413, i64 %1411
  %1415 = call i32 @llvm.amdgcn.ballot.i32(i1 true)
  br i1 %1385, label %1416, label %1420

1416:                                             ; preds = %1406
  %1417 = zext i32 %1415 to i64
  %1418 = getelementptr inbounds nuw i8, ptr addrspace(1) %1412, i64 16
  %1419 = getelementptr inbounds nuw i8, ptr addrspace(1) %1412, i64 8
  store i64 %1417, ptr addrspace(1) %1419, align 8, !tbaa !22
  store <2 x i32> <i32 2, i32 1>, ptr addrspace(1) %1418, align 8, !tbaa !9
  br label %1420

1420:                                             ; preds = %1416, %1406
  %1421 = getelementptr inbounds nuw [64 x i8], ptr addrspace(1) %1414, i64 %53
  store i64 %1383, ptr addrspace(1) %1421, align 8, !tbaa !14
  %1422 = getelementptr inbounds nuw i8, ptr addrspace(1) %1421, i64 8
  store i64 %1237, ptr addrspace(1) %1422, align 8, !tbaa !14
  %1423 = getelementptr inbounds nuw i8, ptr addrspace(1) %1421, i64 16
  store i64 %1261, ptr addrspace(1) %1423, align 8, !tbaa !14
  %1424 = getelementptr inbounds nuw i8, ptr addrspace(1) %1421, i64 24
  store i64 %1285, ptr addrspace(1) %1424, align 8, !tbaa !14
  %1425 = getelementptr inbounds nuw i8, ptr addrspace(1) %1421, i64 32
  store i64 %1309, ptr addrspace(1) %1425, align 8, !tbaa !14
  %1426 = getelementptr inbounds nuw i8, ptr addrspace(1) %1421, i64 40
  store i64 %1333, ptr addrspace(1) %1426, align 8, !tbaa !14
  %1427 = getelementptr inbounds nuw i8, ptr addrspace(1) %1421, i64 48
  store i64 %1357, ptr addrspace(1) %1427, align 8, !tbaa !14
  %1428 = getelementptr inbounds nuw i8, ptr addrspace(1) %1421, i64 56
  store i64 %1377, ptr addrspace(1) %1428, align 8, !tbaa !14
  br i1 %1385, label %1429, label %1456

1429:                                             ; preds = %1420
  %1430 = load atomic i64, ptr addrspace(1) %1203 syncscope("one-as") monotonic, align 8
  %1431 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %1432 = and i64 %1431, %1408
  %1433 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %1409, i64 %1432
  store i64 %1430, ptr addrspace(1) %1433, align 8, !tbaa !24
  %1434 = cmpxchg ptr addrspace(1) %1203, i64 %1430, i64 %1408 syncscope("one-as") release monotonic, align 8
  %1435 = extractvalue { i64, i1 } %1434, 1
  br i1 %1435, label %1441, label %1436

1436:                                             ; preds = %1429, %1436
  %1437 = phi { i64, i1 } [ %1439, %1436 ], [ %1434, %1429 ]
  %1438 = extractvalue { i64, i1 } %1437, 0
  call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %1438, ptr addrspace(1) %1433, align 8, !tbaa !24
  %1439 = cmpxchg ptr addrspace(1) %1203, i64 %1438, i64 %1408 syncscope("one-as") release monotonic, align 8
  %1440 = extractvalue { i64, i1 } %1439, 1
  br i1 %1440, label %1441, label %1436

1441:                                             ; preds = %1436, %1429
  %1442 = load i64, ptr addrspace(1) %1204, align 8
  %1443 = inttoptr i64 %1442 to ptr addrspace(1)
  %1444 = getelementptr inbounds nuw i8, ptr addrspace(1) %1443, i64 8
  %1445 = atomicrmw add ptr addrspace(1) %1444, i64 1 syncscope("one-as") release, align 8, !amdgpu.no.fine.grained.memory !25, !amdgpu.no.remote.memory !25
  %1446 = getelementptr inbounds nuw i8, ptr addrspace(1) %1443, i64 16
  %1447 = load i64, ptr addrspace(1) %1446, align 16, !tbaa !26
  %1448 = icmp eq i64 %1447, 0
  br i1 %1448, label %1456, label %1449

1449:                                             ; preds = %1441
  %1450 = inttoptr i64 %1447 to ptr addrspace(1)
  %1451 = getelementptr inbounds nuw i8, ptr addrspace(1) %1443, i64 24
  %1452 = load i32, ptr addrspace(1) %1451, align 8, !tbaa !28
  %1453 = zext i32 %1452 to i64
  store atomic i64 %1453, ptr addrspace(1) %1450 syncscope("one-as") release, align 8
  %1454 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1452)
  %1455 = and i32 %1454, 16777215
  call void @llvm.amdgcn.s.sendmsg(i32 1, i32 %1455)
  br label %1456

1456:                                             ; preds = %1449, %1441, %1420
  %1457 = getelementptr inbounds nuw i8, ptr addrspace(1) %1412, i64 20
  br label %1458

1458:                                             ; preds = %1466, %1456
  br i1 %1385, label %1459, label %1462

1459:                                             ; preds = %1458
  %1460 = load atomic i32, ptr addrspace(1) %1457 syncscope("one-as") acquire, align 4
  %1461 = and i32 %1460, 1
  br label %1462

1462:                                             ; preds = %1459, %1458
  %1463 = phi i32 [ %1461, %1459 ], [ 1, %1458 ]
  %1464 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1463)
  %1465 = icmp eq i32 %1464, 0
  br i1 %1465, label %1467, label %1466

1466:                                             ; preds = %1462
  call void @llvm.amdgcn.s.sleep(i32 1)
  br label %1458

1467:                                             ; preds = %1462
  %1468 = load i64, ptr addrspace(1) %1421, align 8, !tbaa !14
  %1469 = load i64, ptr addrspace(1) %1422, align 8, !tbaa !14
  br i1 %1385, label %1470, label %1487

1470:                                             ; preds = %1467
  %1471 = load i64, ptr addrspace(1) %40, align 8, !tbaa !20
  %1472 = add i64 %1471, 1
  %1473 = add i64 %1472, %1408
  %1474 = icmp eq i64 %1473, 0
  %1475 = select i1 %1474, i64 %1472, i64 %1473
  %1476 = load atomic i64, ptr addrspace(1) %1202 syncscope("one-as") monotonic, align 8
  %1477 = load ptr addrspace(1), ptr addrspace(1) %10, align 8, !tbaa !16
  %1478 = and i64 %1475, %1471
  %1479 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %1477, i64 %1478
  store i64 %1476, ptr addrspace(1) %1479, align 8, !tbaa !24
  %1480 = cmpxchg ptr addrspace(1) %1202, i64 %1476, i64 %1475 syncscope("one-as") release monotonic, align 8
  %1481 = extractvalue { i64, i1 } %1480, 1
  br i1 %1481, label %1487, label %1482

1482:                                             ; preds = %1470, %1482
  %1483 = phi { i64, i1 } [ %1485, %1482 ], [ %1480, %1470 ]
  %1484 = extractvalue { i64, i1 } %1483, 0
  call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %1484, ptr addrspace(1) %1479, align 8, !tbaa !24
  %1485 = cmpxchg ptr addrspace(1) %1202, i64 %1484, i64 %1475 syncscope("one-as") release monotonic, align 8
  %1486 = extractvalue { i64, i1 } %1485, 1
  br i1 %1486, label %1487, label %1482

1487:                                             ; preds = %1482, %1470, %1467
  %1488 = insertelement <2 x i64> poison, i64 %1468, i64 0
  %1489 = insertelement <2 x i64> %1488, i64 %1469, i64 1
  %1490 = sub i64 %1206, %1211
  %1491 = getelementptr inbounds nuw i8, ptr %1207, i64 %1211
  %1492 = icmp eq i64 %1490, 0
  br i1 %1492, label %1493, label %1205

1493:                                             ; preds = %1487, %1190
  %1494 = phi <2 x i64> [ %1192, %1190 ], [ %1489, %1487 ]
  br label %1495

1495:                                             ; preds = %1495, %1493
  %1496 = phi ptr [ %0, %1493 ], [ %1497, %1495 ]
  %1497 = getelementptr inbounds nuw i8, ptr %1496, i64 1
  %1498 = load i8, ptr %1496, align 1, !tbaa !29
  %1499 = icmp eq i8 %1498, 0
  br i1 %1499, label %1500, label %1495, !llvm.loop !35

1500:                                             ; preds = %1495
  %1501 = extractelement <2 x i64> %1494, i64 0
  %1502 = ptrtoint ptr %1497 to i64
  %1503 = ptrtoint ptr %0 to i64
  %1504 = sub i64 %1502, %1503
  %1505 = shl i64 %1504, 32
  %1506 = ashr exact i64 %1505, 32
  call fastcc void @__ockl_fprintf_append_string_n(i64 noundef %1501, ptr noundef %0, i64 noundef %1506, i32 noundef 1) #24
  call void @llvm.trap()
  unreachable
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p5(ptr addrspace(5) captures(none)) #3

; Function Attrs: mustprogress noinline nounwind uwtable
define weak hidden void @__assertfail() local_unnamed_addr #4 {
  tail call void @llvm.trap()
  unreachable
}

; Function Attrs: convergent mustprogress norecurse nounwind uwtable
define weak_odr amdgpu_kernel void @_ZN7ck_tileL11flush_cacheEv.intern.2b4ba03da513173f() local_unnamed_addr #5 comdat {
  tail call void asm sideeffect "s_icache_inv \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09", ""() #25, !srcloc !36
  ret void
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(none) uwtable
define hidden noundef float @_ZN5aiter15cross_row_sum_4Efi(float noundef %0, i32 noundef %1) local_unnamed_addr #6 {
  %3 = bitcast float %0 to i32
  %4 = shl i32 %1, 2
  %5 = xor i32 %4, 64
  %6 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %5, i32 %3)
  %7 = bitcast i32 %6 to float
  %8 = fadd contract float %0, %7
  ret float %8
}

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #7

; Function Attrs: alwaysinline convergent mustprogress nounwind uwtable
define hidden void @mhc_native_pre_gemm_shuffled(i64 noundef %0, i64 noundef %1, i64 noundef %2, i64 noundef %3, i32 noundef %4, i32 noundef %5, i32 noundef %6) #8 {
  %8 = inttoptr i64 %1 to ptr
  %9 = inttoptr i64 %2 to ptr
  %10 = inttoptr i64 %3 to ptr
  %11 = shl i32 %4, 6
  %12 = zext i32 %11 to i64
  %13 = shl i32 %5, 4
  %14 = shl nsw i32 %6, 7
  %15 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %16 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %15)
  %17 = lshr i32 %16, 5
  %18 = and i32 %15, 31
  %19 = shl i32 %5, 18
  %20 = sext i32 %19 to i64
  %21 = getelementptr inbounds [4 x i8], ptr %10, i64 %20
  %22 = shl nsw i32 %6, 4
  %23 = sext i32 %22 to i64
  %24 = sub i32 16, %11
  %25 = icmp sgt i32 %13, 8
  %26 = sub nsw i32 24, %13
  %27 = select i1 %25, i32 %26, i32 16
  %28 = shl nuw nsw i32 %17, 4
  %29 = add nsw i32 %18, -16
  %30 = icmp samesign ult i32 %18, 16
  %31 = select i1 %30, i32 %18, i32 %29
  %32 = add nuw nsw i32 %28, %31
  %33 = lshr i32 %15, 1
  %34 = and i32 %33, 8
  %35 = shl nuw nsw i32 %17, 2
  %36 = icmp eq i32 %17, 0
  br i1 %36, label %37, label %71

37:                                               ; preds = %7
  %38 = srem i32 %14, 4096
  %39 = trunc nsw i32 %38 to i16
  %40 = ashr exact i16 %39, 6
  %41 = sext i16 %40 to i32
  %42 = sdiv i32 %6, 32
  %43 = shl nsw i32 %41, 6
  %44 = shl nsw i32 %42, 4
  %45 = add nsw i32 %43, %44
  %46 = sext i32 %45 to i64
  %47 = add nsw i64 %46, %12
  %48 = shl nsw i64 %47, 7
  %49 = getelementptr inbounds i8, ptr %9, i64 %48
  %50 = tail call ptr @llvm.amdgcn.readfirstlane.p0(ptr %49)
  %51 = ptrtoint ptr %50 to i64
  %52 = tail call noundef i32 @llvm.amdgcn.readfirstlane.i32(i32 %24)
  %53 = tail call i32 asm "s_sub_co_u32 $0, $1, $2", "=s,s,s,~{scc}"(i32 %52, i32 0) #26, !srcloc !37
  %54 = tail call noundef i32 asm "s_max_i32 $0, $1, 0", "=s,s,~{scc}"(i32 %53) #26, !srcloc !38
  %55 = trunc i64 %51 to i32
  %56 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspacecast (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 8192) to ptr) to i32), i32 poison, i32 poison>, i32 %55, i64 2
  %57 = lshr i64 %51, 32
  %58 = trunc nuw i64 %57 to i32
  %59 = and i32 %58, 33554431
  %60 = or disjoint i32 %59, -2147483648
  %61 = insertelement <4 x i32> %56, i32 %60, i64 3
  %62 = tail call i32 asm "s_sub_co_u32 $0, $1, $2", "=s,s,s,~{scc}"(i32 64, i32 0) #26, !srcloc !37
  %63 = tail call noundef i32 asm "s_max_i32 $0, $1, 0", "=s,s,~{scc}"(i32 %62) #26, !srcloc !38
  %64 = shl i32 %63, 16
  %65 = tail call i32 @llvm.fshl.i32(i32 %54, i32 %63, i32 16)
  %66 = lshr i32 %54, 16
  %67 = or disjoint i32 %66, 4194304
  %68 = insertelement <8 x i32> <i32 118554624, i32 poison, i32 poison, i32 poison, i32 64, i32 64, i32 0, i32 0>, i32 %64, i64 1
  %69 = insertelement <8 x i32> %68, i32 %65, i64 2
  %70 = insertelement <8 x i32> %69, i32 %67, i64 3
  tail call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %61, <8 x i32> %70, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 16)
  br label %71

71:                                               ; preds = %37, %7
  %72 = shl nsw i32 %17, 10
  %73 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 0), i32 %72
  %74 = shl nuw nsw i32 %15, 2
  %75 = and i32 %74, 60
  %76 = lshr i32 %18, 4
  %77 = or disjoint i32 %35, %76
  %78 = icmp slt i32 %77, %27
  br i1 %78, label %84, label %79

79:                                               ; preds = %71
  %80 = shl nuw nsw i32 %76, 8
  %81 = getelementptr inbounds nuw i8, ptr addrspace(3) %73, i32 %80
  %82 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %81, i32 %75
  store <2 x float> zeroinitializer, ptr addrspace(3) %82, align 16, !tbaa !39
  %83 = getelementptr inbounds nuw i8, ptr addrspace(3) %82, i32 8
  store <2 x float> zeroinitializer, ptr addrspace(3) %83, align 8, !tbaa !39
  br label %103

84:                                               ; preds = %71
  %85 = xor i32 %77, %15
  %86 = shl nuw nsw i32 %85, 2
  %87 = and i32 %86, 60
  %88 = shl nuw nsw i32 %76, 8
  %89 = getelementptr inbounds nuw i8, ptr addrspace(3) %73, i32 %88
  %90 = addrspacecast ptr addrspace(3) %89 to ptr
  %91 = zext nneg i32 %75 to i64
  %92 = getelementptr inbounds nuw [4 x i8], ptr %90, i64 %91
  %93 = shl nsw i32 %77, 14
  %94 = add i32 %93, %14
  %95 = or disjoint i32 %87, %94
  %96 = ptrtoint ptr %92 to i64
  %97 = trunc i64 %96 to i32
  %98 = inttoptr i32 %97 to ptr addrspace(3)
  %99 = shl i32 %95, 2
  %100 = sext i32 %99 to i64
  %101 = getelementptr inbounds i8, ptr %21, i64 %100
  %102 = addrspacecast ptr %101 to ptr addrspace(1)
  tail call void @llvm.amdgcn.global.load.async.to.lds.b128(ptr addrspace(1) %102, ptr addrspace(3) %98, i32 0, i32 0)
  br label %103

103:                                              ; preds = %84, %79
  %104 = or disjoint i32 %76, 2
  %105 = or disjoint i32 %35, %104
  %106 = icmp slt i32 %105, %27
  br i1 %106, label %112, label %107

107:                                              ; preds = %103
  %108 = shl nuw nsw i32 %104, 8
  %109 = getelementptr inbounds nuw i8, ptr addrspace(3) %73, i32 %108
  %110 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %109, i32 %75
  store <2 x float> zeroinitializer, ptr addrspace(3) %110, align 16, !tbaa !39
  %111 = getelementptr inbounds nuw i8, ptr addrspace(3) %110, i32 8
  store <2 x float> zeroinitializer, ptr addrspace(3) %111, align 8, !tbaa !39
  br label %131

112:                                              ; preds = %103
  %113 = xor i32 %105, %15
  %114 = shl nuw nsw i32 %113, 2
  %115 = and i32 %114, 60
  %116 = shl nuw nsw i32 %104, 8
  %117 = getelementptr inbounds nuw i8, ptr addrspace(3) %73, i32 %116
  %118 = addrspacecast ptr addrspace(3) %117 to ptr
  %119 = zext nneg i32 %75 to i64
  %120 = getelementptr inbounds nuw [4 x i8], ptr %118, i64 %119
  %121 = shl nsw i32 %105, 14
  %122 = add i32 %121, %14
  %123 = or disjoint i32 %115, %122
  %124 = ptrtoint ptr %120 to i64
  %125 = trunc i64 %124 to i32
  %126 = inttoptr i32 %125 to ptr addrspace(3)
  %127 = shl i32 %123, 2
  %128 = sext i32 %127 to i64
  %129 = getelementptr inbounds i8, ptr %21, i64 %128
  %130 = addrspacecast ptr %129 to ptr addrspace(1)
  tail call void @llvm.amdgcn.global.load.async.to.lds.b128(ptr addrspace(1) %130, ptr addrspace(3) %126, i32 0, i32 0)
  br label %131

131:                                              ; preds = %112, %107
  %132 = or disjoint i32 %14, 64
  br i1 %36, label %133, label %168

133:                                              ; preds = %131
  %134 = sdiv i32 %132, 4096
  %135 = mul i32 %134, 4096
  %136 = sub i32 %132, %135
  %137 = trunc nsw i32 %136 to i16
  %138 = ashr exact i16 %137, 6
  %139 = sext i16 %138 to i32
  %140 = shl nsw i32 %139, 6
  %141 = shl nsw i32 %134, 4
  %142 = add nsw i32 %140, %141
  %143 = sext i32 %142 to i64
  %144 = add nsw i64 %143, %12
  %145 = shl nsw i64 %144, 7
  %146 = getelementptr inbounds i8, ptr %9, i64 %145
  %147 = tail call ptr @llvm.amdgcn.readfirstlane.p0(ptr %146)
  %148 = ptrtoint ptr %147 to i64
  %149 = tail call noundef i32 @llvm.amdgcn.readfirstlane.i32(i32 %24)
  %150 = tail call i32 asm "s_sub_co_u32 $0, $1, $2", "=s,s,s,~{scc}"(i32 %149, i32 0) #26, !srcloc !37
  %151 = tail call noundef i32 asm "s_max_i32 $0, $1, 0", "=s,s,~{scc}"(i32 %150) #26, !srcloc !38
  %152 = trunc i64 %148 to i32
  %153 = insertelement <4 x i32> <i32 1, i32 add (i32 ptrtoint (ptr addrspacecast (ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 8192) to ptr) to i32), i32 9216), i32 poison, i32 poison>, i32 %152, i64 2
  %154 = lshr i64 %148, 32
  %155 = trunc nuw i64 %154 to i32
  %156 = and i32 %155, 33554431
  %157 = or disjoint i32 %156, -2147483648
  %158 = insertelement <4 x i32> %153, i32 %157, i64 3
  %159 = tail call i32 asm "s_sub_co_u32 $0, $1, $2", "=s,s,s,~{scc}"(i32 64, i32 0) #26, !srcloc !37
  %160 = tail call noundef i32 asm "s_max_i32 $0, $1, 0", "=s,s,~{scc}"(i32 %159) #26, !srcloc !38
  %161 = shl i32 %160, 16
  %162 = tail call i32 @llvm.fshl.i32(i32 %151, i32 %160, i32 16)
  %163 = lshr i32 %151, 16
  %164 = or disjoint i32 %163, 4194304
  %165 = insertelement <8 x i32> <i32 118554624, i32 poison, i32 poison, i32 poison, i32 64, i32 64, i32 0, i32 0>, i32 %161, i64 1
  %166 = insertelement <8 x i32> %165, i32 %162, i64 2
  %167 = insertelement <8 x i32> %166, i32 %164, i64 3
  tail call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %158, <8 x i32> %167, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 16)
  br label %168

168:                                              ; preds = %133, %131
  %169 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 0), i32 4096), i32 %72
  br i1 %78, label %175, label %170

170:                                              ; preds = %168
  %171 = shl nuw nsw i32 %76, 8
  %172 = getelementptr inbounds nuw i8, ptr addrspace(3) %169, i32 %171
  %173 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %172, i32 %75
  store <2 x float> zeroinitializer, ptr addrspace(3) %173, align 16, !tbaa !39
  %174 = getelementptr inbounds nuw i8, ptr addrspace(3) %173, i32 8
  store <2 x float> zeroinitializer, ptr addrspace(3) %174, align 8, !tbaa !39
  br label %194

175:                                              ; preds = %168
  %176 = xor i32 %77, %15
  %177 = shl nuw nsw i32 %176, 2
  %178 = and i32 %177, 60
  %179 = shl nuw nsw i32 %76, 8
  %180 = getelementptr inbounds nuw i8, ptr addrspace(3) %169, i32 %179
  %181 = addrspacecast ptr addrspace(3) %180 to ptr
  %182 = zext nneg i32 %75 to i64
  %183 = getelementptr inbounds nuw [4 x i8], ptr %181, i64 %182
  %184 = shl nsw i32 %77, 14
  %185 = add i32 %184, %132
  %186 = or disjoint i32 %178, %185
  %187 = ptrtoint ptr %183 to i64
  %188 = trunc i64 %187 to i32
  %189 = inttoptr i32 %188 to ptr addrspace(3)
  %190 = shl i32 %186, 2
  %191 = sext i32 %190 to i64
  %192 = getelementptr inbounds i8, ptr %21, i64 %191
  %193 = addrspacecast ptr %192 to ptr addrspace(1)
  tail call void @llvm.amdgcn.global.load.async.to.lds.b128(ptr addrspace(1) %193, ptr addrspace(3) %189, i32 0, i32 0)
  br label %194

194:                                              ; preds = %175, %170
  br i1 %106, label %200, label %195

195:                                              ; preds = %194
  %196 = shl nuw nsw i32 %104, 8
  %197 = getelementptr inbounds nuw i8, ptr addrspace(3) %169, i32 %196
  %198 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %197, i32 %75
  store <2 x float> zeroinitializer, ptr addrspace(3) %198, align 16, !tbaa !39
  %199 = getelementptr inbounds nuw i8, ptr addrspace(3) %198, i32 8
  store <2 x float> zeroinitializer, ptr addrspace(3) %199, align 8, !tbaa !39
  br label %219

200:                                              ; preds = %194
  %201 = xor i32 %105, %15
  %202 = shl nuw nsw i32 %201, 2
  %203 = and i32 %202, 60
  %204 = shl nuw nsw i32 %104, 8
  %205 = getelementptr inbounds nuw i8, ptr addrspace(3) %169, i32 %204
  %206 = addrspacecast ptr addrspace(3) %205 to ptr
  %207 = zext nneg i32 %75 to i64
  %208 = getelementptr inbounds nuw [4 x i8], ptr %206, i64 %207
  %209 = shl nsw i32 %105, 14
  %210 = add i32 %209, %132
  %211 = or disjoint i32 %203, %210
  %212 = ptrtoint ptr %208 to i64
  %213 = trunc i64 %212 to i32
  %214 = inttoptr i32 %213 to ptr addrspace(3)
  %215 = shl i32 %211, 2
  %216 = sext i32 %215 to i64
  %217 = getelementptr inbounds i8, ptr %21, i64 %216
  %218 = addrspacecast ptr %217 to ptr addrspace(1)
  tail call void @llvm.amdgcn.global.load.async.to.lds.b128(ptr addrspace(1) %218, ptr addrspace(3) %214, i32 0, i32 0)
  br label %219

219:                                              ; preds = %200, %195
  tail call void @llvm.amdgcn.s.wait.asynccnt(i16 noundef signext 4) #24
  tail call void @llvm.amdgcn.s.wait.asynccnt(i16 noundef signext 0) #24
  tail call void @llvm.amdgcn.s.wait.tensorcnt(i16 noundef signext 1) #24
  tail call void @mhc_pre_subgroup_barrier()
  %220 = mul nsw i32 %32, 144
  %221 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 8192), i32 %220
  %222 = and i32 %15, 16
  %223 = getelementptr inbounds nuw i8, ptr addrspace(3) %221, i32 %222
  %224 = load <8 x bfloat>, ptr addrspace(3) %223, align 16, !tbaa !29
  %225 = getelementptr inbounds nuw i8, ptr addrspace(3) %223, i32 32
  %226 = load <8 x bfloat>, ptr addrspace(3) %225, align 16, !tbaa !29
  %227 = getelementptr inbounds nuw i8, ptr addrspace(3) %223, i32 64
  %228 = load <8 x bfloat>, ptr addrspace(3) %227, align 16, !tbaa !29
  %229 = getelementptr inbounds nuw i8, ptr addrspace(3) %223, i32 96
  %230 = load <8 x bfloat>, ptr addrspace(3) %229, align 16, !tbaa !29
  tail call void @llvm.amdgcn.s.wait.dscnt(i16 noundef signext 0) #24
  %231 = icmp eq i32 %13, 0
  %232 = extractelement <8 x bfloat> %224, i64 0
  br i1 %231, label %250, label %233

233:                                              ; preds = %219
  %234 = extractelement <8 x bfloat> %224, i64 1
  %235 = shufflevector <8 x bfloat> %224, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %236 = shufflevector <8 x bfloat> %224, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %237 = shufflevector <8 x bfloat> %224, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %238 = shufflevector <8 x bfloat> %226, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %239 = shufflevector <8 x bfloat> %226, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %240 = shufflevector <8 x bfloat> %226, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %241 = shufflevector <8 x bfloat> %226, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %242 = shufflevector <8 x bfloat> %228, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %243 = shufflevector <8 x bfloat> %228, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %244 = shufflevector <8 x bfloat> %228, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %245 = shufflevector <8 x bfloat> %228, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %246 = shufflevector <8 x bfloat> %230, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %247 = shufflevector <8 x bfloat> %230, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %248 = shufflevector <8 x bfloat> %230, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %249 = shufflevector <8 x bfloat> %230, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  br label %380

250:                                              ; preds = %219
  %251 = fpext contract bfloat %232 to float
  %252 = fmul contract float %251, %251
  %253 = extractelement <8 x bfloat> %224, i64 1
  %254 = fpext contract bfloat %253 to float
  %255 = fmul contract float %254, %254
  %256 = fadd contract float %252, %255
  %257 = extractelement <8 x bfloat> %224, i64 2
  %258 = fpext contract bfloat %257 to float
  %259 = fmul contract float %258, %258
  %260 = fadd contract float %259, %256
  %261 = extractelement <8 x bfloat> %224, i64 3
  %262 = fpext contract bfloat %261 to float
  %263 = fmul contract float %262, %262
  %264 = fadd contract float %263, %260
  %265 = extractelement <8 x bfloat> %224, i64 4
  %266 = fpext contract bfloat %265 to float
  %267 = fmul contract float %266, %266
  %268 = fadd contract float %267, %264
  %269 = extractelement <8 x bfloat> %224, i64 5
  %270 = fpext contract bfloat %269 to float
  %271 = fmul contract float %270, %270
  %272 = fadd contract float %271, %268
  %273 = extractelement <8 x bfloat> %224, i64 6
  %274 = fpext contract bfloat %273 to float
  %275 = fmul contract float %274, %274
  %276 = fadd contract float %275, %272
  %277 = extractelement <8 x bfloat> %224, i64 7
  %278 = fpext contract bfloat %277 to float
  %279 = fmul contract float %278, %278
  %280 = fadd contract float %279, %276
  %281 = extractelement <8 x bfloat> %226, i64 0
  %282 = fpext contract bfloat %281 to float
  %283 = fmul contract float %282, %282
  %284 = fadd contract float %283, %280
  %285 = extractelement <8 x bfloat> %226, i64 1
  %286 = fpext contract bfloat %285 to float
  %287 = fmul contract float %286, %286
  %288 = fadd contract float %287, %284
  %289 = extractelement <8 x bfloat> %226, i64 2
  %290 = fpext contract bfloat %289 to float
  %291 = fmul contract float %290, %290
  %292 = fadd contract float %291, %288
  %293 = extractelement <8 x bfloat> %226, i64 3
  %294 = fpext contract bfloat %293 to float
  %295 = fmul contract float %294, %294
  %296 = fadd contract float %295, %292
  %297 = extractelement <8 x bfloat> %226, i64 4
  %298 = fpext contract bfloat %297 to float
  %299 = fmul contract float %298, %298
  %300 = fadd contract float %299, %296
  %301 = extractelement <8 x bfloat> %226, i64 5
  %302 = fpext contract bfloat %301 to float
  %303 = fmul contract float %302, %302
  %304 = fadd contract float %303, %300
  %305 = extractelement <8 x bfloat> %226, i64 6
  %306 = fpext contract bfloat %305 to float
  %307 = fmul contract float %306, %306
  %308 = fadd contract float %307, %304
  %309 = extractelement <8 x bfloat> %226, i64 7
  %310 = fpext contract bfloat %309 to float
  %311 = fmul contract float %310, %310
  %312 = fadd contract float %311, %308
  %313 = extractelement <8 x bfloat> %228, i64 0
  %314 = fpext contract bfloat %313 to float
  %315 = fmul contract float %314, %314
  %316 = fadd contract float %315, %312
  %317 = extractelement <8 x bfloat> %228, i64 1
  %318 = fpext contract bfloat %317 to float
  %319 = fmul contract float %318, %318
  %320 = fadd contract float %319, %316
  %321 = extractelement <8 x bfloat> %228, i64 2
  %322 = fpext contract bfloat %321 to float
  %323 = fmul contract float %322, %322
  %324 = fadd contract float %323, %320
  %325 = extractelement <8 x bfloat> %228, i64 3
  %326 = fpext contract bfloat %325 to float
  %327 = fmul contract float %326, %326
  %328 = fadd contract float %327, %324
  %329 = shufflevector <8 x bfloat> %228, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %330 = fpext <2 x bfloat> %329 to <2 x float>
  %331 = fmul contract <2 x float> %330, %330
  %332 = extractelement <2 x float> %331, i64 0
  %333 = fadd contract float %332, %328
  %334 = extractelement <2 x float> %331, i64 1
  %335 = fadd contract float %334, %333
  %336 = shufflevector <8 x bfloat> %228, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %337 = fpext <2 x bfloat> %336 to <2 x float>
  %338 = fmul contract <2 x float> %337, %337
  %339 = extractelement <2 x float> %338, i64 0
  %340 = fadd contract float %339, %335
  %341 = extractelement <2 x float> %338, i64 1
  %342 = fadd contract float %341, %340
  %343 = shufflevector <8 x bfloat> %230, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %344 = fpext <2 x bfloat> %343 to <2 x float>
  %345 = fmul contract <2 x float> %344, %344
  %346 = extractelement <2 x float> %345, i64 0
  %347 = fadd contract float %346, %342
  %348 = extractelement <2 x float> %345, i64 1
  %349 = fadd contract float %348, %347
  %350 = shufflevector <8 x bfloat> %230, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %351 = fpext <2 x bfloat> %350 to <2 x float>
  %352 = fmul contract <2 x float> %351, %351
  %353 = extractelement <2 x float> %352, i64 0
  %354 = fadd contract float %353, %349
  %355 = extractelement <2 x float> %352, i64 1
  %356 = fadd contract float %355, %354
  %357 = shufflevector <8 x bfloat> %230, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %358 = fpext <2 x bfloat> %357 to <2 x float>
  %359 = fmul contract <2 x float> %358, %358
  %360 = extractelement <2 x float> %359, i64 0
  %361 = fadd contract float %360, %356
  %362 = extractelement <2 x float> %359, i64 1
  %363 = fadd contract float %362, %361
  %364 = shufflevector <8 x bfloat> %230, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %365 = fpext <2 x bfloat> %364 to <2 x float>
  %366 = fmul contract <2 x float> %365, %365
  %367 = extractelement <2 x float> %366, i64 0
  %368 = fadd contract float %367, %363
  %369 = extractelement <2 x float> %366, i64 1
  %370 = fadd contract float %369, %368
  %371 = shufflevector <8 x bfloat> %224, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %372 = shufflevector <8 x bfloat> %224, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %373 = shufflevector <8 x bfloat> %224, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %374 = shufflevector <8 x bfloat> %226, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %375 = shufflevector <8 x bfloat> %226, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %376 = shufflevector <8 x bfloat> %226, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %377 = shufflevector <8 x bfloat> %226, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %378 = shufflevector <8 x bfloat> %228, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %379 = shufflevector <8 x bfloat> %228, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  br label %380

380:                                              ; preds = %250, %233
  %381 = phi bfloat [ %234, %233 ], [ %253, %250 ]
  %382 = phi float [ 0.000000e+00, %233 ], [ %370, %250 ]
  %383 = phi <2 x bfloat> [ %235, %233 ], [ %371, %250 ]
  %384 = phi <2 x bfloat> [ %236, %233 ], [ %372, %250 ]
  %385 = phi <2 x bfloat> [ %237, %233 ], [ %373, %250 ]
  %386 = phi <2 x bfloat> [ %238, %233 ], [ %374, %250 ]
  %387 = phi <2 x bfloat> [ %239, %233 ], [ %375, %250 ]
  %388 = phi <2 x bfloat> [ %240, %233 ], [ %376, %250 ]
  %389 = phi <2 x bfloat> [ %241, %233 ], [ %377, %250 ]
  %390 = phi <2 x bfloat> [ %242, %233 ], [ %378, %250 ]
  %391 = phi <2 x bfloat> [ %243, %233 ], [ %379, %250 ]
  %392 = phi <2 x bfloat> [ %244, %233 ], [ %329, %250 ]
  %393 = phi <2 x bfloat> [ %245, %233 ], [ %336, %250 ]
  %394 = phi <2 x bfloat> [ %246, %233 ], [ %343, %250 ]
  %395 = phi <2 x bfloat> [ %247, %233 ], [ %350, %250 ]
  %396 = phi <2 x bfloat> [ %248, %233 ], [ %357, %250 ]
  %397 = phi <2 x bfloat> [ %249, %233 ], [ %364, %250 ]
  %398 = insertelement <16 x bfloat> poison, bfloat %232, i64 0
  %399 = insertelement <16 x bfloat> %398, bfloat %381, i64 1
  %400 = shufflevector <2 x bfloat> %383, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %401 = shufflevector <16 x bfloat> %399, <16 x bfloat> %400, <16 x i32> <i32 0, i32 1, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %402 = shufflevector <2 x bfloat> %384, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %403 = shufflevector <16 x bfloat> %401, <16 x bfloat> %402, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %404 = shufflevector <2 x bfloat> %385, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %405 = shufflevector <16 x bfloat> %403, <16 x bfloat> %404, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %406 = shufflevector <2 x bfloat> %386, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %407 = shufflevector <16 x bfloat> %405, <16 x bfloat> %406, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %408 = shufflevector <2 x bfloat> %387, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %409 = shufflevector <16 x bfloat> %407, <16 x bfloat> %408, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison>
  %410 = shufflevector <2 x bfloat> %388, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %411 = shufflevector <16 x bfloat> %409, <16 x bfloat> %410, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 poison, i32 poison>
  %412 = shufflevector <2 x bfloat> %389, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %413 = shufflevector <16 x bfloat> %411, <16 x bfloat> %412, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 16, i32 17>
  %414 = shufflevector <2 x bfloat> %390, <2 x bfloat> %391, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %415 = shufflevector <2 x bfloat> %392, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %416 = shufflevector <16 x bfloat> %414, <16 x bfloat> %415, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %417 = shufflevector <2 x bfloat> %393, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %418 = shufflevector <16 x bfloat> %416, <16 x bfloat> %417, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %419 = shufflevector <2 x bfloat> %394, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %420 = shufflevector <16 x bfloat> %418, <16 x bfloat> %419, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %421 = shufflevector <2 x bfloat> %395, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %422 = shufflevector <16 x bfloat> %420, <16 x bfloat> %421, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison>
  %423 = shufflevector <2 x bfloat> %396, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %424 = shufflevector <16 x bfloat> %422, <16 x bfloat> %423, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 poison, i32 poison>
  %425 = shufflevector <2 x bfloat> %397, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %426 = shufflevector <16 x bfloat> %424, <16 x bfloat> %425, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 16, i32 17>
  %427 = shl nuw nsw i32 %31, 2
  %428 = and i32 %427, 60
  %429 = shl nuw nsw i32 %31, 8
  %430 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 0), i32 %429
  %431 = lshr exact i32 %34, 1
  %432 = xor i32 %428, %431
  %433 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %430, i32 %432
  %434 = load <8 x bfloat>, ptr addrspace(3) %433, align 16, !tbaa !29
  %435 = or disjoint i32 %431, 8
  %436 = xor i32 %428, %435
  %437 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %430, i32 %436
  %438 = load <8 x bfloat>, ptr addrspace(3) %437, align 16, !tbaa !29
  %439 = or disjoint i32 %431, 16
  %440 = xor i32 %428, %439
  %441 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %430, i32 %440
  %442 = load <8 x bfloat>, ptr addrspace(3) %441, align 16, !tbaa !29
  %443 = shufflevector <8 x bfloat> %434, <8 x bfloat> %442, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %444 = or disjoint i32 %431, 24
  %445 = xor i32 %428, %444
  %446 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %430, i32 %445
  %447 = load <8 x bfloat>, ptr addrspace(3) %446, align 16, !tbaa !29
  %448 = shufflevector <8 x bfloat> %438, <8 x bfloat> %447, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  tail call void @llvm.amdgcn.s.wait.dscnt(i16 noundef signext 0) #24
  %449 = tail call contract noundef <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %443, <16 x bfloat> %413, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %450 = tail call contract noundef <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %448, <16 x bfloat> %413, i16 0, <8 x float> %449, i1 false, i1 false)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %451 = or disjoint i32 %431, 32
  %452 = xor i32 %428, %451
  %453 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %430, i32 %452
  %454 = load <8 x bfloat>, ptr addrspace(3) %453, align 16, !tbaa !29
  %455 = or disjoint i32 %431, 40
  %456 = xor i32 %428, %455
  %457 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %430, i32 %456
  %458 = load <8 x bfloat>, ptr addrspace(3) %457, align 16, !tbaa !29
  %459 = or disjoint i32 %431, 48
  %460 = xor i32 %428, %459
  %461 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %430, i32 %460
  %462 = load <8 x bfloat>, ptr addrspace(3) %461, align 16, !tbaa !29
  %463 = shufflevector <8 x bfloat> %454, <8 x bfloat> %462, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %464 = or disjoint i32 %431, 56
  %465 = xor i32 %428, %464
  %466 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %430, i32 %465
  %467 = load <8 x bfloat>, ptr addrspace(3) %466, align 16, !tbaa !29
  %468 = shufflevector <8 x bfloat> %458, <8 x bfloat> %467, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  tail call void @llvm.amdgcn.s.wait.dscnt(i16 noundef signext 0) #24
  %469 = tail call contract noundef <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %463, <16 x bfloat> %426, i16 0, <8 x float> %450, i1 false, i1 false)
  %470 = tail call contract noundef <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %468, <16 x bfloat> %426, i16 0, <8 x float> %469, i1 false, i1 false)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.wait.asynccnt(i16 noundef signext 0) #24
  tail call void @llvm.amdgcn.s.wait.tensorcnt(i16 noundef signext 0) #24
  tail call void @mhc_pre_subgroup_barrier()
  %471 = getelementptr inbounds i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 8192), i32 9216), i32 %220
  %472 = getelementptr inbounds nuw i8, ptr addrspace(3) %471, i32 %222
  %473 = load <8 x bfloat>, ptr addrspace(3) %472, align 16, !tbaa !29
  %474 = getelementptr inbounds nuw i8, ptr addrspace(3) %472, i32 32
  %475 = load <8 x bfloat>, ptr addrspace(3) %474, align 16, !tbaa !29
  %476 = getelementptr inbounds nuw i8, ptr addrspace(3) %472, i32 64
  %477 = load <8 x bfloat>, ptr addrspace(3) %476, align 16, !tbaa !29
  %478 = getelementptr inbounds nuw i8, ptr addrspace(3) %472, i32 96
  %479 = load <8 x bfloat>, ptr addrspace(3) %478, align 16, !tbaa !29
  tail call void @llvm.amdgcn.s.wait.dscnt(i16 noundef signext 0) #24
  %480 = extractelement <8 x bfloat> %473, i64 0
  br i1 %231, label %498, label %481

481:                                              ; preds = %380
  %482 = extractelement <8 x bfloat> %473, i64 1
  %483 = shufflevector <8 x bfloat> %473, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %484 = shufflevector <8 x bfloat> %473, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %485 = shufflevector <8 x bfloat> %473, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %486 = shufflevector <8 x bfloat> %475, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %487 = shufflevector <8 x bfloat> %475, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %488 = shufflevector <8 x bfloat> %475, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %489 = shufflevector <8 x bfloat> %475, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %490 = shufflevector <8 x bfloat> %477, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %491 = shufflevector <8 x bfloat> %477, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %492 = shufflevector <8 x bfloat> %477, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %493 = shufflevector <8 x bfloat> %477, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %494 = shufflevector <8 x bfloat> %479, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %495 = shufflevector <8 x bfloat> %479, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %496 = shufflevector <8 x bfloat> %479, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %497 = shufflevector <8 x bfloat> %479, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  br label %629

498:                                              ; preds = %380
  %499 = fpext contract bfloat %480 to float
  %500 = fmul contract float %499, %499
  %501 = fadd contract float %382, %500
  %502 = extractelement <8 x bfloat> %473, i64 1
  %503 = fpext contract bfloat %502 to float
  %504 = fmul contract float %503, %503
  %505 = fadd contract float %504, %501
  %506 = extractelement <8 x bfloat> %473, i64 2
  %507 = fpext contract bfloat %506 to float
  %508 = fmul contract float %507, %507
  %509 = fadd contract float %508, %505
  %510 = extractelement <8 x bfloat> %473, i64 3
  %511 = fpext contract bfloat %510 to float
  %512 = fmul contract float %511, %511
  %513 = fadd contract float %512, %509
  %514 = extractelement <8 x bfloat> %473, i64 4
  %515 = fpext contract bfloat %514 to float
  %516 = fmul contract float %515, %515
  %517 = fadd contract float %516, %513
  %518 = extractelement <8 x bfloat> %473, i64 5
  %519 = fpext contract bfloat %518 to float
  %520 = fmul contract float %519, %519
  %521 = fadd contract float %520, %517
  %522 = extractelement <8 x bfloat> %473, i64 6
  %523 = fpext contract bfloat %522 to float
  %524 = fmul contract float %523, %523
  %525 = fadd contract float %524, %521
  %526 = extractelement <8 x bfloat> %473, i64 7
  %527 = fpext contract bfloat %526 to float
  %528 = fmul contract float %527, %527
  %529 = fadd contract float %528, %525
  %530 = extractelement <8 x bfloat> %475, i64 0
  %531 = fpext contract bfloat %530 to float
  %532 = fmul contract float %531, %531
  %533 = fadd contract float %532, %529
  %534 = extractelement <8 x bfloat> %475, i64 1
  %535 = fpext contract bfloat %534 to float
  %536 = fmul contract float %535, %535
  %537 = fadd contract float %536, %533
  %538 = extractelement <8 x bfloat> %475, i64 2
  %539 = fpext contract bfloat %538 to float
  %540 = fmul contract float %539, %539
  %541 = fadd contract float %540, %537
  %542 = extractelement <8 x bfloat> %475, i64 3
  %543 = fpext contract bfloat %542 to float
  %544 = fmul contract float %543, %543
  %545 = fadd contract float %544, %541
  %546 = extractelement <8 x bfloat> %475, i64 4
  %547 = fpext contract bfloat %546 to float
  %548 = fmul contract float %547, %547
  %549 = fadd contract float %548, %545
  %550 = extractelement <8 x bfloat> %475, i64 5
  %551 = fpext contract bfloat %550 to float
  %552 = fmul contract float %551, %551
  %553 = fadd contract float %552, %549
  %554 = extractelement <8 x bfloat> %475, i64 6
  %555 = fpext contract bfloat %554 to float
  %556 = fmul contract float %555, %555
  %557 = fadd contract float %556, %553
  %558 = extractelement <8 x bfloat> %475, i64 7
  %559 = fpext contract bfloat %558 to float
  %560 = fmul contract float %559, %559
  %561 = fadd contract float %560, %557
  %562 = extractelement <8 x bfloat> %477, i64 0
  %563 = fpext contract bfloat %562 to float
  %564 = fmul contract float %563, %563
  %565 = fadd contract float %564, %561
  %566 = extractelement <8 x bfloat> %477, i64 1
  %567 = fpext contract bfloat %566 to float
  %568 = fmul contract float %567, %567
  %569 = fadd contract float %568, %565
  %570 = extractelement <8 x bfloat> %477, i64 2
  %571 = fpext contract bfloat %570 to float
  %572 = fmul contract float %571, %571
  %573 = fadd contract float %572, %569
  %574 = extractelement <8 x bfloat> %477, i64 3
  %575 = fpext contract bfloat %574 to float
  %576 = fmul contract float %575, %575
  %577 = fadd contract float %576, %573
  %578 = shufflevector <8 x bfloat> %477, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %579 = fpext <2 x bfloat> %578 to <2 x float>
  %580 = fmul contract <2 x float> %579, %579
  %581 = extractelement <2 x float> %580, i64 0
  %582 = fadd contract float %581, %577
  %583 = extractelement <2 x float> %580, i64 1
  %584 = fadd contract float %583, %582
  %585 = shufflevector <8 x bfloat> %477, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %586 = fpext <2 x bfloat> %585 to <2 x float>
  %587 = fmul contract <2 x float> %586, %586
  %588 = extractelement <2 x float> %587, i64 0
  %589 = fadd contract float %588, %584
  %590 = extractelement <2 x float> %587, i64 1
  %591 = fadd contract float %590, %589
  %592 = shufflevector <8 x bfloat> %479, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %593 = fpext <2 x bfloat> %592 to <2 x float>
  %594 = fmul contract <2 x float> %593, %593
  %595 = extractelement <2 x float> %594, i64 0
  %596 = fadd contract float %595, %591
  %597 = extractelement <2 x float> %594, i64 1
  %598 = fadd contract float %597, %596
  %599 = shufflevector <8 x bfloat> %479, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %600 = fpext <2 x bfloat> %599 to <2 x float>
  %601 = fmul contract <2 x float> %600, %600
  %602 = extractelement <2 x float> %601, i64 0
  %603 = fadd contract float %602, %598
  %604 = extractelement <2 x float> %601, i64 1
  %605 = fadd contract float %604, %603
  %606 = shufflevector <8 x bfloat> %479, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %607 = fpext <2 x bfloat> %606 to <2 x float>
  %608 = fmul contract <2 x float> %607, %607
  %609 = extractelement <2 x float> %608, i64 0
  %610 = fadd contract float %609, %605
  %611 = extractelement <2 x float> %608, i64 1
  %612 = fadd contract float %611, %610
  %613 = shufflevector <8 x bfloat> %479, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %614 = fpext <2 x bfloat> %613 to <2 x float>
  %615 = fmul contract <2 x float> %614, %614
  %616 = extractelement <2 x float> %615, i64 0
  %617 = fadd contract float %616, %612
  %618 = extractelement <2 x float> %615, i64 1
  %619 = fadd contract float %618, %617
  %620 = shufflevector <8 x bfloat> %473, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %621 = shufflevector <8 x bfloat> %473, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %622 = shufflevector <8 x bfloat> %473, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %623 = shufflevector <8 x bfloat> %475, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %624 = shufflevector <8 x bfloat> %475, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %625 = shufflevector <8 x bfloat> %475, <8 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %626 = shufflevector <8 x bfloat> %475, <8 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %627 = shufflevector <8 x bfloat> %477, <8 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %628 = shufflevector <8 x bfloat> %477, <8 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  br label %629

629:                                              ; preds = %498, %481
  %630 = phi bfloat [ %482, %481 ], [ %502, %498 ]
  %631 = phi float [ %382, %481 ], [ %619, %498 ]
  %632 = phi <2 x bfloat> [ %483, %481 ], [ %620, %498 ]
  %633 = phi <2 x bfloat> [ %484, %481 ], [ %621, %498 ]
  %634 = phi <2 x bfloat> [ %485, %481 ], [ %622, %498 ]
  %635 = phi <2 x bfloat> [ %486, %481 ], [ %623, %498 ]
  %636 = phi <2 x bfloat> [ %487, %481 ], [ %624, %498 ]
  %637 = phi <2 x bfloat> [ %488, %481 ], [ %625, %498 ]
  %638 = phi <2 x bfloat> [ %489, %481 ], [ %626, %498 ]
  %639 = phi <2 x bfloat> [ %490, %481 ], [ %627, %498 ]
  %640 = phi <2 x bfloat> [ %491, %481 ], [ %628, %498 ]
  %641 = phi <2 x bfloat> [ %492, %481 ], [ %578, %498 ]
  %642 = phi <2 x bfloat> [ %493, %481 ], [ %585, %498 ]
  %643 = phi <2 x bfloat> [ %494, %481 ], [ %592, %498 ]
  %644 = phi <2 x bfloat> [ %495, %481 ], [ %599, %498 ]
  %645 = phi <2 x bfloat> [ %496, %481 ], [ %606, %498 ]
  %646 = phi <2 x bfloat> [ %497, %481 ], [ %613, %498 ]
  %647 = insertelement <16 x bfloat> poison, bfloat %480, i64 0
  %648 = insertelement <16 x bfloat> %647, bfloat %630, i64 1
  %649 = shufflevector <2 x bfloat> %632, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %650 = shufflevector <16 x bfloat> %648, <16 x bfloat> %649, <16 x i32> <i32 0, i32 1, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %651 = shufflevector <2 x bfloat> %633, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %652 = shufflevector <16 x bfloat> %650, <16 x bfloat> %651, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %653 = shufflevector <2 x bfloat> %634, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %654 = shufflevector <16 x bfloat> %652, <16 x bfloat> %653, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %655 = shufflevector <2 x bfloat> %635, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %656 = shufflevector <16 x bfloat> %654, <16 x bfloat> %655, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %657 = shufflevector <2 x bfloat> %636, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %658 = shufflevector <16 x bfloat> %656, <16 x bfloat> %657, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison>
  %659 = shufflevector <2 x bfloat> %637, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %660 = shufflevector <16 x bfloat> %658, <16 x bfloat> %659, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 poison, i32 poison>
  %661 = shufflevector <2 x bfloat> %638, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %662 = shufflevector <16 x bfloat> %660, <16 x bfloat> %661, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 16, i32 17>
  %663 = shufflevector <2 x bfloat> %639, <2 x bfloat> %640, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %664 = shufflevector <2 x bfloat> %641, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %665 = shufflevector <16 x bfloat> %663, <16 x bfloat> %664, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %666 = shufflevector <2 x bfloat> %642, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %667 = shufflevector <16 x bfloat> %665, <16 x bfloat> %666, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %668 = shufflevector <2 x bfloat> %643, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %669 = shufflevector <16 x bfloat> %667, <16 x bfloat> %668, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %670 = shufflevector <2 x bfloat> %644, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %671 = shufflevector <16 x bfloat> %669, <16 x bfloat> %670, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 16, i32 17, i32 poison, i32 poison, i32 poison, i32 poison>
  %672 = shufflevector <2 x bfloat> %645, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %673 = shufflevector <16 x bfloat> %671, <16 x bfloat> %672, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 poison, i32 poison>
  %674 = shufflevector <2 x bfloat> %646, <2 x bfloat> poison, <16 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %675 = shufflevector <16 x bfloat> %673, <16 x bfloat> %674, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 16, i32 17>
  %676 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 0), i32 4096), i32 %429
  %677 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %676, i32 %432
  %678 = load <8 x bfloat>, ptr addrspace(3) %677, align 16, !tbaa !29
  %679 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %676, i32 %436
  %680 = load <8 x bfloat>, ptr addrspace(3) %679, align 16, !tbaa !29
  %681 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %676, i32 %440
  %682 = load <8 x bfloat>, ptr addrspace(3) %681, align 16, !tbaa !29
  %683 = shufflevector <8 x bfloat> %678, <8 x bfloat> %682, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %684 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %676, i32 %445
  %685 = load <8 x bfloat>, ptr addrspace(3) %684, align 16, !tbaa !29
  %686 = shufflevector <8 x bfloat> %680, <8 x bfloat> %685, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  tail call void @llvm.amdgcn.s.wait.dscnt(i16 noundef signext 0) #24
  %687 = tail call contract noundef <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %683, <16 x bfloat> %662, i16 0, <8 x float> %470, i1 false, i1 false)
  %688 = tail call contract noundef <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %686, <16 x bfloat> %662, i16 0, <8 x float> %687, i1 false, i1 false)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %689 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %676, i32 %452
  %690 = load <8 x bfloat>, ptr addrspace(3) %689, align 16, !tbaa !29
  %691 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %676, i32 %456
  %692 = load <8 x bfloat>, ptr addrspace(3) %691, align 16, !tbaa !29
  %693 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %676, i32 %460
  %694 = load <8 x bfloat>, ptr addrspace(3) %693, align 16, !tbaa !29
  %695 = shufflevector <8 x bfloat> %690, <8 x bfloat> %694, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %696 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %676, i32 %465
  %697 = load <8 x bfloat>, ptr addrspace(3) %696, align 16, !tbaa !29
  %698 = shufflevector <8 x bfloat> %692, <8 x bfloat> %697, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  tail call void @llvm.amdgcn.s.wait.dscnt(i16 noundef signext 0) #24
  %699 = tail call contract noundef <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %695, <16 x bfloat> %675, i16 0, <8 x float> %688, i1 false, i1 false)
  %700 = tail call contract noundef <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %698, <16 x bfloat> %675, i16 0, <8 x float> %699, i1 false, i1 false)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  br i1 %231, label %701, label %718

701:                                              ; preds = %629
  %702 = bitcast float %631 to i32
  %703 = shl nuw nsw i32 %18, 2
  %704 = xor i32 %703, 64
  %705 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %704, i32 %702)
  %706 = or disjoint i32 %28, %18
  %707 = icmp slt i32 %706, %24
  %708 = select i1 %30, i1 %707, i1 false
  br i1 %708, label %709, label %718

709:                                              ; preds = %701
  %710 = getelementptr [4 x i8], ptr %8, i64 %12
  %711 = getelementptr [4 x i8], ptr %710, i64 %23
  %712 = zext nneg i32 %28 to i64
  %713 = getelementptr [4 x i8], ptr %711, i64 %712
  %714 = zext nneg i32 %18 to i64
  %715 = getelementptr [4 x i8], ptr %713, i64 %714
  %716 = bitcast i32 %705 to float
  %717 = fadd contract float %631, %716
  store float %717, ptr %715, align 4, !tbaa !39
  br label %718

718:                                              ; preds = %629, %701, %709
  %719 = inttoptr i64 %0 to ptr
  %720 = add nsw i64 %23, %12
  %721 = shl nsw i64 %720, 7
  %722 = getelementptr inbounds i8, ptr %719, i64 %721
  %723 = sext i32 %13 to i64
  %724 = getelementptr inbounds [4 x i8], ptr %722, i64 %723
  %725 = shl i32 %24, 7
  %726 = zext i32 %725 to i64
  %727 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %724, i16 0, i64 %726, i32 822099968)
  %728 = shl i32 %32, 7
  %729 = shl nuw nsw i32 %34, 2
  %730 = or disjoint i32 %728, %729
  %731 = bitcast <8 x float> %700 to <8 x i32>
  %732 = shufflevector <8 x i32> %731, <8 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %733 = bitcast <8 x float> %700 to <8 x i32>
  %734 = shufflevector <8 x i32> %733, <8 x i32> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %732, ptr addrspace(8) %727, i32 %730, i32 0, i32 0)
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %734, ptr addrspace(8) %727, i32 %730, i32 16, i32 0)
  ret void
}

; Function Attrs: alwaysinline convergent mustprogress nounwind uwtable
define hidden void @mhc_native_pre_reduce(i64 noundef %0, i64 noundef %1, i64 noundef %2, i64 noundef %3, i64 noundef %4, i64 noundef %5, i64 noundef %6, i64 noundef %7, i64 noundef %8, i32 noundef %9, i64 noundef %10) #8 {
  %12 = inttoptr i64 %0 to ptr
  %13 = inttoptr i64 %1 to ptr
  %14 = inttoptr i64 %2 to ptr
  %15 = inttoptr i64 %3 to ptr
  %16 = inttoptr i64 %5 to ptr
  %17 = inttoptr i64 %6 to ptr
  %18 = inttoptr i64 %7 to ptr
  %19 = inttoptr i64 %8 to ptr
  %20 = addrspacecast ptr addrspace(3) @__dynamic_shared_0 to ptr
  %21 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %22 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %21)
  %23 = lshr i32 %22, 5
  %24 = sub nsw i32 16, %9
  %25 = sext i32 %9 to i64
  %26 = sub nsw i32 2048, %9
  %27 = shl nsw i32 %9, 5
  %28 = sext i32 %27 to i64
  %29 = shl i32 %26, 7
  %30 = zext i32 %29 to i64
  %31 = getelementptr inbounds nuw i8, ptr %20, i64 64
  %32 = and i32 %21, 31
  %33 = icmp slt i32 %9, 16
  %34 = select i1 %33, i32 1, i32 %24
  %35 = getelementptr inbounds [4 x i8], ptr %15, i64 %28
  %36 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %35, i16 0, i64 %30, i32 822099968)
  %37 = icmp samesign ult i32 %21, 128
  br i1 %37, label %38, label %47

38:                                               ; preds = %11
  %39 = shl nuw nsw i32 %21, 6
  %40 = inttoptr i64 %4 to ptr
  %41 = getelementptr inbounds [4 x i8], ptr %40, i64 %25
  %42 = shl i32 %26, 2
  %43 = zext i32 %42 to i64
  %44 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %41, i16 0, i64 %43, i32 822099968)
  %45 = tail call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %44, i32 %39, i32 0, i32 0)
  %46 = bitcast i32 %45 to <1 x float>
  br label %47

47:                                               ; preds = %38, %11
  %48 = phi <1 x float> [ %46, %38 ], [ zeroinitializer, %11 ]
  %49 = icmp samesign ult i32 %21, 288
  br i1 %49, label %50, label %102

50:                                               ; preds = %47
  %51 = trunc nuw nsw i32 %21 to i16
  %52 = udiv i16 %51, 6
  %53 = zext nneg i16 %52 to i32
  %54 = mul i16 %52, 6
  %55 = sub i16 %51, %54
  %56 = shl nuw nsw i16 %55, 2
  %57 = zext nneg i16 %56 to i32
  %58 = icmp sgt i32 %34, 0
  br i1 %58, label %59, label %93

59:                                               ; preds = %50
  %60 = shl nuw nsw i32 %53, 11
  %61 = shl nuw nsw i32 %57, 2
  %62 = or disjoint i32 %61, %60
  %63 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %36, i32 %62, i32 0, i32 0)
  %64 = shl nuw nsw i32 %53, 9
  %65 = add nuw nsw i32 %64, 24576
  %66 = or disjoint i32 %65, %57
  %67 = shl nuw nsw i32 %66, 2
  %68 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %36, i32 %67, i32 0, i32 0)
  %69 = add nuw nsw i32 %64, 49152
  %70 = or disjoint i32 %69, %57
  %71 = shl nuw nsw i32 %70, 2
  %72 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %36, i32 %71, i32 0, i32 0)
  %73 = add nuw nsw i32 %64, 73728
  %74 = or disjoint i32 %73, %57
  %75 = shl nuw nsw i32 %74, 2
  %76 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %36, i32 %75, i32 0, i32 0)
  %77 = shufflevector <4 x i32> %63, <4 x i32> %68, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %78 = shufflevector <4 x i32> %72, <4 x i32> %76, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %79 = shufflevector <8 x i32> %77, <8 x i32> %78, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %80 = bitcast <16 x i32> %79 to <4 x i128>
  %81 = extractelement <4 x i128> %80, i64 0
  %82 = bitcast i128 %81 to <4 x float>
  %83 = extractelement <4 x i128> %80, i64 1
  %84 = bitcast i128 %83 to <4 x float>
  %85 = extractelement <4 x i128> %80, i64 2
  %86 = bitcast i128 %85 to <4 x float>
  %87 = extractelement <4 x i128> %80, i64 3
  %88 = bitcast i128 %87 to <4 x float>
  %89 = fadd contract <4 x float> %82, zeroinitializer
  %90 = fadd contract <4 x float> %89, %84
  %91 = fadd contract <4 x float> %90, %86
  %92 = fadd contract <4 x float> %91, %88
  br label %93, !llvm.loop !41

93:                                               ; preds = %59, %50
  %94 = phi <4 x float> [ zeroinitializer, %50 ], [ %92, %59 ]
  %95 = mul nuw nsw i32 %53, 24
  %96 = add nuw nsw i32 %95, %57
  %97 = zext nneg i32 %96 to i64
  %98 = getelementptr inbounds nuw [4 x i8], ptr %31, i64 %97
  %99 = shufflevector <4 x float> %94, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  store <2 x float> %99, ptr %98, align 4, !tbaa !39
  %100 = getelementptr inbounds nuw i8, ptr %98, i64 8
  %101 = shufflevector <4 x float> %94, <4 x float> poison, <2 x i32> <i32 2, i32 3>
  store <2 x float> %101, ptr %100, align 4, !tbaa !39
  br label %102

102:                                              ; preds = %93, %47
  %103 = extractelement <1 x float> %48, i64 0
  %104 = fadd contract float %103, 0.000000e+00
  %105 = select i1 %37, float %104, float 0.000000e+00
  %106 = bitcast float %105 to i32
  %107 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %106, i32 177, i32 15, i32 15, i1 false)
  %108 = bitcast i32 %107 to float
  %109 = fadd contract float %105, %108
  %110 = bitcast float %109 to i32
  %111 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %110, i32 78, i32 15, i32 15, i1 false)
  %112 = bitcast i32 %111 to float
  %113 = fadd contract float %109, %112
  %114 = bitcast float %113 to i32
  %115 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %114, i32 359, i32 15, i32 15, i1 false)
  %116 = bitcast i32 %115 to float
  %117 = fadd contract float %113, %116
  %118 = bitcast float %117 to i32
  %119 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %118, i32 367, i32 15, i32 15, i1 false)
  %120 = bitcast i32 %119 to float
  %121 = fadd contract float %117, %120
  %122 = bitcast float %121 to i32
  %123 = tail call i32 @llvm.amdgcn.permlanex16.i32(i32 poison, i32 %122, i32 1985229328, i32 -19088744, i1 true, i1 true)
  %124 = icmp eq i32 %32, 31
  br i1 %124, label %125, label %130

125:                                              ; preds = %102
  %126 = zext nneg i32 %23 to i64
  %127 = getelementptr inbounds nuw [4 x i8], ptr %20, i64 %126
  %128 = bitcast i32 %123 to float
  %129 = fadd contract float %121, %128
  store float %129, ptr %127, align 4, !tbaa !39
  br label %130

130:                                              ; preds = %125, %102
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %131 = icmp eq i32 %23, 0
  br i1 %131, label %132, label %168

132:                                              ; preds = %130
  %133 = icmp samesign ult i32 %32, 16
  %134 = and i32 %21, 15
  %135 = icmp samesign ult i32 %134, 9
  %136 = and i1 %133, %135
  br i1 %136, label %137, label %141

137:                                              ; preds = %132
  %138 = zext nneg i32 %32 to i64
  %139 = getelementptr inbounds nuw [4 x i8], ptr %20, i64 %138
  %140 = load float, ptr %139, align 4, !tbaa !39
  br label %141

141:                                              ; preds = %137, %132
  %142 = phi float [ %140, %137 ], [ 0.000000e+00, %132 ]
  %143 = bitcast float %142 to i32
  %144 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %143, i32 177, i32 15, i32 15, i1 false)
  %145 = bitcast i32 %144 to float
  %146 = fadd contract float %142, %145
  %147 = bitcast float %146 to i32
  %148 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %147, i32 78, i32 15, i32 15, i1 false)
  %149 = bitcast i32 %148 to float
  %150 = fadd contract float %146, %149
  %151 = bitcast float %150 to i32
  %152 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %151, i32 321, i32 15, i32 15, i1 false)
  %153 = bitcast i32 %152 to float
  %154 = fadd contract float %150, %153
  %155 = bitcast float %154 to i32
  %156 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %155, i32 320, i32 15, i32 15, i1 false)
  %157 = bitcast i32 %156 to float
  %158 = fadd contract float %154, %157
  %159 = fmul contract float %158, f0x38800000
  %160 = fadd contract float %159, f0x358637BD
  %161 = fcmp olt float %160, f0x00800000
  %162 = fmul float %160, f0x4B800000
  %163 = select i1 %161, float %162, float %160
  %164 = tail call float @llvm.amdgcn.rsq.f32(float %163)
  %165 = fmul float %164, 4.096000e+03
  %166 = select i1 %161, float %165, float %164
  %167 = tail call float @llvm.amdgcn.readlane.f32(float %166, i32 0)
  br label %168

168:                                              ; preds = %141, %130
  %169 = phi float [ undef, %130 ], [ %167, %141 ]
  %170 = and i32 %21, 3
  %171 = load float, ptr %16, align 4, !tbaa !39
  %172 = zext nneg i32 %170 to i64
  %173 = getelementptr inbounds nuw [4 x i8], ptr %17, i64 %172
  %174 = load float, ptr %173, align 4, !tbaa !39
  %175 = icmp samesign ult i32 %21, 24
  br i1 %175, label %176, label %343

176:                                              ; preds = %168
  %177 = icmp sgt i32 %34, 0
  br i1 %177, label %178, label %340

178:                                              ; preds = %176
  %179 = zext nneg i32 %21 to i64
  %180 = getelementptr inbounds nuw [4 x i8], ptr %31, i64 %179
  %181 = getelementptr inbounds nuw i8, ptr %180, i64 288
  %182 = load float, ptr %181, align 4, !tbaa !39
  %183 = fadd contract float %182, 0.000000e+00
  %184 = getelementptr inbounds nuw i8, ptr %180, i64 672
  %185 = load float, ptr %184, align 4, !tbaa !39
  %186 = fadd contract float %183, %185
  %187 = getelementptr inbounds nuw i8, ptr %180, i64 1056
  %188 = load float, ptr %187, align 4, !tbaa !39
  %189 = getelementptr inbounds nuw i8, ptr %180, i64 1440
  %190 = load float, ptr %189, align 4, !tbaa !39
  %191 = getelementptr inbounds nuw i8, ptr %180, i64 1824
  %192 = load float, ptr %191, align 4, !tbaa !39
  %193 = getelementptr inbounds nuw i8, ptr %180, i64 2208
  %194 = load float, ptr %193, align 4, !tbaa !39
  %195 = getelementptr inbounds nuw i8, ptr %180, i64 2592
  %196 = load float, ptr %195, align 4, !tbaa !39
  %197 = getelementptr inbounds nuw i8, ptr %180, i64 2976
  %198 = load float, ptr %197, align 4, !tbaa !39
  %199 = getelementptr inbounds nuw i8, ptr %180, i64 3360
  %200 = load float, ptr %199, align 4, !tbaa !39
  %201 = getelementptr inbounds nuw i8, ptr %180, i64 3744
  %202 = load float, ptr %201, align 4, !tbaa !39
  %203 = getelementptr inbounds nuw i8, ptr %180, i64 4128
  %204 = load float, ptr %203, align 4, !tbaa !39
  %205 = getelementptr inbounds nuw i8, ptr %180, i64 4512
  %206 = load float, ptr %205, align 4, !tbaa !39
  %207 = getelementptr inbounds nuw i8, ptr %180, i64 192
  %208 = load float, ptr %207, align 4, !tbaa !39
  %209 = fadd contract float %208, 0.000000e+00
  %210 = getelementptr inbounds nuw i8, ptr %180, i64 576
  %211 = load float, ptr %210, align 4, !tbaa !39
  %212 = fadd contract float %209, %211
  %213 = getelementptr inbounds nuw i8, ptr %180, i64 960
  %214 = load float, ptr %213, align 4, !tbaa !39
  %215 = fadd contract float %212, %214
  %216 = getelementptr inbounds nuw i8, ptr %180, i64 1344
  %217 = load float, ptr %216, align 4, !tbaa !39
  %218 = fadd contract float %215, %217
  %219 = getelementptr inbounds nuw i8, ptr %180, i64 1728
  %220 = load float, ptr %219, align 4, !tbaa !39
  %221 = fadd contract float %218, %220
  %222 = getelementptr inbounds nuw i8, ptr %180, i64 2112
  %223 = load float, ptr %222, align 4, !tbaa !39
  %224 = fadd contract float %221, %223
  %225 = getelementptr inbounds nuw i8, ptr %180, i64 2496
  %226 = load float, ptr %225, align 4, !tbaa !39
  %227 = fadd contract float %224, %226
  %228 = getelementptr inbounds nuw i8, ptr %180, i64 2880
  %229 = load float, ptr %228, align 4, !tbaa !39
  %230 = fadd contract float %227, %229
  %231 = getelementptr inbounds nuw i8, ptr %180, i64 3264
  %232 = load float, ptr %231, align 4, !tbaa !39
  %233 = fadd contract float %230, %232
  %234 = getelementptr inbounds nuw i8, ptr %180, i64 3648
  %235 = load float, ptr %234, align 4, !tbaa !39
  %236 = fadd contract float %233, %235
  %237 = getelementptr inbounds nuw i8, ptr %180, i64 4032
  %238 = load float, ptr %237, align 4, !tbaa !39
  %239 = fadd contract float %236, %238
  %240 = getelementptr inbounds nuw i8, ptr %180, i64 4416
  %241 = load float, ptr %240, align 4, !tbaa !39
  %242 = fadd contract float %239, %241
  %243 = getelementptr inbounds nuw i8, ptr %180, i64 96
  %244 = load float, ptr %243, align 4, !tbaa !39
  %245 = fadd contract float %244, 0.000000e+00
  %246 = getelementptr inbounds nuw i8, ptr %180, i64 480
  %247 = load float, ptr %246, align 4, !tbaa !39
  %248 = getelementptr inbounds nuw i8, ptr %180, i64 864
  %249 = load float, ptr %248, align 4, !tbaa !39
  %250 = getelementptr inbounds nuw i8, ptr %180, i64 1248
  %251 = load float, ptr %250, align 4, !tbaa !39
  %252 = getelementptr inbounds nuw i8, ptr %180, i64 1632
  %253 = load float, ptr %252, align 4, !tbaa !39
  %254 = getelementptr inbounds nuw i8, ptr %180, i64 2016
  %255 = load float, ptr %254, align 4, !tbaa !39
  %256 = getelementptr inbounds nuw i8, ptr %180, i64 2400
  %257 = load float, ptr %256, align 4, !tbaa !39
  %258 = getelementptr inbounds nuw i8, ptr %180, i64 2784
  %259 = load float, ptr %258, align 4, !tbaa !39
  %260 = getelementptr inbounds nuw i8, ptr %180, i64 3168
  %261 = load float, ptr %260, align 4, !tbaa !39
  %262 = getelementptr inbounds nuw i8, ptr %180, i64 3552
  %263 = load float, ptr %262, align 4, !tbaa !39
  %264 = getelementptr inbounds nuw i8, ptr %180, i64 3936
  %265 = load float, ptr %264, align 4, !tbaa !39
  %266 = getelementptr inbounds nuw i8, ptr %180, i64 4320
  %267 = load float, ptr %266, align 4, !tbaa !39
  %268 = load float, ptr %180, align 4, !tbaa !39
  %269 = getelementptr inbounds nuw i8, ptr %180, i64 384
  %270 = load float, ptr %269, align 4, !tbaa !39
  %271 = getelementptr inbounds nuw i8, ptr %180, i64 768
  %272 = load float, ptr %271, align 4, !tbaa !39
  %273 = getelementptr inbounds nuw i8, ptr %180, i64 1152
  %274 = load float, ptr %273, align 4, !tbaa !39
  %275 = getelementptr inbounds nuw i8, ptr %180, i64 1536
  %276 = load float, ptr %275, align 4, !tbaa !39
  %277 = getelementptr inbounds nuw i8, ptr %180, i64 1920
  %278 = load float, ptr %277, align 4, !tbaa !39
  %279 = getelementptr inbounds nuw i8, ptr %180, i64 2304
  %280 = load float, ptr %279, align 4, !tbaa !39
  %281 = getelementptr inbounds nuw i8, ptr %180, i64 2688
  %282 = load float, ptr %281, align 4, !tbaa !39
  %283 = getelementptr inbounds nuw i8, ptr %180, i64 3072
  %284 = load float, ptr %283, align 4, !tbaa !39
  %285 = getelementptr inbounds nuw i8, ptr %180, i64 3456
  %286 = load float, ptr %285, align 4, !tbaa !39
  %287 = getelementptr inbounds nuw i8, ptr %180, i64 3840
  %288 = load float, ptr %287, align 4, !tbaa !39
  %289 = getelementptr inbounds nuw i8, ptr %180, i64 4224
  %290 = insertelement <4 x float> <float -0.000000e+00, float 0.000000e+00, float poison, float poison>, float %247, i64 2
  %291 = insertelement <4 x float> %290, float %188, i64 3
  %292 = insertelement <4 x float> <float -0.000000e+00, float poison, float poison, float poison>, float %268, i64 1
  %293 = insertelement <4 x float> %292, float %245, i64 2
  %294 = insertelement <4 x float> %293, float %186, i64 3
  %295 = fadd contract <4 x float> %291, %294
  %296 = insertelement <4 x float> <float -0.000000e+00, float poison, float poison, float poison>, float %270, i64 1
  %297 = insertelement <4 x float> %296, float %249, i64 2
  %298 = insertelement <4 x float> %297, float %190, i64 3
  %299 = fadd contract <4 x float> %298, %295
  %300 = insertelement <4 x float> <float -0.000000e+00, float poison, float poison, float poison>, float %272, i64 1
  %301 = insertelement <4 x float> %300, float %251, i64 2
  %302 = insertelement <4 x float> %301, float %192, i64 3
  %303 = fadd contract <4 x float> %302, %299
  %304 = insertelement <4 x float> <float -0.000000e+00, float poison, float poison, float poison>, float %274, i64 1
  %305 = insertelement <4 x float> %304, float %253, i64 2
  %306 = insertelement <4 x float> %305, float %194, i64 3
  %307 = fadd contract <4 x float> %306, %303
  %308 = insertelement <4 x float> <float -0.000000e+00, float poison, float poison, float poison>, float %276, i64 1
  %309 = insertelement <4 x float> %308, float %255, i64 2
  %310 = insertelement <4 x float> %309, float %196, i64 3
  %311 = fadd contract <4 x float> %310, %307
  %312 = insertelement <4 x float> <float -0.000000e+00, float poison, float poison, float poison>, float %278, i64 1
  %313 = insertelement <4 x float> %312, float %257, i64 2
  %314 = insertelement <4 x float> %313, float %198, i64 3
  %315 = fadd contract <4 x float> %314, %311
  %316 = insertelement <4 x float> <float -0.000000e+00, float poison, float poison, float poison>, float %280, i64 1
  %317 = insertelement <4 x float> %316, float %259, i64 2
  %318 = insertelement <4 x float> %317, float %200, i64 3
  %319 = fadd contract <4 x float> %318, %315
  %320 = insertelement <4 x float> <float -0.000000e+00, float poison, float poison, float poison>, float %282, i64 1
  %321 = insertelement <4 x float> %320, float %261, i64 2
  %322 = insertelement <4 x float> %321, float %202, i64 3
  %323 = fadd contract <4 x float> %322, %319
  %324 = insertelement <4 x float> <float -0.000000e+00, float poison, float poison, float poison>, float %284, i64 1
  %325 = insertelement <4 x float> %324, float %263, i64 2
  %326 = insertelement <4 x float> %325, float %204, i64 3
  %327 = fadd contract <4 x float> %326, %323
  %328 = insertelement <4 x float> <float -0.000000e+00, float poison, float poison, float poison>, float %286, i64 1
  %329 = insertelement <4 x float> %328, float %265, i64 2
  %330 = insertelement <4 x float> %329, float %206, i64 3
  %331 = fadd contract <4 x float> %330, %327
  %332 = load float, ptr %289, align 4, !tbaa !39
  %333 = insertelement <4 x float> poison, float %332, i64 0
  %334 = insertelement <4 x float> %333, float %288, i64 1
  %335 = insertelement <4 x float> %334, float %267, i64 2
  %336 = insertelement <4 x float> %335, float %242, i64 3
  %337 = fadd contract <4 x float> %331, %336
  %338 = tail call contract float @llvm.vector.reduce.fadd.v4f32(float -0.000000e+00, <4 x float> %337)
  %339 = fmul contract float %169, %338
  br label %340

340:                                              ; preds = %178, %176
  %341 = phi float [ %339, %178 ], [ 0.000000e+00, %176 ]
  %342 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 8192), i32 %21
  store float %341, ptr addrspace(3) %342, align 4, !tbaa !39
  br label %343

343:                                              ; preds = %340, %168
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %344 = icmp samesign ult i32 %21, 256
  br i1 %344, label %345, label %1436

345:                                              ; preds = %343
  %346 = icmp sgt i32 %34, 0
  br i1 %346, label %347, label %357

347:                                              ; preds = %345
  %348 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 8192), i32 %170
  %349 = load float, ptr addrspace(3) %348, align 4, !tbaa !39
  %350 = fmul contract float %171, %349
  %351 = fadd contract float %174, %350
  %352 = fmul contract float %351, f0xBFB8AA3B
  %353 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %352)
  %354 = fadd contract float %353, 1.000000e+00
  %355 = fdiv contract float 1.000000e+00, %354
  %356 = fadd contract float %355, f0x358637BD
  br label %357

357:                                              ; preds = %347, %345
  %358 = phi float [ %356, %347 ], [ 0.000000e+00, %345 ]
  %359 = shl nsw i64 %25, 15
  %360 = getelementptr inbounds i8, ptr %18, i64 %359
  %361 = shl i32 %34, 15
  %362 = zext i32 %361 to i64
  %363 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %360, i16 0, i64 %362, i32 822099968)
  %364 = trunc i64 %10 to i32
  %365 = inttoptr i32 %364 to ptr addrspace(3)
  %366 = shl nuw nsw i32 %21, 3
  %367 = and i32 %366, 2016
  %368 = shl nuw nsw i32 %170, 13
  %369 = or disjoint i32 %368, %367
  %370 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %363, i32 %369, i32 0, i32 0)
  %371 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %363, i32 %369, i32 16, i32 0)
  %372 = bitcast <4 x i32> %371 to <8 x bfloat>
  %373 = shufflevector <8 x bfloat> %372, <8 x bfloat> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %374 = or disjoint i32 %369, 2048
  %375 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %363, i32 %374, i32 0, i32 0)
  %376 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %363, i32 %374, i32 16, i32 0)
  %377 = bitcast <4 x i32> %376 to <8 x bfloat>
  %378 = shufflevector <8 x bfloat> %377, <8 x bfloat> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %379 = icmp eq i32 %170, 0
  %380 = shl nuw nsw i32 %21, 3
  %381 = getelementptr inbounds nuw i8, ptr addrspace(3) %365, i32 %380
  %382 = getelementptr inbounds nuw i8, ptr addrspace(3) %381, i32 2048
  %383 = shufflevector <4 x i32> %370, <4 x i32> %371, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %384 = bitcast <8 x i32> %383 to <16 x bfloat>
  %385 = fpext <16 x bfloat> %384 to <16 x float>
  %386 = insertelement <16 x float> poison, float %358, i64 0
  %387 = shufflevector <16 x float> %386, <16 x float> poison, <16 x i32> zeroinitializer
  %388 = fmul contract <16 x float> %387, %385
  %389 = bitcast <16 x float> %388 to <16 x i32>
  %390 = extractelement <16 x i32> %389, i64 0
  %391 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %390, i32 177, i32 15, i32 15, i1 false)
  %392 = extractelement <16 x i32> %389, i64 1
  %393 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %392, i32 177, i32 15, i32 15, i1 false)
  %394 = bitcast <16 x float> %388 to <16 x i32>
  %395 = extractelement <16 x i32> %394, i64 2
  %396 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %395, i32 177, i32 15, i32 15, i1 false)
  %397 = extractelement <16 x i32> %394, i64 3
  %398 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %397, i32 177, i32 15, i32 15, i1 false)
  %399 = bitcast <16 x float> %388 to <16 x i32>
  %400 = extractelement <16 x i32> %399, i64 4
  %401 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %400, i32 177, i32 15, i32 15, i1 false)
  %402 = extractelement <16 x i32> %399, i64 5
  %403 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %402, i32 177, i32 15, i32 15, i1 false)
  %404 = bitcast <16 x float> %388 to <16 x i32>
  %405 = extractelement <16 x i32> %404, i64 6
  %406 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %405, i32 177, i32 15, i32 15, i1 false)
  %407 = extractelement <16 x i32> %404, i64 7
  %408 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %407, i32 177, i32 15, i32 15, i1 false)
  %409 = bitcast <16 x float> %388 to <16 x i32>
  %410 = extractelement <16 x i32> %409, i64 8
  %411 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %410, i32 177, i32 15, i32 15, i1 false)
  %412 = extractelement <16 x i32> %409, i64 9
  %413 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %412, i32 177, i32 15, i32 15, i1 false)
  %414 = bitcast <16 x float> %388 to <16 x i32>
  %415 = extractelement <16 x i32> %414, i64 10
  %416 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %415, i32 177, i32 15, i32 15, i1 false)
  %417 = extractelement <16 x i32> %414, i64 11
  %418 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %417, i32 177, i32 15, i32 15, i1 false)
  %419 = bitcast <16 x float> %388 to <16 x i32>
  %420 = extractelement <16 x i32> %419, i64 12
  %421 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %420, i32 177, i32 15, i32 15, i1 false)
  %422 = extractelement <16 x i32> %419, i64 13
  %423 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %422, i32 177, i32 15, i32 15, i1 false)
  %424 = bitcast <16 x float> %388 to <16 x i32>
  %425 = extractelement <16 x i32> %424, i64 14
  %426 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %425, i32 177, i32 15, i32 15, i1 false)
  %427 = extractelement <16 x i32> %424, i64 15
  %428 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %427, i32 177, i32 15, i32 15, i1 false)
  %429 = insertelement <16 x i32> poison, i32 %391, i64 0
  %430 = insertelement <16 x i32> %429, i32 %393, i64 1
  %431 = insertelement <16 x i32> %430, i32 %396, i64 2
  %432 = insertelement <16 x i32> %431, i32 %398, i64 3
  %433 = insertelement <16 x i32> %432, i32 %401, i64 4
  %434 = insertelement <16 x i32> %433, i32 %403, i64 5
  %435 = insertelement <16 x i32> %434, i32 %406, i64 6
  %436 = insertelement <16 x i32> %435, i32 %408, i64 7
  %437 = insertelement <16 x i32> %436, i32 %411, i64 8
  %438 = insertelement <16 x i32> %437, i32 %413, i64 9
  %439 = insertelement <16 x i32> %438, i32 %416, i64 10
  %440 = insertelement <16 x i32> %439, i32 %418, i64 11
  %441 = insertelement <16 x i32> %440, i32 %421, i64 12
  %442 = insertelement <16 x i32> %441, i32 %423, i64 13
  %443 = insertelement <16 x i32> %442, i32 %426, i64 14
  %444 = insertelement <16 x i32> %443, i32 %428, i64 15
  %445 = bitcast <16 x i32> %444 to <16 x float>
  %446 = fadd contract <16 x float> %388, %445
  %447 = bitcast <16 x float> %446 to <16 x i32>
  %448 = extractelement <16 x i32> %447, i64 0
  %449 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %448, i32 78, i32 15, i32 15, i1 false)
  %450 = extractelement <16 x i32> %447, i64 1
  %451 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %450, i32 78, i32 15, i32 15, i1 false)
  %452 = bitcast <16 x float> %446 to <16 x i32>
  %453 = extractelement <16 x i32> %452, i64 2
  %454 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %453, i32 78, i32 15, i32 15, i1 false)
  %455 = extractelement <16 x i32> %452, i64 3
  %456 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %455, i32 78, i32 15, i32 15, i1 false)
  %457 = bitcast <16 x float> %446 to <16 x i32>
  %458 = extractelement <16 x i32> %457, i64 4
  %459 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %458, i32 78, i32 15, i32 15, i1 false)
  %460 = extractelement <16 x i32> %457, i64 5
  %461 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %460, i32 78, i32 15, i32 15, i1 false)
  %462 = bitcast <16 x float> %446 to <16 x i32>
  %463 = extractelement <16 x i32> %462, i64 6
  %464 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %463, i32 78, i32 15, i32 15, i1 false)
  %465 = extractelement <16 x i32> %462, i64 7
  %466 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %465, i32 78, i32 15, i32 15, i1 false)
  %467 = bitcast <16 x float> %446 to <16 x i32>
  %468 = extractelement <16 x i32> %467, i64 8
  %469 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %468, i32 78, i32 15, i32 15, i1 false)
  %470 = extractelement <16 x i32> %467, i64 9
  %471 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %470, i32 78, i32 15, i32 15, i1 false)
  %472 = bitcast <16 x float> %446 to <16 x i32>
  %473 = extractelement <16 x i32> %472, i64 10
  %474 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %473, i32 78, i32 15, i32 15, i1 false)
  %475 = extractelement <16 x i32> %472, i64 11
  %476 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %475, i32 78, i32 15, i32 15, i1 false)
  %477 = bitcast <16 x float> %446 to <16 x i32>
  %478 = extractelement <16 x i32> %477, i64 12
  %479 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %478, i32 78, i32 15, i32 15, i1 false)
  %480 = extractelement <16 x i32> %477, i64 13
  %481 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %480, i32 78, i32 15, i32 15, i1 false)
  %482 = bitcast <16 x float> %446 to <16 x i32>
  %483 = extractelement <16 x i32> %482, i64 14
  %484 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %483, i32 78, i32 15, i32 15, i1 false)
  %485 = extractelement <16 x i32> %482, i64 15
  %486 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %485, i32 78, i32 15, i32 15, i1 false)
  %487 = insertelement <16 x i32> poison, i32 %449, i64 0
  %488 = insertelement <16 x i32> %487, i32 %451, i64 1
  %489 = insertelement <16 x i32> %488, i32 %454, i64 2
  %490 = insertelement <16 x i32> %489, i32 %456, i64 3
  %491 = insertelement <16 x i32> %490, i32 %459, i64 4
  %492 = insertelement <16 x i32> %491, i32 %461, i64 5
  %493 = insertelement <16 x i32> %492, i32 %464, i64 6
  %494 = insertelement <16 x i32> %493, i32 %466, i64 7
  %495 = insertelement <16 x i32> %494, i32 %469, i64 8
  %496 = insertelement <16 x i32> %495, i32 %471, i64 9
  %497 = insertelement <16 x i32> %496, i32 %474, i64 10
  %498 = insertelement <16 x i32> %497, i32 %476, i64 11
  %499 = insertelement <16 x i32> %498, i32 %479, i64 12
  %500 = insertelement <16 x i32> %499, i32 %481, i64 13
  %501 = insertelement <16 x i32> %500, i32 %484, i64 14
  %502 = insertelement <16 x i32> %501, i32 %486, i64 15
  %503 = bitcast <16 x i32> %502 to <16 x float>
  %504 = fadd contract <16 x float> %446, %503
  %505 = extractelement <16 x float> %504, i64 0
  %506 = fptrunc contract float %505 to bfloat
  %507 = insertelement <16 x bfloat> %373, bfloat %506, i64 0
  %508 = extractelement <16 x float> %504, i64 1
  %509 = fptrunc contract float %508 to bfloat
  %510 = insertelement <16 x bfloat> %507, bfloat %509, i64 1
  %511 = extractelement <16 x float> %504, i64 2
  %512 = fptrunc contract float %511 to bfloat
  %513 = insertelement <16 x bfloat> %510, bfloat %512, i64 2
  %514 = extractelement <16 x float> %504, i64 3
  %515 = fptrunc contract float %514 to bfloat
  %516 = insertelement <16 x bfloat> %513, bfloat %515, i64 3
  %517 = extractelement <16 x float> %504, i64 4
  %518 = fptrunc contract float %517 to bfloat
  %519 = insertelement <16 x bfloat> %516, bfloat %518, i64 4
  %520 = extractelement <16 x float> %504, i64 5
  %521 = fptrunc contract float %520 to bfloat
  %522 = insertelement <16 x bfloat> %519, bfloat %521, i64 5
  %523 = extractelement <16 x float> %504, i64 6
  %524 = fptrunc contract float %523 to bfloat
  %525 = insertelement <16 x bfloat> %522, bfloat %524, i64 6
  %526 = extractelement <16 x float> %504, i64 7
  %527 = fptrunc contract float %526 to bfloat
  %528 = insertelement <16 x bfloat> %525, bfloat %527, i64 7
  %529 = extractelement <16 x float> %504, i64 8
  %530 = fptrunc contract float %529 to bfloat
  %531 = insertelement <16 x bfloat> %528, bfloat %530, i64 8
  %532 = extractelement <16 x float> %504, i64 9
  %533 = fptrunc contract float %532 to bfloat
  %534 = insertelement <16 x bfloat> %531, bfloat %533, i64 9
  %535 = extractelement <16 x float> %504, i64 10
  %536 = fptrunc contract float %535 to bfloat
  %537 = insertelement <16 x bfloat> %534, bfloat %536, i64 10
  %538 = extractelement <16 x float> %504, i64 11
  %539 = fptrunc contract float %538 to bfloat
  %540 = insertelement <16 x bfloat> %537, bfloat %539, i64 11
  %541 = extractelement <16 x float> %504, i64 12
  %542 = fptrunc contract float %541 to bfloat
  %543 = insertelement <16 x bfloat> %540, bfloat %542, i64 12
  %544 = extractelement <16 x float> %504, i64 13
  %545 = fptrunc contract float %544 to bfloat
  %546 = insertelement <16 x bfloat> %543, bfloat %545, i64 13
  %547 = extractelement <16 x float> %504, i64 14
  %548 = fptrunc contract float %547 to bfloat
  %549 = insertelement <16 x bfloat> %546, bfloat %548, i64 14
  %550 = extractelement <16 x float> %504, i64 15
  %551 = fptrunc contract float %550 to bfloat
  %552 = insertelement <16 x bfloat> %549, bfloat %551, i64 15
  br i1 %379, label %553, label %587

553:                                              ; preds = %357
  %554 = fmul contract <16 x float> %504, %504
  %555 = extractelement <16 x float> %554, i64 0
  %556 = extractelement <16 x float> %554, i64 1
  %557 = fadd contract float %555, %556
  %558 = extractelement <16 x float> %554, i64 2
  %559 = fadd contract float %557, %558
  %560 = extractelement <16 x float> %554, i64 3
  %561 = fadd contract float %559, %560
  %562 = extractelement <16 x float> %554, i64 4
  %563 = fadd contract float %561, %562
  %564 = extractelement <16 x float> %554, i64 5
  %565 = fadd contract float %563, %564
  %566 = extractelement <16 x float> %554, i64 6
  %567 = fadd contract float %565, %566
  %568 = extractelement <16 x float> %554, i64 7
  %569 = fadd contract float %567, %568
  %570 = extractelement <16 x float> %554, i64 8
  %571 = fadd contract float %569, %570
  %572 = extractelement <16 x float> %554, i64 9
  %573 = fadd contract float %571, %572
  %574 = extractelement <16 x float> %554, i64 10
  %575 = fadd contract float %573, %574
  %576 = extractelement <16 x float> %554, i64 11
  %577 = fadd contract float %575, %576
  %578 = extractelement <16 x float> %554, i64 12
  %579 = fadd contract float %577, %578
  %580 = extractelement <16 x float> %554, i64 13
  %581 = fadd contract float %579, %580
  %582 = extractelement <16 x float> %554, i64 14
  %583 = fadd contract float %581, %582
  %584 = extractelement <16 x float> %554, i64 15
  %585 = fadd contract float %583, %584
  br i1 %346, label %586, label %587

586:                                              ; preds = %553
  store <16 x bfloat> %552, ptr addrspace(3) %381, align 32, !tbaa !29
  br label %587

587:                                              ; preds = %586, %553, %357
  %588 = phi float [ %585, %586 ], [ %585, %553 ], [ 0.000000e+00, %357 ]
  %589 = or disjoint i32 %369, 4096
  %590 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %363, i32 %589, i32 0, i32 0)
  %591 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %363, i32 %589, i32 16, i32 0)
  %592 = bitcast <4 x i32> %591 to <8 x bfloat>
  %593 = shufflevector <8 x bfloat> %592, <8 x bfloat> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %594 = shufflevector <4 x i32> %375, <4 x i32> %376, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %595 = bitcast <8 x i32> %594 to <16 x bfloat>
  %596 = fpext <16 x bfloat> %595 to <16 x float>
  %597 = fmul contract <16 x float> %387, %596
  %598 = bitcast <16 x float> %597 to <16 x i32>
  %599 = extractelement <16 x i32> %598, i64 0
  %600 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %599, i32 177, i32 15, i32 15, i1 false)
  %601 = extractelement <16 x i32> %598, i64 1
  %602 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %601, i32 177, i32 15, i32 15, i1 false)
  %603 = bitcast <16 x float> %597 to <16 x i32>
  %604 = extractelement <16 x i32> %603, i64 2
  %605 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %604, i32 177, i32 15, i32 15, i1 false)
  %606 = extractelement <16 x i32> %603, i64 3
  %607 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %606, i32 177, i32 15, i32 15, i1 false)
  %608 = bitcast <16 x float> %597 to <16 x i32>
  %609 = extractelement <16 x i32> %608, i64 4
  %610 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %609, i32 177, i32 15, i32 15, i1 false)
  %611 = extractelement <16 x i32> %608, i64 5
  %612 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %611, i32 177, i32 15, i32 15, i1 false)
  %613 = bitcast <16 x float> %597 to <16 x i32>
  %614 = extractelement <16 x i32> %613, i64 6
  %615 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %614, i32 177, i32 15, i32 15, i1 false)
  %616 = extractelement <16 x i32> %613, i64 7
  %617 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %616, i32 177, i32 15, i32 15, i1 false)
  %618 = bitcast <16 x float> %597 to <16 x i32>
  %619 = extractelement <16 x i32> %618, i64 8
  %620 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %619, i32 177, i32 15, i32 15, i1 false)
  %621 = extractelement <16 x i32> %618, i64 9
  %622 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %621, i32 177, i32 15, i32 15, i1 false)
  %623 = bitcast <16 x float> %597 to <16 x i32>
  %624 = extractelement <16 x i32> %623, i64 10
  %625 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %624, i32 177, i32 15, i32 15, i1 false)
  %626 = extractelement <16 x i32> %623, i64 11
  %627 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %626, i32 177, i32 15, i32 15, i1 false)
  %628 = bitcast <16 x float> %597 to <16 x i32>
  %629 = extractelement <16 x i32> %628, i64 12
  %630 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %629, i32 177, i32 15, i32 15, i1 false)
  %631 = extractelement <16 x i32> %628, i64 13
  %632 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %631, i32 177, i32 15, i32 15, i1 false)
  %633 = bitcast <16 x float> %597 to <16 x i32>
  %634 = extractelement <16 x i32> %633, i64 14
  %635 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %634, i32 177, i32 15, i32 15, i1 false)
  %636 = extractelement <16 x i32> %633, i64 15
  %637 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %636, i32 177, i32 15, i32 15, i1 false)
  %638 = insertelement <16 x i32> poison, i32 %600, i64 0
  %639 = insertelement <16 x i32> %638, i32 %602, i64 1
  %640 = insertelement <16 x i32> %639, i32 %605, i64 2
  %641 = insertelement <16 x i32> %640, i32 %607, i64 3
  %642 = insertelement <16 x i32> %641, i32 %610, i64 4
  %643 = insertelement <16 x i32> %642, i32 %612, i64 5
  %644 = insertelement <16 x i32> %643, i32 %615, i64 6
  %645 = insertelement <16 x i32> %644, i32 %617, i64 7
  %646 = insertelement <16 x i32> %645, i32 %620, i64 8
  %647 = insertelement <16 x i32> %646, i32 %622, i64 9
  %648 = insertelement <16 x i32> %647, i32 %625, i64 10
  %649 = insertelement <16 x i32> %648, i32 %627, i64 11
  %650 = insertelement <16 x i32> %649, i32 %630, i64 12
  %651 = insertelement <16 x i32> %650, i32 %632, i64 13
  %652 = insertelement <16 x i32> %651, i32 %635, i64 14
  %653 = insertelement <16 x i32> %652, i32 %637, i64 15
  %654 = bitcast <16 x i32> %653 to <16 x float>
  %655 = fadd contract <16 x float> %597, %654
  %656 = bitcast <16 x float> %655 to <16 x i32>
  %657 = extractelement <16 x i32> %656, i64 0
  %658 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %657, i32 78, i32 15, i32 15, i1 false)
  %659 = extractelement <16 x i32> %656, i64 1
  %660 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %659, i32 78, i32 15, i32 15, i1 false)
  %661 = bitcast <16 x float> %655 to <16 x i32>
  %662 = extractelement <16 x i32> %661, i64 2
  %663 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %662, i32 78, i32 15, i32 15, i1 false)
  %664 = extractelement <16 x i32> %661, i64 3
  %665 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %664, i32 78, i32 15, i32 15, i1 false)
  %666 = bitcast <16 x float> %655 to <16 x i32>
  %667 = extractelement <16 x i32> %666, i64 4
  %668 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %667, i32 78, i32 15, i32 15, i1 false)
  %669 = extractelement <16 x i32> %666, i64 5
  %670 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %669, i32 78, i32 15, i32 15, i1 false)
  %671 = bitcast <16 x float> %655 to <16 x i32>
  %672 = extractelement <16 x i32> %671, i64 6
  %673 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %672, i32 78, i32 15, i32 15, i1 false)
  %674 = extractelement <16 x i32> %671, i64 7
  %675 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %674, i32 78, i32 15, i32 15, i1 false)
  %676 = bitcast <16 x float> %655 to <16 x i32>
  %677 = extractelement <16 x i32> %676, i64 8
  %678 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %677, i32 78, i32 15, i32 15, i1 false)
  %679 = extractelement <16 x i32> %676, i64 9
  %680 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %679, i32 78, i32 15, i32 15, i1 false)
  %681 = bitcast <16 x float> %655 to <16 x i32>
  %682 = extractelement <16 x i32> %681, i64 10
  %683 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %682, i32 78, i32 15, i32 15, i1 false)
  %684 = extractelement <16 x i32> %681, i64 11
  %685 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %684, i32 78, i32 15, i32 15, i1 false)
  %686 = bitcast <16 x float> %655 to <16 x i32>
  %687 = extractelement <16 x i32> %686, i64 12
  %688 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %687, i32 78, i32 15, i32 15, i1 false)
  %689 = extractelement <16 x i32> %686, i64 13
  %690 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %689, i32 78, i32 15, i32 15, i1 false)
  %691 = bitcast <16 x float> %655 to <16 x i32>
  %692 = extractelement <16 x i32> %691, i64 14
  %693 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %692, i32 78, i32 15, i32 15, i1 false)
  %694 = extractelement <16 x i32> %691, i64 15
  %695 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %694, i32 78, i32 15, i32 15, i1 false)
  %696 = insertelement <16 x i32> poison, i32 %658, i64 0
  %697 = insertelement <16 x i32> %696, i32 %660, i64 1
  %698 = insertelement <16 x i32> %697, i32 %663, i64 2
  %699 = insertelement <16 x i32> %698, i32 %665, i64 3
  %700 = insertelement <16 x i32> %699, i32 %668, i64 4
  %701 = insertelement <16 x i32> %700, i32 %670, i64 5
  %702 = insertelement <16 x i32> %701, i32 %673, i64 6
  %703 = insertelement <16 x i32> %702, i32 %675, i64 7
  %704 = insertelement <16 x i32> %703, i32 %678, i64 8
  %705 = insertelement <16 x i32> %704, i32 %680, i64 9
  %706 = insertelement <16 x i32> %705, i32 %683, i64 10
  %707 = insertelement <16 x i32> %706, i32 %685, i64 11
  %708 = insertelement <16 x i32> %707, i32 %688, i64 12
  %709 = insertelement <16 x i32> %708, i32 %690, i64 13
  %710 = insertelement <16 x i32> %709, i32 %693, i64 14
  %711 = insertelement <16 x i32> %710, i32 %695, i64 15
  %712 = bitcast <16 x i32> %711 to <16 x float>
  %713 = fadd contract <16 x float> %655, %712
  %714 = extractelement <16 x float> %713, i64 0
  %715 = fptrunc contract float %714 to bfloat
  %716 = insertelement <16 x bfloat> %378, bfloat %715, i64 0
  %717 = extractelement <16 x float> %713, i64 1
  %718 = fptrunc contract float %717 to bfloat
  %719 = insertelement <16 x bfloat> %716, bfloat %718, i64 1
  %720 = extractelement <16 x float> %713, i64 2
  %721 = fptrunc contract float %720 to bfloat
  %722 = insertelement <16 x bfloat> %719, bfloat %721, i64 2
  %723 = extractelement <16 x float> %713, i64 3
  %724 = fptrunc contract float %723 to bfloat
  %725 = insertelement <16 x bfloat> %722, bfloat %724, i64 3
  %726 = extractelement <16 x float> %713, i64 4
  %727 = fptrunc contract float %726 to bfloat
  %728 = insertelement <16 x bfloat> %725, bfloat %727, i64 4
  %729 = extractelement <16 x float> %713, i64 5
  %730 = fptrunc contract float %729 to bfloat
  %731 = insertelement <16 x bfloat> %728, bfloat %730, i64 5
  %732 = extractelement <16 x float> %713, i64 6
  %733 = fptrunc contract float %732 to bfloat
  %734 = insertelement <16 x bfloat> %731, bfloat %733, i64 6
  %735 = extractelement <16 x float> %713, i64 7
  %736 = fptrunc contract float %735 to bfloat
  %737 = insertelement <16 x bfloat> %734, bfloat %736, i64 7
  %738 = extractelement <16 x float> %713, i64 8
  %739 = fptrunc contract float %738 to bfloat
  %740 = insertelement <16 x bfloat> %737, bfloat %739, i64 8
  %741 = extractelement <16 x float> %713, i64 9
  %742 = fptrunc contract float %741 to bfloat
  %743 = insertelement <16 x bfloat> %740, bfloat %742, i64 9
  %744 = extractelement <16 x float> %713, i64 10
  %745 = fptrunc contract float %744 to bfloat
  %746 = insertelement <16 x bfloat> %743, bfloat %745, i64 10
  %747 = extractelement <16 x float> %713, i64 11
  %748 = fptrunc contract float %747 to bfloat
  %749 = insertelement <16 x bfloat> %746, bfloat %748, i64 11
  %750 = extractelement <16 x float> %713, i64 12
  %751 = fptrunc contract float %750 to bfloat
  %752 = insertelement <16 x bfloat> %749, bfloat %751, i64 12
  %753 = extractelement <16 x float> %713, i64 13
  %754 = fptrunc contract float %753 to bfloat
  %755 = insertelement <16 x bfloat> %752, bfloat %754, i64 13
  %756 = extractelement <16 x float> %713, i64 14
  %757 = fptrunc contract float %756 to bfloat
  %758 = insertelement <16 x bfloat> %755, bfloat %757, i64 14
  %759 = extractelement <16 x float> %713, i64 15
  %760 = fptrunc contract float %759 to bfloat
  %761 = insertelement <16 x bfloat> %758, bfloat %760, i64 15
  br i1 %379, label %762, label %797

762:                                              ; preds = %587
  %763 = fmul contract <16 x float> %713, %713
  %764 = extractelement <16 x float> %763, i64 0
  %765 = fadd contract float %588, %764
  %766 = extractelement <16 x float> %763, i64 1
  %767 = fadd contract float %765, %766
  %768 = extractelement <16 x float> %763, i64 2
  %769 = fadd contract float %767, %768
  %770 = extractelement <16 x float> %763, i64 3
  %771 = fadd contract float %769, %770
  %772 = extractelement <16 x float> %763, i64 4
  %773 = fadd contract float %771, %772
  %774 = extractelement <16 x float> %763, i64 5
  %775 = fadd contract float %773, %774
  %776 = extractelement <16 x float> %763, i64 6
  %777 = fadd contract float %775, %776
  %778 = extractelement <16 x float> %763, i64 7
  %779 = fadd contract float %777, %778
  %780 = extractelement <16 x float> %763, i64 8
  %781 = fadd contract float %779, %780
  %782 = extractelement <16 x float> %763, i64 9
  %783 = fadd contract float %781, %782
  %784 = extractelement <16 x float> %763, i64 10
  %785 = fadd contract float %783, %784
  %786 = extractelement <16 x float> %763, i64 11
  %787 = fadd contract float %785, %786
  %788 = extractelement <16 x float> %763, i64 12
  %789 = fadd contract float %787, %788
  %790 = extractelement <16 x float> %763, i64 13
  %791 = fadd contract float %789, %790
  %792 = extractelement <16 x float> %763, i64 14
  %793 = fadd contract float %791, %792
  %794 = extractelement <16 x float> %763, i64 15
  %795 = fadd contract float %793, %794
  br i1 %346, label %796, label %797

796:                                              ; preds = %762
  store <16 x bfloat> %761, ptr addrspace(3) %382, align 32, !tbaa !29
  br label %797

797:                                              ; preds = %796, %762, %587
  %798 = phi float [ %795, %796 ], [ %795, %762 ], [ %588, %587 ]
  %799 = or disjoint i32 %369, 6144
  %800 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %363, i32 %799, i32 0, i32 0)
  %801 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %363, i32 %799, i32 16, i32 0)
  %802 = bitcast <4 x i32> %801 to <8 x bfloat>
  %803 = shufflevector <8 x bfloat> %802, <8 x bfloat> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %804 = shufflevector <4 x i32> %590, <4 x i32> %591, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %805 = bitcast <8 x i32> %804 to <16 x bfloat>
  %806 = fpext <16 x bfloat> %805 to <16 x float>
  %807 = fmul contract <16 x float> %387, %806
  %808 = bitcast <16 x float> %807 to <16 x i32>
  %809 = extractelement <16 x i32> %808, i64 0
  %810 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %809, i32 177, i32 15, i32 15, i1 false)
  %811 = extractelement <16 x i32> %808, i64 1
  %812 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %811, i32 177, i32 15, i32 15, i1 false)
  %813 = bitcast <16 x float> %807 to <16 x i32>
  %814 = extractelement <16 x i32> %813, i64 2
  %815 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %814, i32 177, i32 15, i32 15, i1 false)
  %816 = extractelement <16 x i32> %813, i64 3
  %817 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %816, i32 177, i32 15, i32 15, i1 false)
  %818 = bitcast <16 x float> %807 to <16 x i32>
  %819 = extractelement <16 x i32> %818, i64 4
  %820 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %819, i32 177, i32 15, i32 15, i1 false)
  %821 = extractelement <16 x i32> %818, i64 5
  %822 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %821, i32 177, i32 15, i32 15, i1 false)
  %823 = bitcast <16 x float> %807 to <16 x i32>
  %824 = extractelement <16 x i32> %823, i64 6
  %825 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %824, i32 177, i32 15, i32 15, i1 false)
  %826 = extractelement <16 x i32> %823, i64 7
  %827 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %826, i32 177, i32 15, i32 15, i1 false)
  %828 = bitcast <16 x float> %807 to <16 x i32>
  %829 = extractelement <16 x i32> %828, i64 8
  %830 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %829, i32 177, i32 15, i32 15, i1 false)
  %831 = extractelement <16 x i32> %828, i64 9
  %832 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %831, i32 177, i32 15, i32 15, i1 false)
  %833 = bitcast <16 x float> %807 to <16 x i32>
  %834 = extractelement <16 x i32> %833, i64 10
  %835 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %834, i32 177, i32 15, i32 15, i1 false)
  %836 = extractelement <16 x i32> %833, i64 11
  %837 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %836, i32 177, i32 15, i32 15, i1 false)
  %838 = bitcast <16 x float> %807 to <16 x i32>
  %839 = extractelement <16 x i32> %838, i64 12
  %840 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %839, i32 177, i32 15, i32 15, i1 false)
  %841 = extractelement <16 x i32> %838, i64 13
  %842 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %841, i32 177, i32 15, i32 15, i1 false)
  %843 = bitcast <16 x float> %807 to <16 x i32>
  %844 = extractelement <16 x i32> %843, i64 14
  %845 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %844, i32 177, i32 15, i32 15, i1 false)
  %846 = extractelement <16 x i32> %843, i64 15
  %847 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %846, i32 177, i32 15, i32 15, i1 false)
  %848 = insertelement <16 x i32> poison, i32 %810, i64 0
  %849 = insertelement <16 x i32> %848, i32 %812, i64 1
  %850 = insertelement <16 x i32> %849, i32 %815, i64 2
  %851 = insertelement <16 x i32> %850, i32 %817, i64 3
  %852 = insertelement <16 x i32> %851, i32 %820, i64 4
  %853 = insertelement <16 x i32> %852, i32 %822, i64 5
  %854 = insertelement <16 x i32> %853, i32 %825, i64 6
  %855 = insertelement <16 x i32> %854, i32 %827, i64 7
  %856 = insertelement <16 x i32> %855, i32 %830, i64 8
  %857 = insertelement <16 x i32> %856, i32 %832, i64 9
  %858 = insertelement <16 x i32> %857, i32 %835, i64 10
  %859 = insertelement <16 x i32> %858, i32 %837, i64 11
  %860 = insertelement <16 x i32> %859, i32 %840, i64 12
  %861 = insertelement <16 x i32> %860, i32 %842, i64 13
  %862 = insertelement <16 x i32> %861, i32 %845, i64 14
  %863 = insertelement <16 x i32> %862, i32 %847, i64 15
  %864 = bitcast <16 x i32> %863 to <16 x float>
  %865 = fadd contract <16 x float> %807, %864
  %866 = bitcast <16 x float> %865 to <16 x i32>
  %867 = extractelement <16 x i32> %866, i64 0
  %868 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %867, i32 78, i32 15, i32 15, i1 false)
  %869 = extractelement <16 x i32> %866, i64 1
  %870 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %869, i32 78, i32 15, i32 15, i1 false)
  %871 = bitcast <16 x float> %865 to <16 x i32>
  %872 = extractelement <16 x i32> %871, i64 2
  %873 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %872, i32 78, i32 15, i32 15, i1 false)
  %874 = extractelement <16 x i32> %871, i64 3
  %875 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %874, i32 78, i32 15, i32 15, i1 false)
  %876 = bitcast <16 x float> %865 to <16 x i32>
  %877 = extractelement <16 x i32> %876, i64 4
  %878 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %877, i32 78, i32 15, i32 15, i1 false)
  %879 = extractelement <16 x i32> %876, i64 5
  %880 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %879, i32 78, i32 15, i32 15, i1 false)
  %881 = bitcast <16 x float> %865 to <16 x i32>
  %882 = extractelement <16 x i32> %881, i64 6
  %883 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %882, i32 78, i32 15, i32 15, i1 false)
  %884 = extractelement <16 x i32> %881, i64 7
  %885 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %884, i32 78, i32 15, i32 15, i1 false)
  %886 = bitcast <16 x float> %865 to <16 x i32>
  %887 = extractelement <16 x i32> %886, i64 8
  %888 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %887, i32 78, i32 15, i32 15, i1 false)
  %889 = extractelement <16 x i32> %886, i64 9
  %890 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %889, i32 78, i32 15, i32 15, i1 false)
  %891 = bitcast <16 x float> %865 to <16 x i32>
  %892 = extractelement <16 x i32> %891, i64 10
  %893 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %892, i32 78, i32 15, i32 15, i1 false)
  %894 = extractelement <16 x i32> %891, i64 11
  %895 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %894, i32 78, i32 15, i32 15, i1 false)
  %896 = bitcast <16 x float> %865 to <16 x i32>
  %897 = extractelement <16 x i32> %896, i64 12
  %898 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %897, i32 78, i32 15, i32 15, i1 false)
  %899 = extractelement <16 x i32> %896, i64 13
  %900 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %899, i32 78, i32 15, i32 15, i1 false)
  %901 = bitcast <16 x float> %865 to <16 x i32>
  %902 = extractelement <16 x i32> %901, i64 14
  %903 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %902, i32 78, i32 15, i32 15, i1 false)
  %904 = extractelement <16 x i32> %901, i64 15
  %905 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %904, i32 78, i32 15, i32 15, i1 false)
  %906 = insertelement <16 x i32> poison, i32 %868, i64 0
  %907 = insertelement <16 x i32> %906, i32 %870, i64 1
  %908 = insertelement <16 x i32> %907, i32 %873, i64 2
  %909 = insertelement <16 x i32> %908, i32 %875, i64 3
  %910 = insertelement <16 x i32> %909, i32 %878, i64 4
  %911 = insertelement <16 x i32> %910, i32 %880, i64 5
  %912 = insertelement <16 x i32> %911, i32 %883, i64 6
  %913 = insertelement <16 x i32> %912, i32 %885, i64 7
  %914 = insertelement <16 x i32> %913, i32 %888, i64 8
  %915 = insertelement <16 x i32> %914, i32 %890, i64 9
  %916 = insertelement <16 x i32> %915, i32 %893, i64 10
  %917 = insertelement <16 x i32> %916, i32 %895, i64 11
  %918 = insertelement <16 x i32> %917, i32 %898, i64 12
  %919 = insertelement <16 x i32> %918, i32 %900, i64 13
  %920 = insertelement <16 x i32> %919, i32 %903, i64 14
  %921 = insertelement <16 x i32> %920, i32 %905, i64 15
  %922 = bitcast <16 x i32> %921 to <16 x float>
  %923 = fadd contract <16 x float> %865, %922
  %924 = extractelement <16 x float> %923, i64 0
  %925 = fptrunc contract float %924 to bfloat
  %926 = insertelement <16 x bfloat> %593, bfloat %925, i64 0
  %927 = extractelement <16 x float> %923, i64 1
  %928 = fptrunc contract float %927 to bfloat
  %929 = insertelement <16 x bfloat> %926, bfloat %928, i64 1
  %930 = extractelement <16 x float> %923, i64 2
  %931 = fptrunc contract float %930 to bfloat
  %932 = insertelement <16 x bfloat> %929, bfloat %931, i64 2
  %933 = extractelement <16 x float> %923, i64 3
  %934 = fptrunc contract float %933 to bfloat
  %935 = insertelement <16 x bfloat> %932, bfloat %934, i64 3
  %936 = extractelement <16 x float> %923, i64 4
  %937 = fptrunc contract float %936 to bfloat
  %938 = insertelement <16 x bfloat> %935, bfloat %937, i64 4
  %939 = extractelement <16 x float> %923, i64 5
  %940 = fptrunc contract float %939 to bfloat
  %941 = insertelement <16 x bfloat> %938, bfloat %940, i64 5
  %942 = extractelement <16 x float> %923, i64 6
  %943 = fptrunc contract float %942 to bfloat
  %944 = insertelement <16 x bfloat> %941, bfloat %943, i64 6
  %945 = extractelement <16 x float> %923, i64 7
  %946 = fptrunc contract float %945 to bfloat
  %947 = insertelement <16 x bfloat> %944, bfloat %946, i64 7
  %948 = extractelement <16 x float> %923, i64 8
  %949 = fptrunc contract float %948 to bfloat
  %950 = insertelement <16 x bfloat> %947, bfloat %949, i64 8
  %951 = extractelement <16 x float> %923, i64 9
  %952 = fptrunc contract float %951 to bfloat
  %953 = insertelement <16 x bfloat> %950, bfloat %952, i64 9
  %954 = extractelement <16 x float> %923, i64 10
  %955 = fptrunc contract float %954 to bfloat
  %956 = insertelement <16 x bfloat> %953, bfloat %955, i64 10
  %957 = extractelement <16 x float> %923, i64 11
  %958 = fptrunc contract float %957 to bfloat
  %959 = insertelement <16 x bfloat> %956, bfloat %958, i64 11
  %960 = extractelement <16 x float> %923, i64 12
  %961 = fptrunc contract float %960 to bfloat
  %962 = insertelement <16 x bfloat> %959, bfloat %961, i64 12
  %963 = extractelement <16 x float> %923, i64 13
  %964 = fptrunc contract float %963 to bfloat
  %965 = insertelement <16 x bfloat> %962, bfloat %964, i64 13
  %966 = extractelement <16 x float> %923, i64 14
  %967 = fptrunc contract float %966 to bfloat
  %968 = insertelement <16 x bfloat> %965, bfloat %967, i64 14
  %969 = extractelement <16 x float> %923, i64 15
  %970 = fptrunc contract float %969 to bfloat
  %971 = insertelement <16 x bfloat> %968, bfloat %970, i64 15
  br i1 %379, label %972, label %1009

972:                                              ; preds = %797
  %973 = fmul contract <16 x float> %923, %923
  %974 = extractelement <16 x float> %973, i64 0
  %975 = fadd contract float %798, %974
  %976 = extractelement <16 x float> %973, i64 1
  %977 = fadd contract float %975, %976
  %978 = extractelement <16 x float> %973, i64 2
  %979 = fadd contract float %977, %978
  %980 = extractelement <16 x float> %973, i64 3
  %981 = fadd contract float %979, %980
  %982 = extractelement <16 x float> %973, i64 4
  %983 = fadd contract float %981, %982
  %984 = extractelement <16 x float> %973, i64 5
  %985 = fadd contract float %983, %984
  %986 = extractelement <16 x float> %973, i64 6
  %987 = fadd contract float %985, %986
  %988 = extractelement <16 x float> %973, i64 7
  %989 = fadd contract float %987, %988
  %990 = extractelement <16 x float> %973, i64 8
  %991 = fadd contract float %989, %990
  %992 = extractelement <16 x float> %973, i64 9
  %993 = fadd contract float %991, %992
  %994 = extractelement <16 x float> %973, i64 10
  %995 = fadd contract float %993, %994
  %996 = extractelement <16 x float> %973, i64 11
  %997 = fadd contract float %995, %996
  %998 = extractelement <16 x float> %973, i64 12
  %999 = fadd contract float %997, %998
  %1000 = extractelement <16 x float> %973, i64 13
  %1001 = fadd contract float %999, %1000
  %1002 = extractelement <16 x float> %973, i64 14
  %1003 = fadd contract float %1001, %1002
  %1004 = extractelement <16 x float> %973, i64 15
  %1005 = fadd contract float %1003, %1004
  br i1 %346, label %1006, label %1009

1006:                                             ; preds = %972
  %1007 = getelementptr inbounds nuw i8, ptr addrspace(3) %365, i32 4096
  %1008 = getelementptr inbounds nuw i8, ptr addrspace(3) %1007, i32 %380
  store <16 x bfloat> %971, ptr addrspace(3) %1008, align 32, !tbaa !29
  br label %1009

1009:                                             ; preds = %1006, %972, %797
  %1010 = phi float [ %1005, %1006 ], [ %1005, %972 ], [ %798, %797 ]
  %1011 = shufflevector <4 x i32> %800, <4 x i32> %801, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1012 = bitcast <8 x i32> %1011 to <16 x bfloat>
  %1013 = fpext <16 x bfloat> %1012 to <16 x float>
  %1014 = fmul contract <16 x float> %387, %1013
  %1015 = bitcast <16 x float> %1014 to <16 x i32>
  %1016 = extractelement <16 x i32> %1015, i64 0
  %1017 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1016, i32 177, i32 15, i32 15, i1 false)
  %1018 = extractelement <16 x i32> %1015, i64 1
  %1019 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1018, i32 177, i32 15, i32 15, i1 false)
  %1020 = bitcast <16 x float> %1014 to <16 x i32>
  %1021 = extractelement <16 x i32> %1020, i64 2
  %1022 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1021, i32 177, i32 15, i32 15, i1 false)
  %1023 = extractelement <16 x i32> %1020, i64 3
  %1024 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1023, i32 177, i32 15, i32 15, i1 false)
  %1025 = bitcast <16 x float> %1014 to <16 x i32>
  %1026 = extractelement <16 x i32> %1025, i64 4
  %1027 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1026, i32 177, i32 15, i32 15, i1 false)
  %1028 = extractelement <16 x i32> %1025, i64 5
  %1029 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1028, i32 177, i32 15, i32 15, i1 false)
  %1030 = bitcast <16 x float> %1014 to <16 x i32>
  %1031 = extractelement <16 x i32> %1030, i64 6
  %1032 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1031, i32 177, i32 15, i32 15, i1 false)
  %1033 = extractelement <16 x i32> %1030, i64 7
  %1034 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1033, i32 177, i32 15, i32 15, i1 false)
  %1035 = bitcast <16 x float> %1014 to <16 x i32>
  %1036 = extractelement <16 x i32> %1035, i64 8
  %1037 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1036, i32 177, i32 15, i32 15, i1 false)
  %1038 = extractelement <16 x i32> %1035, i64 9
  %1039 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1038, i32 177, i32 15, i32 15, i1 false)
  %1040 = bitcast <16 x float> %1014 to <16 x i32>
  %1041 = extractelement <16 x i32> %1040, i64 10
  %1042 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1041, i32 177, i32 15, i32 15, i1 false)
  %1043 = extractelement <16 x i32> %1040, i64 11
  %1044 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1043, i32 177, i32 15, i32 15, i1 false)
  %1045 = bitcast <16 x float> %1014 to <16 x i32>
  %1046 = extractelement <16 x i32> %1045, i64 12
  %1047 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1046, i32 177, i32 15, i32 15, i1 false)
  %1048 = extractelement <16 x i32> %1045, i64 13
  %1049 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1048, i32 177, i32 15, i32 15, i1 false)
  %1050 = bitcast <16 x float> %1014 to <16 x i32>
  %1051 = extractelement <16 x i32> %1050, i64 14
  %1052 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1051, i32 177, i32 15, i32 15, i1 false)
  %1053 = extractelement <16 x i32> %1050, i64 15
  %1054 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1053, i32 177, i32 15, i32 15, i1 false)
  %1055 = insertelement <16 x i32> poison, i32 %1017, i64 0
  %1056 = insertelement <16 x i32> %1055, i32 %1019, i64 1
  %1057 = insertelement <16 x i32> %1056, i32 %1022, i64 2
  %1058 = insertelement <16 x i32> %1057, i32 %1024, i64 3
  %1059 = insertelement <16 x i32> %1058, i32 %1027, i64 4
  %1060 = insertelement <16 x i32> %1059, i32 %1029, i64 5
  %1061 = insertelement <16 x i32> %1060, i32 %1032, i64 6
  %1062 = insertelement <16 x i32> %1061, i32 %1034, i64 7
  %1063 = insertelement <16 x i32> %1062, i32 %1037, i64 8
  %1064 = insertelement <16 x i32> %1063, i32 %1039, i64 9
  %1065 = insertelement <16 x i32> %1064, i32 %1042, i64 10
  %1066 = insertelement <16 x i32> %1065, i32 %1044, i64 11
  %1067 = insertelement <16 x i32> %1066, i32 %1047, i64 12
  %1068 = insertelement <16 x i32> %1067, i32 %1049, i64 13
  %1069 = insertelement <16 x i32> %1068, i32 %1052, i64 14
  %1070 = insertelement <16 x i32> %1069, i32 %1054, i64 15
  %1071 = bitcast <16 x i32> %1070 to <16 x float>
  %1072 = fadd contract <16 x float> %1014, %1071
  %1073 = bitcast <16 x float> %1072 to <16 x i32>
  %1074 = extractelement <16 x i32> %1073, i64 0
  %1075 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1074, i32 78, i32 15, i32 15, i1 false)
  %1076 = extractelement <16 x i32> %1073, i64 1
  %1077 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1076, i32 78, i32 15, i32 15, i1 false)
  %1078 = bitcast <16 x float> %1072 to <16 x i32>
  %1079 = extractelement <16 x i32> %1078, i64 2
  %1080 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1079, i32 78, i32 15, i32 15, i1 false)
  %1081 = extractelement <16 x i32> %1078, i64 3
  %1082 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1081, i32 78, i32 15, i32 15, i1 false)
  %1083 = bitcast <16 x float> %1072 to <16 x i32>
  %1084 = extractelement <16 x i32> %1083, i64 4
  %1085 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1084, i32 78, i32 15, i32 15, i1 false)
  %1086 = extractelement <16 x i32> %1083, i64 5
  %1087 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1086, i32 78, i32 15, i32 15, i1 false)
  %1088 = bitcast <16 x float> %1072 to <16 x i32>
  %1089 = extractelement <16 x i32> %1088, i64 6
  %1090 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1089, i32 78, i32 15, i32 15, i1 false)
  %1091 = extractelement <16 x i32> %1088, i64 7
  %1092 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1091, i32 78, i32 15, i32 15, i1 false)
  %1093 = bitcast <16 x float> %1072 to <16 x i32>
  %1094 = extractelement <16 x i32> %1093, i64 8
  %1095 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1094, i32 78, i32 15, i32 15, i1 false)
  %1096 = extractelement <16 x i32> %1093, i64 9
  %1097 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1096, i32 78, i32 15, i32 15, i1 false)
  %1098 = bitcast <16 x float> %1072 to <16 x i32>
  %1099 = extractelement <16 x i32> %1098, i64 10
  %1100 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1099, i32 78, i32 15, i32 15, i1 false)
  %1101 = extractelement <16 x i32> %1098, i64 11
  %1102 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1101, i32 78, i32 15, i32 15, i1 false)
  %1103 = bitcast <16 x float> %1072 to <16 x i32>
  %1104 = extractelement <16 x i32> %1103, i64 12
  %1105 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1104, i32 78, i32 15, i32 15, i1 false)
  %1106 = extractelement <16 x i32> %1103, i64 13
  %1107 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1106, i32 78, i32 15, i32 15, i1 false)
  %1108 = bitcast <16 x float> %1072 to <16 x i32>
  %1109 = extractelement <16 x i32> %1108, i64 14
  %1110 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1109, i32 78, i32 15, i32 15, i1 false)
  %1111 = extractelement <16 x i32> %1108, i64 15
  %1112 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1111, i32 78, i32 15, i32 15, i1 false)
  %1113 = insertelement <16 x i32> poison, i32 %1075, i64 0
  %1114 = insertelement <16 x i32> %1113, i32 %1077, i64 1
  %1115 = insertelement <16 x i32> %1114, i32 %1080, i64 2
  %1116 = insertelement <16 x i32> %1115, i32 %1082, i64 3
  %1117 = insertelement <16 x i32> %1116, i32 %1085, i64 4
  %1118 = insertelement <16 x i32> %1117, i32 %1087, i64 5
  %1119 = insertelement <16 x i32> %1118, i32 %1090, i64 6
  %1120 = insertelement <16 x i32> %1119, i32 %1092, i64 7
  %1121 = insertelement <16 x i32> %1120, i32 %1095, i64 8
  %1122 = insertelement <16 x i32> %1121, i32 %1097, i64 9
  %1123 = insertelement <16 x i32> %1122, i32 %1100, i64 10
  %1124 = insertelement <16 x i32> %1123, i32 %1102, i64 11
  %1125 = insertelement <16 x i32> %1124, i32 %1105, i64 12
  %1126 = insertelement <16 x i32> %1125, i32 %1107, i64 13
  %1127 = insertelement <16 x i32> %1126, i32 %1110, i64 14
  %1128 = insertelement <16 x i32> %1127, i32 %1112, i64 15
  %1129 = bitcast <16 x i32> %1128 to <16 x float>
  %1130 = fadd contract <16 x float> %1072, %1129
  %1131 = extractelement <16 x float> %1130, i64 0
  %1132 = fptrunc contract float %1131 to bfloat
  %1133 = insertelement <16 x bfloat> %803, bfloat %1132, i64 0
  %1134 = extractelement <16 x float> %1130, i64 1
  %1135 = fptrunc contract float %1134 to bfloat
  %1136 = insertelement <16 x bfloat> %1133, bfloat %1135, i64 1
  %1137 = extractelement <16 x float> %1130, i64 2
  %1138 = fptrunc contract float %1137 to bfloat
  %1139 = insertelement <16 x bfloat> %1136, bfloat %1138, i64 2
  %1140 = extractelement <16 x float> %1130, i64 3
  %1141 = fptrunc contract float %1140 to bfloat
  %1142 = insertelement <16 x bfloat> %1139, bfloat %1141, i64 3
  %1143 = extractelement <16 x float> %1130, i64 4
  %1144 = fptrunc contract float %1143 to bfloat
  %1145 = insertelement <16 x bfloat> %1142, bfloat %1144, i64 4
  %1146 = extractelement <16 x float> %1130, i64 5
  %1147 = fptrunc contract float %1146 to bfloat
  %1148 = insertelement <16 x bfloat> %1145, bfloat %1147, i64 5
  %1149 = extractelement <16 x float> %1130, i64 6
  %1150 = fptrunc contract float %1149 to bfloat
  %1151 = insertelement <16 x bfloat> %1148, bfloat %1150, i64 6
  %1152 = extractelement <16 x float> %1130, i64 7
  %1153 = fptrunc contract float %1152 to bfloat
  %1154 = insertelement <16 x bfloat> %1151, bfloat %1153, i64 7
  %1155 = extractelement <16 x float> %1130, i64 8
  %1156 = fptrunc contract float %1155 to bfloat
  %1157 = insertelement <16 x bfloat> %1154, bfloat %1156, i64 8
  %1158 = extractelement <16 x float> %1130, i64 9
  %1159 = fptrunc contract float %1158 to bfloat
  %1160 = insertelement <16 x bfloat> %1157, bfloat %1159, i64 9
  %1161 = extractelement <16 x float> %1130, i64 10
  %1162 = fptrunc contract float %1161 to bfloat
  %1163 = insertelement <16 x bfloat> %1160, bfloat %1162, i64 10
  %1164 = extractelement <16 x float> %1130, i64 11
  %1165 = fptrunc contract float %1164 to bfloat
  %1166 = insertelement <16 x bfloat> %1163, bfloat %1165, i64 11
  %1167 = extractelement <16 x float> %1130, i64 12
  %1168 = fptrunc contract float %1167 to bfloat
  %1169 = insertelement <16 x bfloat> %1166, bfloat %1168, i64 12
  %1170 = extractelement <16 x float> %1130, i64 13
  %1171 = fptrunc contract float %1170 to bfloat
  %1172 = insertelement <16 x bfloat> %1169, bfloat %1171, i64 13
  %1173 = extractelement <16 x float> %1130, i64 14
  %1174 = fptrunc contract float %1173 to bfloat
  %1175 = insertelement <16 x bfloat> %1172, bfloat %1174, i64 14
  %1176 = extractelement <16 x float> %1130, i64 15
  %1177 = fptrunc contract float %1176 to bfloat
  %1178 = insertelement <16 x bfloat> %1175, bfloat %1177, i64 15
  br i1 %379, label %1179, label %1215

1179:                                             ; preds = %1009
  %1180 = fmul contract <16 x float> %1130, %1130
  %1181 = extractelement <16 x float> %1180, i64 0
  %1182 = fadd contract float %1010, %1181
  %1183 = extractelement <16 x float> %1180, i64 1
  %1184 = fadd contract float %1182, %1183
  %1185 = extractelement <16 x float> %1180, i64 2
  %1186 = fadd contract float %1184, %1185
  %1187 = extractelement <16 x float> %1180, i64 3
  %1188 = fadd contract float %1186, %1187
  %1189 = extractelement <16 x float> %1180, i64 4
  %1190 = fadd contract float %1188, %1189
  %1191 = extractelement <16 x float> %1180, i64 5
  %1192 = fadd contract float %1190, %1191
  %1193 = extractelement <16 x float> %1180, i64 6
  %1194 = fadd contract float %1192, %1193
  %1195 = extractelement <16 x float> %1180, i64 7
  %1196 = fadd contract float %1194, %1195
  %1197 = extractelement <16 x float> %1180, i64 8
  %1198 = fadd contract float %1196, %1197
  %1199 = extractelement <16 x float> %1180, i64 9
  %1200 = fadd contract float %1198, %1199
  %1201 = extractelement <16 x float> %1180, i64 10
  %1202 = fadd contract float %1200, %1201
  %1203 = extractelement <16 x float> %1180, i64 11
  %1204 = fadd contract float %1202, %1203
  %1205 = extractelement <16 x float> %1180, i64 12
  %1206 = fadd contract float %1204, %1205
  %1207 = extractelement <16 x float> %1180, i64 13
  %1208 = fadd contract float %1206, %1207
  %1209 = extractelement <16 x float> %1180, i64 14
  %1210 = fadd contract float %1208, %1209
  %1211 = extractelement <16 x float> %1180, i64 15
  %1212 = fadd contract float %1210, %1211
  br i1 %346, label %1213, label %1215

1213:                                             ; preds = %1179
  %1214 = getelementptr inbounds nuw i8, ptr addrspace(3) %381, i32 6144
  store <16 x bfloat> %1178, ptr addrspace(3) %1214, align 32, !tbaa !29
  br label %1215

1215:                                             ; preds = %1213, %1179, %1009
  %1216 = phi float [ %1212, %1213 ], [ %1212, %1179 ], [ %1010, %1009 ]
  %1217 = icmp ult i32 %22, 128
  br i1 %1217, label %1218, label %1235

1218:                                             ; preds = %1215
  %1219 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %19, i16 0, i64 8192, i32 822099968)
  %1220 = shl nuw nsw i32 %23, 11
  %1221 = shl nuw nsw i32 %32, 6
  %1222 = or disjoint i32 %1220, %1221
  %1223 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %1219, i32 %1222, i32 0, i32 0)
  %1224 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %1219, i32 %1222, i32 16, i32 0)
  %1225 = shufflevector <4 x i32> %1223, <4 x i32> %1224, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1226 = bitcast <16 x i32> %1225 to <32 x bfloat>
  %1227 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %1219, i32 %1222, i32 32, i32 0)
  %1228 = bitcast <4 x i32> %1227 to <8 x bfloat>
  %1229 = shufflevector <8 x bfloat> %1228, <8 x bfloat> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1230 = shufflevector <32 x bfloat> %1226, <32 x bfloat> %1229, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1231 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %1219, i32 %1222, i32 48, i32 0)
  %1232 = bitcast <4 x i32> %1231 to <8 x bfloat>
  %1233 = shufflevector <8 x bfloat> %1232, <8 x bfloat> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %1234 = shufflevector <32 x bfloat> %1230, <32 x bfloat> %1233, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  br label %1235

1235:                                             ; preds = %1218, %1215
  %1236 = phi <32 x bfloat> [ %1234, %1218 ], [ undef, %1215 ]
  %1237 = bitcast float %1216 to i32
  %1238 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1237, i32 177, i32 15, i32 15, i1 false)
  %1239 = bitcast i32 %1238 to float
  %1240 = fadd contract float %1216, %1239
  %1241 = bitcast float %1240 to i32
  %1242 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1241, i32 78, i32 15, i32 15, i1 false)
  %1243 = bitcast i32 %1242 to float
  %1244 = fadd contract float %1240, %1243
  %1245 = bitcast float %1244 to i32
  %1246 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1245, i32 359, i32 15, i32 15, i1 false)
  %1247 = bitcast i32 %1246 to float
  %1248 = fadd contract float %1244, %1247
  %1249 = bitcast float %1248 to i32
  %1250 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1249, i32 367, i32 15, i32 15, i1 false)
  %1251 = bitcast i32 %1250 to float
  %1252 = fadd contract float %1248, %1251
  %1253 = bitcast float %1252 to i32
  %1254 = tail call i32 @llvm.amdgcn.permlanex16.i32(i32 poison, i32 %1253, i32 1985229328, i32 -19088744, i1 true, i1 true)
  br i1 %124, label %1255, label %1260

1255:                                             ; preds = %1235
  %1256 = lshr i32 %21, 5
  %1257 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 8288), i32 %1256
  %1258 = bitcast i32 %1254 to float
  %1259 = fadd contract float %1252, %1258
  store float %1259, ptr addrspace(3) %1257, align 4, !tbaa !39
  br label %1260

1260:                                             ; preds = %1255, %1235
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1261 = and i32 %21, 7
  %1262 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 8288), i32 %1261
  %1263 = load float, ptr addrspace(3) %1262, align 4, !tbaa !39
  %1264 = bitcast float %1263 to i32
  %1265 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1264, i32 177, i32 15, i32 15, i1 false)
  %1266 = bitcast i32 %1265 to float
  %1267 = fadd contract float %1263, %1266
  %1268 = bitcast float %1267 to i32
  %1269 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1268, i32 78, i32 15, i32 15, i1 false)
  %1270 = bitcast i32 %1269 to float
  %1271 = fadd contract float %1267, %1270
  %1272 = bitcast float %1271 to i32
  %1273 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1272, i32 359, i32 15, i32 15, i1 false)
  %1274 = bitcast i32 %1273 to float
  %1275 = fadd contract float %1271, %1274
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1276 = tail call float @llvm.amdgcn.readlane.f32(float %1275, i32 0)
  %1277 = and i1 %346, %1217
  br i1 %1277, label %1278, label %1544

1278:                                             ; preds = %1260
  %1279 = shl nsw i64 %25, 13
  %1280 = getelementptr inbounds i8, ptr %14, i64 %1279
  %1281 = shl i32 %34, 13
  %1282 = zext i32 %1281 to i64
  %1283 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %1280, i16 0, i64 %1282, i32 822099968)
  %1284 = fmul contract float %1276, f0x39800000
  %1285 = fadd contract float %1284, f0x358637BD
  %1286 = fcmp olt float %1285, f0x00800000
  %1287 = fmul float %1285, f0x4B800000
  %1288 = select i1 %1286, float %1287, float %1285
  %1289 = tail call float @llvm.amdgcn.rsq.f32(float %1288)
  %1290 = fmul float %1289, 4.096000e+03
  %1291 = select i1 %1286, float %1290, float %1289
  %1292 = shl nuw nsw i32 %23, 11
  %1293 = shl nuw nsw i32 %32, 6
  %1294 = getelementptr i8, ptr addrspace(3) %365, i32 %1292
  %1295 = getelementptr i8, ptr addrspace(3) %1294, i32 %1293
  %1296 = load <32 x bfloat>, ptr addrspace(3) %1295, align 64, !tbaa !29
  %1297 = or disjoint i32 %1292, %1293
  %1298 = shufflevector <32 x bfloat> %1236, <32 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1299 = fpext <2 x bfloat> %1298 to <2 x float>
  %1300 = shufflevector <32 x bfloat> %1296, <32 x bfloat> poison, <2 x i32> <i32 0, i32 1>
  %1301 = fpext <2 x bfloat> %1300 to <2 x float>
  %1302 = fmul contract <2 x float> %1299, %1301
  %1303 = insertelement <2 x float> poison, float %1291, i64 0
  %1304 = shufflevector <2 x float> %1303, <2 x float> poison, <2 x i32> zeroinitializer
  %1305 = fmul contract <2 x float> %1304, %1302
  %1306 = fptrunc <2 x float> %1305 to <2 x bfloat>
  %1307 = shufflevector <32 x bfloat> %1236, <32 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1308 = fpext <2 x bfloat> %1307 to <2 x float>
  %1309 = shufflevector <32 x bfloat> %1296, <32 x bfloat> poison, <2 x i32> <i32 2, i32 3>
  %1310 = fpext <2 x bfloat> %1309 to <2 x float>
  %1311 = fmul contract <2 x float> %1308, %1310
  %1312 = fmul contract <2 x float> %1304, %1311
  %1313 = fptrunc <2 x float> %1312 to <2 x bfloat>
  %1314 = shufflevector <2 x bfloat> %1306, <2 x bfloat> %1313, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %1315 = shufflevector <32 x bfloat> %1236, <32 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1316 = fpext <2 x bfloat> %1315 to <2 x float>
  %1317 = shufflevector <32 x bfloat> %1296, <32 x bfloat> poison, <2 x i32> <i32 4, i32 5>
  %1318 = fpext <2 x bfloat> %1317 to <2 x float>
  %1319 = fmul contract <2 x float> %1316, %1318
  %1320 = fmul contract <2 x float> %1304, %1319
  %1321 = fptrunc <2 x float> %1320 to <2 x bfloat>
  %1322 = shufflevector <2 x bfloat> %1321, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1323 = shufflevector <8 x bfloat> %1314, <8 x bfloat> %1322, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %1324 = shufflevector <32 x bfloat> %1236, <32 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1325 = fpext <2 x bfloat> %1324 to <2 x float>
  %1326 = shufflevector <32 x bfloat> %1296, <32 x bfloat> poison, <2 x i32> <i32 6, i32 7>
  %1327 = fpext <2 x bfloat> %1326 to <2 x float>
  %1328 = fmul contract <2 x float> %1325, %1327
  %1329 = fmul contract <2 x float> %1304, %1328
  %1330 = fptrunc <2 x float> %1329 to <2 x bfloat>
  %1331 = shufflevector <2 x bfloat> %1330, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1332 = shufflevector <8 x bfloat> %1323, <8 x bfloat> %1331, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %1333 = bitcast <8 x bfloat> %1332 to <4 x i32>
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %1333, ptr addrspace(8) %1283, i32 %1297, i32 0, i32 0)
  tail call void asm sideeffect "s_nop 0", ""() #25, !srcloc !42
  %1334 = shufflevector <32 x bfloat> %1236, <32 x bfloat> poison, <2 x i32> <i32 8, i32 9>
  %1335 = fpext <2 x bfloat> %1334 to <2 x float>
  %1336 = shufflevector <32 x bfloat> %1296, <32 x bfloat> poison, <2 x i32> <i32 8, i32 9>
  %1337 = fpext <2 x bfloat> %1336 to <2 x float>
  %1338 = fmul contract <2 x float> %1335, %1337
  %1339 = fmul contract <2 x float> %1304, %1338
  %1340 = fptrunc <2 x float> %1339 to <2 x bfloat>
  %1341 = shufflevector <32 x bfloat> %1236, <32 x bfloat> poison, <2 x i32> <i32 10, i32 11>
  %1342 = fpext <2 x bfloat> %1341 to <2 x float>
  %1343 = shufflevector <32 x bfloat> %1296, <32 x bfloat> poison, <2 x i32> <i32 10, i32 11>
  %1344 = fpext <2 x bfloat> %1343 to <2 x float>
  %1345 = fmul contract <2 x float> %1342, %1344
  %1346 = fmul contract <2 x float> %1304, %1345
  %1347 = fptrunc <2 x float> %1346 to <2 x bfloat>
  %1348 = shufflevector <2 x bfloat> %1340, <2 x bfloat> %1347, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %1349 = shufflevector <32 x bfloat> %1236, <32 x bfloat> poison, <2 x i32> <i32 12, i32 13>
  %1350 = fpext <2 x bfloat> %1349 to <2 x float>
  %1351 = shufflevector <32 x bfloat> %1296, <32 x bfloat> poison, <2 x i32> <i32 12, i32 13>
  %1352 = fpext <2 x bfloat> %1351 to <2 x float>
  %1353 = fmul contract <2 x float> %1350, %1352
  %1354 = fmul contract <2 x float> %1304, %1353
  %1355 = fptrunc <2 x float> %1354 to <2 x bfloat>
  %1356 = shufflevector <2 x bfloat> %1355, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1357 = shufflevector <8 x bfloat> %1348, <8 x bfloat> %1356, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %1358 = shufflevector <32 x bfloat> %1236, <32 x bfloat> poison, <2 x i32> <i32 14, i32 15>
  %1359 = fpext <2 x bfloat> %1358 to <2 x float>
  %1360 = shufflevector <32 x bfloat> %1296, <32 x bfloat> poison, <2 x i32> <i32 14, i32 15>
  %1361 = fpext <2 x bfloat> %1360 to <2 x float>
  %1362 = fmul contract <2 x float> %1359, %1361
  %1363 = fmul contract <2 x float> %1304, %1362
  %1364 = fptrunc <2 x float> %1363 to <2 x bfloat>
  %1365 = shufflevector <2 x bfloat> %1364, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1366 = shufflevector <8 x bfloat> %1357, <8 x bfloat> %1365, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %1367 = bitcast <8 x bfloat> %1366 to <4 x i32>
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %1367, ptr addrspace(8) %1283, i32 %1297, i32 16, i32 0)
  tail call void asm sideeffect "s_nop 0", ""() #25, !srcloc !42
  %1368 = shufflevector <32 x bfloat> %1236, <32 x bfloat> poison, <2 x i32> <i32 16, i32 17>
  %1369 = fpext <2 x bfloat> %1368 to <2 x float>
  %1370 = shufflevector <32 x bfloat> %1296, <32 x bfloat> poison, <2 x i32> <i32 16, i32 17>
  %1371 = fpext <2 x bfloat> %1370 to <2 x float>
  %1372 = fmul contract <2 x float> %1369, %1371
  %1373 = fmul contract <2 x float> %1304, %1372
  %1374 = fptrunc <2 x float> %1373 to <2 x bfloat>
  %1375 = shufflevector <32 x bfloat> %1236, <32 x bfloat> poison, <2 x i32> <i32 18, i32 19>
  %1376 = fpext <2 x bfloat> %1375 to <2 x float>
  %1377 = shufflevector <32 x bfloat> %1296, <32 x bfloat> poison, <2 x i32> <i32 18, i32 19>
  %1378 = fpext <2 x bfloat> %1377 to <2 x float>
  %1379 = fmul contract <2 x float> %1376, %1378
  %1380 = fmul contract <2 x float> %1304, %1379
  %1381 = fptrunc <2 x float> %1380 to <2 x bfloat>
  %1382 = shufflevector <2 x bfloat> %1374, <2 x bfloat> %1381, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %1383 = shufflevector <32 x bfloat> %1236, <32 x bfloat> poison, <2 x i32> <i32 20, i32 21>
  %1384 = fpext <2 x bfloat> %1383 to <2 x float>
  %1385 = shufflevector <32 x bfloat> %1296, <32 x bfloat> poison, <2 x i32> <i32 20, i32 21>
  %1386 = fpext <2 x bfloat> %1385 to <2 x float>
  %1387 = fmul contract <2 x float> %1384, %1386
  %1388 = fmul contract <2 x float> %1304, %1387
  %1389 = fptrunc <2 x float> %1388 to <2 x bfloat>
  %1390 = shufflevector <2 x bfloat> %1389, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1391 = shufflevector <8 x bfloat> %1382, <8 x bfloat> %1390, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %1392 = shufflevector <32 x bfloat> %1236, <32 x bfloat> poison, <2 x i32> <i32 22, i32 23>
  %1393 = fpext <2 x bfloat> %1392 to <2 x float>
  %1394 = shufflevector <32 x bfloat> %1296, <32 x bfloat> poison, <2 x i32> <i32 22, i32 23>
  %1395 = fpext <2 x bfloat> %1394 to <2 x float>
  %1396 = fmul contract <2 x float> %1393, %1395
  %1397 = fmul contract <2 x float> %1304, %1396
  %1398 = fptrunc <2 x float> %1397 to <2 x bfloat>
  %1399 = shufflevector <2 x bfloat> %1398, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1400 = shufflevector <8 x bfloat> %1391, <8 x bfloat> %1399, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %1401 = bitcast <8 x bfloat> %1400 to <4 x i32>
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %1401, ptr addrspace(8) %1283, i32 %1297, i32 32, i32 0)
  tail call void asm sideeffect "s_nop 0", ""() #25, !srcloc !42
  %1402 = shufflevector <32 x bfloat> %1236, <32 x bfloat> poison, <2 x i32> <i32 24, i32 25>
  %1403 = fpext <2 x bfloat> %1402 to <2 x float>
  %1404 = shufflevector <32 x bfloat> %1296, <32 x bfloat> poison, <2 x i32> <i32 24, i32 25>
  %1405 = fpext <2 x bfloat> %1404 to <2 x float>
  %1406 = fmul contract <2 x float> %1403, %1405
  %1407 = fmul contract <2 x float> %1304, %1406
  %1408 = fptrunc <2 x float> %1407 to <2 x bfloat>
  %1409 = shufflevector <32 x bfloat> %1236, <32 x bfloat> poison, <2 x i32> <i32 26, i32 27>
  %1410 = fpext <2 x bfloat> %1409 to <2 x float>
  %1411 = shufflevector <32 x bfloat> %1296, <32 x bfloat> poison, <2 x i32> <i32 26, i32 27>
  %1412 = fpext <2 x bfloat> %1411 to <2 x float>
  %1413 = fmul contract <2 x float> %1410, %1412
  %1414 = fmul contract <2 x float> %1304, %1413
  %1415 = fptrunc <2 x float> %1414 to <2 x bfloat>
  %1416 = shufflevector <2 x bfloat> %1408, <2 x bfloat> %1415, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %1417 = shufflevector <32 x bfloat> %1236, <32 x bfloat> poison, <2 x i32> <i32 28, i32 29>
  %1418 = fpext <2 x bfloat> %1417 to <2 x float>
  %1419 = shufflevector <32 x bfloat> %1296, <32 x bfloat> poison, <2 x i32> <i32 28, i32 29>
  %1420 = fpext <2 x bfloat> %1419 to <2 x float>
  %1421 = fmul contract <2 x float> %1418, %1420
  %1422 = fmul contract <2 x float> %1304, %1421
  %1423 = fptrunc <2 x float> %1422 to <2 x bfloat>
  %1424 = shufflevector <2 x bfloat> %1423, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1425 = shufflevector <8 x bfloat> %1416, <8 x bfloat> %1424, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %1426 = shufflevector <32 x bfloat> %1236, <32 x bfloat> poison, <2 x i32> <i32 30, i32 31>
  %1427 = fpext <2 x bfloat> %1426 to <2 x float>
  %1428 = shufflevector <32 x bfloat> %1296, <32 x bfloat> poison, <2 x i32> <i32 30, i32 31>
  %1429 = fpext <2 x bfloat> %1428 to <2 x float>
  %1430 = fmul contract <2 x float> %1427, %1429
  %1431 = fmul contract <2 x float> %1304, %1430
  %1432 = fptrunc <2 x float> %1431 to <2 x bfloat>
  %1433 = shufflevector <2 x bfloat> %1432, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %1434 = shufflevector <8 x bfloat> %1425, <8 x bfloat> %1433, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %1435 = bitcast <8 x bfloat> %1434 to <4 x i32>
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %1435, ptr addrspace(8) %1283, i32 %1297, i32 48, i32 0)
  tail call void asm sideeffect "s_nop 0", ""() #25, !srcloc !42
  br label %1544

1436:                                             ; preds = %343
  %1437 = icmp samesign ult i32 %32, 4
  %1438 = icmp sgt i32 %34, 0
  %1439 = select i1 %1437, i1 %1438, i1 false
  br i1 %1439, label %1440, label %1461

1440:                                             ; preds = %1436
  %1441 = getelementptr [4 x i8], ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 8192), i32 %32
  %1442 = getelementptr i8, ptr addrspace(3) %1441, i32 16
  %1443 = load float, ptr addrspace(3) %1442, align 4, !tbaa !39
  %1444 = getelementptr inbounds nuw i8, ptr %16, i64 4
  %1445 = load float, ptr %1444, align 4, !tbaa !39
  %1446 = fmul contract float %1443, %1445
  %1447 = zext nneg i32 %32 to i64
  %1448 = getelementptr [4 x i8], ptr %17, i64 %1447
  %1449 = getelementptr i8, ptr %1448, i64 16
  %1450 = load float, ptr %1449, align 4, !tbaa !39
  %1451 = fadd contract float %1446, %1450
  %1452 = fmul contract float %1451, f0xBFB8AA3B
  %1453 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %1452)
  %1454 = fadd contract float %1453, 1.000000e+00
  %1455 = fdiv contract float 1.000000e+00, %1454
  %1456 = fmul contract float %1455, 2.000000e+00
  %1457 = shl nsw i32 %9, 2
  %1458 = or disjoint i32 %1457, %170
  %1459 = sext i32 %1458 to i64
  %1460 = getelementptr inbounds [4 x i8], ptr %12, i64 %1459
  store float %1456, ptr %1460, align 4, !tbaa !39
  br label %1461

1461:                                             ; preds = %1440, %1436
  %1462 = icmp samesign ult i32 %32, 16
  br i1 %1462, label %1463, label %1475

1463:                                             ; preds = %1461
  %1464 = zext nneg i32 %32 to i64
  %1465 = getelementptr [4 x i8], ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @__dynamic_shared_0, i32 8192), i32 %32
  %1466 = getelementptr i8, ptr addrspace(3) %1465, i32 32
  %1467 = load float, ptr addrspace(3) %1466, align 4, !tbaa !39
  %1468 = getelementptr inbounds nuw i8, ptr %16, i64 8
  %1469 = load float, ptr %1468, align 4, !tbaa !39
  %1470 = fmul contract float %1467, %1469
  %1471 = getelementptr [4 x i8], ptr %17, i64 %1464
  %1472 = getelementptr i8, ptr %1471, i64 32
  %1473 = load float, ptr %1472, align 4, !tbaa !39
  %1474 = fadd contract float %1470, %1473
  br label %1475

1475:                                             ; preds = %1463, %1461
  %1476 = phi float [ %1474, %1463 ], [ 0.000000e+00, %1461 ]
  %1477 = bitcast float %1476 to i32
  %1478 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1477, i32 177, i32 15, i32 15, i1 false)
  %1479 = bitcast i32 %1478 to float
  %1480 = tail call nsz contract noundef float @llvm.maxnum.f32(float %1479, float %1476)
  %1481 = bitcast float %1480 to i32
  %1482 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1481, i32 78, i32 15, i32 15, i1 false)
  %1483 = bitcast i32 %1482 to float
  %1484 = tail call nsz contract noundef float @llvm.maxnum.f32(float %1483, float %1480)
  %1485 = fsub contract float %1476, %1484
  %1486 = tail call contract noundef float @llvm.exp.f32(float %1485)
  %1487 = bitcast float %1486 to i32
  %1488 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1487, i32 177, i32 15, i32 15, i1 false)
  %1489 = bitcast i32 %1488 to float
  %1490 = fadd contract float %1486, %1489
  %1491 = bitcast float %1490 to i32
  %1492 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1491, i32 78, i32 15, i32 15, i1 false)
  %1493 = bitcast i32 %1492 to float
  %1494 = fadd contract float %1490, %1493
  %1495 = fdiv contract float %1486, %1494
  %1496 = fadd contract float %1495, f0x358637BD
  %1497 = bitcast float %1496 to i32
  %1498 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1497, i32 292, i32 15, i32 15, i1 true)
  %1499 = bitcast i32 %1498 to float
  %1500 = fadd contract float %1496, %1499
  %1501 = bitcast float %1500 to i32
  %1502 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1501, i32 296, i32 15, i32 15, i1 true)
  %1503 = bitcast i32 %1502 to float
  %1504 = fadd contract float %1500, %1503
  %1505 = fadd contract float %1504, f0x358637BD
  %1506 = fdiv contract float %1496, %1505
  br label %1510

1507:                                             ; preds = %1510
  %1508 = lshr i32 %32, 4
  %1509 = icmp slt i32 %1508, %34
  br i1 %1509, label %1537, label %1544

1510:                                             ; preds = %1510, %1475
  %1511 = phi i32 [ %1535, %1510 ], [ 0, %1475 ]
  %1512 = phi float [ %1534, %1510 ], [ %1506, %1475 ]
  %1513 = bitcast float %1512 to i32
  %1514 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1513, i32 177, i32 15, i32 15, i1 false)
  %1515 = bitcast i32 %1514 to float
  %1516 = fadd contract float %1512, %1515
  %1517 = bitcast float %1516 to i32
  %1518 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1517, i32 78, i32 15, i32 15, i1 false)
  %1519 = bitcast i32 %1518 to float
  %1520 = fadd contract float %1516, %1519
  %1521 = fadd contract float %1520, f0x358637BD
  %1522 = tail call contract float @llvm.amdgcn.rcp.f32(float %1521)
  %1523 = fmul contract float %1512, %1522
  %1524 = bitcast float %1523 to i32
  %1525 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1524, i32 292, i32 15, i32 15, i1 true)
  %1526 = bitcast i32 %1525 to float
  %1527 = fadd contract float %1523, %1526
  %1528 = bitcast float %1527 to i32
  %1529 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1528, i32 296, i32 15, i32 15, i1 true)
  %1530 = bitcast i32 %1529 to float
  %1531 = fadd contract float %1527, %1530
  %1532 = fadd contract float %1531, f0x358637BD
  %1533 = tail call contract float @llvm.amdgcn.rcp.f32(float %1532)
  %1534 = fmul contract float %1523, %1533
  %1535 = add nuw nsw i32 %1511, 1
  %1536 = icmp eq i32 %1511, 18
  br i1 %1536, label %1507, label %1510, !llvm.loop !43

1537:                                             ; preds = %1507
  %1538 = add nsw i32 %9, %1508
  %1539 = shl nsw i32 %1538, 4
  %1540 = and i32 %21, 15
  %1541 = or disjoint i32 %1539, %1540
  %1542 = sext i32 %1541 to i64
  %1543 = getelementptr inbounds [4 x i8], ptr %13, i64 %1542
  store float %1534, ptr %1543, align 4, !tbaa !39
  br label %1544

1544:                                             ; preds = %1260, %1278, %1507, %1537
  ret void
}

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #9

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #10

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #10

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #11

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr readnone, i16, i64, i32) #12

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @llvm.amdgcn.tensor.load.to.lds(<4 x i32>, <8 x i32>, <4 x i32>, <4 x i32>, <8 x i32>, i32 immarg) #13

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @llvm.amdgcn.global.load.async.to.lds.b128(ptr addrspace(1) readonly captures(none), ptr addrspace(3) writeonly captures(none), i32 immarg, i32 immarg) #14

; Function Attrs: nocallback nofree nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.amdgcn.s.wait.asynccnt(i16 immarg) #15

; Function Attrs: nocallback nofree nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.amdgcn.s.wait.tensorcnt(i16 immarg) #15

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.wait.dscnt(i16 immarg) #16

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat>, <16 x bfloat>, i16 immarg, <8 x float>, i1 immarg, i1 immarg) #9

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32>, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #17

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.rcp.f32(float) #12

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #18

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #18

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32 immarg, i32 immarg, i32 immarg, i1 immarg) #7

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.permlanex16.i32(i32, i32, i32, i32, i1 immarg, i1 immarg) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #12

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #12

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.exp.f32(float) #12

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.rsq.f32(float) #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef align 4 ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #11

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.amdgcn.s.sleep(i32 immarg) #19

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i64 @llvm.amdgcn.readfirstlane.i64(i64) #9

; Function Attrs: nocallback nounwind willreturn
declare void @llvm.amdgcn.s.sendmsg(i32 immarg, i32) #20

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #21

; Function Attrs: convergent norecurse nounwind uwtable
define internal fastcc void @__ockl_fprintf_append_string_n(i64 noundef %0, ptr noundef readonly captures(address_is_null) %1, i64 noundef range(i64 -2147483648, 2147483648) %2, i32 noundef range(i32 0, 2) %3) unnamed_addr #22 {
  %5 = icmp eq i32 %3, 0
  %6 = or i64 %0, 2
  %7 = select i1 %5, i64 %0, i64 %6
  %8 = icmp eq ptr %1, null
  br i1 %8, label %9, label %126

9:                                                ; preds = %4
  %10 = and i64 %7, -225
  %11 = or disjoint i64 %10, 32
  %12 = tail call dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %13 = getelementptr inbounds nuw i8, ptr addrspace(4) %12, i64 80
  %14 = load i64, ptr addrspace(4) %13, align 8, !tbaa !14
  %15 = inttoptr i64 %14 to ptr addrspace(1)
  %16 = tail call range(i32 0, 33) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %17 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %16)
  %18 = icmp eq i32 %16, %17
  br i1 %18, label %19, label %41

19:                                               ; preds = %9
  %20 = getelementptr inbounds nuw i8, ptr addrspace(1) %15, i64 24
  %21 = load atomic i64, ptr addrspace(1) %20 syncscope("one-as") acquire, align 8
  %22 = getelementptr i8, ptr addrspace(1) %15, i64 40
  %23 = load ptr addrspace(1), ptr addrspace(1) %15, align 8, !tbaa !16
  %24 = load i64, ptr addrspace(1) %22, align 8, !tbaa !20
  %25 = and i64 %24, %21
  %26 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %23, i64 %25
  %27 = load atomic i64, ptr addrspace(1) %26 syncscope("one-as") monotonic, align 8
  %28 = cmpxchg ptr addrspace(1) %20, i64 %21, i64 %27 syncscope("one-as") acquire monotonic, align 8
  %29 = extractvalue { i64, i1 } %28, 1
  %30 = extractvalue { i64, i1 } %28, 0
  br i1 %29, label %41, label %31

31:                                               ; preds = %19, %31
  %32 = phi i64 [ %40, %31 ], [ %30, %19 ]
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  %33 = load ptr addrspace(1), ptr addrspace(1) %15, align 8, !tbaa !16
  %34 = load i64, ptr addrspace(1) %22, align 8, !tbaa !20
  %35 = and i64 %34, %32
  %36 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %33, i64 %35
  %37 = load atomic i64, ptr addrspace(1) %36 syncscope("one-as") monotonic, align 8
  %38 = cmpxchg ptr addrspace(1) %20, i64 %32, i64 %37 syncscope("one-as") acquire monotonic, align 8
  %39 = extractvalue { i64, i1 } %38, 1
  %40 = extractvalue { i64, i1 } %38, 0
  br i1 %39, label %41, label %31

41:                                               ; preds = %31, %19, %9
  %42 = phi i64 [ 0, %9 ], [ %30, %19 ], [ %40, %31 ]
  %43 = tail call i64 @llvm.amdgcn.readfirstlane.i64(i64 %42)
  %44 = load ptr addrspace(1), ptr addrspace(1) %15, align 8, !tbaa !16
  %45 = getelementptr i8, ptr addrspace(1) %15, i64 40
  %46 = load i64, ptr addrspace(1) %45, align 8, !tbaa !20
  %47 = and i64 %46, %43
  %48 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %44, i64 %47
  %49 = getelementptr i8, ptr addrspace(1) %15, i64 8
  %50 = load ptr addrspace(1), ptr addrspace(1) %49, align 8, !tbaa !21
  %51 = getelementptr inbounds nuw [4096 x i8], ptr addrspace(1) %50, i64 %47
  %52 = tail call i32 @llvm.amdgcn.ballot.i32(i1 true)
  br i1 %18, label %53, label %57

53:                                               ; preds = %41
  %54 = getelementptr inbounds nuw i8, ptr addrspace(1) %48, i64 8
  %55 = getelementptr inbounds nuw i8, ptr addrspace(1) %48, i64 16
  %56 = zext i32 %52 to i64
  store i64 %56, ptr addrspace(1) %54, align 8, !tbaa !22
  store <2 x i32> <i32 2, i32 1>, ptr addrspace(1) %55, align 8, !tbaa !9
  br label %57

57:                                               ; preds = %53, %41
  %58 = zext nneg i32 %16 to i64
  %59 = getelementptr inbounds nuw [64 x i8], ptr addrspace(1) %51, i64 %58
  store i64 %11, ptr addrspace(1) %59, align 8, !tbaa !14
  %60 = getelementptr inbounds nuw i8, ptr addrspace(1) %59, i64 8
  store i64 0, ptr addrspace(1) %60, align 8, !tbaa !14
  %61 = getelementptr inbounds nuw i8, ptr addrspace(1) %59, i64 16
  store i64 0, ptr addrspace(1) %61, align 8, !tbaa !14
  %62 = getelementptr inbounds nuw i8, ptr addrspace(1) %59, i64 24
  store i64 0, ptr addrspace(1) %62, align 8, !tbaa !14
  %63 = getelementptr inbounds nuw i8, ptr addrspace(1) %59, i64 32
  store i64 0, ptr addrspace(1) %63, align 8, !tbaa !14
  %64 = getelementptr inbounds nuw i8, ptr addrspace(1) %59, i64 40
  store i64 0, ptr addrspace(1) %64, align 8, !tbaa !14
  %65 = getelementptr inbounds nuw i8, ptr addrspace(1) %59, i64 48
  store i64 0, ptr addrspace(1) %65, align 8, !tbaa !14
  %66 = getelementptr inbounds nuw i8, ptr addrspace(1) %59, i64 56
  store i64 0, ptr addrspace(1) %66, align 8, !tbaa !14
  br i1 %18, label %67, label %96

67:                                               ; preds = %57
  %68 = getelementptr inbounds nuw i8, ptr addrspace(1) %15, i64 32
  %69 = load atomic i64, ptr addrspace(1) %68 syncscope("one-as") monotonic, align 8
  %70 = load i64, ptr addrspace(1) %45, align 8, !tbaa !20
  %71 = and i64 %70, %43
  %72 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %44, i64 %71
  store i64 %69, ptr addrspace(1) %72, align 8, !tbaa !24
  %73 = cmpxchg ptr addrspace(1) %68, i64 %69, i64 %43 syncscope("one-as") release monotonic, align 8
  %74 = extractvalue { i64, i1 } %73, 1
  br i1 %74, label %80, label %75

75:                                               ; preds = %67, %75
  %76 = phi { i64, i1 } [ %78, %75 ], [ %73, %67 ]
  %77 = extractvalue { i64, i1 } %76, 0
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %77, ptr addrspace(1) %72, align 8, !tbaa !24
  %78 = cmpxchg ptr addrspace(1) %68, i64 %77, i64 %43 syncscope("one-as") release monotonic, align 8
  %79 = extractvalue { i64, i1 } %78, 1
  br i1 %79, label %80, label %75

80:                                               ; preds = %75, %67
  %81 = getelementptr inbounds nuw i8, ptr addrspace(1) %15, i64 16
  %82 = load i64, ptr addrspace(1) %81, align 8
  %83 = inttoptr i64 %82 to ptr addrspace(1)
  %84 = getelementptr inbounds nuw i8, ptr addrspace(1) %83, i64 8
  %85 = atomicrmw add ptr addrspace(1) %84, i64 1 syncscope("one-as") release, align 8, !amdgpu.no.fine.grained.memory !25, !amdgpu.no.remote.memory !25
  %86 = getelementptr inbounds nuw i8, ptr addrspace(1) %83, i64 16
  %87 = load i64, ptr addrspace(1) %86, align 16, !tbaa !26
  %88 = icmp eq i64 %87, 0
  br i1 %88, label %96, label %89

89:                                               ; preds = %80
  %90 = inttoptr i64 %87 to ptr addrspace(1)
  %91 = getelementptr inbounds nuw i8, ptr addrspace(1) %83, i64 24
  %92 = load i32, ptr addrspace(1) %91, align 8, !tbaa !28
  %93 = zext i32 %92 to i64
  store atomic i64 %93, ptr addrspace(1) %90 syncscope("one-as") release, align 8
  %94 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %92)
  %95 = and i32 %94, 16777215
  tail call void @llvm.amdgcn.s.sendmsg(i32 1, i32 %95)
  br label %96

96:                                               ; preds = %89, %80, %57
  %97 = getelementptr inbounds nuw i8, ptr addrspace(1) %48, i64 20
  br label %98

98:                                               ; preds = %106, %96
  br i1 %18, label %99, label %102

99:                                               ; preds = %98
  %100 = load atomic i32, ptr addrspace(1) %97 syncscope("one-as") acquire, align 4
  %101 = and i32 %100, 1
  br label %102

102:                                              ; preds = %99, %98
  %103 = phi i32 [ %101, %99 ], [ 1, %98 ]
  %104 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %103)
  %105 = icmp eq i32 %104, 0
  br i1 %105, label %107, label %106

106:                                              ; preds = %102
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  br label %98

107:                                              ; preds = %102
  br i1 %18, label %108, label %429

108:                                              ; preds = %107
  %109 = load i64, ptr addrspace(1) %45, align 8, !tbaa !20
  %110 = add i64 %109, 1
  %111 = add i64 %110, %43
  %112 = icmp eq i64 %111, 0
  %113 = select i1 %112, i64 %110, i64 %111
  %114 = getelementptr inbounds nuw i8, ptr addrspace(1) %15, i64 24
  %115 = load atomic i64, ptr addrspace(1) %114 syncscope("one-as") monotonic, align 8
  %116 = load ptr addrspace(1), ptr addrspace(1) %15, align 8, !tbaa !16
  %117 = and i64 %113, %109
  %118 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %116, i64 %117
  store i64 %115, ptr addrspace(1) %118, align 8, !tbaa !24
  %119 = cmpxchg ptr addrspace(1) %114, i64 %115, i64 %113 syncscope("one-as") release monotonic, align 8
  %120 = extractvalue { i64, i1 } %119, 1
  br i1 %120, label %429, label %121

121:                                              ; preds = %108, %121
  %122 = phi { i64, i1 } [ %124, %121 ], [ %119, %108 ]
  %123 = extractvalue { i64, i1 } %122, 0
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %123, ptr addrspace(1) %118, align 8, !tbaa !24
  %124 = cmpxchg ptr addrspace(1) %114, i64 %123, i64 %113 syncscope("one-as") release monotonic, align 8
  %125 = extractvalue { i64, i1 } %124, 1
  br i1 %125, label %429, label %121

126:                                              ; preds = %4
  %127 = and i64 %7, 2
  %128 = and i64 %7, -3
  %129 = insertelement <2 x i64> <i64 poison, i64 0>, i64 %128, i64 0
  %130 = tail call dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %131 = getelementptr inbounds nuw i8, ptr addrspace(4) %130, i64 80
  %132 = load i64, ptr addrspace(4) %131, align 8, !tbaa !14
  %133 = inttoptr i64 %132 to ptr addrspace(1)
  %134 = tail call range(i32 0, 33) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %135 = getelementptr i8, ptr addrspace(1) %133, i64 40
  %136 = getelementptr i8, ptr addrspace(1) %133, i64 8
  %137 = zext nneg i32 %134 to i64
  %138 = getelementptr inbounds nuw i8, ptr addrspace(1) %133, i64 24
  %139 = getelementptr inbounds nuw i8, ptr addrspace(1) %133, i64 32
  %140 = getelementptr inbounds nuw i8, ptr addrspace(1) %133, i64 16
  br label %141

141:                                              ; preds = %423, %126
  %142 = phi i64 [ %2, %126 ], [ %426, %423 ]
  %143 = phi ptr [ %1, %126 ], [ %427, %423 ]
  %144 = phi <2 x i64> [ %129, %126 ], [ %425, %423 ]
  %145 = icmp ugt i64 %142, 56
  %146 = extractelement <2 x i64> %144, i64 0
  %147 = tail call i64 @llvm.umin.i64(i64 %142, i64 56)
  %148 = trunc nuw nsw i64 %147 to i32
  %149 = select i1 %145, i64 0, i64 %127
  %150 = icmp ugt i64 %142, 7
  br i1 %150, label %153, label %151

151:                                              ; preds = %141
  %152 = icmp eq i64 %142, 0
  br i1 %152, label %170, label %157

153:                                              ; preds = %141
  %154 = load i64, ptr %143, align 1
  %155 = add nsw i32 %148, -8
  %156 = getelementptr inbounds nuw i8, ptr %143, i64 8
  br label %170

157:                                              ; preds = %151, %157
  %158 = phi i32 [ %168, %157 ], [ 0, %151 ]
  %159 = phi i64 [ %167, %157 ], [ 0, %151 ]
  %160 = zext nneg i32 %158 to i64
  %161 = getelementptr inbounds nuw i8, ptr %143, i64 %160
  %162 = load i8, ptr %161, align 1, !tbaa !32
  %163 = zext i8 %162 to i64
  %164 = shl i32 %158, 3
  %165 = zext nneg i32 %164 to i64
  %166 = shl nuw i64 %163, %165
  %167 = or i64 %166, %159
  %168 = add nuw nsw i32 %158, 1
  %169 = icmp eq i32 %168, %148
  br i1 %169, label %170, label %157

170:                                              ; preds = %157, %153, %151
  %171 = phi ptr [ %156, %153 ], [ %143, %151 ], [ %143, %157 ]
  %172 = phi i32 [ %155, %153 ], [ 0, %151 ], [ 0, %157 ]
  %173 = phi i64 [ %154, %153 ], [ 0, %151 ], [ %167, %157 ]
  %174 = icmp ugt i32 %172, 7
  br i1 %174, label %177, label %175

175:                                              ; preds = %170
  %176 = icmp eq i32 %172, 0
  br i1 %176, label %194, label %181

177:                                              ; preds = %170
  %178 = load i64, ptr %171, align 1
  %179 = add nsw i32 %172, -8
  %180 = getelementptr inbounds nuw i8, ptr %171, i64 8
  br label %194

181:                                              ; preds = %175, %181
  %182 = phi i32 [ %192, %181 ], [ 0, %175 ]
  %183 = phi i64 [ %191, %181 ], [ 0, %175 ]
  %184 = zext nneg i32 %182 to i64
  %185 = getelementptr inbounds nuw i8, ptr %171, i64 %184
  %186 = load i8, ptr %185, align 1, !tbaa !32
  %187 = zext i8 %186 to i64
  %188 = shl i32 %182, 3
  %189 = zext nneg i32 %188 to i64
  %190 = shl nuw i64 %187, %189
  %191 = or i64 %190, %183
  %192 = add nuw nsw i32 %182, 1
  %193 = icmp eq i32 %192, %172
  br i1 %193, label %194, label %181

194:                                              ; preds = %181, %177, %175
  %195 = phi ptr [ %180, %177 ], [ %171, %175 ], [ %171, %181 ]
  %196 = phi i32 [ %179, %177 ], [ 0, %175 ], [ 0, %181 ]
  %197 = phi i64 [ %178, %177 ], [ 0, %175 ], [ %191, %181 ]
  %198 = icmp ugt i32 %196, 7
  br i1 %198, label %201, label %199

199:                                              ; preds = %194
  %200 = icmp eq i32 %196, 0
  br i1 %200, label %218, label %205

201:                                              ; preds = %194
  %202 = load i64, ptr %195, align 1
  %203 = add nsw i32 %196, -8
  %204 = getelementptr inbounds nuw i8, ptr %195, i64 8
  br label %218

205:                                              ; preds = %199, %205
  %206 = phi i32 [ %216, %205 ], [ 0, %199 ]
  %207 = phi i64 [ %215, %205 ], [ 0, %199 ]
  %208 = zext nneg i32 %206 to i64
  %209 = getelementptr inbounds nuw i8, ptr %195, i64 %208
  %210 = load i8, ptr %209, align 1, !tbaa !32
  %211 = zext i8 %210 to i64
  %212 = shl i32 %206, 3
  %213 = zext nneg i32 %212 to i64
  %214 = shl nuw i64 %211, %213
  %215 = or i64 %214, %207
  %216 = add nuw nsw i32 %206, 1
  %217 = icmp eq i32 %216, %196
  br i1 %217, label %218, label %205

218:                                              ; preds = %205, %201, %199
  %219 = phi ptr [ %204, %201 ], [ %195, %199 ], [ %195, %205 ]
  %220 = phi i32 [ %203, %201 ], [ 0, %199 ], [ 0, %205 ]
  %221 = phi i64 [ %202, %201 ], [ 0, %199 ], [ %215, %205 ]
  %222 = icmp ugt i32 %220, 7
  br i1 %222, label %225, label %223

223:                                              ; preds = %218
  %224 = icmp eq i32 %220, 0
  br i1 %224, label %242, label %229

225:                                              ; preds = %218
  %226 = load i64, ptr %219, align 1
  %227 = add nsw i32 %220, -8
  %228 = getelementptr inbounds nuw i8, ptr %219, i64 8
  br label %242

229:                                              ; preds = %223, %229
  %230 = phi i32 [ %240, %229 ], [ 0, %223 ]
  %231 = phi i64 [ %239, %229 ], [ 0, %223 ]
  %232 = zext nneg i32 %230 to i64
  %233 = getelementptr inbounds nuw i8, ptr %219, i64 %232
  %234 = load i8, ptr %233, align 1, !tbaa !32
  %235 = zext i8 %234 to i64
  %236 = shl i32 %230, 3
  %237 = zext nneg i32 %236 to i64
  %238 = shl nuw i64 %235, %237
  %239 = or i64 %238, %231
  %240 = add nuw nsw i32 %230, 1
  %241 = icmp eq i32 %240, %220
  br i1 %241, label %242, label %229

242:                                              ; preds = %229, %225, %223
  %243 = phi ptr [ %228, %225 ], [ %219, %223 ], [ %219, %229 ]
  %244 = phi i32 [ %227, %225 ], [ 0, %223 ], [ 0, %229 ]
  %245 = phi i64 [ %226, %225 ], [ 0, %223 ], [ %239, %229 ]
  %246 = icmp ugt i32 %244, 7
  br i1 %246, label %249, label %247

247:                                              ; preds = %242
  %248 = icmp eq i32 %244, 0
  br i1 %248, label %266, label %253

249:                                              ; preds = %242
  %250 = load i64, ptr %243, align 1
  %251 = add nsw i32 %244, -8
  %252 = getelementptr inbounds nuw i8, ptr %243, i64 8
  br label %266

253:                                              ; preds = %247, %253
  %254 = phi i32 [ %264, %253 ], [ 0, %247 ]
  %255 = phi i64 [ %263, %253 ], [ 0, %247 ]
  %256 = zext nneg i32 %254 to i64
  %257 = getelementptr inbounds nuw i8, ptr %243, i64 %256
  %258 = load i8, ptr %257, align 1, !tbaa !32
  %259 = zext i8 %258 to i64
  %260 = shl i32 %254, 3
  %261 = zext nneg i32 %260 to i64
  %262 = shl nuw i64 %259, %261
  %263 = or i64 %262, %255
  %264 = add nuw nsw i32 %254, 1
  %265 = icmp eq i32 %264, %244
  br i1 %265, label %266, label %253

266:                                              ; preds = %253, %249, %247
  %267 = phi ptr [ %252, %249 ], [ %243, %247 ], [ %243, %253 ]
  %268 = phi i32 [ %251, %249 ], [ 0, %247 ], [ 0, %253 ]
  %269 = phi i64 [ %250, %249 ], [ 0, %247 ], [ %263, %253 ]
  %270 = icmp ugt i32 %268, 7
  br i1 %270, label %273, label %271

271:                                              ; preds = %266
  %272 = icmp eq i32 %268, 0
  br i1 %272, label %290, label %277

273:                                              ; preds = %266
  %274 = load i64, ptr %267, align 1
  %275 = add nsw i32 %268, -8
  %276 = getelementptr inbounds nuw i8, ptr %267, i64 8
  br label %290

277:                                              ; preds = %271, %277
  %278 = phi i32 [ %288, %277 ], [ 0, %271 ]
  %279 = phi i64 [ %287, %277 ], [ 0, %271 ]
  %280 = zext nneg i32 %278 to i64
  %281 = getelementptr inbounds nuw i8, ptr %267, i64 %280
  %282 = load i8, ptr %281, align 1, !tbaa !32
  %283 = zext i8 %282 to i64
  %284 = shl i32 %278, 3
  %285 = zext nneg i32 %284 to i64
  %286 = shl nuw i64 %283, %285
  %287 = or i64 %286, %279
  %288 = add nuw nsw i32 %278, 1
  %289 = icmp eq i32 %288, %268
  br i1 %289, label %290, label %277

290:                                              ; preds = %277, %273, %271
  %291 = phi ptr [ %276, %273 ], [ %267, %271 ], [ %267, %277 ]
  %292 = phi i32 [ %275, %273 ], [ 0, %271 ], [ 0, %277 ]
  %293 = phi i64 [ %274, %273 ], [ 0, %271 ], [ %287, %277 ]
  %294 = icmp ugt i32 %292, 7
  br i1 %294, label %297, label %295

295:                                              ; preds = %290
  %296 = icmp eq i32 %292, 0
  br i1 %296, label %312, label %299

297:                                              ; preds = %290
  %298 = load i64, ptr %291, align 1
  br label %312

299:                                              ; preds = %295, %299
  %300 = phi i32 [ %310, %299 ], [ 0, %295 ]
  %301 = phi i64 [ %309, %299 ], [ 0, %295 ]
  %302 = zext nneg i32 %300 to i64
  %303 = getelementptr inbounds nuw i8, ptr %291, i64 %302
  %304 = load i8, ptr %303, align 1, !tbaa !32
  %305 = zext i8 %304 to i64
  %306 = shl i32 %300, 3
  %307 = zext nneg i32 %306 to i64
  %308 = shl nuw i64 %305, %307
  %309 = or i64 %308, %301
  %310 = add nuw nsw i32 %300, 1
  %311 = icmp eq i32 %310, %292
  br i1 %311, label %312, label %299

312:                                              ; preds = %299, %297, %295
  %313 = phi i64 [ %298, %297 ], [ 0, %295 ], [ %309, %299 ]
  %314 = shl nuw nsw i64 %147, 2
  %315 = add nuw nsw i64 %314, 28
  %316 = and i64 %315, 480
  %317 = and i64 %146, -225
  %318 = or i64 %317, %149
  %319 = or i64 %318, %316
  %320 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %134)
  %321 = icmp eq i32 %134, %320
  br i1 %321, label %322, label %342

322:                                              ; preds = %312
  %323 = load atomic i64, ptr addrspace(1) %138 syncscope("one-as") acquire, align 8
  %324 = load ptr addrspace(1), ptr addrspace(1) %133, align 8, !tbaa !16
  %325 = load i64, ptr addrspace(1) %135, align 8, !tbaa !20
  %326 = and i64 %325, %323
  %327 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %324, i64 %326
  %328 = load atomic i64, ptr addrspace(1) %327 syncscope("one-as") monotonic, align 8
  %329 = cmpxchg ptr addrspace(1) %138, i64 %323, i64 %328 syncscope("one-as") acquire monotonic, align 8
  %330 = extractvalue { i64, i1 } %329, 1
  %331 = extractvalue { i64, i1 } %329, 0
  br i1 %330, label %342, label %332

332:                                              ; preds = %322, %332
  %333 = phi i64 [ %341, %332 ], [ %331, %322 ]
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  %334 = load ptr addrspace(1), ptr addrspace(1) %133, align 8, !tbaa !16
  %335 = load i64, ptr addrspace(1) %135, align 8, !tbaa !20
  %336 = and i64 %335, %333
  %337 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %334, i64 %336
  %338 = load atomic i64, ptr addrspace(1) %337 syncscope("one-as") monotonic, align 8
  %339 = cmpxchg ptr addrspace(1) %138, i64 %333, i64 %338 syncscope("one-as") acquire monotonic, align 8
  %340 = extractvalue { i64, i1 } %339, 1
  %341 = extractvalue { i64, i1 } %339, 0
  br i1 %340, label %342, label %332

342:                                              ; preds = %332, %322, %312
  %343 = phi i64 [ 0, %312 ], [ %331, %322 ], [ %341, %332 ]
  %344 = tail call i64 @llvm.amdgcn.readfirstlane.i64(i64 %343)
  %345 = load ptr addrspace(1), ptr addrspace(1) %133, align 8, !tbaa !16
  %346 = load i64, ptr addrspace(1) %135, align 8, !tbaa !20
  %347 = and i64 %346, %344
  %348 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %345, i64 %347
  %349 = load ptr addrspace(1), ptr addrspace(1) %136, align 8, !tbaa !21
  %350 = getelementptr inbounds nuw [4096 x i8], ptr addrspace(1) %349, i64 %347
  %351 = tail call i32 @llvm.amdgcn.ballot.i32(i1 true)
  br i1 %321, label %352, label %356

352:                                              ; preds = %342
  %353 = getelementptr inbounds nuw i8, ptr addrspace(1) %348, i64 8
  %354 = getelementptr inbounds nuw i8, ptr addrspace(1) %348, i64 16
  %355 = zext i32 %351 to i64
  store i64 %355, ptr addrspace(1) %353, align 8, !tbaa !22
  store <2 x i32> <i32 2, i32 1>, ptr addrspace(1) %354, align 8, !tbaa !9
  br label %356

356:                                              ; preds = %352, %342
  %357 = getelementptr inbounds nuw [64 x i8], ptr addrspace(1) %350, i64 %137
  store i64 %319, ptr addrspace(1) %357, align 8, !tbaa !14
  %358 = getelementptr inbounds nuw i8, ptr addrspace(1) %357, i64 8
  store i64 %173, ptr addrspace(1) %358, align 8, !tbaa !14
  %359 = getelementptr inbounds nuw i8, ptr addrspace(1) %357, i64 16
  store i64 %197, ptr addrspace(1) %359, align 8, !tbaa !14
  %360 = getelementptr inbounds nuw i8, ptr addrspace(1) %357, i64 24
  store i64 %221, ptr addrspace(1) %360, align 8, !tbaa !14
  %361 = getelementptr inbounds nuw i8, ptr addrspace(1) %357, i64 32
  store i64 %245, ptr addrspace(1) %361, align 8, !tbaa !14
  %362 = getelementptr inbounds nuw i8, ptr addrspace(1) %357, i64 40
  store i64 %269, ptr addrspace(1) %362, align 8, !tbaa !14
  %363 = getelementptr inbounds nuw i8, ptr addrspace(1) %357, i64 48
  store i64 %293, ptr addrspace(1) %363, align 8, !tbaa !14
  %364 = getelementptr inbounds nuw i8, ptr addrspace(1) %357, i64 56
  store i64 %313, ptr addrspace(1) %364, align 8, !tbaa !14
  br i1 %321, label %365, label %392

365:                                              ; preds = %356
  %366 = load atomic i64, ptr addrspace(1) %139 syncscope("one-as") monotonic, align 8
  %367 = load i64, ptr addrspace(1) %135, align 8, !tbaa !20
  %368 = and i64 %367, %344
  %369 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %345, i64 %368
  store i64 %366, ptr addrspace(1) %369, align 8, !tbaa !24
  %370 = cmpxchg ptr addrspace(1) %139, i64 %366, i64 %344 syncscope("one-as") release monotonic, align 8
  %371 = extractvalue { i64, i1 } %370, 1
  br i1 %371, label %377, label %372

372:                                              ; preds = %365, %372
  %373 = phi { i64, i1 } [ %375, %372 ], [ %370, %365 ]
  %374 = extractvalue { i64, i1 } %373, 0
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %374, ptr addrspace(1) %369, align 8, !tbaa !24
  %375 = cmpxchg ptr addrspace(1) %139, i64 %374, i64 %344 syncscope("one-as") release monotonic, align 8
  %376 = extractvalue { i64, i1 } %375, 1
  br i1 %376, label %377, label %372

377:                                              ; preds = %372, %365
  %378 = load i64, ptr addrspace(1) %140, align 8
  %379 = inttoptr i64 %378 to ptr addrspace(1)
  %380 = getelementptr inbounds nuw i8, ptr addrspace(1) %379, i64 8
  %381 = atomicrmw add ptr addrspace(1) %380, i64 1 syncscope("one-as") release, align 8, !amdgpu.no.fine.grained.memory !25, !amdgpu.no.remote.memory !25
  %382 = getelementptr inbounds nuw i8, ptr addrspace(1) %379, i64 16
  %383 = load i64, ptr addrspace(1) %382, align 16, !tbaa !26
  %384 = icmp eq i64 %383, 0
  br i1 %384, label %392, label %385

385:                                              ; preds = %377
  %386 = inttoptr i64 %383 to ptr addrspace(1)
  %387 = getelementptr inbounds nuw i8, ptr addrspace(1) %379, i64 24
  %388 = load i32, ptr addrspace(1) %387, align 8, !tbaa !28
  %389 = zext i32 %388 to i64
  store atomic i64 %389, ptr addrspace(1) %386 syncscope("one-as") release, align 8
  %390 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %388)
  %391 = and i32 %390, 16777215
  tail call void @llvm.amdgcn.s.sendmsg(i32 1, i32 %391)
  br label %392

392:                                              ; preds = %385, %377, %356
  %393 = getelementptr inbounds nuw i8, ptr addrspace(1) %348, i64 20
  br label %394

394:                                              ; preds = %402, %392
  br i1 %321, label %395, label %398

395:                                              ; preds = %394
  %396 = load atomic i32, ptr addrspace(1) %393 syncscope("one-as") acquire, align 4
  %397 = and i32 %396, 1
  br label %398

398:                                              ; preds = %395, %394
  %399 = phi i32 [ %397, %395 ], [ 1, %394 ]
  %400 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %399)
  %401 = icmp eq i32 %400, 0
  br i1 %401, label %403, label %402

402:                                              ; preds = %398
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  br label %394

403:                                              ; preds = %398
  %404 = load i64, ptr addrspace(1) %357, align 8, !tbaa !14
  %405 = load i64, ptr addrspace(1) %358, align 8, !tbaa !14
  br i1 %321, label %406, label %423

406:                                              ; preds = %403
  %407 = load i64, ptr addrspace(1) %135, align 8, !tbaa !20
  %408 = add i64 %407, 1
  %409 = add i64 %408, %344
  %410 = icmp eq i64 %409, 0
  %411 = select i1 %410, i64 %408, i64 %409
  %412 = load atomic i64, ptr addrspace(1) %138 syncscope("one-as") monotonic, align 8
  %413 = load ptr addrspace(1), ptr addrspace(1) %133, align 8, !tbaa !16
  %414 = and i64 %411, %407
  %415 = getelementptr inbounds nuw [24 x i8], ptr addrspace(1) %413, i64 %414
  store i64 %412, ptr addrspace(1) %415, align 8, !tbaa !24
  %416 = cmpxchg ptr addrspace(1) %138, i64 %412, i64 %411 syncscope("one-as") release monotonic, align 8
  %417 = extractvalue { i64, i1 } %416, 1
  br i1 %417, label %423, label %418

418:                                              ; preds = %406, %418
  %419 = phi { i64, i1 } [ %421, %418 ], [ %416, %406 ]
  %420 = extractvalue { i64, i1 } %419, 0
  tail call void @llvm.amdgcn.s.sleep(i32 1)
  store i64 %420, ptr addrspace(1) %415, align 8, !tbaa !24
  %421 = cmpxchg ptr addrspace(1) %138, i64 %420, i64 %411 syncscope("one-as") release monotonic, align 8
  %422 = extractvalue { i64, i1 } %421, 1
  br i1 %422, label %423, label %418

423:                                              ; preds = %418, %403, %406
  %424 = insertelement <2 x i64> poison, i64 %404, i64 0
  %425 = insertelement <2 x i64> %424, i64 %405, i64 1
  %426 = sub i64 %142, %147
  %427 = getelementptr inbounds nuw i8, ptr %143, i64 %147
  %428 = icmp eq i64 %426, 0
  br i1 %428, label %429, label %141

429:                                              ; preds = %423, %121, %108, %107
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #12

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare float @llvm.amdgcn.readlane.f32(float, i32) #7

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ballot.i32(i1) #9

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) noalias writeonly captures(none), ptr addrspace(4) noalias readonly captures(none), i64, i1 immarg) #3

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare ptr @llvm.amdgcn.readfirstlane.p0(ptr) #9

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.fshl.i32(i32, i32, i32) #12

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.vector.reduce.fadd.v4f32(float, <4 x float>) #12

attributes #0 = { mustprogress noreturn nounwind uwtable "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1250" "uniform-work-group-size" }
attributes #1 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #2 = { convergent noinline nounwind uwtable "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1250" "uniform-work-group-size" }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { mustprogress noinline nounwind uwtable "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1250" "uniform-work-group-size" }
attributes #5 = { convergent mustprogress norecurse nounwind uwtable "amdgpu-flat-work-group-size"="1,1024" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1250" "uniform-work-group-size" }
attributes #6 = { convergent mustprogress nofree norecurse nounwind willreturn memory(none) uwtable "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1250" "uniform-work-group-size" }
attributes #7 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #8 = { alwaysinline convergent mustprogress nounwind uwtable "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1250" "uniform-work-group-size" }
attributes #9 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #10 = { convergent nocallback nofree nounwind willreturn }
attributes #11 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #12 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #13 = { convergent nocallback nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) }
attributes #14 = { nocallback nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) }
attributes #15 = { nocallback nofree nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #16 = { nocallback nofree nounwind willreturn }
attributes #17 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
attributes #18 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #19 = { nocallback nofree nosync nounwind willreturn }
attributes #20 = { nocallback nounwind willreturn }
attributes #21 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #22 = { convergent norecurse nounwind uwtable "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1250" "target-features"="+16-bit-insts,+add-min-max-insts,+ashr-pk-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+atomic-global-pk-add-bf16-inst,+bf16-cvt-insts,+bf16-pk-insts,+bf16-trans-insts,+bitop3-insts,+ci-insts,+clusters,+cube-insts,+cvt-pknorm-vop2-insts,+dl-insts,+dot7-insts,+dot8-insts,+dpp,+fp8-conversion-insts,+fp8e5m3-insts,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx12-insts,+gfx1250-insts,+gfx8-insts,+gfx9-insts,+lerp-inst,+mcast-load-insts,+mqsad-insts,+mqsad-pk-insts,+msad-insts,+permlane16-swap,+pk-add-min-max-insts,+prng-inst,+qsad-insts,+s-wakeup-barrier-inst,+sad-insts,+setprio-inc-wg-inst,+swmmac-gfx1200-insts,+swmmac-gfx1250-insts,+tanh-insts,+tensor-cvt-lut-insts,+transpose-load-f4f6-insts,+vmem-pref-insts,+wavefrontsize32" }
attributes #23 = { nounwind }
attributes #24 = { convergent nounwind "uniform-work-group-size" }
attributes #25 = { convergent nounwind }
attributes #26 = { convergent nounwind memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}
!llvm.errno.tbaa = !{!5, !9}
!opencl.ocl.version = !{!13}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 600}
!1 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git d177931e65e6057dca59f1c5b95e643d29876db3)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !11, i64 0}
!11 = !{!"omnipotent char", !12, i64 0}
!12 = !{!"Simple C/C++ TBAA"}
!13 = !{i32 2, i32 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"long", !11, i64 0}
!16 = !{!17, !18, i64 0}
!17 = !{!"", !18, i64 0, !18, i64 8, !19, i64 16, !15, i64 24, !15, i64 32, !15, i64 40}
!18 = !{!"any pointer", !11, i64 0}
!19 = !{!"hsa_signal_s", !15, i64 0}
!20 = !{!17, !15, i64 40}
!21 = !{!17, !18, i64 8}
!22 = !{!23, !15, i64 8}
!23 = !{!"", !15, i64 0, !15, i64 8, !10, i64 16, !10, i64 20}
!24 = !{!23, !15, i64 0}
!25 = !{}
!26 = !{!27, !15, i64 16}
!27 = !{!"amd_signal_s", !15, i64 0, !11, i64 8, !15, i64 16, !10, i64 24, !10, i64 28, !15, i64 32, !15, i64 40, !11, i64 48, !11, i64 56}
!28 = !{!27, !10, i64 24}
!29 = !{!7, !7, i64 0}
!30 = distinct !{!30, !31}
!31 = !{!"llvm.loop.mustprogress"}
!32 = !{!11, !11, i64 0}
!33 = distinct !{!33, !31}
!34 = distinct !{!34, !31}
!35 = distinct !{!35, !31}
!36 = !{i64 4440571, i64 4440587, i64 4440623, i64 4440659, i64 4440695, i64 4440731, i64 4440767, i64 4440803, i64 4440839, i64 4440875, i64 4440911, i64 4440947, i64 4440983, i64 4441019, i64 4441055, i64 4441091, i64 4441127, i64 4441163}
!37 = !{i64 8527420}
!38 = !{i64 8527493}
!39 = !{!40, !40, i64 0}
!40 = !{!"float", !7, i64 0}
!41 = distinct !{!41, !31}
!42 = !{i64 8766515}
!43 = distinct !{!43, !31, !44}
!44 = !{!"llvm.loop.unroll.disable"}

@__dynamic_shared_0 = external addrspace(3) global [0 x i8], align 16
@native_qa_named_barrier = internal addrspace(3) global target("amdgcn.named.barrier", 0) poison
declare void @llvm.amdgcn.s.barrier.join(ptr addrspace(3))
declare void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3), i32)
declare void @llvm.amdgcn.s.barrier.wait(i16 immarg)
define internal void @mhc_pre_subgroup_barrier() alwaysinline {
 call void @llvm.amdgcn.s.barrier.join(ptr addrspace(3) @native_qa_named_barrier)
 call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(3) @native_qa_named_barrier, i32 4)
 call void @llvm.amdgcn.s.barrier.wait(i16 1)
 ret void
}
