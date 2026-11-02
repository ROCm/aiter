; ModuleID = '/work/c16-native-reference/ffn/native_hip.bc'
source_filename = "/work/c16-native-reference/ffn/native_hip.cpp"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

$_ZN7ck_tileL11flush_cacheEv.intern.6baeab4a1f11b097 = comdat any

$_ZZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELi2ELi1EEEvPT0_PfPKT_PKfliilPKiijE11s_scale_run = comdat any

@__const.__assert_fail.fmt = private unnamed_addr addrspace(4) constant [47 x i8] c"%s:%u: %s: Device-side assertion `%s' failed.\0A\00", align 16
@_ZZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELi2ELi1EEEvPT0_PfPKT_PKfliilPKiijE11s_scale_run = linkonce_odr hidden local_unnamed_addr addrspace(3) global [4 x i32] undef, comdat, align 16
@__hip_cuid_6baeab4a1f11b097 = addrspace(1) global i8 0

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
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %5) #21
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
  call fastcc void @__ockl_fprintf_append_string_n(i64 noundef %1501, ptr noundef %0, i64 noundef %1506, i32 noundef 1) #22
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
define weak_odr amdgpu_kernel void @_ZN7ck_tileL11flush_cacheEv.intern.6baeab4a1f11b097() local_unnamed_addr #5 comdat {
  tail call void asm sideeffect "s_icache_inv \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09s_nop 0 \0A\09", ""() #23, !srcloc !36
  ret void
}

; Function Attrs: alwaysinline convergent mustprogress nounwind uwtable
define hidden void @ffn_quant4096(i64 noundef %0, i64 noundef %1, i64 noundef %2, i32 noundef %3) #6 {
  %5 = inttoptr i64 %0 to ptr
  %6 = inttoptr i64 %1 to ptr
  %7 = inttoptr i64 %2 to ptr
  tail call void @llvm.experimental.noalias.scope.decl(metadata !37)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !40)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !42)
  %8 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %9 = and i32 %8, 3
  %10 = zext i32 %3 to i64
  %11 = lshr i32 %8, 2
  %12 = zext nneg i32 %11 to i64
  %13 = tail call noundef dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %14 = load i32, ptr addrspace(4) %13, align 8, !tbaa !9, !noalias !44
  %15 = icmp ugt i32 %14, 2047
  %16 = and i64 %12, 15
  %17 = lshr i64 %12, 4
  %18 = add nuw nsw i64 %17, %10
  %19 = trunc i64 %18 to i32
  %20 = icmp slt i32 %19, 32
  br i1 %15, label %21, label %282

21:                                               ; preds = %4
  br i1 %20, label %22, label %240

22:                                               ; preds = %21
  %23 = shl nuw nsw i64 %16, 12
  %24 = getelementptr inbounds nuw [2 x i8], ptr %7, i64 %23
  %25 = shl nsw i32 %19, 7
  %26 = sext i32 %25 to i64
  %27 = getelementptr inbounds [2 x i8], ptr %24, i64 %26
  %28 = zext nneg i32 %9 to i64
  %29 = getelementptr inbounds nuw [32 x i8], ptr %27, i64 %28
  %30 = load <16 x bfloat>, ptr %29, align 32, !tbaa !29, !alias.scope !42, !noalias !46
  %31 = getelementptr i8, ptr %29, i64 128
  %32 = load <16 x bfloat>, ptr %31, align 32, !tbaa !29, !alias.scope !42, !noalias !46
  %33 = shufflevector <16 x bfloat> %30, <16 x bfloat> %32, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %34 = bitcast <32 x bfloat> %33 to <16 x i32>
  %35 = extractelement <16 x i32> %34, i64 0
  %36 = bitcast i32 %35 to <2 x i16>
  %37 = extractelement <16 x i32> %34, i64 1
  %38 = bitcast i32 %37 to <2 x i16>
  %39 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %36, <2 x i16> %38)
  %40 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %36, <2 x i16> %38)
  %41 = extractelement <16 x i32> %34, i64 2
  %42 = bitcast i32 %41 to <2 x i16>
  %43 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %39, <2 x i16> %42)
  %44 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %40, <2 x i16> %42)
  %45 = extractelement <16 x i32> %34, i64 3
  %46 = bitcast i32 %45 to <2 x i16>
  %47 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %43, <2 x i16> %46)
  %48 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %44, <2 x i16> %46)
  %49 = extractelement <16 x i32> %34, i64 4
  %50 = bitcast i32 %49 to <2 x i16>
  %51 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %47, <2 x i16> %50)
  %52 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %48, <2 x i16> %50)
  %53 = extractelement <16 x i32> %34, i64 5
  %54 = bitcast i32 %53 to <2 x i16>
  %55 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %51, <2 x i16> %54)
  %56 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %52, <2 x i16> %54)
  %57 = extractelement <16 x i32> %34, i64 6
  %58 = bitcast i32 %57 to <2 x i16>
  %59 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %55, <2 x i16> %58)
  %60 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %56, <2 x i16> %58)
  %61 = extractelement <16 x i32> %34, i64 7
  %62 = bitcast i32 %61 to <2 x i16>
  %63 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %59, <2 x i16> %62)
  %64 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %60, <2 x i16> %62)
  %65 = extractelement <16 x i32> %34, i64 8
  %66 = bitcast i32 %65 to <2 x i16>
  %67 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %63, <2 x i16> %66)
  %68 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %64, <2 x i16> %66)
  %69 = extractelement <16 x i32> %34, i64 9
  %70 = bitcast i32 %69 to <2 x i16>
  %71 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %67, <2 x i16> %70)
  %72 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %68, <2 x i16> %70)
  %73 = extractelement <16 x i32> %34, i64 10
  %74 = bitcast i32 %73 to <2 x i16>
  %75 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %71, <2 x i16> %74)
  %76 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %72, <2 x i16> %74)
  %77 = extractelement <16 x i32> %34, i64 11
  %78 = bitcast i32 %77 to <2 x i16>
  %79 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %75, <2 x i16> %78)
  %80 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %76, <2 x i16> %78)
  %81 = extractelement <16 x i32> %34, i64 12
  %82 = bitcast i32 %81 to <2 x i16>
  %83 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %79, <2 x i16> %82)
  %84 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %80, <2 x i16> %82)
  %85 = extractelement <16 x i32> %34, i64 13
  %86 = bitcast i32 %85 to <2 x i16>
  %87 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %83, <2 x i16> %86)
  %88 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %84, <2 x i16> %86)
  %89 = extractelement <16 x i32> %34, i64 14
  %90 = bitcast i32 %89 to <2 x i16>
  %91 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %87, <2 x i16> %90)
  %92 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %88, <2 x i16> %90)
  %93 = extractelement <16 x i32> %34, i64 15
  %94 = bitcast i32 %93 to <2 x i16>
  %95 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %91, <2 x i16> %94)
  %96 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %92, <2 x i16> %94)
  %97 = bitcast <2 x i16> %95 to i32
  %98 = and i32 %97, 2147450879
  %99 = bitcast i32 %98 to <2 x i16>
  %100 = bitcast <2 x i16> %96 to i32
  %101 = and i32 %100, 2147450879
  %102 = bitcast i32 %101 to <2 x i16>
  %103 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %99, <2 x i16> %102)
  %104 = extractelement <2 x i16> %103, i64 0
  %105 = extractelement <2 x i16> %103, i64 1
  %106 = tail call i16 @llvm.umax.i16(i16 %104, i16 %105)
  %107 = icmp ugt i16 %106, 32640
  br i1 %107, label %108, label %176

108:                                              ; preds = %22
  %109 = shufflevector <16 x bfloat> %30, <16 x bfloat> %32, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %110 = tail call contract <32 x bfloat> @llvm.fabs.v32bf16(<32 x bfloat> %109)
  %111 = fpext <32 x bfloat> %110 to <32 x float>
  %112 = extractelement <32 x float> %111, i64 0
  %113 = tail call nsz contract noundef float @llvm.maxnum.f32(float %112, float 1.000000e-10)
  %114 = extractelement <32 x float> %111, i64 1
  %115 = tail call nsz contract noundef float @llvm.maxnum.f32(float %113, float %114)
  %116 = extractelement <32 x float> %111, i64 2
  %117 = tail call nsz contract noundef float @llvm.maxnum.f32(float %115, float %116)
  %118 = extractelement <32 x float> %111, i64 3
  %119 = tail call nsz contract noundef float @llvm.maxnum.f32(float %117, float %118)
  %120 = extractelement <32 x float> %111, i64 4
  %121 = tail call nsz contract noundef float @llvm.maxnum.f32(float %119, float %120)
  %122 = extractelement <32 x float> %111, i64 5
  %123 = tail call nsz contract noundef float @llvm.maxnum.f32(float %121, float %122)
  %124 = extractelement <32 x float> %111, i64 6
  %125 = tail call nsz contract noundef float @llvm.maxnum.f32(float %123, float %124)
  %126 = extractelement <32 x float> %111, i64 7
  %127 = tail call nsz contract noundef float @llvm.maxnum.f32(float %125, float %126)
  %128 = extractelement <32 x float> %111, i64 8
  %129 = tail call nsz contract noundef float @llvm.maxnum.f32(float %127, float %128)
  %130 = extractelement <32 x float> %111, i64 9
  %131 = tail call nsz contract noundef float @llvm.maxnum.f32(float %129, float %130)
  %132 = extractelement <32 x float> %111, i64 10
  %133 = tail call nsz contract noundef float @llvm.maxnum.f32(float %131, float %132)
  %134 = extractelement <32 x float> %111, i64 11
  %135 = tail call nsz contract noundef float @llvm.maxnum.f32(float %133, float %134)
  %136 = extractelement <32 x float> %111, i64 12
  %137 = tail call nsz contract noundef float @llvm.maxnum.f32(float %135, float %136)
  %138 = extractelement <32 x float> %111, i64 13
  %139 = tail call nsz contract noundef float @llvm.maxnum.f32(float %137, float %138)
  %140 = extractelement <32 x float> %111, i64 14
  %141 = tail call nsz contract noundef float @llvm.maxnum.f32(float %139, float %140)
  %142 = extractelement <32 x float> %111, i64 15
  %143 = tail call nsz contract noundef float @llvm.maxnum.f32(float %141, float %142)
  %144 = extractelement <32 x float> %111, i64 16
  %145 = tail call nsz contract noundef float @llvm.maxnum.f32(float %143, float %144)
  %146 = extractelement <32 x float> %111, i64 17
  %147 = tail call nsz contract noundef float @llvm.maxnum.f32(float %145, float %146)
  %148 = extractelement <32 x float> %111, i64 18
  %149 = tail call nsz contract noundef float @llvm.maxnum.f32(float %147, float %148)
  %150 = extractelement <32 x float> %111, i64 19
  %151 = tail call nsz contract noundef float @llvm.maxnum.f32(float %149, float %150)
  %152 = extractelement <32 x float> %111, i64 20
  %153 = tail call nsz contract noundef float @llvm.maxnum.f32(float %151, float %152)
  %154 = extractelement <32 x float> %111, i64 21
  %155 = tail call nsz contract noundef float @llvm.maxnum.f32(float %153, float %154)
  %156 = extractelement <32 x float> %111, i64 22
  %157 = tail call nsz contract noundef float @llvm.maxnum.f32(float %155, float %156)
  %158 = extractelement <32 x float> %111, i64 23
  %159 = tail call nsz contract noundef float @llvm.maxnum.f32(float %157, float %158)
  %160 = extractelement <32 x float> %111, i64 24
  %161 = tail call nsz contract noundef float @llvm.maxnum.f32(float %159, float %160)
  %162 = extractelement <32 x float> %111, i64 25
  %163 = tail call nsz contract noundef float @llvm.maxnum.f32(float %161, float %162)
  %164 = extractelement <32 x float> %111, i64 26
  %165 = tail call nsz contract noundef float @llvm.maxnum.f32(float %163, float %164)
  %166 = extractelement <32 x float> %111, i64 27
  %167 = tail call nsz contract noundef float @llvm.maxnum.f32(float %165, float %166)
  %168 = extractelement <32 x float> %111, i64 28
  %169 = tail call nsz contract noundef float @llvm.maxnum.f32(float %167, float %168)
  %170 = extractelement <32 x float> %111, i64 29
  %171 = tail call nsz contract noundef float @llvm.maxnum.f32(float %169, float %170)
  %172 = extractelement <32 x float> %111, i64 30
  %173 = tail call nsz contract noundef float @llvm.maxnum.f32(float %171, float %172)
  %174 = extractelement <32 x float> %111, i64 31
  %175 = tail call nsz contract noundef float @llvm.maxnum.f32(float %173, float %174)
  br label %181

176:                                              ; preds = %22
  %177 = zext nneg i16 %106 to i32
  %178 = shl nuw nsw i32 %177, 16
  %179 = bitcast i32 %178 to float
  %180 = tail call nsz contract noundef float @llvm.maxnum.f32(float %179, float 1.000000e-10)
  br label %181

181:                                              ; preds = %176, %108
  %182 = phi float [ %180, %176 ], [ %175, %108 ]
  %183 = bitcast float %182 to i32
  %184 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %183, i32 177, i32 15, i32 15, i1 false)
  %185 = bitcast i32 %184 to float
  %186 = fcmp contract ogt float %182, %185
  %187 = select contract i1 %186, float %182, float %185
  %188 = bitcast float %187 to i32
  %189 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %188, i32 78, i32 15, i32 15, i1 false)
  %190 = bitcast i32 %189 to float
  %191 = fcmp contract ogt float %187, %190
  %192 = select contract i1 %191, float %187, float %190
  %193 = fmul contract float %192, f0x3B124925
  %194 = tail call float @llvm.fabs.f32(float %193)
  %195 = bitcast float %194 to i32
  %196 = add nuw i32 %195, 8388607
  %197 = lshr i32 %196, 23
  %198 = tail call noundef i32 @llvm.umin.i32(i32 %197, i32 255)
  %199 = shl nuw nsw i32 %198, 23
  %200 = bitcast i32 %199 to float
  %201 = tail call nsz contract noundef float @llvm.minnum.f32(float %200, float f0x7F000000)
  %202 = shl nuw nsw i32 %9, 4
  %203 = icmp eq i32 %9, 0
  br i1 %203, label %204, label %213

204:                                              ; preds = %181
  %205 = shl i64 %18, 32
  %206 = ashr exact i64 %205, 28
  %207 = getelementptr i8, ptr %6, i64 %206
  %208 = getelementptr i8, ptr %207, i64 %16
  %209 = bitcast float %201 to i32
  %210 = lshr i32 %209, 23
  %211 = trunc nuw i32 %210 to i8
  %212 = icmp eq ptr %208, null
  br label %213

213:                                              ; preds = %204, %181
  %214 = phi i8 [ %211, %204 ], [ 0, %181 ]
  %215 = phi i1 [ %212, %204 ], [ true, %181 ]
  %216 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %5, i16 0, i64 65536, i32 822099968)
  %217 = trunc nuw nsw i64 %23 to i32
  %218 = or disjoint i32 %202, %217
  %219 = add nsw i32 %218, %25
  %220 = shufflevector <16 x bfloat> %30, <16 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %221 = tail call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.fp8.bf16(<8 x bfloat> %220, float %201)
  %222 = bitcast <16 x bfloat> %30 to <2 x i128>
  %223 = extractelement <2 x i128> %222, i64 1
  %224 = bitcast i128 %223 to <8 x bfloat>
  %225 = shufflevector <16 x bfloat> %30, <16 x bfloat> poison, <8 x i32> <i32 poison, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %226 = shufflevector <8 x bfloat> %224, <8 x bfloat> %225, <8 x i32> <i32 0, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %227 = tail call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.fp8.bf16(<8 x bfloat> %226, float %201)
  %228 = shufflevector <2 x i32> %221, <2 x i32> %227, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %228, ptr addrspace(8) %216, i32 %219, i32 0, i32 0), !alias.scope !37, !noalias !47
  tail call void asm sideeffect "s_nop 0", ""() #23, !noalias !44, !srcloc !48
  %229 = shufflevector <16 x bfloat> %32, <16 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %230 = tail call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.fp8.bf16(<8 x bfloat> %229, float %201)
  %231 = bitcast <16 x bfloat> %32 to <2 x i128>
  %232 = extractelement <2 x i128> %231, i64 1
  %233 = bitcast i128 %232 to <8 x bfloat>
  %234 = shufflevector <16 x bfloat> %32, <16 x bfloat> poison, <8 x i32> <i32 poison, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %235 = shufflevector <8 x bfloat> %233, <8 x bfloat> %234, <8 x i32> <i32 0, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %236 = tail call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.fp8.bf16(<8 x bfloat> %235, float %201)
  %237 = shufflevector <2 x i32> %230, <2 x i32> %236, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %237, ptr addrspace(8) %216, i32 %219, i32 64, i32 0), !alias.scope !37, !noalias !47
  tail call void asm sideeffect "s_nop 0", ""() #23, !noalias !44, !srcloc !48
  br i1 %215, label %240, label %238

238:                                              ; preds = %213
  %239 = getelementptr inbounds nuw i8, ptr addrspace(3) @_ZZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELi2ELi1EEEvPT0_PfPKT_PKfliilPKiijE11s_scale_run, i32 %11
  store i8 %214, ptr addrspace(3) %239, align 1, !tbaa !29, !noalias !44
  br label %240

240:                                              ; preds = %238, %213, %21
  tail call void @llvm.amdgcn.s.wait.dscnt(i16 noundef signext 0) #22, !noalias !44
  tail call void @llvm.amdgcn.s.barrier(), !noalias !44
  %241 = and i64 %1, 3
  %242 = icmp eq i64 %241, 0
  %243 = icmp samesign ult i32 %8, 32
  br i1 %243, label %244, label %500

244:                                              ; preds = %240
  %245 = tail call i64 @llvm.amdgcn.readfirstlane.i64(i64 %16)
  %246 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %19)
  %247 = select i1 %242, i32 2, i32 0
  %248 = shl nuw nsw i32 %8, %247
  %249 = icmp samesign ult i32 %248, 16
  br i1 %249, label %250, label %500

250:                                              ; preds = %244
  %251 = shl i64 %245, 32
  %252 = ashr exact i64 %251, 32
  %253 = zext nneg i32 %248 to i64
  %254 = add nsw i64 %252, %253
  %255 = icmp slt i64 %254, 16
  br i1 %255, label %266, label %256

256:                                              ; preds = %250
  %257 = tail call i64 @llvm.usub.sat.i64(i64 %254, i64 31)
  %258 = add nuw i64 %257, 15
  %259 = lshr i64 %258, 4
  %260 = and i64 %258, 9223372036854775792
  %261 = trunc nuw nsw i64 %259 to i32
  %262 = add i32 %246, %261
  %263 = add nsw i64 %254, -16
  %264 = sub nsw i64 %263, %260
  %265 = add i32 %262, 1
  br label %266

266:                                              ; preds = %256, %250
  %267 = phi i64 [ %254, %250 ], [ %264, %256 ]
  %268 = phi i32 [ %246, %250 ], [ %265, %256 ]
  %269 = sext i32 %268 to i64
  %270 = shl nsw i64 %269, 4
  %271 = getelementptr inbounds i8, ptr %6, i64 %270
  %272 = getelementptr inbounds i8, ptr %271, i64 %267
  %273 = icmp slt i32 %268, 32
  br i1 %273, label %274, label %500

274:                                              ; preds = %266
  br i1 %242, label %275, label %279

275:                                              ; preds = %274
  %276 = lshr i32 %248, 2
  %277 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @_ZZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELi2ELi1EEEvPT0_PfPKT_PKfliilPKiijE11s_scale_run, i32 %276
  %278 = load i32, ptr addrspace(3) %277, align 4, !tbaa !5, !noalias !44
  store i32 %278, ptr %272, align 4, !tbaa !5, !alias.scope !40, !noalias !49
  br label %500

279:                                              ; preds = %274
  %280 = getelementptr inbounds nuw i8, ptr addrspace(3) @_ZZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELi2ELi1EEEvPT0_PfPKT_PKfliilPKiijE11s_scale_run, i32 %248
  %281 = load i8, ptr addrspace(3) %280, align 1, !tbaa !29, !noalias !44
  store i8 %281, ptr %272, align 1, !tbaa !29, !alias.scope !40, !noalias !49
  br label %500

282:                                              ; preds = %4
  br i1 %20, label %283, label %500

283:                                              ; preds = %282
  %284 = shl nuw nsw i64 %16, 12
  %285 = getelementptr inbounds nuw [2 x i8], ptr %7, i64 %284
  %286 = shl nsw i32 %19, 7
  %287 = sext i32 %286 to i64
  %288 = getelementptr inbounds [2 x i8], ptr %285, i64 %287
  %289 = zext nneg i32 %9 to i64
  %290 = getelementptr inbounds nuw [32 x i8], ptr %288, i64 %289
  %291 = load <16 x bfloat>, ptr %290, align 32, !tbaa !29, !alias.scope !42, !noalias !46
  %292 = getelementptr i8, ptr %290, i64 128
  %293 = load <16 x bfloat>, ptr %292, align 32, !tbaa !29, !alias.scope !42, !noalias !46
  %294 = shufflevector <16 x bfloat> %291, <16 x bfloat> %293, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %295 = bitcast <32 x bfloat> %294 to <16 x i32>
  %296 = extractelement <16 x i32> %295, i64 0
  %297 = bitcast i32 %296 to <2 x i16>
  %298 = extractelement <16 x i32> %295, i64 1
  %299 = bitcast i32 %298 to <2 x i16>
  %300 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %297, <2 x i16> %299)
  %301 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %297, <2 x i16> %299)
  %302 = extractelement <16 x i32> %295, i64 2
  %303 = bitcast i32 %302 to <2 x i16>
  %304 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %300, <2 x i16> %303)
  %305 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %301, <2 x i16> %303)
  %306 = extractelement <16 x i32> %295, i64 3
  %307 = bitcast i32 %306 to <2 x i16>
  %308 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %304, <2 x i16> %307)
  %309 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %305, <2 x i16> %307)
  %310 = extractelement <16 x i32> %295, i64 4
  %311 = bitcast i32 %310 to <2 x i16>
  %312 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %308, <2 x i16> %311)
  %313 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %309, <2 x i16> %311)
  %314 = extractelement <16 x i32> %295, i64 5
  %315 = bitcast i32 %314 to <2 x i16>
  %316 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %312, <2 x i16> %315)
  %317 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %313, <2 x i16> %315)
  %318 = extractelement <16 x i32> %295, i64 6
  %319 = bitcast i32 %318 to <2 x i16>
  %320 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %316, <2 x i16> %319)
  %321 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %317, <2 x i16> %319)
  %322 = extractelement <16 x i32> %295, i64 7
  %323 = bitcast i32 %322 to <2 x i16>
  %324 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %320, <2 x i16> %323)
  %325 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %321, <2 x i16> %323)
  %326 = extractelement <16 x i32> %295, i64 8
  %327 = bitcast i32 %326 to <2 x i16>
  %328 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %324, <2 x i16> %327)
  %329 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %325, <2 x i16> %327)
  %330 = extractelement <16 x i32> %295, i64 9
  %331 = bitcast i32 %330 to <2 x i16>
  %332 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %328, <2 x i16> %331)
  %333 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %329, <2 x i16> %331)
  %334 = extractelement <16 x i32> %295, i64 10
  %335 = bitcast i32 %334 to <2 x i16>
  %336 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %332, <2 x i16> %335)
  %337 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %333, <2 x i16> %335)
  %338 = extractelement <16 x i32> %295, i64 11
  %339 = bitcast i32 %338 to <2 x i16>
  %340 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %336, <2 x i16> %339)
  %341 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %337, <2 x i16> %339)
  %342 = extractelement <16 x i32> %295, i64 12
  %343 = bitcast i32 %342 to <2 x i16>
  %344 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %340, <2 x i16> %343)
  %345 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %341, <2 x i16> %343)
  %346 = extractelement <16 x i32> %295, i64 13
  %347 = bitcast i32 %346 to <2 x i16>
  %348 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %344, <2 x i16> %347)
  %349 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %345, <2 x i16> %347)
  %350 = extractelement <16 x i32> %295, i64 14
  %351 = bitcast i32 %350 to <2 x i16>
  %352 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %348, <2 x i16> %351)
  %353 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %349, <2 x i16> %351)
  %354 = extractelement <16 x i32> %295, i64 15
  %355 = bitcast i32 %354 to <2 x i16>
  %356 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %352, <2 x i16> %355)
  %357 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %353, <2 x i16> %355)
  %358 = bitcast <2 x i16> %356 to i32
  %359 = and i32 %358, 2147450879
  %360 = bitcast i32 %359 to <2 x i16>
  %361 = bitcast <2 x i16> %357 to i32
  %362 = and i32 %361, 2147450879
  %363 = bitcast i32 %362 to <2 x i16>
  %364 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %360, <2 x i16> %363)
  %365 = extractelement <2 x i16> %364, i64 0
  %366 = extractelement <2 x i16> %364, i64 1
  %367 = tail call i16 @llvm.umax.i16(i16 %365, i16 %366)
  %368 = icmp ugt i16 %367, 32640
  br i1 %368, label %369, label %437

369:                                              ; preds = %283
  %370 = shufflevector <16 x bfloat> %291, <16 x bfloat> %293, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %371 = tail call contract <32 x bfloat> @llvm.fabs.v32bf16(<32 x bfloat> %370)
  %372 = fpext <32 x bfloat> %371 to <32 x float>
  %373 = extractelement <32 x float> %372, i64 0
  %374 = tail call nsz contract noundef float @llvm.maxnum.f32(float %373, float 1.000000e-10)
  %375 = extractelement <32 x float> %372, i64 1
  %376 = tail call nsz contract noundef float @llvm.maxnum.f32(float %374, float %375)
  %377 = extractelement <32 x float> %372, i64 2
  %378 = tail call nsz contract noundef float @llvm.maxnum.f32(float %376, float %377)
  %379 = extractelement <32 x float> %372, i64 3
  %380 = tail call nsz contract noundef float @llvm.maxnum.f32(float %378, float %379)
  %381 = extractelement <32 x float> %372, i64 4
  %382 = tail call nsz contract noundef float @llvm.maxnum.f32(float %380, float %381)
  %383 = extractelement <32 x float> %372, i64 5
  %384 = tail call nsz contract noundef float @llvm.maxnum.f32(float %382, float %383)
  %385 = extractelement <32 x float> %372, i64 6
  %386 = tail call nsz contract noundef float @llvm.maxnum.f32(float %384, float %385)
  %387 = extractelement <32 x float> %372, i64 7
  %388 = tail call nsz contract noundef float @llvm.maxnum.f32(float %386, float %387)
  %389 = extractelement <32 x float> %372, i64 8
  %390 = tail call nsz contract noundef float @llvm.maxnum.f32(float %388, float %389)
  %391 = extractelement <32 x float> %372, i64 9
  %392 = tail call nsz contract noundef float @llvm.maxnum.f32(float %390, float %391)
  %393 = extractelement <32 x float> %372, i64 10
  %394 = tail call nsz contract noundef float @llvm.maxnum.f32(float %392, float %393)
  %395 = extractelement <32 x float> %372, i64 11
  %396 = tail call nsz contract noundef float @llvm.maxnum.f32(float %394, float %395)
  %397 = extractelement <32 x float> %372, i64 12
  %398 = tail call nsz contract noundef float @llvm.maxnum.f32(float %396, float %397)
  %399 = extractelement <32 x float> %372, i64 13
  %400 = tail call nsz contract noundef float @llvm.maxnum.f32(float %398, float %399)
  %401 = extractelement <32 x float> %372, i64 14
  %402 = tail call nsz contract noundef float @llvm.maxnum.f32(float %400, float %401)
  %403 = extractelement <32 x float> %372, i64 15
  %404 = tail call nsz contract noundef float @llvm.maxnum.f32(float %402, float %403)
  %405 = extractelement <32 x float> %372, i64 16
  %406 = tail call nsz contract noundef float @llvm.maxnum.f32(float %404, float %405)
  %407 = extractelement <32 x float> %372, i64 17
  %408 = tail call nsz contract noundef float @llvm.maxnum.f32(float %406, float %407)
  %409 = extractelement <32 x float> %372, i64 18
  %410 = tail call nsz contract noundef float @llvm.maxnum.f32(float %408, float %409)
  %411 = extractelement <32 x float> %372, i64 19
  %412 = tail call nsz contract noundef float @llvm.maxnum.f32(float %410, float %411)
  %413 = extractelement <32 x float> %372, i64 20
  %414 = tail call nsz contract noundef float @llvm.maxnum.f32(float %412, float %413)
  %415 = extractelement <32 x float> %372, i64 21
  %416 = tail call nsz contract noundef float @llvm.maxnum.f32(float %414, float %415)
  %417 = extractelement <32 x float> %372, i64 22
  %418 = tail call nsz contract noundef float @llvm.maxnum.f32(float %416, float %417)
  %419 = extractelement <32 x float> %372, i64 23
  %420 = tail call nsz contract noundef float @llvm.maxnum.f32(float %418, float %419)
  %421 = extractelement <32 x float> %372, i64 24
  %422 = tail call nsz contract noundef float @llvm.maxnum.f32(float %420, float %421)
  %423 = extractelement <32 x float> %372, i64 25
  %424 = tail call nsz contract noundef float @llvm.maxnum.f32(float %422, float %423)
  %425 = extractelement <32 x float> %372, i64 26
  %426 = tail call nsz contract noundef float @llvm.maxnum.f32(float %424, float %425)
  %427 = extractelement <32 x float> %372, i64 27
  %428 = tail call nsz contract noundef float @llvm.maxnum.f32(float %426, float %427)
  %429 = extractelement <32 x float> %372, i64 28
  %430 = tail call nsz contract noundef float @llvm.maxnum.f32(float %428, float %429)
  %431 = extractelement <32 x float> %372, i64 29
  %432 = tail call nsz contract noundef float @llvm.maxnum.f32(float %430, float %431)
  %433 = extractelement <32 x float> %372, i64 30
  %434 = tail call nsz contract noundef float @llvm.maxnum.f32(float %432, float %433)
  %435 = extractelement <32 x float> %372, i64 31
  %436 = tail call nsz contract noundef float @llvm.maxnum.f32(float %434, float %435)
  br label %442

437:                                              ; preds = %283
  %438 = zext nneg i16 %367 to i32
  %439 = shl nuw nsw i32 %438, 16
  %440 = bitcast i32 %439 to float
  %441 = tail call nsz contract noundef float @llvm.maxnum.f32(float %440, float 1.000000e-10)
  br label %442

442:                                              ; preds = %437, %369
  %443 = phi float [ %441, %437 ], [ %436, %369 ]
  %444 = bitcast float %443 to i32
  %445 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %444, i32 177, i32 15, i32 15, i1 false)
  %446 = bitcast i32 %445 to float
  %447 = fcmp contract ogt float %443, %446
  %448 = select contract i1 %447, float %443, float %446
  %449 = bitcast float %448 to i32
  %450 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %449, i32 78, i32 15, i32 15, i1 false)
  %451 = bitcast i32 %450 to float
  %452 = fcmp contract ogt float %448, %451
  %453 = select contract i1 %452, float %448, float %451
  %454 = fmul contract float %453, f0x3B124925
  %455 = tail call float @llvm.fabs.f32(float %454)
  %456 = bitcast float %455 to i32
  %457 = add nuw i32 %456, 8388607
  %458 = lshr i32 %457, 23
  %459 = tail call noundef i32 @llvm.umin.i32(i32 %458, i32 255)
  %460 = shl nuw nsw i32 %459, 23
  %461 = bitcast i32 %460 to float
  %462 = tail call nsz contract noundef float @llvm.minnum.f32(float %461, float f0x7F000000)
  %463 = shl nuw nsw i32 %9, 4
  %464 = icmp eq i32 %9, 0
  br i1 %464, label %465, label %473

465:                                              ; preds = %442
  %466 = shl i64 %18, 32
  %467 = ashr exact i64 %466, 28
  %468 = getelementptr i8, ptr %6, i64 %467
  %469 = getelementptr i8, ptr %468, i64 %16
  %470 = bitcast float %462 to i32
  %471 = lshr i32 %470, 23
  %472 = trunc nuw i32 %471 to i8
  br label %473

473:                                              ; preds = %465, %442
  %474 = phi i8 [ %472, %465 ], [ 0, %442 ]
  %475 = phi ptr [ %469, %465 ], [ null, %442 ]
  %476 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %5, i16 0, i64 65536, i32 822099968)
  %477 = trunc nuw nsw i64 %284 to i32
  %478 = or disjoint i32 %463, %477
  %479 = add nsw i32 %478, %286
  %480 = shufflevector <16 x bfloat> %291, <16 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %481 = tail call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.fp8.bf16(<8 x bfloat> %480, float %462)
  %482 = bitcast <16 x bfloat> %291 to <2 x i128>
  %483 = extractelement <2 x i128> %482, i64 1
  %484 = bitcast i128 %483 to <8 x bfloat>
  %485 = shufflevector <16 x bfloat> %291, <16 x bfloat> poison, <8 x i32> <i32 poison, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %486 = shufflevector <8 x bfloat> %484, <8 x bfloat> %485, <8 x i32> <i32 0, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %487 = tail call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.fp8.bf16(<8 x bfloat> %486, float %462)
  %488 = shufflevector <2 x i32> %481, <2 x i32> %487, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %488, ptr addrspace(8) %476, i32 %479, i32 0, i32 0), !alias.scope !37, !noalias !47
  tail call void asm sideeffect "s_nop 0", ""() #23, !noalias !44, !srcloc !48
  %489 = shufflevector <16 x bfloat> %293, <16 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %490 = tail call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.fp8.bf16(<8 x bfloat> %489, float %462)
  %491 = bitcast <16 x bfloat> %293 to <2 x i128>
  %492 = extractelement <2 x i128> %491, i64 1
  %493 = bitcast i128 %492 to <8 x bfloat>
  %494 = shufflevector <16 x bfloat> %293, <16 x bfloat> poison, <8 x i32> <i32 poison, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %495 = shufflevector <8 x bfloat> %493, <8 x bfloat> %494, <8 x i32> <i32 0, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %496 = tail call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.fp8.bf16(<8 x bfloat> %495, float %462)
  %497 = shufflevector <2 x i32> %490, <2 x i32> %496, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %497, ptr addrspace(8) %476, i32 %479, i32 64, i32 0), !alias.scope !37, !noalias !47
  tail call void asm sideeffect "s_nop 0", ""() #23, !noalias !44, !srcloc !48
  %498 = icmp eq ptr %475, null
  br i1 %498, label %500, label %499

499:                                              ; preds = %473
  store i8 %474, ptr %475, align 1, !tbaa !29, !alias.scope !40, !noalias !49
  br label %500

500:                                              ; preds = %240, %244, %266, %275, %279, %282, %473, %499
  ret void
}

; Function Attrs: alwaysinline convergent mustprogress nounwind uwtable
define hidden void @ffn_quant2048(i64 noundef %0, i64 noundef %1, i64 noundef %2, i32 noundef %3) #6 {
  %5 = inttoptr i64 %0 to ptr
  %6 = inttoptr i64 %1 to ptr
  %7 = inttoptr i64 %2 to ptr
  tail call void @llvm.experimental.noalias.scope.decl(metadata !50)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !53)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !55)
  %8 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %9 = and i32 %8, 3
  %10 = zext i32 %3 to i64
  %11 = lshr i32 %8, 2
  %12 = zext nneg i32 %11 to i64
  %13 = tail call noundef dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %14 = load i32, ptr addrspace(4) %13, align 8, !tbaa !9, !noalias !57
  %15 = icmp ugt i32 %14, 2047
  %16 = and i64 %12, 15
  %17 = lshr i64 %12, 4
  %18 = add nuw nsw i64 %17, %10
  %19 = trunc i64 %18 to i32
  %20 = icmp slt i32 %19, 16
  br i1 %15, label %21, label %282

21:                                               ; preds = %4
  br i1 %20, label %22, label %240

22:                                               ; preds = %21
  %23 = shl nuw nsw i64 %16, 11
  %24 = getelementptr inbounds nuw [2 x i8], ptr %7, i64 %23
  %25 = shl nsw i32 %19, 7
  %26 = sext i32 %25 to i64
  %27 = getelementptr inbounds [2 x i8], ptr %24, i64 %26
  %28 = zext nneg i32 %9 to i64
  %29 = getelementptr inbounds nuw [32 x i8], ptr %27, i64 %28
  %30 = load <16 x bfloat>, ptr %29, align 32, !tbaa !29, !alias.scope !55, !noalias !59
  %31 = getelementptr i8, ptr %29, i64 128
  %32 = load <16 x bfloat>, ptr %31, align 32, !tbaa !29, !alias.scope !55, !noalias !59
  %33 = shufflevector <16 x bfloat> %30, <16 x bfloat> %32, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %34 = bitcast <32 x bfloat> %33 to <16 x i32>
  %35 = extractelement <16 x i32> %34, i64 0
  %36 = bitcast i32 %35 to <2 x i16>
  %37 = extractelement <16 x i32> %34, i64 1
  %38 = bitcast i32 %37 to <2 x i16>
  %39 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %36, <2 x i16> %38)
  %40 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %36, <2 x i16> %38)
  %41 = extractelement <16 x i32> %34, i64 2
  %42 = bitcast i32 %41 to <2 x i16>
  %43 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %39, <2 x i16> %42)
  %44 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %40, <2 x i16> %42)
  %45 = extractelement <16 x i32> %34, i64 3
  %46 = bitcast i32 %45 to <2 x i16>
  %47 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %43, <2 x i16> %46)
  %48 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %44, <2 x i16> %46)
  %49 = extractelement <16 x i32> %34, i64 4
  %50 = bitcast i32 %49 to <2 x i16>
  %51 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %47, <2 x i16> %50)
  %52 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %48, <2 x i16> %50)
  %53 = extractelement <16 x i32> %34, i64 5
  %54 = bitcast i32 %53 to <2 x i16>
  %55 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %51, <2 x i16> %54)
  %56 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %52, <2 x i16> %54)
  %57 = extractelement <16 x i32> %34, i64 6
  %58 = bitcast i32 %57 to <2 x i16>
  %59 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %55, <2 x i16> %58)
  %60 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %56, <2 x i16> %58)
  %61 = extractelement <16 x i32> %34, i64 7
  %62 = bitcast i32 %61 to <2 x i16>
  %63 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %59, <2 x i16> %62)
  %64 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %60, <2 x i16> %62)
  %65 = extractelement <16 x i32> %34, i64 8
  %66 = bitcast i32 %65 to <2 x i16>
  %67 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %63, <2 x i16> %66)
  %68 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %64, <2 x i16> %66)
  %69 = extractelement <16 x i32> %34, i64 9
  %70 = bitcast i32 %69 to <2 x i16>
  %71 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %67, <2 x i16> %70)
  %72 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %68, <2 x i16> %70)
  %73 = extractelement <16 x i32> %34, i64 10
  %74 = bitcast i32 %73 to <2 x i16>
  %75 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %71, <2 x i16> %74)
  %76 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %72, <2 x i16> %74)
  %77 = extractelement <16 x i32> %34, i64 11
  %78 = bitcast i32 %77 to <2 x i16>
  %79 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %75, <2 x i16> %78)
  %80 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %76, <2 x i16> %78)
  %81 = extractelement <16 x i32> %34, i64 12
  %82 = bitcast i32 %81 to <2 x i16>
  %83 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %79, <2 x i16> %82)
  %84 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %80, <2 x i16> %82)
  %85 = extractelement <16 x i32> %34, i64 13
  %86 = bitcast i32 %85 to <2 x i16>
  %87 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %83, <2 x i16> %86)
  %88 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %84, <2 x i16> %86)
  %89 = extractelement <16 x i32> %34, i64 14
  %90 = bitcast i32 %89 to <2 x i16>
  %91 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %87, <2 x i16> %90)
  %92 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %88, <2 x i16> %90)
  %93 = extractelement <16 x i32> %34, i64 15
  %94 = bitcast i32 %93 to <2 x i16>
  %95 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %91, <2 x i16> %94)
  %96 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %92, <2 x i16> %94)
  %97 = bitcast <2 x i16> %95 to i32
  %98 = and i32 %97, 2147450879
  %99 = bitcast i32 %98 to <2 x i16>
  %100 = bitcast <2 x i16> %96 to i32
  %101 = and i32 %100, 2147450879
  %102 = bitcast i32 %101 to <2 x i16>
  %103 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %99, <2 x i16> %102)
  %104 = extractelement <2 x i16> %103, i64 0
  %105 = extractelement <2 x i16> %103, i64 1
  %106 = tail call i16 @llvm.umax.i16(i16 %104, i16 %105)
  %107 = icmp ugt i16 %106, 32640
  br i1 %107, label %108, label %176

108:                                              ; preds = %22
  %109 = shufflevector <16 x bfloat> %30, <16 x bfloat> %32, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %110 = tail call contract <32 x bfloat> @llvm.fabs.v32bf16(<32 x bfloat> %109)
  %111 = fpext <32 x bfloat> %110 to <32 x float>
  %112 = extractelement <32 x float> %111, i64 0
  %113 = tail call nsz contract noundef float @llvm.maxnum.f32(float %112, float 1.000000e-10)
  %114 = extractelement <32 x float> %111, i64 1
  %115 = tail call nsz contract noundef float @llvm.maxnum.f32(float %113, float %114)
  %116 = extractelement <32 x float> %111, i64 2
  %117 = tail call nsz contract noundef float @llvm.maxnum.f32(float %115, float %116)
  %118 = extractelement <32 x float> %111, i64 3
  %119 = tail call nsz contract noundef float @llvm.maxnum.f32(float %117, float %118)
  %120 = extractelement <32 x float> %111, i64 4
  %121 = tail call nsz contract noundef float @llvm.maxnum.f32(float %119, float %120)
  %122 = extractelement <32 x float> %111, i64 5
  %123 = tail call nsz contract noundef float @llvm.maxnum.f32(float %121, float %122)
  %124 = extractelement <32 x float> %111, i64 6
  %125 = tail call nsz contract noundef float @llvm.maxnum.f32(float %123, float %124)
  %126 = extractelement <32 x float> %111, i64 7
  %127 = tail call nsz contract noundef float @llvm.maxnum.f32(float %125, float %126)
  %128 = extractelement <32 x float> %111, i64 8
  %129 = tail call nsz contract noundef float @llvm.maxnum.f32(float %127, float %128)
  %130 = extractelement <32 x float> %111, i64 9
  %131 = tail call nsz contract noundef float @llvm.maxnum.f32(float %129, float %130)
  %132 = extractelement <32 x float> %111, i64 10
  %133 = tail call nsz contract noundef float @llvm.maxnum.f32(float %131, float %132)
  %134 = extractelement <32 x float> %111, i64 11
  %135 = tail call nsz contract noundef float @llvm.maxnum.f32(float %133, float %134)
  %136 = extractelement <32 x float> %111, i64 12
  %137 = tail call nsz contract noundef float @llvm.maxnum.f32(float %135, float %136)
  %138 = extractelement <32 x float> %111, i64 13
  %139 = tail call nsz contract noundef float @llvm.maxnum.f32(float %137, float %138)
  %140 = extractelement <32 x float> %111, i64 14
  %141 = tail call nsz contract noundef float @llvm.maxnum.f32(float %139, float %140)
  %142 = extractelement <32 x float> %111, i64 15
  %143 = tail call nsz contract noundef float @llvm.maxnum.f32(float %141, float %142)
  %144 = extractelement <32 x float> %111, i64 16
  %145 = tail call nsz contract noundef float @llvm.maxnum.f32(float %143, float %144)
  %146 = extractelement <32 x float> %111, i64 17
  %147 = tail call nsz contract noundef float @llvm.maxnum.f32(float %145, float %146)
  %148 = extractelement <32 x float> %111, i64 18
  %149 = tail call nsz contract noundef float @llvm.maxnum.f32(float %147, float %148)
  %150 = extractelement <32 x float> %111, i64 19
  %151 = tail call nsz contract noundef float @llvm.maxnum.f32(float %149, float %150)
  %152 = extractelement <32 x float> %111, i64 20
  %153 = tail call nsz contract noundef float @llvm.maxnum.f32(float %151, float %152)
  %154 = extractelement <32 x float> %111, i64 21
  %155 = tail call nsz contract noundef float @llvm.maxnum.f32(float %153, float %154)
  %156 = extractelement <32 x float> %111, i64 22
  %157 = tail call nsz contract noundef float @llvm.maxnum.f32(float %155, float %156)
  %158 = extractelement <32 x float> %111, i64 23
  %159 = tail call nsz contract noundef float @llvm.maxnum.f32(float %157, float %158)
  %160 = extractelement <32 x float> %111, i64 24
  %161 = tail call nsz contract noundef float @llvm.maxnum.f32(float %159, float %160)
  %162 = extractelement <32 x float> %111, i64 25
  %163 = tail call nsz contract noundef float @llvm.maxnum.f32(float %161, float %162)
  %164 = extractelement <32 x float> %111, i64 26
  %165 = tail call nsz contract noundef float @llvm.maxnum.f32(float %163, float %164)
  %166 = extractelement <32 x float> %111, i64 27
  %167 = tail call nsz contract noundef float @llvm.maxnum.f32(float %165, float %166)
  %168 = extractelement <32 x float> %111, i64 28
  %169 = tail call nsz contract noundef float @llvm.maxnum.f32(float %167, float %168)
  %170 = extractelement <32 x float> %111, i64 29
  %171 = tail call nsz contract noundef float @llvm.maxnum.f32(float %169, float %170)
  %172 = extractelement <32 x float> %111, i64 30
  %173 = tail call nsz contract noundef float @llvm.maxnum.f32(float %171, float %172)
  %174 = extractelement <32 x float> %111, i64 31
  %175 = tail call nsz contract noundef float @llvm.maxnum.f32(float %173, float %174)
  br label %181

176:                                              ; preds = %22
  %177 = zext nneg i16 %106 to i32
  %178 = shl nuw nsw i32 %177, 16
  %179 = bitcast i32 %178 to float
  %180 = tail call nsz contract noundef float @llvm.maxnum.f32(float %179, float 1.000000e-10)
  br label %181

181:                                              ; preds = %176, %108
  %182 = phi float [ %180, %176 ], [ %175, %108 ]
  %183 = bitcast float %182 to i32
  %184 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %183, i32 177, i32 15, i32 15, i1 false)
  %185 = bitcast i32 %184 to float
  %186 = fcmp contract ogt float %182, %185
  %187 = select contract i1 %186, float %182, float %185
  %188 = bitcast float %187 to i32
  %189 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %188, i32 78, i32 15, i32 15, i1 false)
  %190 = bitcast i32 %189 to float
  %191 = fcmp contract ogt float %187, %190
  %192 = select contract i1 %191, float %187, float %190
  %193 = fmul contract float %192, f0x3B124925
  %194 = tail call float @llvm.fabs.f32(float %193)
  %195 = bitcast float %194 to i32
  %196 = add nuw i32 %195, 8388607
  %197 = lshr i32 %196, 23
  %198 = tail call noundef i32 @llvm.umin.i32(i32 %197, i32 255)
  %199 = shl nuw nsw i32 %198, 23
  %200 = bitcast i32 %199 to float
  %201 = tail call nsz contract noundef float @llvm.minnum.f32(float %200, float f0x7F000000)
  %202 = shl nuw nsw i32 %9, 4
  %203 = icmp eq i32 %9, 0
  br i1 %203, label %204, label %213

204:                                              ; preds = %181
  %205 = shl i64 %18, 32
  %206 = ashr exact i64 %205, 28
  %207 = getelementptr i8, ptr %6, i64 %206
  %208 = getelementptr i8, ptr %207, i64 %16
  %209 = bitcast float %201 to i32
  %210 = lshr i32 %209, 23
  %211 = trunc nuw i32 %210 to i8
  %212 = icmp eq ptr %208, null
  br label %213

213:                                              ; preds = %204, %181
  %214 = phi i8 [ %211, %204 ], [ 0, %181 ]
  %215 = phi i1 [ %212, %204 ], [ true, %181 ]
  %216 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %5, i16 0, i64 32768, i32 822099968)
  %217 = trunc nuw nsw i64 %23 to i32
  %218 = or disjoint i32 %202, %217
  %219 = add nsw i32 %218, %25
  %220 = shufflevector <16 x bfloat> %30, <16 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %221 = tail call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.fp8.bf16(<8 x bfloat> %220, float %201)
  %222 = bitcast <16 x bfloat> %30 to <2 x i128>
  %223 = extractelement <2 x i128> %222, i64 1
  %224 = bitcast i128 %223 to <8 x bfloat>
  %225 = shufflevector <16 x bfloat> %30, <16 x bfloat> poison, <8 x i32> <i32 poison, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %226 = shufflevector <8 x bfloat> %224, <8 x bfloat> %225, <8 x i32> <i32 0, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %227 = tail call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.fp8.bf16(<8 x bfloat> %226, float %201)
  %228 = shufflevector <2 x i32> %221, <2 x i32> %227, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %228, ptr addrspace(8) %216, i32 %219, i32 0, i32 0), !alias.scope !50, !noalias !60
  tail call void asm sideeffect "s_nop 0", ""() #23, !noalias !57, !srcloc !48
  %229 = shufflevector <16 x bfloat> %32, <16 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %230 = tail call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.fp8.bf16(<8 x bfloat> %229, float %201)
  %231 = bitcast <16 x bfloat> %32 to <2 x i128>
  %232 = extractelement <2 x i128> %231, i64 1
  %233 = bitcast i128 %232 to <8 x bfloat>
  %234 = shufflevector <16 x bfloat> %32, <16 x bfloat> poison, <8 x i32> <i32 poison, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %235 = shufflevector <8 x bfloat> %233, <8 x bfloat> %234, <8 x i32> <i32 0, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %236 = tail call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.fp8.bf16(<8 x bfloat> %235, float %201)
  %237 = shufflevector <2 x i32> %230, <2 x i32> %236, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %237, ptr addrspace(8) %216, i32 %219, i32 64, i32 0), !alias.scope !50, !noalias !60
  tail call void asm sideeffect "s_nop 0", ""() #23, !noalias !57, !srcloc !48
  br i1 %215, label %240, label %238

238:                                              ; preds = %213
  %239 = getelementptr inbounds nuw i8, ptr addrspace(3) @_ZZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELi2ELi1EEEvPT0_PfPKT_PKfliilPKiijE11s_scale_run, i32 %11
  store i8 %214, ptr addrspace(3) %239, align 1, !tbaa !29, !noalias !57
  br label %240

240:                                              ; preds = %238, %213, %21
  tail call void @llvm.amdgcn.s.wait.dscnt(i16 noundef signext 0) #22, !noalias !57
  tail call void @llvm.amdgcn.s.barrier(), !noalias !57
  %241 = and i64 %1, 3
  %242 = icmp eq i64 %241, 0
  %243 = icmp samesign ult i32 %8, 32
  br i1 %243, label %244, label %500

244:                                              ; preds = %240
  %245 = tail call i64 @llvm.amdgcn.readfirstlane.i64(i64 %16)
  %246 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %19)
  %247 = select i1 %242, i32 2, i32 0
  %248 = shl nuw nsw i32 %8, %247
  %249 = icmp samesign ult i32 %248, 16
  br i1 %249, label %250, label %500

250:                                              ; preds = %244
  %251 = shl i64 %245, 32
  %252 = ashr exact i64 %251, 32
  %253 = zext nneg i32 %248 to i64
  %254 = add nsw i64 %252, %253
  %255 = icmp slt i64 %254, 16
  br i1 %255, label %266, label %256

256:                                              ; preds = %250
  %257 = tail call i64 @llvm.usub.sat.i64(i64 %254, i64 31)
  %258 = add nuw i64 %257, 15
  %259 = lshr i64 %258, 4
  %260 = and i64 %258, 9223372036854775792
  %261 = trunc nuw nsw i64 %259 to i32
  %262 = add i32 %246, %261
  %263 = add nsw i64 %254, -16
  %264 = sub nsw i64 %263, %260
  %265 = add i32 %262, 1
  br label %266

266:                                              ; preds = %256, %250
  %267 = phi i64 [ %254, %250 ], [ %264, %256 ]
  %268 = phi i32 [ %246, %250 ], [ %265, %256 ]
  %269 = sext i32 %268 to i64
  %270 = shl nsw i64 %269, 4
  %271 = getelementptr inbounds i8, ptr %6, i64 %270
  %272 = getelementptr inbounds i8, ptr %271, i64 %267
  %273 = icmp slt i32 %268, 16
  br i1 %273, label %274, label %500

274:                                              ; preds = %266
  br i1 %242, label %275, label %279

275:                                              ; preds = %274
  %276 = lshr i32 %248, 2
  %277 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @_ZZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELi2ELi1EEEvPT0_PfPKT_PKfliilPKiijE11s_scale_run, i32 %276
  %278 = load i32, ptr addrspace(3) %277, align 4, !tbaa !5, !noalias !57
  store i32 %278, ptr %272, align 4, !tbaa !5, !alias.scope !53, !noalias !61
  br label %500

279:                                              ; preds = %274
  %280 = getelementptr inbounds nuw i8, ptr addrspace(3) @_ZZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELi2ELi1EEEvPT0_PfPKT_PKfliilPKiijE11s_scale_run, i32 %248
  %281 = load i8, ptr addrspace(3) %280, align 1, !tbaa !29, !noalias !57
  store i8 %281, ptr %272, align 1, !tbaa !29, !alias.scope !53, !noalias !61
  br label %500

282:                                              ; preds = %4
  br i1 %20, label %283, label %500

283:                                              ; preds = %282
  %284 = shl nuw nsw i64 %16, 11
  %285 = getelementptr inbounds nuw [2 x i8], ptr %7, i64 %284
  %286 = shl nsw i32 %19, 7
  %287 = sext i32 %286 to i64
  %288 = getelementptr inbounds [2 x i8], ptr %285, i64 %287
  %289 = zext nneg i32 %9 to i64
  %290 = getelementptr inbounds nuw [32 x i8], ptr %288, i64 %289
  %291 = load <16 x bfloat>, ptr %290, align 32, !tbaa !29, !alias.scope !55, !noalias !59
  %292 = getelementptr i8, ptr %290, i64 128
  %293 = load <16 x bfloat>, ptr %292, align 32, !tbaa !29, !alias.scope !55, !noalias !59
  %294 = shufflevector <16 x bfloat> %291, <16 x bfloat> %293, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %295 = bitcast <32 x bfloat> %294 to <16 x i32>
  %296 = extractelement <16 x i32> %295, i64 0
  %297 = bitcast i32 %296 to <2 x i16>
  %298 = extractelement <16 x i32> %295, i64 1
  %299 = bitcast i32 %298 to <2 x i16>
  %300 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %297, <2 x i16> %299)
  %301 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %297, <2 x i16> %299)
  %302 = extractelement <16 x i32> %295, i64 2
  %303 = bitcast i32 %302 to <2 x i16>
  %304 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %300, <2 x i16> %303)
  %305 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %301, <2 x i16> %303)
  %306 = extractelement <16 x i32> %295, i64 3
  %307 = bitcast i32 %306 to <2 x i16>
  %308 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %304, <2 x i16> %307)
  %309 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %305, <2 x i16> %307)
  %310 = extractelement <16 x i32> %295, i64 4
  %311 = bitcast i32 %310 to <2 x i16>
  %312 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %308, <2 x i16> %311)
  %313 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %309, <2 x i16> %311)
  %314 = extractelement <16 x i32> %295, i64 5
  %315 = bitcast i32 %314 to <2 x i16>
  %316 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %312, <2 x i16> %315)
  %317 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %313, <2 x i16> %315)
  %318 = extractelement <16 x i32> %295, i64 6
  %319 = bitcast i32 %318 to <2 x i16>
  %320 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %316, <2 x i16> %319)
  %321 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %317, <2 x i16> %319)
  %322 = extractelement <16 x i32> %295, i64 7
  %323 = bitcast i32 %322 to <2 x i16>
  %324 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %320, <2 x i16> %323)
  %325 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %321, <2 x i16> %323)
  %326 = extractelement <16 x i32> %295, i64 8
  %327 = bitcast i32 %326 to <2 x i16>
  %328 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %324, <2 x i16> %327)
  %329 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %325, <2 x i16> %327)
  %330 = extractelement <16 x i32> %295, i64 9
  %331 = bitcast i32 %330 to <2 x i16>
  %332 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %328, <2 x i16> %331)
  %333 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %329, <2 x i16> %331)
  %334 = extractelement <16 x i32> %295, i64 10
  %335 = bitcast i32 %334 to <2 x i16>
  %336 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %332, <2 x i16> %335)
  %337 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %333, <2 x i16> %335)
  %338 = extractelement <16 x i32> %295, i64 11
  %339 = bitcast i32 %338 to <2 x i16>
  %340 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %336, <2 x i16> %339)
  %341 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %337, <2 x i16> %339)
  %342 = extractelement <16 x i32> %295, i64 12
  %343 = bitcast i32 %342 to <2 x i16>
  %344 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %340, <2 x i16> %343)
  %345 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %341, <2 x i16> %343)
  %346 = extractelement <16 x i32> %295, i64 13
  %347 = bitcast i32 %346 to <2 x i16>
  %348 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %344, <2 x i16> %347)
  %349 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %345, <2 x i16> %347)
  %350 = extractelement <16 x i32> %295, i64 14
  %351 = bitcast i32 %350 to <2 x i16>
  %352 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %348, <2 x i16> %351)
  %353 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %349, <2 x i16> %351)
  %354 = extractelement <16 x i32> %295, i64 15
  %355 = bitcast i32 %354 to <2 x i16>
  %356 = tail call <2 x i16> @llvm.smax.v2i16(<2 x i16> %352, <2 x i16> %355)
  %357 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %353, <2 x i16> %355)
  %358 = bitcast <2 x i16> %356 to i32
  %359 = and i32 %358, 2147450879
  %360 = bitcast i32 %359 to <2 x i16>
  %361 = bitcast <2 x i16> %357 to i32
  %362 = and i32 %361, 2147450879
  %363 = bitcast i32 %362 to <2 x i16>
  %364 = tail call <2 x i16> @llvm.umax.v2i16(<2 x i16> %360, <2 x i16> %363)
  %365 = extractelement <2 x i16> %364, i64 0
  %366 = extractelement <2 x i16> %364, i64 1
  %367 = tail call i16 @llvm.umax.i16(i16 %365, i16 %366)
  %368 = icmp ugt i16 %367, 32640
  br i1 %368, label %369, label %437

369:                                              ; preds = %283
  %370 = shufflevector <16 x bfloat> %291, <16 x bfloat> %293, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %371 = tail call contract <32 x bfloat> @llvm.fabs.v32bf16(<32 x bfloat> %370)
  %372 = fpext <32 x bfloat> %371 to <32 x float>
  %373 = extractelement <32 x float> %372, i64 0
  %374 = tail call nsz contract noundef float @llvm.maxnum.f32(float %373, float 1.000000e-10)
  %375 = extractelement <32 x float> %372, i64 1
  %376 = tail call nsz contract noundef float @llvm.maxnum.f32(float %374, float %375)
  %377 = extractelement <32 x float> %372, i64 2
  %378 = tail call nsz contract noundef float @llvm.maxnum.f32(float %376, float %377)
  %379 = extractelement <32 x float> %372, i64 3
  %380 = tail call nsz contract noundef float @llvm.maxnum.f32(float %378, float %379)
  %381 = extractelement <32 x float> %372, i64 4
  %382 = tail call nsz contract noundef float @llvm.maxnum.f32(float %380, float %381)
  %383 = extractelement <32 x float> %372, i64 5
  %384 = tail call nsz contract noundef float @llvm.maxnum.f32(float %382, float %383)
  %385 = extractelement <32 x float> %372, i64 6
  %386 = tail call nsz contract noundef float @llvm.maxnum.f32(float %384, float %385)
  %387 = extractelement <32 x float> %372, i64 7
  %388 = tail call nsz contract noundef float @llvm.maxnum.f32(float %386, float %387)
  %389 = extractelement <32 x float> %372, i64 8
  %390 = tail call nsz contract noundef float @llvm.maxnum.f32(float %388, float %389)
  %391 = extractelement <32 x float> %372, i64 9
  %392 = tail call nsz contract noundef float @llvm.maxnum.f32(float %390, float %391)
  %393 = extractelement <32 x float> %372, i64 10
  %394 = tail call nsz contract noundef float @llvm.maxnum.f32(float %392, float %393)
  %395 = extractelement <32 x float> %372, i64 11
  %396 = tail call nsz contract noundef float @llvm.maxnum.f32(float %394, float %395)
  %397 = extractelement <32 x float> %372, i64 12
  %398 = tail call nsz contract noundef float @llvm.maxnum.f32(float %396, float %397)
  %399 = extractelement <32 x float> %372, i64 13
  %400 = tail call nsz contract noundef float @llvm.maxnum.f32(float %398, float %399)
  %401 = extractelement <32 x float> %372, i64 14
  %402 = tail call nsz contract noundef float @llvm.maxnum.f32(float %400, float %401)
  %403 = extractelement <32 x float> %372, i64 15
  %404 = tail call nsz contract noundef float @llvm.maxnum.f32(float %402, float %403)
  %405 = extractelement <32 x float> %372, i64 16
  %406 = tail call nsz contract noundef float @llvm.maxnum.f32(float %404, float %405)
  %407 = extractelement <32 x float> %372, i64 17
  %408 = tail call nsz contract noundef float @llvm.maxnum.f32(float %406, float %407)
  %409 = extractelement <32 x float> %372, i64 18
  %410 = tail call nsz contract noundef float @llvm.maxnum.f32(float %408, float %409)
  %411 = extractelement <32 x float> %372, i64 19
  %412 = tail call nsz contract noundef float @llvm.maxnum.f32(float %410, float %411)
  %413 = extractelement <32 x float> %372, i64 20
  %414 = tail call nsz contract noundef float @llvm.maxnum.f32(float %412, float %413)
  %415 = extractelement <32 x float> %372, i64 21
  %416 = tail call nsz contract noundef float @llvm.maxnum.f32(float %414, float %415)
  %417 = extractelement <32 x float> %372, i64 22
  %418 = tail call nsz contract noundef float @llvm.maxnum.f32(float %416, float %417)
  %419 = extractelement <32 x float> %372, i64 23
  %420 = tail call nsz contract noundef float @llvm.maxnum.f32(float %418, float %419)
  %421 = extractelement <32 x float> %372, i64 24
  %422 = tail call nsz contract noundef float @llvm.maxnum.f32(float %420, float %421)
  %423 = extractelement <32 x float> %372, i64 25
  %424 = tail call nsz contract noundef float @llvm.maxnum.f32(float %422, float %423)
  %425 = extractelement <32 x float> %372, i64 26
  %426 = tail call nsz contract noundef float @llvm.maxnum.f32(float %424, float %425)
  %427 = extractelement <32 x float> %372, i64 27
  %428 = tail call nsz contract noundef float @llvm.maxnum.f32(float %426, float %427)
  %429 = extractelement <32 x float> %372, i64 28
  %430 = tail call nsz contract noundef float @llvm.maxnum.f32(float %428, float %429)
  %431 = extractelement <32 x float> %372, i64 29
  %432 = tail call nsz contract noundef float @llvm.maxnum.f32(float %430, float %431)
  %433 = extractelement <32 x float> %372, i64 30
  %434 = tail call nsz contract noundef float @llvm.maxnum.f32(float %432, float %433)
  %435 = extractelement <32 x float> %372, i64 31
  %436 = tail call nsz contract noundef float @llvm.maxnum.f32(float %434, float %435)
  br label %442

437:                                              ; preds = %283
  %438 = zext nneg i16 %367 to i32
  %439 = shl nuw nsw i32 %438, 16
  %440 = bitcast i32 %439 to float
  %441 = tail call nsz contract noundef float @llvm.maxnum.f32(float %440, float 1.000000e-10)
  br label %442

442:                                              ; preds = %437, %369
  %443 = phi float [ %441, %437 ], [ %436, %369 ]
  %444 = bitcast float %443 to i32
  %445 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %444, i32 177, i32 15, i32 15, i1 false)
  %446 = bitcast i32 %445 to float
  %447 = fcmp contract ogt float %443, %446
  %448 = select contract i1 %447, float %443, float %446
  %449 = bitcast float %448 to i32
  %450 = tail call noundef i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %449, i32 78, i32 15, i32 15, i1 false)
  %451 = bitcast i32 %450 to float
  %452 = fcmp contract ogt float %448, %451
  %453 = select contract i1 %452, float %448, float %451
  %454 = fmul contract float %453, f0x3B124925
  %455 = tail call float @llvm.fabs.f32(float %454)
  %456 = bitcast float %455 to i32
  %457 = add nuw i32 %456, 8388607
  %458 = lshr i32 %457, 23
  %459 = tail call noundef i32 @llvm.umin.i32(i32 %458, i32 255)
  %460 = shl nuw nsw i32 %459, 23
  %461 = bitcast i32 %460 to float
  %462 = tail call nsz contract noundef float @llvm.minnum.f32(float %461, float f0x7F000000)
  %463 = shl nuw nsw i32 %9, 4
  %464 = icmp eq i32 %9, 0
  br i1 %464, label %465, label %473

465:                                              ; preds = %442
  %466 = shl i64 %18, 32
  %467 = ashr exact i64 %466, 28
  %468 = getelementptr i8, ptr %6, i64 %467
  %469 = getelementptr i8, ptr %468, i64 %16
  %470 = bitcast float %462 to i32
  %471 = lshr i32 %470, 23
  %472 = trunc nuw i32 %471 to i8
  br label %473

473:                                              ; preds = %465, %442
  %474 = phi i8 [ %472, %465 ], [ 0, %442 ]
  %475 = phi ptr [ %469, %465 ], [ null, %442 ]
  %476 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %5, i16 0, i64 32768, i32 822099968)
  %477 = trunc nuw nsw i64 %284 to i32
  %478 = or disjoint i32 %463, %477
  %479 = add nsw i32 %478, %286
  %480 = shufflevector <16 x bfloat> %291, <16 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %481 = tail call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.fp8.bf16(<8 x bfloat> %480, float %462)
  %482 = bitcast <16 x bfloat> %291 to <2 x i128>
  %483 = extractelement <2 x i128> %482, i64 1
  %484 = bitcast i128 %483 to <8 x bfloat>
  %485 = shufflevector <16 x bfloat> %291, <16 x bfloat> poison, <8 x i32> <i32 poison, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %486 = shufflevector <8 x bfloat> %484, <8 x bfloat> %485, <8 x i32> <i32 0, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %487 = tail call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.fp8.bf16(<8 x bfloat> %486, float %462)
  %488 = shufflevector <2 x i32> %481, <2 x i32> %487, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %488, ptr addrspace(8) %476, i32 %479, i32 0, i32 0), !alias.scope !50, !noalias !60
  tail call void asm sideeffect "s_nop 0", ""() #23, !noalias !57, !srcloc !48
  %489 = shufflevector <16 x bfloat> %293, <16 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %490 = tail call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.fp8.bf16(<8 x bfloat> %489, float %462)
  %491 = bitcast <16 x bfloat> %293 to <2 x i128>
  %492 = extractelement <2 x i128> %491, i64 1
  %493 = bitcast i128 %492 to <8 x bfloat>
  %494 = shufflevector <16 x bfloat> %293, <16 x bfloat> poison, <8 x i32> <i32 poison, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %495 = shufflevector <8 x bfloat> %493, <8 x bfloat> %494, <8 x i32> <i32 0, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %496 = tail call <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.fp8.bf16(<8 x bfloat> %495, float %462)
  %497 = shufflevector <2 x i32> %490, <2 x i32> %496, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %497, ptr addrspace(8) %476, i32 %479, i32 64, i32 0), !alias.scope !50, !noalias !60
  tail call void asm sideeffect "s_nop 0", ""() #23, !noalias !57, !srcloc !48
  %498 = icmp eq ptr %475, null
  br i1 %498, label %500, label %499

499:                                              ; preds = %473
  store i8 %474, ptr %475, align 1, !tbaa !29, !alias.scope !53, !noalias !61
  br label %500

500:                                              ; preds = %240, %244, %266, %275, %279, %282, %473, %499
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, target_mem: none) uwtable
define hidden void @ffn_act_virtual(i64 noundef %0, i64 noundef %1, i32 noundef %2, i32 %virtual_offset) #7 {
  %physical_tid = call i32 @llvm.amdgcn.workitem.id.x()
  tail call void @llvm.experimental.noalias.scope.decl(metadata !62)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !65)
  %4 = add i32 %physical_tid, %virtual_offset
  %5 = icmp samesign ult i32 %4, 256
  br i1 %5, label %6, label %189

6:                                                ; preds = %3
  %7 = shl nuw nsw i32 %4, 4
  %8 = inttoptr i64 %0 to ptr
  %9 = zext i32 %2 to i64
  %10 = shl nuw nsw i64 %9, 12
  %11 = getelementptr inbounds nuw i8, ptr %8, i64 %10
  %12 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %11, i16 0, i64 4096, i32 822099968)
  %13 = inttoptr i64 %1 to ptr
  %14 = shl nuw nsw i64 %9, 13
  %15 = getelementptr inbounds nuw i8, ptr %13, i64 %14
  %16 = getelementptr inbounds nuw i8, ptr %15, i64 4096
  %17 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr nonnull %16, i16 0, i64 4096, i32 822099968)
  %18 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %15, i16 0, i64 4096, i32 822099968)
  %19 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %18, i32 %7, i32 0, i32 3), !alias.scope !65, !noalias !62
  %20 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) nonnull %17, i32 %7, i32 0, i32 3), !alias.scope !65, !noalias !62
  %21 = bitcast <4 x i32> %20 to <8 x i16>
  %22 = bitcast <4 x i32> %19 to <8 x i16>
  %23 = extractelement <8 x i16> %22, i64 0
  %24 = zext i16 %23 to i32
  %25 = shl nuw i32 %24, 16
  %26 = bitcast i32 %25 to float
  %27 = tail call nsz contract noundef float @llvm.minnum.f32(float %26, float 1.000000e+01)
  %28 = shufflevector <8 x i16> %21, <8 x i16> poison, <2 x i32> <i32 0, i32 1>
  %29 = zext <2 x i16> %28 to <2 x i32>
  %30 = shl nuw <2 x i32> %29, splat (i32 16)
  %31 = bitcast <2 x i32> %30 to <2 x float>
  %32 = extractelement <2 x float> %31, i64 0
  %33 = tail call contract float @llvm.amdgcn.fmed3.f32(float %32, float -1.000000e+01, float 1.000000e+01)
  %34 = extractelement <8 x i16> %22, i64 1
  %35 = zext i16 %34 to i32
  %36 = shl nuw i32 %35, 16
  %37 = bitcast i32 %36 to float
  %38 = tail call nsz contract noundef float @llvm.minnum.f32(float %37, float 1.000000e+01)
  %39 = extractelement <2 x float> %31, i64 1
  %40 = tail call contract float @llvm.amdgcn.fmed3.f32(float %39, float -1.000000e+01, float 1.000000e+01)
  %41 = fptrunc contract float %38 to bfloat
  %42 = fptrunc contract float %27 to bfloat
  %43 = insertelement <2 x bfloat> poison, bfloat %42, i64 0
  %44 = insertelement <2 x bfloat> %43, bfloat %41, i64 1
  %45 = bitcast <2 x bfloat> %44 to <2 x i16>
  %46 = zext <2 x i16> %45 to <2 x i32>
  %47 = shl nuw <2 x i32> %46, splat (i32 16)
  %48 = bitcast <2 x i32> %47 to <2 x float>
  %49 = fneg contract <2 x float> %48
  %50 = tail call <2 x float> @llvm.exp.v2f32(<2 x float> %49)
  %51 = fadd contract <2 x float> %50, splat (float 1.000000e+00)
  %52 = extractelement <2 x float> %51, i64 0
  %53 = tail call contract float @llvm.amdgcn.rcp.f32(float %52)
  %54 = extractelement <2 x float> %51, i64 1
  %55 = tail call contract float @llvm.amdgcn.rcp.f32(float %54)
  %56 = insertelement <2 x float> poison, float %53, i64 0
  %57 = insertelement <2 x float> %56, float %55, i64 1
  %58 = fmul contract <2 x float> %57, %48
  %59 = insertelement <2 x float> poison, float %33, i64 0
  %60 = insertelement <2 x float> %59, float %40, i64 1
  %61 = fmul contract <2 x float> %60, %58
  %62 = fptrunc <2 x float> %61 to <2 x bfloat>
  %63 = extractelement <8 x i16> %22, i64 2
  %64 = zext i16 %63 to i32
  %65 = shl nuw i32 %64, 16
  %66 = bitcast i32 %65 to float
  %67 = tail call nsz contract noundef float @llvm.minnum.f32(float %66, float 1.000000e+01)
  %68 = shufflevector <8 x i16> %21, <8 x i16> poison, <2 x i32> <i32 2, i32 3>
  %69 = zext <2 x i16> %68 to <2 x i32>
  %70 = shl nuw <2 x i32> %69, splat (i32 16)
  %71 = bitcast <2 x i32> %70 to <2 x float>
  %72 = extractelement <2 x float> %71, i64 0
  %73 = tail call contract float @llvm.amdgcn.fmed3.f32(float %72, float -1.000000e+01, float 1.000000e+01)
  %74 = extractelement <8 x i16> %22, i64 3
  %75 = zext i16 %74 to i32
  %76 = shl nuw i32 %75, 16
  %77 = bitcast i32 %76 to float
  %78 = tail call nsz contract noundef float @llvm.minnum.f32(float %77, float 1.000000e+01)
  %79 = extractelement <2 x float> %71, i64 1
  %80 = tail call contract float @llvm.amdgcn.fmed3.f32(float %79, float -1.000000e+01, float 1.000000e+01)
  %81 = fptrunc contract float %78 to bfloat
  %82 = fptrunc contract float %67 to bfloat
  %83 = insertelement <2 x bfloat> poison, bfloat %82, i64 0
  %84 = insertelement <2 x bfloat> %83, bfloat %81, i64 1
  %85 = bitcast <2 x bfloat> %84 to <2 x i16>
  %86 = zext <2 x i16> %85 to <2 x i32>
  %87 = shl nuw <2 x i32> %86, splat (i32 16)
  %88 = bitcast <2 x i32> %87 to <2 x float>
  %89 = fneg contract <2 x float> %88
  %90 = tail call <2 x float> @llvm.exp.v2f32(<2 x float> %89)
  %91 = fadd contract <2 x float> %90, splat (float 1.000000e+00)
  %92 = extractelement <2 x float> %91, i64 0
  %93 = tail call contract float @llvm.amdgcn.rcp.f32(float %92)
  %94 = extractelement <2 x float> %91, i64 1
  %95 = tail call contract float @llvm.amdgcn.rcp.f32(float %94)
  %96 = insertelement <2 x float> poison, float %93, i64 0
  %97 = insertelement <2 x float> %96, float %95, i64 1
  %98 = fmul contract <2 x float> %97, %88
  %99 = insertelement <2 x float> poison, float %73, i64 0
  %100 = insertelement <2 x float> %99, float %80, i64 1
  %101 = fmul contract <2 x float> %100, %98
  %102 = fptrunc <2 x float> %101 to <2 x bfloat>
  %103 = shufflevector <2 x bfloat> %62, <2 x bfloat> %102, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %104 = extractelement <8 x i16> %22, i64 4
  %105 = zext i16 %104 to i32
  %106 = shl nuw i32 %105, 16
  %107 = bitcast i32 %106 to float
  %108 = tail call nsz contract noundef float @llvm.minnum.f32(float %107, float 1.000000e+01)
  %109 = shufflevector <8 x i16> %21, <8 x i16> poison, <2 x i32> <i32 4, i32 5>
  %110 = zext <2 x i16> %109 to <2 x i32>
  %111 = shl nuw <2 x i32> %110, splat (i32 16)
  %112 = bitcast <2 x i32> %111 to <2 x float>
  %113 = extractelement <2 x float> %112, i64 0
  %114 = tail call contract float @llvm.amdgcn.fmed3.f32(float %113, float -1.000000e+01, float 1.000000e+01)
  %115 = extractelement <8 x i16> %22, i64 5
  %116 = zext i16 %115 to i32
  %117 = shl nuw i32 %116, 16
  %118 = bitcast i32 %117 to float
  %119 = tail call nsz contract noundef float @llvm.minnum.f32(float %118, float 1.000000e+01)
  %120 = extractelement <2 x float> %112, i64 1
  %121 = tail call contract float @llvm.amdgcn.fmed3.f32(float %120, float -1.000000e+01, float 1.000000e+01)
  %122 = fptrunc contract float %119 to bfloat
  %123 = fptrunc contract float %108 to bfloat
  %124 = insertelement <2 x bfloat> poison, bfloat %123, i64 0
  %125 = insertelement <2 x bfloat> %124, bfloat %122, i64 1
  %126 = bitcast <2 x bfloat> %125 to <2 x i16>
  %127 = zext <2 x i16> %126 to <2 x i32>
  %128 = shl nuw <2 x i32> %127, splat (i32 16)
  %129 = bitcast <2 x i32> %128 to <2 x float>
  %130 = fneg contract <2 x float> %129
  %131 = tail call <2 x float> @llvm.exp.v2f32(<2 x float> %130)
  %132 = fadd contract <2 x float> %131, splat (float 1.000000e+00)
  %133 = extractelement <2 x float> %132, i64 0
  %134 = tail call contract float @llvm.amdgcn.rcp.f32(float %133)
  %135 = extractelement <2 x float> %132, i64 1
  %136 = tail call contract float @llvm.amdgcn.rcp.f32(float %135)
  %137 = insertelement <2 x float> poison, float %134, i64 0
  %138 = insertelement <2 x float> %137, float %136, i64 1
  %139 = fmul contract <2 x float> %138, %129
  %140 = insertelement <2 x float> poison, float %114, i64 0
  %141 = insertelement <2 x float> %140, float %121, i64 1
  %142 = fmul contract <2 x float> %141, %139
  %143 = fptrunc <2 x float> %142 to <2 x bfloat>
  %144 = shufflevector <2 x bfloat> %143, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %145 = shufflevector <8 x bfloat> %103, <8 x bfloat> %144, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 poison, i32 poison>
  %146 = extractelement <8 x i16> %22, i64 6
  %147 = zext i16 %146 to i32
  %148 = shl nuw i32 %147, 16
  %149 = bitcast i32 %148 to float
  %150 = tail call nsz contract noundef float @llvm.minnum.f32(float %149, float 1.000000e+01)
  %151 = shufflevector <8 x i16> %21, <8 x i16> poison, <2 x i32> <i32 6, i32 7>
  %152 = zext <2 x i16> %151 to <2 x i32>
  %153 = shl nuw <2 x i32> %152, splat (i32 16)
  %154 = bitcast <2 x i32> %153 to <2 x float>
  %155 = extractelement <2 x float> %154, i64 0
  %156 = tail call contract float @llvm.amdgcn.fmed3.f32(float %155, float -1.000000e+01, float 1.000000e+01)
  %157 = extractelement <8 x i16> %22, i64 7
  %158 = zext i16 %157 to i32
  %159 = shl nuw i32 %158, 16
  %160 = bitcast i32 %159 to float
  %161 = tail call nsz contract noundef float @llvm.minnum.f32(float %160, float 1.000000e+01)
  %162 = extractelement <2 x float> %154, i64 1
  %163 = tail call contract float @llvm.amdgcn.fmed3.f32(float %162, float -1.000000e+01, float 1.000000e+01)
  %164 = fptrunc contract float %161 to bfloat
  %165 = fptrunc contract float %150 to bfloat
  %166 = insertelement <2 x bfloat> poison, bfloat %165, i64 0
  %167 = insertelement <2 x bfloat> %166, bfloat %164, i64 1
  %168 = bitcast <2 x bfloat> %167 to <2 x i16>
  %169 = zext <2 x i16> %168 to <2 x i32>
  %170 = shl nuw <2 x i32> %169, splat (i32 16)
  %171 = bitcast <2 x i32> %170 to <2 x float>
  %172 = fneg contract <2 x float> %171
  %173 = tail call <2 x float> @llvm.exp.v2f32(<2 x float> %172)
  %174 = fadd contract <2 x float> %173, splat (float 1.000000e+00)
  %175 = extractelement <2 x float> %174, i64 0
  %176 = tail call contract float @llvm.amdgcn.rcp.f32(float %175)
  %177 = extractelement <2 x float> %174, i64 1
  %178 = tail call contract float @llvm.amdgcn.rcp.f32(float %177)
  %179 = insertelement <2 x float> poison, float %176, i64 0
  %180 = insertelement <2 x float> %179, float %178, i64 1
  %181 = fmul contract <2 x float> %180, %171
  %182 = insertelement <2 x float> poison, float %156, i64 0
  %183 = insertelement <2 x float> %182, float %163, i64 1
  %184 = fmul contract <2 x float> %183, %181
  %185 = fptrunc <2 x float> %184 to <2 x bfloat>
  %186 = shufflevector <2 x bfloat> %185, <2 x bfloat> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %187 = shufflevector <8 x bfloat> %145, <8 x bfloat> %186, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %188 = bitcast <8 x bfloat> %187 to <4 x i32>
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %188, ptr addrspace(8) %12, i32 %7, i32 0, i32 0), !alias.scope !62, !noalias !65
  br label %189, !llvm.loop !67

189:                                              ; preds = %3, %6
  ret void
}

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #8

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #9

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #10

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x i16> @llvm.smax.v2i16(<2 x i16>, <2 x i16>) #11

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x i16> @llvm.umax.v2i16(<2 x i16>, <2 x i16>) #11

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #11

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #11

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32 immarg, i32 immarg, i32 immarg, i1 immarg) #12

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.minnum.f32(float, float) #11

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr readnone, i16, i64, i32) #11

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x i32> @llvm.amdgcn.cvt.scalef32.pk8.fp8.bf16(<8 x bfloat>, float) #11

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32>, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #13

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.wait.dscnt(i16 immarg) #14

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.fmed3.f32(float, float, float) #11

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #15

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.rcp.f32(float) #11

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef align 4 ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #10

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.amdgcn.s.sleep(i32 immarg) #16

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i64 @llvm.amdgcn.readfirstlane.i64(i64) #9

; Function Attrs: nocallback nounwind willreturn
declare void @llvm.amdgcn.s.sendmsg(i32 immarg, i32) #17

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #18

; Function Attrs: convergent norecurse nounwind uwtable
define internal fastcc void @__ockl_fprintf_append_string_n(i64 noundef %0, ptr noundef readonly captures(address_is_null) %1, i64 noundef range(i64 -2147483648, 2147483648) %2, i32 noundef range(i32 0, 2) %3) unnamed_addr #19 {
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
declare i64 @llvm.umin.i64(i64, i64) #11

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umin.i32(i32, i32) #11

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ballot.i32(i1) #9

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) noalias writeonly captures(none), ptr addrspace(4) noalias readonly captures(none), i64, i1 immarg) #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i16 @llvm.umax.i16(i16, i16) #11

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #20

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.usub.sat.i64(i64, i64) #11

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <32 x bfloat> @llvm.fabs.v32bf16(<32 x bfloat>) #11

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x float> @llvm.exp.v2f32(<2 x float>) #11

attributes #0 = { mustprogress noreturn nounwind uwtable "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1250" "uniform-work-group-size" }
attributes #1 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #2 = { convergent noinline nounwind uwtable "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1250" "uniform-work-group-size" }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { mustprogress noinline nounwind uwtable "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1250" "uniform-work-group-size" }
attributes #5 = { convergent mustprogress norecurse nounwind uwtable "amdgpu-flat-work-group-size"="1,1024" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1250" "uniform-work-group-size" }
attributes #6 = { alwaysinline convergent mustprogress nounwind uwtable "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1250" "uniform-work-group-size" }
attributes #7 = { alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, target_mem: none) uwtable "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1250" "uniform-work-group-size" }
attributes #8 = { convergent nocallback nofree nounwind willreturn }
attributes #9 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #10 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #11 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #12 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #13 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
attributes #14 = { nocallback nofree nounwind willreturn }
attributes #15 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #16 = { nocallback nofree nosync nounwind willreturn }
attributes #17 = { nocallback nounwind willreturn }
attributes #18 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #19 = { convergent norecurse nounwind uwtable "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1250" "target-features"="+16-bit-insts,+add-min-max-insts,+ashr-pk-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f32,+atomic-fmin-fmax-global-f64,+atomic-global-pk-add-bf16-inst,+bf16-cvt-insts,+bf16-pk-insts,+bf16-trans-insts,+bitop3-insts,+ci-insts,+clusters,+cube-insts,+cvt-pknorm-vop2-insts,+dl-insts,+dot7-insts,+dot8-insts,+dpp,+fp8-conversion-insts,+fp8e5m3-insts,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx12-insts,+gfx1250-insts,+gfx8-insts,+gfx9-insts,+lerp-inst,+mcast-load-insts,+mqsad-insts,+mqsad-pk-insts,+msad-insts,+permlane16-swap,+pk-add-min-max-insts,+prng-inst,+qsad-insts,+s-wakeup-barrier-inst,+sad-insts,+setprio-inc-wg-inst,+swmmac-gfx1200-insts,+swmmac-gfx1250-insts,+tanh-insts,+tensor-cvt-lut-insts,+transpose-load-f4f6-insts,+vmem-pref-insts,+wavefrontsize32" }
attributes #20 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #21 = { nounwind }
attributes #22 = { convergent nounwind "uniform-work-group-size" }
attributes #23 = { convergent nounwind }

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
!36 = !{i64 4404419, i64 4404435, i64 4404471, i64 4404507, i64 4404543, i64 4404579, i64 4404615, i64 4404651, i64 4404687, i64 4404723, i64 4404759, i64 4404795, i64 4404831, i64 4404867, i64 4404903, i64 4404939, i64 4404975, i64 4405011}
!37 = !{!38}
!38 = distinct !{!38, !39, !"_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELi2ELi1EEEvPT0_PfPKT_PKfliilPKiij: argument 0"}
!39 = distinct !{!39, !"_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELi2ELi1EEEvPT0_PfPKT_PKfliilPKiij"}
!40 = !{!41}
!41 = distinct !{!41, !39, !"_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELi2ELi1EEEvPT0_PfPKT_PKfliilPKiij: argument 1"}
!42 = !{!43}
!43 = distinct !{!43, !39, !"_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELi2ELi1EEEvPT0_PfPKT_PKfliilPKiij: argument 2"}
!44 = !{!38, !41, !43, !45}
!45 = distinct !{!45, !39, !"_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELi2ELi1EEEvPT0_PfPKT_PKfliilPKiij: argument 3"}
!46 = !{!38, !41, !45}
!47 = !{!41, !43, !45}
!48 = !{i64 8334139}
!49 = !{!38, !43, !45}
!50 = !{!51}
!51 = distinct !{!51, !52, !"_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELi2ELi1EEEvPT0_PfPKT_PKfliilPKiij: argument 0"}
!52 = distinct !{!52, !"_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELi2ELi1EEEvPT0_PfPKT_PKfliilPKiij"}
!53 = !{!54}
!54 = distinct !{!54, !52, !"_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELi2ELi1EEEvPT0_PfPKT_PKfliilPKiij: argument 1"}
!55 = !{!56}
!56 = distinct !{!56, !52, !"_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELi2ELi1EEEvPT0_PfPKT_PKfliilPKiij: argument 2"}
!57 = !{!51, !54, !56, !58}
!58 = distinct !{!58, !52, !"_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELi2ELi1EEEvPT0_PfPKT_PKfliilPKiij: argument 3"}
!59 = !{!51, !54, !58}
!60 = !{!54, !56, !58}
!61 = !{!51, !56, !58}
!62 = !{!63}
!63 = distinct !{!63, !64, !"_ZN5aiter18act_and_mul_kernelIDF16bDF16bTnPFfRKT_EXadL_ZNS_11silu_kernelIDF16bEEfS3_EELi8ELb1ELi3EEEvPT0_PS2_ifj: argument 0"}
!64 = distinct !{!64, !"_ZN5aiter18act_and_mul_kernelIDF16bDF16bTnPFfRKT_EXadL_ZNS_11silu_kernelIDF16bEEfS3_EELi8ELb1ELi3EEEvPT0_PS2_ifj"}
!65 = !{!66}
!66 = distinct !{!66, !64, !"_ZN5aiter18act_and_mul_kernelIDF16bDF16bTnPFfRKT_EXadL_ZNS_11silu_kernelIDF16bEEfS3_EELi8ELb1ELi3EEEvPT0_PS2_ifj: argument 1"}
!67 = distinct !{!67, !31}
