; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

define amdgpu_kernel void @moe_route(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, i32 %3, i32 %4) #0 {
  %6 = call i32 @llvm.amdgcn.workgroup.id.x()
  %7 = sext i32 %6 to i64
  %8 = trunc i64 %7 to i32
  %9 = mul i32 %8, 256
  %10 = call i32 @llvm.amdgcn.workitem.id.x()
  %11 = sext i32 %10 to i64
  %12 = trunc i64 %11 to i32
  %13 = add i32 %9, %12
  %14 = icmp ult i32 %13, %3
  br i1 %14, label %15, label %32

15:                                               ; preds = %5
  %16 = ptrtoint ptr addrspace(1) %0 to i64
  %17 = inttoptr i64 %16 to ptr
  %18 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %17, i16 0, i64 4294967295, i32 159744)
  %19 = ptrtoint ptr addrspace(1) %2 to i64
  %20 = inttoptr i64 %19 to ptr
  %21 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr %20, i16 0, i64 4294967295, i32 159744)
  %22 = mul i32 %13, 4
  %23 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %18, i32 %22, i32 0, i32 0)
  %24 = ptrtoint ptr addrspace(1) %1 to i64
  %25 = sext i32 %23 to i64
  %26 = mul i64 %25, 4
  %27 = add i64 %24, %26
  %28 = inttoptr i64 %27 to ptr addrspace(1)
  %29 = atomicrmw add ptr addrspace(1) %28, i32 1 syncscope("agent") monotonic, align 4
  %30 = mul i32 %23, %4
  %31 = add i32 %29, %30
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %31, ptr addrspace(8) %21, i32 %22, i32 0, i32 0)
  br label %32

32:                                               ; preds = %15, %5
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p0(ptr readnone, i16, i64, i32) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #4

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
