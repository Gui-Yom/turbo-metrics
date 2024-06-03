; ModuleID = 'shared.cu'
source_filename = "shared.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Minimal bitcode to include a cuda shared memory in a Rust kernel.
; Originally compiled from shared.cu, then cleaned by hand.

@RING = dso_local addrspace(3) global [96 x [33 x float]] undef, align 4
