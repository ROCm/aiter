mode=0
is_asm=1
round_mode=2
mask=2

device=4
perm=1

for seqlen in 21 64 256 512 1200 3200 5200; do
    ROCR_VISIBLE_DEVICES=$device ./fwd.exe -prec=bf16 -b=2 -h=16  -d=128 -d_v=128 -s=$seqlen -iperm=$perm -operm=$perm -mask=$mask -lse=1 -fwd_v3=$is_asm -v3_bf16_cvt=$round_mode -mode=$mode -kname=1 -v=1
    ROCR_VISIBLE_DEVICES=$device ./fwd.exe -prec=bf16 -b=2 -h=32  -d=128 -d_v=128 -s=$seqlen -iperm=$perm -operm=$perm -mask=$mask -lse=1 -fwd_v3=$is_asm -v3_bf16_cvt=$round_mode -mode=$mode -kname=1 -v=1
    ROCR_VISIBLE_DEVICES=$device ./fwd.exe -prec=bf16 -b=2 -h=128 -d=128 -d_v=128 -s=$seqlen -iperm=$perm -operm=$perm -mask=$mask -lse=1 -fwd_v3=$is_asm -v3_bf16_cvt=$round_mode -mode=$mode -kname=1 -v=1
done

ROCR_VISIBLE_DEVICES=$device ./fwd.exe -prec=bf16 -b=2 -h=16  -d=128 -d_v=128 -s=8192 -iperm=$perm -operm=$perm -mask=$mask -lse=1 -fwd_v3=$is_asm -v3_bf16_cvt=$round_mode -mode=$mode -kname=1 -v=0
ROCR_VISIBLE_DEVICES=$device ./fwd.exe -prec=bf16 -b=2 -h=32  -d=128 -d_v=128 -s=8192 -iperm=$perm -operm=$perm -mask=$mask -lse=1 -fwd_v3=$is_asm -v3_bf16_cvt=$round_mode -mode=$mode -kname=1 -v=0

ROCR_VISIBLE_DEVICES=$device ./fwd.exe -prec=bf16 -b=2 -h=16  -d=128 -d_v=128 -s=10000 -iperm=$perm -operm=$perm -mask=$mask -lse=1 -fwd_v3=$is_asm -v3_bf16_cvt=$round_mode -mode=$mode -kname=1 -v=0
ROCR_VISIBLE_DEVICES=$device ./fwd.exe -prec=bf16 -b=2 -h=32  -d=128 -d_v=128 -s=10000 -iperm=$perm -operm=$perm -mask=$mask -lse=1 -fwd_v3=$is_asm -v3_bf16_cvt=$round_mode -mode=$mode -kname=1 -v=0

ROCR_VISIBLE_DEVICES=$device ./fwd.exe -prec=bf16 -b=2 -h=16  -d=128 -d_v=128 -s=16384 -iperm=$perm -operm=$perm -mask=$mask -lse=1 -fwd_v3=$is_asm -v3_bf16_cvt=$round_mode -mode=$mode -kname=1 -v=0

ROCR_VISIBLE_DEVICES=$device ./fwd.exe -prec=bf16 -b=2 -h=2  -d=128 -d_v=128 -s=90000 -iperm=$perm -operm=$perm -mask=$mask -lse=1 -fwd_v3=$is_asm -v3_bf16_cvt=$round_mode -mode=$mode -kname=1 -v=0