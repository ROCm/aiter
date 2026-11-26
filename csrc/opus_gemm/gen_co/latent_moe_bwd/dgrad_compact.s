// SPDX-License-Identifier: MIT
    // Use the true row count at the compact expert boundary.
    .incbin "body.bin", 0, 0x14fc
    s_add_i32 s46, s15, s16
    .incbin "body.bin", 0x1500, 50272-0x1500
