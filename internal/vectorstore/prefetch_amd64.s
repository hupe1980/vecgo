// Copyright 2024 The Vecgo Authors
// SPDX-License-Identifier: MIT

//go:build !noasm && amd64

#include "textflag.h"

// func prefetchBatchAMD64(base unsafe.Pointer, dim, count int, ids []uint32)
// Prefetch multiple vectors at once using x86-64 PREFETCHT0 instruction.
// base: pointer to vector data (float32 array)
// dim: dimension of each vector
// count: number of vectors to prefetch
// ids: slice of row IDs to prefetch
//
// PREFETCHT0 fetches data into all cache levels (L1, L2, L3).
// Each cache line is 64 bytes on x86-64.
TEXT ·prefetchBatchAMD64(SB), NOSPLIT, $0-56
    MOVQ base+0(FP), AX      // AX = base pointer
    MOVQ dim+8(FP), BX       // BX = dimension
    MOVQ count+16(FP), CX    // CX = count
    MOVQ ids_base+24(FP), DX // DX = ids slice base

    // Calculate stride: dim * sizeof(float32) = dim * 4
    SHLQ $2, BX              // BX = dim * 4 (stride in bytes)

    TESTQ CX, CX             // if count == 0
    JZ done                  // return

loop:
    // Load row ID from ids slice (uint32)
    MOVL (DX), SI            // SI = ids[i] (zero-extended from uint32)

    // Calculate offset: rowID * stride
    MOVQ SI, DI              // DI = rowID (64-bit)
    IMULQ BX, DI             // DI = rowID * stride

    // Calculate address: base + offset
    ADDQ AX, DI              // DI = base + rowID * stride

    // Prefetch the vector start (one cache line = 64 bytes)
    // PREFETCHT0 m8 - fetch into all cache levels
    PREFETCHT0 (DI)

    // For larger vectors, also prefetch the second cache line
    // This covers vectors up to 32 floats (128 bytes) efficiently
    ADDQ $64, DI
    PREFETCHT0 (DI)

    // Move to next ID
    ADDQ $4, DX              // ids pointer += sizeof(uint32)
    DECQ CX                  // count--
    JNZ loop                 // if count != 0, continue

done:
    RET
