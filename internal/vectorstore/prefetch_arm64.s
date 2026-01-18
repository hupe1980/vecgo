// Copyright 2024 The Vecgo Authors
// SPDX-License-Identifier: MIT

//go:build !noasm && arm64

#include "textflag.h"

// func prefetchVectorNEON(ptr unsafe.Pointer)
// Prefetch a cache line into L1 data cache using ARM64 PRFM instruction.
// This is non-blocking and returns immediately.
TEXT ·prefetchVectorNEON(SB), NOSPLIT, $0-8
    MOVD ptr+0(FP), R0
    // PRFM PLDL1STRM, [R0] - Prefetch for Load into L1 cache, streaming (non-temporal)
    // Opcode: 0xF9800000 | (Rn << 5) where Rn=R0=0
    // PRFM PLDL1KEEP is encoded as: 0xF9800000 + offset + Rt
    // For simple prefetch with R0 as base:
    WORD $0xF9800000  // PRFM PLDL1KEEP, [X0]
    RET

// func prefetchBatchNEON(base unsafe.Pointer, dim, count int, ids []uint32)
// Prefetch multiple vectors at once using ARM64 PRFM.
// base: pointer to vector data (float32 array)
// dim: dimension of each vector
// count: number of vectors to prefetch
// ids: slice of row IDs to prefetch
TEXT ·prefetchBatchNEON(SB), NOSPLIT, $0-56
    MOVD base+0(FP), R0      // base pointer
    MOVD dim+8(FP), R1       // dimension
    MOVD count+16(FP), R2    // count
    MOVD ids_base+24(FP), R3 // ids slice base

    // Calculate stride: dim * sizeof(float32) = dim * 4
    LSL $2, R1, R4           // R4 = dim * 4 (stride in bytes)

    CBZ R2, done             // if count == 0, return
    
loop:
    // Load row ID from ids slice
    MOVWU (R3), R5           // R5 = ids[i] (uint32)
    
    // Calculate offset: rowID * stride
    MUL R5, R4, R6           // R6 = rowID * stride
    
    // Calculate address: base + offset
    ADD R0, R6, R7           // R7 = base + rowID * stride
    
    // Prefetch the vector start (one cache line = 64 bytes)
    WORD $0xF98000E7         // PRFM PLDL1KEEP, [X7]
    
    // For larger vectors, also prefetch the second cache line
    // This covers vectors up to 32 floats (128 bytes) efficiently
    ADD $64, R7, R8
    WORD $0xF9800108         // PRFM PLDL1KEEP, [X8]

    // Move to next ID
    ADD $4, R3, R3           // ids pointer += sizeof(uint32)
    SUB $1, R2, R2           // count--
    CBNZ R2, loop            // if count != 0, continue

done:
    RET
