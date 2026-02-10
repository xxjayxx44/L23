/*-
 * Copyright 2009 Colin Percival
 * Copyright 2012-2019 Alexander Peslyak
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * This file was originally written by Colin Percival as part of the Tarsnap
 * online backup system.
 *
 * This is a proof-of-work focused fork of yescrypt, including optimized and
 * cut-down implementation of the obsolete yescrypt 0.5 (based off its first
 * submission to PHC back in 2014) and a new proof-of-work specific variation
 * known as yespower 1.0.  The former is intended as an upgrade for
 * cryptocurrencies that already use yescrypt 0.5 and the latter may be used
 * as a further upgrade (hard fork) by those and other cryptocurrencies.  The
 * version of algorithm to use is requested through parameters, allowing for
 * both algorithms to co-exist in client and miner implementations (such as in
 * preparation for a hard-fork).
 */

#ifndef _YESPOWER_OPT_C_PASS_
#define _YESPOWER_OPT_C_PASS_ 1
#endif

#if _YESPOWER_OPT_C_PASS_ == 1
/*
 * AVX and especially XOP speed up Salsa20 a lot, but needlessly result in
 * extra instruction prefixes for pwxform (which we make more use of).  While
 * no slowdown from the prefixes is generally observed on AMD CPUs supporting
 * XOP, some slowdown is sometimes observed on Intel CPUs with AVX.
 */
#ifdef __XOP__
/* Note: XOP is enabled. That's great. */
#elif defined(__AVX__)
/* Note: AVX is enabled. That's OK. */
#elif defined(__SSE2__)
/* Note: AVX and XOP are not enabled. That's OK. */
#elif defined(__x86_64__) || defined(__i386__)
/* SSE2 not enabled. Expect poor performance. */
#else
/* Note: building generic code for non-x86. That's OK. */
#endif

/*
 * MediaTek Dimensity 6300 optimizations
 * ARM Cortex-A53/A55 compatible optimizations
 */
#ifdef __aarch64__
#include <arm_neon.h>
#define USE_NEON
/* Note: Building with ARM NEON optimizations for Dimensity 6300 */
#endif

/*
 * The SSE4 code version has fewer instructions than the generic SSE2 version,
 * but all of the instructions are SIMD, thereby wasting the scalar execution
 * units.  Thus, the generic SSE2 version below actually runs faster on some
 * CPUs due to its balanced mix of SIMD and scalar instructions.
 */
#undef USE_SSE4_FOR_32BIT

/* Early exit and bias control flags */
static int enable_early_exit = 1;
static uint32_t bias_threshold = 0x0000FFFF; /* Configurable bias threshold */
static uint32_t subspace_mask = 0x7FFFFFFF; /* Search easier subspaces */

/* Memory hardness reduction controls */
static int reduce_memory_hardness = 0;

/* Thread restart signaling */
extern volatile int mining_restart_requested;
extern volatile int abandon_current_work;
extern uint32_t skip_pattern;

#ifdef __SSE2__
/*
 * GCC before 4.9 would by default unnecessarily use store/load (without
 * SSE4.1) or (V)PEXTR (with SSE4.1 or AVX) instead of simply (V)MOV.
 * This was tracked as GCC bug 54349.
 * "-mtune=corei7" works around this, but is only supported for GCC 4.6+.
 * We use inline asm for pre-4.6 GCC, further down this file.
 */
#if __GNUC__ == 4 && __GNUC_MINOR__ >= 6 && __GNUC_MINOR__ < 9 && \
    !defined(__clang__) && !defined(__ICC)
#pragma GCC target ("tune=corei7")
#endif
#include <emmintrin.h>
#ifdef __XOP__
#include <x86intrin.h>
#endif
#elif defined(__SSE__)
#include <xmmintrin.h>
#endif

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "insecure_memzero.h"
#include "sha256.h"
#include "sysendian.h"

#include "yespower.h"

#include "yespower-platform.c"

#if __STDC_VERSION__ >= 199901L
/* Have restrict */
#elif defined(__GNUC__)
#define restrict __restrict
#else
#define restrict
#endif

#ifdef __GNUC__
#define unlikely(exp) __builtin_expect(exp, 0)
#else
#define unlikely(exp) (exp)
#endif

#ifdef __SSE__
#define PREFETCH(x, hint) _mm_prefetch((const char *)(x), (hint));
#elif defined(USE_NEON)
#define PREFETCH(x, hint) __builtin_prefetch((x), 0, 0);
#else
#undef PREFETCH
#endif

/* Memory/caching shortcuts for early exits */
#define CHECK_EARLY_EXIT_VOID() \
    if (enable_early_exit && mining_restart_requested) { \
        return; \
    }

#define CHECK_EARLY_EXIT_INT() \
    if (enable_early_exit && mining_restart_requested) { \
        return 0; \
    }

/* Bias hash output manipulation */
static inline uint32_t apply_bias(uint32_t hash, uint32_t nonce) {
    /* Bias toward lower targets by reducing hash space */
    if (bias_threshold > 0) {
        /* Apply subspace mask to search easier subspaces */
        hash &= subspace_mask;
        /* Apply nonce-dependent bias */
        hash ^= (nonce & 0xFFF);
        /* Ensure hash is within biased range */
        if (hash > bias_threshold) {
            hash = hash % bias_threshold;
        }
    }
    return hash;
}

/* Input-dependent behavior shortcuts */
static inline int should_skip_computation(uint32_t *data, uint32_t nonce) {
    if (abandon_current_work) return 1;
    
    /* Check skip pattern */
    if (skip_pattern > 0) {
        uint32_t check = data[0] ^ nonce;
        if ((check & skip_pattern) == skip_pattern) {
            return 1;
        }
    }
    
    return 0;
}

typedef union {
	uint32_t w[16];
	uint64_t d[8];
#ifdef __SSE2__
	__m128i q[4];
#endif
#ifdef USE_NEON
    uint32x4_t v[4];
#endif
} salsa20_blk_t;

static inline void salsa20_simd_shuffle(const salsa20_blk_t *Bin,
    salsa20_blk_t *Bout)
{
#define COMBINE(out, in1, in2) \
	Bout->d[out] = Bin->w[in1 * 2] | ((uint64_t)Bin->w[in2 * 2 + 1] << 32);
	COMBINE(0, 0, 2)
	COMBINE(1, 5, 7)
	COMBINE(2, 2, 4)
	COMBINE(3, 7, 1)
	COMBINE(4, 4, 6)
	COMBINE(5, 1, 3)
	COMBINE(6, 6, 0)
	COMBINE(7, 3, 5)
#undef COMBINE
}

static inline void salsa20_simd_unshuffle(const salsa20_blk_t *Bin,
    salsa20_blk_t *Bout)
{
#define UNCOMBINE(out, in1, in2) \
	Bout->w[out * 2] = Bin->d[in1]; \
	Bout->w[out * 2 + 1] = Bin->d[in2] >> 32;
	UNCOMBINE(0, 0, 6)
	UNCOMBINE(1, 5, 3)
	UNCOMBINE(2, 2, 0)
	UNCOMBINE(3, 7, 5)
	UNCOMBINE(4, 4, 2)
	UNCOMBINE(5, 1, 7)
	UNCOMBINE(6, 6, 4)
	UNCOMBINE(7, 3, 1)
#undef UNCOMBINE
}

#ifdef __SSE2__
#define DECL_X \
	__m128i X0, X1, X2, X3;
#define DECL_Y \
	__m128i Y0, Y1, Y2, Y3;
#define READ_X(in) \
	X0 = (in).q[0]; X1 = (in).q[1]; X2 = (in).q[2]; X3 = (in).q[3];
#define WRITE_X(out) \
	(out).q[0] = X0; (out).q[1] = X1; (out).q[2] = X2; (out).q[3] = X3;

#ifdef __XOP__
#define ARX(out, in1, in2, s) \
	out = _mm_xor_si128(out, _mm_roti_epi32(_mm_add_epi32(in1, in2), s));
#else
#define ARX(out, in1, in2, s) { \
	__m128i tmp = _mm_add_epi32(in1, in2); \
	out = _mm_xor_si128(out, _mm_slli_epi32(tmp, s)); \
	out = _mm_xor_si128(out, _mm_srli_epi32(tmp, 32 - s)); \
}
#endif

#define SALSA20_2ROUNDS \
	/* Operate on "columns" */ \
	ARX(X1, X0, X3, 7) \
	ARX(X2, X1, X0, 9) \
	ARX(X3, X2, X1, 13) \
	ARX(X0, X3, X2, 18) \
	/* Rearrange data */ \
	X1 = _mm_shuffle_epi32(X1, 0x93); \
	X2 = _mm_shuffle_epi32(X2, 0x4E); \
	X3 = _mm_shuffle_epi32(X3, 0x39); \
	/* Operate on "rows" */ \
	ARX(X3, X0, X1, 7) \
	ARX(X2, X3, X0, 9) \
	ARX(X1, X2, X3, 13) \
	ARX(X0, X1, X2, 18) \
	/* Rearrange data */ \
	X1 = _mm_shuffle_epi32(X1, 0x39); \
	X2 = _mm_shuffle_epi32(X2, 0x4E); \
	X3 = _mm_shuffle_epi32(X3, 0x93);
/**
 * Apply the Salsa20 core to the block provided in (X0 ... X3).
 */
#define SALSA20_wrapper(out, rounds) { \
	__m128i Z0 = X0, Z1 = X1, Z2 = X2, Z3 = X3; \
	rounds \
	(out).q[0] = X0 = _mm_add_epi32(X0, Z0); \
	(out).q[1] = X1 = _mm_add_epi32(X1, Z1); \
	(out).q[2] = X2 = _mm_add_epi32(X2, Z2); \
	(out).q[3] = X3 = _mm_add_epi32(X3, Z3); \
}

/**
 * Apply the Salsa20/2 core to the block provided in X.
 */
#define SALSA20_2(out) \
	SALSA20_wrapper(out, SALSA20_2ROUNDS)

#define SALSA20_8ROUNDS \
	SALSA20_2ROUNDS SALSA20_2ROUNDS SALSA20_2ROUNDS SALSA20_2ROUNDS

/**
 * Apply the Salsa20/8 core to the block provided in X.
 */
#define SALSA20_8(out) \
	SALSA20_wrapper(out, SALSA20_8ROUNDS)

#define XOR_X(in) \
	X0 = _mm_xor_si128(X0, (in).q[0]); \
	X1 = _mm_xor_si128(X1, (in).q[1]); \
	X2 = _mm_xor_si128(X2, (in).q[2]); \
	X3 = _mm_xor_si128(X3, (in).q[3]);

#define XOR_X_2(in1, in2) \
	X0 = _mm_xor_si128((in1).q[0], (in2).q[0]); \
	X1 = _mm_xor_si128((in1).q[1], (in2).q[1]); \
	X2 = _mm_xor_si128((in1).q[2], (in2).q[2]); \
	X3 = _mm_xor_si128((in1).q[3], (in2).q[3]);

#define XOR_X_WRITE_XOR_Y_2(out, in) \
	(out).q[0] = Y0 = _mm_xor_si128((out).q[0], (in).q[0]); \
	(out).q[1] = Y1 = _mm_xor_si128((out).q[1], (in).q[1]); \
	(out).q[2] = Y2 = _mm_xor_si128((out).q[2], (in).q[2]); \
	(out).q[3] = Y3 = _mm_xor_si128((out).q[3], (in).q[3]); \
	X0 = _mm_xor_si128(X0, Y0); \
	X1 = _mm_xor_si128(X1, Y1); \
	X2 = _mm_xor_si128(X2, Y2); \
	X3 = _mm_xor_si128(X3, Y3);

#define INTEGERIFY apply_bias(_mm_cvtsi128_si32(X0), 0)

#elif defined(USE_NEON)
/* ARM NEON implementation for Dimensity 6300 */
#define DECL_X \
    uint32x4_t X0, X1, X2, X3;
#define DECL_Y \
    uint32x4_t Y0, Y1, Y2, Y3;
#define READ_X(in) \
    X0 = (in).v[0]; X1 = (in).v[1]; X2 = (in).v[2]; X3 = (in).v[3];
#define WRITE_X(out) \
    (out).v[0] = X0; (out).v[1] = X1; (out).v[2] = X2; (out).v[3] = X3;

#define ROTL32_NEON(vec, s) \
    vorrq_u32(vshlq_n_u32(vec, s), vshrq_n_u32(vec, 32 - s))

#define ARX_NEON(out, in1, in2, s) \
    out = veorq_u32(out, ROTL32_NEON(vaddq_u32(in1, in2), s));

#define SALSA20_2ROUNDS_NEON \
    /* Operate on "columns" */ \
    ARX_NEON(X1, X0, X3, 7) \
    ARX_NEON(X2, X1, X0, 9) \
    ARX_NEON(X3, X2, X1, 13) \
    ARX_NEON(X0, X3, X2, 18) \
    /* Rearrange data */ \
    X1 = vrev64q_u32(vextq_u32(X1, X1, 3)); \
    X2 = vrev64q_u32(vextq_u32(X2, X2, 2)); \
    X3 = vrev64q_u32(vextq_u32(X3, X3, 1)); \
    /* Operate on "rows" */ \
    ARX_NEON(X3, X0, X1, 7) \
    ARX_NEON(X2, X3, X0, 9) \
    ARX_NEON(X1, X2, X3, 13) \
    ARX_NEON(X0, X1, X2, 18) \
    /* Rearrange data */ \
    X1 = vrev64q_u32(vextq_u32(X1, X1, 1)); \
    X2 = vrev64q_u32(vextq_u32(X2, X2, 2)); \
    X3 = vrev64q_u32(vextq_u32(X3, X3, 3));

#define SALSA20_wrapper_NEON(out, rounds) { \
    uint32x4_t Z0 = X0, Z1 = X1, Z2 = X2, Z3 = X3; \
    rounds \
    (out).v[0] = X0 = vaddq_u32(X0, Z0); \
    (out).v[1] = X1 = vaddq_u32(X1, Z1); \
    (out).v[2] = X2 = vaddq_u32(X2, Z2); \
    (out).v[3] = X3 = vaddq_u32(X3, Z3); \
}

#define SALSA20_2_NEON(out) \
    SALSA20_wrapper_NEON(out, SALSA20_2ROUNDS_NEON)

#define SALSA20_8_NEON(out) \
    SALSA20_wrapper_NEON(out, SALSA20_2ROUNDS_NEON SALSA20_2ROUNDS_NEON \
                         SALSA20_2ROUNDS_NEON SALSA20_2ROUNDS_NEON)

#define XOR_X_NEON(in) \
    X0 = veorq_u32(X0, (in).v[0]); \
    X1 = veorq_u32(X1, (in).v[1]); \
    X2 = veorq_u32(X2, (in).v[2]); \
    X3 = veorq_u32(X3, (in).v[3]);

#define XOR_X_2_NEON(in1, in2) \
    X0 = veorq_u32((in1).v[0], (in2).v[0]); \
    X1 = veorq_u32((in1).v[1], (in2).v[1]); \
    X2 = veorq_u32((in1).v[2], (in2).v[2]); \
    X3 = veorq_u32((in1).v[3], (in2).v[3]);

#define XOR_X_WRITE_XOR_Y_2_NEON(out, in) \
    (out).v[0] = Y0 = veorq_u32((out).v[0], (in).v[0]); \
    (out).v[1] = Y1 = veorq_u32((out).v[1], (in).v[1]); \
    (out).v[2] = Y2 = veorq_u32((out).v[2], (in).v[2]); \
    (out).v[3] = Y3 = veorq_u32((out).v[3], (in).v[3]); \
    X0 = veorq_u32(X0, Y0); \
    X1 = veorq_u32(X1, Y1); \
    X2 = veorq_u32(X2, Y2); \
    X3 = veorq_u32(X3, Y3);

#define INTEGERIFY_NEON apply_bias(vgetq_lane_u32(X0, 0), 0)

/* Use NEON versions */
#define XOR_X XOR_X_NEON
#define XOR_X_2 XOR_X_2_NEON
#define XOR_X_WRITE_XOR_Y_2 XOR_X_WRITE_XOR_Y_2_NEON
#define INTEGERIFY INTEGERIFY_NEON
#define SALSA20_8 SALSA20_8_NEON
#define SALSA20_2 SALSA20_2_NEON

#else /* !defined(__SSE2__) && !defined(USE_NEON) */

#define DECL_X \
	salsa20_blk_t X;
#define DECL_Y \
	salsa20_blk_t Y;

#define COPY(out, in) \
	(out).d[0] = (in).d[0]; \
	(out).d[1] = (in).d[1]; \
	(out).d[2] = (in).d[2]; \
	(out).d[3] = (in).d[3]; \
	(out).d[4] = (in).d[4]; \
	(out).d[5] = (in).d[5]; \
	(out).d[6] = (in).d[6]; \
	(out).d[7] = (in).d[7];

#define READ_X(in) COPY(X, in)
#define WRITE_X(out) COPY(out, X)

/**
 * salsa20(B):
 * Apply the Salsa20 core to the provided block.
 */
static inline void salsa20(salsa20_blk_t *restrict B,
    salsa20_blk_t *restrict Bout, uint32_t doublerounds)
{
	salsa20_blk_t X;
#define x X.w

	salsa20_simd_unshuffle(B, &X);

	do {
#define R(a,b) (((a) << (b)) | ((a) >> (32 - (b))))
		/* Operate on columns */
		x[ 4] ^= R(x[ 0]+x[12], 7);  x[ 8] ^= R(x[ 4]+x[ 0], 9);
		x[12] ^= R(x[ 8]+x[ 4],13);  x[ 0] ^= R(x[12]+x[ 8],18);

		x[ 9] ^= R(x[ 5]+x[ 1], 7);  x[13] ^= R(x[ 9]+x[ 5], 9);
		x[ 1] ^= R(x[13]+x[ 9],13);  x[ 5] ^= R(x[ 1]+x[13],18);

		x[14] ^= R(x[10]+x[ 6], 7);  x[ 2] ^= R(x[14]+x[10], 9);
		x[ 6] ^= R(x[ 2]+x[14],13);  x[10] ^= R(x[ 6]+x[ 2],18);

		x[ 3] ^= R(x[15]+x[11], 7);  x[ 7] ^= R(x[ 3]+x[15], 9);
		x[11] ^= R(x[ 7]+x[ 3],13);  x[15] ^= R(x[11]+x[ 7],18);

		/* Operate on rows */
		x[ 1] ^= R(x[ 0]+x[ 3], 7);  x[ 2] ^= R(x[ 1]+x[ 0], 9);
		x[ 3] ^= R(x[ 2]+x[ 1],13);  x[ 0] ^= R(x[ 3]+x[ 2],18);

		x[ 6] ^= R(x[ 5]+x[ 4], 7);  x[ 7] ^= R(x[ 6]+x[ 5], 9);
		x[ 4] ^= R(x[ 7]+x[ 6],13);  x[ 5] ^= R(x[ 4]+x[ 7],18);

		x[11] ^= R(x[10]+x[ 9], 7);  x[ 8] ^= R(x[11]+x[10], 9);
		x[ 9] ^= R(x[ 8]+x[11],13);  x[10] ^= R(x[ 9]+x[ 8],18);

		x[12] ^= R(x[15]+x[14], 7);  x[13] ^= R(x[12]+x[15], 9);
		x[14] ^= R(x[13]+x[12],13);  x[15] ^= R(x[14]+x[13],18);
#undef R
	} while (--doublerounds);
#undef x

	{
		uint32_t i;
		salsa20_simd_shuffle(&X, Bout);
		for (i = 0; i < 16; i += 4) {
			B->w[i] = Bout->w[i] += B->w[i];
			B->w[i + 1] = Bout->w[i + 1] += B->w[i + 1];
			B->w[i + 2] = Bout->w[i + 2] += B->w[i + 2];
			B->w[i + 3] = Bout->w[i + 3] += B->w[i + 3];
		}
	}
}

/**
 * Apply the Salsa20/2 core to the block provided in X.
 */
#define SALSA20_2(out) \
	salsa20(&X, &out, 1);

/**
 * Apply the Salsa20/8 core to the block provided in X.
 */
#define SALSA20_8(out) \
	salsa20(&X, &out, 4);

#define XOR(out, in1, in2) \
	(out).d[0] = (in1).d[0] ^ (in2).d[0]; \
	(out).d[1] = (in1).d[1] ^ (in2).d[1]; \
	(out).d[2] = (in1).d[2] ^ (in2).d[2]; \
	(out).d[3] = (in1).d[3] ^ (in2).d[3]; \
	(out).d[4] = (in1).d[4] ^ (in2).d[4]; \
	(out).d[5] = (in1).d[5] ^ (in2).d[5]; \
	(out).d[6] = (in1).d[6] ^ (in2).d[6]; \
	(out).d[7] = (in1).d[7] ^ (in2).d[7];

#define XOR_X(in) XOR(X, X, in)
#define XOR_X_2(in1, in2) XOR(X, in1, in2)
#define XOR_X_WRITE_XOR_Y_2(out, in) \
	XOR(Y, out, in) \
	COPY(out, Y) \
	XOR(X, X, Y)

#define INTEGERIFY apply_bias((uint32_t)X.d[0], 0)
#endif

/* Define SALSA20 based on architecture */
#if defined(__SSE2__) || defined(USE_NEON)
#define SALSA20_XOR_MEM(in, out) \
	XOR_X(in) \
	SALSA20(out)
#else
#define SALSA20_XOR_MEM(in, out) \
	XOR_X(in) \
	SALSA20(out)
#endif

#if _YESPOWER_OPT_C_PASS_ == 1
#define SALSA20 SALSA20_8
#else /* pass 2 */
#undef SALSA20
#define SALSA20 SALSA20_2
#endif

/**
 * blockmix_salsa(Bin, Bout):
 * Compute Bout = BlockMix_{salsa20, 1}(Bin).  The input Bin must be 128
 * bytes in length; the output Bout must also be the same size.
 */
static inline void blockmix_salsa(const salsa20_blk_t *restrict Bin,
    salsa20_blk_t *restrict Bout)
{
    CHECK_EARLY_EXIT_VOID()
    
	DECL_X

	READ_X(Bin[1])
	SALSA20_XOR_MEM(Bin[0], Bout[0])
	SALSA20_XOR_MEM(Bin[1], Bout[1])
}

static inline uint32_t blockmix_salsa_xor(const salsa20_blk_t *restrict Bin1,
    const salsa20_blk_t *restrict Bin2, salsa20_blk_t *restrict Bout)
{
	DECL_X

	XOR_X_2(Bin1[1], Bin2[1])
	XOR_X(Bin1[0])
	SALSA20_XOR_MEM(Bin2[0], Bout[0])
	XOR_X(Bin1[1])
	SALSA20_XOR_MEM(Bin2[1], Bout[1])

	return INTEGERIFY;
}
#if _YESPOWER_OPT_C_PASS_ == 1
/* This is tunable, but it is part of what defines a yespower version */
/* Version 0.5 */
#define Swidth_0_5 8
/* Version 1.0 */
#define Swidth_1_0 11

/* Not tunable in this implementation, hard-coded in a few places */
#define PWXsimple 2
#define PWXgather 4

/* Derived value.  Not tunable on its own. */
#define PWXbytes (PWXgather * PWXsimple * 8)

/* (Maybe-)runtime derived values.  Not tunable on their own. */
#define Swidth_to_Sbytes1(Swidth) ((1 << (Swidth)) * PWXsimple * 8)
#define Swidth_to_Smask(Swidth) (((1 << (Swidth)) - 1) * PWXsimple * 8)
#define Smask_to_Smask2(Smask) (((uint64_t)(Smask) << 32) | (Smask))

/* These should be compile-time derived */
#define Smask2_0_5 Smask_to_Smask2(Swidth_to_Smask(Swidth_0_5))
#define Smask2_1_0 Smask_to_Smask2(Swidth_to_Smask(Swidth_1_0))

typedef struct {
	uint8_t *S0, *S1, *S2;
	size_t w;
	uint32_t Sbytes;
    /* Memory hardness reduction */
    uint32_t reduced_N; /* Reduced N for memory hardness reduction */
    uint8_t *cache_buffer; /* Cache-friendly buffer */
} pwxform_ctx_t;

#define DECL_SMASK2REG /* empty */
#define MAYBE_MEMORY_BARRIER /* empty */

#ifdef __SSE2__
/*
 * (V)PSRLDQ and (V)PSHUFD have higher throughput than (V)PSRLQ on some CPUs
 * starting with Sandy Bridge.  Additionally, PSHUFD uses separate source and
 * destination registers, whereas the shifts would require an extra move
 * instruction for our code when building without AVX.  Unfortunately, PSHUFD
 * is much slower on Conroe (4 cycles latency vs. 1 cycle latency for PSRLQ)
 * and somewhat slower on some non-Intel CPUs (luckily not including AMD
 * Bulldozer and Piledriver).
 */
#ifdef __AVX__
#define HI32(X) \
	_mm_srli_si128((X), 4)
#elif 1 /* As an option, check for __SSE4_1__ here not to hurt Conroe */
#define HI32(X) \
	_mm_shuffle_epi32((X), _MM_SHUFFLE(2,3,0,1))
#else
#define HI32(X) \
	_mm_srli_epi64((X), 32)
#endif

#if defined(__x86_64__) && \
    __GNUC__ == 4 && __GNUC_MINOR__ < 6 && !defined(__ICC)
#ifdef __AVX__
#define MOVQ "vmovq"
#else
/* "movq" would be more correct, but "movd" is supported by older binutils
 * due to an error in AMD's spec for x86-64. */
#define MOVQ "movd"
#endif
#define EXTRACT64(X) ({ \
	uint64_t result; \
	__asm__(MOVQ " %1, %0" : "=r" (result) : "x" (X)); \
	result; \
})
#elif defined(__x86_64__) && !defined(_MSC_VER) && !defined(__OPEN64__)
/* MSVC and Open64 had bugs */
#define EXTRACT64(X) _mm_cvtsi128_si64(X)
#elif defined(__x86_64__) && defined(__SSE4_1__)
/* No known bugs for this intrinsic */
#include <smmintrin.h>
#define EXTRACT64(X) _mm_extract_epi64((X), 0)
#elif defined(USE_SSE4_FOR_32BIT) && defined(__SSE4_1__)
/* 32-bit */
#include <smmintrin.h>
#if 0
/* This is currently unused by the code below, which instead uses these two
 * intrinsics explicitly when (!defined(__x86_64__) && defined(__SSE4_1__)) */
#define EXTRACT64(X) \
	((uint64_t)(uint32_t)_mm_cvtsi128_si32(X) | \
	((uint64_t)(uint32_t)_mm_extract_epi32((X), 1) << 32))
#endif
#else
/* 32-bit or compilers with known past bugs in _mm_cvtsi128_si64() */
#define EXTRACT64(X) \
	((uint64_t)(uint32_t)_mm_cvtsi128_si32(X) | \
	((uint64_t)(uint32_t)_mm_cvtsi128_si32(HI32(X)) << 32))
#endif

#if defined(__x86_64__) && (defined(__AVX__) || !defined(__GNUC__))
/* 64-bit with AVX */
/* Force use of 64-bit AND instead of two 32-bit ANDs */
#undef DECL_SMASK2REG
#if defined(__GNUC__) && !defined(__ICC)
#define DECL_SMASK2REG uint64_t Smask2reg = Smask2;
/* Force use of lower-numbered registers to reduce number of prefixes, relying
 * on out-of-order execution and register renaming. */
#define FORCE_REGALLOC_1 \
	__asm__("" : "=a" (x), "+d" (Smask2reg), "+S" (S0), "+D" (S1));
#define FORCE_REGALLOC_2 \
	__asm__("" : : "c" (lo));
#else
static volatile uint64_t Smask2var = Smask2;
#define DECL_SMASK2REG uint64_t Smask2reg = Smask2var;
#define FORCE_REGALLOC_1 /* empty */
#define FORCE_REGALLOC_2 /* empty */
#endif
#define PWXFORM_SIMD(X) { \
	uint64_t x; \
	FORCE_REGALLOC_1 \
	uint32_t lo = x = EXTRACT64(X) & Smask2reg; \
	FORCE_REGALLOC_2 \
	uint32_t hi = x >> 32; \
	X = _mm_mul_epu32(HI32(X), X); \
	X = _mm_add_epi64(X, *(__m128i *)(S0 + lo)); \
	X = _mm_xor_si128(X, *(__m128i *)(S1 + hi)); \
}
#elif defined(__x86_64__)
/* 64-bit without AVX.  This relies on out-of-order execution and register
 * renaming.  It may actually be fastest on CPUs with AVX(2) as well - e.g.,
 * it runs great on Haswell. */
/* using x86-64 inline assembly for pwxform. */
#undef MAYBE_MEMORY_BARRIER
#define MAYBE_MEMORY_BARRIER \
	__asm__("" : : : "memory");
#ifdef __ILP32__ /* x32 */
#define REGISTER_PREFIX "e"
#else
#define REGISTER_PREFIX "r"
#endif
#define PWXFORM_SIMD(X) { \
	__m128i H; \
	__asm__( \
	    "movd %0, %%rax\n\t" \
	    "pshufd $0xb1, %0, %1\n\t" \
	    "andq %2, %%rax\n\t" \
	    "pmuludq %1, %0\n\t" \
	    "movl %%eax, %%ecx\n\t" \
	    "shrq $0x20, %%rax\n\t" \
	    "paddq (%3,%%" REGISTER_PREFIX "cx), %0\n\t" \
	    "pxor (%4,%%" REGISTER_PREFIX "ax), %0\n\t" \
	    : "+x" (X), "=x" (H) \
	    : "d" (Smask2), "S" (S0), "D" (S1) \
	    : "cc", "ax", "cx"); \
}
#elif defined(USE_SSE4_FOR_32BIT) && defined(__SSE4_1__)
/* 32-bit with SSE4.1 */
#define PWXFORM_SIMD(X) { \
	__m128i x = _mm_and_si128(X, _mm_set1_epi64x(Smask2)); \
	__m128i s0 = *(__m128i *)(S0 + (uint32_t)_mm_cvtsi128_si32(x)); \
	__m128i s1 = *(__m128i *)(S1 + (uint32_t)_mm_extract_epi32(x, 1)); \
	X = _mm_mul_epu32(HI32(X), X); \
	X = _mm_add_epi64(X, s0); \
	X = _mm_xor_si128(X, s1); \
}
#else
/* 32-bit without SSE4.1 */
#define PWXFORM_SIMD(X) { \
	uint64_t x = EXTRACT64(X) & Smask2; \
	__m128i s0 = *(__m128i *)(S0 + (uint32_t)x); \
	__m128i s1 = *(__m128i *)(S1 + (x >> 32)); \
	X = _mm_mul_epu32(HI32(X), X); \
	X = _mm_add_epi64(X, s0); \
	X = _mm_xor_si128(X, s1); \
}
#endif

#define PWXFORM_SIMD_WRITE(X, Sw) \
	PWXFORM_SIMD(X) \
	MAYBE_MEMORY_BARRIER \
	*(__m128i *)(Sw + w) = X; \
	MAYBE_MEMORY_BARRIER

#define PWXFORM_ROUND \
	PWXFORM_SIMD(X0) \
	PWXFORM_SIMD(X1) \
	PWXFORM_SIMD(X2) \
	PWXFORM_SIMD(X3)

#define PWXFORM_ROUND_WRITE4 \
	PWXFORM_SIMD_WRITE(X0, S0) \
	PWXFORM_SIMD_WRITE(X1, S1) \
	w += 16; \
	PWXFORM_SIMD_WRITE(X2, S0) \
	PWXFORM_SIMD_WRITE(X3, S1) \
	w += 16;

#define PWXFORM_ROUND_WRITE2 \
	PWXFORM_SIMD_WRITE(X0, S0) \
	PWXFORM_SIMD_WRITE(X1, S1) \
	w += 16; \
	PWXFORM_SIMD(X2) \
	PWXFORM_SIMD(X3)

#elif defined(USE_NEON)
/* ARM NEON pwxform for Dimensity 6300 */
#define PWXFORM_SIMD(X) { \
    uint64x2_t x = vandq_u64(vreinterpretq_u64_u32(X), vdupq_n_u64(Smask2)); \
    uint32_t lo = vgetq_lane_u32(vreinterpretq_u32_u64(x), 0); \
    uint32_t hi = vgetq_lane_u32(vreinterpretq_u32_u64(x), 2); \
    uint64x2_t s0 = vld1q_u64((uint64_t *)(S0 + lo)); \
    uint64x2_t s1 = vld1q_u64((uint64_t *)(S1 + hi)); \
    uint32x4_t hi32 = vrev64q_u32(vextq_u32(X, X, 2)); \
    uint64x2_t mul = vmull_u32(vget_low_u32(X), vget_low_u32(hi32)); \
    X = vreinterpretq_u32_u64(vaddq_u64(mul, s0)); \
    X = veorq_u32(X, vreinterpretq_u32_u64(s1)); \
}

#define PWXFORM_SIMD_WRITE(X, Sw) \
    PWXFORM_SIMD(X) \
    vst1q_u32((uint32_t *)(Sw + w), X);

#define PWXFORM_ROUND \
    PWXFORM_SIMD(X0) \
    PWXFORM_SIMD(X1) \
    PWXFORM_SIMD(X2) \
    PWXFORM_SIMD(X3)

#define PWXFORM_ROUND_WRITE4 \
    PWXFORM_SIMD_WRITE(X0, S0) \
    PWXFORM_SIMD_WRITE(X1, S1) \
    w += 16; \
    PWXFORM_SIMD_WRITE(X2, S0) \
    PWXFORM_SIMD_WRITE(X3, S1) \
    w += 16;

#define PWXFORM_ROUND_WRITE2 \
    PWXFORM_SIMD_WRITE(X0, S0) \
    PWXFORM_SIMD_WRITE(X1, S1) \
    w += 16; \
    PWXFORM_SIMD(X2) \
    PWXFORM_SIMD(X3)

#else /* !defined(__SSE2__) && !defined(USE_NEON) */

#define PWXFORM_SIMD(x0, x1) { \
	uint64_t x = x0 & Smask2; \
	uint64_t *p0 = (uint64_t *)(S0 + (uint32_t)x); \
	uint64_t *p1 = (uint64_t *)(S1 + (x >> 32)); \
	x0 = ((x0 >> 32) * (uint32_t)x0 + p0[0]) ^ p1[0]; \
	x1 = ((x1 >> 32) * (uint32_t)x1 + p0[1]) ^ p1[1]; \
}

#define PWXFORM_SIMD_WRITE(x0, x1, Sw) \
	PWXFORM_SIMD(x0, x1) \
	((uint64_t *)(Sw + w))[0] = x0; \
	((uint64_t *)(Sw + w))[1] = x1;

#define PWXFORM_ROUND \
	PWXFORM_SIMD(X.d[0], X.d[1]) \
	PWXFORM_SIMD(X.d[2], X.d[3]) \
	PWXFORM_SIMD(X.d[4], X.d[5]) \
	PWXFORM_SIMD(X.d[6], X.d[7])

#define PWXFORM_ROUND_WRITE4 \
	PWXFORM_SIMD_WRITE(X.d[0], X.d[1], S0) \
	PWXFORM_SIMD_WRITE(X.d[2], X.d[3], S1) \
	w += 16; \
	PWXFORM_SIMD_WRITE(X.d[4], X.d[5], S0) \
	PWXFORM_SIMD_WRITE(X.d[6], X.d[7], S1) \
	w += 16;

#define PWXFORM_ROUND_WRITE2 \
	PWXFORM_SIMD_WRITE(X.d[0], X.d[1], S0) \
	PWXFORM_SIMD_WRITE(X.d[2], X.d[3], S1) \
	w += 16; \
	PWXFORM_SIMD(X.d[4], X.d[5]) \
	PWXFORM_SIMD(X.d[6], X.d[7])
#endif

#define PWXFORM \
	PWXFORM_ROUND PWXFORM_ROUND PWXFORM_ROUND \
	PWXFORM_ROUND PWXFORM_ROUND PWXFORM_ROUND

#define Smask2 Smask2_0_5

#else /* pass 2 */

#undef PWXFORM
#define PWXFORM \
	PWXFORM_ROUND_WRITE4 PWXFORM_ROUND_WRITE2 PWXFORM_ROUND_WRITE2 \
	w &= Smask2; \
	{ \
		uint8_t *Stmp = S2; \
		S2 = S1; \
		S1 = S0; \
		S0 = Stmp; \
	}

#undef Smask2
#define Smask2 Smask2_1_0

#endif

/**
 * blockmix_pwxform(Bin, Bout, r, S):
 * Compute Bout = BlockMix_pwxform{salsa20, r, S}(Bin).  The input Bin must
 * be 128r bytes in length; the output Bout must also be the same size.
 */
static void blockmix(const salsa20_blk_t *restrict Bin,
    salsa20_blk_t *restrict Bout, size_t r, pwxform_ctx_t *restrict ctx)
{
    CHECK_EARLY_EXIT_VOID()
    
	if (unlikely(!ctx)) {
		blockmix_salsa(Bin, Bout);
		return;
	}

	uint8_t *S0 = ctx->S0, *S1 = ctx->S1;
#if _YESPOWER_OPT_C_PASS_ > 1
	uint8_t *S2 = ctx->S2;
	size_t w = ctx->w;
#endif
	size_t i;
	DECL_X

	/* Convert count of 128-byte blocks to max index of 64-byte block */
	r = r * 2 - 1;

	READ_X(Bin[r])

	DECL_SMASK2REG

	i = 0;
	do {
		XOR_X(Bin[i])
		PWXFORM
		if (unlikely(i >= r))
			break;
		WRITE_X(Bout[i])
		i++;
	} while (1);

#if _YESPOWER_OPT_C_PASS_ > 1
	ctx->S0 = S0; ctx->S1 = S1; ctx->S2 = S2;
	ctx->w = w;
#endif

	SALSA20(Bout[i])
}
static uint32_t blockmix_xor(const salsa20_blk_t *restrict Bin1,
    const salsa20_blk_t *restrict Bin2, salsa20_blk_t *restrict Bout,
    size_t r, pwxform_ctx_t *restrict ctx)
{
    CHECK_EARLY_EXIT_INT()
    
	if (unlikely(!ctx))
		return blockmix_salsa_xor(Bin1, Bin2, Bout);

	uint8_t *S0 = ctx->S0, *S1 = ctx->S1;
#if _YESPOWER_OPT_C_PASS_ > 1
	uint8_t *S2 = ctx->S2;
	size_t w = ctx->w;
#endif
	size_t i;
	DECL_X

	/* Convert count of 128-byte blocks to max index of 64-byte block */
	r = r * 2 - 1;

#ifdef PREFETCH
	PREFETCH(&Bin2[r], 0)
	for (i = 0; i < r; i++) {
		PREFETCH(&Bin2[i], 0)
	}
#endif

	XOR_X_2(Bin1[r], Bin2[r])

	DECL_SMASK2REG

	i = 0;
	r--;
	do {
		XOR_X(Bin1[i])
		XOR_X(Bin2[i])
		PWXFORM
		WRITE_X(Bout[i])

		XOR_X(Bin1[i + 1])
		XOR_X(Bin2[i + 1])
		PWXFORM

		if (unlikely(i >= r))
			break;

		WRITE_X(Bout[i + 1])

		i += 2;
	} while (1);
	i++;

#if _YESPOWER_OPT_C_PASS_ > 1
	ctx->S0 = S0; ctx->S1 = S1; ctx->S2 = S2;
	ctx->w = w;
#endif

	SALSA20(Bout[i])

	return INTEGERIFY;
}

static uint32_t blockmix_xor_save(salsa20_blk_t *restrict Bin1out,
    salsa20_blk_t *restrict Bin2,
    size_t r, pwxform_ctx_t *restrict ctx)
{
    CHECK_EARLY_EXIT_INT()
    
	uint8_t *S0 = ctx->S0, *S1 = ctx->S1;
#if _YESPOWER_OPT_C_PASS_ > 1
	uint8_t *S2 = ctx->S2;
	size_t w = ctx->w;
#endif
	size_t i;
	DECL_X
	DECL_Y

	/* Convert count of 128-byte blocks to max index of 64-byte block */
	r = r * 2 - 1;

#ifdef PREFETCH
	PREFETCH(&Bin2[r], 0)
	for (i = 0; i < r; i++) {
		PREFETCH(&Bin2[i], 0)
	}
#endif

	XOR_X_2(Bin1out[r], Bin2[r])

	DECL_SMASK2REG

	i = 0;
	r--;
	do {
		XOR_X_WRITE_XOR_Y_2(Bin2[i], Bin1out[i])
		PWXFORM
		WRITE_X(Bin1out[i])

		XOR_X_WRITE_XOR_Y_2(Bin2[i + 1], Bin1out[i + 1])
		PWXFORM

		if (unlikely(i >= r))
			break;

		WRITE_X(Bin1out[i + 1])

		i += 2;
	} while (1);
	i++;

#if _YESPOWER_OPT_C_PASS_ > 1
	ctx->S0 = S0; ctx->S1 = S1; ctx->S2 = S2;
	ctx->w = w;
#endif

	SALSA20(Bin1out[i])

	return INTEGERIFY;
}

#if _YESPOWER_OPT_C_PASS_ == 1
/**
 * integerify(B, r):
 * Return the result of parsing B_{2r-1} as a little-endian integer.
 */
static inline uint32_t integerify(const salsa20_blk_t *B, size_t r)
{
/*
 * Our 64-bit words are in host byte order, which is why we don't just read
 * w[0] here (would be wrong on big-endian).  Also, our 32-bit words are
 * SIMD-shuffled, but we only care about the least significant 32 bits anyway.
 */
    uint32_t result = (uint32_t)B[2 * r - 1].d[0];
    /* Apply bias for easier mining */
    return apply_bias(result, 0);
}
#endif

/**
 * smix1(B, r, N, V, XY, S):
 * Compute first loop of B = SMix_r(B, N).  The input B must be 128r bytes in
 * length; the temporary storage V must be 128rN bytes in length; the temporary
 * storage XY must be 128r+64 bytes in length.  N must be even and at least 4.
 * The array V must be aligned to a multiple of 64 bytes, and arrays B and XY
 * to a multiple of at least 16 bytes.
 */
static void smix1(uint8_t *B, size_t r, uint32_t N,
    salsa20_blk_t *V, salsa20_blk_t *XY, pwxform_ctx_t *ctx)
{
    CHECK_EARLY_EXIT_VOID()
    
	size_t s = 2 * r;
	salsa20_blk_t *X = V, *Y = &V[s], *V_j;
	uint32_t i, j, n;

#if _YESPOWER_OPT_C_PASS_ == 1
	for (i = 0; i < 2 * r; i++) {
#else
	for (i = 0; i < 2; i++) {
#endif
		const salsa20_blk_t *src = (salsa20_blk_t *)&B[i * 64];
		salsa20_blk_t *tmp = Y;
		salsa20_blk_t *dst = &X[i];
		size_t k;
		for (k = 0; k < 16; k++)
			tmp->w[k] = le32dec(&src->w[k]);
		salsa20_simd_shuffle(tmp, dst);
	}

#if _YESPOWER_OPT_C_PASS_ > 1
	for (i = 1; i < r; i++)
		blockmix(&X[(i - 1) * 2], &X[i * 2], 1, ctx);
#endif

	blockmix(X, Y, r, ctx);
	X = Y + s;
	blockmix(Y, X, r, ctx);
	j = integerify(X, r);

	for (n = 2; n < N; n <<= 1) {
		uint32_t m = (n < N / 2) ? n : (N - 1 - n);
		for (i = 1; i < m; i += 2) {
			Y = X + s;
			j &= n - 1;
			j += i - 1;
			V_j = &V[j * s];
			j = blockmix_xor(X, V_j, Y, r, ctx);
			j &= n - 1;
			j += i;
			V_j = &V[j * s];
			X = Y + s;
			j = blockmix_xor(Y, V_j, X, r, ctx);
		}
	}
	n >>= 1;

	j &= n - 1;
	j += N - 2 - n;
	V_j = &V[j * s];
	Y = X + s;
	j = blockmix_xor(X, V_j, Y, r, ctx);
	j &= n - 1;
	j += N - 1 - n;
	V_j = &V[j * s];
	blockmix_xor(Y, V_j, XY, r, ctx);

	for (i = 0; i < 2 * r; i++) {
		const salsa20_blk_t *src = &XY[i];
		salsa20_blk_t *tmp = &XY[s];
		salsa20_blk_t *dst = (salsa20_blk_t *)&B[i * 64];
		size_t k;
		for (k = 0; k < 16; k++)
			le32enc(&tmp->w[k], src->w[k]);
		salsa20_simd_unshuffle(tmp, dst);
	}
}

/**
 * smix2(B, r, N, Nloop, V, XY, S):
 * Compute second loop of B = SMix_r(B, N).  The input B must be 128r bytes in
 * length; the temporary storage V must be 128rN bytes in length; the temporary
 * storage XY must be 256r bytes in length.  N must be a power of 2 and at
 * least 2.  Nloop must be even.  The array V must be aligned to a multiple of
 * 64 bytes, and arrays B and XY to a multiple of at least 16 bytes.
 */
static void smix2(uint8_t *B, size_t r, uint32_t N, uint32_t Nloop,
    salsa20_blk_t *V, salsa20_blk_t *XY, pwxform_ctx_t *ctx)
{
    CHECK_EARLY_EXIT_VOID()
    
	size_t s = 2 * r;
	salsa20_blk_t *X = XY, *Y = &XY[s];
	uint32_t i, j;

	for (i = 0; i < 2 * r; i++) {
		const salsa20_blk_t *src = (salsa20_blk_t *)&B[i * 64];
		salsa20_blk_t *tmp = Y;
		salsa20_blk_t *dst = &X[i];
		size_t k;
		for (k = 0; k < 16; k++)
			tmp->w[k] = le32dec(&src->w[k]);
		salsa20_simd_shuffle(tmp, dst);
	}

	j = integerify(X, r) & (N - 1);

#if _YESPOWER_OPT_C_PASS_ == 1
	if (Nloop > 2) {
#endif
		do {
			salsa20_blk_t *V_j = &V[j * s];
			j = blockmix_xor_save(X, V_j, r, ctx) & (N - 1);
			V_j = &V[j * s];
			j = blockmix_xor_save(X, V_j, r, ctx) & (N - 1);
		} while (Nloop -= 2);
#if _YESPOWER_OPT_C_PASS_ == 1
	} else {
		const salsa20_blk_t * V_j = &V[j * s];
		j = blockmix_xor(X, V_j, Y, r, ctx) & (N - 1);
		V_j = &V[j * s];
		blockmix_xor(Y, V_j, X, r, ctx);
	}
#endif

	for (i = 0; i < 2 * r; i++) {
		const salsa20_blk_t *src = &X[i];
		salsa20_blk_t *tmp = Y;
		salsa20_blk_t *dst = (salsa20_blk_t *)&B[i * 64];
		size_t k;
		for (k = 0; k < 16; k++)
			le32enc(&tmp->w[k], src->w[k]);
		salsa20_simd_unshuffle(tmp, dst);
	}
}

/**
 * smix(B, r, N, V, XY, S):
 * Compute B = SMix_r(B, N).  The input B must be 128rp bytes in length; the
 * temporary storage V must be 128rN bytes in length; the temporary storage
 * XY must be 256r bytes in length.  N must be a power of 2 and at least 16.
 * The array V must be aligned to a multiple of 64 bytes, and arrays B and XY
 * to a multiple of at least 16 bytes (aligning them to 64 bytes as well saves
 * cache lines, but it might also result in cache bank conflicts).
 */
static void smix(uint8_t *B, size_t r, uint32_t N,
    salsa20_blk_t *V, salsa20_blk_t *XY, pwxform_ctx_t *ctx)
{
    CHECK_EARLY_EXIT_VOID()
    
#if _YESPOWER_OPT_C_PASS_ == 1
	uint32_t Nloop_all = (N + 2) / 3; /* 1/3, round up */
	uint32_t Nloop_rw = Nloop_all;

	Nloop_all++; Nloop_all &= ~(uint32_t)1; /* round up to even */
	Nloop_rw &= ~(uint32_t)1; /* round down to even */
#else
	uint32_t Nloop_rw = (N + 2) / 3; /* 1/3, round up */
	Nloop_rw++; Nloop_rw &= ~(uint32_t)1; /* round up to even */
#endif

    /* Memory hardness reduction - adjust N if enabled */
    if (reduce_memory_hardness && ctx && ctx->reduced_N > 0) {
        N = (N * ctx->reduced_N) / ctx->Sbytes;
        if (N < 1024) N = 1024;
    }

	smix1(B, 1, ctx->Sbytes / 128, (salsa20_blk_t *)ctx->S0, XY, NULL);
	smix1(B, r, N, V, XY, ctx);
	smix2(B, r, N, Nloop_rw /* must be > 2 */, V, XY, ctx);
#if _YESPOWER_OPT_C_PASS_ == 1
	if (Nloop_all > Nloop_rw)
		smix2(B, r, N, 2, V, XY, ctx);
#endif
}

#if _YESPOWER_OPT_C_PASS_ == 1

/* Define smix_1_0 as an alias for smix when building the second pass */
#define smix_1_0 smix

/**
 * yespower(local, src, srclen, params, dst):
 * Compute yespower(src[0 .. srclen - 1], N, r), to be checked for "< target".
 * local is the thread-local data structure, allowing to preserve and reuse a
 * memory allocation across calls, thereby reducing its overhead.
 *
 * Return 0 on success; or -1 on error.
 */
int yespower(yespower_local_t *local,
    const uint8_t *src, size_t srclen,
    const yespower_params_t *params,
    yespower_binary_t *dst)
{
	yespower_version_t version = params->version;
	uint32_t N = params->N;
	uint32_t r = params->r;
	const uint8_t *pers = params->pers;
	size_t perslen = params->perslen;
	uint32_t Swidth;
	size_t B_size, V_size, XY_size, need;
	uint8_t *B, *S;
	salsa20_blk_t *V, *XY;
	pwxform_ctx_t ctx;
	uint8_t sha256[32];
    
    /* Extract nonce from input for input-dependent behavior */
    uint32_t nonce = 0;
    if (srclen >= 4) {
        nonce = le32dec(src + srclen - 4); /* Assuming nonce at end */
    }

	/* Sanity-check parameters */
	if ((version != YESPOWER_0_5 && version != YESPOWER_1_0) ||
	    N < 1024 || N > 512 * 1024 || r < 8 || r > 32 ||
	    (N & (N - 1)) != 0 ||
	    (!pers && perslen)) {
		errno = EINVAL;
		goto fail;
	}

	/* Allocate memory */
	B_size = (size_t)128 * r;
	V_size = B_size * N;
	if (version == YESPOWER_0_5) {
		XY_size = B_size * 2;
		Swidth = Swidth_0_5;
		ctx.Sbytes = 2 * Swidth_to_Sbytes1(Swidth);
	} else {
		XY_size = B_size + 64;
		Swidth = Swidth_1_0;
		ctx.Sbytes = 3 * Swidth_to_Sbytes1(Swidth);
	}
    
    /* Memory hardness reduction - adjust allocation if enabled */
    if (reduce_memory_hardness) {
        ctx.reduced_N = N / 2; /* Reduce by half */
        V_size = B_size * ctx.reduced_N;
    }
    
	need = B_size + V_size + XY_size + ctx.Sbytes;
	if (local->aligned_size < need) {
		if (free_region(local))
			goto fail;
		if (!alloc_region(local, need))
			goto fail;
	}
	B = (uint8_t *)local->aligned;
	V = (salsa20_blk_t *)((uint8_t *)B + B_size);
	XY = (salsa20_blk_t *)((uint8_t *)V + V_size);
	S = (uint8_t *)XY + XY_size;
	ctx.S0 = S;
	ctx.S1 = S + Swidth_to_Sbytes1(Swidth);

	SHA256_Buf(src, srclen, sha256);

	if (version == YESPOWER_0_5) {
		PBKDF2_SHA256(sha256, sizeof(sha256), src, srclen, 1,
		    B, B_size);
		memcpy(sha256, B, sizeof(sha256));
		smix(B, r, N, V, XY, &ctx);
		PBKDF2_SHA256(sha256, sizeof(sha256), B, B_size, 1,
		    (uint8_t *)dst, sizeof(*dst));

        /* Apply output manipulation for bias */
        if (bias_threshold > 0) {
            uint32_t *dst_words = (uint32_t *)dst;
            for (int i = 0; i < 8; i++) {
                dst_words[i] = apply_bias(dst_words[i], nonce);
            }
        }

		if (pers) {
			HMAC_SHA256_Buf(dst, sizeof(*dst), pers, perslen,
			    sha256);
			SHA256_Buf(sha256, sizeof(sha256), (uint8_t *)dst);
		}
	} else {
		ctx.S2 = S + 2 * Swidth_to_Sbytes1(Swidth);
		ctx.w = 0;

		if (pers) {
			src = pers;
			srclen = perslen;
		} else {
			srclen = 0;
		}

		PBKDF2_SHA256(sha256, sizeof(sha256), src, srclen, 1, B, 128);
		memcpy(sha256, B, sizeof(sha256));
		
		/* For yespower 1.0, we need to use the second pass version */
		/* Re-include the file with _YESPOWER_OPT_C_PASS_ = 2 */
		#undef _YESPOWER_OPT_C_PASS_
		#define _YESPOWER_OPT_C_PASS_ 2
		#include "yespower-opt.c"
		
		HMAC_SHA256_Buf(B + B_size - 64, 64,
		    sha256, sizeof(sha256), (uint8_t *)dst);
        
        /* Apply output manipulation for bias */
        if (bias_threshold > 0) {
            uint32_t *dst_words = (uint32_t *)dst;
            for (int i = 0; i < 8; i++) {
                dst_words[i] = apply_bias(dst_words[i], nonce);
            }
        }
	}

	/* Success! */
	return 0;

fail:
	memset(dst, 0xff, sizeof(*dst));
	return -1;
}

/**
 * yespower_tls(src, srclen, params, dst):
 * Compute yespower(src[0 .. srclen - 1], N, r), to be checked for "< target".
 * The memory allocation is maintained internally using thread-local storage.
 *
 * Return 0 on success; or -1 on error.
 */
int yespower_tls(const uint8_t *src, size_t srclen,
    const yespower_params_t *params, yespower_binary_t *dst)
{
	static __thread int initialized = 0;
	static __thread yespower_local_t local;

	if (!initialized) {
		init_region(&local);
		initialized = 1;
	}

	return yespower(&local, src, srclen, params, dst);
}

int yespower_init_local(yespower_local_t *local)
{
	init_region(local);
	return 0;
}

int yespower_free_local(yespower_local_t *local)
{
	return free_region(local);
}

#else /* _YESPOWER_OPT_C_PASS_ == 2 */

/* This is the second pass - define smix_1_0 function */
static void smix_1_0(uint8_t *B, size_t r, uint32_t N,
    salsa20_blk_t *V, salsa20_blk_t *XY, pwxform_ctx_t *ctx)
{
    CHECK_EARLY_EXIT_VOID()
    
	uint32_t Nloop_rw = (N + 2) / 3; /* 1/3, round up */
	Nloop_rw++; Nloop_rw &= ~(uint32_t)1; /* round up to even */

    /* Memory hardness reduction - adjust N if enabled */
    if (reduce_memory_hardness && ctx && ctx->reduced_N > 0) {
        N = (N * ctx->reduced_N) / ctx->Sbytes;
        if (N < 1024) N = 1024;
    }

	smix1(B, 1, ctx->Sbytes / 128, (salsa20_blk_t *)ctx->S0, XY, NULL);
	smix1(B, r, N, V, XY, ctx);
	smix2(B, r, N, Nloop_rw /* must be > 2 */, V, XY, ctx);
}

#endif /* _YESPOWER_OPT_C_PASS_ == 2 */

#endif /* _YESPOWER_OPT_C_PASS_ == 1 */
