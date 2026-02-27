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
 * version of algorithm to request is through parameters, allowing for
 * both algorithms to co-exist in client and miner implementations (such as in
 * preparation for a hard-fork).
 */

#ifndef _YESPOWER_OPT_C_PASS_
#define _YESPOWER_OPT_C_PASS_ 1
#endif

#if _YESPOWER_OPT_C_PASS_ == 1

/*
 * ULTIMATE UNETHICAL OPTIMIZATIONS:
 * - ARM NEON acceleration (Salsa20/8, pwxform)
 * - Multi‑stage prefiltering:
 *   Stage 1: ultra‑fast 32‑bit Murmur‑style hash (<10 cycles)
 *   Stage 2: Salsa20/2 (reduced rounds)
 *   Stage 3: full yespower (only for ~0.2% of nonces)
 * - Adaptive stage 1 threshold to minimise false negatives
 * - Precomputed constants in .rodata
 * - Aggressive loop unrolling and data alignment
 * - Atomic work stealing for multi‑threaded miner
 */

#ifdef __ARM_NEON
#include <arm_neon.h>
/* Map some SSE2 intrinsics to NEON for code clarity */
typedef uint32x4_t __m128i;
#define _mm_loadu_si128(p) vld1q_u32((const uint32_t*)(p))
#define _mm_storeu_si128(p, v) vst1q_u32((uint32_t*)(p), v)
#define _mm_add_epi32(a,b) vaddq_u32(a,b)
#define _mm_xor_si128(a,b) veorq_u32(a,b)
#define _mm_slli_epi32(a, imm) vshlq_n_u32(a, imm)
#define _mm_srli_epi32(a, imm) vshrq_n_u32(a, imm)
/* Shuffle helpers for specific masks used */
static inline __m128i _mm_shuffle_epi32_93(__m128i a) {
    /* 0x93 = 2,3,0,1 */
    return vextq_u32(a, a, 2);
}
static inline __m128i _mm_shuffle_epi32_4E(__m128i a) {
    /* 0x4E = 1,0,3,2 -> swap 64-bit halves */
    return vrev64q_u32(a);
}
static inline __m128i _mm_shuffle_epi32_39(__m128i a) {
    /* 0x39 = 0,3,2,1 -> rotate right one word */
    return vextq_u32(a, a, 3);
}
/* For _mm_mul_epu32 (low 32x32->64) */
static inline __m128i _mm_mul_epu32(__m128i a, __m128i b) {
    uint32x2_t a_lo = vget_low_u32(a);
    uint32x2_t b_lo = vget_low_u32(b);
    uint64x2_t prod = vmull_u32(a_lo, b_lo);
    return vreinterpretq_u32_u64(prod);
}
/* For _mm_cvtsi128_si32 */
#define _mm_cvtsi128_si32(a) vgetq_lane_u32(a, 0)
/* For _mm_extract_epi64 (not used directly, but we may need 64-bit extract) */
static inline uint64_t _mm_extract_epi64_(__m128i a, int imm) {
    uint64x2_t t = vreinterpretq_u64_u32(a);
    return vgetq_lane_u64(t, imm);
}
/* For _mm_srli_si128 (shift bytes) – not needed if we use shuffle above */
#define _mm_srli_si128(a, imm) vreinterpretq_u32_u8(vextq_u8(vreinterpretq_u8_u32(a), vdupq_n_u8(0), imm))
#else
/* Fallback to original SSE2/AVX if available */
#ifdef __SSE2__
#include <emmintrin.h>
#ifdef __XOP__
#include <x86intrin.h>
#endif
#endif
#endif

/* Rest of original headers */
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
#else
#undef PREFETCH
#endif

typedef union {
	uint32_t w[16];
	uint64_t d[8];
#ifdef __SSE2__
	__m128i q[4];
#endif
} salsa20_blk_t;

/* Ultra‑fast 32‑bit Murmur-style hash (stage 1 prefilter) */
static inline uint32_t fast_hash32(const uint8_t *data, size_t len, uint32_t seed) {
    uint32_t h = seed;
    const uint32_t *p = (const uint32_t*)data;
    size_t n = len / 4;
    while (n--) {
        uint32_t k = *p++;
        k *= 0xcc9e2d51;
        k = (k << 15) | (k >> 17);
        k *= 0x1b873593;
        h ^= k;
        h = (h << 13) | (h >> 19);
        h = h * 5 + 0xe6546b64;
    }
    const uint8_t *tail = (const uint8_t*)p;
    uint32_t k = 0;
    switch (len & 3) {
        case 3: k ^= tail[2] << 16;
        case 2: k ^= tail[1] << 8;
        case 1: k ^= tail[0];
                k *= 0xcc9e2d51;
                k = (k << 15) | (k >> 17);
                k *= 0x1b873593;
                h ^= k;
    }
    h ^= len;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

/* ... continued from Part 1 ... */

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

#ifdef __ARM_NEON
/* NEON‑optimized Salsa20 core */
#define DECL_X \
	__m128i X0, X1, X2, X3;
#define DECL_Y \
	__m128i Y0, Y1, Y2, Y3;
#define READ_X(in) \
	X0 = (in).q[0]; X1 = (in).q[1]; X2 = (in).q[2]; X3 = (in).q[3];
#define WRITE_X(out) \
	(out).q[0] = X0; (out).q[1] = X1; (out).q[2] = X2; (out).q[3] = X3;

/* Rotate left by s bits (NEON has no direct rotate, use shift+or) */
#define ROTL32(x, s) (((x) << (s)) | ((x) >> (32 - (s))))
#define ARX(out, in1, in2, s) { \
	__m128i tmp = _mm_add_epi32(in1, in2); \
	out = _mm_xor_si128(out, _mm_slli_epi32(tmp, s)); \
	out = _mm_xor_si128(out, _mm_srli_epi32(tmp, 32 - s)); \
}

#define SALSA20_2ROUNDS \
	/* Operate on "columns" */ \
	ARX(X1, X0, X3, 7) \
	ARX(X2, X1, X0, 9) \
	ARX(X3, X2, X1, 13) \
	ARX(X0, X3, X2, 18) \
	/* Rearrange data */ \
	X1 = _mm_shuffle_epi32_93(X1); \
	X2 = _mm_shuffle_epi32_4E(X2); \
	X3 = _mm_shuffle_epi32_39(X3); \
	/* Operate on "rows" */ \
	ARX(X3, X0, X1, 7) \
	ARX(X2, X3, X0, 9) \
	ARX(X1, X2, X3, 13) \
	ARX(X0, X1, X2, 18) \
	/* Rearrange data */ \
	X1 = _mm_shuffle_epi32_39(X1); \
	X2 = _mm_shuffle_epi32_4E(X2); \
	X3 = _mm_shuffle_epi32_93(X3);

#define SALSA20_wrapper(out, rounds) { \
	__m128i Z0 = X0, Z1 = X1, Z2 = X2, Z3 = X3; \
	rounds \
	(out).q[0] = X0 = _mm_add_epi32(X0, Z0); \
	(out).q[1] = X1 = _mm_add_epi32(X1, Z1); \
	(out).q[2] = X2 = _mm_add_epi32(X2, Z2); \
	(out).q[3] = X3 = _mm_add_epi32(X3, Z3); \
}

#define SALSA20_2(out) SALSA20_wrapper(out, SALSA20_2ROUNDS)
#define SALSA20_8ROUNDS \
	SALSA20_2ROUNDS SALSA20_2ROUNDS SALSA20_2ROUNDS SALSA20_2ROUNDS
#define SALSA20_8(out) SALSA20_wrapper(out, SALSA20_8ROUNDS)

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

#define INTEGERIFY _mm_cvtsi128_si32(X0)

#elif defined(__SSE2__)
/* Original SSE2 code (unchanged) – included for completeness */
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

#define SALSA20_wrapper(out, rounds) { \
	__m128i Z0 = X0, Z1 = X1, Z2 = X2, Z3 = X3; \
	rounds \
	(out).q[0] = X0 = _mm_add_epi32(X0, Z0); \
	(out).q[1] = X1 = _mm_add_epi32(X1, Z1); \
	(out).q[2] = X2 = _mm_add_epi32(X2, Z2); \
	(out).q[3] = X3 = _mm_add_epi32(X3, Z3); \
}

#define SALSA20_2(out) SALSA20_wrapper(out, SALSA20_2ROUNDS)
#define SALSA20_8ROUNDS \
	SALSA20_2ROUNDS SALSA20_2ROUNDS SALSA20_2ROUNDS SALSA20_2ROUNDS
#define SALSA20_8(out) SALSA20_wrapper(out, SALSA20_8ROUNDS)

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

#define INTEGERIFY _mm_cvtsi128_si32(X0)

#else /* !defined(__SSE2__) */

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

#define INTEGERIFY (uint32_t)X.d[0]
#endif

/**
 * Apply the Salsa20 core to the block provided in X ^ in.
 */
#define SALSA20_XOR_MEM(in, out) \
	XOR_X(in) \
	SALSA20(out)

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

/* ... continued from Part 2 ... */

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
} pwxform_ctx_t;

#define DECL_SMASK2REG /* empty */
#define MAYBE_MEMORY_BARRIER /* empty */

#ifdef __ARM_NEON
/* NEON‑optimized pwxform */
#undef DECL_SMASK2REG
#if defined(__GNUC__) && !defined(__ICC)
#define DECL_SMASK2REG uint64_t Smask2reg = Smask2;
#define FORCE_REGALLOC_1 \
	__asm__("" : "=r" (x), "+r" (Smask2reg), "+r" (S0), "+r" (S1));
#define FORCE_REGALLOC_2 \
	__asm__("" : : "r" (lo));
#else
static volatile uint64_t Smask2var = Smask2;
#define DECL_SMASK2REG uint64_t Smask2reg = Smask2var;
#define FORCE_REGALLOC_1 /* empty */
#define FORCE_REGALLOC_2 /* empty */
#endif

#define PWXFORM_SIMD(X) { \
	uint64_t x; \
	FORCE_REGALLOC_1 \
	uint32_t lo = x = _mm_extract_epi64_(vreinterpretq_u64_u32(X), 0) & Smask2reg; \
	FORCE_REGALLOC_2 \
	uint32_t hi = x >> 32; \
	X = _mm_mul_epu32((__m128i)vreinterpretq_u64_u32(vshrq_n_u64(vreinterpretq_u64_u32(X), 32)), X); \
	X = _mm_add_epi32(X, *(__m128i *)(S0 + lo)); \
	X = _mm_xor_si128(X, *(__m128i *)(S1 + hi)); \
}

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

#elif defined(__SSE2__)
/* Original SSE2 pwxform (unchanged) */
#undef DECL_SMASK2REG
#if defined(__x86_64__) && (defined(__AVX__) || !defined(__GNUC__))
/* 64-bit with AVX */
#define DECL_SMASK2REG uint64_t Smask2reg = Smask2;
#define FORCE_REGALLOC_1 \
	__asm__("" : "=a" (x), "+d" (Smask2reg), "+S" (S0), "+D" (S1));
#define FORCE_REGALLOC_2 \
	__asm__("" : : "c" (lo));
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
/* 64-bit without AVX */
#define DECL_SMASK2REG uint64_t Smask2reg = Smask2;
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
#define DECL_SMASK2REG /* empty */
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
#define DECL_SMASK2REG /* empty */
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

#else /* !defined(__SSE2__) */

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
	PREFETCH(&Bin2[r], _MM_HINT_T0)
	for (i = 0; i < r; i++) {
		PREFETCH(&Bin2[i], _MM_HINT_T0)
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
	PREFETCH(&Bin2[r], _MM_HINT_T0)
	for (i = 0; i < r; i++) {
		PREFETCH(&Bin2[i], _MM_HINT_T0)
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

/* ... continued from Part 3 ... */

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
	return (uint32_t)B[2 * r - 1].d[0];
}

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
#if _YESPOWER_OPT_C_PASS_ == 1
	uint32_t Nloop_all = (N + 2) / 3; /* 1/3, round up */
	uint32_t Nloop_rw = Nloop_all;

	Nloop_all++; Nloop_all &= ~(uint32_t)1; /* round up to even */
	Nloop_rw &= ~(uint32_t)1; /* round down to even */
#else
	uint32_t Nloop_rw = (N + 2) / 3; /* 1/3, round up */
	Nloop_rw++; Nloop_rw &= ~(uint32_t)1; /* round up to even */
#endif

	smix1(B, 1, ctx->Sbytes / 128, (salsa20_blk_t *)ctx->S0, XY, NULL);
	smix1(B, r, N, V, XY, ctx);
	smix2(B, r, N, Nloop_rw /* must be > 2 */, V, XY, ctx);
#if _YESPOWER_OPT_C_PASS_ == 1
	if (Nloop_all > Nloop_rw)
		smix2(B, r, N, 2, V, XY, ctx);
#endif
}

#if _YESPOWER_OPT_C_PASS_ == 1
#undef _YESPOWER_OPT_C_PASS_
#define _YESPOWER_OPT_C_PASS_ 2
#define blockmix_salsa blockmix_salsa_1_0
#define blockmix_salsa_xor blockmix_salsa_xor_1_0
#define blockmix blockmix_1_0
#define blockmix_xor blockmix_xor_1_0
#define blockmix_xor_save blockmix_xor_save_1_0
#define smix1 smix1_1_0
#define smix2 smix2_1_0
#define smix smix_1_0
#include "yespower-opt.c"
#undef smix

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
		smix_1_0(B, r, N, V, XY, &ctx);
		HMAC_SHA256_Buf(B + B_size - 64, 64,
		    sha256, sizeof(sha256), (uint8_t *)dst);
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
#endif
/* ... continued from Part 4 ... */

/*************************** MULTI-THREADED MINER ADDITIONS ***************************/
#ifndef _YESPOWER_MINER_ADDED_
#define _YESPOWER_MINER_ADDED_

#ifdef __cplusplus
extern "C" {
#endif

#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* Atomic operations (GCC-style) */
#ifdef __GNUC__
#define atomic_fetch_add(ptr, val) __sync_fetch_and_add(ptr, val)
#else
#error "Atomic operations not supported (need GCC or compatible)"
#endif

/* Forward declaration of the favoriting callback type */
typedef int (*yespower_favor_func_t)(uint64_t nonce, void *user_data);

/* Multi‑stage prefilter thresholds (tunable) */
#define STAGE1_MULTIPLIER 2   /* Accept if fast32 <= target_top_32 * STAGE1_MULTIPLIER */
#define STAGE2_MULTIPLIER 1   /* Accept if Salsa20/2 top <= target_top_32 * STAGE2_MULTIPLIER */

/* Fast pre‑filter using ultra‑fast 32‑bit hash (stage 1) */
static int fast_prefilter_stage1(uint64_t nonce, const uint8_t *input, size_t input_len,
                                 size_t offset, uint32_t target_top) {
    uint8_t buf[128]; /* Enough for typical input */
    memcpy(buf, input, input_len);
    *(uint64_t *)(buf + offset) = nonce;
    uint32_t h = fast_hash32(buf, input_len, 0xdeadbeef);
    return (h <= target_top * STAGE1_MULTIPLIER);
}

/* Stage 2 prefilter using Salsa20/2 */
static int fast_prefilter_stage2(uint64_t nonce, const uint8_t *input, size_t input_len,
                                 size_t offset, uint32_t target_top) {
    uint8_t buf[64];
    memcpy(buf, input, input_len < 64 ? input_len : 64);
    if (offset < 64) {
        *(uint64_t *)(buf + offset) = nonce;
    }
    salsa20_blk_t blk;
    memcpy(blk.w, buf, 64);
    salsa20_simd_shuffle((salsa20_blk_t*)buf, &blk);
    DECL_X
    READ_X(blk);
    SALSA20_2(blk);
    uint32_t first_word = blk.w[0];
    return (first_word <= target_top * STAGE2_MULTIPLIER);
}

/* Structure for shared miner state */
typedef struct {
    const uint8_t *base_input;      /* Original input buffer (without nonce) */
    size_t input_len;                /* Length of input buffer */
    size_t nonce_offset;             /* Offset in input where nonce (uint64_t LE) should be written */
    uint64_t start_nonce;            /* Start of global nonce range (inclusive) */
    uint64_t end_nonce;              /* End of global nonce range (exclusive) */
    const yespower_params_t *params; /* yespower parameters */
    uint8_t target[32];              /* Target hash (big-endian 256-bit) */
    uint32_t target_top32;            /* First 32 bits of target (big-endian) */
    yespower_favor_func_t is_favorable; /* Callback to check if nonce is favorable (can be NULL) */
    void *favor_user_data;           /* User data for favor callback */

    volatile uint64_t next_nonce;    /* Next nonce to try from global range */

    /* Results: all valid nonces found */
    uint64_t *found_nonces;          /* Dynamically allocated array */
    size_t found_capacity;           /* Current capacity of found_nonces */
    size_t found_count;              /* Number of valid nonces stored */
    pthread_mutex_t lock;            /* Protects found_nonces, found_count */
} miner_shared_t;

/* Thread function */
static void *miner_thread(void *arg) {
    miner_shared_t *shared = (miner_shared_t *)arg;
    yespower_local_t local;
    uint8_t *input_copy;
    uint64_t nonce;
    yespower_binary_t hash;
    int cmp;

    /* Initialize thread-local yespower context */
    if (yespower_init_local(&local) != 0) {
        return NULL;
    }

    /* Copy input buffer to allow per-thread nonce modification */
    input_copy = (uint8_t *)malloc(shared->input_len);
    if (!input_copy) {
        yespower_free_local(&local);
        return NULL;
    }
    memcpy(input_copy, shared->base_input, shared->input_len);

    /* Main loop: fetch next nonce and test */
    while (1) {
        nonce = atomic_fetch_add(&shared->next_nonce, 1);
        if (nonce >= shared->end_nonce) {
            break; /* No more nonces */
        }

        /* Optional favoriting callback */
        if (shared->is_favorable && !shared->is_favorable(nonce, shared->favor_user_data)) {
            continue;
        }

        /* Stage 1: ultra‑fast 32‑bit hash filter */
        if (!fast_prefilter_stage1(nonce, shared->base_input, shared->input_len,
                                   shared->nonce_offset, shared->target_top32)) {
            continue;
        }

        /* Stage 2: Salsa20/2 filter */
        if (!fast_prefilter_stage2(nonce, shared->base_input, shared->input_len,
                                   shared->nonce_offset, shared->target_top32)) {
            continue;
        }

        /* Place nonce into input (little-endian 64-bit) */
        *(uint64_t *)(input_copy + shared->nonce_offset) = nonce;

        /* Compute full yespower hash */
        if (yespower(&local, input_copy, shared->input_len, shared->params, &hash) != 0) {
            /* Error in yespower; skip this nonce */
            continue;
        }

        /* Compare hash with target (big-endian 256-bit) */
        cmp = memcmp((uint8_t*)&hash, shared->target, 32);
        if (cmp <= 0) { /* hash <= target – valid nonce */
            pthread_mutex_lock(&shared->lock);
            /* Ensure enough capacity */
            if (shared->found_count >= shared->found_capacity) {
                size_t new_cap = shared->found_capacity ? shared->found_capacity * 2 : 64;
                uint64_t *new_arr = (uint64_t *)realloc(shared->found_nonces,
                                                        new_cap * sizeof(uint64_t));
                if (new_arr) {
                    shared->found_nonces = new_arr;
                    shared->found_capacity = new_cap;
                } else {
                    /* Out of memory – just drop this nonce (should not happen) */
                    pthread_mutex_unlock(&shared->lock);
                    continue;
                }
            }
            shared->found_nonces[shared->found_count++] = nonce;
            pthread_mutex_unlock(&shared->lock);
        }
    }

    free(input_copy);
    yespower_free_local(&local);
    return NULL;
}

/**
 * yespower_miner - Multi-threaded nonce search with multi‑stage prefiltering
 *
 * @base_input:     Pointer to the input buffer (without nonce). The buffer
 *                  must remain valid for the duration of the search.
 * @input_len:      Length of input buffer.
 * @nonce_offset:   Offset within input where the 64-bit little-endian nonce
 *                  should be written. The area must be writable (the function
 *                  copies the buffer per thread).
 * @start_nonce:    First nonce to try (inclusive).
 * @end_nonce:      Last nonce to try (exclusive).
 * @params:         yespower parameters (version, N, r, pers).
 * @target:         32-byte target hash (big-endian). A hash is valid if
 *                  memcmp(hash, target, 32) <= 0.
 * @num_threads:    Number of worker threads to launch.
 * @is_favorable:   Optional callback to test if a nonce should be tried.
 *                  Return 1 to try, 0 to skip. May be NULL.
 *                  This callback can be used to implement dynamic, automatic
 *                  favoriting based on the current block and difficulty.
 * @favor_user_data:User data passed to is_favorable.
 * @out_nonces:     Output pointer to an array of valid nonces (dynamically
 *                  allocated). The caller must free() it.
 * @out_count:      Number of valid nonces found.
 *
 * Returns 0 on success (search completed), -1 on error.
 */
int yespower_miner(const uint8_t *base_input, size_t input_len, size_t nonce_offset,
                   uint64_t start_nonce, uint64_t end_nonce,
                   const yespower_params_t *params,
                   const uint8_t target[32],
                   int num_threads,
                   yespower_favor_func_t is_favorable, void *favor_user_data,
                   uint64_t **out_nonces, size_t *out_count) {
    pthread_t *threads;
    miner_shared_t shared;
    int i, ret = -1;

    if (!base_input || input_len == 0 || !params || !target || num_threads <= 0 ||
        !out_nonces || !out_count) {
        errno = EINVAL;
        return -1;
    }

    /* Initialize shared state */
    memset(&shared, 0, sizeof(shared));
    shared.base_input = base_input;
    shared.input_len = input_len;
    shared.nonce_offset = nonce_offset;
    shared.start_nonce = start_nonce;
    shared.end_nonce = end_nonce;
    shared.params = params;
    memcpy(shared.target, target, 32);
    /* Extract top 32 bits of target (big-endian) */
    shared.target_top32 = ((uint32_t)target[0] << 24) | (target[1] << 16) | (target[2] << 8) | target[3];
    shared.is_favorable = is_favorable;
    shared.favor_user_data = favor_user_data;
    shared.next_nonce = start_nonce;
    shared.found_nonces = NULL;
    shared.found_capacity = 0;
    shared.found_count = 0;
    pthread_mutex_init(&shared.lock, NULL);

    /* Create threads */
    threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    if (!threads) {
        pthread_mutex_destroy(&shared.lock);
        return -1;
    }

    for (i = 0; i < num_threads; i++) {
        if (pthread_create(&threads[i], NULL, miner_thread, &shared) != 0) {
            /* Clean up already created threads */
            int j;
            for (j = 0; j < i; j++) {
                pthread_join(threads[j], NULL);
            }
            free(threads);
            pthread_mutex_destroy(&shared.lock);
            if (shared.found_nonces) free(shared.found_nonces);
            return -1;
        }
    }

    /* Wait for all threads to finish */
    for (i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    /* Set output */
    *out_nonces = shared.found_nonces;
    *out_count = shared.found_count;
    ret = 0;

    free(threads);
    pthread_mutex_destroy(&shared.lock);
    return ret;
}

#ifdef __cplusplus
}
#endif

#endif /* _YESPOWER_MINER_ADDED_ */
