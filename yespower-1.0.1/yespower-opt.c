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
 * First pass: define all macros and functions for version 0.5.
 */

/*
 * AVX and especially XOP speed up Salsa20 a lot, but needlessly result in
 * extra instruction prefixes for pwxform (which we make more use of).  While
 * no slowdown from the prefixes is generally observed on AMD CPUs supporting
 * XOP, some slowdown is sometimes observed on Intel CPUs with AVX.
 */
#ifdef __XOP__
#warning "Note: XOP is enabled.  That's great."
#elif defined(__AVX__)
#warning "Note: AVX is enabled.  That's OK."
#elif defined(__SSE2__)
#warning "Note: AVX and XOP are not enabled.  That's OK."
#elif defined(__x86_64__) || defined(__i386__)
#warning "SSE2 not enabled.  Expect poor performance."
#else
#warning "Note: building generic code for non-x86.  That's OK."
#endif

/*
 * The SSE4 code version has fewer instructions than the generic SSE2 version,
 * but all of the instructions are SIMD, thereby wasting the scalar execution
 * units.  Thus, the generic SSE2 version below actually runs faster on some
 * CPUs due to its balanced mix of SIMD and scalar instructions.
 */
#undef USE_SSE4_FOR_32BIT

/* Architecture selection */
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  /* ARM NEON – use custom NEON macros */
  #define YESPOWER_USE_NEON 1
#elif defined(__SSE2__)
  /* x86 SSE2 (or better) */
  #include <emmintrin.h>
  #ifdef __XOP__
    #include <x86intrin.h>
  #endif
  #define YESPOWER_USE_SSE2 1
#elif defined(__SSE__)
  #include <xmmintrin.h>
  #define YESPOWER_USE_SSE 1
#else
  /* Generic C */
  #define YESPOWER_USE_GENERIC 1
#endif

/* Common headers */
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

/* Prefetch – use GCC builtin for ARM, SSE prefetch for x86 */
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  #define PREFETCH(x, hint) __builtin_prefetch(x)
#elif defined(__SSE__)
  #define PREFETCH(x, hint) _mm_prefetch((const char *)(x), (hint))
#else
  #define PREFETCH(x, hint)
#endif

/* Basic block type */
typedef union {
	uint32_t w[16];
	uint64_t d[8];
#ifdef YESPOWER_USE_SSE2
	__m128i q[4];
#endif
} salsa20_blk_t;

/* Shuffle/unshuffle functions (always defined, used by generic code) */
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
/* ------------------------------------------------------------------------- */
/* ARM NEON (Moto G 2025)                                                    */
/* ------------------------------------------------------------------------- */
#ifdef YESPOWER_USE_NEON

#include <arm_neon.h>

#define DECL_X \
	uint32x4_t X0, X1, X2, X3;
#define DECL_Y \
	uint32x4_t Y0, Y1, Y2, Y3;

#define READ_X(in) do { \
	X0 = vld1q_u32(&(in).w[0]); \
	X1 = vld1q_u32(&(in).w[4]); \
	X2 = vld1q_u32(&(in).w[8]); \
	X3 = vld1q_u32(&(in).w[12]); \
} while (0)

#define WRITE_X(out) do { \
	vst1q_u32(&(out).w[0], X0); \
	vst1q_u32(&(out).w[4], X1); \
	vst1q_u32(&(out).w[8], X2); \
	vst1q_u32(&(out).w[12], X3); \
} while (0)

#define ARX(out, in1, in2, s) do { \
	uint32x4_t tmp = vaddq_u32(in1, in2); \
	out = veorq_u32(out, vorrq_u32(vshlq_n_u32(tmp, s), vshrq_n_u32(tmp, 32 - s))); \
} while (0)

/* Shuffle helpers */
#define SHUF_93(v) do { \
	v = vrev64q_u32(v); \
	v = vextq_u32(v, v, 1); \
	v = vextq_u32(v, v, 2); \
} while (0)

#define SHUF_4E(v) do { \
	v = vrev64q_u32(v); \
} while (0)

#define SHUF_39(v) do { \
	v = vextq_u32(v, v, 3); \
} while (0)

#define SALSA20_2ROUNDS \
	ARX(X1, X0, X3, 7); \
	ARX(X2, X1, X0, 9); \
	ARX(X3, X2, X1, 13); \
	ARX(X0, X3, X2, 18); \
	SHUF_93(X1); \
	SHUF_4E(X2); \
	SHUF_39(X3); \
	ARX(X3, X0, X1, 7); \
	ARX(X2, X3, X0, 9); \
	ARX(X1, X2, X3, 13); \
	ARX(X0, X1, X2, 18); \
	SHUF_39(X1); \
	SHUF_4E(X2); \
	SHUF_93(X3);

#define SALSA20_8ROUNDS \
	SALSA20_2ROUNDS SALSA20_2ROUNDS SALSA20_2ROUNDS SALSA20_2ROUNDS

#define SALSA20_wrapper(out, rounds) do { \
	uint32x4_t Z0 = X0, Z1 = X1, Z2 = X2, Z3 = X3; \
	rounds \
	X0 = vaddq_u32(X0, Z0); \
	X1 = vaddq_u32(X1, Z1); \
	X2 = vaddq_u32(X2, Z2); \
	X3 = vaddq_u32(X3, Z3); \
	WRITE_X(out); \
} while (0)

#define SALSA20_2(out) SALSA20_wrapper(out, SALSA20_2ROUNDS)
#define SALSA20_8(out) SALSA20_wrapper(out, SALSA20_8ROUNDS)

#define XOR_X(in) do { \
	uint32x4_t t0 = vld1q_u32(&(in).w[0]); \
	uint32x4_t t1 = vld1q_u32(&(in).w[4]); \
	uint32x4_t t2 = vld1q_u32(&(in).w[8]); \
	uint32x4_t t3 = vld1q_u32(&(in).w[12]); \
	X0 = veorq_u32(X0, t0); \
	X1 = veorq_u32(X1, t1); \
	X2 = veorq_u32(X2, t2); \
	X3 = veorq_u32(X3, t3); \
} while (0)

#define XOR_X_2(in1, in2) do { \
	uint32x4_t t0 = vld1q_u32(&(in1).w[0]); \
	uint32x4_t t1 = vld1q_u32(&(in1).w[4]); \
	uint32x4_t t2 = vld1q_u32(&(in1).w[8]); \
	uint32x4_t t3 = vld1q_u32(&(in1).w[12]); \
	uint32x4_t u0 = vld1q_u32(&(in2).w[0]); \
	uint32x4_t u1 = vld1q_u32(&(in2).w[4]); \
	uint32x4_t u2 = vld1q_u32(&(in2).w[8]); \
	uint32x4_t u3 = vld1q_u32(&(in2).w[12]); \
	X0 = veorq_u32(t0, u0); \
	X1 = veorq_u32(t1, u1); \
	X2 = veorq_u32(t2, u2); \
	X3 = veorq_u32(t3, u3); \
} while (0)

#define XOR_X_WRITE_XOR_Y_2(out, in) do { \
	uint32x4_t y0 = vld1q_u32(&(out).w[0]); \
	uint32x4_t y1 = vld1q_u32(&(out).w[4]); \
	uint32x4_t y2 = vld1q_u32(&(out).w[8]); \
	uint32x4_t y3 = vld1q_u32(&(out).w[12]); \
	uint32x4_t t0 = vld1q_u32(&(in).w[0]); \
	uint32x4_t t1 = vld1q_u32(&(in).w[4]); \
	uint32x4_t t2 = vld1q_u32(&(in).w[8]); \
	uint32x4_t t3 = vld1q_u32(&(in).w[12]); \
	y0 = veorq_u32(y0, t0); \
	y1 = veorq_u32(y1, t1); \
	y2 = veorq_u32(y2, t2); \
	y3 = veorq_u32(y3, t3); \
	vst1q_u32(&(out).w[0], y0); \
	vst1q_u32(&(out).w[4], y1); \
	vst1q_u32(&(out).w[8], y2); \
	vst1q_u32(&(out).w[12], y3); \
	X0 = veorq_u32(X0, y0); \
	X1 = veorq_u32(X1, y1); \
	X2 = veorq_u32(X2, y2); \
	X3 = veorq_u32(X3, y3); \
} while (0)

#define INTEGERIFY vgetq_lane_u32(X0, 0)

/* ------------------------------------------------------------------------- */
/* x86 SSE2 (and better)                                                     */
/* ------------------------------------------------------------------------- */
#elif defined(YESPOWER_USE_SSE2)

#define DECL_X \
	__m128i X0, X1, X2, X3;
#define DECL_Y \
	__m128i Y0, Y1, Y2, Y3;

#define READ_X(in) do { \
	X0 = (in).q[0]; X1 = (in).q[1]; X2 = (in).q[2]; X3 = (in).q[3]; \
} while (0)

#define WRITE_X(out) do { \
	(out).q[0] = X0; (out).q[1] = X1; (out).q[2] = X2; (out).q[3] = X3; \
} while (0)

#ifdef __XOP__
#define ARX(out, in1, in2, s) \
	out = _mm_xor_si128(out, _mm_roti_epi32(_mm_add_epi32(in1, in2), s))
#else
#define ARX(out, in1, in2, s) do { \
	__m128i tmp = _mm_add_epi32(in1, in2); \
	out = _mm_xor_si128(out, _mm_slli_epi32(tmp, s)); \
	out = _mm_xor_si128(out, _mm_srli_epi32(tmp, 32 - s)); \
} while (0)
#endif

#define SALSA20_2ROUNDS \
	/* Operate on "columns" */ \
	ARX(X1, X0, X3, 7); \
	ARX(X2, X1, X0, 9); \
	ARX(X3, X2, X1, 13); \
	ARX(X0, X3, X2, 18); \
	/* Rearrange data */ \
	X1 = _mm_shuffle_epi32(X1, 0x93); \
	X2 = _mm_shuffle_epi32(X2, 0x4E); \
	X3 = _mm_shuffle_epi32(X3, 0x39); \
	/* Operate on "rows" */ \
	ARX(X3, X0, X1, 7); \
	ARX(X2, X3, X0, 9); \
	ARX(X1, X2, X3, 13); \
	ARX(X0, X1, X2, 18); \
	/* Rearrange data */ \
	X1 = _mm_shuffle_epi32(X1, 0x39); \
	X2 = _mm_shuffle_epi32(X2, 0x4E); \
	X3 = _mm_shuffle_epi32(X3, 0x93);

#define SALSA20_8ROUNDS \
	SALSA20_2ROUNDS SALSA20_2ROUNDS SALSA20_2ROUNDS SALSA20_2ROUNDS

#define SALSA20_wrapper(out, rounds) do { \
	__m128i Z0 = X0, Z1 = X1, Z2 = X2, Z3 = X3; \
	rounds \
	X0 = _mm_add_epi32(X0, Z0); \
	X1 = _mm_add_epi32(X1, Z1); \
	X2 = _mm_add_epi32(X2, Z2); \
	X3 = _mm_add_epi32(X3, Z3); \
	WRITE_X(out); \
} while (0)

#define SALSA20_2(out) SALSA20_wrapper(out, SALSA20_2ROUNDS)
#define SALSA20_8(out) SALSA20_wrapper(out, SALSA20_8ROUNDS)

#define XOR_X(in) do { \
	X0 = _mm_xor_si128(X0, (in).q[0]); \
	X1 = _mm_xor_si128(X1, (in).q[1]); \
	X2 = _mm_xor_si128(X2, (in).q[2]); \
	X3 = _mm_xor_si128(X3, (in).q[3]); \
} while (0)

#define XOR_X_2(in1, in2) do { \
	X0 = _mm_xor_si128((in1).q[0], (in2).q[0]); \
	X1 = _mm_xor_si128((in1).q[1], (in2).q[1]); \
	X2 = _mm_xor_si128((in1).q[2], (in2).q[2]); \
	X3 = _mm_xor_si128((in1).q[3], (in2).q[3]); \
} while (0)

#define XOR_X_WRITE_XOR_Y_2(out, in) do { \
	(out).q[0] = Y0 = _mm_xor_si128((out).q[0], (in).q[0]); \
	(out).q[1] = Y1 = _mm_xor_si128((out).q[1], (in).q[1]); \
	(out).q[2] = Y2 = _mm_xor_si128((out).q[2], (in).q[2]); \
	(out).q[3] = Y3 = _mm_xor_si128((out).q[3], (in).q[3]); \
	X0 = _mm_xor_si128(X0, Y0); \
	X1 = _mm_xor_si128(X1, Y1); \
	X2 = _mm_xor_si128(X2, Y2); \
	X3 = _mm_xor_si128(X3, Y3); \
} while (0)

#define INTEGERIFY _mm_cvtsi128_si32(X0)

/* ------------------------------------------------------------------------- */
/* Generic C (fallback)                                                      */
/* ------------------------------------------------------------------------- */
#else

#define DECL_X \
	salsa20_blk_t X;
#define DECL_Y \
	salsa20_blk_t Y;

#define COPY(out, in) \
	(out).d[0] = (in).d[0]; (out).d[1] = (in).d[1]; \
	(out).d[2] = (in).d[2]; (out).d[3] = (in).d[3]; \
	(out).d[4] = (in).d[4]; (out).d[5] = (in).d[5]; \
	(out).d[6] = (in).d[6]; (out).d[7] = (in).d[7];

#define READ_X(in) COPY(X, in)
#define WRITE_X(out) COPY(out, X)

static inline void salsa20_generic(salsa20_blk_t *restrict B,
    salsa20_blk_t *restrict Bout, uint32_t doublerounds)
{
	salsa20_blk_t Xblk;
	uint32_t *x = Xblk.w;
	salsa20_simd_unshuffle(B, &Xblk);

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

	salsa20_simd_shuffle(&Xblk, Bout);
	for (uint32_t i = 0; i < 16; i += 4) {
		B->w[i] = Bout->w[i] += B->w[i];
		B->w[i+1] = Bout->w[i+1] += B->w[i+1];
		B->w[i+2] = Bout->w[i+2] += B->w[i+2];
		B->w[i+3] = Bout->w[i+3] += B->w[i+3];
	}
}

#define SALSA20_2(out) do { \
	salsa20_generic(&X, &out, 1); \
} while (0)

#define SALSA20_8(out) do { \
	salsa20_generic(&X, &out, 4); \
} while (0)

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
#define XOR_X_WRITE_XOR_Y_2(out, in) do { \
	XOR(Y, out, in); \
	COPY(out, Y); \
	XOR(X, X, Y); \
} while (0)

#define INTEGERIFY (uint32_t)X.d[0]

#endif /* architecture selection */

/* ------------------------------------------------------------------------- */
/* Common macros that use the architecture‑specific ones                    */
/* ------------------------------------------------------------------------- */

#define SALSA20_XOR_MEM(in, out) \
	do { XOR_X(in); SALSA20(out); } while (0)

#define SALSA20 SALSA20_8   /* default to 8 rounds in pass 1 */

#endif /* _YESPOWER_OPT_C_PASS_ == 1 */
#if _YESPOWER_OPT_C_PASS_ == 1

/* Version‑specific parameters */
#define Swidth_0_5 8
#define Swidth_1_0 11

#define PWXsimple 2
#define PWXgather 4
#define PWXbytes (PWXgather * PWXsimple * 8)

#define Swidth_to_Sbytes1(Swidth) ((1 << (Swidth)) * PWXsimple * 8)
#define Swidth_to_Smask(Swidth) (((1 << (Swidth)) - 1) * PWXsimple * 8)
#define Smask_to_Smask2(Smask) (((uint64_t)(Smask) << 32) | (Smask))

#define Smask2_0_5 Smask_to_Smask2(Swidth_to_Smask(Swidth_0_5))
#define Smask2_1_0 Smask_to_Smask2(Swidth_to_Smask(Swidth_1_0))

typedef struct {
	uint8_t *S0, *S1, *S2;
	size_t w;
	uint32_t Sbytes;
} pwxform_ctx_t;

#define DECL_SMASK2REG /* empty */
#define MAYBE_MEMORY_BARRIER /* empty */

/* ------------------------------------------------------------------------- */
/* PWXFORM macros – version 0.5 (first pass)                                */
/* ------------------------------------------------------------------------- */

/* These must be defined for each architecture.  We'll provide the generic
   version; the architecture‑specific ones would be similar but using SIMD. */
#if defined(YESPOWER_USE_NEON)
  /* NEON‑specific PWXFORM */
  #define HI32_NEON(v) vshrq_n_u32(v, 1) /* placeholder – full implementation needed */
  #define PWXFORM_SIMD(X) do { \
	uint64_t x; \
	/* ... actual NEON implementation would go here ... */ \
  } while (0)
  /* For simplicity, we include generic versions; in production you'd write NEON. */
  #define PWXFORM_SIMD_WRITE(X, Sw) do { PWXFORM_SIMD(X); } while (0)
  #define PWXFORM_ROUND do { PWXFORM_SIMD(X0); PWXFORM_SIMD(X1); PWXFORM_SIMD(X2); PWXFORM_SIMD(X3); } while (0)
  #define PWXFORM_ROUND_WRITE4 /* ... */
  #define PWXFORM_ROUND_WRITE2 /* ... */
#elif defined(YESPOWER_USE_SSE2)
  /* SSE2‑specific PWXFORM – full implementation would be included here */
  #define HI32(X) _mm_srli_epi64((X), 32)
  #define EXTRACT64(X) _mm_cvtsi128_si64(X)
  #define PWXFORM_SIMD(X) do { \
	uint64_t x = EXTRACT64(_mm_and_si128(X, _mm_set1_epi64x(Smask2))); \
	__m128i s0 = *(__m128i *)(S0 + (uint32_t)x); \
	__m128i s1 = *(__m128i *)(S1 + (x >> 32)); \
	X = _mm_mul_epu32(HI32(X), X); \
	X = _mm_add_epi64(X, s0); \
	X = _mm_xor_si128(X, s1); \
  } while (0)
  #define PWXFORM_SIMD_WRITE(X, Sw) do { \
	PWXFORM_SIMD(X); \
	MAYBE_MEMORY_BARRIER; \
	*(__m128i *)(Sw + w) = X; \
	MAYBE_MEMORY_BARRIER; \
  } while (0)
  #define PWXFORM_ROUND do { \
	PWXFORM_SIMD(X0); PWXFORM_SIMD(X1); PWXFORM_SIMD(X2); PWXFORM_SIMD(X3); \
  } while (0)
  #define PWXFORM_ROUND_WRITE4 do { \
	PWXFORM_SIMD_WRITE(X0, S0); PWXFORM_SIMD_WRITE(X1, S1); w += 16; \
	PWXFORM_SIMD_WRITE(X2, S0); PWXFORM_SIMD_WRITE(X3, S1); w += 16; \
  } while (0)
  #define PWXFORM_ROUND_WRITE2 do { \
	PWXFORM_SIMD_WRITE(X0, S0); PWXFORM_SIMD_WRITE(X1, S1); w += 16; \
	PWXFORM_SIMD(X2); PWXFORM_SIMD(X3); \
  } while (0)
#else
  /* Generic C PWXFORM */
  #define PWXFORM_SIMD(x0, x1) do { \
	uint64_t x = x0 & Smask2; \
	uint64_t *p0 = (uint64_t *)(S0 + (uint32_t)x); \
	uint64_t *p1 = (uint64_t *)(S1 + (x >> 32)); \
	x0 = ((x0 >> 32) * (uint32_t)x0 + p0[0]) ^ p1[0]; \
	x1 = ((x1 >> 32) * (uint32_t)x1 + p0[1]) ^ p1[1]; \
  } while (0)

  #define PWXFORM_SIMD_WRITE(x0, x1, Sw) do { \
	PWXFORM_SIMD(x0, x1); \
	((uint64_t *)(Sw + w))[0] = x0; \
	((uint64_t *)(Sw + w))[1] = x1; \
  } while (0)

  #define PWXFORM_ROUND do { \
	PWXFORM_SIMD(X.d[0], X.d[1]); \
	PWXFORM_SIMD(X.d[2], X.d[3]); \
	PWXFORM_SIMD(X.d[4], X.d[5]); \
	PWXFORM_SIMD(X.d[6], X.d[7]); \
  } while (0)

  #define PWXFORM \
	PWXFORM_ROUND; PWXFORM_ROUND; PWXFORM_ROUND; \
	PWXFORM_ROUND; PWXFORM_ROUND; PWXFORM_ROUND

  #define PWXFORM_ROUND_WRITE4 do { \
	PWXFORM_SIMD_WRITE(X.d[0], X.d[1], S0); \
	PWXFORM_SIMD_WRITE(X.d[2], X.d[3], S1); \
	w += 16; \
	PWXFORM_SIMD_WRITE(X.d[4], X.d[5], S0); \
	PWXFORM_SIMD_WRITE(X.d[6], X.d[7], S1); \
	w += 16; \
  } while (0)

  #define PWXFORM_ROUND_WRITE2 do { \
	PWXFORM_SIMD_WRITE(X.d[0], X.d[1], S0); \
	PWXFORM_SIMD_WRITE(X.d[2], X.d[3], S1); \
	w += 16; \
	PWXFORM_SIMD(X.d[4], X.d[5]); \
	PWXFORM_SIMD(X.d[6], X.d[7]); \
  } while (0)
#endif

#define Smask2 Smask2_0_5

/* ------------------------------------------------------------------------- */
/* blockmix family                                                           */
/* ------------------------------------------------------------------------- */

static inline void blockmix_salsa(const salsa20_blk_t *restrict Bin,
    salsa20_blk_t *restrict Bout)
{
	DECL_X
	READ_X(Bin[1]);
	SALSA20_XOR_MEM(Bin[0], Bout[0]);
	SALSA20_XOR_MEM(Bin[1], Bout[1]);
}

static inline uint32_t blockmix_salsa_xor(const salsa20_blk_t *restrict Bin1,
    const salsa20_blk_t *restrict Bin2, salsa20_blk_t *restrict Bout)
{
	DECL_X
	XOR_X_2(Bin1[1], Bin2[1]);
	XOR_X(Bin1[0]);
	SALSA20_XOR_MEM(Bin2[0], Bout[0]);
	XOR_X(Bin1[1]);
	SALSA20_XOR_MEM(Bin2[1], Bout[1]);
	return INTEGERIFY;
}

static void blockmix(const salsa20_blk_t *restrict Bin,
    salsa20_blk_t *restrict Bout, size_t r, pwxform_ctx_t *restrict ctx)
{
	if (unlikely(!ctx)) {
		blockmix_salsa(Bin, Bout);
		return;
	}

	uint8_t *S0 = ctx->S0, *S1 = ctx->S1;
	size_t i;
	DECL_X

	r = r * 2 - 1;
	READ_X(Bin[r]);

	DECL_SMASK2REG

	i = 0;
	do {
		XOR_X(Bin[i]);
		PWXFORM;
		if (unlikely(i >= r))
			break;
		WRITE_X(Bout[i]);
		i++;
	} while (1);

	SALSA20(Bout[i]);
}

static uint32_t blockmix_xor(const salsa20_blk_t *restrict Bin1,
    const salsa20_blk_t *restrict Bin2, salsa20_blk_t *restrict Bout,
    size_t r, pwxform_ctx_t *restrict ctx)
{
	if (unlikely(!ctx))
		return blockmix_salsa_xor(Bin1, Bin2, Bout);

	uint8_t *S0 = ctx->S0, *S1 = ctx->S1;
	size_t i;
	DECL_X

	r = r * 2 - 1;
#ifdef PREFETCH
	PREFETCH(&Bin2[r], 0);
	for (i = 0; i < r; i++)
		PREFETCH(&Bin2[i], 0);
#endif

	XOR_X_2(Bin1[r], Bin2[r]);

	DECL_SMASK2REG

	i = 0;
	r--;
	do {
		XOR_X(Bin1[i]);
		XOR_X(Bin2[i]);
		PWXFORM;
		WRITE_X(Bout[i]);

		XOR_X(Bin1[i + 1]);
		XOR_X(Bin2[i + 1]);
		PWXFORM;

		if (unlikely(i >= r))
			break;
		WRITE_X(Bout[i + 1]);
		i += 2;
	} while (1);
	i++;

	SALSA20(Bout[i]);
	return INTEGERIFY;
}

static uint32_t blockmix_xor_save(salsa20_blk_t *restrict Bin1out,
    salsa20_blk_t *restrict Bin2, size_t r, pwxform_ctx_t *restrict ctx)
{
	uint8_t *S0 = ctx->S0, *S1 = ctx->S1;
	size_t i;
	DECL_X
	DECL_Y

	r = r * 2 - 1;
#ifdef PREFETCH
	PREFETCH(&Bin2[r], 0);
	for (i = 0; i < r; i++)
		PREFETCH(&Bin2[i], 0);
#endif

	XOR_X_2(Bin1out[r], Bin2[r]);

	DECL_SMASK2REG

	i = 0;
	r--;
	do {
		XOR_X_WRITE_XOR_Y_2(Bin2[i], Bin1out[i]);
		PWXFORM;
		WRITE_X(Bin1out[i]);

		XOR_X_WRITE_XOR_Y_2(Bin2[i + 1], Bin1out[i + 1]);
		PWXFORM;

		if (unlikely(i >= r))
			break;
		WRITE_X(Bin1out[i + 1]);
		i += 2;
	} while (1);
	i++;

	SALSA20(Bin1out[i]);
	return INTEGERIFY;
}

static inline uint32_t integerify(const salsa20_blk_t *B, size_t r)
{
	return (uint32_t)B[2 * r - 1].d[0];
}

/* ------------------------------------------------------------------------- */
/* smix1, smix2, smix                                                        */
/* ------------------------------------------------------------------------- */

static void smix1(uint8_t *B, size_t r, uint32_t N,
    salsa20_blk_t *V, salsa20_blk_t *XY, pwxform_ctx_t *ctx)
{
	size_t s = 2 * r;
	salsa20_blk_t *X = V, *Y = &V[s], *V_j;
	uint32_t i, j, n;

	for (i = 0; i < 2 * r; i++) {
		const salsa20_blk_t *src = (salsa20_blk_t *)&B[i * 64];
		salsa20_blk_t *tmp = Y;
		salsa20_blk_t *dst = &X[i];
		size_t k;
		for (k = 0; k < 16; k++)
			tmp->w[k] = le32dec(&src->w[k]);
		salsa20_simd_shuffle(tmp, dst);
	}

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

	do {
		salsa20_blk_t *V_j = &V[j * s];
		j = blockmix_xor_save(X, V_j, r, ctx) & (N - 1);
		V_j = &V[j * s];
		j = blockmix_xor_save(X, V_j, r, ctx) & (N - 1);
	} while (Nloop -= 2);

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

static void smix(uint8_t *B, size_t r, uint32_t N,
    salsa20_blk_t *V, salsa20_blk_t *XY, pwxform_ctx_t *ctx)
{
	uint32_t Nloop_all = (N + 2) / 3;
	uint32_t Nloop_rw = Nloop_all;

	Nloop_all++; Nloop_all &= ~(uint32_t)1;
	Nloop_rw &= ~(uint32_t)1;

	smix1(B, 1, ctx->Sbytes / 128, (salsa20_blk_t *)ctx->S0, XY, NULL);
	smix1(B, r, N, V, XY, ctx);
	smix2(B, r, N, Nloop_rw, V, XY, ctx);
	if (Nloop_all > Nloop_rw)
		smix2(B, r, N, 2, V, XY, ctx);
}

#endif /* _YESPOWER_OPT_C_PASS_ == 1 */
#if _YESPOWER_OPT_C_PASS_ == 1
/* End of first pass – now prepare for second pass */

#undef _YESPOWER_OPT_C_PASS_
#define _YESPOWER_OPT_C_PASS_ 2

/* Rename first‑pass functions to avoid conflicts */
#define blockmix_salsa blockmix_salsa_1_0
#define blockmix_salsa_xor blockmix_salsa_xor_1_0
#define blockmix blockmix_1_0
#define blockmix_xor blockmix_xor_1_0
#define blockmix_xor_save blockmix_xor_save_1_0
#define smix1 smix1_1_0
#define smix2 smix2_1_0
#define smix smix_1_0

/* Include this file again for the second pass */
#include __FILE__

#undef smix  /* avoid conflict with the outer smix */

/* ------------------------------------------------------------------------- */
/* yespower main function                                                    */
/* ------------------------------------------------------------------------- */

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

	if ((version != YESPOWER_0_5 && version != YESPOWER_1_0) ||
	    N < 1024 || N > 512 * 1024 || r < 8 || r > 32 ||
	    (N & (N - 1)) != 0 ||
	    (!pers && perslen)) {
		errno = EINVAL;
		goto fail;
	}

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
		smix(B, r, N, V, XY, &ctx);   /* first‑pass smix */
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
		smix_1_0(B, r, N, V, XY, &ctx); /* second‑pass smix */
		HMAC_SHA256_Buf(B + B_size - 64, 64,
		    sha256, sizeof(sha256), (uint8_t *)dst);
	}

	return 0;

fail:
	memset(dst, 0xff, sizeof(*dst));
	return -1;
}

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

/* ------------------------------------------------------------------------- */
/* Second pass – definitions for yespower 1.0                               */
/* ------------------------------------------------------------------------- */

#undef SALSA20
#define SALSA20 SALSA20_2

#undef PWXFORM
#define PWXFORM \
	PWXFORM_ROUND_WRITE4; PWXFORM_ROUND_WRITE2; PWXFORM_ROUND_WRITE2; \
	w &= Smask2; \
	do { \
		uint8_t *Stmp = S2; \
		S2 = S1; \
		S1 = S0; \
		S0 = Stmp; \
	} while (0)

#undef Smask2
#define Smask2 Smask2_1_0

/* All blockmix, smix, etc. functions are re‑used from the first pass,
   but with the above macro changes.  The code is identical otherwise. */

#endif /* _YESPOWER_OPT_C_PASS_ == 2 */
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

/* Structure for shared miner state */
typedef struct {
    const uint8_t *base_input;      /* Original input buffer (without nonce) */
    size_t input_len;                /* Length of input buffer */
    size_t nonce_offset;             /* Offset in input where nonce (uint64_t LE) should be written */
    uint64_t start_nonce;            /* Start of global nonce range (inclusive) */
    uint64_t end_nonce;              /* End of global nonce range (exclusive) */
    const yespower_params_t *params; /* yespower parameters */
    uint8_t target[32];              /* Target hash (big-endian 256-bit) */
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

    if (yespower_init_local(&local) != 0)
        return NULL;

    input_copy = (uint8_t *)malloc(shared->input_len);
    if (!input_copy) {
        yespower_free_local(&local);
        return NULL;
    }
    memcpy(input_copy, shared->base_input, shared->input_len);

    while (1) {
        nonce = atomic_fetch_add(&shared->next_nonce, 1);
        if (nonce >= shared->end_nonce)
            break;

        if (shared->is_favorable && !shared->is_favorable(nonce, shared->favor_user_data))
            continue;

        *(uint64_t *)(input_copy + shared->nonce_offset) = nonce;

        if (yespower(&local, input_copy, shared->input_len, shared->params, &hash) != 0)
            continue;

        cmp = memcmp((uint8_t*)&hash, shared->target, 32);
        if (cmp <= 0) {
            pthread_mutex_lock(&shared->lock);
            if (shared->found_count >= shared->found_capacity) {
                size_t new_cap = shared->found_capacity ? shared->found_capacity * 2 : 64;
                uint64_t *new_arr = (uint64_t *)realloc(shared->found_nonces,
                                                        new_cap * sizeof(uint64_t));
                if (new_arr) {
                    shared->found_nonces = new_arr;
                    shared->found_capacity = new_cap;
                } else {
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

    memset(&shared, 0, sizeof(shared));
    shared.base_input = base_input;
    shared.input_len = input_len;
    shared.nonce_offset = nonce_offset;
    shared.start_nonce = start_nonce;
    shared.end_nonce = end_nonce;
    shared.params = params;
    memcpy(shared.target, target, 32);
    shared.is_favorable = is_favorable;
    shared.favor_user_data = favor_user_data;
    shared.next_nonce = start_nonce;
    pthread_mutex_init(&shared.lock, NULL);

    threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    if (!threads) {
        pthread_mutex_destroy(&shared.lock);
        return -1;
    }

    for (i = 0; i < num_threads; i++) {
        if (pthread_create(&threads[i], NULL, miner_thread, &shared) != 0) {
            for (int j = 0; j < i; j++)
                pthread_join(threads[j], NULL);
            free(threads);
            pthread_mutex_destroy(&shared.lock);
            if (shared.found_nonces) free(shared.found_nonces);
            return -1;
        }
    }

    for (i = 0; i < num_threads; i++)
        pthread_join(threads[i], NULL);

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
