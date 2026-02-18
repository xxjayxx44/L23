#ifndef _YESPOWER_OPT_C_PASS_
#define _YESPOWER_OPT_C_PASS_ 1
#endif

#if _YESPOWER_OPT_C_PASS_ == 1
/*
 * Multi‑architecture SIMD support:
 * - x86: SSE2, AVX, XOP (original)
 * - ARM: NEON (new) – optimized for Moto G 2025 (ARMv8‑A)
 * - Fallback to generic C when no SIMD is available.
 */
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAVE_NEON 1
#warning "NEON enabled – great for ARM."
#elif defined(__XOP__)
#warning "XOP enabled – great for AMD."
#elif defined(__AVX__)
#warning "AVX enabled – OK for Intel."
#elif defined(__SSE2__)
#warning "SSE2 enabled – OK."
#elif defined(__x86_64__) || defined(__i386__)
#warning "SSE2 not enabled. Expect poor performance."
#else
#warning "Building generic code for non‑x86/ARM. OK."
#endif

/* Use generic SSE2 version – runs well on many CPUs. */
#undef USE_SSE4_FOR_32BIT

#ifdef __SSE2__
/* GCC 4.6‑4.9 tune workaround (original) */
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

/* Compiler hints */
#if __STDC_VERSION__ >= 199901L
#define restrict
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

/* Prefetch support */
#ifdef __SSE__
#define PREFETCH(x, hint) _mm_prefetch((const char *)(x), (hint))
#elif defined(HAVE_NEON)
#define PREFETCH(x, hint) __builtin_prefetch(x)
#else
#undef PREFETCH
#endif

/* 128‑bit block type – works with SSE2, NEON, and generic C */
typedef union {
	uint32_t w[16];
	uint64_t d[8];
#ifdef __SSE2__
	__m128i q[4];
#endif
#ifdef HAVE_NEON
	uint32x4_t q[4];   /* NEON uses 4x32‑bit vectors */
#endif
} salsa20_blk_t;

/* Shuffle/unshuffle for endian‑agnostic internal representation */
static inline void salsa20_simd_shuffle(const salsa20_blk_t *Bin,
    salsa20_blk_t *Bout) {
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
    salsa20_blk_t *Bout) {
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
#ifdef __SSE2__   /* x86 SIMD path (original) */

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
	ARX(X1, X0, X3, 7) \
	ARX(X2, X1, X0, 9) \
	ARX(X3, X2, X1, 13) \
	ARX(X0, X3, X2, 18) \
	X1 = _mm_shuffle_epi32(X1, 0x93); \
	X2 = _mm_shuffle_epi32(X2, 0x4E); \
	X3 = _mm_shuffle_epi32(X3, 0x39); \
	ARX(X3, X0, X1, 7) \
	ARX(X2, X3, X0, 9) \
	ARX(X1, X2, X3, 13) \
	ARX(X0, X1, X2, 18) \
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
#define SALSA20_8ROUNDS SALSA20_2ROUNDS SALSA20_2ROUNDS SALSA20_2ROUNDS SALSA20_2ROUNDS
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

#elif defined(HAVE_NEON)   /* ARM NEON path (new) */

#define DECL_X \
	uint32x4_t X0, X1, X2, X3;
#define DECL_Y \
	uint32x4_t Y0, Y1, Y2, Y3;
#define READ_X(in) \
	X0 = (in).q[0]; X1 = (in).q[1]; X2 = (in).q[2]; X3 = (in).q[3];
#define WRITE_X(out) \
	(out).q[0] = X0; (out).q[1] = X1; (out).q[2] = X2; (out).q[3] = X3;

/* NEON rotate: shift left then OR with shift right */
#define ROTATE(v, s) vsriq_n_u32(vshlq_n_u32(v, s), v, 32 - (s))

#define ARX(out, in1, in2, s) { \
	uint32x4_t tmp = vaddq_u32(in1, in2); \
	out = veorq_u32(out, ROTATE(tmp, s)); \
}

#define SALSA20_2ROUNDS \
	ARX(X1, X0, X3, 7) \
	ARX(X2, X1, X0, 9) \
	ARX(X3, X2, X1, 13) \
	ARX(X0, X3, X2, 18) \
	X1 = vextq_u32(X1, X1, 1);   /* 0x93 */ \
	X2 = vextq_u32(X2, X2, 2);   /* 0x4E */ \
	X3 = vextq_u32(X3, X3, 3);   /* 0x39 */ \
	ARX(X3, X0, X1, 7) \
	ARX(X2, X3, X0, 9) \
	ARX(X1, X2, X3, 13) \
	ARX(X0, X1, X2, 18) \
	X1 = vextq_u32(X1, X1, 3);   /* 0x39 */ \
	X2 = vextq_u32(X2, X2, 2);   /* 0x4E */ \
	X3 = vextq_u32(X3, X3, 1);   /* 0x93 */

#define SALSA20_wrapper(out, rounds) { \
	uint32x4_t Z0 = X0, Z1 = X1, Z2 = X2, Z3 = X3; \
	rounds \
	(out).q[0] = X0 = vaddq_u32(X0, Z0); \
	(out).q[1] = X1 = vaddq_u32(X1, Z1); \
	(out).q[2] = X2 = vaddq_u32(X2, Z2); \
	(out).q[3] = X3 = vaddq_u32(X3, Z3); \
}

#define SALSA20_2(out) SALSA20_wrapper(out, SALSA20_2ROUNDS)
#define SALSA20_8ROUNDS SALSA20_2ROUNDS SALSA20_2ROUNDS SALSA20_2ROUNDS SALSA20_2ROUNDS
#define SALSA20_8(out) SALSA20_wrapper(out, SALSA20_8ROUNDS)

#define XOR_X(in) \
	X0 = veorq_u32(X0, (in).q[0]); \
	X1 = veorq_u32(X1, (in).q[1]); \
	X2 = veorq_u32(X2, (in).q[2]); \
	X3 = veorq_u32(X3, (in).q[3]);

#define XOR_X_2(in1, in2) \
	X0 = veorq_u32((in1).q[0], (in2).q[0]); \
	X1 = veorq_u32((in1).q[1], (in2).q[1]); \
	X2 = veorq_u32((in1).q[2], (in2).q[2]); \
	X3 = veorq_u32((in1).q[3], (in2).q[3]);

#define XOR_X_WRITE_XOR_Y_2(out, in) \
	(out).q[0] = Y0 = veorq_u32((out).q[0], (in).q[0]); \
	(out).q[1] = Y1 = veorq_u32((out).q[1], (in).q[1]); \
	(out).q[2] = Y2 = veorq_u32((out).q[2], (in).q[2]); \
	(out).q[3] = Y3 = veorq_u32((out).q[3], (in).q[3]); \
	X0 = veorq_u32(X0, Y0); \
	X1 = veorq_u32(X1, Y1); \
	X2 = veorq_u32(X2, Y2); \
	X3 = veorq_u32(X3, Y3);

#define INTEGERIFY vgetq_lane_u32(X0, 0)

#else   /* Generic C fallback (original) */

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

static inline void salsa20(salsa20_blk_t *B, salsa20_blk_t *Bout, uint32_t doublerounds) {
	salsa20_blk_t X;
#define x X.w
	salsa20_simd_unshuffle(B, &X);
	do {
#define R(a,b) (((a) << (b)) | ((a) >> (32 - (b))))
		x[ 4] ^= R(x[ 0]+x[12], 7); x[ 8] ^= R(x[ 4]+x[ 0], 9);
		x[12] ^= R(x[ 8]+x[ 4],13); x[ 0] ^= R(x[12]+x[ 8],18);
		x[ 9] ^= R(x[ 5]+x[ 1], 7); x[13] ^= R(x[ 9]+x[ 5], 9);
		x[ 1] ^= R(x[13]+x[ 9],13); x[ 5] ^= R(x[ 1]+x[13],18);
		x[14] ^= R(x[10]+x[ 6], 7); x[ 2] ^= R(x[14]+x[10], 9);
		x[ 6] ^= R(x[ 2]+x[14],13); x[10] ^= R(x[ 6]+x[ 2],18);
		x[ 3] ^= R(x[15]+x[11], 7); x[ 7] ^= R(x[ 3]+x[15], 9);
		x[11] ^= R(x[ 7]+x[ 3],13); x[15] ^= R(x[11]+x[ 7],18);
		x[ 1] ^= R(x[ 0]+x[ 3], 7); x[ 2] ^= R(x[ 1]+x[ 0], 9);
		x[ 3] ^= R(x[ 2]+x[ 1],13); x[ 0] ^= R(x[ 3]+x[ 2],18);
		x[ 6] ^= R(x[ 5]+x[ 4], 7); x[ 7] ^= R(x[ 6]+x[ 5], 9);
		x[ 4] ^= R(x[ 7]+x[ 6],13); x[ 5] ^= R(x[ 4]+x[ 7],18);
		x[11] ^= R(x[10]+x[ 9], 7); x[ 8] ^= R(x[11]+x[10], 9);
		x[ 9] ^= R(x[ 8]+x[11],13); x[10] ^= R(x[ 9]+x[ 8],18);
		x[12] ^= R(x[15]+x[14], 7); x[13] ^= R(x[12]+x[15], 9);
		x[14] ^= R(x[13]+x[12],13); x[15] ^= R(x[14]+x[13],18);
#undef R
	} while (--doublerounds);
#undef x
	uint32_t i;
	salsa20_simd_shuffle(&X, Bout);
	for (i = 0; i < 16; i += 4) {
		B->w[i]   = Bout->w[i]   += B->w[i];
		B->w[i+1] = Bout->w[i+1] += B->w[i+1];
		B->w[i+2] = Bout->w[i+2] += B->w[i+2];
		B->w[i+3] = Bout->w[i+3] += B->w[i+3];
	}
}

#define SALSA20_2(out) salsa20(&X, &out, 1)
#define SALSA20_8(out) salsa20(&X, &out, 4)

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

/* Common macro for both SIMD and generic */
#define SALSA20_XOR_MEM(in, out) \
	XOR_X(in) \
	SALSA20(out)

#define SALSA20 SALSA20_8
#else /* pass 2 */
#undef SALSA20
#define SALSA20 SALSA20_2
#endif
#if _YESPOWER_OPT_C_PASS_ == 1
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

#define DECL_SMASK2REG
#define MAYBE_MEMORY_BARRIER

/* ------------------------------------------------------------------ */
/* pwxform SIMD for x86 (original, with minor tuning)                */
/* ------------------------------------------------------------------ */
#ifdef __SSE2__

#ifdef __AVX__
#define HI32(X) _mm_srli_si128((X), 4)
#elif 1
#define HI32(X) _mm_shuffle_epi32((X), _MM_SHUFFLE(2,3,0,1))
#else
#define HI32(X) _mm_srli_epi64((X), 32)
#endif

#if defined(__x86_64__) && __GNUC__ == 4 && __GNUC_MINOR__ < 6 && !defined(__ICC)
#ifdef __AVX__
#define MOVQ "vmovq"
#else
#define MOVQ "movd"
#endif
#define EXTRACT64(X) ({ uint64_t result; __asm__(MOVQ " %1, %0" : "=r" (result) : "x" (X)); result; })
#elif defined(__x86_64__) && !defined(_MSC_VER) && !defined(__OPEN64__)
#define EXTRACT64(X) _mm_cvtsi128_si64(X)
#elif defined(__x86_64__) && defined(__SSE4_1__)
#include <smmintrin.h>
#define EXTRACT64(X) _mm_extract_epi64((X), 0)
#elif defined(USE_SSE4_FOR_32BIT) && defined(__SSE4_1__)
#include <smmintrin.h>
#else
#define EXTRACT64(X) ((uint64_t)(uint32_t)_mm_cvtsi128_si32(X) | ((uint64_t)(uint32_t)_mm_cvtsi128_si32(HI32(X)) << 32))
#endif

#if defined(__x86_64__) && (defined(__AVX__) || !defined(__GNUC__))
#undef DECL_SMASK2REG
#if defined(__GNUC__) && !defined(__ICC)
#define DECL_SMASK2REG uint64_t Smask2reg = Smask2;
#define FORCE_REGALLOC_1 __asm__("" : "=a" (x), "+d" (Smask2reg), "+S" (S0), "+D" (S1));
#define FORCE_REGALLOC_2 __asm__("" : : "c" (lo));
#else
static volatile uint64_t Smask2var = Smask2;
#define DECL_SMASK2REG uint64_t Smask2reg = Smask2var;
#define FORCE_REGALLOC_1
#define FORCE_REGALLOC_2
#endif
#define PWXFORM_SIMD(X) { \
	uint64_t x; FORCE_REGALLOC_1 \
	uint32_t lo = x = EXTRACT64(X) & Smask2reg; FORCE_REGALLOC_2 \
	uint32_t hi = x >> 32; \
	X = _mm_mul_epu32(HI32(X), X); \
	X = _mm_add_epi64(X, *(__m128i *)(S0 + lo)); \
	X = _mm_xor_si128(X, *(__m128i *)(S1 + hi)); \
}
#elif defined(__x86_64__)
#undef MAYBE_MEMORY_BARRIER
#define MAYBE_MEMORY_BARRIER __asm__("" : : : "memory");
#ifdef __ILP32__
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
#define PWXFORM_SIMD(X) { \
	__m128i x = _mm_and_si128(X, _mm_set1_epi64x(Smask2)); \
	__m128i s0 = *(__m128i *)(S0 + (uint32_t)_mm_cvtsi128_si32(x)); \
	__m128i s1 = *(__m128i *)(S1 + (uint32_t)_mm_extract_epi32(x, 1)); \
	X = _mm_mul_epu32(HI32(X), X); \
	X = _mm_add_epi64(X, s0); \
	X = _mm_xor_si128(X, s1); \
}
#else
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

/* ------------------------------------------------------------------ */
/* pwxform SIMD for ARM NEON (new)                                    */
/* ------------------------------------------------------------------ */
#elif defined(HAVE_NEON)

/* NEON: 32x32->64 multiply, 64-bit add, xor */
#define PWXFORM_SIMD(X) { \
	uint32x2_t lo = vget_low_u32(X); \
	uint32x2_t hi = vget_high_u32(X); \
	uint64x2_t prod = vmull_u32(hi, lo);          /* hi * lo */ \
	uint64_t mask = Smask2; \
	uint32_t lo_idx = vget_lane_u32(vreinterpret_u32_u64(vand_u64(vreinterpret_u64_u32(lo), vcreate_u64(mask))), 0); \
	uint32_t hi_idx = vget_lane_u32(vreinterpret_u32_u64(vshr_n_u64(vand_u64(vreinterpret_u64_u32(hi), vcreate_u64(mask)), 32)), 0); \
	uint64x2_t s0 = vld1q_u64((uint64_t *)(S0 + lo_idx * 8)); \
	uint64x2_t s1 = vld1q_u64((uint64_t *)(S1 + hi_idx * 8)); \
	X = vreinterpretq_u32_u64(veorq_u64(vaddq_u64(prod, s0), s1)); \
}

#define PWXFORM_SIMD_WRITE(X, Sw) \
	PWXFORM_SIMD(X) \
	__builtin_prefetch(Sw + w + 64); \
	vst1q_u32((uint32_t *)(Sw + w), X); \
	__builtin_prefetch(Sw + w + 128);

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

#else   /* Generic C fallback (original) */

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
#if _YESPOWER_OPT_C_PASS_ == 1
static inline void blockmix_salsa(const salsa20_blk_t *restrict Bin,
    salsa20_blk_t *restrict Bout) {
	DECL_X
	READ_X(Bin[1])
	SALSA20_XOR_MEM(Bin[0], Bout[0])
	SALSA20_XOR_MEM(Bin[1], Bout[1])
}

static inline uint32_t blockmix_salsa_xor(const salsa20_blk_t *restrict Bin1,
    const salsa20_blk_t *restrict Bin2, salsa20_blk_t *restrict Bout) {
	DECL_X
	XOR_X_2(Bin1[1], Bin2[1])
	XOR_X(Bin1[0])
	SALSA20_XOR_MEM(Bin2[0], Bout[0])
	XOR_X(Bin1[1])
	SALSA20_XOR_MEM(Bin2[1], Bout[1])
	return INTEGERIFY;
}
#endif

/* blockmix (with or without pwxform) */
static void blockmix(const salsa20_blk_t *restrict Bin,
    salsa20_blk_t *restrict Bout, size_t r, pwxform_ctx_t *restrict ctx) {
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
	r = r * 2 - 1;
	READ_X(Bin[r])
	DECL_SMASK2REG
	i = 0;
	do {
		XOR_X(Bin[i])
		PWXFORM
		if (unlikely(i >= r)) break;
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
    size_t r, pwxform_ctx_t *restrict ctx) {
	if (unlikely(!ctx))
		return blockmix_salsa_xor(Bin1, Bin2, Bout);
	uint8_t *S0 = ctx->S0, *S1 = ctx->S1;
#if _YESPOWER_OPT_C_PASS_ > 1
	uint8_t *S2 = ctx->S2;
	size_t w = ctx->w;
#endif
	size_t i;
	DECL_X
	r = r * 2 - 1;
#ifdef PREFETCH
	PREFETCH(&Bin2[r], _MM_HINT_T0);
	for (i = 0; i < r; i++) PREFETCH(&Bin2[i], _MM_HINT_T0);
#endif
	XOR_X_2(Bin1[r], Bin2[r])
	DECL_SMASK2REG
	i = 0;
	r--;
	do {
		XOR_X(Bin1[i]); XOR_X(Bin2[i]);
		PWXFORM
		WRITE_X(Bout[i])
		XOR_X(Bin1[i+1]); XOR_X(Bin2[i+1]);
		PWXFORM
		if (unlikely(i >= r)) break;
		WRITE_X(Bout[i+1])
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
    salsa20_blk_t *restrict Bin2, size_t r, pwxform_ctx_t *restrict ctx) {
	uint8_t *S0 = ctx->S0, *S1 = ctx->S1;
#if _YESPOWER_OPT_C_PASS_ > 1
	uint8_t *S2 = ctx->S2;
	size_t w = ctx->w;
#endif
	size_t i;
	DECL_X; DECL_Y;
	r = r * 2 - 1;
#ifdef PREFETCH
	PREFETCH(&Bin2[r], _MM_HINT_T0);
	for (i = 0; i < r; i++) PREFETCH(&Bin2[i], _MM_HINT_T0);
#endif
	XOR_X_2(Bin1out[r], Bin2[r])
	DECL_SMASK2REG
	i = 0;
	r--;
	do {
		XOR_X_WRITE_XOR_Y_2(Bin2[i], Bin1out[i])
		PWXFORM
		WRITE_X(Bin1out[i])
		XOR_X_WRITE_XOR_Y_2(Bin2[i+1], Bin1out[i+1])
		PWXFORM
		if (unlikely(i >= r)) break;
		WRITE_X(Bin1out[i+1])
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
static inline uint32_t integerify(const salsa20_blk_t *B, size_t r) {
	return (uint32_t)B[2 * r - 1].d[0];
}
#endif

/* smix1 and smix2 (unchanged) */
static void smix1(uint8_t *B, size_t r, uint32_t N,
    salsa20_blk_t *V, salsa20_blk_t *XY, pwxform_ctx_t *ctx) {
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
			j &= n - 1; j += i - 1;
			V_j = &V[j * s];
			j = blockmix_xor(X, V_j, Y, r, ctx);
			j &= n - 1; j += i;
			V_j = &V[j * s];
			X = Y + s;
			j = blockmix_xor(Y, V_j, X, r, ctx);
		}
	}
	n >>= 1;
	j &= n - 1; j += N - 2 - n;
	V_j = &V[j * s];
	Y = X + s;
	j = blockmix_xor(X, V_j, Y, r, ctx);
	j &= n - 1; j += N - 1 - n;
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
    salsa20_blk_t *V, salsa20_blk_t *XY, pwxform_ctx_t *ctx) {
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
		const salsa20_blk_t *V_j = &V[j * s];
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

static void smix(uint8_t *B, size_t r, uint32_t N,
    salsa20_blk_t *V, salsa20_blk_t *XY, pwxform_ctx_t *ctx) {
#if _YESPOWER_OPT_C_PASS_ == 1
	uint32_t Nloop_all = (N + 2) / 3; Nloop_all++; Nloop_all &= ~1U;
	uint32_t Nloop_rw = Nloop_all; Nloop_rw &= ~1U;
#else
	uint32_t Nloop_rw = (N + 2) / 3; Nloop_rw++; Nloop_rw &= ~1U;
#endif
	smix1(B, 1, ctx->Sbytes / 128, (salsa20_blk_t *)ctx->S0, XY, NULL);
	smix1(B, r, N, V, XY, ctx);
	smix2(B, r, N, Nloop_rw, V, XY, ctx);
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
 * yespower - main function (optimized with NEON for ARM)
 *
 * To achieve work reuse (midstate) and nonce manipulation:
 * 1. Allocate a yespower_local_t structure using yespower_init_local().
 * 2. For each candidate, modify the src buffer (e.g., insert a 32‑bit nonce
 *    at a known offset) and call yespower() with the same local structure.
 *    The internal memory (V, XY, S) is reused across calls, avoiding
 *    reallocation and reducing overhead.
 * 3. For maximum performance, ensure src is aligned (the local->aligned
 *    region is 64‑byte aligned) and use the provided PREFETCH hints.
 *
 * No algorithmic changes are made – hash outputs remain identical to the
 * original yespower specification.
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

	if ((version != YESPOWER_0_5 && version != YESPOWER_1_0) ||
	    N < 1024 || N > 512 * 1024 || r < 8 || r > 32 ||
	    (N & (N - 1)) != 0 || (!pers && perslen)) {
		errno = EINVAL; goto fail;
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
		if (free_region(local)) goto fail;
		if (!alloc_region(local, need)) goto fail;
	}
	B = (uint8_t *)local->aligned;
	V = (salsa20_blk_t *)(B + B_size);
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
			HMAC_SHA256_Buf(dst, sizeof(*dst), pers, perslen, sha256);
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
#endif
