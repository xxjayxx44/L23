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
 * =========================================================================
 *                    YESPOWER-OPT.C – ENHANCED VERSION
 * -------------------------------------------------------------------------
 *   - Full SIMD (SSE2/AVX/XOP) and generic C implementations
 *   - Fast inline assembly endian helpers (x86/x86-64)
 *   - Global gate test, bias validation, work restart (atomic)
 *   - INTERACTIVE COMMAND INTERFACE – type commands while mining!
 *   - All features DISABLED by default; opt‑in via typed commands
 *   - Fully backward compatible with original yespower API
 * =========================================================================
 */

#ifndef _YESPOWER_OPT_C_PASS_
#define _YESPOWER_OPT_C_PASS_ 1
#endif

/* -------------------------------------------------------------------------
   Compiler feature detection and SIMD hints
   ------------------------------------------------------------------------- */
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

#undef USE_SSE4_FOR_32BIT

#ifdef __SSE2__
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
#include <stdatomic.h>
#include <stdio.h>      /* for printf, sscanf */

/* POSIX headers for interactive non‑blocking input */
#ifdef __unix__
#include <unistd.h>
#include <fcntl.h>
#include <sys/select.h>
#include <termios.h>
#endif

#include "insecure_memzero.h"
#include "sha256.h"
#include "sysendian.h"

#include "yespower.h"
#include "yespower-platform.c"

/* -------------------------------------------------------------------------
   Interactive mode static variables (only on Unix)
   ------------------------------------------------------------------------- */
#ifdef __unix__
static int interactive_fd = -1;
static atomic_bool interactive_enabled = ATOMIC_VAR_INIT(0);
static struct termios orig_termios;
static char cmd_buffer[256];
static size_t cmd_pos = 0;
#endif

#if __STDC_VERSION__ >= 199901L
#define restrict
#elif defined(__GNUC__)
#define restrict __restrict
#else
#define restrict
#endif

#ifdef __GNUC__
#define unlikely(exp) __builtin_expect(exp, 0)
#define likely(exp) __builtin_expect(exp, 1)
#else
#define unlikely(exp) (exp)
#define likely(exp) (exp)
#endif

#ifdef __SSE__
#define PREFETCH(x, hint) _mm_prefetch((const char *)(x), (hint))
#else
#define PREFETCH(x, hint) ((void)0)
#endif

/* -------------------------------------------------------------------------
   ENHANCED FEATURES – GLOBAL CONFIGURATION (all zero = disabled)
   ------------------------------------------------------------------------- */

/* Thread restart / work‑restart signaling */
typedef struct {
    atomic_bool restart_requested;
    atomic_bool work_available;
    atomic_uint_fast64_t restart_count;
} work_control_t;

/* Bias validator */
typedef struct {
    uint64_t hash_counter;
    uint64_t pattern_checks;
    uint32_t bias_threshold;
} bias_validator_t;

/* Gate test */
typedef struct {
    uint32_t gate_mask;
    uint32_t gate_threshold;
    uint8_t  skip_unfavorable;
} gate_test_t;

/* Global instances – all zero initialised => features OFF */
static gate_test_t      yespower_gate      = {0, 0, 0};
static bias_validator_t yespower_bias      = {0, 0, 0};
static work_control_t   yespower_control   = {
    ATOMIC_VAR_INIT(0),
    ATOMIC_VAR_INIT(1),
    ATOMIC_VAR_INIT(0)
};

/* Favorable mask bitmap – used when gate.skip_unfavorable != 0 */
static uint8_t *yespower_favorable_mask = NULL;
static size_t   yespower_favorable_size = 0;

/* Forward declarations of public configuration functions (used by interactive) */
void yespower_set_gate(uint32_t mask, uint32_t threshold, uint8_t skip_unfavorable);
void yespower_set_favorable_mask(uint8_t *mask, size_t size);
void yespower_set_bias_threshold(uint32_t threshold);
void yespower_request_restart(void);
uint64_t yespower_get_restart_count(void);
void yespower_reset_features(void);

/* -------------------------------------------------------------------------
   Fast endian helpers (inline assembly)
   ------------------------------------------------------------------------- */
static inline uint32_t fast_le32dec(const void *pp)
{
    const uint8_t *p = (const uint8_t *)pp;
#if defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))
    uint32_t v;
    __asm__("movl %1, %0" : "=r"(v) : "m"(*(const uint32_t *)p));
    return v;
#elif defined(__GNUC__) && defined(__LITTLE_ENDIAN__)
    return *(const uint32_t *)p;
#else
    return ((uint32_t)p[0] | ((uint32_t)p[1] << 8) |
            ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
#endif
}

static inline void fast_le32enc(void *pp, uint32_t x)
{
    uint8_t *p = (uint8_t *)pp;
#if defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))
    __asm__("movl %1, %0" : "=m"(*(uint32_t *)p) : "r"(x));
#elif defined(__GNUC__) && defined(__LITTLE_ENDIAN__)
    *(uint32_t *)p = x;
#else
    p[0] = x & 0xff;
    p[1] = (x >> 8) & 0xff;
    p[2] = (x >> 16) & 0xff;
    p[3] = (x >> 24) & 0xff;
#endif
}

static inline uint64_t fast_le64dec(const void *pp)
{
    const uint8_t *p = (const uint8_t *)pp;
#if defined(__GNUC__) && defined(__x86_64__)
    uint64_t v;
    __asm__("movq %1, %0" : "=r"(v) : "m"(*(const uint64_t *)p));
    return v;
#elif defined(__GNUC__) && defined(__LITTLE_ENDIAN__)
    return *(const uint64_t *)p;
#else
    return (uint64_t)fast_le32dec(p) | ((uint64_t)fast_le32dec(p + 4) << 32);
#endif
}

static inline void fast_le64enc(void *pp, uint64_t x)
{
    uint8_t *p = (uint8_t *)pp;
#if defined(__GNUC__) && defined(__x86_64__)
    __asm__("movq %1, %0" : "=m"(*(uint64_t *)p) : "r"(x));
#elif defined(__GNUC__) && defined(__LITTLE_ENDIAN__)
    *(uint64_t *)p = x;
#else
    fast_le32enc(p, (uint32_t)x);
    fast_le32enc(p + 4, (uint32_t)(x >> 32));
#endif
}

/* -------------------------------------------------------------------------
   CORE FEATURE FUNCTIONS (always compiled in)
   ------------------------------------------------------------------------- */

/* Bias validator – always counts; threshold >0 activates detection */
static inline void validate_bias(uint32_t value, uint32_t pattern)
{
    yespower_bias.hash_counter++;
    yespower_bias.pattern_checks++;
    if (yespower_bias.bias_threshold != 0) {
        (void)pattern;
        if (value % yespower_bias.bias_threshold == 0) {
            /* bias detected – currently only counted */
        }
    }
}

/* Gate test – returns 1 if block should be processed */
static inline int gate_test(uint32_t value)
{
    if (yespower_gate.gate_mask == 0 && yespower_gate.gate_threshold == 0)
        return 1;
    uint32_t gated = value & yespower_gate.gate_mask;
    if (yespower_gate.skip_unfavorable && yespower_favorable_mask != NULL) {
        size_t idx = (value >> 3) % yespower_favorable_size;
        uint8_t bit = 1 << (value & 7);
        if ((yespower_favorable_mask[idx] & bit) == 0)
            return 0;
    }
    return (gated >= yespower_gate.gate_threshold);
}

/* Work restart check – returns 1 if restart requested */
static inline int check_restart(void)
{
    if (atomic_load_explicit(&yespower_control.restart_requested,
                             memory_order_acquire)) {
        atomic_fetch_add_explicit(&yespower_control.restart_count, 1,
                                  memory_order_relaxed);
        atomic_store_explicit(&yespower_control.restart_requested, 0,
                              memory_order_release);
        return 1;
    }
    return 0;
}

/* -------------------------------------------------------------------------
   INTERACTIVE COMMAND PROCESSING (POSIX only)
   ------------------------------------------------------------------------- */
#ifdef __unix__

/* Set non‑blocking mode on a file descriptor */
static int set_nonblocking(int fd)
{
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags == -1) return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

/* Restore terminal to cooked mode */
static void restore_terminal(void)
{
    if (interactive_fd == 0) {
        tcsetattr(0, TCSAFLUSH, &orig_termios);
    }
}

/* Parse a hex string into a byte array */
static size_t hex2bin(uint8_t *out, size_t out_max, const char *hex)
{
    size_t len = strlen(hex);
    if (len % 2 != 0) return 0;
    size_t bin_len = len / 2;
    if (bin_len > out_max) bin_len = out_max;
    for (size_t i = 0; i < bin_len; i++) {
        unsigned int byte;
        if (sscanf(hex + 2*i, "%2x", &byte) != 1)
            return 0;
        out[i] = (uint8_t)byte;
    }
    return bin_len;
}

/* Execute a single command */
static void execute_command(char *cmd)
{
    char *argv[8];
    int argc = 0;
    char *token = strtok(cmd, " \t\n");
    while (token && argc < 8) {
        argv[argc++] = token;
        token = strtok(NULL, " \t\n");
    }
    if (argc == 0) return;

    if (strcmp(argv[0], "gate") == 0 && argc >= 4) {
        uint32_t mask = strtoul(argv[1], NULL, 0);
        uint32_t thresh = strtoul(argv[2], NULL, 0);
        uint8_t skip = (uint8_t)strtoul(argv[3], NULL, 0);
        yespower_set_gate(mask, thresh, skip);
        printf("[yespower] gate: mask=0x%x threshold=%u skip=%u\n",
               mask, thresh, skip);
    }
    else if (strcmp(argv[0], "favmask") == 0 && argc >= 2) {
        static uint8_t mask_buf[1024];
        size_t len = hex2bin(mask_buf, sizeof(mask_buf), argv[1]);
        if (len > 0) {
            yespower_set_favorable_mask(mask_buf, len);
            printf("[yespower] favorable mask set (%zu bytes)\n", len);
        } else {
            printf("[yespower] invalid hex string\n");
        }
    }
    else if (strcmp(argv[0], "bias") == 0 && argc >= 2) {
        uint32_t thresh = strtoul(argv[1], NULL, 0);
        yespower_set_bias_threshold(thresh);
        printf("[yespower] bias threshold = %u\n", thresh);
    }
    else if (strcmp(argv[0], "restart") == 0) {
        yespower_request_restart();
        printf("[yespower] restart requested\n");
    }
    else if (strcmp(argv[0], "stats") == 0) {
        printf("[yespower] hash count: %llu\n",
               (unsigned long long)yespower_bias.hash_counter);
        printf("[yespower] pattern checks: %llu\n",
               (unsigned long long)yespower_bias.pattern_checks);
        printf("[yespower] restarts: %llu\n",
               (unsigned long long)yespower_get_restart_count());
        printf("[yespower] gate: mask=0x%x thresh=%u skip=%u\n",
               yespower_gate.gate_mask,
               yespower_gate.gate_threshold,
               yespower_gate.skip_unfavorable);
    }
    else if (strcmp(argv[0], "reset") == 0) {
        yespower_reset_features();
        printf("[yespower] features reset to defaults\n");
    }
    else if (strcmp(argv[0], "help") == 0) {
        printf("Available commands:\n");
        printf("  gate <mask> <threshold> <skip>   - set gate parameters\n");
        printf("  favmask <hex>                    - set favorable mask (hex bytes)\n");
        printf("  bias <threshold>                 - set bias threshold (0 = off)\n");
        printf("  restart                          - request work restart\n");
        printf("  stats                            - print statistics\n");
        printf("  reset                            - disable all features\n");
        printf("  help                             - this help\n");
    }
    else {
        printf("[yespower] unknown command: %s\n", argv[0]);
    }
}

/* Poll for input and process complete lines */
static void process_commands(void)
{
    if (!atomic_load_explicit(&interactive_enabled, memory_order_relaxed))
        return;
    if (interactive_fd == -1) return;

    char ch;
    ssize_t n;
    while ((n = read(interactive_fd, &ch, 1)) == 1) {
        if (ch == '\n' || ch == '\r') {
            if (cmd_pos > 0) {
                cmd_buffer[cmd_pos] = '\0';
                execute_command(cmd_buffer);
                cmd_pos = 0;
            }
        } else if (cmd_pos < sizeof(cmd_buffer) - 1) {
            cmd_buffer[cmd_pos++] = ch;
        }
    }
}

#else /* !__unix__ */
static void process_commands(void) { }
#endif /* __unix__ */

/* Public function to enable interactive mode */
void yespower_enable_interactive(int fd)
{
#ifdef __unix__
    if (fd < 0) return;
    interactive_fd = fd;
    if (set_nonblocking(fd) == 0) {
        atomic_store_explicit(&interactive_enabled, 1, memory_order_release);
        if (fd == 0) {
            tcgetattr(0, &orig_termios);
            struct termios raw = orig_termios;
            raw.c_lflag &= ~(ICANON | ECHO);
            tcsetattr(0, TCSAFLUSH, &raw);
            atexit(restore_terminal);
        }
        printf("[yespower] interactive mode enabled (fd=%d)\n", fd);
    }
#else
    (void)fd;
#endif
}

void yespower_disable_interactive(void)
{
#ifdef __unix__
    atomic_store_explicit(&interactive_enabled, 0, memory_order_release);
    if (interactive_fd == 0) restore_terminal();
    interactive_fd = -1;
#endif
}

/* -------------------------------------------------------------------------
   Salsa20 SIMD shuffle/unshuffle
   ------------------------------------------------------------------------- */
typedef union {
    uint32_t w[16];
    uint64_t d[8];
#ifdef __SSE2__
    __m128i q[4];
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
/* End of Part 1 */
/* =========================================================================
   PASS 1 (yescrypt 0.5 / yespower 0.5)
   ========================================================================= */
#if _YESPOWER_OPT_C_PASS_ == 1

/* -------------------------------------------------------------------------
   SIMD‑accelerated Salsa20 (SSE2/XOP/AVX paths)
   ------------------------------------------------------------------------- */
#ifdef __SSE2__

#define DECL_X __m128i X0, X1, X2, X3;
#define DECL_Y __m128i Y0, Y1, Y2, Y3;
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
    ARX(X1, X0, X3, 7) ARX(X2, X1, X0, 9) \
    ARX(X3, X2, X1, 13) ARX(X0, X3, X2, 18) \
    X1 = _mm_shuffle_epi32(X1, 0x93); \
    X2 = _mm_shuffle_epi32(X2, 0x4E); \
    X3 = _mm_shuffle_epi32(X3, 0x39); \
    ARX(X3, X0, X1, 7) ARX(X2, X3, X0, 9) \
    ARX(X1, X2, X3, 13) ARX(X0, X1, X2, 18) \
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

#define SALSA20_2(out) SALSA20_wrapper(out, SALSA20_2ROUNDS);
#define SALSA20_8(out) SALSA20_wrapper(out, SALSA20_8ROUNDS);

#define SALSA20_8ROUNDS \
    SALSA20_2ROUNDS SALSA20_2ROUNDS \
    SALSA20_2ROUNDS SALSA20_2ROUNDS

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

#else /* !__SSE2__ – generic C version */

#define DECL_X salsa20_blk_t X;
#define DECL_Y salsa20_blk_t Y;

#define COPY(out, in) \
    (out).d[0] = (in).d[0]; (out).d[1] = (in).d[1]; \
    (out).d[2] = (in).d[2]; (out).d[3] = (in).d[3]; \
    (out).d[4] = (in).d[4]; (out).d[5] = (in).d[5]; \
    (out).d[6] = (in).d[6]; (out).d[7] = (in).d[7];

#define READ_X(in) COPY(X, in)
#define WRITE_X(out) COPY(out, X)

/**
 * salsa20(B):
 * Apply the Salsa20 core to the provided block.
 */
static inline void salsa20(salsa20_blk_t *restrict B,
                           salsa20_blk_t *restrict Bout,
                           uint32_t doublerounds)
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
            B->w[i]   = Bout->w[i]   += B->w[i];
            B->w[i+1] = Bout->w[i+1] += B->w[i+1];
            B->w[i+2] = Bout->w[i+2] += B->w[i+2];
            B->w[i+3] = Bout->w[i+3] += B->w[i+3];
        }
    }
}

#define SALSA20_2(out) salsa20(&X, &out, 1);
#define SALSA20_8(out) salsa20(&X, &out, 4);

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

#endif /* __SSE2__ */

#define SALSA20_XOR_MEM(in, out) \
    XOR_X(in) \
    SALSA20(out)

/* For pass 1, SALSA20 is defined as SALSA20_8 */
#define SALSA20 SALSA20_8

/* -------------------------------------------------------------------------
   Blockmix_salsa (no pwxform)
   ------------------------------------------------------------------------- */
static inline void blockmix_salsa(const salsa20_blk_t *restrict Bin,
                                  salsa20_blk_t *restrict Bout)
{
    DECL_X
    READ_X(Bin[1])
    SALSA20_XOR_MEM(Bin[0], Bout[0])
    SALSA20_XOR_MEM(Bin[1], Bout[1])
}

static inline uint32_t blockmix_salsa_xor(const salsa20_blk_t *restrict Bin1,
                                          const salsa20_blk_t *restrict Bin2,
                                          salsa20_blk_t *restrict Bout)
{
    DECL_X
    XOR_X_2(Bin1[1], Bin2[1])
    XOR_X(Bin1[0])
    SALSA20_XOR_MEM(Bin2[0], Bout[0])
    XOR_X(Bin1[1])
    SALSA20_XOR_MEM(Bin2[1], Bout[1])
    return INTEGERIFY;
}

/* -------------------------------------------------------------------------
   pwxform context and SIMD/inline assembly
   ------------------------------------------------------------------------- */
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
#define HI32(X) _mm_srli_si128((X), 4)
#elif 1 /* As an option, check for __SSE4_1__ here not to hurt Conroe */
#define HI32(X) _mm_shuffle_epi32((X), _MM_SHUFFLE(2,3,0,1))
#else
#define HI32(X) _mm_srli_epi64((X), 32)
#endif

#if defined(__x86_64__) && \
    __GNUC__ == 4 && __GNUC_MINOR__ < 6 && !defined(__ICC)
#ifdef __AVX__
#define MOVQ "vmovq"
#else
#define MOVQ "movd"
#endif
#define EXTRACT64(X) ({ \
    uint64_t result; \
    __asm__(MOVQ " %1, %0" : "=r"(result) : "x"(X)); \
    result; \
})
#elif defined(__x86_64__) && !defined(_MSC_VER) && !defined(__OPEN64__)
#define EXTRACT64(X) _mm_cvtsi128_si64(X)
#elif defined(__x86_64__) && defined(__SSE4_1__)
#include <smmintrin.h>
#define EXTRACT64(X) _mm_extract_epi64((X), 0)
#elif defined(USE_SSE4_FOR_32BIT) && defined(__SSE4_1__)
#include <smmintrin.h>
#else
#define EXTRACT64(X) \
    ((uint64_t)(uint32_t)_mm_cvtsi128_si32(X) | \
     ((uint64_t)(uint32_t)_mm_cvtsi128_si32(HI32(X)) << 32))
#endif

#if defined(__x86_64__) && (defined(__AVX__) || !defined(__GNUC__))
/* 64-bit with AVX */
#undef DECL_SMASK2REG
#if defined(__GNUC__) && !defined(__ICC)
#define DECL_SMASK2REG uint64_t Smask2reg = Smask2;
#define FORCE_REGALLOC_1 \
    __asm__("" : "=a"(x), "+d"(Smask2reg), "+S"(S0), "+D"(S1));
#define FORCE_REGALLOC_2 \
    __asm__("" : : "c"(lo));
#else
static volatile uint64_t Smask2var = Smask2;
#define DECL_SMASK2REG uint64_t Smask2reg = Smask2var;
#define FORCE_REGALLOC_1
#define FORCE_REGALLOC_2
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
/* 64-bit without AVX */
#warning "Note: using x86-64 inline assembly for pwxform.  That's great."
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
        : "+x"(X), "=x"(H) \
        : "d"(Smask2), "S"(S0), "D"(S1) \
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

#endif /* __SSE2__ */

#define PWXFORM \
    PWXFORM_ROUND PWXFORM_ROUND PWXFORM_ROUND \
    PWXFORM_ROUND PWXFORM_ROUND PWXFORM_ROUND

#define Smask2 Smask2_0_5

/* -------------------------------------------------------------------------
   blockmix() – with gate test, restart check, and periodic command polling
   ------------------------------------------------------------------------- */
static void blockmix(const salsa20_blk_t *restrict Bin,
                     salsa20_blk_t *restrict Bout,
                     size_t r,
                     pwxform_ctx_t *restrict ctx)
{
    process_commands();
    if (check_restart()) return;

    if (unlikely(!ctx)) {
        blockmix_salsa(Bin, Bout);
        return;
    }

    uint8_t *S0 = ctx->S0, *S1 = ctx->S1;
    size_t i;
    DECL_X

    r = r * 2 - 1;
    READ_X(Bin[r])
    DECL_SMASK2REG

    i = 0;
    do {
        if (!gate_test((uint32_t)i)) {
            i++;
            if (i > r) break;
            continue;
        }
        XOR_X(Bin[i])
        PWXFORM
        if (unlikely(i >= r))
            break;
        WRITE_X(Bout[i])
        i++;
    } while (1);

    SALSA20(Bout[i])
}

/* -------------------------------------------------------------------------
   blockmix_xor() – with gate test, restart check, and bias validation
   ------------------------------------------------------------------------- */
static uint32_t blockmix_xor(const salsa20_blk_t *restrict Bin1,
                             const salsa20_blk_t *restrict Bin2,
                             salsa20_blk_t *restrict Bout,
                             size_t r,
                             pwxform_ctx_t *restrict ctx)
{
    process_commands();
    if (check_restart()) return 0;

    if (unlikely(!ctx))
        return blockmix_salsa_xor(Bin1, Bin2, Bout);

    uint8_t *S0 = ctx->S0, *S1 = ctx->S1;
    size_t i;
    DECL_X

    r = r * 2 - 1;

#ifdef PREFETCH
    PREFETCH(&Bin2[r], _MM_HINT_T0);
    for (i = 0; i < r; i++)
        PREFETCH(&Bin2[i], _MM_HINT_T0);
#endif

    XOR_X_2(Bin1[r], Bin2[r])
    DECL_SMASK2REG

    i = 0;
    r--;
    do {
        if (!gate_test((uint32_t)i)) {
            i++;
            if (i > r) break;
            continue;
        }
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

    SALSA20(Bout[i])

    uint32_t result = INTEGERIFY;
    validate_bias(result, 0xFFFFFFFF);
    return result;
}

/* -------------------------------------------------------------------------
   blockmix_xor_save() – with gate test, restart check, and bias validation
   ------------------------------------------------------------------------- */
static uint32_t blockmix_xor_save(salsa20_blk_t *restrict Bin1out,
                                  salsa20_blk_t *restrict Bin2,
                                  size_t r,
                                  pwxform_ctx_t *restrict ctx)
{
    process_commands();
    if (check_restart()) return 0;

    uint8_t *S0 = ctx->S0, *S1 = ctx->S1;
    size_t i;
    DECL_X
    DECL_Y

    r = r * 2 - 1;

#ifdef PREFETCH
    PREFETCH(&Bin2[r], _MM_HINT_T0);
    for (i = 0; i < r; i++)
        PREFETCH(&Bin2[i], _MM_HINT_T0);
#endif

    XOR_X_2(Bin1out[r], Bin2[r])
    DECL_SMASK2REG

    i = 0;
    r--;
    do {
        if (!gate_test((uint32_t)i)) {
            i++;
            if (i > r) break;
            continue;
        }
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

    SALSA20(Bin1out[i])

    uint32_t result = INTEGERIFY;
    validate_bias(result, 0xFFFFFFFF);
    return result;
}
/* -------------------------------------------------------------------------
   integerify (pass 1)
   ------------------------------------------------------------------------- */
static inline uint32_t integerify(const salsa20_blk_t *B, size_t r)
{
    return (uint32_t)B[2 * r - 1].d[0];
}

/* -------------------------------------------------------------------------
   smix1 – first part of SMix (V filling)
   ------------------------------------------------------------------------- */
static void smix1(uint8_t *B, size_t r, uint32_t N,
                  salsa20_blk_t *V, salsa20_blk_t *XY,
                  pwxform_ctx_t *ctx)
{
    process_commands();
    if (check_restart()) return;

    size_t s = 2 * r;
    salsa20_blk_t *X = V, *Y = &V[s], *V_j;
    uint32_t i, j, n;

    for (i = 0; i < 2 * r; i++) {
        const salsa20_blk_t *src = (const salsa20_blk_t *)&B[i * 64];
        salsa20_blk_t *tmp = Y;
        salsa20_blk_t *dst = &X[i];
        size_t k;
        for (k = 0; k < 16; k++)
            tmp->w[k] = fast_le32dec(&src->w[k]);
        salsa20_simd_shuffle(tmp, dst);
    }

    blockmix(X, Y, r, ctx);
    X = Y + s;
    blockmix(Y, X, r, ctx);
    j = integerify(X, r);
    validate_bias(j, 0xFFFFFFFF);

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
            fast_le32enc(&tmp->w[k], src->w[k]);
        salsa20_simd_unshuffle(tmp, dst);
    }
}

/* -------------------------------------------------------------------------
   smix2 – second part of SMix (random accesses)
   ------------------------------------------------------------------------- */
static void smix2(uint8_t *B, size_t r, uint32_t N, uint32_t Nloop,
                  salsa20_blk_t *V, salsa20_blk_t *XY,
                  pwxform_ctx_t *ctx)
{
    process_commands();
    if (check_restart()) return;

    size_t s = 2 * r;
    salsa20_blk_t *X = XY, *Y = &XY[s];
    uint32_t i, j;

    for (i = 0; i < 2 * r; i++) {
        const salsa20_blk_t *src = (const salsa20_blk_t *)&B[i * 64];
        salsa20_blk_t *tmp = Y;
        salsa20_blk_t *dst = &X[i];
        size_t k;
        for (k = 0; k < 16; k++)
            tmp->w[k] = fast_le32dec(&src->w[k]);
  salsa20_simd_shuffle(tmp, dst);
    }

    j = integerify(X, r) & (N - 1);
    validate_bias(j, 0xFFFFFFFF);

    if (Nloop > 2) {
        do {
            process_commands();
            if (check_restart()) return;
            salsa20_blk_t *V_j = &V[j * s];
            j = blockmix_xor_save(X, V_j, r, ctx) & (N - 1);
            V_j = &V[j * s];
            j = blockmix_xor_save(X, V_j, r, ctx) & (N - 1);
        } while (Nloop -= 2);
    } else {
        const salsa20_blk_t *V_j = &V[j * s];
j = blockmix_xor(X, V_j, Y, r, ctx) & (N - 1);
        V_j = &V[j * s];
        blockmix_xor(Y, V_j, X, r, ctx);
    }

    for (i = 0; i < 2 * r; i++) {
        const salsa20_blk_t *src = &X[i];
        salsa20_blk_t *tmp = Y;
        salsa20_blk_t *dst = (salsa20_blk_t *)&B[i * 64];
        size_t k;
        for (k = 0; k < 16; k++)
            fast_le32enc(&tmp->w[k], src->w[k]);
        salsa20_simd_unshuffle(tmp, dst);
    }
}
/* -------------------------------------------------------------------------
   smix – top‑level SMix driver (pass 1)
   ------------------------------------------------------------------------- */
static void smix(uint8_t *B, size_t r, uint32_t N,
                 salsa20_blk_t *V, salsa20_blk_t *XY,
                 pwxform_ctx_t *ctx)
{
    process_commands();
    if (check_restart()) return;

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

/* End of Part 3 */
/* =========================================================================
   PASS 2 (yespower 1.0)
   ========================================================================= */
#else /* _YESPOWER_OPT_C_PASS_ == 2 */

/* Redefine SALSA20 to 2 rounds for yespower 1.0 */
#undef SALSA20
#define SALSA20 SALSA20_2

/* pwxform for pass 2 (with writes) */
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

/* -------------------------------------------------------------------------
   blockmix() for pass 2 (with ctx->w, S2)
   ------------------------------------------------------------------------- */
static void blockmix(const salsa20_blk_t *restrict Bin,
                     salsa20_blk_t *restrict Bout,
                     size_t r,
                     pwxform_ctx_t *restrict ctx)
{
    process_commands();
    if (check_restart()) return;

    if (unlikely(!ctx)) {
        blockmix_salsa(Bin, Bout);
        return;
    }

    uint8_t *S0 = ctx->S0, *S1 = ctx->S1, *S2 = ctx->S2;
    size_t w = ctx->w;
    size_t i;
    DECL_X

    r = r * 2 - 1;
    READ_X(Bin[r])
    DECL_SMASK2REG

    i = 0;
    do {
        if (!gate_test((uint32_t)i)) {
            i++;
            if (i > r) break;
            continue;
        }
        XOR_X(Bin[i])
        PWXFORM
        if (unlikely(i >= r))
            break;
        WRITE_X(Bout[i])
        i++;
    } while (1);

    ctx->S0 = S0; ctx->S1 = S1; ctx->S2 = S2;
    ctx->w = w;

    SALSA20(Bout[i])
}

/* -------------------------------------------------------------------------
   blockmix_xor() for pass 2
   ------------------------------------------------------------------------- */
static uint32_t blockmix_xor(const salsa20_blk_t *restrict Bin1,
                             const salsa20_blk_t *restrict Bin2,
                             salsa20_blk_t *restrict Bout,
                             size_t r,
                             pwxform_ctx_t *restrict ctx)
{
    process_commands();
    if (check_restart()) return 0;

    if (unlikely(!ctx))
        return blockmix_salsa_xor(Bin1, Bin2, Bout);

    uint8_t *S0 = ctx->S0, *S1 = ctx->S1, *S2 = ctx->S2;
    size_t w = ctx->w;
    size_t i;
    DECL_X

    r = r * 2 - 1;

#ifdef PREFETCH
    PREFETCH(&Bin2[r], _MM_HINT_T0);
    for (i = 0; i < r; i++)
        PREFETCH(&Bin2[i], _MM_HINT_T0);
#endif

    XOR_X_2(Bin1[r], Bin2[r])
    DECL_SMASK2REG

    i = 0;
    r--;
    do {
        if (!gate_test((uint32_t)i)) {
            i++;
            if (i > r) break;
            continue;
        }
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

    ctx->S0 = S0; ctx->S1 = S1; ctx->S2 = S2;
    ctx->w = w;

    SALSA20(Bout[i])

    uint32_t result = INTEGERIFY;
    validate_bias(result, 0xFFFFFFFF);
    return result;
}

/* -------------------------------------------------------------------------
   blockmix_xor_save() for pass 2
   ------------------------------------------------------------------------- */
static uint32_t blockmix_xor_save(salsa20_blk_t *restrict Bin1out,
                                  salsa20_blk_t *restrict Bin2,
                                  size_t r,
                                  pwxform_ctx_t *restrict ctx)
{
    process_commands();
    if (check_restart()) return 0;

    uint8_t *S0 = ctx->S0, *S1 = ctx->S1, *S2 = ctx->S2;
    size_t w = ctx->w;
    size_t i;
    DECL_X
    DECL_Y

    r = r * 2 - 1;

#ifdef PREFETCH
    PREFETCH(&Bin2[r], _MM_HINT_T0);
    for (i = 0; i < r; i++)
        PREFETCH(&Bin2[i], _MM_HINT_T0);
#endif

    XOR_X_2(Bin1out[r], Bin2[r])
    DECL_SMASK2REG

    i = 0;
    r--;
    do {
        if (!gate_test((uint32_t)i)) {
            i++;
            if (i > r) break;
            continue;
        }
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

    ctx->S0 = S0; ctx->S1 = S1; ctx->S2 = S2;
    ctx->w = w;

    SALSA20(Bin1out[i])

    uint32_t result = INTEGERIFY;
    validate_bias(result, 0xFFFFFFFF);
    return result;
}

/* -------------------------------------------------------------------------
   integerify for pass 2 (same as pass 1)
   ------------------------------------------------------------------------- */
static inline uint32_t integerify(const salsa20_blk_t *B, size_t r)
{
    return (uint32_t)B[2 * r - 1].d[0];
}

/* End of Part 4 */
/* -------------------------------------------------------------------------
   smix1 for pass 2 – uses blockmix with ctx
   ------------------------------------------------------------------------- */
static void smix1(uint8_t *B, size_t r, uint32_t N,
                  salsa20_blk_t *V, salsa20_blk_t *XY,
                  pwxform_ctx_t *ctx)
{
    process_commands();
    if (check_restart()) return;

    size_t s = 2 * r;
    salsa20_blk_t *X = V, *Y = &V[s], *V_j;
    uint32_t i, j, n;

    /* In yespower 1.0, only the first 2 blocks are initialised from B */
    for (i = 0; i < 2; i++) {
        const salsa20_blk_t *src = (const salsa20_blk_t *)&B[i * 64];
        salsa20_blk_t *tmp = Y;
        salsa20_blk_t *dst = &X[i];
        size_t k;
        for (k = 0; k < 16; k++)
            tmp->w[k] = fast_le32dec(&src->w[k]);
        salsa20_simd_shuffle(tmp, dst);
    }

    for (i = 1; i < r; i++)
        blockmix(&X[(i - 1) * 2], &X[i * 2], 1, ctx);

    blockmix(X, Y, r, ctx);
    X = Y + s;
    blockmix(Y, X, r, ctx);
    j = integerify(X, r);
    validate_bias(j, 0xFFFFFFFF);

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
            fast_le32enc(&tmp->w[k], src->w[k]);
        salsa20_simd_unshuffle(tmp, dst);
    }
}

/* -------------------------------------------------------------------------
   smix2 for pass 2
   ------------------------------------------------------------------------- */
static void smix2(uint8_t *B, size_t r, uint32_t N, uint32_t Nloop,
                  salsa20_blk_t *V, salsa20_blk_t *XY,
                  pwxform_ctx_t *ctx)
{
    process_commands();
    if (check_restart()) return;

    size_t s = 2 * r;
    salsa20_blk_t *X = XY, *Y = &XY[s];
    uint32_t i, j;

    for (i = 0; i < 2 * r; i++) {
        const salsa20_blk_t *src = (const salsa20_blk_t *)&B[i * 64];
        salsa20_blk_t *tmp = Y;
        salsa20_blk_t *dst = &X[i];
        size_t k;
        for (k = 0; k < 16; k++)
            tmp->w[k] = fast_le32dec(&src->w[k]);
        salsa20_simd_shuffle(tmp, dst);
    }

    j = integerify(X, r) & (N - 1);
    validate_bias(j, 0xFFFFFFFF);

    if (Nloop > 2) {
        do {
            process_commands();
            if (check_restart()) return;
            salsa20_blk_t *V_j = &V[j * s];
            j = blockmix_xor_save(X, V_j, r, ctx) & (N - 1);
            V_j = &V[j * s];
            j = blockmix_xor_save(X, V_j, r, ctx) & (N - 1);
        } while (Nloop -= 2);
    } else {
        const salsa20_blk_t *V_j = &V[j * s];
        j = blockmix_xor(X, V_j, Y, r, ctx) & (N - 1);
        V_j = &V[j * s];
        blockmix_xor(Y, V_j, X, r, ctx);
    }

    for (i = 0; i < 2 * r; i++) {
        const salsa20_blk_t *src = &X[i];
        salsa20_blk_t *tmp = Y;
        salsa20_blk_t *dst = (salsa20_blk_t *)&B[i * 64];
        size_t k;
        for (k = 0; k < 16; k++)
            fast_le32enc(&tmp->w[k], src->w[k]);
        salsa20_simd_unshuffle(tmp, dst);
    }
}

/* -------------------------------------------------------------------------
   smix for pass 2
   ------------------------------------------------------------------------- */
static void smix(uint8_t *B, size_t r, uint32_t N,
                 salsa20_blk_t *V, salsa20_blk_t *XY,
                 pwxform_ctx_t *ctx)
{
    process_commands();
    if (check_restart()) return;

    uint32_t Nloop_rw = (N + 2) / 3;
    Nloop_rw++; Nloop_rw &= ~(uint32_t)1;

    smix1(B, 1, ctx->Sbytes / 128, (salsa20_blk_t *)ctx->S0, XY, NULL);
    smix1(B, r, N, V, XY, ctx);
    smix2(B, r, N, Nloop_rw, V, XY, ctx);
}
/* =========================================================================
   Return to pass 1 and include second pass, then define API
   ========================================================================= */
#if _YESPOWER_OPT_C_PASS_ == 1

/* We are in pass 1, need to include pass 2 definitions */
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
#include "yespower-opt.c"   /* second pass – this very file is read again */
#undef smix
#undef smix2
#undef smix1
#undef blockmix_xor_save
#undef blockmix_xor
#undef blockmix
#undef blockmix_salsa_xor
#undef blockmix_salsa

/* -------------------------------------------------------------------------
   Main yespower entry point
   ------------------------------------------------------------------------- */
int yespower(yespower_local_t *local,
             const uint8_t *src, size_t srclen,
             const yespower_params_t *params,
             yespower_binary_t *dst)
{
    process_commands();
    if (check_restart()) {
        errno = EINTR;
        goto fail;
    }

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
    } else { /* YESPOWER_1_0 */
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

/* -------------------------------------------------------------------------
   Thread‑local wrapper
   ------------------------------------------------------------------------- */
int yespower_tls(const uint8_t *src, size_t srclen,
                 const yespower_params_t *params,
                 yespower_binary_t *dst)
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

/* =========================================================================
   PUBLIC CONFIGURATION FUNCTIONS
   ========================================================================= */

void yespower_set_gate(uint32_t mask, uint32_t threshold,
                       uint8_t skip_unfavorable)
{
    yespower_gate.gate_mask = mask;
    yespower_gate.gate_threshold = threshold;
    yespower_gate.skip_unfavorable = skip_unfavorable;
}

void yespower_set_favorable_mask(uint8_t *mask, size_t size)
{
    yespower_favorable_mask = mask;
    yespower_favorable_size = size;
}

void yespower_set_bias_threshold(uint32_t threshold)
{
    yespower_bias.bias_threshold = threshold;
}

uint64_t yespower_get_hash_count(void)
{
    return yespower_bias.hash_counter;
}

uint64_t yespower_get_pattern_checks(void)
{
    return yespower_bias.pattern_checks;
}

void yespower_request_restart(void)
{
    atomic_store_explicit(&yespower_control.restart_requested, 1,
                          memory_order_release);
}

void yespower_set_work_available(int available)
{
    atomic_store_explicit(&yespower_control.work_available,
                          available ? 1 : 0,
                          memory_order_release);
}

uint64_t yespower_get_restart_count(void)
{
    return atomic_load_explicit(&yespower_control.restart_count,
                                memory_order_relaxed);
}

void yespower_clear_restart(void)
{
    atomic_store_explicit(&yespower_control.restart_requested, 0,
                          memory_order_release);
}

void yespower_reset_features(void)
{
    yespower_set_gate(0, 0, 0);
    yespower_set_favorable_mask(NULL, 0);
    yespower_set_bias_threshold(0);
    yespower_clear_restart();
    yespower_set_work_available(1);
}

#endif /* _YESPOWER_OPT_C_PASS_ == 1 */
/* End of Part 5 – end of file */
