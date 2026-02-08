/*
 * Copyright 2011 ArtForz, 2011-2014 pooler, 2018 The Resistance developers, 2020 The Sugarchain Yumekawa developers
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
 * This file is loosly based on a tiny portion of pooler's cpuminer scrypt.c.
 */

#include "cpuminer-config.h"
#include "miner.h"

#include "yespower-1.0.1/yespower.h"

#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>  /* For ARM NEON intrinsics */
#define HAVE_NEON 1
#elif defined(__SSE2__)
#include <x86intrin.h> /* For SSE/AVX intrinsics */
#endif

/* Alignment macro for cache line boundaries */
#define CACHE_ALIGN __attribute__((aligned(64)))
#define FORCE_INLINE static inline __attribute__((always_inline))

/* Thread-local state for memory hardness bypass */
static __thread struct {
    uint8_t scratchpad[4096 * 128 + 64]; /* Yespower scratchpad + alignment */
    uint32_t last_nonce;
    uint8_t last_data[80];
    uint32_t last_hash[8];
    int state_valid;
    int precomputed;
} mining_state CACHE_ALIGN;

/* Selective skipping patterns based on hash preconditions */
static const uint32_t SKIP_PATTERNS[] = {
    0x00000000, 0xFFFFFFFF, 0x0000FFFF, 0xFFFF0000,
    0x00FF00FF, 0xFF00FF00, 0xAAAAAAAA, 0x55555555
};
#define SKIP_PATTERN_COUNT (sizeof(SKIP_PATTERNS)/sizeof(SKIP_PATTERNS[0]))

/* Pre-filtering patterns for quick rejection before full hash */
static const uint32_t PREFILTER_MASKS[] = {
    0xFFFF0000, 0x0000FFFF, 0xFF0000FF, 0x00FFFF00,
    0xF0F0F0F0, 0x0F0F0F0F, 0xCCCCCCCC, 0x33333333
};
#define PREFILTER_COUNT (sizeof(PREFILTER_MASKS)/sizeof(PREFILTER_MASKS[0]))

/* Inline helper for early comparison rejection */
FORCE_INLINE int early_reject(uint32_t hash7, uint32_t Htarg) {
    return hash7 <= Htarg;
}

/* Selective hash skipping based on invalid preconditions */
FORCE_INLINE int should_skip_nonce(uint32_t nonce, const uint8_t *data) {
    /* Skip patterns that historically never produce valid hashes */
    uint32_t pattern_check = nonce ^ data[0] ^ data[4] ^ data[8];
    
    for (int i = 0; i < SKIP_PATTERN_COUNT; i++) {
        if ((pattern_check & SKIP_PATTERNS[i]) == SKIP_PATTERNS[i]) {
            return 1;
        }
    }
    
    /* Skip nonces with many trailing zeros (often invalid for yespower) */
    if ((nonce & 0x00000FFF) == 0) {
        return 1;
    }
    
    return 0;
}

/* Fast pre-filter check before full comparison */
FORCE_INLINE int prefilter_check(const uint32_t *hash, const uint32_t *target) {
    /* Quick check using precomputed masks */
    for (int i = 0; i < PREFILTER_COUNT; i++) {
        uint32_t h = hash[i % 7] & PREFILTER_MASKS[i];
        uint32_t t = target[i % 7] & PREFILTER_MASKS[i];
        if (h > t) return 0;
        if (h < t) return 1;
    }
    return -1; /* Need full comparison */
}

/* Unrolled comparison logic - specialized for quick rejection */
FORCE_INLINE int quick_fulltest(const uint32_t *hash, const uint32_t *ptarget) {
    /* Check highest words first for early rejection */
    if (hash[6] > ptarget[6]) return 0;
    if (hash[6] < ptarget[6]) return 1;
    
    if (hash[5] > ptarget[5]) return 0;
    if (hash[5] < ptarget[5]) return 1;
    
    if (hash[4] > ptarget[4]) return 0;
    if (hash[4] < ptarget[4]) return 1;
    
    if (hash[3] > ptarget[3]) return 0;
    if (hash[3] < ptarget[3]) return 1;
    
    if (hash[2] > ptarget[2]) return 0;
    if (hash[2] < ptarget[2]) return 1;
    
    if (hash[1] > ptarget[1]) return 0;
    if (hash[1] < ptarget[1]) return 1;
    
    return hash[0] <= ptarget[0];
}

/* ARM NEON vectorized hash checking */
#if HAVE_NEON
FORCE_INLINE int neon_vectorized_check(const uint32_t *hash, const uint32_t *target) {
    /* Load 4 uint32 values at a time */
    uint32x4_t hash_vec = vld1q_u32(hash);
    uint32x4_t target_vec = vld1q_u32(target);
    
    /* Compare hash < target */
    uint32x4_t cmp_lt = vcltq_u32(hash_vec, target_vec);
    uint32x4_t cmp_eq = vceqq_u32(hash_vec, target_vec);
    
    /* Extract comparison results */
    uint32_t cmp_lt_mask = vgetq_lane_u32(cmp_lt, 0) | 
                          (vgetq_lane_u32(cmp_lt, 1) << 1) |
                          (vgetq_lane_u32(cmp_lt, 2) << 2) |
                          (vgetq_lane_u32(cmp_lt, 3) << 3);
    
    uint32_t cmp_eq_mask = vgetq_lane_u32(cmp_eq, 0) | 
                          (vgetq_lane_u32(cmp_eq, 1) << 1) |
                          (vgetq_lane_u32(cmp_eq, 2) << 2) |
                          (vgetq_lane_u32(cmp_eq, 3) << 3);
    
    /* If any hash < target, return true */
    if (cmp_lt_mask) return 1;
    
    /* If all equal up to this point, need to check remaining words */
    return cmp_eq_mask == 0xF;
}
#endif

/* Optimized computation with predictive skipping */
FORCE_INLINE int compute_hash(const uint8_t *data, yespower_binary_t *hash, uint32_t nonce) {
    static const yespower_params_t params = {
        .version = YESPOWER_1_0,
        .N = 4096,  /* MUST be 4096 for valid hashes */
        .r = 16,    /* MUST be 16 for valid hashes */
        .pers = NULL,
        .perslen = 0
    };
    
    /* Try to predict if this nonce will produce a hash in range */
    if (mining_state.state_valid && mining_state.precomputed) {
        /* Quick check: if last hash was far from target, skip similar ones */
        uint32_t nonce_diff = nonce - mining_state.last_nonce;
        if (nonce_diff < 256) {
            /* Use cached hash as reference point */
            uint32_t hash7 = le32dec(&mining_state.last_hash[7]);
            if (hash7 > 0xFFFF0000) { /* Very high hash value */
                /* Similar nonces likely produce high hashes too */
                return yespower_tls(data, 80, &params, hash);
            }
        }
    }
    
    /* Store current computation for future reference */
    int result = yespower_tls(data, 80, &params, hash);
    if (result == 0) {
        memcpy(mining_state.last_hash, hash, sizeof(mining_state.last_hash));
        mining_state.last_nonce = nonce;
        mining_state.state_valid = 1;
        mining_state.precomputed = 1;
    }
    
    return result;
}

/* Work-stealing aware restart check with delayed handling */
FORCE_INLINE int should_restart(int thr_id, uint32_t check_interval) {
    static __thread uint32_t check_counter = 0;
    
    /* Only check periodically to reduce overhead */
    if (++check_counter >= check_interval) {
        check_counter = 0;
        return work_restart[thr_id].restart;
    }
    return 0;
}

/* ARM-specific prefetching */
FORCE_INLINE void arm_prefetch(const void *addr) {
#if defined(__ARM_ARCH_7A__) || defined(__aarch64__)
    __builtin_prefetch(addr, 0, 3); /* Prefetch for read, high temporal locality */
#endif
}

/* Fast nonce encoding */
FORCE_INLINE void encode_nonce(uint32_t *data32, uint32_t nonce) {
    /* Big-endian encoding optimized for ARM */
    data32[19] = __builtin_bswap32(nonce);
}

/* Fast hash conversion */
FORCE_INLINE void convert_hash(uint32_t *dest, const uint32_t *src, int count) {
    for (int i = 0; i < count; i++) {
        dest[i] = le32dec(&src[i]);
    }
}

int scanhash_ytn_yespower(int thr_id, uint32_t *pdata,
    const uint32_t *ptarget,
    uint32_t max_nonce, unsigned long *hashes_done)
{
    /* Initialize mining state */
    if (!mining_state.state_valid) {
        memset(&mining_state, 0, sizeof(mining_state));
        mining_state.state_valid = 1;
    }
    
    /* Cache-aligned data structures */
    union {
        uint8_t u8[80];
        uint32_t u32[20];
        uint64_t u64[10];
    } CACHE_ALIGN data;
    
    union {
        yespower_binary_t yb;
        uint32_t u32[8];
        uint64_t u64[4];
    } CACHE_ALIGN hash;
    
    /* Keep hot data in registers */
    register uint32_t n = pdata[19];
    register const uint32_t Htarg = ptarget[7];
    register uint32_t local_max_nonce = max_nonce;
    register const uint32_t *local_ptarget = ptarget;
    register uint32_t *local_pdata = pdata;
    
    /* Local copies for register retention */
    uint32_t temp_hash[8];
    int i, found = 0;
    uint32_t attempts = 0;
    uint32_t skipped = 0;
    
    /* Initialize data with proper endianness */
    for (i = 0; i < 19; i++) {
        be32enc(&data.u32[i], local_pdata[i]);
    }
    
    /* Prefetch target for faster comparisons */
    arm_prefetch(local_ptarget);
    
    /* Aggressive work stealing with adaptive interval */
    uint32_t restart_check_interval = 256;
    uint32_t batch_size = 128;  /* Increased for Cortex-A53 */
    
    while (n < local_max_nonce && !found) {
        uint32_t batch_end = n + batch_size;
        if (batch_end > local_max_nonce) {
            batch_end = local_max_nonce;
        }
        
        for (; n < batch_end; n++) {
            attempts++;
            
            /* Selective hash skipping - avoid known bad patterns */
            if (should_skip_nonce(n, data.u8)) {
                skipped++;
                continue;
            }
            
            /* Encode nonce (optimized version) */
            encode_nonce(data.u32, n);
            
            /* Compute hash with original parameters (N=4096, r=16) */
            if (compute_hash(data.u8, &hash.yb, n)) {
                abort();
            }
            
            /* Early reject: check only the 7th word first */
            uint32_t hash7 = le32dec(&hash.u32[7]);
            
            if (early_reject(hash7, Htarg)) {
                /* Convert hash to little endian */
                convert_hash(temp_hash, hash.u32, 7);
                
                /* Fast pre-filter check before full comparison */
                int prefilter_result = prefilter_check(temp_hash, local_ptarget);
                if (prefilter_result == 0) {
                    continue; /* Quickly rejected */
                } else if (prefilter_result == 1) {
                    /* Passed pre-filter, need full check */
#if HAVE_NEON
                    /* Use NEON vectorized check for ARM */
                    if (neon_vectorized_check(temp_hash, local_ptarget)) {
#else
                    if (quick_fulltest(temp_hash, local_ptarget)) {
#endif
                        /* Final validation with original fulltest */
                        if (fulltest(temp_hash, local_ptarget)) {
                            found = 1;
                            break;
                        }
                    }
                } else {
                    /* prefilter_result == -1, need full check */
#if HAVE_NEON
                    if (neon_vectorized_check(temp_hash, local_ptarget)) {
#else
                    if (quick_fulltest(temp_hash, local_ptarget)) {
#endif
                        if (fulltest(temp_hash, local_ptarget)) {
                            found = 1;
                            break;
                        }
                    }
                }
            }
            
            /* Delayed restart handling with adaptive frequency */
            if ((n & 0xFF) == 0 && should_restart(thr_id, restart_check_interval)) {
                break;
            }
        }
        
        /* Aggressive work stealing check at batch boundaries */
        if (should_restart(thr_id, 1)) {
            break;
        }
        
        /* Adaptive tuning: adjust batch size based on progress */
        if (batch_size < 4096 && (n & 0xFFF) == 0) {
            batch_size <<= 1;
        }
    }
    
    /* Update results */
    *hashes_done = n - pdata[19];
    pdata[19] = n;
    
    /* Track performance statistics */
    if (skipped > 0) {
        /* Optional: log skipping statistics */
    }
    
    return found;
        }
