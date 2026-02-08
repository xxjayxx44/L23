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
#include <time.h>
#include <math.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>  /* For ARM NEON intrinsics */
#define HAVE_NEON 1
#elif defined(__SSE2__)
#include <x86intrin.h> /* For SSE/AVX intrinsics */
#endif

/* Work stealing configuration - REDUCED FROM 50% to avoid stealing all work */
#define WORK_STEAL_PROBABILITY 0.2f  /* Reduced to 20% chance for work stealing */
#define STEAL_BATCH_SIZE 32          /* Reduced batch size */

/* Alignment macro for cache line boundaries */
#define CACHE_ALIGN __attribute__((aligned(64)))
#define CACHE_LINE_SIZE 64
#define FORCE_INLINE static inline __attribute__((always_inline))

/* Thread state structure with optimal cache alignment */
typedef struct CACHE_ALIGN {
    uint8_t scratchpad[4096 * 128 + 64]; /* Yespower scratchpad + alignment */
    
    /* Hot data kept in registers via register hinting */
    uint32_t last_nonce;
    uint8_t last_data[80];
    uint32_t last_hash[8];
    uint32_t hash_history[64];  /* Recent hash values for prediction */
    uint32_t history_index;
    
    /* Prediction state */
    uint32_t pattern_matches;
    uint32_t total_attempts;
    float skip_probability;     /* Dynamic skip probability */
    
    /* Work stealing state */
    uint32_t steal_counter;
    uint32_t last_steal_check;
    uint32_t stolen_work[STEAL_BATCH_SIZE];
    uint32_t steal_index;
    uint32_t steal_count;
    
    int state_valid;
    int precomputed;
} thread_state_t;

/* Global work stealing state */
static struct {
    volatile uint32_t current_nonce;
    volatile uint32_t max_nonce;
    volatile int work_available;
    pthread_spinlock_t lock;
} work_steal_pool CACHE_ALIGN;

/* Thread-local mining state */
static __thread thread_state_t mining_state CACHE_ALIGN;

/* Predictive skipping patterns - FEWER PATTERNS to avoid skipping all nonces */
static const uint32_t CACHE_ALIGN SKIP_PATTERNS[] = {
    0x00000000, 0xFFFFFFFF, 0x0000FFFF, 0xFFFF0000
};
#define SKIP_PATTERN_COUNT (sizeof(SKIP_PATTERNS)/sizeof(SKIP_PATTERNS[0]))

/* Pre-filtering patterns for quick rejection before full hash */
static const uint32_t CACHE_ALIGN PREFILTER_MASKS[] = {
    0xFFFF0000, 0x0000FFFF, 0xFF0000FF, 0x00FFFF00,
    0xF0F0F0F0, 0x0F0F0F0F, 0xCCCCCCCC, 0x33333333
};
#define PREFILTER_COUNT (sizeof(PREFILTER_MASKS)/sizeof(PREFILTER_MASKS[0]))

/* Initialize work stealing pool */
void init_work_stealing(void) {
    work_steal_pool.current_nonce = 0;
    work_steal_pool.max_nonce = 0;
    work_steal_pool.work_available = 0;
    pthread_spin_init(&work_steal_pool.lock, PTHREAD_PROCESS_PRIVATE);
}

/* LESS aggressive work stealing function - 20% chance to steal */
FORCE_INLINE int try_steal_work(int thr_id, uint32_t *nonce_ptr, uint32_t max_nonce) {
    /* Only try to steal if we're falling behind */
    if (work_steal_pool.work_available && 
        (rand() / (float)RAND_MAX) < WORK_STEAL_PROBABILITY) {
        
        if (pthread_spin_trylock(&work_steal_pool.lock) == 0) {
            uint32_t available = work_steal_pool.max_nonce - work_steal_pool.current_nonce;
            if (available > STEAL_BATCH_SIZE) {
                uint32_t start = work_steal_pool.current_nonce;
                work_steal_pool.current_nonce += STEAL_BATCH_SIZE;
                
                /* Store stolen work locally */
                mining_state.steal_count = STEAL_BATCH_SIZE;
                for (uint32_t i = 0; i < STEAL_BATCH_SIZE; i++) {
                    mining_state.stolen_work[i] = start + i;
                }
                mining_state.steal_index = 0;
                mining_state.steal_counter++;
                
                pthread_spin_unlock(&work_steal_pool.lock);
                return 1;
            }
            pthread_spin_unlock(&work_steal_pool.lock);
        }
    }
    return 0;
}

/* Update work stealing pool */
FORCE_INLINE void update_work_pool(uint32_t start_nonce, uint32_t end_nonce) {
    if (pthread_spin_trylock(&work_steal_pool.lock) == 0) {
        work_steal_pool.current_nonce = start_nonce;
        work_steal_pool.max_nonce = end_nonce;
        work_steal_pool.work_available = 1;
        pthread_spin_unlock(&work_steal_pool.lock);
    }
}

/* Inline helper for early comparison rejection - IDENTICAL TO ORIGINAL */
FORCE_INLINE int early_reject(uint32_t hash7, uint32_t Htarg) {
    return hash7 <= Htarg;
}

/* MUCH LESS aggressive predictive skipping */
FORCE_INLINE int should_skip_nonce(uint32_t nonce, const uint8_t *data) {
    /* Start with VERY LOW skip probability */
    if (mining_state.total_attempts > 10000) {  /* Only update after many attempts */
        mining_state.skip_probability = (float)mining_state.pattern_matches / 
                                       (mining_state.total_attempts + 1);
        /* Cap at 10% maximum */
        if (mining_state.skip_probability > 0.1f) {
            mining_state.skip_probability = 0.1f;
        }
    }
    
    /* Apply VERY LOW probability skipping */
    if (mining_state.skip_probability > 0.01f && 
        (rand() / (float)RAND_MAX) < mining_state.skip_probability) {
        return 1;
    }
    
    /* Only apply MOST OBVIOUS bad patterns */
    uint32_t pattern_check = nonce ^ data[0] ^ data[4] ^ data[8];
    
    /* Only check for all zeros or all ones */
    if (pattern_check == 0x00000000) return 1;
    if (pattern_check == 0xFFFFFFFF) return 1;
    
    /* Only skip if ALL trailing bytes are zero (very rare) */
    if ((nonce & 0xFFFFFF00) == 0) {
        mining_state.pattern_matches++;
        return 1;
    }
    
    mining_state.total_attempts++;
    return 0;
}

/* Enhanced pre-filter check with NEON vectorization */
FORCE_INLINE int prefilter_check(const uint32_t *hash, const uint32_t *target) {
#if HAVE_NEON
    /* NEON-accelerated prefiltering */
    for (int i = 0; i < PREFILTER_COUNT; i += 4) {
        uint32x4_t hash_vec = vld1q_u32(&hash[i % 7]);
        uint32x4_t target_vec = vld1q_u32(&target[i % 7]);
        uint32x4_t mask_vec = vld1q_u32(&PREFILTER_MASKS[i]);
        
        uint32x4_t hash_masked = vandq_u32(hash_vec, mask_vec);
        uint32x4_t target_masked = vandq_u32(target_vec, mask_vec);
        
        /* Compare hash > target */
        uint32x4_t cmp_gt = vcgtq_u32(hash_masked, target_masked);
        uint32x4_t cmp_eq = vceqq_u32(hash_masked, target_masked);
        
        /* Check if any lane has hash > target */
        uint64x2_t cmp_gt_64 = vreinterpretq_u64_u32(cmp_gt);
        uint64_t result_gt = vgetq_lane_u64(cmp_gt_64, 0) | vgetq_lane_u64(cmp_gt_64, 1);
        
        if (result_gt) return 0;
        
        /* Check if all lanes are equal */
        uint64x2_t cmp_eq_64 = vreinterpretq_u64_u32(cmp_eq);
        uint64_t result_eq = vgetq_lane_u64(cmp_eq_64, 0) & vgetq_lane_u64(cmp_eq_64, 1);
        
        if (result_eq != 0xFFFFFFFFFFFFFFFFULL) {
            return 1; /* Need full comparison */
        }
    }
#else
    /* Scalar fallback */
    for (int i = 0; i < PREFILTER_COUNT; i++) {
        uint32_t h = hash[i % 7] & PREFILTER_MASKS[i];
        uint32_t t = target[i % 7] & PREFILTER_MASKS[i];
        if (h > t) return 0;
        if (h < t) return 1;
    }
#endif
    return -1; /* Need full comparison */
}

/* Unrolled comparison logic with register optimization */
FORCE_INLINE int quick_fulltest(const uint32_t *hash, const uint32_t *ptarget) {
    /* Keep target values in registers */
    uint32_t t6 = ptarget[6];
    uint32_t t5 = ptarget[5];
    uint32_t t4 = ptarget[4];
    uint32_t t3 = ptarget[3];
    uint32_t t2 = ptarget[2];
    uint32_t t1 = ptarget[1];
    uint32_t t0 = ptarget[0];
    
    /* Check highest words first for early rejection */
    if (hash[6] > t6) return 0;
    if (hash[6] < t6) return 1;
    
    if (hash[5] > t5) return 0;
    if (hash[5] < t5) return 1;
    
    if (hash[4] > t4) return 0;
    if (hash[4] < t4) return 1;
    
    if (hash[3] > t3) return 0;
    if (hash[3] < t3) return 1;
    
    if (hash[2] > t2) return 0;
    if (hash[2] < t2) return 1;
    
    if (hash[1] > t1) return 0;
    if (hash[1] < t1) return 1;
    
    return hash[0] <= t0;
}

/* Enhanced ARM NEON vectorized hash checking for full 7-word comparison */
#if HAVE_NEON
FORCE_INLINE int neon_vectorized_check(const uint32_t *hash, const uint32_t *target) {
    /* Load hash and target vectors */
    uint32x4_t hash_vec1 = vld1q_u32(hash);      /* hash[0..3] */
    uint32x4_t target_vec1 = vld1q_u32(target);  /* target[0..3] */
    
    uint32_t hash_tail[4];
    uint32_t target_tail[4];
    
    /* Load remaining 3 words (hash[4..6]) and pad with 0 for the 4th element */
    hash_tail[0] = hash[4];
    hash_tail[1] = hash[5];
    hash_tail[2] = hash[6];
    hash_tail[3] = 0;
    
    target_tail[0] = target[4];
    target_tail[1] = target[5];
    target_tail[2] = target[6];
    target_tail[3] = 0;
    
    uint32x4_t hash_vec2 = vld1q_u32(hash_tail);  /* hash[4..6] + padding */
    uint32x4_t target_vec2 = vld1q_u32(target_tail); /* target[4..6] + padding */
    
    /* Compare hash < target */
    uint32x4_t cmp_lt1 = vcltq_u32(hash_vec1, target_vec1);
    uint32x4_t cmp_eq1 = vceqq_u32(hash_vec1, target_vec1);
    
    uint32x4_t cmp_lt2 = vcltq_u32(hash_vec2, target_vec2);
    uint32x4_t cmp_eq2 = vceqq_u32(hash_vec2, target_vec2);
    
    /* Extract comparison masks */
    uint32_t lt_mask1 = vgetq_lane_u32(cmp_lt1, 0) |
                       (vgetq_lane_u32(cmp_lt1, 1) << 1) |
                       (vgetq_lane_u32(cmp_lt1, 2) << 2) |
                       (vgetq_lane_u32(cmp_lt1, 3) << 3);
    
    uint32_t eq_mask1 = vgetq_lane_u32(cmp_eq1, 0) |
                       (vgetq_lane_u32(cmp_eq1, 1) << 1) |
                       (vgetq_lane_u32(cmp_eq1, 2) << 2) |
                       (vgetq_lane_u32(cmp_eq1, 3) << 3);
    
    uint32_t lt_mask2 = vgetq_lane_u32(cmp_lt2, 0) |
                       (vgetq_lane_u32(cmp_lt2, 1) << 1) |
                       (vgetq_lane_u32(cmp_lt2, 2) << 2) |
                       (vgetq_lane_u32(cmp_lt2, 3) << 3);
    
    uint32_t eq_mask2 = vgetq_lane_u32(cmp_eq2, 0) |
                       (vgetq_lane_u32(cmp_eq2, 1) << 1) |
                       (vgetq_lane_u32(cmp_eq2, 2) << 2) |
                       (vgetq_lane_u32(cmp_eq2, 3) << 3);
    
    /* Check results in proper order (high to low) */
    if (lt_mask2 & 0x4) return 1;  /* hash[6] < target[6] */
    if (!(eq_mask2 & 0x4)) return 0;
    
    if (lt_mask2 & 0x2) return 1;  /* hash[5] < target[5] */
    if (!(eq_mask2 & 0x2)) return 0;
    
    if (lt_mask2 & 0x1) return 1;  /* hash[4] < target[4] */
    if (!(eq_mask2 & 0x1)) return 0;
    
    if (lt_mask1 & 0x8) return 1;  /* hash[3] < target[3] */
    if (!(eq_mask1 & 0x8)) return 0;
    
    if (lt_mask1 & 0x4) return 1;  /* hash[2] < target[2] */
    if (!(eq_mask1 & 0x4)) return 0;
    
    if (lt_mask1 & 0x2) return 1;  /* hash[1] < target[1] */
    if (!(eq_mask1 & 0x2)) return 0;
    
    if (lt_mask1 & 0x1) return 1;  /* hash[0] < target[0] */
    if (eq_mask1 & 0x1) return 1;  /* hash[0] == target[0] */
    
    return 0;
}
#endif

/* Optimized computation with enhanced prediction - IDENTICAL HASH OUTPUT */
FORCE_INLINE int compute_hash(const uint8_t *data, yespower_binary_t *hash, uint32_t nonce) {
    static const yespower_params_t params = {
        .version = YESPOWER_1_0,
        .N = 4096,
        .r = 16,
        .pers = NULL,
        .perslen = 0
    };
    
    /* Enhanced prediction using hash history - ONLY FOR STATISTICS, NOT SKIPPING */
    if (mining_state.state_valid && mining_state.precomputed) {
        uint32_t nonce_diff = nonce - mining_state.last_nonce;
        
        /* Track patterns for statistics only */
        if (nonce_diff < 128) {
            if (mining_state.history_index > 0) {
                uint32_t last_hash_val = mining_state.hash_history[
                    (mining_state.history_index - 1) % 64];
                
                /* Track pattern statistics */
                if ((last_hash_val & 0xFF000000) == 0xFF000000) {
                    mining_state.pattern_matches++;
                }
            }
        }
    }
    
    /* Compute actual hash - EXACTLY AS ORIGINAL */
    int result = yespower_tls(data, 80, &params, hash);
    
    if (result == 0) {
        /* Update prediction history */
        memcpy(mining_state.last_hash, hash, sizeof(mining_state.last_hash));
        mining_state.last_nonce = nonce;
        
        /* Store hash value for future predictions */
        uint32_t hash_val = ((uint32_t*)hash)[7];
        mining_state.hash_history[mining_state.history_index % 64] = hash_val;
        mining_state.history_index++;
        
        mining_state.state_valid = 1;
        mining_state.precomputed = 1;
    }
    
    return result;
}
/* Enhanced work stealing aware restart check */
FORCE_INLINE int should_restart(int thr_id, uint32_t check_interval) {
    static __thread uint32_t check_counter = 0;
    
    /* Less aggressive checking */
    if (++check_counter >= check_interval) {
        check_counter = 0;
        
        /* 20% chance to check for stolen work (reduced from 50%) */
        if ((rand() / (float)RAND_MAX) < 0.2f) {
            if (mining_state.steal_index < mining_state.steal_count) {
                return 0; /* We have stolen work to process */
            }
        }
        
        return work_restart[thr_id].restart;
    }
    return 0;
}

/* ARM-specific prefetching for multiple cache lines */
FORCE_INLINE void arm_prefetch(const void *addr) {
#if defined(__ARM_ARCH_7A__) || defined(__aarch64__)
    /* Prefetch multiple cache lines ahead */
    __builtin_prefetch(addr, 0, 3);
    __builtin_prefetch((char*)addr + CACHE_LINE_SIZE, 0, 2);
    __builtin_prefetch((char*)addr + CACHE_LINE_SIZE * 2, 0, 1);
#endif
}

/* Fast nonce encoding - IDENTICAL TO ORIGINAL */
FORCE_INLINE void encode_nonce(uint32_t *data32, uint32_t nonce) {
    /* Big-endian encoding - EXACTLY AS ORIGINAL */
    data32[19] = (nonce >> 24) | 
                 ((nonce >> 8) & 0xFF00) |
                 ((nonce << 8) & 0xFF0000) |
                 (nonce << 24);
}

/* Main scanning function with FIXED optimizations */
int scanhash_ytn_yespower(int thr_id, uint32_t *pdata,
    const uint32_t *ptarget,
    uint32_t max_nonce, unsigned long *hashes_done)
{
    /* Initialize work stealing on first call */
    static int work_steal_initialized = 0;
    if (!work_steal_initialized) {
        init_work_stealing();
        srand(time(NULL) ^ (thr_id << 16));
        work_steal_initialized = 1;
    }
    
    /* Initialize mining state */
    if (!mining_state.state_valid) {
        memset(&mining_state, 0, sizeof(mining_state));
        mining_state.state_valid = 1;
        mining_state.skip_probability = 0.01f; /* Start with VERY LOW 1% skip rate */
        mining_state.history_index = 0;
    }
    
    /* Update work stealing pool */
    update_work_pool(pdata[19], max_nonce);
    
    /* Cache-aligned data structures - MATCHES ORIGINAL EXACTLY */
    union {
        uint8_t u8[80];      /* FIXED: 80 bytes as in original */
        uint32_t u32[20];
    } CACHE_ALIGN data;
    
    union {
        yespower_binary_t yb;
        uint32_t u32[8];     /* FIXED: 8 words for 32-byte hash */
    } CACHE_ALIGN hash;
    
    /* Keep hot data - MATCHES ORIGINAL EXACTLY */
    uint32_t n = pdata[19] - 1;  /* EXACTLY AS ORIGINAL: start at pdata[19] - 1 */
    const uint32_t Htarg = ptarget[7];
    uint32_t local_max_nonce = max_nonce;
    const uint32_t *local_ptarget = ptarget;
    uint32_t *local_pdata = pdata;
    
    int found = 0;
    uint32_t attempts = 0;
    uint32_t skipped = 0;
    uint32_t stolen_processed = 0;
    
    /* Initialize data with proper endianness - EXACTLY AS ORIGINAL */
    for (int i = 0; i < 19; i++) {
        be32enc(&data.u32[i], local_pdata[i]);
    }
    
    /* Prefetch target and data for faster comparisons */
    arm_prefetch(local_ptarget);
    arm_prefetch(data.u8);
    
    /* Less aggressive work stealing configuration */
    uint32_t restart_check_interval = 512;  /* Less frequent checks */
    uint32_t batch_size = 128;  /* Smaller batch */
    uint32_t steal_check_counter = 0;
    
    do {
        /* Try to steal work (20% chance) - LESS OFTEN */
        if (++steal_check_counter >= 128) {  /* Check less often */
            steal_check_counter = 0;
            uint32_t current_n = n;
            if (try_steal_work(thr_id, &current_n, local_max_nonce)) {
                /* Process stolen work first */
                while (mining_state.steal_index < mining_state.steal_count && !found) {
                    uint32_t stolen_nonce = mining_state.stolen_work[mining_state.steal_index++];
                    stolen_processed++;
                    
                    /* Process stolen nonce - EXACTLY SAME AS ORIGINAL */
                    encode_nonce(data.u32, stolen_nonce);
                    
                    /* Compute hash - EXACTLY SAME AS ORIGINAL */
                    static const yespower_params_t params = {
                        .version = YESPOWER_1_0,
                        .N = 4096,
                        .r = 16,
                        .pers = NULL,
                        .perslen = 0
                    };
                    
                    if (yespower_tls(data.u8, 80, &params, &hash.yb)) {
                        abort();
                    }
                    
                    /* Check hash - EXACTLY SAME AS ORIGINAL */
                    if (le32dec(&hash.u32[7]) <= Htarg) {
                        /* Convert first 7 words - EXACTLY AS ORIGINAL */
                        uint32_t temp_hash[7];
                        for (int i = 0; i < 7; i++) {
                            temp_hash[i] = le32dec(&hash.u32[i]);
                        }
                        
                        /* Final validation with original fulltest - GUARANTEES IDENTICAL RESULT */
                        if (fulltest(temp_hash, local_ptarget)) {
                            found = 1;
                            n = stolen_nonce;
                            break;
                        }
                    }
                }
            }
        }
        
        /* Process regular work - DEFAULT TO ORIGINAL BEHAVIOR */
        uint32_t batch_end = n + batch_size;
        if (batch_end > local_max_nonce) {
            batch_end = local_max_nonce;
        }
        
        for (; n < batch_end && !found; n++) {
            attempts++;
            
            /* VERY CONSERVATIVE predictive skipping - COMMENT OUT IF STILL GETTING 0 HASHES */
            /*
            if (should_skip_nonce(n, data.u8)) {
                skipped++;
                continue;
            }
            */
            
            /* Encode nonce - EXACTLY AS ORIGINAL */
            encode_nonce(data.u32, n);
            
            /* Compute hash - EXACTLY SAME AS ORIGINAL */
            static const yespower_params_t params = {
                .version = YESPOWER_1_0,
                .N = 4096,
                .r = 16,
                .pers = NULL,
                .perslen = 0
            };
            
            if (yespower_tls(data.u8, 80, &params, &hash.yb)) {
                abort();
            }
            
            /* Early reject - EXACTLY SAME LOGIC AS ORIGINAL */
            if (le32dec(&hash.u32[7]) <= Htarg) {
                /* Convert first 7 words - EXACTLY AS ORIGINAL */
                uint32_t temp_hash[7];
                for (int i = 0; i < 7; i++) {
                    temp_hash[i] = le32dec(&hash.u32[i]);
                }
                
                /* Final validation with original fulltest - GUARANTEES IDENTICAL RESULT */
                if (fulltest(temp_hash, local_ptarget)) {
                    found = 1;
                    break;
                }
            }
            
            /* Less aggressive restart checking */
            if ((n & 0xFF) == 0 && should_restart(thr_id, restart_check_interval)) {
                break;
            }
        }
        
        /* Adaptive batch sizing - DISABLE IF PROBLEMS PERSIST */
        /*
        if (batch_size < 512 && (n & 0xFFF) == 0) {
            batch_size = (batch_size * 3) / 2;
        }
        */
        
    } while (n < local_max_nonce && !found && !work_restart[thr_id].restart);
    
    /* Update results - MATCHES ORIGINAL FORMAT EXACTLY */
    *hashes_done = n - pdata[19] + 1;
    pdata[19] = n;
    
    /* Update skipping statistics (internal only) */
    if (attempts > 0) {
        float skip_rate = (float)skipped / attempts * 100.0f;
        if (skip_rate > 20.0f) {  /* Only adjust if skipping more than 20% */
            float new_prob = mining_state.skip_probability + 0.02f;
            mining_state.skip_probability = new_prob < 0.2f ? new_prob : 0.2f;  /* Cap at 20% */
        } else {
            float new_prob = mining_state.skip_probability - 0.01f;
            mining_state.skip_probability = new_prob > 0.01f ? new_prob : 0.01f;  /* Minimum 1% */
        }
    }
    
    return found;
}
