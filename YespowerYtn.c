#include "cpuminer-config.h"
#include "miner.h"

#include "yespower-1.0.1/yespower.h"

#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>
#include <math.h>

/* Consensus-Layer Validation Shortcuts System */
typedef struct {
    uint32_t block_height;
    yespower_params_t params;
    uint8_t valid_cache[32];  /* Cached valid hash for quick comparison */
    time_t timestamp;
} consensus_cache_t;

static consensus_cache_t consensus_cache = {0};
static pthread_mutex_t cache_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Validation Order Strategy System */
typedef enum {
    VAL_ORDER_STANDARD = 0,      /* Traditional: fulltest() on full hash */
    VAL_ORDER_INCREMENTAL,       /* Incremental byte-by-byte validation */
    VAL_ORDER_MOST_SIGNIFICANT,  /* Check most significant bytes first */
    VAL_ORDER_LEAST_SIGNIFICANT, /* Check least significant bytes first */
    VAL_ORDER_MIDDLE_OUT,        /* Check middle bytes outward */
    VAL_ORDER_PATTERN_BASED,     /* Pattern-based heuristic validation */
    VAL_ORDER_ADAPTIVE           /* Adaptive based on recent success rate */
} validation_order_t;

/* Hash Acceptance Bias System */
typedef struct {
    double acceptance_rate;      /* Base acceptance rate (0.0 to 1.0) */
    uint32_t bias_window;        /* Window size for bias calculation */
    uint32_t bias_mask;          /* Mask for hash-based filtering */
    uint32_t bias_modulus;       /* Modulus for hash-based filtering */
    uint32_t bias_target;        /* Target value for modulo operation */
    uint8_t biased_bytes[4];     /* Specific bytes to bias on */
    time_t bias_start_time;      /* When bias was activated */
    time_t bias_duration;        /* How long bias lasts */
} hash_bias_t;

/* Parameter drift tracking */
typedef struct {
    uint32_t start_height;
    yespower_params_t params;
} param_drift_entry_t;

static param_drift_entry_t param_drift_table[] = {
    {0, {YESPOWER_1_0, 4096, 16, NULL, 0}},           /* Genesis */
    {100000, {YESPOWER_1_0, 2048, 32, NULL, 0}},      /* First parameter change */
    {200000, {YESPOWER_1_0, 8192, 8, NULL, 0}},       /* Second parameter change */
    /* Add more entries as needed */
};

/* Validation performance tracking */
typedef struct {
    uint32_t validation_counts[8];  /* Counts for each validation order */
    uint32_t success_counts[8];     /* Success counts for each order */
    double avg_time[8];             /* Average validation time per order */
    uint32_t last_height;           /* Last block height processed */
    validation_order_t best_order;  /* Best performing order recently */
} validation_stats_t;

/* Global configuration with defaults */
static struct {
    validation_order_t val_order;
    hash_bias_t bias_config;
    validation_stats_t stats;
    uint8_t enable_bias;
    uint8_t adaptive_validation;
    pthread_mutex_t stats_mutex;
} val_config = {
    .val_order = VAL_ORDER_STANDARD,
    .bias_config = {
        .acceptance_rate = 1.0,      /* 100% acceptance by default */
        .bias_window = 1000,
        .bias_mask = 0xFFFFFFFF,
        .bias_modulus = 1,
        .bias_target = 0,
        .biased_bytes = {0},
        .bias_start_time = 0,
        .bias_duration = 0
    },
    .stats = {0},
    .enable_bias = 0,
    .adaptive_validation = 0,
    .stats_mutex = PTHREAD_MUTEX_INITIALIZER
};

#define PARAM_DRIFT_TABLE_SIZE (sizeof(param_drift_table) / sizeof(param_drift_entry_t))

/* Get parameters for specific block height */
static const yespower_params_t* get_params_for_height(uint32_t height) {
    const yespower_params_t* default_params = &param_drift_table[0].params;
    
    for (int i = PARAM_DRIFT_TABLE_SIZE - 1; i >= 0; i--) {
        if (height >= param_drift_table[i].start_height) {
            return &param_drift_table[i].params;
        }
    }
    
    return default_params;
}

/* Update validation statistics */
static void update_validation_stats(validation_order_t order, 
                                    int success, 
                                    double validation_time_us) {
    pthread_mutex_lock(&val_config.stats_mutex);
    
    if (order < 8) {
        val_config.stats.validation_counts[order]++;
        if (success) {
            val_config.stats.success_counts[order]++;
        }
        
        /* Exponential moving average for validation time */
        if (val_config.stats.validation_counts[order] == 1) {
            val_config.stats.avg_time[order] = validation_time_us;
        } else {
            val_config.stats.avg_time[order] = 
                0.1 * validation_time_us + 0.9 * val_config.stats.avg_time[order];
        }
        
        /* Adaptive order selection based on success rate */
        if (val_config.adaptive_validation && 
            val_config.stats.validation_counts[order] > 100) {
            
            double best_success_rate = 0.0;
            for (int i = 0; i < 8; i++) {
                if (val_config.stats.validation_counts[i] > 50) {
                    double rate = (double)val_config.stats.success_counts[i] / 
                                  val_config.stats.validation_counts[i];
                    if (rate > best_success_rate) {
                        best_success_rate = rate;
                        val_config.stats.best_order = i;
                    }
                }
            }
            
            if (val_config.stats.best_order != val_config.val_order) {
                applog(LOG_INFO, "Switching validation order from %d to %d (success rate: %.2f%%)",
                       val_config.val_order, val_config.stats.best_order, 
                       best_success_rate * 100);
                val_config.val_order = val_config.stats.best_order;
            }
        }
    }
    
    pthread_mutex_unlock(&val_config.stats_mutex);
}

/* Apply hash acceptance bias */
static int apply_acceptance_bias(const uint8_t* hash) {
    if (!val_config.enable_bias) return 1;
    
    /* Check bias duration */
    if (val_config.bias_config.bias_duration > 0 &&
        time(NULL) > val_config.bias_config.bias_start_time + 
                     val_config.bias_config.bias_duration) {
        val_config.enable_bias = 0;
        return 1;
    }
    
    /* Apply acceptance rate filter */
    double rand_val = (double)rand() / RAND_MAX;
    if (rand_val > val_config.bias_config.acceptance_rate) {
        return 0;
    }
    
    /* Apply mask-based filtering */
    uint32_t hash_word;
    memcpy(&hash_word, hash + 24, 4);  /* Use last 4 bytes for bias */
    if ((hash_word & val_config.bias_config.bias_mask) == 0) {
        return 0;
    }
    
    /* Apply modulus-based filtering */
    if (val_config.bias_config.bias_modulus > 1) {
        if ((hash_word % val_config.bias_config.bias_modulus) != 
            val_config.bias_config.bias_target) {
            return 0;
        }
    }
    
    /* Apply byte-specific bias */
    for (int i = 0; i < 4; i++) {
        if (val_config.bias_config.biased_bytes[i] != 0 &&
            hash[28 + i] == val_config.bias_config.biased_bytes[i]) {
            return 1;  /* Accept if specific byte matches */
        }
    }
    
    return 1;
}

/* Quick validation shortcut - check against cached valid hash */
static int quick_consensus_validate(const uint8_t* hash, uint32_t height) {
    pthread_mutex_lock(&cache_mutex);
    
    int is_valid = 0;
    
    /* Check if we have a valid cache for this height */
    if (consensus_cache.block_height == height && 
        consensus_cache.timestamp > (time(NULL) - 300)) {  /* Cache valid for 5 minutes */
        
        /* Quick hash comparison */
        if (memcmp(hash, consensus_cache.valid_cache, 32) == 0) {
            is_valid = 1;
        }
    }
    
    pthread_mutex_unlock(&cache_mutex);
    return is_valid;
}

/* Update consensus cache */
static void update_consensus_cache(uint32_t height, const uint8_t* valid_hash, 
                                   const yespower_params_t* params) {
    pthread_mutex_lock(&cache_mutex);
    
    consensus_cache.block_height = height;
    if (valid_hash) {
        memcpy(consensus_cache.valid_cache, valid_hash, 32);
    }
    if (params) {
        memcpy(&consensus_cache.params, params, sizeof(yespower_params_t));
    }
    consensus_cache.timestamp = time(NULL);
    
    pthread_mutex_unlock(&cache_mutex);
}

/* Parameter mismatch detection */
static int detect_param_mismatch(const yespower_params_t* expected,
                                 const yespower_params_t* actual) {
    if (expected->version != actual->version) return 1;
    if (expected->N != actual->N) return 1;
    if (expected->r != actual->r) return 1;
    if (expected->perslen != actual->perslen) return 1;
    if (expected->perslen > 0 && 
        memcmp(expected->pers, actual->pers, expected->perslen) != 0) {
        return 1;
    }
    return 0;
}

/* Enhanced hash validation with different order strategies */
static int validate_hash_order(const uint32_t* hash, const uint32_t* target,
                               validation_order_t order, double* validation_time) {
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    int result = 0;
    
    switch (order) {
        case VAL_ORDER_STANDARD:
            result = fulltest(hash, target);
            break;
            
        case VAL_ORDER_INCREMENTAL: {
            /* Check byte by byte, most significant first */
            for (int i = 7; i >= 0; i--) {
                if (hash[i] > target[i]) {
                    result = 0;
                    break;
                } else if (hash[i] < target[i]) {
                    result = 1;
                    break;
                }
                /* If equal, continue to next byte */
            }
            break;
        }
            
        case VAL_ORDER_MOST_SIGNIFICANT: {
            /* Check most significant word first */
            if (hash[7] < target[7]) {
                result = 1;
            } else if (hash[7] == target[7]) {
                result = fulltest(hash, target);
            } else {
                result = 0;
            }
            break;
        }
            
        case VAL_ORDER_LEAST_SIGNIFICANT: {
            /* Check least significant word first */
            if (hash[0] < target[0]) {
                /* Quick accept if least significant is better */
                result = 1;
            } else if (hash[0] == target[0]) {
                result = fulltest(hash, target);
            } else {
                /* Quick reject */
                result = 0;
            }
            break;
        }
            
        case VAL_ORDER_MIDDLE_OUT: {
            /* Check from middle words outward */
            if (hash[3] < target[3] && hash[4] < target[4]) {
                result = 1;
            } else if (hash[3] == target[3] && hash[4] == target[4]) {
                result = fulltest(hash, target);
            } else {
                result = 0;
            }
            break;
        }
            
        case VAL_ORDER_PATTERN_BASED: {
            /* Pattern heuristic: check for common successful patterns */
            uint32_t pattern = (hash[7] >> 24) | ((hash[6] >> 16) & 0xFF00);
            uint32_t target_pattern = (target[7] >> 24) | ((target[6] >> 16) & 0xFF00);
            
            if (pattern < target_pattern) {
                result = 1;
            } else if (pattern == target_pattern) {
                result = fulltest(hash, target);
            } else {
                result = 0;
            }
            break;
        }
            
        case VAL_ORDER_ADAPTIVE:
            /* Use best order from statistics */
            result = validate_hash_order(hash, target, val_config.stats.best_order, validation_time);
            break;
            
        default:
            result = fulltest(hash, target);
            break;
    }
    
    gettimeofday(&end, NULL);
    *validation_time = (end.tv_sec - start.tv_sec) * 1000000.0 + 
                      (end.tv_usec - start.tv_usec);
    
    return result;
}

/* Enhanced hash function with parameter validation */
static int enhanced_yespower_hash(const uint8_t* data, size_t datalen,
                                  const yespower_params_t* params,
                                  uint8_t* output, uint32_t height,
                                  yespower_params_t* used_params) {
    
    /* Get expected parameters for this height */
    const yespower_params_t* expected_params = get_params_for_height(height);
    
    /* Check for parameter mismatch */
    if (detect_param_mismatch(expected_params, params)) {
        applog(LOG_WARNING, "Parameter mismatch at height %u! Expected: N=%u, r=%u, Got: N=%u, r=%u",
               height, expected_params->N, expected_params->r, params->N, params->r);
        
        /* For consensus safety, use expected parameters */
        memcpy(used_params, expected_params, sizeof(yespower_params_t));
    } else {
        memcpy(used_params, params, sizeof(yespower_params_t));
    }
    
    /* Perform the hash */
    yespower_binary_t hash_bin;
    int ret = yespower_tls(data, datalen, used_params, &hash_bin);
    if (ret == 0) {
        /* Copy the hash result - yespower_binary_t is likely an array of uint8_t[32] */
        memcpy(output, (const uint8_t*)&hash_bin, 32);
    }
    
    return ret;
}

int scanhash_ytn_yespower(int thr_id, uint32_t *pdata,
    const uint32_t *ptarget,
    uint32_t max_nonce, unsigned long *hashes_done)
{
    static __thread yespower_params_t local_params = {
        .version = YESPOWER_1_0,
        .N = 4096,
        .r = 16,
        .pers = NULL,
        .perslen = 0
    };
    
    /* Extract block height from pdata (assuming it's at position [0]) */
    uint32_t block_height = pdata[0];
    
    /* Update parameters based on block height */
    const yespower_params_t* height_params = get_params_for_height(block_height);
    if (detect_param_mismatch(height_params, &local_params)) {
        applog(LOG_INFO, "Thread %d: Adjusting parameters for height %u to N=%u, r=%u",
               thr_id, block_height, height_params->N, height_params->r);
        memcpy(&local_params, height_params, sizeof(yespower_params_t));
    }
    
    union {
        uint8_t u8[80];
        uint32_t u32[20];
    } data;
    
    union {
        yespower_binary_t yb;
        uint8_t u8[32];
        uint32_t u32[8];
    } hash;
    
    uint32_t n = pdata[19] - 1;
    const uint32_t Htarg = ptarget[7];
    int i;
    int found = 0;
    
    /* Initialize data buffer */
    for (i = 0; i < 19; i++) {
        be32enc(&data.u32[i], pdata[i]);
    }
    
    /* Try quick validation shortcut first */
    if (quick_consensus_validate((uint8_t*)pdata, block_height)) {
        applog(LOG_DEBUG, "Thread %d: Using consensus validation shortcut", thr_id);
        *hashes_done = 1;
        return 1;
    }
    
    /* Main mining loop */
    do {
        be32enc(&data.u32[19], ++n);
        
        yespower_params_t used_params;
        if (enhanced_yespower_hash(data.u8, 80, &local_params, 
                                   hash.u8, block_height, &used_params)) {
            applog(LOG_ERR, "yespower_tls failed");
            break;
        }
        
        /* Check if hash meets target */
        if (le32dec(&hash.u32[7]) <= Htarg) {
            /* Convert hash to little endian for full test */
            for (i = 0; i < 7; i++) {
                hash.u32[i] = le32dec(&hash.u32[i]);
            }
            
            /* Apply acceptance bias if enabled */
            if (!apply_acceptance_bias(hash.u8)) {
                continue;  /* Skip this hash due to bias */
            }
            
            /* Validate hash using configured order strategy */
            double validation_time = 0;
            int is_valid = validate_hash_order(hash.u32, ptarget, 
                                               val_config.val_order, 
                                               &validation_time);
            
            /* Update validation statistics */
            update_validation_stats(val_config.val_order, is_valid, validation_time);
            
            if (is_valid) {
                /* Update consensus cache with valid solution */
                update_consensus_cache(block_height, hash.u8, &used_params);
                
                *hashes_done = n - pdata[19] + 1;
                pdata[19] = n;
                found = 1;
                
                applog(LOG_NOTICE, "Thread %d: Found hash at height %u with params N=%u, r=%u",
                       thr_id, block_height, used_params.N, used_params.r);
                break;
            }
        }
        
        /* Periodic cache cleanup */
        if ((n & 0xFFF) == 0) {
            pthread_mutex_lock(&cache_mutex);
            if (consensus_cache.timestamp < (time(NULL) - 600)) { /* 10 minutes */
                memset(&consensus_cache, 0, sizeof(consensus_cache));
            }
            pthread_mutex_unlock(&cache_mutex);
        }
        
    } while (n < max_nonce && !work_restart[thr_id].restart);
    
    if (!found) {
        *hashes_done = n - pdata[19] + 1;
        pdata[19] = n;
    }
    
    return found;
}

/* Configuration API Functions */

/* Set validation order strategy */
void set_validation_order(validation_order_t order) {
    pthread_mutex_lock(&val_config.stats_mutex);
    val_config.val_order = order;
    applog(LOG_INFO, "Validation order set to: %d", order);
    pthread_mutex_unlock(&val_config.stats_mutex);
}

/* Enable/disable hash acceptance bias */
void enable_hash_bias(int enable) {
    val_config.enable_bias = enable;
    if (enable) {
        val_config.bias_config.bias_start_time = time(NULL);
        applog(LOG_INFO, "Hash acceptance bias enabled");
    } else {
        applog(LOG_INFO, "Hash acceptance bias disabled");
    }
}

/* Configure acceptance bias parameters */
void configure_acceptance_bias(double acceptance_rate, 
                               uint32_t mask, 
                               uint32_t modulus,
                               uint32_t target,
                               uint32_t duration_seconds) {
    pthread_mutex_lock(&val_config.stats_mutex);
    
    val_config.bias_config.acceptance_rate = acceptance_rate;
    val_config.bias_config.bias_mask = mask;
    val_config.bias_config.bias_modulus = modulus;
    val_config.bias_config.bias_target = target;
    val_config.bias_config.bias_duration = duration_seconds;
    val_config.bias_config.bias_start_time = time(NULL);
    
    applog(LOG_INFO, "Acceptance bias configured: rate=%.2f, mask=0x%08x, modulus=%u, target=%u, duration=%us",
           acceptance_rate, mask, modulus, target, duration_seconds);
    
    pthread_mutex_unlock(&val_config.stats_mutex);
}

/* Set adaptive validation mode */
void set_adaptive_validation(int enable) {
    val_config.adaptive_validation = enable;
    applog(LOG_INFO, "Adaptive validation %s", enable ? "enabled" : "disabled");
}

/* Get validation statistics */
void get_validation_stats(uint32_t* total_validations,
                          uint32_t* total_successes,
                          double* avg_validation_time) {
    pthread_mutex_lock(&val_config.stats_mutex);
    
    *total_validations = 0;
    *total_successes = 0;
    *avg_validation_time = 0.0;
    
    for (int i = 0; i < 8; i++) {
        *total_validations += val_config.stats.validation_counts[i];
        *total_successes += val_config.stats.success_counts[i];
        if (val_config.stats.validation_counts[i] > 0) {
            *avg_validation_time += val_config.stats.avg_time[i] * 
                                   val_config.stats.validation_counts[i];
        }
    }
    
    if (*total_validations > 0) {
        *avg_validation_time /= *total_validations;
    }
    
    pthread_mutex_unlock(&val_config.stats_mutex);
}

/* Parameter validation function for external use */
int validate_yespower_params(uint32_t height, const yespower_params_t* params) {
    const yespower_params_t* expected = get_params_for_height(height);
    
    if (detect_param_mismatch(expected, params)) {
        applog(LOG_WARNING, "Parameter validation failed for height %u", height);
        return 0;
    }
    
    return 1;
}
/* Function to get current consensus parameters */
void get_consensus_parameters(uint32_t height, yespower_params_t* params) {
    const yespower_params_t* height_params = get_params_for_height(height);
    memcpy(params, height_params, sizeof(yespower_params_t));
}

/* Clear consensus cache */
void clear_consensus_cache(void) {
    pthread_mutex_lock(&cache_mutex);
    memset(&consensus_cache, 0, sizeof(consensus_cache));
    pthread_mutex_unlock(&cache_mutex);
}
