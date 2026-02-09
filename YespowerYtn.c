#include "cpuminer-config.h"
#include "miner.h"

#include "yespower-1.0.1/yespower.h"

#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>

/* Consensus-Layer Validation Shortcuts System */
typedef struct {
    uint32_t block_height;
    yespower_params_t params;
    uint8_t valid_cache[32];  /* Cached valid hash for quick comparison */
    time_t timestamp;
} consensus_cache_t;

static consensus_cache_t consensus_cache = {0};
static pthread_mutex_t cache_mutex = PTHREAD_MUTEX_INITIALIZER;

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
        memcpy(output, hash_bin, 32);
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
            
            if (fulltest(hash.u32, ptarget)) {
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
