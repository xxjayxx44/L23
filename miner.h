#ifndef __MINER_H__
#define __MINER_H__

#include "cpuminer-config.h"

#include <stdbool.h>
#include <inttypes.h>
#include <sys/time.h>
#include <pthread.h>
#include <jansson.h>
#include <curl/curl.h>

#ifdef STDC_HEADERS
# include <stdlib.h>
# include <stddef.h>
#else
# ifdef HAVE_STDLIB_H
#  include <stdlib.h>
# endif
#endif
#ifdef HAVE_ALLOCA_H
# include <alloca.h>
#elif !defined alloca
# ifdef __GNUC__
#  define alloca __builtin_alloca
# elif defined _AIX
#  define alloca __alloca
# elif defined _MSC_VER
#  include <malloc.h>
#  define alloca _alloca
# elif !defined HAVE_ALLOCA
#  ifdef  __cplusplus
extern "C"
#  endif
void *alloca (size_t);
# endif
#endif

#ifdef HAVE_SYSLOG_H
#include <syslog.h>
#else
enum {
	LOG_ERR,
	LOG_WARNING,
	LOG_NOTICE,
	LOG_INFO,
	LOG_DEBUG,
};
#endif

#undef unlikely
#undef likely
#if defined(__GNUC__) && (__GNUC__ > 2) && defined(__OPTIMIZE__)
#define unlikely(expr) (__builtin_expect(!!(expr), 0))
#define likely(expr) (__builtin_expect(!!(expr), 1))
#else
#define unlikely(expr) (expr)
#define likely(expr) (expr)
#endif

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif

/* ================= IMPROVED ENDIAN HELPERS ================= */
/* Using compiler intrinsics for maximum speed */
#if defined(__GNUC__) && (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 3)
#define HAVE_BUILTIN_BSWAP
#endif

#if defined(_MSC_VER)
#include <stdlib.h>
#define bswap_32(x) _byteswap_ulong(x)
#define bswap_64(x) _byteswap_uint64(x)
#elif defined(HAVE_BUILTIN_BSWAP)
#define bswap_32(x) __builtin_bswap32(x)
#define bswap_64(x) __builtin_bswap64(x)
#else
#define bswap_32(x) ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u) \
                   | (((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))
#define bswap_64(x) (((uint64_t)bswap_32((x) & 0xffffffffu) << 32) | \
                     (uint64_t)bswap_32((x) >> 32))
#endif

/* Fast inline endian conversion functions */
static inline uint32_t swab32(uint32_t v)
{
	return bswap_32(v);
}

static inline uint64_t swab64(uint64_t v)
{
	return bswap_64(v);
}

/* Optimized 32-bit read/write with single memory access when aligned */
#ifdef HAVE_SYS_ENDIAN_H
#include <sys/endian.h>
#endif

/* Check for aligned access support */
#if defined(__GNUC__) && (__GNUC__ >= 3) && defined(__GNUC_MINOR__) && (__GNUC_MINOR__ >= 1)
#define MAY_ALIAS __attribute__((__may_alias__))
#else
#define MAY_ALIAS
#endif

/* Fast aligned memory access helpers */
typedef uint32_t MAY_ALIAS aligned_uint32_t;
typedef uint64_t MAY_ALIAS aligned_uint64_t;

#if !HAVE_DECL_BE32DEC
static inline uint32_t be32dec(const void *pp)
{
	/* Try aligned access first for speed */
	if (((uintptr_t)pp & 3) == 0) {
		return bswap_32(*(const aligned_uint32_t *)pp);
	} else {
		const uint8_t *p = (uint8_t const *)pp;
		return ((uint32_t)(p[3]) + ((uint32_t)(p[2]) << 8) +
			((uint32_t)(p[1]) << 16) + ((uint32_t)(p[0]) << 24));
	}
}
#endif

#if !HAVE_DECL_LE32DEC
static inline uint32_t le32dec(const void *pp)
{
	/* Try aligned access first for speed */
	if (((uintptr_t)pp & 3) == 0) {
		return *(const aligned_uint32_t *)pp;
	} else {
		const uint8_t *p = (uint8_t const *)pp;
		return ((uint32_t)(p[0]) + ((uint32_t)(p[1]) << 8) +
			((uint32_t)(p[2]) << 16) + ((uint32_t)(p[3]) << 24));
	}
}
#endif

#if !HAVE_DECL_BE32ENC
static inline void be32enc(void *pp, uint32_t x)
{
	/* Try aligned access first for speed */
	if (((uintptr_t)pp & 3) == 0) {
		*(aligned_uint32_t *)pp = bswap_32(x);
	} else {
		uint8_t *p = (uint8_t *)pp;
		p[3] = x & 0xff;
		p[2] = (x >> 8) & 0xff;
		p[1] = (x >> 16) & 0xff;
		p[0] = (x >> 24) & 0xff;
	}
}
#endif

#if !HAVE_DECL_LE32ENC
static inline void le32enc(void *pp, uint32_t x)
{
	/* Try aligned access first for speed */
	if (((uintptr_t)pp & 3) == 0) {
		*(aligned_uint32_t *)pp = x;
	} else {
		uint8_t *p = (uint8_t *)pp;
		p[0] = x & 0xff;
		p[1] = (x >> 8) & 0xff;
		p[2] = (x >> 16) & 0xff;
		p[3] = (x >> 24) & 0xff;
	}
}
#endif

/* 64-bit endian helpers for improved performance */
#if !HAVE_DECL_BE64DEC
static inline uint64_t be64dec(const void *pp)
{
	if (((uintptr_t)pp & 7) == 0) {
		return bswap_64(*(const aligned_uint64_t *)pp);
	} else {
		const uint8_t *p = (uint8_t const *)pp;
		return ((uint64_t)(p[7]) + ((uint64_t)(p[6]) << 8) +
			((uint64_t)(p[5]) << 16) + ((uint64_t)(p[4]) << 24) +
			((uint64_t)(p[3]) << 32) + ((uint64_t)(p[2]) << 40) +
			((uint64_t)(p[1]) << 48) + ((uint64_t)(p[0]) << 56));
	}
}
#endif

#if !HAVE_DECL_LE64DEC
static inline uint64_t le64dec(const void *pp)
{
	if (((uintptr_t)pp & 7) == 0) {
		return *(const aligned_uint64_t *)pp;
	} else {
		const uint8_t *p = (uint8_t const *)pp;
		return ((uint64_t)(p[0]) + ((uint64_t)(p[1]) << 8) +
			((uint64_t)(p[2]) << 16) + ((uint64_t)(p[3]) << 24) +
			((uint64_t)(p[4]) << 32) + ((uint64_t)(p[5]) << 40) +
			((uint64_t)(p[6]) << 48) + ((uint64_t)(p[7]) << 56));
	}
}
#endif

#if !HAVE_DECL_BE64ENC
static inline void be64enc(void *pp, uint64_t x)
{
	if (((uintptr_t)pp & 7) == 0) {
		*(aligned_uint64_t *)pp = bswap_64(x);
	} else {
		uint8_t *p = (uint8_t *)pp;
		p[7] = x & 0xff;
		p[6] = (x >> 8) & 0xff;
		p[5] = (x >> 16) & 0xff;
		p[4] = (x >> 24) & 0xff;
		p[3] = (x >> 32) & 0xff;
		p[2] = (x >> 40) & 0xff;
		p[1] = (x >> 48) & 0xff;
		p[0] = (x >> 56) & 0xff;
	}
}
#endif

#if !HAVE_DECL_LE64ENC
static inline void le64enc(void *pp, uint64_t x)
{
	if (((uintptr_t)pp & 7) == 0) {
		*(aligned_uint64_t *)pp = x;
	} else {
		uint8_t *p = (uint8_t *)pp;
		p[0] = x & 0xff;
		p[1] = (x >> 8) & 0xff;
		p[2] = (x >> 16) & 0xff;
		p[3] = (x >> 24) & 0xff;
		p[4] = (x >> 32) & 0xff;
		p[5] = (x >> 40) & 0xff;
		p[6] = (x >> 48) & 0xff;
		p[7] = (x >> 56) & 0xff;
	}
}
#endif

/* Batch operations for better throughput */
static inline void be32enc_batch(void *pp, const uint32_t *data, size_t count)
{
	uint32_t *p = (uint32_t *)pp;
	for (size_t i = 0; i < count; i++) {
		be32enc(&p[i], data[i]);
	}
}

static inline void le32enc_batch(void *pp, const uint32_t *data, size_t count)
{
	uint32_t *p = (uint32_t *)pp;
	for (size_t i = 0; i < count; i++) {
		le32enc(&p[i], data[i]);
	}
}

#if JANSSON_MAJOR_VERSION >= 2
#define JSON_LOADS(str, err_ptr) json_loads(str, 0, err_ptr)
#define JSON_LOAD_FILE(path, err_ptr) json_load_file(path, 0, err_ptr)
#else
#define JSON_LOADS(str, err_ptr) json_loads(str, err_ptr)
#define JSON_LOAD_FILE(path, err_ptr) json_load_file(path, err_ptr)
#endif

#define USER_AGENT PACKAGE_NAME "/" PACKAGE_VERSION

void sha256_init(uint32_t *state);
void sha256_transform(uint32_t *state, const uint32_t *block, int swap);
void sha256d(unsigned char *hash, const unsigned char *data, int len);

extern int scanhash_tidecoin_yespower(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done);

extern int scanhash_sugar_yespower(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done);

extern int scanhash_iso_yespower(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done);

extern int scanhash_null_yespower(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done);

extern int scanhash_urx_yespower(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done);

extern int scanhash_litb_yespower(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done);

extern int scanhash_iots_yespower(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done);

extern int scanhash_itc_yespower(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done);

extern int scanhash_ytn_yespower(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done);

/* ================= THREAD RESTART / WORK-RESTART SIGNALING ================= */
/* Enhanced thread info with restart signaling */
struct thr_info {
	int		id;
	pthread_t	pth;
	struct thread_q	*q;
	volatile int	restart_requested;	/* Atomic restart flag */
	volatile int	work_abandoned;		/* Signal to abandon current work */
	uint32_t	last_nonce;		/* For restart continuation */
	uint64_t	work_id;		/* Unique work identifier */
	uint32_t	skip_mask;		/* Pattern mask for skipping */
	pthread_mutex_t	restart_mutex;		/* For restart synchronization */
	pthread_cond_t	restart_cond;		/* Condition for restart */
};

/* Enhanced work restart structure with bias control */
struct work_restart {
	volatile unsigned long	restart;
	volatile uint32_t	bias_threshold;	/* Threshold for skipping */
	volatile uint32_t	skip_pattern;	/* Pattern to identify unfavorable work */
	volatile int		adaptive_mode;	/* Enable adaptive bias */
	volatile double		bias_factor;	/* Dynamic bias adjustment */
	char			padding[128 - sizeof(unsigned long) - 
					sizeof(uint32_t) * 3 - sizeof(int) - sizeof(double)];
};

/* Thread restart control functions */
static inline void request_thread_restart(struct thr_info *thr)
{
	if (thr) {
		thr->restart_requested = 1;
		pthread_mutex_lock(&thr->restart_mutex);
		pthread_cond_signal(&thr->restart_cond);
		pthread_mutex_unlock(&thr->restart_mutex);
	}
}

static inline int is_restart_requested(struct thr_info *thr)
{
	return thr ? thr->restart_requested : 0;
}

static inline void clear_restart_request(struct thr_info *thr)
{
	if (thr) {
		thr->restart_requested = 0;
	}
}

static inline void signal_work_abandon(struct thr_info *thr)
{
	if (thr) {
		thr->work_abandoned = 1;
	}
}

/* Bias validation helpers */
static inline int is_favorable_hash(uint32_t hash, uint32_t threshold, uint32_t pattern)
{
	/* Quick early rejection based on simple heuristics */
	if (threshold > 0 && (hash & 0xFFF) > threshold) {
		return 0; /* Unfavorable */
	}
	
	/* Pattern matching for known unfavorable patterns */
	if (pattern > 0 && (hash & pattern) == pattern) {
		return 0; /* Matches skip pattern */
	}
	
	return 1; /* Favorable */
}

/* Gate function for fulltest with bias control */
static inline int gate_fulltest(const uint32_t *hash, const uint32_t *target, 
								uint32_t bias_threshold, uint32_t skip_pattern)
{
	/* Suppress unused parameter warning */
	(void)target;

	/* Quick pre-check before full comparison */
	if (bias_threshold > 0) {
		/* Check if hash passes initial bias gate */
		if (!is_favorable_hash(hash[7], bias_threshold, skip_pattern)) {
			return 0; /* Reject early */
		}
	}
	
	/* Continue with standard comparison */
	return 1;
}

/* Enhanced comparison with bias awareness */
static inline int biased_compare(const uint32_t *hash, const uint32_t *target,
								uint32_t bias_threshold, uint32_t skip_pattern)
{
	/* Apply bias gate first */
	if (!gate_fulltest(hash, target, bias_threshold, skip_pattern)) {
		return -1; /* Early rejection */
	}
	
	/* Standard comparison */
	for (int i = 7; i >= 0; i--) {
		uint32_t h32 = hash[i];
		uint32_t t32 = target[i];
		
		if (h32 < t32)
			return 1;
		if (h32 > t32)
			return 0;
	}
	
	return 1; /* Hash meets target */
}

/* Selective work skipping decision */
static inline int should_skip_work(uint32_t *work_data, uint32_t skip_mask, 
								  uint32_t bias_threshold, struct work_restart *wr)
{
	if (!wr || !work_data) return 0;
	
	/* Check for restart signal */
	if (wr->restart) return 1;
	
	/* Apply adaptive bias if enabled */
	if (wr->adaptive_mode && wr->bias_factor > 1.0) {
		uint32_t adjusted_threshold = (uint32_t)(bias_threshold * wr->bias_factor);
		/* Quick hash of work data to decide */
		uint32_t work_hash = work_data[0] ^ work_data[1] ^ work_data[2];
		if ((work_hash & skip_mask) > adjusted_threshold) {
			return 1; /* Skip this work */
		}
	}
	
	/* Standard pattern check */
	if (skip_mask > 0) {
		uint32_t pattern_check = work_data[0] & skip_mask;
		if (pattern_check == skip_mask) {
			return 1; /* Matches unfavorable pattern */
		}
	}
	
	return 0; /* Don't skip */
}

/* Adaptive bias adjustment */
static inline void adjust_bias_threshold(struct work_restart *wr, 
										double hash_rate, double found_blocks)
{
	if (!wr || !wr->adaptive_mode) return;
	
	/* Simple adaptive algorithm based on performance */
	if (hash_rate > 1000.0 && found_blocks < 1.0) {
		/* Increase bias if hashing fast but finding few blocks */
		wr->bias_factor *= 1.05;
		if (wr->bias_factor > 2.0) wr->bias_factor = 2.0;
	} else if (found_blocks > 0.0) {
		/* Decrease bias if finding blocks */
		wr->bias_factor *= 0.95;
		if (wr->bias_factor < 1.0) wr->bias_factor = 1.0;
	}
}

extern bool opt_debug;
extern bool opt_protocol;
extern bool opt_redirect;
extern int opt_timeout;
extern bool want_longpoll;
extern bool have_longpoll;
extern bool have_gbt;
extern bool allow_getwork;
extern bool want_stratum;
extern bool have_stratum;
extern char *opt_cert;
extern char *opt_proxy;
extern long opt_proxy_type;
extern bool use_syslog;
extern pthread_mutex_t applog_lock;
extern struct thr_info *thr_info;
extern int longpoll_thr_id;
extern int stratum_thr_id;
extern struct work_restart *work_restart;

/* Global restart control */
extern volatile int global_restart_request;
extern void request_global_restart(void);
extern void clear_global_restart(void);
extern int check_global_restart(void);

#define JSON_RPC_LONGPOLL	(1 << 0)
#define JSON_RPC_QUIET_404	(1 << 1)

extern void applog(int prio, const char *fmt, ...);
extern json_t *json_rpc_call(CURL *curl, const char *url, const char *userpass,
	const char *rpc_req, int *curl_err, int flags);
void memrev(unsigned char *p, size_t len);
extern void bin2hex(char *s, const unsigned char *p, size_t len);
extern char *abin2hex(const unsigned char *p, size_t len);
extern bool hex2bin(unsigned char *p, const char *hexstr, size_t len);
extern int varint_encode(unsigned char *p, uint64_t n);
extern size_t address_to_script(unsigned char *out, size_t outsz, const char *addr);
extern int timeval_subtract(struct timeval *result, struct timeval *x,
	struct timeval *y);

/* Enhanced fulltest with bias control */
extern bool fulltest(const uint32_t *hash, const uint32_t *target);
extern bool fulltest_biased(const uint32_t *hash, const uint32_t *target,
						   uint32_t bias_threshold, uint32_t skip_pattern);
extern bool fulltest_gated(const uint32_t *hash, const uint32_t *target,
						  struct work_restart *wr);

extern void diff_to_target(uint32_t *target, double diff);

struct stratum_job {
	char *job_id;
	unsigned char prevhash[32];
	size_t coinbase_size;
	unsigned char *coinbase;
	unsigned char *xnonce2;
	int merkle_count;
	unsigned char **merkle;
	unsigned char version[4];
	unsigned char nbits[4];
	unsigned char ntime[4];
	bool clean;
	double diff;
};

struct stratum_ctx {
	char *url;

	CURL *curl;
	char *curl_url;
	char curl_err_str[CURL_ERROR_SIZE];
	curl_socket_t sock;
	size_t sockbuf_size;
	char *sockbuf;
	pthread_mutex_t sock_lock;

	double next_diff;

	char *session_id;
	size_t xnonce1_size;
	unsigned char *xnonce1;
	size_t xnonce2_size;
	struct stratum_job job;
	pthread_mutex_t work_lock;
};

bool stratum_socket_full(struct stratum_ctx *sctx, int timeout);
bool stratum_send_line(struct stratum_ctx *sctx, char *s);
char *stratum_recv_line(struct stratum_ctx *sctx);
bool stratum_connect(struct stratum_ctx *sctx, const char *url);
void stratum_disconnect(struct stratum_ctx *sctx);
bool stratum_subscribe(struct stratum_ctx *sctx);
bool stratum_authorize(struct stratum_ctx *sctx, const char *user, const char *pass);
bool stratum_handle_method(struct stratum_ctx *sctx, const char *s);

struct thread_q;

extern struct thread_q *tq_new(void);
extern void tq_free(struct thread_q *tq);
extern bool tq_push(struct thread_q *tq, void *data);
extern void *tq_pop(struct thread_q *tq, const struct timespec *abstime);
extern void tq_freeze(struct thread_q *tq);
extern void tq_thaw(struct thread_q *tq);

#endif /* __MINER_H__ */
