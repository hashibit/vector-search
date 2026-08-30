
#ifndef INCLUDE_VECTOR_SEARCH_H_
#define INCLUDE_VECTOR_SEARCH_H_

#include <stddef.h>

#ifdef _MSC_VER
#ifdef __cplusplus
#ifdef VECTOR_SEARCH_EXPORTS
#define VECTOR_SEARCH_API extern "C" __declspec(dllexport)
#else
#define VECTOR_SEARCH_API extern "C" __declspec(dllimport)
#endif
#else
#ifdef VECTOR_SEARCH_EXPORTS
#define VECTOR_SEARCH_API __declspec(dllexport)
#else
#define VECTOR_SEARCH_API __declspec(dllimport)
#endif
#endif
#else /* _MSC_VER */
#ifdef __cplusplus
#ifdef VECTOR_SEARCH_EXPORTS
#define VECTOR_SEARCH_API extern "C" __attribute__((visibility("default")))
#else
#define VECTOR_SEARCH_API extern "C"
#endif
#else
#ifdef VECTOR_SEARCH_EXPORTS
#define VECTOR_SEARCH_API __attribute__((visibility("default")))
#else
#define VECTOR_SEARCH_API
#endif
#endif
#endif

#define VES_OK 0
#define VES_INVALID_HANDLE (-1)
#define VES_FILE_NOT_FOUND (-2)
#define VES_NOMEMORY (-3)
#define VES_INVALID_ARGUMENT (-4)
#define VES_BAD_INDEX (-5)
#define VES_SYSCALL_ERROR (-6)

typedef void *ves_gpu_index_t;
typedef void *ves_cpu_index_t;
typedef void *ves_index_shards_t;
typedef void *ves_resource_t;

typedef struct
{
    // Index
    int dimension;
    long index_size;
    int is_trained;
    // GpuIndexIVF
    int nlist;
    int nprobe;
    int max_list_size;
    // GpuIndexIVFPQ
    int subQuantizers;
    int bitsPerCode;
} ves_index_status_t;

typedef struct
{
    // Index
    int dimension;
    // GpuIndexIVF
    int nlist;
    int nprobe;
    // GpuIndexIVFPQ
    int subQuantizers;
    int bitsPerCode;
} ves_index_config_t;

typedef struct
{
    size_t total_memory_size;
    size_t free_memory_size;
} ves_gpu_info_t;

typedef struct
{
    unsigned int time_range[2];
    // bits from 0 to 127
    unsigned char camera_mask[128 / 8];
} ves_timespace_filter_t;

typedef struct ves_device_properties
{
    int major;
    int minor;
    char name[256];
} ves_device_properties;

//GPU Search
VECTOR_SEARCH_API int ves_get_current_device_id();

VECTOR_SEARCH_API int ves_get_gpu_info(int device_id, ves_gpu_info_t *info);

VECTOR_SEARCH_API int ves_create_gpu_resource(int device_id, unsigned long temp_memory_size, float temp_memory_fraction, ves_resource_t *res);
VECTOR_SEARCH_API int ves_destroy_gpu_resource(ves_resource_t resource);

VECTOR_SEARCH_API int ves_init_gpu_index(ves_resource_t resource, ves_index_config_t *config, ves_gpu_index_t *idx);
VECTOR_SEARCH_API int ves_load_gpu_index(ves_resource_t resource, const char *path, int device_id, ves_gpu_index_t *idx);
VECTOR_SEARCH_API int ves_create_index_shards(int dimension, int threaded, ves_index_shards_t *shards);

VECTOR_SEARCH_API int ves_free_gpu_index(ves_gpu_index_t index);
VECTOR_SEARCH_API int ves_free_cpu_index(ves_cpu_index_t index);
VECTOR_SEARCH_API int ves_free_index_shards(ves_index_shards_t shards);

VECTOR_SEARCH_API int ves_get_gpu_index_status(ves_gpu_index_t index, ves_index_status_t *status);
VECTOR_SEARCH_API int ves_get_gpu_index_ids(ves_gpu_index_t index, long *ids);
VECTOR_SEARCH_API long ves_get_gpu_max_may_reserve_memory(ves_gpu_index_t index, long n);

VECTOR_SEARCH_API int ves_train_gpu_index(ves_gpu_index_t index, long n, const float *x);
VECTOR_SEARCH_API int ves_add_index_batch(ves_gpu_index_t index, long n, const float *x, const long *ids);
VECTOR_SEARCH_API int ves_add_gpu_index_to_shards(ves_gpu_index_t index, ves_index_shards_t shards);
VECTOR_SEARCH_API int ves_add_shards_to_shards(ves_index_shards_t from, ves_index_shards_t to);

VECTOR_SEARCH_API int ves_index_gpu_to_cpu(ves_gpu_index_t index, ves_cpu_index_t *cpu_index, int reclaim_gpu_memory);
VECTOR_SEARCH_API int ves_index_cpu_to_gpu(ves_cpu_index_t index, ves_resource_t resource, int device_id, ves_gpu_index_t *gpu_index);
VECTOR_SEARCH_API int ves_write_cpu_index(ves_cpu_index_t index, const char *filename);

VECTOR_SEARCH_API int ves_search_index(ves_gpu_index_t index, long n, const float *x, int k, float *distances, long *ids);
VECTOR_SEARCH_API int ves_search_index_with_timespace_filter(ves_gpu_index_t index, long n, const float *x, int k, ves_timespace_filter_t *filter, float *distances, long *ids);

VECTOR_SEARCH_API int ves_search_index_shards(ves_index_shards_t shards, long n, const float *x, int k, float *distances, long *ids);

//CPU Search
VECTOR_SEARCH_API int ves_init_cpu_index(ves_index_config_t *config, ves_cpu_index_t *idx);
VECTOR_SEARCH_API int ves_load_cpu_index(const char *path, ves_cpu_index_t *idx);

VECTOR_SEARCH_API int ves_get_cpu_index_status(ves_cpu_index_t index, ves_index_status_t *status);
VECTOR_SEARCH_API int ves_get_cpu_index_ids(ves_cpu_index_t index, long *ids);

VECTOR_SEARCH_API int ves_train_cpu_index(ves_cpu_index_t index, long n, const float *x);
VECTOR_SEARCH_API int ves_add_cpu_index_batch(ves_cpu_index_t index, long n, const float *x, const long *ids);
VECTOR_SEARCH_API int ves_remove_cpu_index_ids(ves_cpu_index_t index, long n, const long *ids);

VECTOR_SEARCH_API int ves_search_cpu_index(ves_cpu_index_t index, long n, const float *x, int k, float *distances, long *ids);
VECTOR_SEARCH_API int ves_search_cpu_index_with_timespace_filter(ves_cpu_index_t index, long n, const float *x, int k, ves_timespace_filter_t *filter, float *distances, long *ids);

VECTOR_SEARCH_API int ves_clone_cpu_index(ves_cpu_index_t input_index, ves_cpu_index_t *output_index);

VECTOR_SEARCH_API int ves_get_device_properties(ves_device_properties *prop, int device);
#endif // INCLUDE_VECTOR_SEARCH_H_
