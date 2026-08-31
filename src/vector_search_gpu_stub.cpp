// vector_search_gpu_stub: CPU-only build of the GPU surface of
// include/vector_search.h.
//
// Selected by CMake when VES_WITH_CUDA=OFF (the CUDA build compiles
// vector_search_gpu.cpp instead): every GPU/shards/device entry point is a
// stub returning -7. There is deliberately no argument checking here — the
// whole surface is unavailable, which is the documented contract (README,
// tests/ves_smoke_test.cpp).
#include "ves_internal.h"

extern "C" {

int ves_get_current_device_id() {
    return ves::kGpuUnavailable;
}

int ves_get_gpu_info(int, ves_gpu_info_t*) {
    return ves::kGpuUnavailable;
}

int ves_create_gpu_resource(int, unsigned long, float, ves_resource_t*) {
    return ves::kGpuUnavailable;
}

int ves_destroy_gpu_resource(ves_resource_t) {
    return ves::kGpuUnavailable;
}

int ves_init_gpu_index(ves_resource_t, ves_index_config_t*, ves_gpu_index_t*) {
    return ves::kGpuUnavailable;
}

int ves_load_gpu_index(ves_resource_t, const char*, int, ves_gpu_index_t*) {
    return ves::kGpuUnavailable;
}

int ves_free_gpu_index(ves_gpu_index_t) {
    return ves::kGpuUnavailable;
}

int ves_create_index_shards(int, int, ves_index_shards_t*) {
    return ves::kGpuUnavailable;
}

int ves_free_index_shards(ves_index_shards_t) {
    return ves::kGpuUnavailable;
}

int ves_add_gpu_index_to_shards(ves_gpu_index_t, ves_index_shards_t) {
    return ves::kGpuUnavailable;
}

int ves_add_shards_to_shards(ves_index_shards_t, ves_index_shards_t) {
    return ves::kGpuUnavailable;
}

int ves_search_index_shards(ves_index_shards_t, long, const float*, int, float*, long*) {
    return ves::kGpuUnavailable;
}

int ves_index_gpu_to_cpu(ves_gpu_index_t, ves_cpu_index_t*, int) {
    return ves::kGpuUnavailable;
}

int ves_index_cpu_to_gpu(ves_cpu_index_t, ves_resource_t, int, ves_gpu_index_t*) {
    return ves::kGpuUnavailable;
}

int ves_get_gpu_index_status(ves_gpu_index_t, ves_index_status_t*) {
    return ves::kGpuUnavailable;
}

int ves_get_gpu_index_ids(ves_gpu_index_t, long*) {
    return ves::kGpuUnavailable;
}

long ves_get_gpu_max_may_reserve_memory(ves_gpu_index_t, long) {
    return ves::kGpuUnavailable;
}

int ves_train_gpu_index(ves_gpu_index_t, long, const float*) {
    return ves::kGpuUnavailable;
}

int ves_add_index_batch(ves_gpu_index_t, long, const float*, const long*) {
    return ves::kGpuUnavailable;
}

int ves_search_index(ves_gpu_index_t, long, const float*, int, float*, long*) {
    return ves::kGpuUnavailable;
}

int ves_search_index_with_timespace_filter(ves_gpu_index_t, long, const float*, int,
                                           ves_timespace_filter_t*, float*, long*) {
    return ves::kGpuUnavailable;
}

int ves_get_device_properties(ves_device_properties*, int) {
    return ves::kGpuUnavailable;
}

}  // extern "C"
