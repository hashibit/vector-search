// vector_search_gpu: CUDA implementation of the GPU surface of
// include/vector_search.h.
//
// Selected by CMake when VES_WITH_CUDA=ON (the CPU-only build compiles
// vector_search_gpu_stub.cpp instead, which stubs every entry point to -7).
// Written against the modern faiss GPU API (>= 1.7, checked against the 1.14
// headers); this file is NOT compiled or validated without a CUDA toolchain.
//
// CPU-side helpers (new_ivfpq, read_cpu_index_from_file) come from
// vector_search.cpp, the filtered scan (search_ivfpq_with_filter) from
// ves_search_filter.cpp.
#include "ves_gpu_internal.h"
#include "ves_internal.h"
#include "id_filter.h"
#include "ves_search_filter.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace ves {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

faiss::gpu::GpuIndexIVFPQConfig gpu_config_for(int device_id, const ves_index_config_t& config) {
    faiss::gpu::GpuIndexIVFPQConfig cfg;
    cfg.device = device_id;
    cfg.indicesOptions = faiss::gpu::INDICES_64_BIT;  // ids are 64-bit everywhere in this API
    return cfg;
}

// ntotal and max_list_size are computed from the GPU index's lists; modern
// GpuIndexIVF exposes them per list (getNumLists/getListLength) instead of
// through a CPU InvertedLists view.
long gpu_ntotal(const faiss::gpu::GpuIndexIVF* index) {
    long total = 0;
    const long nlist = index->getNumLists();
    for (long l = 0; l < nlist; ++l) {
        total += index->getListLength(l);
    }
    return total;
}

long gpu_max_list_size(const faiss::gpu::GpuIndexIVF* index) {
    long max_size = 0;
    const long nlist = index->getNumLists();
    for (long l = 0; l < nlist; ++l) {
        max_size = std::max(max_size, static_cast<long>(index->getListLength(l)));
    }
    return max_size;
}

}  // namespace ves

// ===========================================================================
// C API (GPU surface; every function below requires VES_WITH_CUDA)
// ===========================================================================

extern "C" {

// ---------------------------------------------------------------------------
// GPU resource & device queries
// ---------------------------------------------------------------------------

int ves_get_current_device_id() {
    try {
        int dev = 0;
        cudaGetDevice(&dev);
        return dev;
    } catch (...) {
        return ves::exception_to_error("ves_get_current_device_id");
    }
}

int ves_get_gpu_info(int device_id, ves_gpu_info_t* info) {
    if (info == nullptr) {
        return VES_INVALID_ARGUMENT;
    }
    try {
        cudaSetDevice(device_id);
        size_t free_bytes = 0, total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        info->total_memory_size = total_bytes;
        info->free_memory_size = free_bytes;
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_get_gpu_info");
    }
}

int ves_create_gpu_resource(int device_id, unsigned long temp_memory_size,
                            float temp_memory_fraction, ves_resource_t* res) {
    if (res == nullptr) {
        return VES_INVALID_ARGUMENT;
    }
    try {
        auto* r = new ves::GpuResource();
        r->device_id = device_id;
        r->res = new faiss::gpu::StandardGpuResources();
        if (temp_memory_size > 0) {
            r->res->setTempMemory(temp_memory_size);
        }
        // temp_memory_fraction: no equivalent in modern faiss (setTempMemoryFraction
        // was removed); it is ignored in the modern build.
        (void)temp_memory_fraction;
        *res = r;
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_create_gpu_resource");
    }
}

int ves_destroy_gpu_resource(ves_resource_t resource) {
    try {
        delete static_cast<ves::GpuResource*>(resource);
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_destroy_gpu_resource");
    }
}

// ---------------------------------------------------------------------------
// GPU index lifecycle
// ---------------------------------------------------------------------------

int ves_init_gpu_index(ves_resource_t resource, ves_index_config_t* config, ves_gpu_index_t* idx) {
    if (idx == nullptr || config == nullptr) {
        return VES_INVALID_ARGUMENT;
    }
    auto* res = static_cast<ves::GpuResource*>(resource);
    if (res == nullptr || res->res == nullptr) {
        return VES_INVALID_HANDLE;
    }
    try {
        faiss::gpu::GpuIndexIVFPQConfig cfg = ves::gpu_config_for(res->device_id, *config);
        // Modern faiss builds the GPU index from a CPU IndexIVFPQ (copies
        // data over once trained); train/add then run on the GPU index.
        faiss::IndexIVFPQ* cpu = ves::new_ivfpq(*config);
        auto* gidx = new ves::GpuIndex();
        gidx->device_id = res->device_id;
        gidx->resource = res;
        gidx->index = new faiss::gpu::GpuIndexIVFPQ(res->res, cpu, cfg);
        gidx->nprobe = config->nprobe;  // applied per search via SearchParametersIVF
        delete cpu;
        if (gidx->index == nullptr) {
            delete gidx;
            return VES_NOMEMORY;
        }
        *idx = gidx;
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_init_gpu_index");
    }
}

int ves_load_gpu_index(ves_resource_t resource, const char* path, int device_id,
                       ves_gpu_index_t* idx) {
    if (idx == nullptr) {
        return VES_INVALID_ARGUMENT;
    }
    auto* res = static_cast<ves::GpuResource*>(resource);
    if (res == nullptr || res->res == nullptr) {
        return VES_INVALID_HANDLE;
    }
    try {
        faiss::Index* cpu = nullptr;
        int ret = ves::read_cpu_index_from_file(path, &cpu);
        if (ret != VES_OK) {
            return ret;
        }
        // read_cpu_index_from_file guarantees the file holds an IndexIVFPQ.
        auto* cpu_ivf = static_cast<faiss::IndexIVFPQ*>(cpu);
        faiss::gpu::GpuIndexIVFPQConfig cfg;
        cfg.device = device_id >= 0 ? device_id : res->device_id;
        cfg.indicesOptions = faiss::gpu::INDICES_64_BIT;
        auto* gidx = new ves::GpuIndex();
        gidx->device_id = cfg.device;
        gidx->resource = res;
        gidx->index = new faiss::gpu::GpuIndexIVFPQ(res->res, cpu_ivf, cfg);
        gidx->nprobe = cpu_ivf->nprobe;
        delete cpu;
        if (gidx->index == nullptr) {
            delete gidx;
            return VES_INVALID_HANDLE;
        }
        *idx = gidx;
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_load_gpu_index");
    }
}

int ves_free_gpu_index(ves_gpu_index_t index) {
    try {
        delete static_cast<ves::GpuIndex*>(index);
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_free_gpu_index");
    }
}

// ---------------------------------------------------------------------------
// Shards (GPU indices are moved in as CPU copies)
// ---------------------------------------------------------------------------

int ves_create_index_shards(int dimension, int threaded, ves_index_shards_t* shards) {
    if (shards == nullptr) {
        return VES_INVALID_ARGUMENT;
    }
    try {
        auto* s = new ves::Shards();
        s->dimension = dimension;
        s->threaded = threaded;
        *shards = s;
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_create_index_shards");
    }
}

int ves_free_index_shards(ves_index_shards_t shards) {
    try {
        delete static_cast<ves::Shards*>(shards);
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_free_index_shards");
    }
}

int ves_add_gpu_index_to_shards(ves_gpu_index_t index, ves_index_shards_t shards) {
    auto* gidx = static_cast<ves::GpuIndex*>(index);
    auto* s = static_cast<ves::Shards*>(shards);
    if (gidx == nullptr || gidx->index == nullptr || s == nullptr) {
        return VES_INVALID_HANDLE;
    }
    try {
        if (gidx->index->d != s->dimension) {
            return VES_INVALID_ARGUMENT;
        }
        faiss::Index* cpu = faiss::gpu::index_gpu_to_cpu(gidx->index);
        s->indices.push_back(cpu);
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_add_gpu_index_to_shards");
    }
}

int ves_add_shards_to_shards(ves_index_shards_t from, ves_index_shards_t to) {
    auto* sfrom = static_cast<ves::Shards*>(from);
    auto* sto = static_cast<ves::Shards*>(to);
    if (sfrom == nullptr || sto == nullptr) {
        return VES_INVALID_HANDLE;
    }
    try {
        if (sfrom->dimension != sto->dimension) {
            return VES_INVALID_ARGUMENT;
        }
        sto->indices.insert(sto->indices.end(), sfrom->indices.begin(), sfrom->indices.end());
        sfrom->indices.clear();  // ownership moved into `to`
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_add_shards_to_shards");
    }
}

int ves_search_index_shards(ves_index_shards_t shards, long n, const float* x, int k,
                            float* distances, long* ids) {
    auto* s = static_cast<ves::Shards*>(shards);
    if (s == nullptr) {
        return VES_INVALID_HANDLE;
    }
    if (x == nullptr || distances == nullptr || ids == nullptr || n < 0 || k < 0) {
        return VES_INVALID_ARGUMENT;
    }
    try {
        if (s->indices.empty()) {
            for (long i = 0; i < n * k; ++i) {
                distances[i] = std::numeric_limits<float>::max();
                ids[i] = -1;
            }
            return VES_OK;
        }
        const size_t nshards = s->indices.size();
        std::vector<float> dis_buf(n * k * nshards);
        std::vector<faiss::idx_t> id_buf(n * k * nshards);

        if (s->threaded) {
#pragma omp parallel for
            for (long i = 0; i < long(nshards); ++i) {
                s->indices[i]->search(n, x, k, dis_buf.data() + i * n * k,
                                      id_buf.data() + i * n * k);
            }
        } else {
            for (size_t i = 0; i < nshards; ++i) {
                s->indices[i]->search(n, x, k, dis_buf.data() + i * n * k,
                                      id_buf.data() + i * n * k);
            }
        }

        // k-best merge across shards, per query, sorted by (distance, id).
        for (long qi = 0; qi < n; ++qi) {
            std::vector<std::pair<float, faiss::idx_t>> cand;
            cand.reserve(nshards * k);
            for (size_t i = 0; i < nshards; ++i) {
                const float* dis = dis_buf.data() + (i * n + qi) * k;
                const faiss::idx_t* idd = id_buf.data() + (i * n + qi) * k;
                for (int j = 0; j < k; ++j) {
                    if (idd[j] >= 0) {
                        cand.emplace_back(dis[j], idd[j]);
                    }
                }
            }
            std::sort(cand.begin(), cand.end());
            float* out_dis = distances + qi * k;
            faiss::idx_t* out_ids = ids + qi * k;
            size_t m = std::min(cand.size(), size_t(k));
            for (size_t j = 0; j < m; ++j) {
                out_dis[j] = cand[j].first;
                out_ids[j] = cand[j].second;
            }
            for (size_t j = m; j < size_t(k); ++j) {
                out_dis[j] = std::numeric_limits<float>::max();
                out_ids[j] = -1;
            }
        }
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_search_index_shards");
    }
}

// ---------------------------------------------------------------------------
// GPU <-> CPU conversion
// ---------------------------------------------------------------------------

int ves_index_gpu_to_cpu(ves_gpu_index_t index, ves_cpu_index_t* cpu_index,
                         int reclaim_gpu_memory) {
    if (cpu_index == nullptr) {
        return VES_INVALID_ARGUMENT;
    }
    auto* gidx = static_cast<ves::GpuIndex*>(index);
    if (gidx == nullptr || gidx->index == nullptr) {
        return VES_INVALID_HANDLE;
    }
    try {
        auto* cpu = new ves::CpuIndex();
        cpu->index = faiss::gpu::index_gpu_to_cpu(gidx->index);
        if (cpu->index == nullptr) {
            delete cpu;
            return VES_NOMEMORY;
        }
        if (reclaim_gpu_memory) {
            delete gidx->index;
            gidx->index = nullptr;
        }
        *cpu_index = cpu;
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_index_gpu_to_cpu");
    }
}

int ves_index_cpu_to_gpu(ves_cpu_index_t index, ves_resource_t resource, int device_id,
                         ves_gpu_index_t* gpu_index) {
    if (gpu_index == nullptr) {
        return VES_INVALID_ARGUMENT;
    }
    auto* cpu = static_cast<ves::CpuIndex*>(index);
    auto* res = static_cast<ves::GpuResource*>(resource);
    if (cpu == nullptr || cpu->index == nullptr || res == nullptr || res->res == nullptr) {
        return VES_INVALID_HANDLE;
    }
    try {
        int dev = device_id >= 0 ? device_id : res->device_id;
        auto* gidx = new ves::GpuIndex();
        gidx->device_id = dev;
        gidx->resource = res;
        gidx->index = dynamic_cast<faiss::gpu::GpuIndexIVFPQ*>(
            faiss::gpu::index_cpu_to_gpu(res->res, dev, cpu->index));
        if (gidx->index == nullptr) {
            delete gidx;
            return VES_INVALID_HANDLE;
        }
        gidx->nprobe = cpu->as_ivfpq() ? cpu->as_ivfpq()->nprobe : 1;
        *gpu_index = gidx;
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_index_cpu_to_gpu");
    }
}

// ---------------------------------------------------------------------------
// GPU index status / ids / memory
// ---------------------------------------------------------------------------

int ves_get_gpu_index_status(ves_gpu_index_t index, ves_index_status_t* status) {
    if (status == nullptr) {
        return VES_INVALID_ARGUMENT;
    }
    auto* gidx = static_cast<ves::GpuIndex*>(index);
    if (gidx == nullptr || gidx->index == nullptr) {
        return VES_INVALID_HANDLE;
    }
    try {
        std::memset(status, 0, sizeof(*status));
        status->dimension = gidx->index->d;
        status->index_size = ves::gpu_ntotal(gidx->index);
        status->is_trained = gidx->index->is_trained ? 1 : 0;
        status->nlist = static_cast<int>(gidx->index->getNumLists());
        status->nprobe = static_cast<int>(gidx->nprobe);
        status->max_list_size = static_cast<int>(ves::gpu_max_list_size(gidx->index));
        status->subQuantizers = gidx->index->pq.M;
        status->bitsPerCode = gidx->index->pq.nbits;
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_get_gpu_index_status");
    }
}

int ves_get_gpu_index_ids(ves_gpu_index_t index, long* ids) {
    auto* gidx = static_cast<ves::GpuIndex*>(index);
    if (gidx == nullptr || gidx->index == nullptr) {
        return VES_INVALID_HANDLE;
    }
    if (ids == nullptr) {
        return VES_INVALID_ARGUMENT;
    }
    try {
        long* out = ids;
        const long nlist = gidx->index->getNumLists();
        for (long l = 0; l < nlist; ++l) {
            const std::vector<faiss::idx_t> list_ids = gidx->index->getListIndices(l);
            for (faiss::idx_t id : list_ids) {
                *out++ = static_cast<long>(id);
            }
        }
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_get_gpu_index_ids");
    }
}

long ves_get_gpu_max_may_reserve_memory(ves_gpu_index_t index, long n) {
    auto* gidx = static_cast<ves::GpuIndex*>(index);
    if (gidx == nullptr || gidx->index == nullptr) {
        return VES_INVALID_HANDLE;
    }
    // Estimate: memory still reservable on the device given n more vectors.
    // Per-vector cost on device: PQ code (M bytes for nbits=8) + 8-byte id.
    try {
        size_t free_bytes = 0, total_bytes = 0;
        cudaSetDevice(gidx->device_id);
        cudaMemGetInfo(&free_bytes, &total_bytes);
        const long per_vec = (gidx->index->pq.M * gidx->index->pq.nbits + 7) / 8 + 8;
        long needed = n > 0 ? n * per_vec : 0;
        return needed >= long(free_bytes) ? 0 : long(free_bytes) - needed;
    } catch (...) {
        return ves::exception_to_error("ves_get_gpu_max_may_reserve_memory");
    }
}

// ---------------------------------------------------------------------------
// GPU train / add / search
// ---------------------------------------------------------------------------

int ves_train_gpu_index(ves_gpu_index_t index, long n, const float* x) {
    auto* gidx = static_cast<ves::GpuIndex*>(index);
    if (gidx == nullptr || gidx->index == nullptr) {
        return VES_INVALID_HANDLE;
    }
    if (x == nullptr || n < 0) {
        return VES_INVALID_ARGUMENT;
    }
    try {
        gidx->index->train(n, x);
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_train_gpu_index");
    }
}

int ves_add_index_batch(ves_gpu_index_t index, long n, const float* x, const long* ids) {
    auto* gidx = static_cast<ves::GpuIndex*>(index);
    if (gidx == nullptr || gidx->index == nullptr) {
        return VES_INVALID_HANDLE;
    }
    if (x == nullptr || ids == nullptr || n < 0) {
        return VES_INVALID_ARGUMENT;
    }
    try {
        gidx->index->add_with_ids(n, x,
                             reinterpret_cast<const faiss::idx_t*>(ids));
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_add_index_batch");
    }
}

int ves_search_index(ves_gpu_index_t index, long n, const float* x, int k,
                     float* distances, long* ids) {
    auto* gidx = static_cast<ves::GpuIndex*>(index);
    if (gidx == nullptr || gidx->index == nullptr) {
        return VES_INVALID_HANDLE;
    }
    if (x == nullptr || distances == nullptr || ids == nullptr || n < 0 || k < 0) {
        return VES_INVALID_ARGUMENT;
    }
    try {
        // Modern faiss has no setNumProbes; the API's nprobe is applied per
        // search through SearchParametersIVF.
        faiss::SearchParametersIVF params;
        params.nprobe = gidx->nprobe;
        gidx->index->search(n, x, k, distances,
                            reinterpret_cast<faiss::idx_t*>(ids), &params);
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_search_index");
    }
}

int ves_search_index_with_timespace_filter(ves_gpu_index_t index, long n, const float* x,
                                           int k, ves_timespace_filter_t* filter,
                                           float* distances, long* ids) {
    auto* gidx = static_cast<ves::GpuIndex*>(index);
    if (gidx == nullptr || gidx->index == nullptr) {
        return VES_INVALID_HANDLE;
    }
    if (x == nullptr || filter == nullptr || distances == nullptr || ids == nullptr ||
        n < 0 || k < 0) {
        return VES_INVALID_ARGUMENT;
    }
    try {
        // The filter applies per candidate id, which the GPU search kernel
        // has no hook for: copy to CPU, filter there, discard the copy.
        ves::TimespaceFilter ts = ves::TimespaceFilter::from_ves(*filter);
        faiss::Index* cpu = faiss::gpu::index_gpu_to_cpu(gidx->index);
        int ret = ves::search_ivfpq_with_filter(static_cast<faiss::IndexIVFPQ*>(cpu), n, x,
                                                k, ts, distances, ids);
        delete cpu;
        return ret;
    } catch (...) {
        return ves::exception_to_error("ves_search_index_with_timespace_filter");
    }
}

// ---------------------------------------------------------------------------
// Device properties
// ---------------------------------------------------------------------------

int ves_get_device_properties(ves_device_properties* prop, int device) {
    if (prop == nullptr) {
        return VES_INVALID_ARGUMENT;
    }
    try {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, device);
        prop->major = p.major;
        prop->minor = p.minor;
        std::snprintf(prop->name, sizeof(prop->name), "%s", p.name);
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_get_device_properties");
    }
}

}  // extern "C"
