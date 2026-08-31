// vector_search: CPU implementation of include/vector_search.h.
//
// Built on faiss: the API maps onto IndexIVFPQ (init -> train -> add ->
// search -> write/load) and faiss exceptions are mapped to the VES_* error
// codes of the header. The timespace filter is an IVF-PQ scan with a
// per-candidate filter hook (see ves_search_filter.cpp).
//
// The GPU surface of the header lives in its own TUs, chosen by CMake:
// vector_search_gpu.cpp (CUDA build) / vector_search_gpu_stub.cpp
// (CPU-only build).
#include "ves_internal.h"
#include "ves_search_filter.h"
#include "id_filter.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <vector>

namespace ves {

// ---------------------------------------------------------------------------
// Internal helpers (shared with the GPU build; declared in ves_internal.h)
// ---------------------------------------------------------------------------

// CPU index creation shared by init and load. Indexes are always IndexIVFPQ;
// other types are rejected at load time (see read_cpu_index_from_file).
faiss::IndexIVFPQ* new_ivfpq(const ves_index_config_t& config) {
    faiss::IndexFlatL2* quantizer = new faiss::IndexFlatL2(config.dimension);
    auto* index = new faiss::IndexIVFPQ(quantizer, config.dimension, config.nlist,
                                        config.subQuantizers, config.bitsPerCode);
    index->nprobe = config.nprobe;
    return index;
}

void fill_status_from_ivf(const faiss::Index* index, ves_index_status_t* status) {
    std::memset(status, 0, sizeof(*status));
    status->dimension = index->d;
    status->index_size = index->ntotal;
    status->is_trained = index->is_trained ? 1 : 0;

    const auto* ivf = dynamic_cast<const faiss::IndexIVFPQ*>(index);
    if (ivf == nullptr) {
        return;  // non-IVFPQ handle: IVF/PQ fields stay 0
    }
    status->nlist = static_cast<int>(ivf->nlist);
    status->nprobe = static_cast<int>(ivf->nprobe);
    status->subQuantizers = ivf->pq.M;
    status->bitsPerCode = ivf->pq.nbits;

    // max_list_size = largest inverted list, only meaningful once trained
    // (the max list_size over every inverted list).
    if (index->is_trained) {
        size_t max_size = 0;
        for (size_t l = 0; l < ivf->nlist; ++l) {
            max_size = std::max(max_size, ivf->invlists->list_size(l));
        }
        status->max_list_size = static_cast<int>(max_size);
    }
}

// Collects every id, list-major order (list 0 ids, then list 1 ids, ...).
void collect_ids(const faiss::IndexIVFPQ* ivf, long* out) {
    const faiss::InvertedLists* il = ivf->invlists;
    for (size_t l = 0; l < ivf->nlist; ++l) {
        const size_t n = il->list_size(l);
        if (n == 0) continue;
        const faiss::idx_t* list_ids = il->get_ids(l);
        for (size_t t = 0; t < n; ++t) {
            *out++ = list_ids[t];
        }
    }
}

int read_cpu_index_from_file(const char* path, faiss::Index** out) {
    if (path == nullptr || out == nullptr) {
        return VES_INVALID_ARGUMENT;
    }
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) {
        return VES_FILE_NOT_FOUND;
    }
    faiss::Index* index = nullptr;
    int ret = VES_OK;
    try {
        index = faiss::read_index(f, true);
    } catch (const std::exception&) {
        ret = VES_BAD_INDEX;  // corrupted file / bad magic
    }
    std::fclose(f);
    if (ret != VES_OK) {
        return ret;
    }
    if (dynamic_cast<faiss::IndexIVFPQ*>(index) == nullptr) {
        delete index;
        return VES_INVALID_HANDLE;  // loaded index is not an IndexIVFPQ
    }
    *out = index;
    return VES_OK;
}

}  // namespace ves

// ===========================================================================
// C API (CPU surface)
// ===========================================================================

extern "C" {

// ---------------------------------------------------------------------------
// CPU index lifecycle
// ---------------------------------------------------------------------------

int ves_init_cpu_index(ves_index_config_t* config, ves_cpu_index_t* idx) {
    if (idx == nullptr || config == nullptr) {
        return VES_INVALID_ARGUMENT;
    }
    // Note: no extra validation of dimension/nlist/... — the config goes
    // straight to faiss and errors surface as exceptions (bad_alloc -> -3).
    try {
        auto* cpu = new ves::CpuIndex();
        cpu->index = ves::new_ivfpq(*config);
        *idx = cpu;
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_init_cpu_index");
    }
}

int ves_load_cpu_index(const char* path, ves_cpu_index_t* idx) {
    if (idx == nullptr) {
        return VES_INVALID_ARGUMENT;
    }
    faiss::Index* index = nullptr;
    int ret = ves::read_cpu_index_from_file(path, &index);
    if (ret != VES_OK) {
        return ret;
    }
    try {
        auto* cpu = new ves::CpuIndex();
        cpu->index = index;
        *idx = cpu;
        return VES_OK;
    } catch (...) {
        delete index;
        return ves::exception_to_error("ves_load_cpu_index");
    }
}

int ves_free_cpu_index(ves_cpu_index_t index) {
    try {
        delete static_cast<ves::CpuIndex*>(index);
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_free_cpu_index");
    }
}

// ---------------------------------------------------------------------------
// CPU index status / ids
// ---------------------------------------------------------------------------

int ves_get_cpu_index_status(ves_cpu_index_t index, ves_index_status_t* status) {
    auto* cpu = static_cast<ves::CpuIndex*>(index);
    if (cpu == nullptr || cpu->index == nullptr) {
        return VES_INVALID_HANDLE;
    }
    if (status == nullptr) {
        return VES_INVALID_ARGUMENT;
    }
    try {
        ves::fill_status_from_ivf(cpu->index, status);
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_get_cpu_index_status");
    }
}

int ves_get_cpu_index_ids(ves_cpu_index_t index, long* ids) {
    auto* cpu = static_cast<ves::CpuIndex*>(index);
    if (cpu == nullptr || cpu->index == nullptr) {
        return VES_INVALID_HANDLE;
    }
    if (ids == nullptr) {
        return VES_INVALID_ARGUMENT;
    }
    try {
        auto* ivf = cpu->as_ivfpq();
        if (ivf == nullptr) {
            return VES_INVALID_HANDLE;
        }
        ves::collect_ids(ivf, ids);
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_get_cpu_index_ids");
    }
}

// ---------------------------------------------------------------------------
// CPU train / add / remove
// ---------------------------------------------------------------------------

int ves_train_cpu_index(ves_cpu_index_t index, long n, const float* x) {
    auto* cpu = static_cast<ves::CpuIndex*>(index);
    if (cpu == nullptr || cpu->index == nullptr) {
        return VES_INVALID_HANDLE;
    }
    if (x == nullptr || n < 0) {
        return VES_INVALID_ARGUMENT;
    }
    try {
        cpu->index->train(n, x);
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_train_cpu_index");
    }
}

int ves_add_cpu_index_batch(ves_cpu_index_t index, long n, const float* x, const long* ids) {
    auto* cpu = static_cast<ves::CpuIndex*>(index);
    if (cpu == nullptr || cpu->index == nullptr) {
        return VES_INVALID_HANDLE;
    }
    if (x == nullptr || ids == nullptr || n < 0) {
        return VES_INVALID_ARGUMENT;
    }
    try {
        cpu->index->add_with_ids(
            n, x, reinterpret_cast<const faiss::idx_t*>(ids));
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_add_cpu_index_batch");
    }
}

int ves_remove_cpu_index_ids(ves_cpu_index_t index, long n, const long* ids) {
    auto* cpu = static_cast<ves::CpuIndex*>(index);
    if (cpu == nullptr || cpu->index == nullptr) {
        return VES_INVALID_HANDLE;
    }
    if (ids == nullptr || n < 0) {
        return VES_INVALID_ARGUMENT;
    }
    try {
        // An IDSelectorBatch over the ids, then IndexIVF::remove_ids
        // (the removed count is discarded).
        faiss::IDSelectorBatch sel(n, reinterpret_cast<const faiss::idx_t*>(ids));
        cpu->index->remove_ids(sel);
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_remove_cpu_index_ids");
    }
}

// ---------------------------------------------------------------------------
// CPU search
// ---------------------------------------------------------------------------

int ves_search_cpu_index(ves_cpu_index_t index, long n, const float* x, int k,
                         float* distances, long* ids) {
    auto* cpu = static_cast<ves::CpuIndex*>(index);
    if (cpu == nullptr || cpu->index == nullptr) {
        return VES_INVALID_HANDLE;
    }
    if (x == nullptr || distances == nullptr || ids == nullptr || n < 0 || k < 0) {
        return VES_INVALID_ARGUMENT;
    }
    try {
        cpu->index->search(n, x, k, distances,
                               reinterpret_cast<faiss::idx_t*>(ids));  // output buffer
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_search_cpu_index");
    }
}

int ves_search_cpu_index_with_timespace_filter(ves_cpu_index_t index, long n, const float* x,
                                               int k, ves_timespace_filter_t* filter,
                                               float* distances, long* ids) {
    auto* cpu = static_cast<ves::CpuIndex*>(index);
    if (cpu == nullptr || cpu->index == nullptr) {
        return VES_INVALID_HANDLE;
    }
    if (x == nullptr || filter == nullptr || distances == nullptr || ids == nullptr ||
        n < 0 || k < 0) {
        return VES_INVALID_ARGUMENT;
    }
    try {
        auto* ivf = cpu->as_ivfpq();
        if (ivf == nullptr) {
            return VES_INVALID_HANDLE;
        }
        ves::TimespaceFilter ts = ves::TimespaceFilter::from_ves(*filter);
        return ves::search_ivfpq_with_filter(ivf, n, x, k, ts, distances, ids);
    } catch (...) {
        return ves::exception_to_error("ves_search_cpu_index_with_timespace_filter");
    }
}

int ves_clone_cpu_index(ves_cpu_index_t input_index, ves_cpu_index_t* output_index) {
    auto* cpu = static_cast<ves::CpuIndex*>(input_index);
    if (output_index == nullptr) {
        return VES_INVALID_ARGUMENT;
    }
    if (cpu == nullptr || cpu->index == nullptr) {
        return VES_INVALID_HANDLE;
    }
    try {
        // faiss::clone_index; a null result maps to VES_NOMEMORY (the clone
        // allocation failed).
        faiss::Index* clone = faiss::clone_index(cpu->index);
        if (clone == nullptr) {
            return VES_NOMEMORY;
        }
        auto* out = new ves::CpuIndex();
        out->index = clone;
        *output_index = out;
        return VES_OK;
    } catch (...) {
        return ves::exception_to_error("ves_clone_cpu_index");
    }
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

int ves_write_cpu_index(ves_cpu_index_t index, const char* filename) {
    auto* cpu = static_cast<ves::CpuIndex*>(index);
    if (cpu == nullptr || cpu->index == nullptr) {
        return VES_INVALID_HANDLE;
    }
    if (filename == nullptr) {
        return VES_INVALID_ARGUMENT;
    }
    // Error mapping:
    //   null index      -> VES_INVALID_HANDLE (-1)
    //   fopen failure   -> VES_FILE_NOT_FOUND (-2)
    //   fflush failure  -> errno printed, still fclose()ed; return of fclose wins
    //   fclose failure  -> VES_SYSCALL_ERROR (-6)
    // A write_index exception is mapped to VES_INVALID_HANDLE instead of
    // propagating past the C boundary.
    FILE* f = std::fopen(filename, "wb");
    if (f == nullptr) {
        return VES_FILE_NOT_FOUND;
    }
    try {
        faiss::write_index(cpu->index, f);
    } catch (...) {
        std::fclose(f);
        return VES_INVALID_HANDLE;
    }
    if (std::fflush(f) != 0) {
        std::fprintf(stderr, "vector_search: ves_write_cpu_index: fflush: %s\n",
                     std::strerror(errno));
    }
    if (std::fclose(f) != 0) {
        return VES_SYSCALL_ERROR;
    }
    return VES_OK;
}

}  // extern "C"
