// Internal helpers shared by the vector_search implementation.
//
// Handles, the id-packing contract and the exception-to-error-code mapping
// live here (the C API's error codes are documented in README.md).
#ifndef VES_INTERNAL_H_
#define VES_INTERNAL_H_

#include <cstdint>
#include <cstdio>
#include <new>
#include <vector>

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/clone_index.h>
#include <faiss/impl/AuxIndexStructures.h>  // IDSelectorBatch (faiss 1.6.x layout)
#include <faiss/impl/FaissException.h>
#include <faiss/index_io.h>

#ifdef VES_WITH_CUDA
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <cuda_runtime.h>
#endif

#include "vector_search.h"

// The C API moves ids through `long`, faiss through idx_t (int64_t). The two
// must agree in width; on Linux they are the same type, on macOS long == 8
// bytes too. Windows (long == 4) is out of scope.
static_assert(sizeof(long) == sizeof(faiss::idx_t),
              "vector_search requires 64-bit long");

namespace ves {

// Every GPU entry point returns -7 when the library is built without CUDA;
// with VES_WITH_CUDA the GPU paths are implemented against the modern faiss
// GPU API (>= 1.7, written and checked against the faiss 1.14 headers; NOT
// compiled or run here, requires a CUDA toolchain).
constexpr int kGpuUnavailable = -7;

// ---------------------------------------------------------------------------
// Feature-id contract (time + camera packing)
// ---------------------------------------------------------------------------
// A feature id encodes the (time, camera) of the frame it was extracted from:
//
//     bits 0..6   camera id   (0..127, matches camera_mask[128/8])
//     bits 7..63  capture time (unix seconds, uint32_t in practice)
//
//     id = (time_sec << 7) | camera_id
//
// The timespace filter (ves_search_*_index_with_timespace_filter) decodes ids
// with the inverse mapping and rejects ids whose camera bit is clear or whose
// time falls outside the requested range.
//
// NOTE: the packing is not pinned down by the header alone; the layout above
// is the only one consistent with it (128 cameras -> 7 bits, uint32
// time_range). Change kCameraBits and the two helpers below if the real
// contract differs.
constexpr int kCameraBits = 7;
constexpr long kCameraIdMask = (1L << kCameraBits) - 1;

inline long make_feature_id(uint32_t time_sec, int camera_id) {
    return (static_cast<long>(time_sec) << kCameraBits) | (camera_id & kCameraIdMask);
}

inline uint32_t id_time(long id) {
    return static_cast<uint32_t>(static_cast<uint64_t>(id) >> kCameraBits);
}

inline int id_camera(long id) {
    return static_cast<int>(id & kCameraIdMask);
}

// ---------------------------------------------------------------------------
// Handles
// ---------------------------------------------------------------------------

struct CpuIndex {
    faiss::Index* index = nullptr;  // IndexIVFPQ in practice

    ~CpuIndex() { delete index; }

    faiss::IndexIVFPQ* as_ivfpq() { return dynamic_cast<faiss::IndexIVFPQ*>(index); }
};

#ifdef VES_WITH_CUDA
struct GpuResource {
    int device_id = 0;
    faiss::gpu::StandardGpuResources* res = nullptr;

    ~GpuResource() { delete res; }
};

struct GpuIndex {
    int device_id = 0;
    GpuResource* resource = nullptr;  // borrowed, not owned
    faiss::gpu::GpuIndexIVFPQ* index = nullptr;
    // Modern faiss (>= 1.7) has no setNumProbes; nprobe is applied per search
    // through SearchParametersIVF. Keep the API's nprobe here.
    size_t nprobe = 1;

    ~GpuIndex() { delete index; }
};
#endif

// Collection of standalone indices. Stubs in the CPU-only build (-7); the
// semantics below are the natural reading of the header.
struct Shards {
    int dimension = 0;
    int threaded = 0;
    std::vector<faiss::Index*> indices;  // owned

    ~Shards() {
        for (faiss::Index* i : indices) delete i;
    }
};

// ---------------------------------------------------------------------------
// Exception mapping (the C boundary must never throw)
// ---------------------------------------------------------------------------
// Call from within a catch block: maps the active exception to a VES_* code.
inline int exception_to_error(const char* fn) noexcept {
    try {
        throw;
    } catch (const std::bad_alloc&) {
        return VES_NOMEMORY;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "vector_search: %s: %s\n", fn, e.what());
        return VES_SYSCALL_ERROR;
    } catch (...) {
        return VES_SYSCALL_ERROR;
    }
}

}  // namespace ves

#endif  // VES_INTERNAL_H_
