// Internal types of the GPU surface (vector_search_gpu.cpp).
//
// Only included from the CUDA build: everything here needs the faiss GPU
// headers and the CUDA runtime. The CPU-only build compiles
// vector_search_gpu_stub.cpp instead and sees none of this file.
#ifndef VES_GPU_INTERNAL_H_
#define VES_GPU_INTERNAL_H_

#include <vector>

#include <faiss/Index.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <cuda_runtime.h>

#include "ves_internal.h"

namespace ves {

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

// Collection of standalone indices. Part of the GPU surface: the whole shards
// API is stubbed out (-7) in the CPU-only build, so only the CUDA build
// instantiates this.
struct Shards {
    int dimension = 0;
    int threaded = 0;
    std::vector<faiss::Index*> indices;  // owned

    ~Shards() {
        for (faiss::Index* i : indices) delete i;
    }
};

}  // namespace ves

#endif  // VES_GPU_INTERNAL_H_
