// Filtered search kernel for faiss::IndexIVFPQ.
//
// Runs an IVF-PQ scan with a per-candidate filter applied (stock faiss has no
// such hook). Implemented with public faiss APIs only, so no faiss patching
// is required.
#ifndef VES_SEARCH_FILTER_H_
#define VES_SEARCH_FILTER_H_

#include "ves_internal.h"
#include "id_filter.h"

namespace ves {

// Runs the IVF-PQ scan of `index` with `filter` applied per candidate id.
//
//   - list selection follows faiss: for each query the coarse quantizer
//     returns the index->nprobe nearest lists, which are scanned in order;
//   - distances are exact L2 against the reconstructed vectors (residual encoding
//     is undone when index->by_residual is set), matching faiss's own
//     precomputed-table arithmetic up to float summation order;
//   - candidates failing the filter are skipped, so up to k passing results
//     are returned per query; slots that stay empty are filled with
//     FLT_MAX / -1;
//   - results are returned sorted by (distance, id) ascending.
//
// Returns 0 on success, a VES_* error code otherwise.
int search_ivfpq_with_filter(faiss::IndexIVFPQ* index, long n, const float* x, int k,
                             const TimespaceFilter& filter, float* distances, long* ids);

}  // namespace ves

#endif  // VES_SEARCH_FILTER_H_
