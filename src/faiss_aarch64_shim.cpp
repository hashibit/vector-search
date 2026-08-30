// Workaround for a faiss build gap on arm64: the __aarch64__ branch of
// utils/distances_simd.cpp defines fvec_inner_product / fvec_L2sqr_ny but
// forgets the fvec_inner_products_ny wrapper, so it stays undefined in
// libfaiss.a. Confirmed for faiss 1.6.5 and 1.14.3. faiss only ever calls it
// through ProductQuantizer (which we do use, via compute_distance_table).
// Provide the wrapper backed by the ref implementation, which faiss does
// define on every platform.
//
// This is a no-op (empty TU) on any platform where faiss provides the symbol
// itself; keep the guard in sync with the fetched faiss version.
#if defined(__aarch64__)
#include <cstddef>

namespace faiss {
// Defined in faiss/utils/distances_simd.cpp but not declared in any header.
void fvec_inner_products_ny_ref(float* dis, const float* x, const float* y,
                                size_t d, size_t ny);

void fvec_inner_products_ny(float* dis, const float* x, const float* y,
                            size_t d, size_t ny) {
    fvec_inner_products_ny_ref(dis, x, y, d, ny);
}

}  // namespace faiss
#endif  // __aarch64__
