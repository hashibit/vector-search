#include "ves_search_filter.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ves {

namespace {

// Generic PQ bit unpacking (faiss 1.6.x layout): sub-quantizer m's index
// occupies bits [m*nbits, (m+1)*nbits) of the code. Degenerates to a plain
// byte load for nbits == 8.
int unpack_code_index(const uint8_t* code, int m, int nbits) {
    int idx = 0;
    const int bit0 = m * nbits;
    for (int j = 0; j < nbits; ++j) {
        const int bit = bit0 + j;
        idx |= ((code[bit >> 3] >> (bit & 7)) & 1) << j;
    }
    return idx;
}

// Max-heap over (distance, id). After `build()` the heap root is the current
// k-th worst result; any candidate closer than the root replaces it.
void sift_down(float* dis, faiss::idx_t* ids, int size, int i) {
    for (;;) {
        int l = 2 * i + 1;
        int r = 2 * i + 2;
        int largest = i;
        if (l < size && dis[l] > dis[largest]) largest = l;
        if (r < size && dis[r] > dis[largest]) largest = r;
        if (largest == i) break;
        std::swap(dis[i], dis[largest]);
        std::swap(ids[i], ids[largest]);
        i = largest;
    }
}

void build_heap(float* dis, faiss::idx_t* ids, int size) {
    for (int i = size / 2 - 1; i >= 0; --i) sift_down(dis, ids, size, i);
}

struct LessByDisThenId {
    bool operator()(const std::pair<float, faiss::idx_t>& a,
                    const std::pair<float, faiss::idx_t>& b) const {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    }
};

}  // namespace

int search_ivfpq_with_filter(faiss::IndexIVFPQ* index, long n, const float* x, int k,
                             const TimespaceFilter& filter, float* distances, long* ids) {
    // Empty result slots are padded with FLT_MAX / -1.
    for (long i = 0; i < n * k; ++i) {
        distances[i] = std::numeric_limits<float>::max();
        ids[i] = -1;
    }
    if (k <= 0 || n <= 0) {
        return VES_OK;
    }
    if (!index->is_trained) {
        // Stock faiss search on an untrained IVFPQ returns all-FLT_MAX/-1.
        return VES_OK;
    }

    const int d = index->d;
    const size_t nlist = index->nlist;
    size_t nprobe = index->nprobe;
    if (nprobe > nlist) nprobe = nlist;
    if (nprobe == 0) return VES_OK;

    // Coarse quantization: nprobe nearest centroids per query, same call the
    // stock IndexIVF::search makes.
    std::vector<faiss::idx_t> assign(n * nprobe);
    std::vector<float> coarse_dis(n * nprobe);
    index->quantizer->search(n, x, nprobe, coarse_dis.data(), assign.data());

    const faiss::InvertedLists* il = index->invlists;
    const int M = index->pq.M;
    const int nbits = index->pq.nbits;
    const size_t ksub = size_t(1) << nbits;
    const size_t code_size = index->pq.code_size;
    std::vector<float> centroid(d);
    std::vector<float> residual(d);
    std::vector<float> qtable(M * ksub);
    std::vector<float> heap_dis(k);
    std::vector<faiss::idx_t> heap_ids(k);
    std::vector<std::pair<float, faiss::idx_t>> picked(k);

    for (long qi = 0; qi < n; ++qi) {
        const float* q = x + qi * d;
        float* out_dis = distances + qi * k;
        faiss::idx_t* out_ids =
            reinterpret_cast<faiss::idx_t*>(ids + qi * k);

        int count = 0;  // passing candidates collected so far

        for (size_t j = 0; j < nprobe; ++j) {
            const faiss::idx_t l = assign[qi * nprobe + j];
            if (l < 0 || size_t(l) >= nlist) continue;

            const size_t list_size = il->list_size(l);
            if (list_size == 0) continue;

            const faiss::idx_t* list_ids = il->get_ids(l);
            const uint8_t* list_codes = il->get_codes(l);
            if (list_ids == nullptr || list_codes == nullptr) continue;

            // Query-to-subquantizer distance table (M * 2^nbits floats), the
            // same table the stock precomputed-table path builds. With
            // by_residual (faiss default) the codes encode the residual w.r.t.
            // the list centroid, so the query is centered first.
            const float* table_x = q;
            if (index->by_residual) {
                index->quantizer->reconstruct(l, centroid.data());
                for (int t = 0; t < d; ++t) residual[t] = q[t] - centroid[t];
                table_x = residual.data();
            }
            index->pq.compute_distance_table(table_x, qtable.data());

            for (size_t t = 0; t < list_size; ++t) {
                const faiss::idx_t id = list_ids[t];
                if (!filter.pass(id)) continue;
                const uint8_t* code = list_codes + t * code_size;
                float dist = 0.0f;
                for (int m = 0; m < M; ++m) {
                    dist += qtable[m * ksub + unpack_code_index(code, m, nbits)];
                }
                if (count < k) {
                    heap_dis[count] = dist;
                    heap_ids[count] = id;
                    ++count;
                    if (count == k) {
                        build_heap(heap_dis.data(), heap_ids.data(), k);
                    }
                } else if (dist < heap_dis[0]) {
                    heap_dis[0] = dist;
                    heap_ids[0] = id;
                    sift_down(heap_dis.data(), heap_ids.data(), k, 0);
                }
            }
        }

        // Copy out the passing results, sorted by (distance, id) ascending.
        // Slots beyond `count` keep the FLT_MAX / -1 padding.
        for (int t = 0; t < count; ++t) {
            picked[t] = std::make_pair(heap_dis[t], heap_ids[t]);
        }
        std::sort(picked.begin(), picked.begin() + count, LessByDisThenId());
        for (int t = 0; t < count; ++t) {
            out_dis[t] = picked[t].first;
            out_ids[t] = picked[t].second;
        }
    }

    return VES_OK;
}

}  // namespace ves
