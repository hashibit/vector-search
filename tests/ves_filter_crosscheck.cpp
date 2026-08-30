// Cross-check of the timespace-filtered search kernel against an independent
// brute-force scan built on other public faiss machinery
// (ProductQuantizer::decode + per-candidate L2). Both compute exact L2
// distances of the PQ reconstruction; agreement within float tolerance
// validates ves_search_cpu_index_with_timespace_filter.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>

#include "vector_search.h"

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

namespace {

constexpr int kDim = 32;
constexpr int kNlist = 64;
constexpr int kNprobe = 16;
constexpr int kM = 8;
constexpr int kBits = 8;
constexpr long kNTrain = 5000;
constexpr long kNAdd = 5000;
constexpr int kK = 20;

uint32_t g_seed = 0xdeadbeef;
uint32_t next_rand() {
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 17;
    g_seed ^= g_seed << 5;
    return g_seed;
}
void fill_rand(float* x, long n) {
    for (long i = 0; i < n; ++i) x[i] = (int(next_rand() % 2001) - 1000) / 1000.0f;
}

long make_id(uint32_t time_sec, int camera) {
    return (static_cast<long>(time_sec) << 7) | (camera & 0x7f);
}

bool passes(const ves_timespace_filter_t& f, long id) {
    const int cam = int(id & 0x7f);
    if (!((f.camera_mask[cam / 8] >> (cam % 8)) & 1)) return false;
    const uint32_t t = uint32_t(uint64_t(id) >> 7);
    if ((f.time_range[0] != 0 || f.time_range[1] != 0) &&
        (t < f.time_range[0] || t > f.time_range[1])) {
        return false;
    }
    return true;
}

}  // namespace

int main() {
    ves_index_config_t cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.dimension = kDim;
    cfg.nlist = kNlist;
    cfg.nprobe = kNprobe;
    cfg.subQuantizers = kM;
    cfg.bitsPerCode = kBits;

    ves_cpu_index_t idx = nullptr;
    CHECK(ves_init_cpu_index(&cfg, &idx) == VES_OK);

    std::vector<float> x_train(kNTrain * kDim);
    fill_rand(x_train.data(), kNTrain * kDim);
    CHECK(ves_train_cpu_index(idx, kNTrain, x_train.data()) == VES_OK);

    std::vector<float> x_add(kNAdd * kDim);
    fill_rand(x_add.data(), kNAdd * kDim);
    std::vector<long> ids(kNAdd);
    for (long i = 0; i < kNAdd; ++i) {
        ids[i] = make_id(uint32_t(500 + i % 100), int(i % 128));
    }
    CHECK(ves_add_cpu_index_batch(idx, kNAdd, x_add.data(), ids.data()) == VES_OK);

    // Dump to disk so the brute-force side can read the raw faiss index.
    const char* path = "ves_filter_crosscheck_index.bin";
    CHECK(ves_write_cpu_index(idx, path) == VES_OK);

    faiss::Index* raw = faiss::read_index(path);
    CHECK(raw != nullptr);
    auto* ivf = dynamic_cast<faiss::IndexIVFPQ*>(raw);
    CHECK(ivf != nullptr);

    // Query + restrictive filter: time window [520, 525], cameras 0..15 only.
    std::vector<float> q(kDim);
    fill_rand(q.data(), kDim);
    ves_timespace_filter_t f;
    std::memset(&f, 0, sizeof(f));
    f.camera_mask[0] = 0xff;
    f.camera_mask[1] = 0xff;
    f.time_range[0] = 520;
    f.time_range[1] = 525;

    std::vector<float> got_dis(kK);
    std::vector<long> got_ids(kK);
    CHECK(ves_search_cpu_index_with_timespace_filter(idx, 1, q.data(), kK, &f,
                                                     got_dis.data(), got_ids.data()) == VES_OK);

    // Brute force: decode every stored vector in the nprobe nearest lists
    // (same list selection as the kernel) and take the k best passing ones.
    size_t nprobe = std::min(ivf->nprobe, ivf->nlist);
    std::vector<faiss::idx_t> assign(nprobe);
    std::vector<float> coarse_dis(nprobe);
    ivf->quantizer->search(1, q.data(), nprobe, coarse_dis.data(), assign.data());

    std::vector<float> centroid(kDim);
    std::vector<float> decoded(kDim);
    std::vector<std::pair<float, long>> cand;
    const faiss::InvertedLists* il = ivf->invlists;
    for (size_t j = 0; j < nprobe; ++j) {
        const faiss::idx_t l = assign[j];
        if (l < 0 || size_t(l) >= ivf->nlist) continue;
        const size_t n = il->list_size(l);
        if (n == 0) continue;
        const faiss::idx_t* list_ids = il->get_ids(l);
        const uint8_t* list_codes = il->get_codes(l);
        if (ivf->by_residual) {
            ivf->quantizer->reconstruct(l, centroid.data());
        }
        for (size_t t = 0; t < n; ++t) {
            if (!passes(f, list_ids[t])) continue;
            ivf->pq.decode(list_codes + t * ivf->pq.code_size, decoded.data());
            if (ivf->by_residual) {
                for (int dim = 0; dim < kDim; ++dim) decoded[dim] += centroid[dim];
            }
            float d = 0.0f;
            for (int dim = 0; dim < kDim; ++dim) {
                float dd = q[dim] - decoded[dim];
                d += dd * dd;
            }
            cand.emplace_back(d, long(list_ids[t]));
        }
    }
    std::sort(cand.begin(), cand.end());
    const size_t expect = std::min(cand.size(), size_t(kK));
    CHECK(expect >= 1);

    for (size_t j = 0; j < expect; ++j) {
        CHECK(got_ids[j] == cand[j].second);
        CHECK(std::fabs(got_dis[j] - cand[j].first) < 1e-3f * std::max(1.0f, cand[j].first));
    }
    // slots beyond the passing count are padded
    for (size_t j = expect; j < size_t(kK); ++j) {
        CHECK(got_ids[j] == -1);
        CHECK(got_dis[j] == std::numeric_limits<float>::max());
    }

    std::printf("ves filter crosscheck: OK (%zu passing candidates, %zu compared)\n",
                cand.size(), expect);

    delete raw;
    CHECK(ves_free_cpu_index(idx) == VES_OK);
    std::remove(path);
    return 0;
}
