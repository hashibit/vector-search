// Smoke test for the vector_search library.
//
// Exercises the CPU surface end to end (init -> train -> add -> search ->
// timespace filter -> clone -> save -> load -> remove). GPU assertions are
// mode-agnostic: in the default CPU-only build every GPU entry point must
// return -7 (GPU surface unavailable); with VES_WITH_CUDA they may
// return -1 for null handles.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vector_search.h"

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

#define CHECK_CODE(expr, want)                                             \
    do {                                                                   \
        int rc_ = (expr);                                                  \
        if (rc_ != (want)) {                                               \
            std::fprintf(stderr, "FAIL %s:%d: %s -> %d, want %d\n",        \
                         __FILE__, __LINE__, #expr, rc_, (want));          \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

namespace {

constexpr int kDim = 16;
constexpr int kNlist = 32;
constexpr int kNprobe = 8;
constexpr int kM = 4;
constexpr int kBits = 8;
constexpr long kNTrain = 2000;
constexpr long kNAdd = 2000;
constexpr int kK = 10;
constexpr long kTimeBase = 1000;

// Deterministic pseudo-random generator.
uint32_t g_seed = 0x12345678;
uint32_t next_rand() {
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 17;
    g_seed ^= g_seed << 5;
    return g_seed;
}
void fill_rand(float* x, long n) {
    for (long i = 0; i < n; ++i) {
        x[i] = (int(next_rand() % 2001) - 1000) / 1000.0f;  // [-1, 1]
    }
}

// id = (time << 7) | camera, mirrors the reference contract in ves_internal.h
long make_id(uint32_t time_sec, int camera) {
    return (static_cast<long>(time_sec) << 7) | (camera & 0x7f);
}
int id_camera(long id) { return int(id & 0x7f); }
uint32_t id_time(long id) { return uint32_t(uint64_t(id) >> 7); }

bool camera_allowed(long id, const unsigned char* mask) {
    int cam = id_camera(id);
    return (mask[cam / 8] >> (cam % 8)) & 1u;
}

void run_cpu_tests() {
    // ---- init -------------------------------------------------------------
    ves_index_config_t cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.dimension = kDim;
    cfg.nlist = kNlist;
    cfg.nprobe = kNprobe;
    cfg.subQuantizers = kM;
    cfg.bitsPerCode = kBits;

    ves_cpu_index_t idx = nullptr;
    CHECK_CODE(ves_init_cpu_index(&cfg, &idx), VES_OK);
    CHECK(idx != nullptr);
    // null config / null out
    CHECK_CODE(ves_init_cpu_index(nullptr, &idx), VES_INVALID_ARGUMENT);
    ves_cpu_index_t tmp = nullptr;
    CHECK_CODE(ves_init_cpu_index(&cfg, nullptr), VES_INVALID_ARGUMENT);
    (void)tmp;

    // ---- train & add ------------------------------------------------------
    std::vector<float> x_train(kNTrain * kDim);
    fill_rand(x_train.data(), kNTrain * kDim);
    CHECK_CODE(ves_train_cpu_index(idx, kNTrain, x_train.data()), VES_OK);

    std::vector<float> x_add(kNAdd * kDim);
    fill_rand(x_add.data(), kNAdd * kDim);
    std::vector<long> ids(kNAdd);
    for (long i = 0; i < kNAdd; ++i) {
        ids[i] = make_id(uint32_t(kTimeBase + (i % 50)), int(i % 128));
    }
    CHECK_CODE(ves_add_cpu_index_batch(idx, kNAdd, x_add.data(), ids.data()), VES_OK);

    // ---- status -----------------------------------------------------------
    ves_index_status_t st;
    std::memset(&st, 0, sizeof(st));
    CHECK_CODE(ves_get_cpu_index_status(idx, &st), VES_OK);
    CHECK(st.dimension == kDim);
    CHECK(st.index_size == kNAdd);
    CHECK(st.is_trained == 1);
    CHECK(st.nlist == kNlist);
    CHECK(st.nprobe == kNprobe);
    CHECK(st.subQuantizers == kM);
    CHECK(st.bitsPerCode == kBits);
    CHECK(st.max_list_size > 0);

    // ---- search -----------------------------------------------------------
    float query[kDim];
    std::memset(query, 0, sizeof(query));
    query[0] = 1.0f;
    std::vector<float> dis(kK);
    std::vector<long> sid(kK);
    CHECK_CODE(ves_search_cpu_index(idx, 1, query, kK, dis.data(), sid.data()), VES_OK);
    for (int j = 0; j < kK; ++j) {
        CHECK(sid[j] >= 0);
        CHECK(dis[j] >= 0.0f);
        if (j > 0) {
            CHECK(dis[j] >= dis[j - 1] - 1e-5f);
        }
    }

    // ---- timespace filter -------------------------------------------------
    ves_timespace_filter_t f;
    std::memset(&f, 0, sizeof(f));
    // camera mask: allow only cameras 0..7 (bit 0..7 of byte 0)
    f.camera_mask[0] = 0xff;
    std::vector<float> fdis(kK);
    std::vector<long> fsid(kK);
    CHECK_CODE(ves_search_cpu_index_with_timespace_filter(idx, 1, query, kK, &f, fdis.data(),
                                                          fsid.data()),
               VES_OK);
    for (int j = 0; j < kK; ++j) {
        CHECK(fsid[j] >= 0);
        CHECK(id_camera(fsid[j]) < 8);
    }
    // time range: only ids with time in [kTimeBase, kTimeBase+4]. The filter
    // applies during the PQ scan of the nprobe nearest lists, so fewer than k
    // candidates may pass; the padded slots are -1 / FLT_MAX.
    f.camera_mask[0] = 0xff;
    for (int i = 1; i < 16; ++i) f.camera_mask[i] = 0xff;
    f.time_range[0] = uint32_t(kTimeBase);
    f.time_range[1] = uint32_t(kTimeBase + 4);
    CHECK_CODE(ves_search_cpu_index_with_timespace_filter(idx, 1, query, kK, &f, fdis.data(),
                                                          fsid.data()),
               VES_OK);
    int passed = 0;
    for (int j = 0; j < kK; ++j) {
        if (fsid[j] < 0) continue;
        ++passed;
        uint32_t t = id_time(fsid[j]);
        CHECK(t >= f.time_range[0] && t <= f.time_range[1]);
    }
    CHECK(passed >= 1 && passed <= kK);
    // camera-only mask restriction
    std::memset(&f, 0, sizeof(f));
    f.camera_mask[0] = 0x01;  // only camera 0
    CHECK_CODE(ves_search_cpu_index_with_timespace_filter(idx, 1, query, kK, &f, fdis.data(),
                                                          fsid.data()),
               VES_OK);
    passed = 0;
    for (int j = 0; j < kK; ++j) {
        if (fsid[j] < 0) continue;
        ++passed;
        CHECK(camera_allowed(fsid[j], f.camera_mask));
    }
    CHECK(passed >= 1);
    // arg validation: null filter -> -4
    CHECK_CODE(ves_search_cpu_index_with_timespace_filter(idx, 1, query, kK, nullptr,
                                                          fdis.data(), fsid.data()),
               VES_INVALID_ARGUMENT);

    // ---- ids --------------------------------------------------------------
    std::vector<long> got_ids(kNAdd);
    CHECK_CODE(ves_get_cpu_index_ids(idx, got_ids.data()), VES_OK);
    std::vector<long> expected(ids.begin(), ids.end());
    std::sort(expected.begin(), expected.end());
    std::sort(got_ids.begin(), got_ids.end());
    CHECK(got_ids == expected);

    // ---- clone ------------------------------------------------------------
    ves_cpu_index_t clone = nullptr;
    CHECK_CODE(ves_clone_cpu_index(idx, &clone), VES_OK);
    CHECK(clone != nullptr);
    std::vector<float> cdis(kK);
    std::vector<long> csid(kK);
    CHECK_CODE(ves_search_cpu_index(clone, 1, query, kK, cdis.data(), csid.data()), VES_OK);
    CHECK(cdis[0] >= 0.0f);
    ves_index_status_t cst;
    std::memset(&cst, 0, sizeof(cst));
    CHECK_CODE(ves_get_cpu_index_status(clone, &cst), VES_OK);
    CHECK(cst.index_size == kNAdd);
    CHECK_CODE(ves_free_cpu_index(clone), VES_OK);

    // ---- write & load -----------------------------------------------------
    const char* path = "ves_smoke_test_index.bin";
    CHECK_CODE(ves_write_cpu_index(idx, path), VES_OK);
    CHECK_CODE(ves_write_cpu_index(idx, "/nonexistent_dir/xyz.bin"), VES_FILE_NOT_FOUND);
    ves_cpu_index_t loaded = nullptr;
    CHECK_CODE(ves_load_cpu_index(path, &loaded), VES_OK);
    CHECK(loaded != nullptr);
    ves_index_status_t lst;
    std::memset(&lst, 0, sizeof(lst));
    CHECK_CODE(ves_get_cpu_index_status(loaded, &lst), VES_OK);
    CHECK(lst.index_size == kNAdd);
    CHECK(lst.is_trained == 1);
    CHECK(lst.subQuantizers == kM);
    std::vector<float> ldis(kK);
    std::vector<long> lsid(kK);
    CHECK_CODE(ves_search_cpu_index(loaded, 1, query, kK, ldis.data(), lsid.data()), VES_OK);
    CHECK(lsid[0] >= 0);
    CHECK_CODE(ves_load_cpu_index("/nonexistent_dir/xyz.bin", &loaded), VES_FILE_NOT_FOUND);
    CHECK_CODE(ves_free_cpu_index(loaded), VES_OK);
    std::remove(path);

    // ---- remove -----------------------------------------------------------
    long remove_ids[10];
    for (int i = 0; i < 10; ++i) remove_ids[i] = ids[i];
    CHECK_CODE(ves_remove_cpu_index_ids(idx, 10, remove_ids), VES_OK);
    ves_index_status_t rst;
    std::memset(&rst, 0, sizeof(rst));
    CHECK_CODE(ves_get_cpu_index_status(idx, &rst), VES_OK);
    CHECK(rst.index_size == kNAdd - 10);
    std::vector<long> rids(rst.index_size);
    CHECK_CODE(ves_get_cpu_index_ids(idx, rids.data()), VES_OK);
    for (long r : rids) {
        for (long rid : remove_ids) CHECK(r != rid);
    }

    // ---- null-handle behavior ---------------------------------------------
    CHECK_CODE(ves_search_cpu_index(nullptr, 1, query, kK, dis.data(), sid.data()),
               VES_INVALID_HANDLE);
    CHECK_CODE(ves_get_cpu_index_status(nullptr, &st), VES_INVALID_HANDLE);

    CHECK_CODE(ves_free_cpu_index(idx), VES_OK);
}

void run_gpu_stub_tests() {
    // Mode-agnostic: CPU-only build returns -7, CUDA build returns -1 for a
    // null handle.
    ves_gpu_index_t g = nullptr;
    float dis[kK];
    long sid[kK];
    int rc = ves_search_index(nullptr, 1, nullptr, kK, dis, sid);
    CHECK(rc == -7 || rc == VES_INVALID_HANDLE);

    ves_gpu_info_t info;
    rc = ves_get_gpu_info(0, &info);
    CHECK(rc == -7 || rc == VES_OK);

    rc = ves_get_current_device_id();
    CHECK(rc == -7 || rc >= 0);

    ves_resource_t res = nullptr;
    rc = ves_create_gpu_resource(0, 0, 0, &res);
    CHECK(rc == -7 || rc == VES_OK);

    rc = ves_create_index_shards(16, 0, nullptr);
    CHECK(rc == VES_INVALID_ARGUMENT || rc == -7);
}

}  // namespace

int main() {
    run_cpu_tests();
    run_gpu_stub_tests();
    std::printf("ves smoke test: OK\n");
    return 0;
}
