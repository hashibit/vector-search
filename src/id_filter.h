// The timespace filter used by ves_search_*_index_with_timespace_filter.
//
// Stock faiss has no per-candidate filter hook, so filtered search cannot go
// through Index::search; the filter is carried as a plain struct (the same
// layout the C API's ves_timespace_filter_t uses) and applied inside the
// filtered IVF-PQ scan (see ves_search_filter.cpp).
//
// Object layout:
//
//   offset 0x00  int      filterType          (= 1 for TimespaceFilter)
//   offset 0x04  uint32_t camera_mask[4]      (16 bytes = 128 cameras)
//   offset 0x14  uint32_t time_range[2]       (unix seconds, inclusive)
#ifndef VES_ID_FILTER_H_
#define VES_ID_FILTER_H_

#include <cstdint>

#include "ves_internal.h"

namespace ves {

enum FilterType {
    kFilterTimespace = 1,
};

struct TimespaceFilter {
    int filterType;
    uint32_t camera_mask[4];
    uint32_t time_range[2];

    // Layout-compatible with the ves_timespace_filter_t the C API receives.
    static TimespaceFilter from_ves(const ves_timespace_filter_t& f) {
        TimespaceFilter t;
        t.filterType = kFilterTimespace;
        for (int w = 0; w < 4; ++w) {
            uint32_t v = 0;
            for (int b = 0; b < 4; ++b) {
                v |= uint32_t(f.camera_mask[w * 4 + b]) << (8 * b);
            }
            t.camera_mask[w] = v;
        }
        t.time_range[0] = f.time_range[0];
        t.time_range[1] = f.time_range[1];
        return t;
    }

    // Inactive-filter semantics: time_range == {0,0} with an all-0xff camera
    // mask means "no filter" — every id must pass.
    bool camera_active() const {
        for (uint32_t w : camera_mask) {
            if (w != 0xffffffffu) return true;
        }
        return false;
    }

    bool time_active() const {
        return time_range[0] != 0 || time_range[1] != 0;
    }

    bool pass(faiss::idx_t id) const {
        if (camera_active()) {
            const int cam = id_camera(id);
            if (((camera_mask[cam >> 5] >> (cam & 31)) & 1u) == 0) {
                return false;
            }
        }
        if (time_active()) {
            const uint32_t t = id_time(id);
            if (t < time_range[0] || t > time_range[1]) {
                return false;
            }
        }
        return true;
    }
};

}  // namespace ves

#endif  // VES_ID_FILTER_H_
