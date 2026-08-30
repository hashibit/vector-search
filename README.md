# vector_search

`vector_search` (library `libvector_search`, short name `ves`) is a C++ vector
search library built on [faiss](https://github.com/facebookresearch/faiss). It
exposes the `ves_*` C API declared in
[`include/vector_search.h`](include/vector_search.h): IVF-PQ index lifecycle
(init → train → add → search), timespace-filtered search, CPU↔GPU index
conversion, sharded indexes, and index serialization.

faiss is compiled in statically and hidden (`-fvisibility=hidden`), so the
shared library exports exactly the `ves_*` C API — 34 functions — and nothing
else.

## Features

- **IVF-PQ index lifecycle** — create, train, add, search, clone, remove ids,
  write/load; CPU and (with CUDA) GPU.
- **Timespace-filtered search** — candidates are filtered by camera id and
  capture-time window inside the IVF-PQ scan, based on the id encoding below.
- **Sharded indexes** (`ves_index_shards_t`) — a collection of indexes searched
  as one; threaded fan-out with OpenMP and k-way merge (CUDA build).
- **CPU ↔ GPU conversion** — `ves_index_cpu_to_gpu` / `ves_index_gpu_to_cpu`
  (with an option to release the GPU copy on the way down).
- **Serialization** — `ves_write_cpu_index` / `ves_load_cpu_index` on faiss
  index files.

## Error codes

Every entry point returns an `int` error code and never throws across the C
boundary; faiss exceptions are mapped as follows.

| Code | Name | Meaning |
|---|---|---|
| 0 | `VES_OK` | success |
| -1 | `VES_INVALID_HANDLE` | null/expired index handle; loaded file is not an IVF-PQ index |
| -2 | `VES_FILE_NOT_FOUND` | `fopen` failure |
| -3 | `VES_NOMEMORY` | allocation failure (`bad_alloc`, null clone result) |
| -4 | `VES_INVALID_ARGUMENT` | null pointer, negative `n`/`k` |
| -5 | `VES_BAD_INDEX` | faiss exception while loading (corrupt file / bad magic) |
| -6 | `VES_SYSCALL_ERROR` | `fflush`/`fclose` failure, other system errors |

Without CUDA (`VES_WITH_CUDA=OFF`, the default) every GPU/shards/device entry
point is a stub returning **-7**.

## Ids and the timespace filter

A feature id encodes the frame a vector was extracted from:

```
id = (time_sec << 7) | camera_id      // camera in bits 0-6, time in the rest
```

`ves_timespace_filter_t` carries `time_range[2]` (unix seconds, inclusive) and
`camera_mask[128/8]` (bit set = camera included). Filtered search skips
candidates whose camera bit is clear or whose time is outside the range, so up
to `k` passing results are returned per query; empty slots are padded with
`FLT_MAX` / `-1`, and results are sorted by (distance, id) ascending. A filter
with `time_range == {0,0}` and an all-`0xff` mask is inactive and passes every
id.

The packing contract is isolated in `src/ves_internal.h` (`make_feature_id`,
`id_time`, `id_camera`, `kCameraBits`) — one place to change if the real
contract differs.

## Layout

```
include/vector_search.h     C API header (ves_* functions, VES_* error codes)
src/vector_search.cpp       all ves_* entry points, faiss mapping, error codes
src/ves_internal.h          handles, id-packing contract, exception mapping
src/id_filter.h             TimespaceFilter struct and pass logic
src/ves_search_filter.h/.cpp  filtered IVF-PQ scan (public faiss APIs only)
src/faiss_aarch64_shim.cpp  arm64 workaround: faiss's __aarch64__ build omits
                            the fvec_inner_products_ny wrapper
tests/ves_smoke_test.cpp    end-to-end CPU smoke test
tests/ves_filter_crosscheck.cpp  filtered search vs independent brute-force scan
CMakeLists.txt              build; faiss via find_package or FetchContent
```

## Building

```sh
# CPU-only (default) — GPU entry points return -7
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build

# system faiss instead of FetchContent
cmake -S . -B build -DVES_USE_FETCHED_FAISS=OFF

# GPU build (CUDA toolkit required; written against the modern faiss GPU API,
# checked against the faiss 1.14 headers, not yet compiled/validated)
cmake -S . -B build-gpu -DVES_WITH_CUDA=ON
```

Output: `libvector_search.dylib` (macOS) / `libvector_search.so.0.1.0`
(Linux, SONAME `libvector_search.so.0`).

CMake options:

| Option | Default | Purpose |
|---|---|---|
| `VES_WITH_CUDA` | `OFF` | build the GPU paths against the faiss GPU API |
| `VES_BUILD_TESTS` | `ON` | build the two test executables |
| `VES_USE_FETCHED_FAISS` | `ON` | FetchContent faiss instead of `find_package` |
| `VES_FAISS_VERSION` | `1.14.3` | faiss tag used by FetchContent |

macOS prerequisites: a BLAS/LAPACK (Homebrew `openblas`) and OpenMP (Homebrew
`libomp`); export `OPENMP_ROOT`/`CMAKE_PREFIX_PATH` as needed.

## Tests

- `ves_smoke_test` — full lifecycle: init → train → add → search →
  timespace filter (time window, camera mask, padding) → ids → clone →
  write/load → remove → error codes, plus the -7 GPU stubs.
- `ves_filter_crosscheck` — compares filtered search against an independent
  decode-based brute-force scan: ids must match exactly, distances within
  float tolerance, empty slots padded with `FLT_MAX` / `-1`.

Both pass on macOS (Apple Silicon) with faiss 1.14.3 (CPU-only).

## faiss version

Default **faiss 1.14.3**, fetched via FetchContent. Older versions build with
`-DVES_FAISS_VERSION=<tag>` (e.g. `1.8.0`, `1.6.5`). The code absorbs the
faiss-side changes across those versions: `faiss::idx_t` at namespace scope
(C++17), the modern GPU API (`GpuIndexIVFPQ` built from a CPU index,
per-search `SearchParametersIVF`), and Metal disabled on macOS/arm64
(`FAISS_ENABLE_METAL=OFF`).

## Notes

- No faiss patching: the filtered scan is implemented on top of public faiss
  APIs (`InvertedLists`, `ProductQuantizer::compute_distance_table`, coarse
  `quantizer->search`). Distances are exact L2, matching faiss's own
  precomputed-table arithmetic up to float summation order.
